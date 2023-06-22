"""
Contains a simple MEM-EKF* Implementation for reference.

Please refer to
    A Tutorial on Multiple Extended Object Tracking
    K. Granström and M. Baum, 2022
    https://www.techrxiv.org/articles/preprint/ATutorialonMultipleExtendedObjectTracking/19115858/1
based on which this implementation has been done.

Furthermore, refer to
    Extended Kalman Filter for Extended Object Tracking
    S. Yang and M. Baum, 2017
    https://ieeexplore.ieee.org/document/7952985
for the MEM-EKF* specifically.
"""
import numpy as np
from scipy.linalg import block_diag

from src.utilities.utils import rot, get_initial_estimate
from src.trackers_elliptical.abstract_tracking import AbtractEllipticalTracker
from src.utilities.utils import vect


class TrackerMEMEKF(AbtractEllipticalTracker):
    def __init__(self, C_m_init, C_p_init, R, H=None, F=None, C_h=None, Q=None, Q_extent=None):
        """
        Initialize a new tracker
        :param C_m_init: Initial covariance of the kinematic estimate (4x4)
        :param C_p_init: Initial covariance of the extent estimate (3x3)
        :param R: Measurement Noise covariance
        :param H: Measurement Matrix or None. If None, will assume 3D state and use standard H
        :param C_h: Covariance of multiplicative Gaussian noise, or None. If none will be I*0.25
        :param Q: Process Noise or None (in which case Q=0) for kinematics
        :param Q_extent: Process Noise or None (in which case Q=I*1e-3) for extent
        """
        self.m = None
        self.p = None

        self.C_m = np.array(C_m_init)
        self.C_p = np.array(C_p_init)

        self.H = np.array(H) if H is not None else np.hstack([np.eye(2), np.zeros((2, 4 - 2))])
        self.R = np.array(R)
        self.C_h = np.array(C_h) if C_h is not None else 0.25 * np.eye(2, 2)
        self.Q = Q if Q is not None else np.diag([0, 0, 0])
        self.Q_extent = Q_extent if Q_extent is not None else np.diag([1e-3, 1e-3, 1e-3])
        F_cv = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.F = np.array(F) if F is not None else F_cv

    def predict(self):
        assert self.m is not None and self.p is not None, "m or p are None - did you predict before first update?"
        self.m, self.p, self.C_m, self.C_p = self.predict_memekf(self.m, self.p, self.C_m, self.C_p)
        return self.get_state()

    def update(self, Z, fix_orientation=None):
        if self.m is None or self.p is None:
            # initialize:
            self.m, self.p = get_initial_estimate(measurements=Z, fix_orientation=fix_orientation)
        else:
            # standard update
            for z in Z:
                self.m, self.p, self.C_m, self.C_p = self.update_memekf(z, self.m, self.p, self.C_m, self.C_p,
                                                                        fix_orientation=fix_orientation)
            return self.get_state()

    def set_R(self, R):
        """
        Update internal measurement noise covariance with equally shaped new measurement noise cov. matrix
        :param R: New covariance matrix
        """
        R = np.array(R)
        assert np.array(self.R).shape == R.shape, "Old ({}) and new ({}) R.shape are different".format(
            np.array(self.R).shape,
            R.shape)
        self.R = R

    def get_state(self):
        # location
        x, y, vel_x, vel_y = self.m.reshape((4,)).astype(float)

        # extent
        theta, length, width = self.p.reshape((3,)).astype(float)
        length, width = length * 2, width * 2

        return np.array([x, y, vel_x, vel_y, theta, length, width])

    def get_state_and_cov(self):
        return self.get_state(), block_diag(self.C_m, self.C_p).reshape((7, 7))

    def update_memekf(self, z, m_minus, p_minus, C_m_minus, C_p_minus, fix_orientation=None):
        """
        Update function for the MEM-EKF* algorithm as described in Table III of the referenced tutorial paper.

        Takes a single measurement z as well as the prior estimates for m, p, C^m and C^p as parameters.
        Returns the updated estimates for m, p, C^m and C^p after incorporating z.
        """
        # unpack p_minus
        alpha_minus, l1_minus, l2_minus = p_minus.reshape((3,))

        # F
        F = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0]
        ])
        # F tilde
        Ft = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])

        S = rot(alpha_minus) @ np.diag([l1_minus, l2_minus])
        S_1 = S[0, :].reshape((1, 2))
        S_2 = S[1, :].reshape((1, 2))

        J_1 = np.block([
            [-1 * l1_minus * np.sin(alpha_minus), np.cos(alpha_minus), 0],
            [-1 * l2_minus * np.cos(alpha_minus), 0, -1 * np.sin(alpha_minus)]
        ])
        J_2 = np.block([
            [l1_minus * np.cos(alpha_minus), np.sin(alpha_minus), 0],
            [-1 * l2_minus * np.sin(alpha_minus), 0, np.cos(alpha_minus)]
        ])
        J = [J_1, J_2]

        C_I = S @ self.C_h @ S.T
        C_II = np.block([
            [np.trace(C_p_minus @ J[m].T @ self.C_h @ J[n]) for m in range(2)] for n in range(2)
        ]).reshape(2, 2)

        M = np.array([
            [2 * S_1 @ self.C_h @ J_1],
            [2 * S_2 @ self.C_h @ J_2],
            [S_1 @ self.C_h @ J_2 + S_2 @ self.C_h @ J_1]
        ])
        M = np.reshape(M, (3, -1))

        C_mz = C_m_minus @ self.H.T
        C_z = self.H @ C_m_minus @ self.H.T + C_I + C_II + self.R
        Z = F @ np.kron(z - self.H @ m_minus, z - self.H @ m_minus)
        Z = np.reshape(Z, (-1, 1))
        Z_bar = F @ vect(C_z)

        C_pZ = C_p_minus @ M.T
        C_Z = F @ np.kron(C_z, C_z) @ (F + Ft).T

        # prepare for final calculations - invert C_z and C_Z
        C_z_inv = np.linalg.inv(C_z)
        C_Z_inv = np.linalg.inv(C_Z)
        p_minus = p_minus.reshape((-1, 1))

        # finally: calculate m, p, C_m, C_p
        m_plus = m_minus + C_mz @ C_z_inv @ (z - self.H @ m_minus)
        C_m_plus = C_m_minus - C_mz @ C_z_inv @ C_mz.T
        p_plus = p_minus + C_pZ @ C_Z_inv @ (Z - Z_bar)
        C_p_plus = C_p_minus - C_pZ @ C_Z_inv @ C_pZ.T

        # enforce symmetry of covariance:
        C_p_plus = (C_p_plus + C_p_plus.T) * 0.5
        C_m_plus = (C_m_plus + C_m_plus.T) * 0.5

        # optionally, overwrite fix orientation
        if fix_orientation is not None:
            p_plus[0] = fix_orientation

        return m_plus, p_plus, C_m_plus, C_p_plus

    def predict_memekf(self, m_minus, p_minus, C_m_minus, C_p_minus):
        """
        Given a set of MEM-EKF* parameters, perform the prediction based on the internal Q/Q_extent variables and
        return the predicted versions of the input parameters

        Uses the constant velocity assumption.
        """
        # predict kinematics
        m_plus = self.F @ m_minus.astype(float)

        C_m_plus = self.F @ C_m_minus @ self.F.T + self.Q

        # predict extent
        p_plus = p_minus

        C_p_plus = C_p_minus + self.Q_extent
        return m_plus, p_plus, C_m_plus, C_p_plus
