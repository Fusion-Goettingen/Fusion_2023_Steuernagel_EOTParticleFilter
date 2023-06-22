import numpy as np
from scipy.linalg import block_diag

from src.utilities.utils import get_initial_estimate
from src.trackers_elliptical.abstract_tracking import AbtractEllipticalTracker


class PrincipalAxesKalmanFilter(AbtractEllipticalTracker):
    """
    PrincipalAxesKalmanFilter, as presented by Fowdur et al. in
        J. S. Fowdur, M. Baum and F. Heymann, "An Elliptical Principal Axes-based Model for Extended Target Tracking
        with Marine Radar Data," 2021 IEEE 24th International Conference on Information Fusion (FUSION), Sun City,
        South Africa, 2021, pp. 1-8, doi: 10.23919/FUSION49465.2021.9627039.

    Direct link: https://ieeexplore.ieee.org/document/9627039
    """

    def __init__(self,
                 C_x_init,
                 C_p_init,
                 R,
                 R_p,
                 H=None,
                 Q=None,
                 Q_extent=None,
                 F=None,
                 time_step_length=1
                 ):

        self.x = None
        self.p = None
        self.C_x = np.array(C_x_init)
        self.C_p = np.array(C_p_init)
        self.R = R
        self.R_p = R_p

        self.H = H if H is not None else np.block([np.eye(2), np.zeros((2, 2))])
        self.H_p = np.eye(3)  # default extent measurement matrix
        self.Q = np.eye(4) * 0.001 if Q is None else Q
        self.Q_extent = Q_extent if Q_extent is not None else np.diag([1e-3, 1e-3, 1e-3])
        self.F = np.array([
            [1, 0, time_step_length, 0],
            [0, 1, 0, time_step_length],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) if F is None else F

    def update(self, measurements: np.ndarray, fix_orientation: float = None):
        if self.x is None or self.p is None:
            self.x, self.p = get_initial_estimate(measurements, fix_orientation)
            # PAKF tracks full axis length
            self.p[1:] *= 2
            return

        # ===
        # Kinematic Update
        y_bar = np.average(measurements, axis=0)
        y_hat = self.H @ self.x
        C_xy = self.C_x @ self.H.T
        C_yy = self.H @ self.C_x @ self.H.T + self.R

        C_yy_inv = np.linalg.inv(C_yy)
        x_plus = self.x + C_xy @ C_yy_inv @ (y_bar - y_hat)
        C_x_plus = self.C_x - C_xy @ C_yy_inv @ C_xy.T

        # Save results
        self.x = x_plus
        self.C_x = C_x_plus

        # EVD Step
        y_shifted = measurements - y_bar
        # NOTE: the sum is replaced with matrix multiplication of the stacked measurements to save compute
        E_d = (1 / (len(measurements) - 1)) * (y_shifted.T @ y_shifted)
        E_d_hat = E_d - self.R
        # NOTE: We catch too small values in E_d - R by thresholding, as suggested in the original paper
        E_d_hat[E_d_hat < 1e-2] = 1e-2
        eig_vals, Q = np.linalg.eig(E_d_hat)
        if np.any(eig_vals < 0):
            eig_vals, Q = np.linalg.eig(E_d)
        # np evd does not guarantee major ev is first, save that fact for later
        inversion_flag = eig_vals[0] < eig_vals[1]

        # NOTE: as a fail-safe if E_d_hat was not pos. def., we skip the extent update if any eig_vals are negative
        a, b = 2 * np.sqrt(eig_vals)  # semi-axes length computation
        # to full axis
        a, b = a * 2, b * 2
        major_ix = 0 if not inversion_flag else 1
        major_ev = Q[:, major_ix]
        theta = np.arctan2(major_ev[1], major_ev[0]) % (2 * np.pi)

        # Extent Update
        # make sure "a" is the major axis
        # orientation is cared for by selecting the major EV already before
        if inversion_flag:
            y_p = np.asarray([theta, b, a])
        else:
            y_p = np.asarray([theta, a, b])
        if fix_orientation is not None:
            y_p[0] = fix_orientation
        y_hat_p = self.H_p @ self.p

        C_py = self.C_p @ self.H_p.T
        R_p = self.R_p if self.R_p is not None else np.diag(
            [0.1, *[np.trace(self.R) / len(measurements)] * 2]  # note that the * 2 only repeats the noise, not double!
        )
        C_yy = self.H_p @ self.C_p @ self.H_p.T + R_p

        C_yy_inv = np.linalg.inv(C_yy)
        # NOTE: paper has C_xy here, but should be C_py for the extent!
        # calculate innovation
        innov_p = y_p - y_hat_p
        # update the orientation innovation to accommodate wrap around 0/2pi
        innov_p[0] = self._orientation_innovation(x=y_hat_p[0], z=y_p[0])
        p_plus = self.p + C_py @ C_yy_inv @ innov_p
        C_p_plus = self.C_p - C_py @ C_yy_inv @ C_py.T

        if np.isnan(p_plus).any():
            raise ValueError(f"p+ in PAKF is {p_plus} - may not contain NaNs though!")

        # ensure orientation is bounded correctly
        p_plus[0] = p_plus[0] % (2 * np.pi)

        # Save results
        self.p = p_plus
        self.C_p = C_p_plus

    def predict(self):
        # predict kinematics
        self.x = self.F @ self.x.astype(float)
        self.C_x = self.F @ self.C_x @ self.F.T + self.Q

        # predict extent
        self.C_p = self.C_p + self.Q_extent

    def get_state(self):
        # location
        x, y, vel_x, vel_y = self.x.reshape((4,)).astype(float)

        # extent
        theta, length, width = self.p.reshape((3,)).astype(float)
        return np.array([x, y, vel_x, vel_y, theta, length, width])

    def get_state_and_cov(self):
        return self.get_state(), block_diag(self.C_x, self.C_p).reshape((7, 7))

    def set_R(self, R, R_p=None):
        self.R = R
        if R_p is not None:
            self.R_p = R_p

    @staticmethod
    def _orientation_innovation(x, z):
        """
        Calculate the innovation between a state x and a measurement z for a circular state space in 0,2pi

        Changes z in order to deal with "wrap-around" cases at 0 and 2pi
        :param x: State
        :param z: Measurement
        :return: Innovation between parameters
        """
        if np.abs(x - (z + 2 * np.pi)) < np.abs(x - z):
            z += 2 * np.pi
        elif np.abs(x - (z - 2 * np.pi)) < np.abs(x - z):
            z -= 2 * np.pi
        return z - x
