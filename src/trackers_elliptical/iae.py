"""
Implements the IAE tracker.

Please refer to
    A Tutorial on Multiple Extended Object Tracking
    K. Granström and M. Baum, 2022
    https://www.techrxiv.org/articles/preprint/ATutorialonMultipleExtendedObjectTracking/19115858/1
based on which this implementation has been done.

Furthermore, refer to
    On Independent Axes Estimation for Extended Target Tracking
    F. Govaers, 2019
    https://ieeexplore.ieee.org/document/8916660
for the IAE specifically.
"""
import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.linalg import block_diag
from src.trackers_elliptical.abstract_tracking import AbtractEllipticalTracker
from src.utilities.utils import rot, get_initial_estimate


class TrackerIAE(AbtractEllipticalTracker):
    """
    Implementation of the IAE tracker using the following state variables:
    x: 4D state
    P: 4D state cov
    l: semi axis length
    c: semi axis uncertainty
    """

    def __init__(self,
                 P_init,
                 c_init,
                 R=None,
                 time_step_length=1,
                 H=None,
                 Q=None,
                 F=None,
                 q_c=0,
                 c_scaling=0.25):
        """
        Create a new tracker given the parameters
        :param P_init: Initial kinematic state uncertainty
        :param c_init: Initial semi axes uncertainty
        :param R: Measurement Noise
        :param time_step_length: Time between discrete steps
        :param H: Measurement Model
        :param Q: Process Noise
        :param F: Motion Model
        :param q_c: Noise parameter modeling object size change over time. Set to 0 due to fixed object size.
        :param c_scaling: Scaling factor, use 0.25 for ellipses
        """
        self.x = None
        self.P = P_init
        self.len_semi_axis = None
        self.c = np.array(c_init)

        self.R = np.eye(2) if R is None else R
        self.q_c = q_c
        self.H = H if H is not None else np.block([np.eye(2), np.zeros((2, 2))])
        self.Q = np.eye(4) * 0.001 if Q is None else Q
        self.F = np.array([
            [1, 0, time_step_length, 0],
            [0, 1, 0, time_step_length],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]) if F is None else F
        self.c_scaling = c_scaling

    def predict(self):
        assert self.x is not None, "x is None - did you predict before first update?"
        self.x, self.P, self.len_semi_axis, self.c = self.predict_iae(self.x, self.P, self.len_semi_axis, self.c)
        return self.get_state()

    def update(self, Z, fix_orientation=None):
        if self.x is None or self.len_semi_axis is None:
            # initialize:
            m, p = get_initial_estimate(measurements=Z, fix_orientation=fix_orientation)
            self.x = m
            self.len_semi_axis = p[1:]
            return self.get_state()
        else:
            self.x, self.P, self.len_semi_axis, self.c = self.update_iae(Z, self.x, self.P, self.len_semi_axis, self.c,
                                                                         fix_orientation=fix_orientation)
            return self.get_state()

    def set_R(self, R):
        self.R = R

    def get_state(self):
        state = np.zeros((7,))
        state[:4] = self.x
        state[4] = np.arctan2(self.x[3], self.x[2])
        state[5:] = np.array(self.len_semi_axis) * 2  # convert semi to full axis length
        return state

    def predict_iae(self, x_minus, P_minus, l_minus, c_minus):
        """
        Predict function for the IAE algorithm. Parameters:
        x_minus: Prior kinematic state estimate
        P_minus: Prior kinematic state covariance
        l_minus: 2D Array of Estimated semi-axis lengths
        c_minus: 2D Arrayof semi-axis length variances.
        """
        x_minus = np.array(x_minus)
        P_minus = np.array(P_minus)
        l_minus = np.array(l_minus)
        c_minus = np.array(c_minus)
        x_plus = self.F @ x_minus
        P_plus = self.F @ P_minus @ self.F.T + self.Q
        l_plus = l_minus
        # q_c: defined in "VARIABLES" section above
        c_plus = c_minus + self.q_c
        return x_plus, P_plus, l_plus, c_plus

    def update_iae(self, Z, x_minus, P_minus, l_minus, c_minus, fix_orientation=None):
        """
        Update ('filtering') function for the IAE algorithm. Parameters:
        :param Z: batch of measurements
        :param x_minus: Prior kinematic state estimate
        :param P_minus: Prior kinematic state covariance
        :param l_minus: 2D Array of Estimated semi-axis lengths
        :param c_minus: 2D Array of semi-axis length variances.
        :param fix_orientation: Given Orientation of target, or None to approximate orientation from velocity vector
        """
        x_minus = np.array(x_minus)
        P_minus = np.array(P_minus)
        l_minus = np.array(l_minus)
        c_minus = np.array(c_minus)

        # (1) Kinematic Update
        n = len(Z)
        z_avg = np.average(Z, axis=0)
        innov = z_avg - self.H @ x_minus

        if fix_orientation is None:
            fix_orientation = np.arctan2(x_minus[3], x_minus[2])
        L = np.diag(l_minus)
        # R: measurement noise covariance
        # c: defined in "VARIABLES" section above
        R_bar = (1 / n) * (self.c_scaling * (rot(fix_orientation) @ L ** 2 @ rot(fix_orientation).T) + self.R)
        S = self.H @ P_minus @ self.H.T + R_bar  # innovation covariance

        W = P_minus @ self.H.T @ np.linalg.inv(S)  # gain

        # update parameters:
        x_plus = x_minus + W @ innov
        P_plus = P_minus - W @ S @ W.T

        # (2) Shape Update
        # measurement observation of half axis length d with corresponding variance v (both as 2D array)
        if len(Z) > 2:
            d, v = self.half_axis_observation(Z=Z, R_k=self.R, alpha=fix_orientation)

            s_l = c_minus + v
            w_l = c_minus / s_l

            l_plus = l_minus + w_l * (d - l_minus)
            # NOTE: paper uses /s_l - causes huge problems - use *s_l instead
            # TODO: reproduce corner case in which this caused problems
            c_plus = c_minus - ((w_l ** 2) * s_l)
        else:
            # not enough measurements for half axis observation model, no change
            l_plus = l_minus
            c_plus = c_minus

        return x_plus, P_plus, l_plus, c_plus

    def half_axis_observation(self, Z, R_k, alpha=None):
        """
        Computing the half axis measurement from given sensor data Z and noise with covariance R_k
        Returns the observation of half axis length d with corresponding variance v, each as 2D arrays.
        """
        n = len(Z)
        if n < 3:
            raise ValueError("IAE half axis observation cant estimate anything for n<3")

        # spread matrix of measurements
        Z_spread = Z - np.average(Z, axis=0).reshape((-1, 2))
        Z_spread = (Z_spread.T @ Z_spread) / (n - 1)

        # calculation of eigenvalues - note that we use the rescaled version Z
        w, V = np.linalg.eig(Z_spread * (1 / self.c_scaling))

        # ---
        # [Kolja Thormann]
        # Code to check if switching eigenvalues is necessary
        if alpha is None:
            alpha = np.arctan2(self.x[3], self.x[2])
        eig0_or_diff = np.minimum(abs(((np.arctan2(V[1, 0], V[0, 0]) - alpha) + np.pi) % (2 * np.pi) - np.pi),
                                  abs(((np.arctan2(-V[1, 0], -V[0, 0]) - alpha) + np.pi) % (2 * np.pi) - np.pi))
        eig1_or_diff = np.minimum(abs(((np.arctan2(V[1, 1], V[0, 1]) - alpha) + np.pi) % (2 * np.pi) - np.pi),
                                  abs(((np.arctan2(-V[1, 1], -V[0, 1]) - alpha) + np.pi) % (2 * np.pi) - np.pi))
        if eig0_or_diff > eig1_or_diff:  # switch eigenvalues to make R==V assumption possible
            eig_save = w[0]
            w[0] = w[1]
            w[1] = eig_save
        # ---

        # approx V by rot based on velocity
        V = rot(alpha)
        K = (1 / self.c_scaling) * (V.T @ R_k @ V)
        k = np.diag(K)

        subtracted_noise = np.array(w - k)

        # eigenvalues are sometimes smaller than noise to be subtracted, setting those entries to 1e-2 before sqrt
        subtracted_noise[subtracted_noise < 1e-2] = 1e-2

        d = np.sqrt(subtracted_noise)
        v = ((d ** 2 + k) ** 2) / (2 * (n - 1) * d ** 2)
        return d, v

    def get_state_and_cov(self):
        state = self.get_state()
        cov = block_diag(self.P, np.diag([0, *self.c]))
        return state, cov

    def get_likelihood(self, Z, alpha=None):
        """
        Given a set of measurements, use the current state of this tracker instance to calculate the likelihood of
        these measurements.

        Calculates the likelihood as the product of the likelihood of the measurement mean and of the two semi-axes
        that can be observed using the IAE model in the measurements.

        :param Z: Received data / measurements
        :param alpha: Fixed orientation
        :return: Likelihood value
        """
        d, v = self.half_axis_observation(Z=Z, R_k=self.R, alpha=alpha)

        R_bar = 1 / len(Z) * (self.c_scaling * rot(alpha) @ np.diag(self.len_semi_axis) ** 2 @ rot(alpha).T + self.R)

        p = np.sum([multivariate_normal(mean=self.H @ self.x,
                                        cov=R_bar + self.H @ self.P @ self.H.T).logpdf(np.average(Z, axis=0)),
                    norm(self.len_semi_axis[0],
                         np.sqrt(self.c[0] + v[0])).logpdf(d[0]),
                    norm(self.len_semi_axis[1],
                         np.sqrt(self.c[1] + v[1])).logpdf(d[1]),
                    ])
        return np.exp(p)
