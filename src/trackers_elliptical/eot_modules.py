"""
Implements reference extended object tracking methods as a single function, to enable easy wrapping. Each function
takes all required parameters and performs a predict+update pair using the specific method, returning the posterior
state mean and covariance.
"""
import numpy as np
from scipy.linalg import block_diag

from src.utilities.utils import rot, vect


def iae_step(Z,
             x, P,
             l, c,
             H, F, R, Q, q_c,
             orientation=None):
    """
    Perform a predict-update pair for the IAE method, given all required parameters, and returning the posterior state
    mean and covariance.
    """
    x_minus, P_minus, l_minus, c_minus = predict_iae(x, P,
                                                     l, c,
                                                     F, Q, q_c)
    x_plus, P_plus, l_plus, c_plus = update_iae(Z,
                                                x_minus, P_minus,
                                                l_minus, c_minus,
                                                H, R,
                                                c_scaling=0.25, fix_orientation=orientation)
    if orientation is None:
        orientation = np.arctan2(x_plus[3], x_plus[2])
    state = np.array([*x_plus, orientation, *l_plus])
    cov = block_diag(P_plus, np.zeros((1, 1)), np.diag(c_plus))
    return state, cov


def memekf_step(Z,
                x, p,
                C_m, C_p,
                H, F, R, Q, q_extent,
                orientation=None):
    """
    Perform a predict-update pair for the MEM-EKF*, given all required parameters, and returning the posterior state
    mean and covariance.
    Note that for the MEM-EKF*, the orientation is part of the extent state p. Passing the explicit orientation
    parameter forces a fixed overwrite of the orientation after each individual update iteration.
    """
    m_plus, p_plus, C_m_plus, C_p_plus = predict_memekf(x, p,
                                                        C_m, C_p,
                                                        F, Q, q_extent)
    for z in Z:
        m_plus, p_plus, C_m_plus, C_p_plus = update_memekf(z,
                                                           m_plus, p_plus,
                                                           C_m_plus, C_p_plus,
                                                           H, R,
                                                           fix_orientation=orientation)
    state = np.hstack([m_plus.reshape(4, ), p_plus.reshape(3, )])
    cov = block_diag(C_m_plus, C_p_plus)
    if orientation is not None:
        cov[:, 4] = 0
        cov[4, :] = 0
    return state, cov


def pakf_step(Z,
              x, p,
              C_x, C_p,
              H, H_p,
              F,
              R, R_p,
              Q, Q_extent,
              orientation):
    """
    Perform a predict-update pair for the PAKF, given all required parameters, and returning the posterior state mean
    and covariance.
    """
    x, p, C_x, C_p = predict_pakf(x, C_x, p, C_p, F, Q, Q_extent)
    x_plus, p_plus, C_x_plus, C_p_plus = update_pakf(Z,
                                                     x, p,
                                                     C_x, C_p,
                                                     H, H_p, R, R_p,
                                                     orientation)
    state = np.hstack([x_plus.reshape(4, ), p_plus.reshape(3, )])
    cov = block_diag(C_x_plus, C_p_plus)

    return state, cov


def predict_iae(x_minus, P_minus, l_minus, c_minus, F, Q, q_c):
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
    x_plus = F @ x_minus
    P_plus = F @ P_minus @ F.T + Q
    l_plus = l_minus
    # q_c: defined in "VARIABLES" section above
    c_plus = c_minus + q_c
    return x_plus, P_plus, l_plus, c_plus


def update_iae(Z, x_minus, P_minus, l_minus, c_minus, H, R, c_scaling=0.25, fix_orientation=None):
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
    innov = z_avg - H @ x_minus

    if fix_orientation is None:
        fix_orientation = np.arctan2(x_minus[3], x_minus[2])
    L = np.diag(l_minus)
    # R: measurement noise covariance
    # c: defined in "VARIABLES" section above
    R_bar = (1 / n) * (c_scaling * (rot(fix_orientation) @ L ** 2 @ rot(fix_orientation).T) + R)
    S = H @ P_minus @ H.T + R_bar  # innovation covariance

    W = P_minus @ H.T @ np.linalg.inv(S)  # gain

    # update parameters:
    x_plus = x_minus + W @ innov
    P_plus = P_minus - W @ S @ W.T

    # (2) Shape Update
    # measurement observation of half axis length d with corresponding variance v (both as 2D array)
    if len(Z) > 2:
        d, v = half_axis_observation(Z=Z, R_k=R, alpha=fix_orientation, c_scaling=c_scaling)

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


def half_axis_observation(Z, R_k, alpha, c_scaling):
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
    w, V = np.linalg.eig(Z_spread * (1 / c_scaling))

    # ---
    # [Kolja Thormann]
    # Code to check if switching eigenvalues is necessary
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
    K = (1 / c_scaling) * (V.T @ R_k @ V)
    k = np.diag(K)

    subtracted_noise = np.array(w - k)

    # eigenvalues are sometimes smaller than noise to be subtracted, setting those entries to 1e-2 before sqrt
    subtracted_noise[subtracted_noise < 1e-2] = 1e-2

    d = np.sqrt(subtracted_noise)
    v = ((d ** 2 + k) ** 2) / (2 * (n - 1) * d ** 2)
    return d, v


def predict_memekf(m_minus, p_minus, C_m_minus, C_p_minus, F, Q, Q_extent):
    """
    Given a set of MEM-EKF* parameters, perform the prediction based on the internal Q/Q_extent variables and
    return the predicted versions of the input parameters

    Uses the constant velocity assumption.
    """
    # predict kinematics
    m_plus = F @ m_minus.astype(float)

    C_m_plus = F @ C_m_minus @ F.T + Q

    # predict extent
    p_plus = p_minus

    C_p_plus = C_p_minus + Q_extent
    return m_plus, p_plus, C_m_plus, C_p_plus


def update_memekf(z, m_minus, p_minus, C_m_minus, C_p_minus, H, R, fix_orientation=None):
    """
    Update function for the MEM-EKF* algorithm as described in Table III of the referenced tutorial paper.

    Takes a single measurement z as well as the prior estimates for m, p, C^m and C^p as parameters.
    Returns the updated estimates for m, p, C^m and C^p after incorporating z.
    """
    C_h = R
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

    C_I = S @ C_h @ S.T
    C_II = np.block([
        [np.trace(C_p_minus @ J[m].T @ C_h @ J[n]) for m in range(2)] for n in range(2)
    ]).reshape(2, 2)

    M = np.array([
        [2 * S_1 @ C_h @ J_1],
        [2 * S_2 @ C_h @ J_2],
        [S_1 @ C_h @ J_2 + S_2 @ C_h @ J_1]
    ])
    M = np.reshape(M, (3, -1))

    C_mz = C_m_minus @ H.T
    C_z = H @ C_m_minus @ H.T + C_I + C_II + R
    Z = F @ np.kron(z - H @ m_minus, z - H @ m_minus)
    Z = np.reshape(Z, (-1, 1))
    Z_bar = F @ vect(C_z)

    C_pZ = C_p_minus @ M.T
    C_Z = F @ np.kron(C_z, C_z) @ (F + Ft).T

    # prepare for final calculations - invert C_z and C_Z
    C_z_inv = np.linalg.inv(C_z)
    C_Z_inv = np.linalg.inv(C_Z)
    p_minus = p_minus.reshape((-1, 1))

    # finally: calculate m, p, C_m, C_p
    m_plus = m_minus + C_mz @ C_z_inv @ (z - H @ m_minus)
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


def update_pakf(measurements, x, p, C_x, C_p, H, H_p, R, R_p, fix_orientation=None):
    # ===
    # Kinematic Update
    y_bar = np.average(measurements, axis=0)
    y_hat = H @ x
    C_xy = C_x @ H.T
    C_yy = H @ C_x @ H.T + R

    C_yy_inv = np.linalg.inv(C_yy)
    x_plus = x + C_xy @ C_yy_inv @ (y_bar - y_hat)
    C_x_plus = C_x - C_xy @ C_yy_inv @ C_xy.T

    # EVD Step
    y_shifted = measurements - y_bar
    # NOTE: the sum is replaced with matrix multiplication of the stacked measurements to save compute
    E_d = (1 / (len(measurements) - 1)) * (y_shifted.T @ y_shifted)
    E_d_hat = E_d - R
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

    y_hat_p = H_p @ p

    C_py = C_p @ H_p.T
    C_yy = H_p @ C_p @ H_p.T + R_p

    C_yy_inv = np.linalg.inv(C_yy)
    # NOTE: paper has C_xy here, but should be C_py for the extent!
    p_plus = p + C_py @ C_yy_inv @ (y_p - y_hat_p)
    C_p_plus = C_p - C_py @ C_yy_inv @ C_py.T

    if np.isnan(p_plus).any():
        raise ValueError(f"p+ in PAKF is {p_plus} - may not contain NaNs though!")

    return x_plus, p_plus, C_x_plus, C_p_plus


def predict_pakf(x, C_x, p, C_p, F, Q, Q_extent):
    # predict kinematics
    x = F @ x.astype(float)
    C_x = F @ C_x @ F.T + Q

    # predict extent
    # p = p
    C_p = C_p + Q_extent
    return x, p, C_x, C_p
