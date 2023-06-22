"""
Contains general utility functions.
"""
import numpy as np


def rot(theta):
    """
    Constructs a rotation matrix for given angle alpha.
    :param theta: angle of orientation
    :return: Rotation matrix in 2D around theta (2x2)
    """
    r = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return r.reshape((2, 2))


def pol2cart(rho, phi):
    """
    Convert Polar coordinates to Cartesian
    :param rho: Distance to origin
    :param phi: Angle to x-Axis
    :return: [x, y] in Cartesian Coordinates as numpy array
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])


def cart2pol(x, y):
    """
    Help function to convert from polar(radius rho, angle phi) to cartesian coordinates.
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    return np.array([rho, phi])


def ellipse_size_to_D(l, w):
    """
    Given full axis lengths, return diagonal shape matrix
    :param l: Full major axis length
    :param w: Full minor axis length
    :return: Diagonal Shape Matrix approximating a uniform distribution on the ellipse surface
    """
    D = np.diag([(l / 2) ** 2, (w / 2) ** 2])
    return 0.25 * D


def identity(x):
    """This is a helper function equivalent to lambda x: x"""
    return x


def matrix_to_params(X):
    """Convert shape matrix X to parameter form [alpha, l1, l2] with semi-axis length"""
    assert X.shape == (2, 2), "X is not a 2x2 matrix"
    val, vec = np.linalg.eig(X)  # eigenvalue decomposition
    alpha = np.arctan2(vec[1][0], vec[0][0])  # calculate angle of orientation
    alpha = (alpha + 2 * np.pi) % (2 * np.pi)  # just in case alpha was negative
    p = [alpha, *np.sqrt(val)]
    return np.array(p)


def params_to_matrix(p):
    """
    Convert parameters [alpha, l1, l2] to shape matrix X (2x2)
    """
    X = rot(p[0]) @ np.diag(np.array(p[1:]) ** 2) @ rot(p[0]).T
    return X


def state_to_srs(extent_state):
    """Convert state parameters [theta, l, w] to square rood space shape matrix"""
    theta, l, w = extent_state
    return rot(theta) @ np.diag([l, w]) @ rot(theta).T


def srs_to_state(srs_mat):
    """convert square root space matrix to shape parameters"""
    # convert to shape matrix by squaring and then calculate parameters
    return matrix_to_params((srs_mat @ srs_mat.T).astype(float))


def vect(M):
    """
    From original MEM-EKF* paper:
    Constructs a column vector from a matrix M by stacking its column vectors
    """
    v = M.flatten(order="F")  # just use ndarray.flatten(), pass `order='F'` for column-major order
    v = np.reshape(v, (len(v), 1))  # ensure output is column vector
    return v


def get_initial_estimate(measurements, fix_orientation=None, merge_return=False):
    """
    Creates an initial estimate from the measurements in a straightforward manner using EVD of the scattering matrix

    :param measurements: Nx2 ndarray of measurements to process
    :param fix_orientation: Fixed Orientation overwriting the estimate
    :param merge_return: If True, will return a single 7D state rather than m, p
    :return: m,p: kinematic, extent estimate as 4D/3D
    """
    # initialize:
    m = np.average(measurements, axis=0)
    m = np.array([*m, 0, 0])
    # get cov mat
    X = measurements - m[:2]
    X = (1 / (len(measurements) - 1)) * (X.T @ X)

    # extract alpha and semi-axis lengths
    val, vec = np.linalg.eig(X)  # eigenvalue decomposition
    inversion_flag = val[0] < val[1]  # check if evd return major axis on second position
    if inversion_flag:
        # flip semi-axes so major axis is first
        val = val[::-1]

    if fix_orientation is None:
        major_ix = 0 if not inversion_flag else 1
        major_ev = vec[:, major_ix]
        theta = np.arctan2(major_ev[1], major_ev[0]) % (2 * np.pi)
    else:
        theta = fix_orientation
    p = np.array([theta, *(2 * np.sqrt(val))])

    if merge_return:
        return np.array([*m, *p])
    else:
        return m, p
