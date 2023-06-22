import numpy as np

from src.utilities.utils import rot


def get_ellipse_measurements(loc, length, width, theta, n_measurements, R, internal_RNG=None, polar_noise=False):
    """
    Given an elliptical extended object, returns n_measurements many measurement sources and noise-corrupted
    measurements across the entire object surface, based on measurement covariance R.

    Uses accept-reject sampling to uniformly generate measurement sources across the ellipse.
    :param loc: [x,y] location of object
    :param length: length of object in direction of orientation
    :param width: width of object, orthogonal to direction of orientation
    :param theta: orientation of object in radians
    :param n_measurements: number of measurements to draw. If <1, will be set to 1 instead!
    :param R: 2x2 measurement noise matrix
    :param internal_RNG: np random Generator or None. If None, a new generator will be created, without seed.
    :param polar_noise: If True, will generate noise in polar coordinates
    :return: Y, Z: measurement sources and measurements, both as np arrays of shape (n_measurements, 2)
    """
    if n_measurements < 1:
        n_measurements = 1

    if internal_RNG is None:
        internal_RNG = np.random.default_rng()

    # half axis length
    half_length = length / 2
    half_width = width / 2
    Y = []
    while len(Y) < n_measurements:
        # draw new candidate point [x,y]
        x = internal_RNG.uniform(low=-half_length, high=half_length)
        y = internal_RNG.uniform(low=-half_width, high=half_width)

        # determine whether to check for <=1 or <1 (entire surface or not)
        # check if it matches ellipse equation:
        if (x ** 2 / half_length ** 2) + (y ** 2 / half_width ** 2) <= 1:
            # measurement y
            y = np.array([x, y])
            # rotate to match orientation
            y = rot(theta) @ y
            # offset based on location of ellipse center
            y += loc
            # save
            Y.append(y)
    Y = np.vstack(Y)
    # apply gaussian noise with cov. matrix R to all measurements
    if R is not None:
        Z = np.vstack([internal_RNG.multivariate_normal(y, R) for y in Y])
    else:
        Z = Y
    return Y, Z
