import numpy as np
from scipy.linalg import sqrtm

from src.utilities.utils import params_to_matrix


def gwd_shape(d1, X1, d2, X2, return_parts=False):
    """
    Squared Gaussian Wasserstein Distance as defined in eq. (63)
    Compares two elliptic extended targets defined by their centers (d) and shape matrices (X)
    """
    d1, X1, d2, X2 = np.array(d1), np.array(X1), np.array(d2), np.array(X2)
    # first: perform preliminary calculations
    X1sqrt = sqrtm(X1.astype(float))
    C = sqrtm(X1sqrt @ X2 @ X1sqrt)

    # finally: construct GW distance
    d1 = np.linalg.norm(d1 - d2) ** 2
    d2 = np.trace(X1 + X2 - 2 * C)
    d = d1 + d2
    # for sgw == 0, rounding errors might cause a minimally negative result. round to avoid
    if d < 0:
        d = np.around(d, 4)
    if return_parts:
        return d, d1, d2
    else:
        return d


def gwd(m1, p1, m2, p2, return_parts=False):
    """
    Calculates the Squared Gaussian Wasserstein Distance, using the 'gwd_shape' function.
    Compares two ellipses with object center m=[x_location, y_location]
    and object extend p=[orientation, length, width].
    """
    d1 = m1
    d2 = m2
    X1 = params_to_matrix(p1)
    X2 = params_to_matrix(p2)
    return gwd_shape(d1, X1, d2, X2, return_parts=return_parts)


def gwd_full_state(s1, s2, return_parts=False):
    """
    Returns the squared Gauss Wasserstein distance for two elliptical extended object.
    Each object is parameterized by a 7D state in the form:
        [loc_x, loc_y, velocity_x, velocity_y, orientation, full_length, full_width]
    :param s1: 7D state of object 1
    :param s2: 7D state of object 2
    :param return_parts:
    :return: Squared Gauss Wasserstein distance
    """
    # split each state into center and extent information
    # use half-axis length
    s1, s2 = np.array(s1), np.array(s2)
    m1 = s1[:2]
    p1 = s1[4:]
    p1[1:] /= 2

    m2 = s2[:2]
    p2 = s2[4:]
    p2[1:] /= 2
    return gwd(m1, p1, m2, p2, return_parts=return_parts)
