import numpy as np

from abc import ABC, abstractmethod


class AbtractEllipticalTracker(ABC):
    is_polar = False

    @abstractmethod
    def update(self, measurements: np.ndarray):
        """
        Perform an update given measurements for the current time step.

        If no prior update was done, this function is used to initialize the tracker estimates.

        :param measurements: N measurements as Nx2 numpy ndarray
        :return: 7D array [x, y, vx, vy, theta, length, width]
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Perform a predict step for delta-t = 1
        :return: 7D array [x, y, vx, vy, theta, length, width]
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        Return the current state as 7D array [x, y, vx, vy, theta, length, width]
        :return: 7D array [x, y, vx, vy, theta, length, width]
        """
        pass

    @abstractmethod
    def get_state_and_cov(self):
        """
        Return the current state as 7D array [x, y, vx, vy, theta, length, width] and its corresponding covariance
        :return: state, cov as tuple of 7D array and 7x7 matrix
        """
        pass

    @abstractmethod
    def set_R(self, R):
        """
        Update the measurement noise covariance to a new value
        :param R: new measurement noise covariance
        """
        pass
