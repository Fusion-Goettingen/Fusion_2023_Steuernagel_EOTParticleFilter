import numpy as np
from src.trackers_elliptical.abstract_tracking import AbtractEllipticalTracker
from src.utilities.utils import rot, get_initial_estimate
from copy import deepcopy
from scipy.stats import multivariate_normal
from src.utilities.utils import ellipse_size_to_D, state_to_srs, srs_to_state


class NaiveSIRTracker(AbtractEllipticalTracker):
    """
    Implements a "naive" standard particle filter for the extended object tracking problem, which uses a constant
    velocity model.
    """

    def __init__(self,
                 R,
                 Q,
                 rng: np.random.Generator,
                 n_particles=100,
                 ):
        """
        Create a new particle filter instance

        :param R: Measurement noise as 2x2 matrix
        :param Q: Process Noise as 7x7 matrix -> [x, y, v_x, v_y, theta, length, width]
        :param rng: np.random.Generator object used for all internal randomized operations
        :param n_particles: Number of particles to use
        """
        self.rng = rng

        # Angles will be bounded in _angle_range_factor * pi - highly suggested as 2, only other sensible choice is 1
        self._angle_range_factor = 2

        # init particles
        self.particles = None
        self.n_particles = n_particles

        self.R = np.array(R).reshape((2, 2))
        self._full_Q = np.array(Q)
        assert self._full_Q.shape == (7, 7), f"Q is {self._full_Q.shape}-d, but should be 7x7!"

    def predict(self):
        """Predict step - for this naive PF, has no effect, as everything is handled in self.update(..)!"""
        pass  # nothing to do

    def init_particles(self, measurements):
        """
        Overwrite particles with initial estimates based on measurements.
        An initial estimate is acquired from the measurements, which is then used as the mean of a Gaussian with
        covariance set to the process noise of the PF for acquiring a set of initial particle estimates.
        The weight of all particles is set uniformly.

        :param measurements: Nx2 measurements around which the initial particle estimates are centered
        """
        # get initial estimate as mean of initial guess
        init = get_initial_estimate(measurements, merge_return=True)
        # create Nx8 array to fill
        self.particles = np.full((self.n_particles, len(init) + 1), 1 / self.n_particles)
        # overwrite everything but weights
        self.particles[:, :-1] = self.rng.multivariate_normal(mean=init,
                                                              cov=self._full_Q,
                                                              size=self.n_particles)

    def update(self, measurements: np.ndarray):
        # 0) Optional init. if no update was done so far
        if self.particles is None:
            return self.init_particles(measurements)

        # 1) Sample new particle states from the proposal density
        # apply cv model
        self.particles[:, :2] += self.particles[:, 2:4]  # add velocity to location
        for i in range(len(self.particles)):
            self.particles[i, :-1] = self.rng.multivariate_normal(mean=self.particles[i, :-1],
                                                                  cov=self._full_Q)
            # normalize orientation
            self.particles[i, 4] = self.particles[i, 4] % (self._angle_range_factor * np.pi)

            # calculate particle weights

            self.particles[i, -1] = self._get_particle_likelihood(Z=measurements,
                                                                  state=self.particles[i, :-1],
                                                                  cov=np.zeros((7, 7)))

        # 2) normalize particle weights
        self.normalize_weights()

        # 3) resample
        self.resample()

        # 4) consolidated estimate
        # return self.get_state()  # code explicitly calls get_state() when needed

    def normalize_weights(self):
        """Normalize the particle weights"""
        if np.sum(self.particles[:, -1]) == 0:
            # TODO: Handling edge case of numerical errors causing problems
            self.particles[:, -1] = 1 / len(self.particles)
        self.particles[:, -1] /= np.sum(self.particles[:, -1])

    def get_state(self):
        """
        Returns the current state estimate, based on a weighted average in square root space of all particles for the
        shape and a standard weighted average for the kinematics.
        :return: 7D consolidated state estimate of the PF
        """
        if self.particles is None:
            raise ValueError("Acquiring PF state before measurements were received!")

        # get all states
        state_list = np.array(self.particles[:, :-1])
        # get weighted average of all states
        avg_kinematics = np.average(state_list[:, :4], axis=0, weights=self.particles[:, -1])

        # get average extent in SquareRootSpace
        extents = deepcopy(state_list[:, 4:]).astype(float)
        srs_stack = np.array([state_to_srs(extents[i]) for i in range(len(extents))])
        srs_avg = np.average(srs_stack, axis=0, weights=self.particles[:, -1])
        avg_extent = srs_to_state(srs_avg)
        return np.asarray([*avg_kinematics, *avg_extent])

    def get_state_and_cov(self):
        raise NotImplementedError

    def set_R(self, R):
        assert R.shape == self.R.shape
        self.R = R

    def resample(self):
        """
        Resample all particles, resetting weights to uniform values.
        """
        n_particles = len(self.particles)
        # pick n_particles-many random rows from the particle array, with replacement, with prob. according to weights
        self.particles = self.particles[np.random.choice(len(self.particles), size=n_particles,
                                                         p=self.particles[:, -1], replace=True)]
        self.particles[:, -1] = 1 / len(self.particles)

    def _get_particle_likelihood(self, Z, state, cov):
        """
        Calculate the likelihood of a given state [x, y, vel_x, vel_y, orientation, length, width] given a set of
        measurements Z

        :param Z: Measurements as |Z|x2 array
        :param state: [x, y, vel_x, vel_y, orientation, length, width]
        :return: Likelihood of state given measurements (and optionally given further internal parameters)
        """
        # p(z|state) ~ N(z; [x, y], 0.25*Rot(theta)@D@Rot(theta).T + meas_noise_cov + P_loc)
        shape_mat = rot(state[4]) @ ellipse_size_to_D(state[5], state[6]) @ rot(state[4]).T
        mvn = multivariate_normal(mean=state[:2],
                                  cov=shape_mat + self.R + cov[:2, :2])
        w = np.sum(mvn.logpdf(Z))
        return np.exp(w)
