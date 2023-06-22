import numpy as np
from src.trackers_elliptical.abstract_tracking import AbtractEllipticalTracker
from src.utilities.utils import rot, get_initial_estimate
from copy import deepcopy
from scipy.stats import multivariate_normal
from src.utilities.utils import ellipse_size_to_D, state_to_srs, srs_to_state
from src.trackers_elliptical.eot_modules import iae_step, memekf_step, pakf_step


class EOTPFTracker(AbtractEllipticalTracker):
    """
    Implements a particle filter-based augmentation of an extended object tracker, where the PF handles orientation
    estimation.
    """

    def __init__(self,
                 R,
                 Q,
                 q_lw,
                 base_tracker,
                 n_particles=100,
                 resampling_var: float = 0.05,
                 rng: np.random.Generator = None,
                 time_step_length=1,
                 ):
        """
        Create a new instance of the PF

        :param R: 2x2 Measurement Noise
        :param Q: 4x4 Process Noise of the kinematic state (location, velocity)
        :param q_lw: 2x1 diagonal of the process noise covariance matrix of the semi-axes lengths.
        :param base_tracker: oneof ['iae', 'memekf', 'pakf'] - determines the way in which the update is handled by
        the tracker
        :param n_particles: Number of particles
        :param resampling_var: Variance used for resampling the orientation estimates of the particles.
        :param rng: Random number generator used for all internal randomization
        :param time_step_length: Length of a time step, defaults to 1.
        """
        self.rng = rng
        self.base_tracker = base_tracker
        self.R = R

        # Angles will be bounded in _angle_range_factor * pi - highly suggested as 2, only other sensible choice is 1
        self._angle_range_factor = 2

        # init particles
        self.particles = None
        self.resampling_var = resampling_var
        self.n_particles = n_particles

        self.init_P_m = Q  # non-zero init in order for velocity estimation to work!
        self.init_p_lw = np.array(q_lw)
        self.H = np.block([np.eye(2), np.zeros((2, 2))])
        self.F = np.array([
            [1, 0, time_step_length, 0],
            [0, 1, 0, time_step_length],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.Q = Q
        self.q_lw = np.array(q_lw)

    def predict(self):
        """
        Predict step - for this PF, has no effect, as everything is handled in self.update(..)!
        The EOT-predict step is called jointly with the update step of the desired wrapped tracking method as part of
        the particle filter update.
        """
        pass  # nothing to do

    def init_particles(self, measurements):
        """
        Overwrite particles with initial estimates based on measurements.
        An initial estimate is acquired from the measurements. This estimate is used for the location and semi-axes
        lengths. Velocity is set to 0, and the orientation is uniformly sampled from [0, 2pi]. Weights are initialized
        uniformly.

        :param measurements: Nx2 measurements around which the initial particle estimates are centered
        """
        m, p = get_initial_estimate(measurements=measurements)

        # set for all particles
        self.particles = np.array([
            [*m,  # location + velocity
             self.rng.uniform(low=0, high=self._angle_range_factor * np.pi),  # theta randomized
             *p[1:],  # l w
             1 / self.n_particles,  # weight
             ]
            for i in range(self.n_particles)
        ])

    def update(self, measurements: np.ndarray):
        # 0) Optional init. if no update was done so far
        if self.particles is None:
            self.init_particles(measurements)
            self.particles: np.ndarray
            for i in range(len(self.particles)):
                next_state = self.particles[i, :-1]
                next_cov = np.zeros((7, 7))
                self.particles[i, -1] = self._get_particle_likelihood(Z=measurements, state=next_state, cov=next_cov)

                # 2) normalize particle weights
            self.normalize_weights()

            # 3) resample
            self.resample()

            return

        pass  # end of optional init
        # ==========================

        # 1) Update all particle states
        # a) sample a new orientation for each particle
        self.particles[:, 4] = self.rng.normal(loc=self.particles[:, 4].astype(float),
                                               scale=np.sqrt(self.resampling_var)) % (
                                       self._angle_range_factor * np.pi)

        # b) extended object tracker based proposal for the current state
        for i in range(len(self.particles)):
            # self.base_tracker controls which EOT algorithm is used
            if self.base_tracker == "iae":
                next_state, next_cov = iae_step(Z=measurements,
                                                x=self.particles[i, :4].astype(float),
                                                P=self.init_P_m,
                                                l=self.particles[i, 5:7].astype(float),
                                                c=self.init_p_lw,
                                                H=self.H,
                                                F=self.F,
                                                R=self.R,
                                                Q=self.Q,
                                                q_c=self.q_lw,
                                                orientation=self.particles[i, 4])

            elif self.base_tracker == "memekf":
                next_state, next_cov = memekf_step(Z=measurements,
                                                   x=self.particles[i, :4].astype(float),
                                                   p=self.particles[i, 4:7].astype(float),
                                                   C_m=self.init_P_m,
                                                   C_p=np.diag([0, *self.init_p_lw]),
                                                   H=self.H,
                                                   F=self.F,
                                                   R=self.R,
                                                   Q=self.Q,
                                                   q_extent=np.diag([0, *self.init_p_lw]),
                                                   orientation=self.particles[i, 4])
            elif self.base_tracker == "pakf":
                r_lw = np.trace(self.R) / len(measurements)
                next_state, next_cov = pakf_step(Z=measurements,
                                                 x=self.particles[i, :4].astype(float),
                                                 p=self.particles[i, 4:7].astype(float),
                                                 C_x=self.init_P_m * 10,
                                                 C_p=np.diag([0, *self.init_p_lw]),
                                                 H=self.H,
                                                 H_p=np.eye(3),
                                                 F=self.F,
                                                 R=self.R,
                                                 R_p=np.diag([1e-10, r_lw, r_lw]),
                                                 Q=self.Q,
                                                 Q_extent=np.diag([0, *self.init_p_lw]),
                                                 orientation=self.particles[i, 4])
            else:
                raise NotImplementedError(f"Unknown base EOT algorithm '{self.base_tracker}'")

            # save state
            self.particles[i, :-1] = next_state

            # c) calculate particle weights
            self.particles[i, -1] = self._get_particle_likelihood(Z=measurements, state=next_state, cov=next_cov)

        # 2) normalize particle weights
        self.normalize_weights()

        # 3) resample
        self.resample()

        # [4) consolidated estimate: code explicitly calls get_state() when needed]

    def normalize_weights(self):
        """Normalize the particle weights"""
        if np.sum(self.particles[:, -1]) == 0:
            # sum of particle weights is zero, typically due to numerical instability
            # replace weights with uniform ones
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
        if self.base_tracker != "pakf":
            avg_extent[1:] *= 2  # to full-axis
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
