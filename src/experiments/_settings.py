import numpy as np

from src.trackers_elliptical.iae import TrackerIAE
from src.trackers_elliptical.rm import TrackerRM
from src.trackers_elliptical.memekf import TrackerMEMEKF
from src.trackers_elliptical.principal_axes_kf import PrincipalAxesKalmanFilter
from src.trackers_elliptical.eot_particle_filter import EOTPFTracker
from src.trackers_elliptical.standard_sir_eot import NaiveSIRTracker


class SettingsManager:
    """
    Small helper class managing creation of all needed settings etc. for the experiments.
    Not parameterized.
    """

    def __init__(self):
        self._n_runs = 200
        self._n_runs_internal = 100
        self._n_runs_grid = 50

        self._obj_len = 10
        self._obj_width = 4

        self._P_init = np.diag([3, 3, 64, 64])

        self._q_loc, self._q_speed = 1, 2
        self._Q = np.diag([self._q_loc, self._q_loc, self._q_speed, self._q_speed])
        self._q_lw = np.array([0.15, 0.15])

        self._Q_extent = np.diag([0.1, 1e-8, 1e-8])

        self._rm_v = 25
        self._rm_tau = 3

        self._c_init = [5, 5]

        # rb settings
        self._default_n_particles = 50
        self._default_resampling_var = 0.1
        self._rng_seed = 42

        # other
        self._main_seed = 101

        self._r_factor = 2
        self._measurement_lambda = 12

    def get_n_runs(self):
        return self._n_runs

    def _get_settings(self, r_factor, measurement_lambda, rotate=True):
        return dict(
            R=np.eye(2) * r_factor,
            measurement_lambda=measurement_lambda,
            object_length=self._obj_len,
            object_width=self._obj_width,
            rotate_orientation=rotate,
        )

    def get_tracker_dict(self, tracker_names, settings):
        # catch single tracker_name passed as parameter
        if type(tracker_names) == str:
            tracker_names = [tracker_names]

        trackers = {}  # dict of trackers to be filled

        # MEM-EKF*
        kwargs_memekf = dict(
            C_m_init=self._P_init,
            C_p_init=np.diag([np.pi / 2, 2, 2]),
            R=settings["R"],
            Q=self._Q,
            Q_extent=self._Q_extent
        )

        # IAE
        kwargs_iae = dict(
            P_init=self._P_init,
            c_init=self._c_init,
            R=settings["R"],
            Q=self._Q
        )

        # RM
        kwargs_rm = dict(
            v=self._rm_v,
            P=self._P_init,
            Q=self._Q,
            R=settings["R"],
            tau=self._rm_tau,
        )

        # PAKF
        kwargs_pakf = dict(
            C_x_init=self._P_init,
            C_p_init=np.diag([np.pi / 2, 2, 2]),
            R=settings["R"],
            R_p=None,
            Q=self._Q,
            Q_extent=self._Q_extent
        )

        # add trackers to dict as desired
        if "MEM-EKF*" in tracker_names:
            trackers["MEM-EKF*"] = {"instance": TrackerMEMEKF(**kwargs_memekf),
                                    "color": 'red'
                                    }
        if "RM" in tracker_names:
            trackers["RM"] = {
                "instance": TrackerRM(**kwargs_rm),
                "color": 'black',
            }
        if "IAE" in tracker_names:
            trackers["IAE"] = {
                "instance": TrackerIAE(**kwargs_iae),
                "color": 'green',
            }
        if "PAKF" in tracker_names:
            trackers["PAKF"] = {
                "instance": PrincipalAxesKalmanFilter(**kwargs_pakf),
                "color": 'fuchsia',
            }
        if "PF+IAE" in tracker_names:
            trackers["PF+IAE"] = {
                "instance": EOTPFTracker(settings["R"],
                                         Q=self._Q,
                                         q_lw=self._q_lw,
                                         base_tracker="iae",
                                         n_particles=self._default_n_particles,
                                         resampling_var=self._default_resampling_var,
                                         rng=np.random.default_rng(self._rng_seed)),
                "color": "blue"
            }
        if "PF+MEM-EKF*" in tracker_names:
            trackers["PF+MEM-EKF*"] = {
                "instance": EOTPFTracker(settings["R"],
                                         Q=self._Q,
                                         q_lw=[1e-4, 1e-4],
                                         base_tracker="memekf",
                                         n_particles=self._default_n_particles,
                                         resampling_var=self._default_resampling_var,
                                         rng=np.random.default_rng(self._rng_seed)),
                "color": "red"
            }
        if "PF+PAKF" in tracker_names:
            trackers["PF+PAKF"] = {
                "instance": EOTPFTracker(settings["R"],
                                         Q=self._Q,
                                         q_lw=self._q_lw,
                                         base_tracker="pakf",
                                         n_particles=self._default_n_particles,
                                         resampling_var=self._default_resampling_var,
                                         rng=np.random.default_rng(self._rng_seed)),
                "color": "red"
            }

        if "Pure PF" in tracker_names:
            full_Q = np.diag([1e-10, 1e-10, 20, 20, 0.1, 0.66, 0.66])
            trackers["Pure PF"] = {
                "instance": NaiveSIRTracker(
                    Q=full_Q,
                    R=settings["R"],
                    n_particles=5000,
                    rng=np.random.default_rng(self._rng_seed)
                ),
                "color": "black"
            }
        # return everything
        return trackers

    def get_tracker_dict_n_particles(self, options, settings):
        # IAE
        kwargs_iae = dict(
            P_init=self._P_init,
            c_init=self._c_init,
            R=settings["R"],
            Q=self._Q
        )

        trackers = {
            f"PF+IAE-{n_particles}": {
                "instance": EOTPFTracker(settings["R"],
                                         Q=self._Q,
                                         q_lw=self._q_lw,
                                         base_tracker="iae",
                                         n_particles=n_particles,
                                         resampling_var=self._default_resampling_var,
                                         rng=np.random.default_rng(self._rng_seed)),
                "color": "blue"
            }
            for n_particles in options
        }
        return trackers

    def over_time(self):
        r_factor = self._r_factor
        measurement_lambda = self._measurement_lambda

        rng = np.random.default_rng(self._main_seed)
        n_runs = self._n_runs
        settings = self._get_settings(r_factor=r_factor,
                                      measurement_lambda=measurement_lambda,
                                      rotate=True)

        trackers = self.get_tracker_dict([
            "MEM-EKF*",
            "RM",
            "IAE",
            "PAKF",
            "PF+IAE"
        ], settings=settings)
        return rng, n_runs, trackers, settings

    def comparison(self, r, lam):
        rng = np.random.default_rng(self._main_seed)
        n_runs = self._n_runs_grid
        settings = self._get_settings(r_factor=r,
                                      measurement_lambda=lam,
                                      rotate=True)

        trackers = self.get_tracker_dict([
            "PF+IAE",
            "MEM-EKF*",
        ], settings=settings)

        return rng, n_runs, trackers, settings

    def particle_count(self):
        r_factor = self._r_factor
        measurement_lambda = self._measurement_lambda

        rng = np.random.default_rng(self._main_seed)
        n_runs = self._n_runs
        settings = self._get_settings(r_factor=r_factor,
                                      measurement_lambda=measurement_lambda,
                                      rotate=True)

        trackers = self.get_tracker_dict_n_particles([
            10, 25, 50, 100
        ], settings=settings)

        return rng, n_runs, trackers, settings

    def predictive_comparison(self):
        rng = np.random.default_rng(self._main_seed)
        n_runs = self._n_runs
        settings = self._get_settings(r_factor=self._r_factor,
                                      measurement_lambda=self._measurement_lambda,
                                      rotate=True)

        trackers = self.get_tracker_dict([
            "PF+IAE",
            "MEM-EKF*",
            # "Pure PF",
            "Diffusion PF+IAE",
        ], settings=settings)

        return rng, n_runs, trackers, settings

    def internal_comparison(self):
        rng = np.random.default_rng(self._main_seed)
        n_runs = self._n_runs_internal
        settings = self._get_settings(r_factor=self._r_factor,
                                      measurement_lambda=self._measurement_lambda,
                                      rotate=True)

        trackers = self.get_tracker_dict([
            "PF+IAE",
            "PF+PAKF",
            "Pure PF",
        ], settings=settings)

        return rng, n_runs, trackers, settings

    def runtime(self, measurement_lambda):
        r_factor = self._r_factor

        rng = np.random.default_rng(self._main_seed)
        n_runs = self._n_runs_grid
        settings = self._get_settings(r_factor=r_factor,
                                      measurement_lambda=measurement_lambda,
                                      rotate=True)

        trackers_ref = self.get_tracker_dict([
            "MEM-EKF*",
            "RM",
            "IAE",
            "PAKF",
            "Pure PF"
        ], settings=settings)
        trackers_pf = self.get_tracker_dict_n_particles(options=[10, 25, 50], settings=settings)
        trackers = {**trackers_pf, **trackers_ref}
        return rng, n_runs, trackers, settings
