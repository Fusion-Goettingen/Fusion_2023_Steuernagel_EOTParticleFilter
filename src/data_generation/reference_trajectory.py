import numpy as np
from scipy.linalg import block_diag

from src.data_generation.elliptical_objects import get_ellipse_measurements
from src.utilities.utils import rot
from src.utilities.repo_structure import get_root

N_STEPS_IN_REF_TRAJ = 58


def get_reference_trajectory_data(rng: np.random.Generator,
                                  R,
                                  measurement_lambda,
                                  object_length,
                                  object_width,
                                  get_measurements=get_ellipse_measurements,
                                  rotate_orientation=True,
                                  min_measurements=3,
                                  polar_noise=False,
                                  init_pos=None
                                  ):
    R = np.array(R).reshape((2, 2))

    if init_pos is None:
        init_pos = [0, 0]

    measurement_state_dict = {
        "gt_state": [],
        "measurements": [],
    }

    speed = max(object_width, object_length) / 1.5
    initial_state = np.array([*rng.multivariate_normal(init_pos, cov=R),
                              speed, speed,
                              np.pi / 4, object_length, object_width])

    current_state = np.array(initial_state)

    # state transition setup
    F_cv = block_diag([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.eye(3))

    steps_in_turn = 6
    steps_in_long_line = 12
    steps_in_short_line = 5

    def update_orientation(state_to_update):
        if rotate_orientation:
            state_to_update[4] = (state_to_update[4] + np.pi / 16) % (2 * np.pi)
        else:
            state_to_update[4] = np.arctan2(state_to_update[3], state_to_update[2])
        return state_to_update

    def perform_linear_step(state_to_update):
        state_to_update = F_cv @ state_to_update
        return update_orientation(state_to_update)

    def perform_rotation_step_right(state_to_update):
        rotation_angle = (np.pi / 4 + np.pi / 2) / steps_in_turn
        rmat = rot(-rotation_angle)
        state_to_update[2:4] = rmat @ state_to_update[2:4]
        return perform_linear_step(state_to_update)

    def perform_rotation_step_left(state_to_update):
        rotation_angle = (np.pi / 4 + np.pi / 2) / steps_in_turn
        rmat = rot(rotation_angle)
        state_to_update[2:4] = rmat @ state_to_update[2:4]

        return perform_linear_step(state_to_update)

    update_function_list = [*[perform_linear_step] * steps_in_long_line,  # long path
                            *[perform_rotation_step_right] * steps_in_turn,  # first turn
                            *[perform_linear_step] * steps_in_short_line,  # down again
                            *[perform_rotation_step_right] * steps_in_turn,  # second turn
                            *[perform_linear_step] * steps_in_long_line,  # long path
                            *[perform_rotation_step_left] * steps_in_turn,  # third turn
                            *[perform_linear_step] * steps_in_short_line,  # down again
                            *[perform_rotation_step_left] * steps_in_turn,  # fourth turn
                            ]

    for step_ix, update_fct in enumerate(update_function_list):
        # sample data
        _, Z = get_measurements(loc=current_state[:2],
                                length=current_state[5],
                                width=current_state[6],
                                theta=current_state[4],
                                R=R,
                                n_measurements=np.max([min_measurements, rng.poisson(lam=measurement_lambda)]),
                                internal_RNG=rng,
                                polar_noise=polar_noise)

        measurement_state_dict["measurements"].append(Z)
        measurement_state_dict["gt_state"].append(current_state)

        current_state = update_fct(current_state)

    measurement_state_dict["gt_state"] = np.asarray(measurement_state_dict["gt_state"])
    return measurement_state_dict


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from src.utilities.visuals import plot_elliptic_extent
    plt.style.use(get_root() + "data/stylesheets/presentation.mplstyle")
    data = get_reference_trajectory_data(rng=np.random.default_rng(42),
                                         R=np.eye(2) * 0.25,
                                         measurement_lambda=12,
                                         object_length=5,
                                         object_width=2,
                                         get_measurements=get_ellipse_measurements)
    print(f"Acquired data for {len(data['gt_state'])} steps")

    colors = np.array(cm.rainbow(np.linspace(0, 1, len(data["gt_state"]))))
    for Z, gt, c in zip(data["measurements"], data["gt_state"], colors):
        plot_elliptic_extent(gt[:4], gt[4:], alpha=0.75, color=c)
        plt.scatter(Z[:, 0], Z[:, 1], color=c, marker='.', alpha=0.5)
    plt.axis('equal')
    plt.show()
