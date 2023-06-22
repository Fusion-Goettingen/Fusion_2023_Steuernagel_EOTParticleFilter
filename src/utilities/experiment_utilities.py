"""
Contains functions used during the conducted experiments, both for result acquisition and visualization.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import time

from src.utilities.metrics import gwd_full_state
from src.data_generation.reference_trajectory import get_reference_trajectory_data, N_STEPS_IN_REF_TRAJ
from src.data_generation.elliptical_objects import get_ellipse_measurements
from src.utilities.visuals import create_grid_plot, plot_elliptic_state, mark_standard_trajectory_turn_in_plot


def get_error_data(tracker_dict: dict,
                   rng: np.random.Generator,
                   reference_trajectory_settings: dict,
                   n_monte_carlo_runs: int,
                   use_tqdm=True,
                   ):
    """
    Given trackers and needed settings, run the trackers on the trajectories, evaluate the errors and runtimes, and
    return the results
    :param tracker_dict: Dictionary of trackers, each entry containing a dict which contains "instance", a not init.
    Tracker instance, and color, a valid matplotlib color string
    :param rng: RNG object used throughout the experiment
    :param reference_trajectory_settings:Setting used to get the reference trajectory data to be run on
    :param n_monte_carlo_runs: Number of monte carlo runs to perform
    :return: (errors, runtimes) as dicts
    """
    tracker_names = np.array(list(tracker_dict.keys()))
    errors = {tracker_id: np.zeros((n_monte_carlo_runs, N_STEPS_IN_REF_TRAJ)) for tracker_id in tracker_names}
    runtimes = {tracker_id: np.zeros((n_monte_carlo_runs, N_STEPS_IN_REF_TRAJ)) for tracker_id in tracker_names}
    for run_ix in tqdm(range(n_monte_carlo_runs), disable=(not use_tqdm)):
        current_run_trackers = deepcopy(tracker_dict)
        this_run_states = {tracker_id: [] for tracker_id in tracker_names}
        data = get_reference_trajectory_data(rng=rng,
                                             get_measurements=get_ellipse_measurements,
                                             **reference_trajectory_settings)

        for step_ix, measurements_and_gt in enumerate(zip(data["measurements"], data["gt_state"])):
            Z, gt = measurements_and_gt
            for tracker_id in tracker_names:
                start_time = time.time()
                # update
                current_run_trackers[tracker_id]["instance"].update(Z)
                # extract state
                this_run_states[tracker_id].append(current_run_trackers[tracker_id]["instance"].get_state())
                # save error
                errors[tracker_id][run_ix, step_ix] = gwd_full_state(this_run_states[tracker_id][-1], gt)
                # predict
                current_run_trackers[tracker_id]["instance"].predict()
                runtimes[tracker_id][run_ix, step_ix] = (time.time() - start_time) * 1000
        pass
    return errors, runtimes


def get_tracking_data(tracker_dict: dict,
                      rng: np.random.Generator,
                      reference_trajectory_settings: dict,
                      ):
    """
    Given trackers and needed settings, run the trackers on the trajectories, evaluate the errors and runtimes, and
    return the results
    :param tracker_dict: Dictionary of trackers, each entry containing a dict which contains "instance", a not init.
    Tracker instance, and color, a valid matplotlib color string
    :param rng: RNG object used throughout the experiment
    :param reference_trajectory_settings:Setting used to get the reference trajectory data to be run on
    :return: (states, errors, runtimes) as dicts
    """
    # init
    tracker_names = np.array(list(tracker_dict.keys()))
    errors = {tracker_id: np.zeros((N_STEPS_IN_REF_TRAJ,)) for tracker_id in tracker_names}
    runtimes = {tracker_id: np.zeros((N_STEPS_IN_REF_TRAJ,)) for tracker_id in tracker_names}
    current_run_trackers = deepcopy(tracker_dict)
    states = {tracker_id: [] for tracker_id in tracker_names}
    data = get_reference_trajectory_data(rng=rng,
                                         get_measurements=get_ellipse_measurements,
                                         **reference_trajectory_settings)

    # iterate
    for step_ix, measurements_and_gt in enumerate(zip(data["measurements"], data["gt_state"])):
        Z, gt = measurements_and_gt
        for tracker_id in tracker_names:
            start_time = time.time()
            # update
            current_run_trackers[tracker_id]["instance"].update(Z)
            # extract state
            states[tracker_id].append(current_run_trackers[tracker_id]["instance"].get_state())
            # save error
            errors[tracker_id][step_ix] = gwd_full_state(states[tracker_id][-1], gt)
            # predict
            current_run_trackers[tracker_id]["instance"].predict()
            runtimes[tracker_id][step_ix] = (time.time() - start_time) * 1000

    # numpy-fy states
    for tracker_id in tracker_names:
        states[tracker_id] = np.array(states[tracker_id])

    # finalize
    return states, errors, runtimes, np.array(data["measurements"]), np.array(data["gt_state"])


def plot_error_over_time(tracker_dict, errors, settings, runtimes, tracker_names=None,
                         filename=None, verbose=False, outside_legend=True, mark_turns=False,
                         include_median_for=None):
    """Create a plot of the errors of a set of trackers, as a line plot over time"""
    if tracker_names is None:
        tracker_names = np.array(list(tracker_dict.keys()))

    to_pop = []
    for t_name in tracker_names:
        if t_name not in tracker_dict.keys():
            print(f"{t_name} not in trackers, skipping")
            to_pop.append(t_name)
    for name in to_pop:
        tracker_names.pop(tracker_names.index(name))
    if len(tracker_names) == 0:
        print("All trackers skipped or none given, skipping")
        return

    longest_name_length = np.max([len(t) for t in tracker_names])
    # iterate over trackers_elliptical
    if verbose:
        print("GWD:")
    for t_id in tracker_names:
        color = tracker_dict[t_id]["color"]
        tracker_error_avg = errors[t_id]
        if include_median_for is None or t_id not in include_median_for:
            avg_over_runs = np.average(tracker_error_avg, axis=0)
            plt.plot(avg_over_runs, color=color, label=f"{t_id}: {np.average(avg_over_runs):5.2f}")
            if verbose:
                print(f"\t{t_id:{longest_name_length}s}: {np.average(avg_over_runs):5.2f}")
        else:
            avg_over_runs = np.average(tracker_error_avg, axis=0)
            plt.plot(avg_over_runs, color=color, label=f"{t_id} [Mean]: {np.average(avg_over_runs):5.2f}",
                     linestyle='--')
            median_over_runs = np.median(tracker_error_avg, axis=0)
            plt.plot(median_over_runs, color=color, label=f"{t_id} [Median]: {np.average(median_over_runs):5.2f}")
    if verbose:
        print("Time / ms:")
    for t_id in tracker_names:
        tracker_time_avg = runtimes[t_id]
        time_avg_over_runs = np.average(tracker_time_avg, axis=0)
        if verbose:
            print(f"\t{t_id:{longest_name_length}s}: {np.average(time_avg_over_runs):5.2f}")
    if mark_turns:
        mark_standard_trajectory_turn_in_plot()

    # show plot
    if outside_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    else:
        if len(tracker_dict) > 4:
            plt.ylim([plt.ylim()[0], plt.ylim()[1] + 4])
        # plt.legend(loc='upper right')
        plt.legend()
    plt.xlabel("Step in Trajectory")
    plt.ylabel(r"Squared GWD / m$^2$")
    title = f"Errors over Time - Velocity and Orientation {'de' if settings['rotate_orientation'] else ''}coupled"
    # plt.title(title)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(filename)
        plt.close()


def plot_grid(grid_proposed, grid_reference, name_proposed, name_reference, filename, parameters):
    """Given grids of two trackers, visualize the diff-grid between the two using a diverging colormap"""
    grid_comparison = grid_proposed - grid_reference
    # show grid
    create_grid_plot(grid_data=grid_comparison.T,
                     xlabel=r"Noise Level $r$ for $\mathbf{R} = \mathbf{I}\cdot r$", xticks=parameters["noise_options"],
                     ylabel=r"Measurement Rate Mean $\lambda$", yticks=parameters["lambda_options"],
                     cmap="seismic",
                     vmin=-np.max(np.abs(grid_comparison)),
                     vmax=np.max(np.abs(grid_comparison)),
                     show_cbar=True)

    # set up plot
    plt.tight_layout()

    # save
    if filename is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(filename)
        plt.close()


def plot_error_barplot(tracker_dict, errors, settings, runtimes, tracker_names=None,
                       filename=None, verbose=False, keep_order=False, outside_legend=True):
    """Visualize the average errors of a set of trackers as a barplot"""
    if tracker_names is None:
        tracker_names = np.array(list(tracker_dict.keys()))

    to_pop = []
    for t_name in tracker_names:
        if t_name not in tracker_dict.keys():
            print(f"{t_name} not in trackers, skipping")
            to_pop.append(t_name)
    for name in to_pop:
        tracker_names.pop(tracker_names.index(name))
    if len(tracker_names) == 0:
        print("All trackers skipped or none given, skipping")
        return

    longest_name_length = np.max([len(t) for t in tracker_names])
    # iterate over trackers_elliptical
    if verbose:
        print("GWD:")
    avg_over_runs = {}
    for t_id in tracker_names:
        avg_over_runs[t_id] = np.average(np.average(errors[t_id], axis=0), axis=0)
    if not keep_order:
        sorting_ix = np.argsort([avg_over_runs[t_id] for t_id in tracker_names])
        tracker_names = np.array(tracker_names)[sorting_ix][::-1]

    for t_id in tracker_names:
        color = tracker_dict[t_id]["color"]
        plt.bar(x=t_id, height=avg_over_runs[t_id], color=color,
                label=f"{t_id}: {avg_over_runs[t_id]:5.2f}")
        if verbose:
            print(f"\t{t_id:{longest_name_length}s}: {np.average(avg_over_runs):5.2f}")
    if verbose:
        print("Time / ms:")
    for t_id in tracker_names:
        tracker_time_avg = runtimes[t_id]
        time_avg_over_runs = np.average(tracker_time_avg, axis=0)
        if verbose:
            print(f"\t{t_id:{longest_name_length}s}: {np.average(time_avg_over_runs):5.2f}")
    # show plot
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    if outside_legend:
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    else:
        plt.legend()
    plt.ylabel("GWD / $m^2$")
    plt.xlabel("Tracking Method")
    title = f"Errors over Time - Velocity and Orientation {'de' if settings['rotate_orientation'] else ''}coupled"
    # plt.title(title)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(filename)
        plt.close()


def plot_particle_count_barplot(tracker_dict, errors, settings, runtimes, tracker_names=None,
                                filename=None, verbose=False, keep_order=False, outside_legend=True):
    """Create an average error barplot, specifically for visualizing the impact of the number of particles"""
    if tracker_names is None:
        tracker_names = np.array(list(tracker_dict.keys()))

    to_pop = []
    for t_name in tracker_names:
        if t_name not in tracker_dict.keys():
            print(f"{t_name} not in trackers, skipping")
            to_pop.append(t_name)
    for name in to_pop:
        tracker_names.pop(tracker_names.index(name))
    if len(tracker_names) == 0:
        print("All trackers skipped or none given, skipping")
        return

    longest_name_length = np.max([len(t) for t in tracker_names])
    # iterate over trackers_elliptical
    if verbose:
        print("GWD:")
    avg_over_runs = {}
    for t_id in tracker_names:
        avg_over_runs[t_id] = np.average(np.average(errors[t_id], axis=0), axis=0)
    if not keep_order:
        sorting_ix = np.argsort([avg_over_runs[t_id] for t_id in tracker_names])
        tracker_names = np.array(tracker_names)[sorting_ix][::-1]

    color = '#1f77b4'
    xs = np.array([
        str(t_id).replace('RBPF', '').replace('IAE', '').replace('-', '').replace('+', '').replace('PF', '')
        for t_id in tracker_names
    ])
    order_ixs = np.argsort(xs.astype(float))
    averages = np.array([
        avg_over_runs[t_id]
        for t_id in tracker_names
    ]).round(2)
    bar_plot_object = plt.bar(x=xs[order_ixs], height=averages[order_ixs], color=color)
    plt.gca().bar_label(bar_plot_object)

    if verbose:
        print("Time / ms:")
    for t_id in tracker_names:
        tracker_time_avg = runtimes[t_id]
        time_avg_over_runs = np.average(tracker_time_avg, axis=0)
        if verbose:
            print(f"\t{t_id:{longest_name_length}s}: {np.average(time_avg_over_runs):5.2f}")
    # show plot
    plt.ylabel(r"Squared GWD / m$^2$")
    plt.xlabel("Number of Particles")
    plt.ylim([0, plt.ylim()[1] * 1.1])
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(filename)
        plt.close()


def create_runtime_barplot(tracker_dict, errors, settings, runtimes, tracker_names=None,
                           filename=None, verbose=False, keep_order=False):
    """Create a barplot of the runtime of the individual methods"""
    if tracker_names is None:
        tracker_names = np.array(list(tracker_dict.keys()))

    to_pop = []
    for t_name in tracker_names:
        if t_name not in tracker_dict.keys():
            print(f"{t_name} not in trackers, skipping")
            to_pop.append(t_name)
    for name in to_pop:
        tracker_names.pop(tracker_names.index(name))
    if len(tracker_names) == 0:
        print("All trackers skipped or none given, skipping")
        return

    # iterate over trackers_elliptical
    avg_over_runs = {}
    for t_id in tracker_names:
        avg_over_runs[t_id] = np.average(np.average(runtimes[t_id], axis=0), axis=0)
    if not keep_order:
        sorting_ix = np.argsort([avg_over_runs[t_id] for t_id in tracker_names])
        tracker_names = np.array(tracker_names)[sorting_ix][::-1]

    for t_id in tracker_names:
        color = tracker_dict[t_id]["color"]
        plt.bar(x=t_id, height=avg_over_runs[t_id], color=color,
                label=f"{t_id}: {avg_over_runs[t_id]:5.2f}")
    # show plot
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.ylabel("Time / ms")
    plt.xlabel("Tracking Method")
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(filename)
        plt.close()


def create_trajectory_visualization(filename, adapt_colors_in_turn):
    """
    Visualize the experimental trajectory, without showing the target extend.
    Can optionally adapt the colors during the first turn, matching the create_extent_visualization(...) function.
    """
    R = np.eye(2) * 1

    rng = np.random.default_rng(42)

    data = get_reference_trajectory_data(rng=rng,
                                         measurement_lambda=5,
                                         R=R,
                                         object_length=10,
                                         object_width=4,
                                         min_measurements=3,
                                         )

    locations = data["gt_state"][:, :2]

    plt.scatter(*locations[0, :], marker='o', s=850, c='#1f77b4')
    if adapt_colors_in_turn:
        color = np.array(['#1f77b4'] * len(locations))
        ixs = range(9, 20)
        color[ixs] = '#ff7f0e'
    else:
        color = '#1f77b4'
    plt.plot(*locations.T, linestyle='--', markersize=15, color='#1f77b4', zorder=-1)
    plt.scatter(*locations.T, marker='o', s=150, color=color, zorder=1)

    plt.xlabel("$m_1$ / m")
    plt.ylabel("$m_2$ / m")
    plt.axis('equal')

    if filename is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(filename)
        plt.close()


def create_extent_visualization(filename):
    """Visualize the extent of the target in the first turn of the reference trajectory"""
    R = np.eye(2) * 1

    rng = np.random.default_rng(101)

    data = get_reference_trajectory_data(rng=rng,
                                         measurement_lambda=5,
                                         R=R,
                                         object_length=10,
                                         object_width=4,
                                         min_measurements=3,
                                         )

    ixs = range(9, 20)
    states = data["gt_state"][ixs, :]
    measurements = np.array(data["measurements"], dtype=object)[ixs]

    for state in states:
        plot_elliptic_state(state, color='#ff7f0e', fill=True, alpha=0.5)
    for Z in measurements:
        plt.scatter(*Z.T, c='k', marker='.')

    plt.xlabel("$m_1$ / m")
    plt.ylabel("$m_2$ / m")
    plt.axis('equal')

    if filename is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(filename)
        plt.close()
