"""
Contains function for loading previously generated experiment data and creating corresponding visualization.
When this file is executed, all plots are generated.
"""
import matplotlib.pyplot as plt
import os.path
import numpy as np
from pandas import DataFrame

from src.utilities.experiment_utilities import plot_error_over_time, plot_grid, create_trajectory_visualization, \
    create_extent_visualization, plot_particle_count_barplot
from src.utilities.repo_structure import get_root


def plot_over_time_experiment(tracker_dict, errors, settings, runtimes, target_dir):
    """Generate plot for average error per time step of different trackers over the trajectory"""
    mark_turns = True
    plt.style.use(get_root() + "data/stylesheets/paper.mplstyle")
    plot_error_over_time(tracker_dict, errors, settings, runtimes,
                         tracker_names=list(tracker_dict.keys()),
                         filename=os.path.join(target_dir, "paper", "error_over_time"),
                         outside_legend=False,
                         mark_turns=mark_turns)

    plt.style.use(get_root() + "data/stylesheets/presentation.mplstyle")
    plot_error_over_time(tracker_dict, errors, settings, runtimes,
                         tracker_names=list(tracker_dict.keys()),
                         filename=os.path.join(target_dir, "presentation", "error_over_time"),
                         mark_turns=mark_turns)


def plot_grid_comparison(grid_proposed, grid_reference, proposed_name, reference_name, target_dir, parameters):
    """
    Plot the grid-based detailed experiment comparing the performance of two trackers with each other throughout a set
    of options for the intensity of the noise and the expected number of measurements.
    """
    plt.style.use(get_root() + "data/stylesheets/presentation.mplstyle")
    plot_grid(grid_proposed, grid_reference, proposed_name, reference_name,
              filename=os.path.join(target_dir, "presentation", "error_grid"),
              parameters=parameters)

    plt.style.use(get_root() + "data/stylesheets/paper.mplstyle")
    plot_grid(grid_proposed, grid_reference, proposed_name, reference_name,
              filename=os.path.join(target_dir, "paper", "error_grid"),
              parameters=parameters)


def plot_n_particles(tracker_dict, errors, settings, runtimes, target_dir):
    """Plot the evaluation of the impact of the number of particles"""
    plt.style.use(get_root() + "data/stylesheets/paper.mplstyle")
    plot_particle_count_barplot(tracker_dict, errors, settings, runtimes,
                                tracker_names=list(tracker_dict.keys()),
                                filename=os.path.join(target_dir, "paper", "n_particles"),
                                outside_legend=False)

    plt.style.use(get_root() + "data/stylesheets/presentation.mplstyle")
    plot_particle_count_barplot(tracker_dict, errors, settings, runtimes,
                                tracker_names=list(tracker_dict.keys()),
                                filename=os.path.join(target_dir, "presentation", "n_particles"))


def plot_traj(target_dir):
    """Generate the visualization of the overall trajectory as well as of the object extent in the first turn"""
    adapt_colors_in_turn = True

    plt.style.use(get_root() + "data/stylesheets/paper.mplstyle")
    create_trajectory_visualization(filename=os.path.join(target_dir, "paper", "trajectory_overview"),
                                    adapt_colors_in_turn=adapt_colors_in_turn)
    create_extent_visualization(filename=os.path.join(target_dir, "paper", "extent_in_turn"))

    plt.style.use(get_root() + "data/stylesheets/presentation.mplstyle")
    create_trajectory_visualization(filename=os.path.join(target_dir, "presentation", "trajectory_overview"),
                                    adapt_colors_in_turn=adapt_colors_in_turn)
    create_extent_visualization(filename=os.path.join(target_dir, "presentation", "extent_in_turn"))


def plot_internal_comparison(tracker_dict, errors, settings, runtimes, target_dir):
    """Plot data evaluating different options for the internal wrapped EOT method used"""
    mark_turns = True
    plt.style.use(get_root() + "data/stylesheets/paper.mplstyle")
    plot_error_over_time(tracker_dict, errors, settings, runtimes,
                         tracker_names=list(tracker_dict.keys()),
                         filename=os.path.join(target_dir, "paper", "method_comparison"),
                         outside_legend=False,
                         mark_turns=mark_turns,
                         include_median_for=["Pure PF"])

    plt.style.use(get_root() + "data/stylesheets/presentation.mplstyle")
    plot_error_over_time(tracker_dict, errors, settings, runtimes,
                         tracker_names=list(tracker_dict.keys()),
                         filename=os.path.join(target_dir, "presentation", "method_comparison"),
                         mark_turns=mark_turns,
                         include_median_for=["Pure PF"])


def plot_runtime(runtimes_over_lambdas, parameters, target_dir):
    """Plot the evaluation of the runtime of different trackers for different number of measurements"""
    tracker_names = []
    tracker_runtimes = []
    lambdas = parameters['lambda_options']
    for k in runtimes_over_lambdas.keys():
        tracker_names.append(k)
        tracker_runtimes.append(runtimes_over_lambdas[k])
    df = DataFrame(np.asarray(tracker_runtimes).T)
    df = df.rename(
        {i: f"{tracker_names[i]}" for i in range(len(tracker_runtimes))},
        axis='columns'
    )  # set tracker name labels
    df: DataFrame = df.rename(
        {i: f"{lambdas[i]}" for i in range(len(lambdas))},
        axis='rows'
    )  # set n_measurements labels
    df = df.round(2)

    # save to tex file
    with open(os.path.join(target_dir, "paper", "runtimes.tex"), "w") as f:
        df.to_latex(buf=f)

    with open(os.path.join(target_dir, "paper", "runtimes_ref.tex"), "w") as f:
        df[[key for key in df.keys() if "PF" not in key]].to_latex(buf=f)

    with open(os.path.join(target_dir, "paper", "runtimes_pf.tex"), "w") as f:
        df[[key for key in df.keys() if "PF" in key]].to_latex(buf=f)


def main(target_dir):
    """
    Visualize all experiments, by doing:
    For all experiments:
        Try to load the corresponding data from file
        If successful: Call the respective visualization function for the loaded data
    """
    try:
        fp_over_time = os.path.join(target_dir, "data", "over_time_data.npy")
        over_time_data: dict = np.load(fp_over_time, allow_pickle=True).astype(dict).item()
        print(f"Successfully loaded data from {fp_over_time}")
        plot_over_time_experiment(target_dir=target_dir,
                                  **over_time_data
                                  )
    except FileNotFoundError as e:
        print(f"File not found error {e}")

    try:
        fp_grid = os.path.join(target_dir, "data", "grid_data.npy")
        grid_data: dict = np.load(fp_grid, allow_pickle=True).astype(dict).item()
        print(f"Successfully loaded data from {fp_grid}")
        plot_grid_comparison(target_dir=target_dir,
                             **grid_data
                             )
    except FileNotFoundError as e:
        print(f"File not found error {e}")

    try:
        fp_nparticles = os.path.join(target_dir, "data", "n_particles_data.npy")
        nparticle_data: dict = np.load(fp_nparticles, allow_pickle=True).astype(dict).item()
        print(f"Successfully loaded data from {fp_nparticles}")
        print("Generating n_particle plots")
        plot_n_particles(target_dir=target_dir,
                         **nparticle_data
                         )
    except FileNotFoundError as e:
        print(f"File not found error {e}")

    # ---
    try:
        print(f"Generating trajectory plots")
        plot_traj(target_dir=target_dir)
    except FileNotFoundError as e:
        print(f"File not found error {e}")

    try:
        fp_internal = os.path.join(target_dir, "data", "method_comparison.npy")
        internal_comp_data: dict = np.load(fp_internal, allow_pickle=True).astype(dict).item()
        print(f"Successfully loaded data from {fp_internal}")
        plot_internal_comparison(target_dir=target_dir, **internal_comp_data)
    except FileNotFoundError as e:
        print(f"File not found error {e}")

    try:
        fp_runtime = os.path.join(target_dir, "data", "runtime_data.npy")
        runtime_data: dict = np.load(fp_runtime, allow_pickle=True).astype(dict).item()
        print(f"Successfully loaded data from {fp_runtime}")
        plot_runtime(target_dir=target_dir, **runtime_data)
    except FileNotFoundError as e:
        print(f"File not found error {e}")


if __name__ == '__main__':
    target_dir = get_root() + "output/results/"
    main(target_dir)
