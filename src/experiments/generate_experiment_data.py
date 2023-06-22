"""
Contains functions that each generate data for a different experiment. When called directly, this script runs all
experiments presented in the paper.
"""
import numpy as np
from tqdm import tqdm
from itertools import product
import os.path

from src.utilities.experiment_utilities import get_error_data
from src.experiments._settings import SettingsManager
from src.utilities.repo_structure import get_root


def experiment_over_time(manager, target_dir):
    """Generate average error per time step of different trackers over the trajectory"""
    rng, n_runs, trackers, settings = manager.over_time()

    err_over_time, timings_over_time = get_error_data(tracker_dict=trackers, rng=rng,
                                                      reference_trajectory_settings=settings,
                                                      n_monte_carlo_runs=n_runs)
    tqdm.write("\nGenerating over Time data...")
    data_dict = {
        "tracker_dict": trackers,
        "errors": err_over_time,
        "runtimes": timings_over_time,
        "settings": settings
    }
    target_file = os.path.join(target_dir, "over_time_data.npy")
    np.save(target_file, data_dict)
    tqdm.write(f"Saved data to {target_file}")


def experiment_grid(manager: SettingsManager, target_dir, proposed_name="PF+IAE", reference_name="MEM-EKF*"):
    """
    Grid-based detailed experiment comparing the performance of two trackers with each other throughout a set of
    options for the intensity of the noise and the expected number of measurements.
    """
    noise_options = [0.1, 0.5, 1, 2, 3, 4]
    lambda_options = [4, 8, 12, 20, 40]

    grid_proposed = np.zeros((len(noise_options), len(lambda_options)))
    grid_reference = np.zeros((len(noise_options), len(lambda_options)))

    tqdm.write("\nGenerating Grid data...")
    for current_r, current_lambda in tqdm(product(noise_options, lambda_options),
                                          total=len(noise_options) * len(lambda_options)):
        rng, n_runs, trackers, settings = manager.comparison(current_r, current_lambda)
        err_over_time, _ = get_error_data(tracker_dict=trackers, rng=rng,
                                          reference_trajectory_settings=settings,
                                          n_monte_carlo_runs=n_runs, use_tqdm=False)

        avg_proposed = np.average(np.average(err_over_time[proposed_name], axis=0), axis=0)
        avg_reference = np.average(np.average(err_over_time[reference_name], axis=0), axis=0)
        # insert into array
        grid_proposed[noise_options.index(current_r), lambda_options.index(current_lambda)] = avg_proposed
        grid_reference[noise_options.index(current_r), lambda_options.index(current_lambda)] = avg_reference

    data_dict = {
        "grid_proposed": grid_proposed,
        "grid_reference": grid_reference,
        "proposed_name": proposed_name,
        "reference_name": reference_name,
        "parameters": {
            "noise_options": noise_options,
            "lambda_options": lambda_options,
            "n_runs": n_runs
        }
    }
    target_file = os.path.join(target_dir, "grid_data.npy")
    np.save(target_file, data_dict)
    tqdm.write(f"Saved data to {target_file}")


def experiment_runtime(manager: SettingsManager, target_dir):
    """Evaluation of the runtime of different trackers for different number of measurements"""
    lambda_options = [8, 40]
    rng, n_runs, trackers, settings = manager.runtime(42)  # only to get trackers once
    runtimes_over_lambdas = {t_id: [] for t_id in trackers.keys()}
    tqdm.write("\nGenerating Runtime data...")
    for current_lambda in tqdm(lambda_options):
        rng, n_runs, trackers, settings = manager.runtime(current_lambda)
        _, runtimes = get_error_data(tracker_dict=trackers, rng=rng,
                                     reference_trajectory_settings=settings,
                                     n_monte_carlo_runs=n_runs, use_tqdm=False)
        for t_id in trackers.keys():
            avg_for_tracker = np.average(np.average(runtimes[t_id], axis=0), axis=0)
            runtimes_over_lambdas[t_id].append(avg_for_tracker)

    data_dict = {
        "runtimes_over_lambdas": runtimes_over_lambdas,
        "parameters": {
            "lambda_options": lambda_options,
            "n_runs": n_runs
        }
    }
    target_file = os.path.join(target_dir, "runtime_data.npy")
    np.save(target_file, data_dict)
    tqdm.write(f"Saved data to {target_file}")


def experiment_nparticles(manager, target_dir):
    """Perform the evaluation of the impact of the number of particles"""
    rng, n_runs, trackers, settings = manager.particle_count()
    tqdm.write("\nGenerating #particles data...")
    err_over_time, timings_over_time = get_error_data(tracker_dict=trackers, rng=rng,
                                                      reference_trajectory_settings=settings,
                                                      n_monte_carlo_runs=n_runs)
    data_dict = {
        "tracker_dict": trackers,
        "errors": err_over_time,
        "runtimes": timings_over_time,
        "settings": settings
    }
    target_file = os.path.join(target_dir, "n_particles_data.npy")
    np.save(target_file, data_dict)
    tqdm.write(f"Saved data to {target_file}")


def experiment_method_comparison(manager: SettingsManager, target_dir):
    """Generate data to evaluate different options for the internal wrapped EOT method used"""
    rng, n_runs, trackers, settings = manager.internal_comparison()
    tqdm.write("\nGenerating Internal Comparison data...")
    err_over_time, timings_over_time = get_error_data(tracker_dict=trackers, rng=rng,
                                                      reference_trajectory_settings=settings,
                                                      n_monte_carlo_runs=n_runs)
    data_dict = {
        "tracker_dict": trackers,
        "errors": err_over_time,
        "runtimes": timings_over_time,
        "settings": settings
    }
    target_file = os.path.join(target_dir, "method_comparison.npy")
    np.save(target_file, data_dict)
    tqdm.write(f"Saved data to {target_file}")


def main():
    """
    Sequentially calls all individual experiment functions that each generate (and save) data to their respective
    output files
    """
    target_dir = get_root() + "output/results/data"
    manager = SettingsManager()

    # Experiment: All Methods over time
    experiment_over_time(manager, target_dir)

    # Experiment: Performance Grid
    experiment_grid(manager, target_dir)

    # Experiment: Number of Particles
    experiment_nparticles(manager, target_dir)

    # Experiment: Internal comparison over time
    experiment_method_comparison(manager, target_dir)

    # Experiment: Runtime analysis
    experiment_runtime(manager, target_dir)


if __name__ == '__main__':
    main()
