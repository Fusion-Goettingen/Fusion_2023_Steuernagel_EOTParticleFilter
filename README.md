# Improved Extended Object Tracking with Efficient Particle-based Orientation Estimation

This repository contains the code for both the proposed algorithm and the conducted experiments for the paper
```
"Improved Extended Object Tracking with Efficient Particle-based Orientation Estimation"
Simon Steuernagel, Kolja Thormann and Marcus Baum
```

The proposed algorithm is a method for elliptical single-extended object tracking developed with orientation uncertainty 
in mind. It requires an additional analytical extended object tracker which is embedded in the method. Implementations
for different methods are included in this repository. In the paper, the 
[IAE](https://ieeexplore.ieee.org/document/8916660) was found to be a highly suitable method that performs well.

Implementations of the reference methods as well as of the conducted experiments can also be found in this repository.
In order to recreate the experiments, the simulation can be run using 
[the file `generate_experiment_data.py`](src/experiments/generate_experiment_data.py). This generates the data
necessary to then create all the presented visualizations, which can be done using
[the file `plot_data.py`](src/experiments/plot_data.py).
Both the hyperparameters of the experiments and the method parameters are extracted from 
[`_settings.py`](src/experiments/_settings.py).

The code for the proposed method itself can be found [in this file](./src/trackers_elliptical/eot_particle_filter.py).
Three different EOT algorithms are supported for the importance sampling:
- [IAE](https://ieeexplore.ieee.org/document/8916660)
- [PAKF](https://ieeexplore.ieee.org/document/9627039)
- [MEM-EKF*](https://ieeexplore.ieee.org/document/8770112)