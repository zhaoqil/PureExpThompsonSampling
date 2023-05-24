# Optimal Exploration is No Harder Than Thompson Sampling

Code for the paper "Optimal Exploration is No Harder Than Thompson Sampling". 

## Setting up

The code requires the following packages in Python: numpy, scipy, matplotlib, pandas. For running benchmark algorithms, it needs a version of Julia later or equal to 1.6.4, with the following packages: JLD2, StatsPlots, LaTeXStrings, IterTools, Distributions, JuMP, Tulip. 

## Code structure

The code is organized in the following files:

- library_linear.py implements the Thompson sampling and our PEPS algorithms. 
- instance.py includes the three problem instances we consider. 
- run.py includes the functions to run experiments and store outputs. 
- utils.py includes some utility functions. 
- bandit-elimination-main is a directory containing files we modify from the repo [Elimination Strategies for Bandit Identification
](https://github.com/AndreaTirinzoni/bandit-elimination) for running benchmark algorithms such as LinGame, LinGapE, and oracle strategy in our instances. 

## Reproducing our experiments

To reproduce our experiments, one should first run benchmark algorithms in Julia, then run our algorithms in Python, and run the Jupyter notebook to produce the figures. To run the benchmark algorithms, please add other files to the "bandit-elimination-main" folder from the repo [Elimination Strategies for Bandit Identification
](https://github.com/AndreaTirinzoni/bandit-elimination), change to the "experiments" folder, and enter the command in terminal for each instance: 
```
julia hard_linear.jl
julia linear_bai.jl
julia linear_topm.jl
```

To run TS and PEPS algorithm, please change to the PureExpThompsonSampling directory and enter the command: 
```
python run.py --path PATH --config configs/config_soare.json
python run.py --path PATH --config configs/config_sphere.json
python run.py --path PATH --config configs/config_topm.json
```

In the above command, please change path to the corresponding path. Then, running the scripts in plot.ipynb should reproduce the figures. 