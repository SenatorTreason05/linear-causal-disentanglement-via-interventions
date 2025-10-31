# Linear Causal Disentanglement via Interventions

This repository implements algorithms for recovering latent causal structures from interventional data. It includes both the original single-view algorithm and a new **multi-view extension** where different views observe the same latent variables through different mixing matrices.

## Setup 

To set up a virtual environment with the required packages, run:
```
bash setup.sh
```
Note that this project depends on two packages which can be difficult to install,
`pygraphviz` and `gurobipy`.
The dependence on `pygraphviz` is only necessary for plotting the learned latent DAG over the real data, so almost everything can be run without this dependency.
The dependence on `gurobipy` is required for efficient computation to see how well two permutations match, up to the partial order defined by a DAG.
This dependence can be removed by changing `DEFAULT_PERMUTATION_METHOD` in `src/run_experiment.py` from `"ilp"` to `"naive"`.

## Synthetic Data Result Reproduction
To reproduce the synthetic data results, run:
```
python3 -m experiments.experiment1.noisy_recovery --seed 8164 --nnodes 5 --nnodes_obs 10
```

## Real Data Result Reproduction
To reproduce the real data results, first download the data:
```
python3 -m experiments.real_data.step1_download_and_pickle
```
For the semi-synthetic results, run
```
bash experiments/real_data/semisynthetic.sh
```
For real-data results, run
```
bash experiments/real_data/real.sh
```

## Multi-View Extension

This repository now includes a **multi-view extension** that allows recovery of latent causal structures when multiple views observe the same latent variables through different mixing matrices.

### Mathematical Setup

In the multi-view setting:
- **Shared latent variables** Z with causal structure
- **Multiple views** (e.g., View X and View Y) observe: X = H_X^T Z, Y = H_Y^T Z
- **Different mixing matrices** H_X and H_Y for each view
- **Interventions distributed across views**

The algorithm recovers all mixing matrices and the latent causal structure using observational data plus one intervention per view.

### Running Multi-View Experiments

**D=2 case (two latent variables, no edges):**
```bash
# Single experiment
python -m experiments.multiview.experiment_d2 --seed 42 --nsamples 1000 --verbose

# Multiple experiments with varying sample sizes
python -m experiments.multiview.experiment_d2 --multiple --num_experiments 10
```

**D=3 case (three latent variables):**
```bash
# Two sources configuration
python -m experiments.multiview.experiment_d3 --config two_sources --seed 42

# One source configuration
python -m experiments.multiview.experiment_d3 --config one_source --seed 42
```

**Quick test:**
```bash
python test_multiview_simple.py
```

### Key Features

- **Fewer interventions needed**: Multi-view setting can recover full structure with p interventions (one per latent variable) distributed across views
- **More identifiable**: Additional constraints from multiple views improve identifiability
- **Flexible intervention distribution**: Interventions can be performed in any view

### Implementation Files

- `src/dataset_multiview.py`: Multi-view data structures
- `src/rand_multiview.py`: Multi-view data generation
- `src/solver_multiview.py`: Multi-view solver with RQ decomposition
- `experiments/multiview/`: Multi-view experiments

See `experiments/multiview/README.md` for detailed documentation.