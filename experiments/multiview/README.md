# Multi-View Causal Disentanglement

This directory contains the implementation of multi-view causal disentanglement as described in the proof document.

## Overview

In the multi-view setting, we have:
- **Shared latent variables** Z = (z₁, z₂, ..., zₚ) with causal structure
- **Multiple views** (e.g., View X and View Y) that observe linear combinations of the same latent variables
- **Different mixing matrices** for each view: Hₓ, Hᵧ
- **Interventional data** from different views

The goal is to recover:
1. The mixing matrices Hₓ and Hᵧ for each view
2. The latent causal structure (B matrices)
3. The intervention targets

## Mathematical Setup

For two views X and Y observing latent variables Z:

```
X = Hₓᵀ Z
Y = Hᵧᵀ Z
```

Where Z follows a latent causal model:
```
Z = B₀⁻¹ ε  (observational)
Z = Bₖ⁻¹ ε  (interventional)
```

The observed precision matrices are:
```
Θₒbs,ₓ = Hₓᵀ B₀ᵀ B₀ Hₓ
Θₖ,ₓ = Hₓᵀ Bₖᵀ Bₖ Hₓ
```

And similarly for View Y.

## Implementation

### Core Modules

1. **`src/dataset_multiview.py`**: Data structures for multi-view datasets
   - `MultiViewDataset`: Stores true parameters (H_views, B matrices, etc.)
   - `ObservedMultiViewDataset`: Stores observed precision matrices

2. **`src/rand_multiview.py`**: Data generation for multi-view settings
   - `rand_multiview_model()`: General multi-view data generator
   - `rand_multiview_model_d2_no_edges()`: d=2 case (no edges)
   - `rand_multiview_model_d3_two_sources()`: d=3 case (two sources)
   - `rand_multiview_model_d3_one_source()`: d=3 case (one source)

3. **`src/solver_multiview.py`**: Multi-view solver implementation
   - `MultiViewSolver`: Main solver class
   - `solve_d2_no_edges()`: Algorithm for d=2 case
   - Uses RQ decomposition and Cholesky decomposition

### Experiments

1. **`experiment_d2.py`**: Experiments for d=2 case
   - Tests recovery with two latent variables (both sources)
   - One intervention per view
   - Validates the RQ decomposition approach

2. **`experiment_d3.py`**: Experiments for d=3 case
   - Tests recovery with three latent variables
   - Different configurations (two sources, one source)
   - More complex scenarios

## Algorithms

### D=2 Algorithm (No Edges)

**Input**:
- Θₒbs,ₓ, Θₒbs,ᵧ (observational precision matrices)
- Θ₁,ₓ (intervention in View X on source z₀)
- Θ₂,ᵧ (intervention in View Y on source z₁)

**Output**: Hₓ, Hᵧ, B₀, B₁, B₂

**Steps**:

1. **Recover one row of Hₓ**:
   - Compute Δₓ = Θ₁,ₓ - Θₒbs,ₓ
   - Since z₀ is a source and intervened: Δₓ = γ hᵀ h (rank-1)
   - Extract h via eigendecomposition (top eigenvector)

2. **Recover full Hₓ via RQ decomposition**:
   - Set q₁ = h / ||h||
   - Find orthogonal q₂
   - Construct Q = [q₁; q₂] (orthonormal rows)
   - Construct R (upper triangular) such that Hₓ = RQ

3. **Recover B matrices from Hₓ**:
   - Compute B₀ᵀB₀ = (Hₓ⁻¹)ᵀ Θₒbs,ₓ Hₓ⁻¹
   - Apply Cholesky decomposition: B₀ = chol(B₀ᵀB₀)ᵀ
   - Similarly for B₁

4. **Recover Hᵧ from View Y**:
   - Use analogous process with Θ₂,ᵧ - Θₒbs,ᵧ
   - Or use B₀ to solve: Hᵧᵀ B₀ᵀ B₀ Hᵧ = Θₒbs,ᵧ

### D=3 Algorithm (Two Sources)

Similar approach but with three latent variables:
- If z₀ and z₁ are both sources: recover two rows directly
- Use orthogonal complement for third row
- More complex permutation matching required

### D=3 Algorithm (One Source)

**Uses whitening approach** (from Section 2 of proof):
- Whiten the observational data: Wᵀ Θₒbs,ₓ W = I
- Analyze whitened differences: Aₖ = Wᵀ Δₖ W
- Decompose as Aₖ = RᵀSₖR where R ∈ O(p)
- Use additional constraints to identify parameters

## Usage

### Basic Example

```python
from src.rand_multiview import rand_multiview_model_d2_no_edges
from src.solver_multiview import MultiViewSolver

# Generate synthetic data
dataset = rand_multiview_model_d2_no_edges(seed=42)

# Sample observations
observed_data = dataset.sample_thetas(nsamples=1000)

# Run solver
solver = MultiViewSolver(observed_data)
result = solver.solve(method="d2_no_edges")

# Access recovered parameters
H_X = result['H_views'][0]
H_Y = result['H_views'][1]
B_obs = result['B_obs']
```

### Running Experiments

```bash
# D=2 experiment (single run)
python -m experiments.multiview.experiment_d2 --seed 42 --nsamples 1000 --verbose

# D=2 experiment (multiple runs)
python -m experiments.multiview.experiment_d2 --multiple --num_experiments 10

# D=3 experiment (two sources)
python -m experiments.multiview.experiment_d3 --config two_sources --seed 42

# D=3 experiment (one source)
python -m experiments.multiview.experiment_d3 --config one_source --seed 42
```

## Key Results

### Theoretical Guarantees

From the proof document:

**Theorem (d=2, no edges)**: With observational data and one intervention per view (one per source), the mixing matrices Hₓ and Hᵧ can be uniquely recovered up to:
- Permutation of latent variables
- Scale/sign of each latent variable

**Theorem (d=3, two sources)**: With two sources and one intervention per source (distributed across views), full recovery is possible.

### Identifiability Conditions

1. **Intervention targets must be sources** (for rank-1 approach)
2. **Interventions must be hard** (perfect interventions)
3. **Sufficient samples** for covariance estimation
4. **Non-degenerate mixing matrices** (Hₓ, Hᵧ must be full rank)

## Comparison with Single-View Case

| Aspect | Single View | Multi-View |
|--------|-------------|------------|
| Data | X = HᵀZ | X = Hₓᵀ Z, Y = Hᵧᵀ Z |
| Parameters | One H matrix | Multiple H matrices (Hₓ, Hᵧ, ...) |
| Interventions | All in same view | Distributed across views |
| Recovery | May need p interventions | Can recover with fewer interventions |
| Advantage | Simpler | More identifiable with fewer interventions |

## Limitations and Future Work

### Current Limitations

1. **Source node requirement**: Current d=2 implementation requires intervention targets to be source nodes
2. **Hard interventions only**: Does not yet support soft interventions
3. **D=3 one-source case**: Whitening approach not fully implemented
4. **Permutation matching**: Simple approach for d=2, needs improvement for higher dimensions

### Future Extensions

1. Implement whitening approach for d=3 one-source case
2. Support for soft interventions
3. More sophisticated permutation matching
4. Extension to more than 2 views
5. Handling of missing or incomplete interventions
6. Robustness to model misspecification

## References

Based on the proof document provided, which extends the LCD via Interventions paper to the multi-view setting.

## Testing

Run the simple test to verify installation:

```bash
python test_multiview_simple.py
```

This will test:
1. Module imports
2. Data generation
3. Observation sampling
4. Solver execution
5. Basic accuracy checks
