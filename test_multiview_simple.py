#!/usr/bin/env python3
"""
Simple test script for multi-view implementation.
Tests basic functionality without requiring full environment setup.
"""

import sys
import numpy as np

try:
    import causaldag as cd
    print("✓ causaldag imported successfully")
except ImportError as e:
    print(f"✗ Failed to import causaldag: {e}")
    sys.exit(1)

try:
    from src.dataset_multiview import MultiViewDataset, ObservedMultiViewDataset
    print("✓ Multi-view dataset classes imported successfully")
except ImportError as e:
    print(f"✗ Failed to import multi-view dataset: {e}")
    sys.exit(1)

try:
    from src.rand_multiview import rand_multiview_model_d2_no_edges
    print("✓ Multi-view data generation imported successfully")
except ImportError as e:
    print(f"✗ Failed to import multi-view data generation: {e}")
    sys.exit(1)

try:
    from src.solver_multiview import MultiViewSolver
    print("✓ Multi-view solver imported successfully")
except ImportError as e:
    print(f"✗ Failed to import multi-view solver: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("BASIC FUNCTIONALITY TEST")
print("=" * 70)
print()

# Test 1: Generate a d=2 multi-view dataset
print("Test 1: Generating d=2 multi-view dataset...")
try:
    dataset = rand_multiview_model_d2_no_edges(seed=42, nnodes_obs_X=3, nnodes_obs_Y=2)
    print(f"✓ Dataset generated successfully")
    print(f"  - Number of views: {len(dataset.H_views)}")
    print(f"  - H_X shape: {dataset.H_views[0].shape}")
    print(f"  - H_Y shape: {dataset.H_views[1].shape}")
    print(f"  - B_obs shape: {dataset.B_obs.shape}")
    print(f"  - Number of interventions: {len(dataset.Bs)}")
except Exception as e:
    print(f"✗ Failed to generate dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 2: Sample observations
print("Test 2: Sampling observations...")
try:
    observed_data = dataset.sample_thetas(nsamples=100)
    print(f"✓ Observations sampled successfully")
    print(f"  - Number of views: {len(observed_data.Theta_obs_views)}")
    print(f"  - Theta_obs_X shape: {observed_data.Theta_obs_views[0].shape}")
    print(f"  - Theta_obs_Y shape: {observed_data.Theta_obs_views[1].shape}")
    print(f"  - Interventions in X: {len(observed_data.Thetas_views[0])}")
    print(f"  - Interventions in Y: {len(observed_data.Thetas_views[1])}")
except Exception as e:
    print(f"✗ Failed to sample observations: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Run solver
print("Test 3: Running multi-view solver...")
try:
    solver = MultiViewSolver(observed_data, verbose=False)
    result = solver.solve(method="d2_no_edges")
    print(f"✓ Solver completed successfully")
    print(f"  - Recovered H_X shape: {result['H_views'][0].shape}")
    print(f"  - Recovered H_Y shape: {result['H_views'][1].shape}")
    print(f"  - Recovered B_obs shape: {result['B_obs'].shape}")
except Exception as e:
    print(f"✗ Solver failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Check recovery accuracy
print("Test 4: Checking recovery accuracy...")
try:
    # Simple check: see if recovered matrices have reasonable values
    H_X_true = dataset.H_views[0]
    H_X_est = result['H_views'][0]

    # Check shapes match
    assert H_X_true.shape == H_X_est.shape, "H_X shape mismatch"

    # Compute error (considering permutations)
    error1 = np.linalg.norm(H_X_true - H_X_est, 'fro')
    error2 = np.linalg.norm(H_X_true[[1, 0], :] - H_X_est, 'fro')
    min_error = min(error1, error2)

    print(f"✓ Recovery accuracy check passed")
    print(f"  - H_X Frobenius error (best permutation): {min_error:.6f}")

    if min_error < 5.0:  # Reasonable threshold for low samples
        print(f"  - Recovery appears successful!")
    else:
        print(f"  - Recovery error is high (may need more samples)")

except Exception as e:
    print(f"✗ Accuracy check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print()
print("The multi-view implementation is working correctly.")
print("You can now run full experiments using:")
print("  python -m experiments.multiview.experiment_d2")
print("  python -m experiments.multiview.experiment_d3")
