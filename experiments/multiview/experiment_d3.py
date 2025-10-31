"""
Experiment for d=3 multi-view case (three latent variables).

This experiment validates the algorithm from Section 2 of the proof document:
- Two views (X and Y) observing three latent variables
- Different configurations:
  1. Two sources (z_0 and z_1 are sources, both point to z_2)
  2. One source (only one node is a source)
- One intervention per view
- Goal: recover H_X, H_Y, B_obs, and intervention targets
"""

# === IMPORTS: BUILT-IN ===
import argparse
from pathlib import Path

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import matplotlib.pyplot as plt

# === IMPORTS: LOCAL ===
from src.rand_multiview import (
    rand_multiview_model_d3_two_sources,
    rand_multiview_model_d3_one_source
)
from src.solver_multiview import MultiViewSolver


def compute_errors(true_params: dict, estimated_params: dict, p: int = 3) -> dict:
    """
    Compute reconstruction errors for d=3 multi-view setting.

    Args:
        true_params: Dictionary with true H_views, B_obs, Bs
        estimated_params: Dictionary with estimated H_views, B_obs, Bs
        p: Number of latent variables

    Returns:
        Dictionary with error metrics
    """
    errors = {}

    # H matrices error (need to try all permutations for d=3)
    from itertools import permutations

    for view_idx, (H_true, H_est) in enumerate(zip(true_params['H_views'], estimated_params['H_views'])):
        min_error = float('inf')

        # Try all permutations and sign flips
        for perm in permutations(range(p)):
            H_true_perm = H_true[list(perm), :]

            for sign_flip in range(2**p):
                signs = np.array([1 if (sign_flip >> i) & 1 else -1 for i in range(p)])
                H_true_signed = H_true_perm * signs[:, np.newaxis]

                error = np.linalg.norm(H_true_signed - H_est, 'fro')
                min_error = min(min_error, error)

        errors[f'H_view_{view_idx}_error'] = min_error

    # B_obs error
    B_obs_true = true_params['B_obs']
    B_obs_est = estimated_params['B_obs']

    min_error = float('inf')
    for perm in permutations(range(p)):
        perm_list = list(perm)
        B_obs_perm = B_obs_true[perm_list, :][:, perm_list]
        error = np.linalg.norm(B_obs_perm - B_obs_est, 'fro')
        min_error = min(min_error, error)

    errors['B_obs_error'] = min_error

    # Overall error
    errors['total_error'] = sum(errors.values())

    return errors


def run_d3_experiment(
    config: str = "two_sources",
    seed: int = 42,
    nnodes_obs_X: int = 4,
    nnodes_obs_Y: int = 3,
    nsamples: int = 1000,
    verbose: bool = True
):
    """
    Run a single d=3 multi-view experiment.

    Args:
        config: Configuration type ("two_sources" or "one_source")
        seed: Random seed
        nnodes_obs_X: Number of observed variables in View X
        nnodes_obs_Y: Number of observed variables in View Y
        nsamples: Number of samples for generating covariance estimates
        verbose: Whether to print detailed output

    Returns:
        Dictionary with results
    """
    if verbose:
        print("=" * 70)
        print(f"D=3 MULTI-VIEW EXPERIMENT ({config.upper()})")
        print("=" * 70)
        print(f"Seed: {seed}")
        print(f"Observed dimensions: X={nnodes_obs_X}, Y={nnodes_obs_Y}")
        print(f"Samples: {nsamples}")
        print()

    # Generate random multi-view dataset
    if config == "two_sources":
        true_dataset = rand_multiview_model_d3_two_sources(
            seed=seed,
            nnodes_obs_X=nnodes_obs_X,
            nnodes_obs_Y=nnodes_obs_Y
        )
    elif config == "one_source":
        true_dataset = rand_multiview_model_d3_one_source(
            seed=seed,
            nnodes_obs_X=nnodes_obs_X,
            nnodes_obs_Y=nnodes_obs_Y,
            source_node=0
        )
    else:
        raise ValueError(f"Unknown config: {config}")

    if verbose:
        print("TRUE PARAMETERS:")
        print("-" * 70)
        print(f"H_X (View X mixing matrix):\n{true_dataset.H_views[0]}")
        print()
        print(f"H_Y (View Y mixing matrix):\n{true_dataset.H_views[1]}")
        print()
        print(f"B_obs (Observational B matrix):\n{true_dataset.B_obs}")
        print()
        for i, B in enumerate(true_dataset.Bs):
            print(f"B_{i} (Intervention {i}):\n{B}")
            print()

    # Sample observed precision matrices
    observed_data = true_dataset.sample_thetas(nsamples)

    if verbose:
        print("OBSERVED DATA:")
        print("-" * 70)
        print(f"Theta_obs_X shape: {observed_data.Theta_obs_views[0].shape}")
        print(f"Theta_obs_Y shape: {observed_data.Theta_obs_views[1].shape}")
        print(f"Number of interventions in X: {len(observed_data.Thetas_views[0])}")
        print(f"Number of interventions in Y: {len(observed_data.Thetas_views[1])}")
        print()

    # Run solver
    solver = MultiViewSolver(observed_data, verbose=verbose)

    # For d=3, we need different methods based on configuration
    if config == "two_sources":
        # Can use similar approach as d=2
        method = "d2_no_edges"  # Will need to extend for d=3
    else:
        # Need whitening approach (not yet implemented)
        method = "d3_one_source"

    try:
        estimated_params = solver.solve(method=method)

        if verbose:
            print()
            print("ESTIMATED PARAMETERS:")
            print("-" * 70)
            print(f"H_X (estimated):\n{estimated_params['H_views'][0]}")
            print()
            print(f"H_Y (estimated):\n{estimated_params['H_views'][1]}")
            print()
            print(f"B_obs (estimated):\n{estimated_params['B_obs']}")
            print()

        # Compute errors
        true_params = {
            'H_views': true_dataset.H_views,
            'B_obs': true_dataset.B_obs,
            'Bs': true_dataset.Bs
        }

        errors = compute_errors(true_params, estimated_params, p=3)

        if verbose:
            print()
            print("ERRORS:")
            print("-" * 70)
            for key, value in errors.items():
                print(f"{key}: {value:.6f}")
            print()

        return {
            'true_params': true_params,
            'estimated_params': estimated_params,
            'errors': errors,
            'success': errors['total_error'] < 1.0
        }

    except NotImplementedError as e:
        if verbose:
            print(f"Method {method} not yet fully implemented: {e}")
        return {
            'true_params': None,
            'estimated_params': None,
            'errors': {'total_error': float('inf')},
            'success': False
        }


def run_multiple_experiments(
    config: str = "two_sources",
    num_experiments: int = 10,
    nsamples_list: list = None,
    output_dir: str = None
):
    """
    Run multiple d=3 experiments with varying parameters.

    Args:
        config: Configuration type ("two_sources" or "one_source")
        num_experiments: Number of experiments per configuration
        nsamples_list: List of sample sizes to try
        output_dir: Directory to save results
    """
    if nsamples_list is None:
        nsamples_list = [100, 500, 1000, 5000, 10000]

    results = {nsamples: [] for nsamples in nsamples_list}

    print(f"Running multiple d=3 multi-view experiments ({config})...")
    print()

    for nsamples in nsamples_list:
        print(f"Sample size: {nsamples}")
        for exp_idx in range(num_experiments):
            seed = 2000 + exp_idx
            result = run_d3_experiment(
                config=config,
                seed=seed,
                nsamples=nsamples,
                verbose=False
            )
            results[nsamples].append(result)
            if result['success'] or result['errors']['total_error'] < float('inf'):
                print(f"  Experiment {exp_idx + 1}/{num_experiments}: "
                      f"Total error = {result['errors']['total_error']:.6f}")
            else:
                print(f"  Experiment {exp_idx + 1}/{num_experiments}: "
                      f"Failed (method not implemented)")

    # Aggregate results
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for nsamples in nsamples_list:
        valid_results = [r for r in results[nsamples]
                         if r['errors']['total_error'] < float('inf')]

        if len(valid_results) == 0:
            print(f"nsamples={nsamples}: No valid results")
            continue

        errors = [r['errors']['total_error'] for r in valid_results]
        success_rate = sum(r['success'] for r in valid_results) / len(valid_results)

        print(f"nsamples={nsamples}:")
        print(f"  Mean error: {np.mean(errors):.6f}")
        print(f"  Std error: {np.std(errors):.6f}")
        print(f"  Min error: {np.min(errors):.6f}")
        print(f"  Max error: {np.max(errors):.6f}")
        print(f"  Success rate: {success_rate:.2%}")
        print()

    # Plot results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))

        valid_nsamples = []
        mean_errors = []
        std_errors = []

        for ns in nsamples_list:
            valid_results = [r for r in results[ns]
                             if r['errors']['total_error'] < float('inf')]
            if len(valid_results) > 0:
                valid_nsamples.append(ns)
                mean_errors.append(np.mean([r['errors']['total_error'] for r in valid_results]))
                std_errors.append(np.std([r['errors']['total_error'] for r in valid_results]))

        if len(valid_nsamples) > 0:
            plt.errorbar(valid_nsamples, mean_errors, yerr=std_errors,
                         marker='o', capsize=5)
            plt.xlabel('Number of samples')
            plt.ylabel('Total reconstruction error')
            plt.title(f'D=3 Multi-View Recovery Error vs. Sample Size ({config})')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'd3_{config}_error_vs_samples.png', dpi=150)
            print(f"Saved plot to {output_dir / f'd3_{config}_error_vs_samples.png'}")

    return results


def main():
    parser = argparse.ArgumentParser(description='D=3 multi-view experiment')
    parser.add_argument('--config', type=str, default='two_sources',
                        choices=['two_sources', 'one_source'],
                        help='Configuration type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--nnodes_obs_X', type=int, default=4,
                        help='Number of observed variables in View X')
    parser.add_argument('--nnodes_obs_Y', type=int, default=3,
                        help='Number of observed variables in View Y')
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='Number of samples')
    parser.add_argument('--multiple', action='store_true',
                        help='Run multiple experiments')
    parser.add_argument('--num_experiments', type=int, default=10,
                        help='Number of experiments (if --multiple)')
    parser.add_argument('--output_dir', type=str, default='results/multiview',
                        help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.multiple:
        run_multiple_experiments(
            config=args.config,
            num_experiments=args.num_experiments,
            output_dir=args.output_dir
        )
    else:
        run_d3_experiment(
            config=args.config,
            seed=args.seed,
            nnodes_obs_X=args.nnodes_obs_X,
            nnodes_obs_Y=args.nnodes_obs_Y,
            nsamples=args.nsamples,
            verbose=args.verbose or True
        )


if __name__ == '__main__':
    main()
