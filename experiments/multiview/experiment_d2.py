"""
Experiment for d=2 multi-view case (two latent variables, no edges).

This experiment validates the algorithm from Section 1 of the proof document:
- Two views (X and Y) observing the same two latent variables
- No edges in the latent DAG (both z_0 and z_1 are sources)
- One intervention per view (one on each source)
- Goal: recover H_X, H_Y, B_obs, and intervention targets
"""

# === IMPORTS: BUILT-IN ===
import argparse
from pathlib import Path

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import matplotlib.pyplot as plt

# === IMPORTS: LOCAL ===
from src.rand_multiview import rand_multiview_model_d2_no_edges
from src.solver_multiview import MultiViewSolver


def compute_errors(true_params: dict, estimated_params: dict) -> dict:
    """
    Compute reconstruction errors for multi-view setting.

    Args:
        true_params: Dictionary with true H_views, B_obs, Bs
        estimated_params: Dictionary with estimated H_views, B_obs, Bs

    Returns:
        Dictionary with error metrics
    """
    errors = {}

    # H matrices error (Frobenius norm, accounting for scale/sign ambiguity)
    for view_idx, (H_true, H_est) in enumerate(zip(true_params['H_views'], estimated_params['H_views'])):
        # Need to account for permutation of rows (latent variables)
        # For d=2, try both permutations
        error_perm1 = np.linalg.norm(H_true - H_est, 'fro')
        error_perm2 = np.linalg.norm(H_true[[1, 0], :] - H_est, 'fro')
        error_perm3 = np.linalg.norm(-H_true - H_est, 'fro')
        error_perm4 = np.linalg.norm(-H_true[[1, 0], :] - H_est, 'fro')

        min_error = min(error_perm1, error_perm2, error_perm3, error_perm4)
        errors[f'H_view_{view_idx}_error'] = min_error

    # B_obs error
    B_obs_true = true_params['B_obs']
    B_obs_est = estimated_params['B_obs']

    # Try different permutations and signs
    error_perm1 = np.linalg.norm(B_obs_true - B_obs_est, 'fro')
    error_perm2 = np.linalg.norm(B_obs_true[[1, 0], :][:, [1, 0]] - B_obs_est, 'fro')

    errors['B_obs_error'] = min(error_perm1, error_perm2)

    # Overall error
    errors['total_error'] = sum(errors.values())

    return errors


def run_d2_experiment(
    seed: int = 42,
    nnodes_obs_X: int = 3,
    nnodes_obs_Y: int = 2,
    nsamples: int = 1000,
    verbose: bool = True
):
    """
    Run a single d=2 multi-view experiment.

    Args:
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
        print("D=2 MULTI-VIEW EXPERIMENT (NO EDGES)")
        print("=" * 70)
        print(f"Seed: {seed}")
        print(f"Observed dimensions: X={nnodes_obs_X}, Y={nnodes_obs_Y}")
        print(f"Samples: {nsamples}")
        print()

    # Generate random multi-view dataset
    true_dataset = rand_multiview_model_d2_no_edges(
        seed=seed,
        nnodes_obs_X=nnodes_obs_X,
        nnodes_obs_Y=nnodes_obs_Y
    )

    if verbose:
        print("TRUE PARAMETERS:")
        print("-" * 70)
        print(f"H_X (View X mixing matrix):\n{true_dataset.H_views[0]}")
        print()
        print(f"H_Y (View Y mixing matrix):\n{true_dataset.H_views[1]}")
        print()
        print(f"B_obs (Observational B matrix):\n{true_dataset.B_obs}")
        print()
        print(f"B_0 (Intervention in View X):\n{true_dataset.Bs[0]}")
        print()
        print(f"B_1 (Intervention in View Y):\n{true_dataset.Bs[1]}")
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
    estimated_params = solver.solve(method="d2_no_edges")

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

    errors = compute_errors(true_params, estimated_params)

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
        'success': errors['total_error'] < 1.0  # Threshold for success
    }


def run_multiple_experiments(
    num_experiments: int = 10,
    nsamples_list: list = None,
    output_dir: str = None
):
    """
    Run multiple d=2 experiments with varying parameters.

    Args:
        num_experiments: Number of experiments per configuration
        nsamples_list: List of sample sizes to try
        output_dir: Directory to save results
    """
    if nsamples_list is None:
        nsamples_list = [100, 500, 1000, 5000, 10000]

    results = {nsamples: [] for nsamples in nsamples_list}

    print("Running multiple d=2 multi-view experiments...")
    print()

    for nsamples in nsamples_list:
        print(f"Sample size: {nsamples}")
        for exp_idx in range(num_experiments):
            seed = 1000 + exp_idx
            result = run_d2_experiment(
                seed=seed,
                nsamples=nsamples,
                verbose=False
            )
            results[nsamples].append(result)
            print(f"  Experiment {exp_idx + 1}/{num_experiments}: "
                  f"Total error = {result['errors']['total_error']:.6f}")

    # Aggregate results
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for nsamples in nsamples_list:
        errors = [r['errors']['total_error'] for r in results[nsamples]]
        success_rate = sum(r['success'] for r in results[nsamples]) / num_experiments

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
        mean_errors = [np.mean([r['errors']['total_error'] for r in results[ns]])
                       for ns in nsamples_list]
        std_errors = [np.std([r['errors']['total_error'] for r in results[ns]])
                      for ns in nsamples_list]

        plt.errorbar(nsamples_list, mean_errors, yerr=std_errors,
                     marker='o', capsize=5)
        plt.xlabel('Number of samples')
        plt.ylabel('Total reconstruction error')
        plt.title('D=2 Multi-View Recovery Error vs. Sample Size')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'd2_error_vs_samples.png', dpi=150)
        print(f"Saved plot to {output_dir / 'd2_error_vs_samples.png'}")

    return results


def main():
    parser = argparse.ArgumentParser(description='D=2 multi-view experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--nnodes_obs_X', type=int, default=3,
                        help='Number of observed variables in View X')
    parser.add_argument('--nnodes_obs_Y', type=int, default=2,
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
            num_experiments=args.num_experiments,
            output_dir=args.output_dir
        )
    else:
        run_d2_experiment(
            seed=args.seed,
            nnodes_obs_X=args.nnodes_obs_X,
            nnodes_obs_Y=args.nnodes_obs_Y,
            nsamples=args.nsamples,
            verbose=args.verbose or True
        )


if __name__ == '__main__':
    main()
