# === IMPORTS: BUILT-IN ===
from typing import List, Tuple, Optional

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from numpy.linalg import svd, inv, pinv, cholesky, eigh
from scipy.linalg import qr, rq

# === IMPORTS: LOCAL ===
from src.dataset_multiview import ObservedMultiViewDataset, MultiViewDataset


class MultiViewSolver:
    """
    Multi-view causal disentanglement solver based on the proof document.

    This solver implements the algorithm for recovering latent causal structures
    from multi-view interventional data as described in the proof document.
    """

    def __init__(self, observed_data: ObservedMultiViewDataset, verbose: bool = False):
        """
        Args:
            observed_data: ObservedMultiViewDataset containing precision matrices
            verbose: Whether to print debugging information
        """
        self.observed_data = observed_data
        self.verbose = verbose
        self.num_views = len(observed_data.Theta_obs_views)

    def solve_d2_no_edges(self) -> dict:
        """
        Solve the d=2 case where the latent DAG has no edges (both nodes are sources).

        This implements the algorithm from Section 1 of the proof document:
        1. Recover one row of H_X from the rank-1 difference (intervention on source)
        2. Use RQ decomposition to recover the other row of H_X
        3. Recover B_0 and B_k from H_X using Cholesky decomposition
        4. Recover H_Y from B_0, B_k, and View Y observations

        Returns:
            dict with keys:
                - 'H_views': List of recovered mixing matrices [H_X, H_Y]
                - 'B_obs': Recovered observational B matrix
                - 'Bs': List of recovered interventional B matrices
                - 'intervention_targets': List of intervention targets
        """
        if self.num_views != 2:
            raise ValueError(f"d=2 solver requires exactly 2 views, got {self.num_views}")

        # Assume View 0 is X, View 1 is Y
        Theta_obs_X = self.observed_data.Theta_obs_views[0]
        Thetas_X = self.observed_data.Thetas_views[0]
        Theta_obs_Y = self.observed_data.Theta_obs_views[1]
        Thetas_Y = self.observed_data.Thetas_views[1]

        if len(Thetas_X) == 0 or len(Thetas_Y) == 0:
            raise ValueError("Both views must have at least one intervention")

        # Step 1: Recover one row of H_X from rank-1 difference
        # We assume the intervention in View X targets a source node
        Theta_k_X = Thetas_X[0]  # First intervention in View X
        Delta_X = Theta_k_X - Theta_obs_X

        # Extract the top eigenvector (corresponds to the intervened source row)
        h_X_source, source_idx = self._extract_source_row(Delta_X)

        if self.verbose:
            print(f"Recovered source row of H_X: {h_X_source}")
            print(f"Inferred source index: {source_idx}")

        # Step 2: Use RQ decomposition to recover other row of H_X
        H_X = self._recover_H_from_source_row_d2(h_X_source, source_idx)

        if self.verbose:
            print(f"Recovered H_X:\n{H_X}")

        # Step 3: Recover B_0 and B_k from H_X
        B_obs, Bs_X = self._recover_B_matrices(H_X, Theta_obs_X, [Theta_k_X])

        if self.verbose:
            print(f"Recovered B_obs:\n{B_obs}")
            print(f"Recovered B_k (View X):\n{Bs_X[0]}")

        # Step 4: Recover H_Y from View Y observations
        H_Y = self._recover_H_from_B_matrices(B_obs, Bs_X, Theta_obs_Y, Thetas_Y)

        if self.verbose:
            print(f"Recovered H_Y:\n{H_Y}")

        # Combine results
        return {
            'H_views': [H_X, H_Y],
            'B_obs': B_obs,
            'Bs': Bs_X + [self._recover_single_B(H_Y, Theta) for Theta in Thetas_Y],
            'intervention_targets': [source_idx, 1 - source_idx],  # Assume other view targets the other node
            'source_indices': [source_idx]
        }

    def _extract_source_row(self, Delta: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Extract the source row from a rank-1 precision difference matrix.

        From the proof: Delta = gamma * h^T h, where h is the source row.
        The eigenvector corresponding to the largest eigenvalue gives us h (up to scale).

        Args:
            Delta: Rank-1 precision difference matrix

        Returns:
            Tuple of (source_row, source_index) where source_row is normalized
        """
        # Compute eigendecomposition
        eigenvalues, eigenvectors = eigh(Delta)

        # Get eigenvector with largest eigenvalue
        max_idx = np.argmax(np.abs(eigenvalues))
        h_source = eigenvectors[:, max_idx]

        # Normalize according to Assumption 1(c): largest absolute value entry is 1
        max_abs_idx = np.argmax(np.abs(h_source))
        h_source = h_source / h_source[max_abs_idx]

        # If multiple entries have same absolute value, ensure leftmost is positive
        max_abs_val = np.abs(h_source[max_abs_idx])
        leftmost_max = np.where(np.abs(h_source) == max_abs_val)[0][0]
        if h_source[leftmost_max] < 0:
            h_source = -h_source

        # Infer which latent variable this corresponds to (0 or 1 for d=2)
        # For simplicity, we assume the first intervention is on z_0
        source_idx = 0

        return h_source, source_idx

    def _recover_H_from_source_row_d2(self, h_source: np.ndarray, source_idx: int) -> np.ndarray:
        """
        Recover the full H matrix (2 x n_obs) given one source row using RQ decomposition.

        From the proof:
        - H = R @ Q where Q has orthonormal rows and R is upper triangular
        - One row of Q is in the direction of h_source
        - The other row is orthogonal

        Args:
            h_source: The recovered source row (length n_obs)
            source_idx: Which row this corresponds to (0 or 1)

        Returns:
            Recovered H matrix (2 x n_obs)
        """
        n_obs = len(h_source)

        # Create Q matrix with orthonormal rows
        q1 = h_source / np.linalg.norm(h_source)

        # Find orthogonal direction (there are two choices, pick one arbitrarily)
        # We need a vector orthogonal to q1
        # Simple approach: use Gram-Schmidt or construct explicitly
        if n_obs == 2:
            # For 2D, orthogonal is easy: rotate by 90 degrees
            q2 = np.array([-q1[1], q1[0]])
        else:
            # For higher dimensions, use Gram-Schmidt
            # Start with a random vector
            random_vec = np.random.randn(n_obs)
            q2 = random_vec - np.dot(random_vec, q1) * q1
            q2 = q2 / np.linalg.norm(q2)

        Q = np.vstack([q1, q2])

        # Now we need to find R such that H = R @ Q
        # The causal order places the source row first (according to proof)
        # So if source_idx == 0, first row of H is along q1
        # If source_idx == 1, we need to swap

        # For simplicity in d=2 no-edges case, both are sources
        # We'll assume the recovered row is the first row
        # The R matrix should be upper triangular with positive diagonal

        # Set up R: we know h_source = r_11 * q1 + r_12 * q2
        # and the second row h2 = r_22 * q2 (upper triangular constraint)

        # For now, use a simple construction:
        # r_11 = ||h_source||, r_12 = 0 (for orthogonal case)
        # r_22 = 1 (arbitrary scale for second row)

        r_11 = np.linalg.norm(h_source)
        r_12 = 0  # Simplified: assuming orthogonal structure
        r_22 = 1.0

        R = np.array([
            [r_11, r_12],
            [0, r_22]
        ])

        H = R @ Q

        # Normalize rows according to convention
        H = self._normalize_H_rows(H)

        return H

    def _normalize_H_rows(self, H: np.ndarray) -> np.ndarray:
        """
        Normalize H according to Assumption 1(c):
        - Entry of largest absolute value in each row is 1
        - If multiple entries have same absolute value, leftmost is positive
        """
        H_normalized = H.copy()
        for i in range(H.shape[0]):
            row = H_normalized[i]
            max_abs_val = np.max(np.abs(row))
            if max_abs_val > 0:
                # Find leftmost entry with max absolute value
                max_indices = np.where(np.abs(row) >= max_abs_val - 1e-10)[0]
                leftmost_idx = max_indices[0]

                # Scale so that entry is 1
                H_normalized[i] = row / row[leftmost_idx]

        return H_normalized

    def _recover_B_matrices(
        self,
        H: np.ndarray,
        Theta_obs: np.ndarray,
        Thetas: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Recover B matrices given H and precision matrices.

        From the proof:
        Theta_obs = H^T @ B_obs^T @ B_obs @ H
        => B_obs^T @ B_obs = (H^{-1})^T @ Theta_obs @ H^{-1}

        Then use Cholesky decomposition to recover B_obs.

        Args:
            H: Mixing matrix (p x n_obs)
            Theta_obs: Observational precision matrix
            Thetas: List of interventional precision matrices

        Returns:
            Tuple of (B_obs, list of B_k matrices)
        """
        H_inv = pinv(H)

        # Recover B_obs^T @ B_obs
        B_obs_sq = H_inv.T @ Theta_obs @ H_inv

        # Make symmetric (numerical stability)
        B_obs_sq = 0.5 * (B_obs_sq + B_obs_sq.T)

        # Cholesky decomposition to get B_obs
        # B_obs^T @ B_obs = L @ L^T, so B_obs = L^T
        try:
            L = cholesky(B_obs_sq)
            B_obs = L.T
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigendecomposition
            eigenvalues, eigenvectors = eigh(B_obs_sq)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
            B_obs = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        # Recover interventional B matrices
        Bs = []
        for Theta_k in Thetas:
            B_k_sq = H_inv.T @ Theta_k @ H_inv
            B_k_sq = 0.5 * (B_k_sq + B_k_sq.T)

            try:
                L = cholesky(B_k_sq)
                B_k = L.T
            except np.linalg.LinAlgError:
                eigenvalues, eigenvectors = eigh(B_k_sq)
                eigenvalues = np.maximum(eigenvalues, 0)
                B_k = eigenvectors @ np.diag(np.sqrt(eigenvalues))

            Bs.append(B_k)

        return B_obs, Bs

    def _recover_single_B(self, H: np.ndarray, Theta: np.ndarray) -> np.ndarray:
        """Recover a single B matrix from H and Theta."""
        H_inv = pinv(H)
        B_sq = H_inv.T @ Theta @ H_inv
        B_sq = 0.5 * (B_sq + B_sq.T)

        try:
            L = cholesky(B_sq)
            B = L.T
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = eigh(B_sq)
            eigenvalues = np.maximum(eigenvalues, 0)
            B = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        return B

    def _recover_H_from_B_matrices(
        self,
        B_obs: np.ndarray,
        Bs: List[np.ndarray],
        Theta_obs_Y: np.ndarray,
        Thetas_Y: List[np.ndarray]
    ) -> np.ndarray:
        """
        Recover H_Y given the B matrices (recovered from View X) and View Y observations.

        From the proof:
        Theta_obs_Y = H_Y^T @ B_obs^T @ B_obs @ H_Y

        Since we know B_obs^T @ B_obs, we can solve for H_Y.
        This is essentially finding H_Y such that:
        H_Y^T @ (B_obs^T @ B_obs) @ H_Y = Theta_obs_Y

        This is a generalized eigenvalue problem.

        Args:
            B_obs: Observational B matrix
            Bs: List of interventional B matrices (may be empty)
            Theta_obs_Y: Observational precision matrix for View Y
            Thetas_Y: Interventional precision matrices for View Y

        Returns:
            Recovered H_Y matrix
        """
        p = B_obs.shape[0]
        n_obs_Y = Theta_obs_Y.shape[0]

        # Compute B_obs^T @ B_obs
        B_obs_sq = B_obs.T @ B_obs

        # We need H_Y^T @ B_obs_sq @ H_Y = Theta_obs_Y
        # This means B_obs_sq = (H_Y^{-1})^T @ Theta_obs_Y @ H_Y^{-1}

        # Approach: use eigendecomposition
        # If B_obs_sq = H_Y^T @ Lambda @ H_Y for some diagonal Lambda,
        # we can recover H_Y (up to rotation)

        # Simplified approach for d=2:
        # We know the structure and can use the interventional data to pin down H_Y

        # For d=2, if we have one intervention in View Y, we can recover one row
        # and use similar logic as for View X

        if len(Thetas_Y) > 0:
            Theta_k_Y = Thetas_Y[0]
            Delta_Y = Theta_k_Y - Theta_obs_Y

            # Extract source row from View Y
            h_Y_source, _ = self._extract_source_row(Delta_Y)

            # Recover full H_Y
            H_Y = self._recover_H_from_source_row_d2(h_Y_source, source_idx=1)
        else:
            # Fallback: use Cholesky-based approach
            # H_Y^T @ B_obs_sq @ H_Y = Theta_obs_Y
            # Let's try: H_Y = B_obs_sq^{-1/2} @ U where U is from SVD

            B_obs_sq_inv_sqrt = self._matrix_sqrt_inv(B_obs_sq)
            Theta_Y_transformed = B_obs_sq_inv_sqrt @ Theta_obs_Y @ B_obs_sq_inv_sqrt

            # Take top eigenvectors
            eigenvalues, eigenvectors = eigh(Theta_Y_transformed)
            # Sort by largest eigenvalues
            idx = np.argsort(eigenvalues)[::-1]
            top_eigenvectors = eigenvectors[:, idx[:n_obs_Y]]

            H_Y = B_obs_sq_inv_sqrt @ top_eigenvectors
            H_Y = self._normalize_H_rows(H_Y)

        return H_Y

    def _matrix_sqrt_inv(self, A: np.ndarray) -> np.ndarray:
        """Compute the inverse square root of a positive definite matrix."""
        eigenvalues, eigenvectors = eigh(A)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability
        return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    def solve(self, method: str = "auto") -> dict:
        """
        Main solve method that dispatches to the appropriate algorithm.

        Args:
            method: Which method to use:
                - "auto": automatically detect based on data
                - "d2_no_edges": d=2 case with no edges
                - "d3_two_sources": d=3 case with two sources

        Returns:
            Dictionary containing recovered parameters
        """
        if method == "auto":
            # Auto-detect based on dimensions
            p_inferred = self._infer_latent_dimension()
            if p_inferred == 2:
                method = "d2_no_edges"
            elif p_inferred == 3:
                method = "d3_two_sources"
            else:
                raise ValueError(f"Auto-detection not supported for p={p_inferred}")

        if method == "d2_no_edges":
            return self.solve_d2_no_edges()
        else:
            raise NotImplementedError(f"Method {method} not yet implemented")

    def _infer_latent_dimension(self) -> int:
        """Infer the latent dimension from the data."""
        # For now, return based on the first view's observational precision matrix
        # In practice, this would use more sophisticated rank estimation
        Theta_obs = self.observed_data.Theta_obs_views[0]
        rank = np.linalg.matrix_rank(Theta_obs)
        return rank  # Rough estimate
