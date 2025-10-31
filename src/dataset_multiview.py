# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass
from typing import Dict, Any, List
from collections import defaultdict

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from numpy.linalg import inv, pinv
from scipy.stats import multivariate_normal, invwishart


@dataclass
class ObservedMultiViewDataset:
    """
    Stores observed precision matrices for multiple views.

    Attributes:
        Theta_obs_views: List of observational precision matrices, one per view
        Thetas_views: List of lists - for each view, list of interventional precision matrices
        view_names: Optional list of view names (e.g., ["X", "Y"])
    """
    Theta_obs_views: List[np.ndarray]
    Thetas_views: List[List[np.ndarray]]
    view_names: List[str] = None

    def __post_init__(self):
        if self.view_names is None:
            self.view_names = [f"View_{i}" for i in range(len(self.Theta_obs_views))]
        assert len(self.Theta_obs_views) == len(self.Thetas_views)


@dataclass
class MultiViewDataset:
    """
    Multi-view extension of the Dataset class.

    In the multi-view setting, we have:
    - Shared latent variables Z with causal structure
    - Multiple views, each with its own mixing matrix H_view
    - Interventions can be performed in any view
    - All views share the same latent B matrices (B_obs and Bs)

    Attributes:
        B_obs: Observational B matrix (latent causal structure)
        P: Permutation matrix
        H_views: List of mixing matrices, one per view (e.g., [H_X, H_Y])
        Bs: List of interventional B matrices
        ix2target: Mapping from intervention index to latent target
        ix2view: Mapping from intervention index to view index
        view_names: Optional list of view names (e.g., ["X", "Y"])
    """
    B_obs: np.ndarray
    P: np.ndarray
    H_views: List[np.ndarray]
    Bs: List[np.ndarray]
    ix2target: dict
    ix2view: dict  # Maps intervention index to which view it was performed in
    view_names: List[str] = None

    def __post_init__(self):
        if self.view_names is None:
            self.view_names = [f"View_{i}" for i in range(len(self.H_views))]

        # === COMPUTE OBSERVATIONAL MATRICES FOR EACH VIEW ===
        self.precision_latent = self.B_obs.T @ self.B_obs
        self.C_obs_views = []
        self.Theta_obs_views = []

        for H in self.H_views:
            C_obs = self.B_obs @ self.P @ H
            Theta_obs = C_obs.T @ C_obs
            self.C_obs_views.append(C_obs)
            self.Theta_obs_views.append(Theta_obs)

        # === COMPUTE THETA_k MATRICES FOR EACH VIEW ===
        self.Thetas_views = [[] for _ in range(len(self.H_views))]
        self.precisions_latent = []

        for k, B in enumerate(self.Bs):
            assert B.shape == self.B_obs.shape
            view_idx = self.ix2view[k]
            H = self.H_views[view_idx]

            C = B @ self.P @ H
            Theta = C.T @ C
            self.Thetas_views[view_idx].append(Theta)
            self.precisions_latent.append(B.T @ B)

    def sample_thetas(self, nsamples):
        """Sample precision matrices from inverse Wishart distributions."""
        Theta_latent_obs = invwishart(nsamples, self.precision_latent).rvs(1) * nsamples

        # Sample observational precisions for each view
        Theta_obs_views = []
        for H in self.H_views:
            Theta_obs = H.T @ Theta_latent_obs @ H
            Theta_obs_views.append(Theta_obs)

        # Sample interventional precisions
        Thetas_views = [[] for _ in range(len(self.H_views))]
        for k, precision_latent in enumerate(self.precisions_latent):
            view_idx = self.ix2view[k]
            H = self.H_views[view_idx]

            Theta_latent = invwishart(nsamples, precision_latent).rvs(1) * nsamples
            Theta = H.T @ Theta_latent @ H
            Thetas_views[view_idx].append(Theta)

        return ObservedMultiViewDataset(Theta_obs_views, Thetas_views, self.view_names)

    def sample(self, nsamples):
        """Sample observations from multivariate normal distributions."""
        # Sample observational data for each view
        observational_samples = []
        for Theta_obs in self.Theta_obs_views:
            mv = multivariate_normal(cov=pinv(Theta_obs), allow_singular=True)
            samples = mv.rvs(nsamples)
            observational_samples.append(samples)

        # Sample interventional data for each view
        interventional_samples_views = [[] for _ in range(len(self.H_views))]
        for view_idx, Thetas in enumerate(self.Thetas_views):
            for Theta in Thetas:
                mv = multivariate_normal(cov=pinv(Theta), allow_singular=True)
                samples = mv.rvs(nsamples)
                interventional_samples_views[view_idx].append(samples)

        return observational_samples, interventional_samples_views

    def get_view_interventions(self, view_idx):
        """Get all intervention indices for a specific view."""
        return [k for k, v in self.ix2view.items() if v == view_idx]

    def get_interventions_by_view(self):
        """Group interventions by view."""
        view_interventions = defaultdict(list)
        for k, view_idx in self.ix2view.items():
            view_interventions[view_idx].append(k)
        return dict(view_interventions)
