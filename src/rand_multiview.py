# === IMPORTS: BUILT-IN ===
import random

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import causaldag as cd
from scipy.linalg import orth

# === IMPORTS: LOCAL ===
from src.utils.permutations import get_inverse_permutation, get_permutation_matrix
from src.dataset_multiview import MultiViewDataset
from src.utils.linalg import normalize_H
from src.rand import get_intervention


def rand_multiview_model(
    latent_dag: cd.DAG,
    view_configs: list,
    nodes2num_ivs_per_view: list,
    perm: list = None,
    seed: int = None,
    no_perm: bool = False,
    iv_type: str = "hard"
):
    """
    Generate a random multi-view causal model.

    Args:
        latent_dag: The latent causal DAG structure
        view_configs: List of dicts, each with keys:
            - 'nnodes_obs': number of observed nodes in this view
            - 'name': optional name for the view
            - 'orthogonal_h': whether to make H orthogonal
            - 'upper_triangular_h': whether to make H upper triangular
        nodes2num_ivs_per_view: List of dicts (one per view), each mapping
            latent node index to number of interventions on that node in this view
        perm: Permutation to apply to latent variables
        seed: Random seed
        no_perm: If True, use identity permutation
        iv_type: Type of intervention ("hard" or "soft")

    Returns:
        MultiViewDataset containing all the multi-view data

    Example:
        For d=2 with 2 views (X and Y):
        view_configs = [
            {'nnodes_obs': 3, 'name': 'X'},  # View X: 3 observed variables
            {'nnodes_obs': 2, 'name': 'Y'}   # View Y: 2 observed variables
        ]
        nodes2num_ivs_per_view = [
            {0: 1},  # View X: 1 intervention on z_0
            {1: 1}   # View Y: 1 intervention on z_1
        ]
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    g = cd.rand.rand_weights(latent_dag)
    p = latent_dag.nnodes

    # === CREATE OBSERVATIONAL B MATRIX (shared across all views) ===
    I = np.eye(p)
    A = g.to_amat()
    Omega_half = np.diag(np.random.uniform(2, 4, p))
    B_obs = np.round(Omega_half @ (I - A), 3)

    # === CREATE H MATRICES (one per view) ===
    H_views = []
    view_names = []

    for view_config in view_configs:
        nnodes_obs = view_config.get('nnodes_obs', p)
        orthogonal_h = view_config.get('orthogonal_h', False)
        upper_triangular_h = view_config.get('upper_triangular_h', False)
        view_name = view_config.get('name', f'View_{len(H_views)}')

        if upper_triangular_h:
            H = np.triu(np.round(np.random.uniform(-2, 2, size=(p, nnodes_obs)), 2), k=0)
        else:
            H = np.random.uniform(-2, 2, size=(p, nnodes_obs))
            H = normalize_H(H)

        if orthogonal_h:
            H = orth(H)

        H_views.append(H)
        view_names.append(view_name)

    # === CREATE P MATRIX ===
    perm = list(range(p)) if perm is None else perm
    invperm = get_inverse_permutation(perm)
    P = get_permutation_matrix(perm)
    if no_perm:
        P = np.eye(p, dtype=int)

    # === CREATE INTERVENTIONAL B MATRICES ===
    # For multi-view, we track which intervention was performed in which view
    Bs = []
    ix2target = {}
    ix2view = {}
    intervention_counter = 0

    for view_idx, nodes2num_ivs in enumerate(nodes2num_ivs_per_view):
        for node, num_ivs in nodes2num_ivs.items():
            ix = invperm[node]
            for _ in range(num_ivs):
                B = get_intervention(B_obs, ix, iv_type=iv_type)
                Bs.append(B)
                ix2target[intervention_counter] = ix
                ix2view[intervention_counter] = view_idx
                intervention_counter += 1

    return MultiViewDataset(
        B_obs=B_obs,
        P=P,
        H_views=H_views,
        Bs=Bs,
        ix2target=ix2target,
        ix2view=ix2view,
        view_names=view_names
    )


def rand_multiview_model_d2_no_edges(
    seed: int = None,
    nnodes_obs_X: int = 3,
    nnodes_obs_Y: int = 2,
):
    """
    Generate d=2 multi-view model with no edges (both latent nodes are sources).
    This corresponds to the d=2 case in the proof document.

    Args:
        seed: Random seed
        nnodes_obs_X: Number of observed variables in View X
        nnodes_obs_Y: Number of observed variables in View Y

    Returns:
        MultiViewDataset for d=2 case
    """
    # Create empty DAG with 2 nodes (no edges)
    latent_dag = cd.DAG(nodes=[0, 1])

    view_configs = [
        {'nnodes_obs': nnodes_obs_X, 'name': 'X'},
        {'nnodes_obs': nnodes_obs_Y, 'name': 'Y'}
    ]

    # One intervention on each node, one in each view
    nodes2num_ivs_per_view = [
        {0: 1},  # View X: intervene on z_0
        {1: 1}   # View Y: intervene on z_1
    ]

    return rand_multiview_model(
        latent_dag=latent_dag,
        view_configs=view_configs,
        nodes2num_ivs_per_view=nodes2num_ivs_per_view,
        seed=seed,
        no_perm=True  # No permutation for simplicity
    )


def rand_multiview_model_d3_one_source(
    seed: int = None,
    nnodes_obs_X: int = 4,
    nnodes_obs_Y: int = 3,
    source_node: int = 0
):
    """
    Generate d=3 multi-view model where only one node is a source.
    This corresponds to the d=3 case in the proof document.

    Args:
        seed: Random seed
        nnodes_obs_X: Number of observed variables in View X
        nnodes_obs_Y: Number of observed variables in View Y
        source_node: Which node is the source (0, 1, or 2)

    Returns:
        MultiViewDataset for d=3 case with one source
    """
    # Create DAG with 3 nodes, where source_node has no parents
    latent_dag = cd.DAG(nodes=[0, 1, 2])

    # Add edges to make only source_node a source
    # For example, if source_node=0, add edges 0->1, 0->2
    if source_node == 0:
        latent_dag.add_arc(0, 1)
        latent_dag.add_arc(0, 2)
    elif source_node == 1:
        latent_dag.add_arc(1, 0)
        latent_dag.add_arc(1, 2)
    else:  # source_node == 2
        latent_dag.add_arc(2, 0)
        latent_dag.add_arc(2, 1)

    view_configs = [
        {'nnodes_obs': nnodes_obs_X, 'name': 'X'},
        {'nnodes_obs': nnodes_obs_Y, 'name': 'Y'}
    ]

    # Intervene on source in View X, intervene on another node in View Y
    nodes2num_ivs_per_view = [
        {source_node: 1},  # View X: intervene on source
        {(source_node + 1) % 3: 1}  # View Y: intervene on non-source
    ]

    return rand_multiview_model(
        latent_dag=latent_dag,
        view_configs=view_configs,
        nodes2num_ivs_per_view=nodes2num_ivs_per_view,
        seed=seed,
        no_perm=True
    )


def rand_multiview_model_d3_two_sources(
    seed: int = None,
    nnodes_obs_X: int = 4,
    nnodes_obs_Y: int = 3,
):
    """
    Generate d=3 multi-view model where two nodes are sources.
    This corresponds to the favorable d=3 case in the proof document.

    Args:
        seed: Random seed
        nnodes_obs_X: Number of observed variables in View X
        nnodes_obs_Y: Number of observed variables in View Y

    Returns:
        MultiViewDataset for d=3 case with two sources
    """
    # Create DAG with 3 nodes where nodes 0 and 1 are sources
    # z_0 and z_1 both point to z_2
    latent_dag = cd.DAG(nodes=[0, 1, 2])
    latent_dag.add_arc(0, 2)
    latent_dag.add_arc(1, 2)

    view_configs = [
        {'nnodes_obs': nnodes_obs_X, 'name': 'X'},
        {'nnodes_obs': nnodes_obs_Y, 'name': 'Y'}
    ]

    # Intervene on both sources, one in each view
    nodes2num_ivs_per_view = [
        {0: 1},  # View X: intervene on z_0 (source)
        {1: 1}   # View Y: intervene on z_1 (source)
    ]

    return rand_multiview_model(
        latent_dag=latent_dag,
        view_configs=view_configs,
        nodes2num_ivs_per_view=nodes2num_ivs_per_view,
        seed=seed,
        no_perm=True
    )
