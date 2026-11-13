import numpy as np
from sklearn.cluster import DBSCAN
import networkx as nx
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_networkx
import logging
from typing import List, Optional
from ase.data import covalent_radii
from .substituent_builder import select_fragment, place_fragment

logger = logging.getLogger(__name__)

CONNECTORS_KEY = "connectors"
_CONNECTOR_ALIASES = ("connector_dicts", "connector_indices")


def resolve_connectors(outpaint_cfgs) -> dict:
    """Read the growth-site spec out of an outpaint config block.

    Canonical form is ``connectors: {atom_index: [n_bonds]}``. The degree is the number
    of new bonds the connector should form, and is only honoured by plain ``outpaint`` --
    ``outpaintft`` and the guided variants apply no bonding constraint and use the keys
    alone.

    ``connector_dicts`` (same shape) and ``connector_indices`` (a bare list, degree 0)
    are accepted as deprecated aliases so existing configs keep running.

    Returns {int index: [int degree]}; empty dict if nothing is specified. Callers decide
    whether empty is an error -- ``init_method: seed`` with phantom atoms legitimately has
    no connectors.
    """
    if not outpaint_cfgs:
        return {}

    spec = outpaint_cfgs.get(CONNECTORS_KEY, None)
    if not spec:
        for alias in _CONNECTOR_ALIASES:
            spec = outpaint_cfgs.get(alias, None)
            if spec:
                logger.warning(
                    "'%s' is deprecated; use '%s: {atom_index: [n_bonds]}'. "
                    "Mapping it across for now.", alias, CONNECTORS_KEY,
                )
                break
    if not spec:
        return {}

    if isinstance(spec, dict):
        return {int(k): list(v) if isinstance(v, (list, tuple)) else [int(v)]
                for k, v in spec.items()}
    # Bare list of indices: no bonding constraint, which degree 0 expresses.
    return {int(i): [0] for i in spec}

EDGE_THRESHOLD = 2
WEIGHT_FACTOR = 2

# Nominal valence by atomic number for the common (organic) element subset.
# Used only as a heuristic to locate open attachment sites on a scaffold when
# outpaint connectors are not supplied explicitly.
_NOMINAL_VALENCE = {
    1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1,
    13: 3, 14: 4, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1,
}


def derive_connectors_from_scaffold(
    pos, atomic_numbers, scale_factor=1.3, valence_map=None,
):
    """Heuristically locate open attachment sites on a bare scaffold.

    Builds a covalent-bond graph over the scaffold atoms (radius graph filtered
    by summed covalent radii) and returns the under-coordinated heavy atoms
    (nominal valence > current bond degree) as outpaint connectors, with each
    value being the number of free valences. Falls back to the atom farthest
    from the centroid when the scaffold is fully saturated, so outpaint growth
    always has at least one anchor.

    Args:
        pos: (N, 3) atom coordinates (physical Angstrom units).
        atomic_numbers: (N,) integer atomic numbers.
        scale_factor: covalent-radii relaxation factor for bond detection.
        valence_map: optional {atomic_number: nominal_valence} override.

    Returns:
        dict ``{atom_index: [n_free_valence]}`` suitable for
        ``outpaint_cfgs['connector_dicts']``.
    """
    valence_map = valence_map or _NOMINAL_VALENCE
    pos = pos.detach()
    z = atomic_numbers.detach().long()
    n = pos.size(0)
    if n == 0:
        return {}

    device = pos.device
    ei = radius_graph(pos, r=5.0)
    radii = torch.tensor(
        [covalent_radii[int(zi)] if 0 <= int(zi) < len(covalent_radii) else 0.77
         for zi in z.tolist()],
        device=device, dtype=pos.dtype,
    )
    src, dst = ei
    bond_len = (pos[src] - pos[dst]).norm(dim=1)
    keep = bond_len <= (radii[src] + radii[dst]) * scale_factor
    src = src[keep]
    degree = torch.zeros(n, device=device, dtype=torch.long)
    if src.numel() > 0:
        degree.scatter_add_(0, src, torch.ones_like(src))

    connectors = {}
    for i in range(n):
        zi = int(z[i])
        if zi <= 1:  # skip hydrogen and padding
            continue
        free = valence_map.get(zi, 0) - int(degree[i])
        if free >= 1:
            connectors[i] = [int(free)]

    if not connectors:
        centroid = pos.mean(dim=0)
        far = int(torch.argmax((pos - centroid).norm(dim=1)))
        connectors[far] = [1]
    return connectors


def initialize_extra_nodes_seed(xh_cond, connector_indices, n_extra, seed_dist=1.0, min_dist=1.0, spread=0.5, n_bq_atom=0):
    """
    Initializes extra nodes based on seed locations.
    
    If n_bq_atom > 0, uses the last n_bq_atom positions in xh_cond as seed locations.
    Otherwise, calculates seed locations around connector atoms maximizing distance to other atoms.
    
    Args:
        xh_cond (torch.Tensor): Tensor of shape (B, N, D).
                                Dims: [coords(3) | node_features(D-3)].
        connector_indices (list or torch.Tensor): Indices of connector atoms.
        n_extra (int): Number of extra nodes to initialize per batch item.
        seed_dist (float): Distance of the seed from the connector atom (used if n_bq_atom == 0).
        min_dist (float): Minimum distance from any existing atom in xh_cond (except the connector itself).
        spread (float): Standard deviation for the normal distribution when sampling new node positions.
        n_bq_atom (int): Number of "boundary/query" atoms at the end of xh_cond to use as seeds.
                         If > 0, connector_indices and seed_dist are ignored for seed location determination.

    Returns:
        torch.Tensor: New nodes tensor of shape (B, n_extra, D).
    """
    B, N, D = xh_cond.shape
    device = xh_cond.device
    
    # Get all atom positions for distance checking
    all_pos = xh_cond[:, :, :3] # (B, N, 3)

    new_nodes_list = []
    
    for b in range(B):
        batch_new_nodes = []
        curr_mol_pos = all_pos[b] # (N, 3)
        seed_positions = {} # {seed_idx: seed_pos_tensor}
        
        if n_bq_atom > 0:
            # --- BQ ATOM MODE ---
            # Use last n_bq_atom positions as seeds
            if n_bq_atom > N:
                raise ValueError(f"n_bq_atom ({n_bq_atom}) cannot be larger than total atoms ({N})")
                
            # Extract seeds from the end of the molecule coordinates
            # Indices: N-n_bq_atom to N-1
            bq_start_idx = N - n_bq_atom
            for i in range(n_bq_atom):
                seed_idx = bq_start_idx + i
                seed_positions[i] = curr_mol_pos[seed_idx]
                
            # Connector indices for exclusion logic during sampling:
            # If we use BQ atoms, which atom should we exclude from distance check?
            # Typically BQ atoms are phantom or guides. If they are part of xh_cond, 
            # we should probably exclude the BQ atom itself corresponding to the seed.
            # Let's map seed_local_idx -> real_atom_idx
            seed_to_real_idx = {i: bq_start_idx + i for i in range(n_bq_atom)}
            
        else:
            # --- CONNECTOR MODE (Original) ---
            if isinstance(connector_indices, list):
                connector_indices = torch.tensor(connector_indices, device=device)
            
            # Extract position of connector atoms
            connector_pos = xh_cond[:, connector_indices, :3]
            
            # Identify valid seed locations for each connector
            for i, conn_idx in enumerate(connector_indices):
                conn_p = connector_pos[b, i] # (3,)
                
                # Find the direction that maximizes distance to other atoms ("away" from rest)
                num_attempts = 100
                
                # Random directions on unit sphere
                directions = torch.randn(num_attempts, 3, device=device)
                directions = directions / torch.norm(directions, dim=1, keepdim=True)
                
                # Candidate seed positions
                candidate_seeds = conn_p.unsqueeze(0) + directions * seed_dist # (num_attempts, 3)
                
                # Calculate distances to all atoms: (num_attempts, N)
                diffs = candidate_seeds.unsqueeze(1) - curr_mol_pos.unsqueeze(0)
                dists = torch.norm(diffs, dim=2)
                
                # Mask out the connector atom itself
                mask = torch.ones(N, dtype=torch.bool, device=device)
                mask[conn_idx] = False
                
                # Distances to non-connector atoms
                valid_dists = dists[:, mask] # (num_attempts, N-1)
                
                if valid_dists.shape[1] > 0:
                    # For each candidate, find distance to nearest neighbor
                    min_dists = torch.min(valid_dists, dim=1).values # (num_attempts,)
                    
                    # Pick candidate with MAX minimum distance (Maximin)
                    best_idx = torch.argmax(min_dists)
                    seed_positions[i] = candidate_seeds[best_idx]
                else:
                    seed_positions[i] = candidate_seeds[0]
            
            # In connector mode, real index is the connector index
            seed_to_real_idx = {i: connector_indices[i].item() for i in range(len(connector_indices))}

        # --- SAMPLE EXTRA NODES ---
        valid_seed_indices = torch.tensor(list(seed_positions.keys()), device=device)
        
        # Sample pool logic (ensure coverage then random)
        sample_pool = []
        
        # Ensure we have enough indices to pick from
        num_seeds = len(valid_seed_indices)
        if num_seeds == 0:
             # Should ideally not happen given the logic above
             raise ValueError("No valid seed positions found.")

        # First, ensure each is picked once if we have room
        if n_extra >= num_seeds:
            sample_pool.append(valid_seed_indices)
            remaining = n_extra - num_seeds
            if remaining > 0:
                # Sample remaining randomly with replacement
                random_indices = torch.randint(0, num_seeds, (remaining,), device=device)
                sample_pool.append(valid_seed_indices[random_indices])
        else:
             # Pick n_extra distinct ones
             # torch.randperm produces indices 0..num_seeds-1
             perm = torch.randperm(num_seeds, device=device)
             selected_indices = perm[:n_extra]
             sample_pool.append(valid_seed_indices[selected_indices])
             
        # Flatten pool
        sample_pool_tensor = torch.cat(sample_pool)
        
        # Shuffle pool
        perm = torch.randperm(len(sample_pool_tensor), device=device)
        sample_pool_tensor = sample_pool_tensor[perm]

        for i in range(n_extra):
            seed_local_idx = sample_pool_tensor[i].item()
            seed_pos = seed_positions[seed_local_idx] # (3,) tensor
            
            # Sample location from N(seed_pos, spread)
            valid_sample = False
            for _ in range(50):
                 sampled_pos = torch.normal(mean=seed_pos, std=spread)
                 
                 # Check distance against xh_cond
                 dists = torch.norm(curr_mol_pos - sampled_pos, dim=1)
                 
                 # Exclude the "parent" atom (connector or BQ atom) from distance check
                 real_parent_idx = seed_to_real_idx[seed_local_idx]
                 
                 mask = torch.ones(N, dtype=torch.bool, device=device)
                 mask[real_parent_idx] = False
                 
                 min_d = torch.min(dists[mask]) if mask.any() else float('inf')
                 
                 if min_d >= min_dist:
                     valid_sample = True
                     break
            
            # Sample features: N(0, 1)
            sampled_feats = torch.randn(D-3, device=device)
            
            node = torch.cat([sampled_pos, sampled_feats])
            batch_new_nodes.append(node)
            
        new_nodes_list.append(torch.stack(batch_new_nodes))

    new_nodes_tensor = torch.stack(new_nodes_list)
    return new_nodes_tensor


# ---------------------------------------------------------------------------
# Clean placeholder ("skeleton") builders for outpaint latent initialization.
#
# These return CLEAN, connector-anchored, clash-free coordinates for the
# to-be-generated atoms. Unlike ``initialize_extra_nodes_seed`` (which emits a
# tight blob used directly as the start latent), the output here is meant to be
# *forward-noised* (z = alpha_t * x0 + sigma_t * eps) to a partial t_start so the
# inference initialization matches the training forward marginal. Only the
# coordinates matter; node features are randomized (denoised during sampling).
#
# Builders are looked up in SKELETON_BUILDERS by a ``skeleton_type`` / family
# name. The first pass implements the random self-avoiding family
# ("random_walk", "globular"); other families are registered as TODO stubs so
# the config surface exists without wiring the geometry yet.
# ---------------------------------------------------------------------------


def _connector_seed_direction(
    conn_p, curr_mol_pos, conn_idx, seed_dist, device, num_attempts=100
):
    """Maximin outward direction from a connector (away from the scaffold).

    Returns a unit direction (3,) that maximizes the minimum distance from a
    seed placed at ``conn_p + dir * seed_dist`` to every other scaffold atom.
    """
    N = curr_mol_pos.shape[0]
    directions = torch.randn(num_attempts, 3, device=device)
    directions = directions / torch.norm(directions, dim=1, keepdim=True)
    candidate_seeds = conn_p.unsqueeze(0) + directions * seed_dist
    diffs = candidate_seeds.unsqueeze(1) - curr_mol_pos.unsqueeze(0)
    dists = torch.norm(diffs, dim=2)
    mask = torch.ones(N, dtype=torch.bool, device=device)
    if 0 <= int(conn_idx) < N:
        mask[int(conn_idx)] = False
    valid = dists[:, mask]
    if valid.shape[1] > 0:
        best = int(torch.argmax(torch.min(valid, dim=1).values))
    else:
        best = 0
    return directions[best]


def _min_clearance(cand, scaffold_pos, placed):
    """Minimum distance from ``cand`` (3,) to scaffold atoms and placed atoms."""
    d_scaf = torch.min(torch.norm(scaffold_pos - cand, dim=1)) \
        if scaffold_pos.shape[0] > 0 else torch.tensor(float("inf"))
    if placed:
        d_plac = torch.min(
            torch.norm(torch.stack(placed) - cand, dim=1)
        )
    else:
        d_plac = torch.tensor(float("inf"), device=cand.device)
    return torch.min(d_scaf, d_plac)


def _build_random_walk(
    curr_mol_pos, connector_pos, connector_real_idx, n_extra, n_feat, device,
    seed_dist=1.5, min_dist=1.0, spread=1.0, bond_len=1.5, max_tries=50,
):
    """Connected self-avoiding 3D walk grown outward from the connector(s).

    ``spread`` controls angular dispersion: 0 produces a straight walk, 1
    preserves the original equal blend of the previous and random directions,
    and larger values make the walk more tortuous.
    """
    spread = max(0.0, float(spread))
    n_conn = connector_pos.shape[0]
    heads = [connector_pos[c].clone() for c in range(n_conn)]
    dirs = [
        _connector_seed_direction(
            connector_pos[c], curr_mol_pos, connector_real_idx[c],
            seed_dist, device,
        )
        for c in range(n_conn)
    ]
    placed = []
    for i in range(n_extra):
        c = i % n_conn
        prev = heads[c]
        prev_dir = dirs[c]
        chosen = None
        chosen_dir = prev_dir
        for _ in range(max_tries):
            rd = torch.randn(3, device=device)
            rd = rd / torch.norm(rd)
            new_dir = prev_dir + rd * spread
            new_dir = new_dir / torch.norm(new_dir).clamp_min(1e-8)
            cand = prev + new_dir * bond_len
            if _min_clearance(cand, curr_mol_pos, placed) >= min_dist * 0.9:
                chosen = cand
                chosen_dir = new_dir
                break
        if chosen is None:
            chosen = prev + prev_dir * bond_len  # fallback: keep growing
        placed.append(chosen)
        heads[c] = chosen
        dirs[c] = chosen_dir
    pos = torch.stack(placed)
    feats = torch.randn(n_extra, n_feat, device=device)
    return torch.cat([pos, feats], dim=1)


def _build_globular(
    curr_mol_pos, connector_pos, connector_real_idx, n_extra, n_feat, device,
    seed_dist=2.0, min_dist=1.0, bond_len=1.5, max_tries=80,
):
    """Compact min-distance-packed blob anchored outside the scaffold."""
    centroid = connector_pos.mean(dim=0)
    ref_idx = int(connector_real_idx[0]) if len(connector_real_idx) else -1
    direction = _connector_seed_direction(
        centroid, curr_mol_pos, ref_idx, seed_dist, device,
    )
    # Radius grows with atom count (~sphere packing).
    radius = bond_len * max(1.0, float(n_extra) ** (1.0 / 3.0))
    center = centroid + direction * (seed_dist + radius * 0.5)
    placed = []
    for _ in range(n_extra):
        chosen = None
        for _ in range(max_tries):
            cand = center + torch.randn(3, device=device) * radius * 0.45
            if _min_clearance(cand, curr_mol_pos, placed) >= min_dist * 0.9:
                chosen = cand
                break
        if chosen is None:
            chosen = center + torch.randn(3, device=device) * radius * 0.45
        placed.append(chosen)
    pos = torch.stack(placed)
    feats = torch.randn(n_extra, n_feat, device=device)
    return torch.cat([pos, feats], dim=1)


def _self_avoiding_step(
    prev, prev_dir, scaffold_pos, placed, device, min_dist, bond_len, max_tries=50
):
    """One self-avoiding growth step from ``prev`` along a biased random dir."""
    for _ in range(max_tries):
        rd = torch.randn(3, device=device)
        rd = rd / torch.norm(rd)
        new_dir = prev_dir * 0.5 + rd * 0.5
        new_dir = new_dir / torch.norm(new_dir)
        cand = prev + new_dir * bond_len
        if _min_clearance(cand, scaffold_pos, placed) >= min_dist * 0.9:
            return cand, new_dir
    return prev + prev_dir * bond_len, prev_dir


def _build_fragment_family(
    curr_mol_pos, connector_pos, connector_real_idx, n_extra, n_feat, device,
    tags_all=(), seed_dist=1.5, min_dist=1.0, bond_len=1.5,
):
    """Place a bundled substituent of a given family at the connector.

    Selects the bundled fragment matching ``tags_all`` closest to ``n_extra``,
    aligns it so its bond points along the maximin outward direction, then:
      * if larger than n_extra, keeps the n_extra atoms nearest the attachment
        (a connected core);
      * if smaller, extends with a self-avoiding walk (e.g. a ring + tail).
    Falls back to a random walk when no fragment matches the family. Atom
    features are randomized.
    """
    c0 = connector_pos[0]
    ref0 = int(connector_real_idx[0]) if len(connector_real_idx) else -1
    direction = _connector_seed_direction(
        c0, curr_mol_pos, ref0, max(seed_dist, bond_len), device
    )
    _, frag = select_fragment(tags_all, n_extra)
    if frag is None:
        return _build_random_walk(
            curr_mol_pos, connector_pos, connector_real_idx, n_extra, n_feat,
            device, seed_dist=seed_dist, min_dist=min_dist, bond_len=bond_len,
        )

    roll = torch.rand((), device=device) * float(2 * np.pi)
    placed = place_fragment(frag, c0, direction, roll_angle=roll)  # (M, 3)
    M = placed.shape[0]

    if M >= n_extra:
        d_attach = torch.norm(placed - placed[0:1], dim=1)
        keep = torch.argsort(d_attach)[:n_extra]
        pos = placed[keep]
    else:
        placed_list = [placed[i] for i in range(M)]
        d_from_conn = torch.norm(placed - c0.unsqueeze(0), dim=1)
        head = placed[int(torch.argmax(d_from_conn))]
        head_dir = head - c0
        head_dir = head_dir / (torch.norm(head_dir) + 1e-8)
        for _ in range(n_extra - M):
            head, head_dir = _self_avoiding_step(
                head, head_dir, curr_mol_pos, placed_list, device,
                min_dist, bond_len,
            )
            placed_list.append(head)
        pos = torch.stack(placed_list)

    feats = torch.randn(n_extra, n_feat, device=device)
    return torch.cat([pos, feats], dim=1)


def _family_builder(tags_all):
    """Make a SKELETON_BUILDERS entry that places a fragment of a given family."""
    def _builder(
        curr_mol_pos, connector_pos, connector_real_idx, n_extra, n_feat, device,
        seed_dist=1.5, min_dist=1.0, bond_len=1.5,
    ):
        return _build_fragment_family(
            curr_mol_pos, connector_pos, connector_real_idx, n_extra, n_feat,
            device, tags_all=tags_all, seed_dist=seed_dist, min_dist=min_dist,
            bond_len=bond_len,
        )
    return _builder


def _build_auto(
    curr_mol_pos, connector_pos, connector_real_idx, n_extra, n_feat, device,
    seed_dist=1.5, min_dist=1.0, bond_len=1.5,
):
    """Size-driven family heuristic: small->chain, mid->ring, large->cage."""
    if n_extra <= 4:
        tags = ["aliphatic", "chain"]
    elif n_extra <= 8:
        tags = ["aliphatic", "branched"]
    elif n_extra <= 18:
        tags = ["aliphatic", "ring"]
    else:
        tags = ["aliphatic", "cage"]
    return _build_fragment_family(
        curr_mol_pos, connector_pos, connector_real_idx, n_extra, n_feat,
        device, tags_all=tags, seed_dist=seed_dist, min_dist=min_dist,
        bond_len=bond_len,
    )


# Registry of clean-skeleton builders keyed by family name.
# random_walk/globular are procedural; the rest place a bundled substituent
# fragment of the matching family (aromatic_fused has no fused fragment bundled,
# so it extends an aromatic ring).
SKELETON_BUILDERS = {
    "random_walk": _build_random_walk,
    "globular": _build_globular,
    "aliphatic_chain": _family_builder(["aliphatic", "chain"]),
    "aliphatic_branched": _family_builder(["aliphatic", "branched"]),
    "aliphatic_ring": _family_builder(["aliphatic", "ring"]),
    "aromatic_ring": _family_builder(["aromatic", "ring"]),
    "aromatic_fused": _family_builder(["aromatic"]),
    "cage": _family_builder(["aliphatic", "cage"]),
    "ring_tail": _family_builder(["ring"]),
    "mixed": _family_builder([]),
    "auto": _build_auto,
}


def build_extra_node_template(
    xh_cond,
    connector_indices,
    n_extra,
    method="skeleton",
    skeleton_type="random_walk",
    seed_dist=1.5,
    min_dist=1.0,
    spread=0.5,
    n_bq_atom=0,
    bond_len=1.5,
):
    """Build clean extra-node coordinates for outpaint latent initialization.

    Args:
        xh_cond: (B, N_ref, D) scaffold tensor [coords(3) | features(D-3)].
        connector_indices: indices of connector atoms in the scaffold.
        n_extra: number of extra nodes to create per batch item.
        method: "skeleton" (procedural family) or "fragment" (bundled
            substituent family) — both dispatch via skeleton_type; "seed"
            (legacy blob). "data" is handled by the caller (model).
        skeleton_type: family name when method in {"skeleton", "fragment"} (see
            SKELETON_BUILDERS). For "fragment", default to an aromatic ring when
            left at the procedural default.
        seed_dist, min_dist, spread, n_bq_atom, bond_len: geometry knobs.
            For ``random_walk``, spread controls angular dispersion. For
            legacy ``method="seed"``, it is the Gaussian position standard
            deviation.

    Returns:
        (B, n_extra, D) clean coordinates + randomized features.
    """
    if method == "seed":
        return initialize_extra_nodes_seed(
            xh_cond, connector_indices, n_extra,
            seed_dist=seed_dist, min_dist=min_dist, spread=spread,
            n_bq_atom=n_bq_atom,
        )
    if method == "fragment" and skeleton_type in ("random_walk", "globular"):
        # 'fragment' implies a substituent family; pick a sensible default.
        skeleton_type = "aromatic_ring"
    if method not in ("skeleton", "fragment"):
        raise ValueError(
            f"build_extra_node_template got method={method!r}; only 'skeleton', "
            "'fragment', and 'seed' are dispatched here ('data' is handled by "
            "the model)."
        )

    builder = SKELETON_BUILDERS.get(skeleton_type)
    if builder is None:
        raise ValueError(
            f"Unknown skeleton_type={skeleton_type!r}. "
            f"Choose from {sorted(SKELETON_BUILDERS)}."
        )

    B, N, D = xh_cond.shape
    device = xh_cond.device
    n_feat = D - 3
    if isinstance(connector_indices, list):
        connector_indices = torch.tensor(connector_indices, device=device)
    connector_indices = connector_indices.to(device=device, dtype=torch.long)

    out = []
    for b in range(B):
        curr_mol_pos = xh_cond[b, :, :3]
        connector_pos = curr_mol_pos[connector_indices]
        builder_kwargs = {
            "seed_dist": seed_dist,
            "min_dist": min_dist,
            "bond_len": bond_len,
        }
        if skeleton_type == "random_walk":
            builder_kwargs["spread"] = spread
        node = builder(
            curr_mol_pos,
            connector_pos,
            connector_indices,
            n_extra,
            n_feat,
            device,
            **builder_kwargs,
        )
        out.append(node)
    return torch.stack(out)


def initialize_extra_nodes(xh_cond, connector_indices, n_extra, eps=2.0, min_samples=1):
    """
    Initializes extra nodes based on clusters of connector atoms.

    Args:
        xh_cond (torch.Tensor): Tensor of shape (B, N, D).
                                Dims: [coords(3) | node_features(D-3)].
        connector_indices (list or torch.Tensor): Indices of connector atoms.
        n_extra (int): Number of extra nodes to initialize per batch item.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
        torch.Tensor: New nodes tensor of shape (B, n_extra, D).
    """
    B, N, D = xh_cond.shape
    device = xh_cond.device
    
    # Ensure connector_indices is a tensor
    if isinstance(connector_indices, list):
        connector_indices = torch.tensor(connector_indices, device=device)
        
    # 1. Extract position of connector atoms
    # connector_pos: (B, n_connectors, 3)
    connector_pos = xh_cond[:, connector_indices, :3]
    
    new_nodes_list = []
    
    # Process each batch item separately as clustering is per-molecule
    for b in range(B):
        pos = connector_pos[b].cpu().numpy() # (n_connectors, 3)
        
        # 2. Identify clusters (distance criteria)
        # Using DBSCAN with eps=2.0 (Angstroms, assuming unnormalized or consistently normalized) 
        # and min_samples=1.
        # If coords are normalized, eps should be adjusted. 
        # The prompt says normalize_coords=1, so eps=2.0 is reasonable for Angstroms.
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pos)
        labels = clustering.labels_
        unique_labels = set(labels)
        
        # 3. Determine mean location of clusters
        cluster_means = {} # Dict[cluster_id: mean_pos]
        for label in unique_labels:
            if label == -1: continue # Noise, shouldn't happen with min_samples=1
            mask = labels == label
            cluster_points = pos[mask]
            mean_p = np.mean(cluster_points, axis=0)
            cluster_means[label] = mean_p
            
        # 4. Sample extra nodes
        # randomly pick cluster-id and sample location from N(mean-pos, 1)
        # sample node feature from N(0, 1)
        
        batch_new_nodes = []
        available_clusters = list(cluster_means.keys())
        
        for _ in range(n_extra):
            # Pick random cluster
            if not available_clusters:
                 # Fallback if no clusters found (shouldn't happen if connectors exist)
                 mean_pos = np.mean(pos, axis=0) if len(pos) > 0 else np.zeros(3)
            else:
                cluster_id = np.random.choice(available_clusters)
                mean_pos = cluster_means[cluster_id]
            
            # Sample position: N(mean_pos, 1)
            # Shape (3,)
            sampled_pos = np.random.normal(loc=mean_pos, scale=1.0, size=(3,))
            
            # Sample features: N(0, 1)
            # Shape (D-3,)
            sampled_feats = np.random.normal(loc=0.0, scale=1.0, size=(D-3,))
            
            # Concatenate
            node = np.concatenate([sampled_pos, sampled_feats])
            batch_new_nodes.append(node)
            
        # Stack new nodes for this batch: (n_extra, D)
        new_nodes_list.append(np.stack(batch_new_nodes))
        
    # Stack all batches: (B, n_extra, D)
    new_nodes_tensor = torch.tensor(np.stack(new_nodes_list), dtype=xh_cond.dtype, device=device)
    
    return new_nodes_tensor


#%% For OP
def find_close_points_torch_and_push_op2(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    connector_indices: torch.Tensor = None,
    d_threshold_f: float = 1.0,
    d_threshold_c: float = 1.0,
    d_fixed_move: Optional[float] = None,
    alpha_max: float = 10.0,
    steps: int = 40,
    w_b: float = 0.5,
    search_method: str = "binary",
    all_frozen: bool = False,
    z_ref: torch.Tensor = None,
    z_tgt: torch.Tensor = None,
    scale_factor: float = 1.1,
    debug: bool = False ,
):  
    """
    Identifies points in the target tensor that are too close to the reference tensor and adjusts their positions.
    Also detects outliers based on subgraph connectivity using a distance threshold.
    
    Order of operations:
      1. Detect and move outliers to merge with their closest target subgraph, ensuring a minimum distance of d_threshold_c.
      2. Recalculate distances and detect violating nodes (relative to frozen points) and adjust them.
      3. Additionally, detect target nodes that are closer to any frozen nodes than the minimal distance of their 
         closest connector node to any frozen node.
    
    Args:
        ref (torch.Tensor): Reference tensor of shape (N, D).
        tgt (torch.Tensor): Target tensor of shape (M, D).
        connector_indices (torch.Tensor, optional): Indices of connector points in the reference tensor.
        d_threshold_f (float, optional): Distance threshold for frozen points.
        d_threshold_c (float, optional): Distance threshold for connector points (and outlier merging).
        d_fixed_move (float, optional): Fixed distance for moving points.
        alpha_max (float, optional): Maximum step size for adjustments.
        steps (int, optional): Number of steps for binary/adaptive search.
        w_b (float, optional): Weight for the secondary direction in adjustment.
        search_method (str, optional): Search method for finding the best adjustment ("binary", "adaptive", or "log").
        all_frozen (bool, optional): Whether all points in the reference tensor are frozen. Default is False.
        
        Toggle this if you want few bonds with the connectors.
        This is useful when there is too few frozen nodes to push the target nodes away to form valid subgraphs to the connector nodes,
        resulting in these target nodes developing in the middle of rerfernce structures, forming too many bonds with the connector nodes.
        
        
        z_ref (torch.Tensor, optional): Atomic numbers of the reference tensor. Default is None.
        z_tgt (torch.Tensor, optional): Atomic numbers of the target tensor. Default is None.
        scale_factor (float, optional): Scale factor for the covalent radii. Default is 1.1.
        debug (bool, optional): Whether to print debug information. Default is False.
        
        If these are provided, the function will consider covanlent radii of the atoms to adjust the distances
        instead of d_threshold_f.
        
        For example, see 60.xyz
        NOTE: When this is toggled, recommend to decrease the d_threshold_c to around 1.4
        and set t_critical to 0.8
        
        debug (bool, optional): Whether to print debug information. Default is False.
        
    Returns:
        torch.Tensor: Indices of the points that were adjusted.
        torch.Tensor: Updated target tensor with adjusted points.
    """
    device = ref.device
    tgt = tgt.to(device)
    COV_R = torch.tensor(covalent_radii, dtype=torch.float32, device=device)
    updated_tgt = tgt.clone()
    
    if (z_ref is not None and z_tgt is None) or (z_ref is None and z_tgt is not None):
        raise ValueError("Both z_ref and z_tgt must be specified if either is provided.")

    
    if connector_indices is None or connector_indices.numel() == 0:
        return torch.tensor([]), updated_tgt
    
    # Set up frozen and connector points.
    all_indices = torch.arange(ref.size(0), device=device)
    frozen_mask = torch.ones(ref.size(0), dtype=torch.bool, device=device)
    frozen_mask[connector_indices] = False
    if len(connector_indices) == ref.size(0):
        frozen_mask[connector_indices] = True
    frozen_indices = all_indices[frozen_mask]
    if all_frozen:
        ref_frozen = ref
    else:
        ref_frozen = ref[frozen_indices]
    ref_conn = ref[connector_indices]
    centroid = ref.mean(dim=0)  # Centroid of ref
    
    # Compute KDTree for connector points.
    tree_conn = KDTree(ref_conn.cpu().numpy())
    min_dist_conn, idx_conn = tree_conn.query(tgt.cpu().numpy())
    min_dist_conn = torch.tensor(min_dist_conn, device=device)
    idx_conn = torch.tensor(idx_conn, device=device)

    adjusted_indices = []  # Record indices that have been moved.


    # --- Step 2: Detect and Handle Violating Nodes ---
    # Recompute distances using the updated target positions.
    dist_frozen = torch.cdist(updated_tgt, ref_frozen, p=2) if ref_frozen.numel() > 0 else torch.full((tgt.size(0), 1), float('inf'), device=device)
    dist_centroid = torch.norm(updated_tgt - centroid, dim=1)
    dist_connector_to_centroid = torch.norm(ref_conn[idx_conn] - centroid, dim=1)
    dist_to_connector = torch.norm(updated_tgt - ref_conn[idx_conn], dim=1)

    if z_ref is not None and z_tgt is not None:
        r1 = COV_R[z_tgt] / scale_factor  # (M,)
        r2 = COV_R[z_ref] / scale_factor  # (N,)
        min_bond_length = r1[:, None] + r2[None, :]  # (M+N, N)
        min_bond_length = min_bond_length[:-z_ref.size(0)]
        violating_indices = torch.where(
            (dist_frozen.min(dim=1).values < min_bond_length.min(dim=1).values) & 
            ((dist_centroid <= dist_connector_to_centroid) | (dist_to_connector >= d_threshold_c))
        )[0].tolist()
    else:
    
        violating_indices = torch.where(
            (dist_frozen.min(dim=1).values < d_threshold_f) & 
            ((dist_centroid <= dist_connector_to_centroid) | (dist_to_connector >= d_threshold_c))
        )[0].tolist()
    
    # --- Additional Detection: Find target nodes that are closer to frozen nodes than
    # the minimal distance of their closest connector node to any frozen node.
    # For each target, compute its minimum distance to any frozen node.
    d_frozen_target = dist_frozen.min(dim=1).values  # Shape: (M,)
    # For each connector, compute its minimum distance to any frozen node.
    if ref_frozen.numel() > 0:
        dist_conn_frozen = torch.cdist(ref_conn, ref_frozen, p=2)  # shape: (number_of_connectors, num_frozen)
        d_min_conn = dist_conn_frozen.min(dim=1).values  # minimal distance for each connector
        # For each target, get the minimal frozen distance of its closest connector.
        d_threshold_target = d_min_conn[idx_conn]
        # Find targets where the target's minimum distance to frozen is less than that threshold.
        additional_violation_indices = torch.where(d_frozen_target < d_threshold_target)[0].tolist()
    else:
        additional_violation_indices = []
    
    # Combine both violation lists (avoiding duplicates)
    all_violation_indices = list(set(violating_indices +  additional_violation_indices))
    
    # Move each violating node to satisfy the frozen distance constraint.
    for idx in all_violation_indices:
        point = updated_tgt[idx]
        c_idx = idx_conn[idx]
        c2 = ref_conn[c_idx]
        
        a = c2 - point
        b = c2 - centroid
        direction = a + w_b * b
        dist_dir = direction.norm()
        if dist_dir < 1e-12:
            direction = torch.randn_like(point)
            dist_dir = direction.norm()
        direction_unit = direction / dist_dir
        
        if d_fixed_move is not None:
            best_alpha = d_fixed_move
        else:
            low, high = 0.0, alpha_max
            best_alpha = alpha_max
            if search_method == "adaptive":
                alpha = best_alpha
                for _ in range(steps):
                    candidate_pt = point + alpha * direction_unit
                    if torch.norm(candidate_pt - ref_frozen, dim=1).min() >= d_threshold_f:
                        best_alpha = alpha
                        alpha *= 0.5
                    else:
                        alpha *= 0.7
            elif search_method == "binary":
                for _ in range(steps):
                    mid = 0.5 * (low + high)
                    candidate_pt = point + mid * direction_unit
                    if torch.norm(candidate_pt - ref_frozen, dim=1).min() >= d_threshold_f:
                        best_alpha = mid
                        high = mid
                    else:
                        low = mid
            elif search_method == "log":
                for _ in range(steps):
                    alpha = alpha_max / (1 + torch.exp(torch.tensor(-0.1 * (_ - steps / 2))))
                    candidate_pt = point + alpha * direction_unit
                    if torch.norm(candidate_pt - ref_frozen, dim=1).min() >= d_threshold_f:
                        best_alpha = alpha  
        if debug:
            print("Mover 2: Moving point", idx, "by", best_alpha, "to satisfy frozen distance constraint.")
        updated_tgt[idx] = point + best_alpha * direction_unit
        adjusted_indices.append(idx)

 
    return torch.tensor(adjusted_indices, device=device), updated_tgt


def ensure_intact(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    connector_indices: torch.Tensor = None,
    d_threshold: float = 1.0,
    d_threshold_e: float = EDGE_THRESHOLD,
    d_fixed_move: Optional[float] = None,
    steps: int = 40,
    debug: bool = False,
):  
    """
    Identifies subgraphs of target nodes, detects outlier subgraphs based on their distance to connector nodes,
    and moves them toward the closest non-outlier subgraph.
    
    Args:
        ref (torch.Tensor): Reference tensor of shape (N, D).
        tgt (torch.Tensor): Target tensor of shape (M, D).
        connector_indices (torch.Tensor, optional): Indices of connector points in the reference tensor.
        d_threshold (float, optional): Distance threshold to determine outlier subgraphs.
        d_threshold_e (float, optional): Distance threshold for edge formation in the graph.
        d_fixed_move (float, optional): Fixed distance for moving points.   
        steps (int, optional): Number of steps for binary search while moving outlier subgraphs.
        debug (bool, optional): Whether to print debug information.
        
    Returns:
        torch.Tensor: Indices of the points that were adjusted.
        torch.Tensor: Updated target tensor with adjusted points.
    """
    device = ref.device
    tgt = tgt.to(device)
    updated_tgt = tgt.clone()
    
    if connector_indices is None or connector_indices.numel() == 0:
        return torch.tensor([]), updated_tgt

    # Compute connected components on the target using KDTree.
    tree_tgt = KDTree(tgt.cpu().numpy())
    adjacency_matrix = tree_tgt.sparse_distance_matrix(tree_tgt, max_distance=d_threshold_e, output_type='coo_matrix')
    sparse_graph = csr_matrix(
        (adjacency_matrix.data, (adjacency_matrix.row, adjacency_matrix.col)),
        shape=(tgt.size(0), tgt.size(0))
    )
    _, labels = connected_components(sparse_graph)

    # Identify subgraphs; keep indices as tensors for safe indexing.
    unique_labels = torch.tensor(labels, device=device).unique()
    subgraphs = {
        label.item(): torch.from_numpy(
            (labels == label.item()).nonzero()[0]
        ).to(device)
        for label in unique_labels
    }

    # Outliers = any component that is not the largest connected component.
    # enforce_min_nodes_per_connector handles proximity to connectors; this
    # function's sole job is to restore global connectivity.
    component_sizes = {label: len(indices) for label, indices in subgraphs.items()}
    main_label = max(component_sizes, key=component_sizes.get)
    outlier_subgraphs = [indices for label, indices in subgraphs.items() if label != main_label]
    non_outlier_indices = subgraphs[main_label]
    
    adjusted_indices = []
    if non_outlier_indices.numel() > 0 and len(outlier_subgraphs) > 0:
        for indices in outlier_subgraphs:
            subgraph_points = updated_tgt[indices]
            
            dists = torch.cdist(subgraph_points, updated_tgt[non_outlier_indices])
            closest_idx = dists.argmin(dim=1)
            closest_points = updated_tgt[non_outlier_indices][closest_idx]
            
            direction = closest_points.mean(dim=0) - subgraph_points.mean(dim=0)
            dist_dir = direction.norm()
            if dist_dir < 1e-12:
                direction = torch.randn_like(direction)
                dist_dir = direction.norm()
            direction_unit = direction / dist_dir
            
            if d_fixed_move:
                best_alpha = d_fixed_move
            else:
                # Binary search to find minimal movement
                low, high = 0.0, dist_dir
                best_alpha = high
                for _ in range(steps):
                    mid = 0.5 * (low + high)
                    candidate_pts = subgraph_points + mid * direction_unit
                    candidate_dists = torch.norm(candidate_pts - closest_points.mean(dim=0), dim=1)

                    if candidate_dists.max() > d_threshold:
                        low = mid
                    else:
                        best_alpha = mid
                        high = mid

            updated_tgt[indices] += best_alpha * direction_unit
            adjusted_indices.extend(indices.tolist())
            
            if debug:
                print(f"Moving outlier subgraph {indices.tolist()} toward closest non-outlier subgraph by {best_alpha}")
    
    return torch.tensor(adjusted_indices, device=device), updated_tgt


def enforce_min_nodes_per_connector(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    connector_indices: torch.Tensor,
    N: List[int],
    d_threshold_c: List[float],
    debug: bool = False,
) -> torch.Tensor:
    device = ref.device
    tgt = tgt.to(device)
    updated_tgt = tgt.clone()

    # Extract connector nodes from ref
    connectors = ref[connector_indices]
    num_connectors = connectors.shape[0]
    
    # Repeat a single value across connectors. N/d_threshold_c may arrive as tensors, for
    # which `x * n` scales values instead of repeating the sequence.
    if len(N) == 1:
        N = [N[0]] * num_connectors
    if len(d_threshold_c) == 1:
        d_threshold_c = [d_threshold_c[0]] * num_connectors

    if len(N) != num_connectors or len(d_threshold_c) != num_connectors:
        raise ValueError("Number of N and d_threshold_c must match number of connector nodes.")

    # Assign each target to its closest connector
    ref_conn = ref[connector_indices]
    tree_conn = KDTree(ref_conn.cpu().numpy())
    min_dist_conn, idx_conn = tree_conn.query(tgt.cpu().numpy())
    min_dist_conn = torch.tensor(min_dist_conn, device=device)
    idx_conn = torch.tensor(idx_conn, device=device)
    

    # Group target indices by their assigned connector
    groups = {i: [] for i in range(num_connectors)}
    for i, c_idx in enumerate(idx_conn.tolist()):
        groups[c_idx].append(i)

    # Process each group
    for i, tgt_inds in groups.items():
        connector_node = connectors[i]
        # Set a maximum number of iterations to avoid infinite loops.
        iteration = 0
        max_iter = len(tgt_inds)  # adjust as needed
        while iteration < max_iter:
            iteration += 1
            pts = updated_tgt[tgt_inds]  # positions for nodes in this group
            dists = torch.norm(pts - connector_node, dim=1)
            close_mask = dists <= d_threshold_c[i]
            num_close = close_mask.sum().item()
            # if debug:
            #     print(f"Group {i}: {num_close} nodes within threshold") # for debugging
            #     print(f"Group {i}: Close distances: {dists[close_mask].tolist()}")
            if num_close >= N[i]:
                break

            sorted_indices = torch.argsort(dists)
            candidate_found = False

            for j in sorted_indices:
                if close_mask[j]:
                    continue  # skip nodes already within threshold
                global_idx = tgt_inds[j]
                candidate_found = True
                chosen_global = global_idx
                chosen_pos = updated_tgt[chosen_global]

                # Compute unit direction from candidate toward the connector
                a_vec = connector_node - chosen_pos

                # # Find the target node in the group that is closest to the connector.
                # dists_group = torch.norm(updated_tgt[tgt_inds] - connector_node, dim=1)
                # min_idx = torch.argmin(dists_group)
                # closest_global = tgt_inds[min_idx]
                # b = (chosen_pos) - (position of the closest node)
                close_indices = [idx for idx in tgt_inds if torch.norm(updated_tgt[idx] - connector_node) <= d_threshold_c[i]]
                if close_indices:
                    close_positions = torch.stack([updated_tgt[idx] for idx in close_indices], dim=0)
                    centroid = close_positions.mean(dim=0)
                    b_vec = chosen_pos - centroid
                else:
                    b_vec = torch.zeros_like(a_vec)
                    
                move_vector = a_vec + WEIGHT_FACTOR * b_vec

                # Build connectivity graph using radius_graph from PyTorch Geometric.
                group_graph = nx.Graph()
                for idx in tgt_inds:
                    group_graph.add_node(idx)
                pts_group = updated_tgt[tgt_inds]
                edge_index = radius_graph(pts_group, r=EDGE_THRESHOLD, loop=False)
                # Convert local indices (relative to pts_group) to global indices.
                global_indices = torch.tensor(tgt_inds, device=device)
                global_edge_index = edge_index.clone()
                global_edge_index[0] = global_indices[edge_index[0]]
                global_edge_index[1] = global_indices[edge_index[1]]
                for m, n in global_edge_index.t().tolist():
                    group_graph.add_edge(m, n)
                
                # Extract the connected subgraph (component) containing the candidate.
                sub_nodes = list(nx.node_connected_component(group_graph, chosen_global))
                # Filter out nodes that are already within threshold.
                sub_nodes_to_move = [n for n in sub_nodes if torch.norm(updated_tgt[n] - connector_node) > d_threshold_c[i]]
                # If no nodes need to be moved, skip this candidate.
                if not sub_nodes_to_move:
                    continue

                # Determine maximum allowed translation so that none overshoot the connector.
                safe_factors = []
                mv_dot = torch.dot(move_vector, move_vector)
                if mv_dot == 0:
                    continue  # Cannot move if move_vector is zero.
                for n_idx in sub_nodes_to_move:
                    pos_n = updated_tgt[n_idx]
                    # Maximum scalar such that pos_n + s*move_vector does not overshoot.
                    s_max_n = torch.dot(connector_node - pos_n, move_vector) / mv_dot
                    safe_factors.append(s_max_n)
                safe_max = min(safe_factors)

                # Compute the distance the candidate must move to reach the threshold.
                d_chosen = torch.norm(connector_node - chosen_pos)
                desired_alpha = d_chosen - d_threshold_c[i]
                if desired_alpha < 0:
                    continue  # candidate is already within threshold (should not occur)
                desired_alpha = min(desired_alpha, safe_max)

                # Use binary search to refine the minimal translation needed.
                low = 0.0
                high = desired_alpha
                tol = 1e-5
                while high - low > tol:
                    mid = (low + high) / 2.0
                    new_distance = torch.norm((chosen_pos + mid * move_vector) - connector_node)
                    if new_distance > d_threshold_c[i]:
                        low = mid
                    else:
                        high = mid
                alpha_move = high

                # Move every node in the filtered subgraph.
                for n_idx in sub_nodes_to_move:
                    if debug:
                        print(f"Mover 3: Group {i}: Moving node {n_idx} by {alpha_move}")
                    updated_tgt[n_idx] = updated_tgt[n_idx] + alpha_move * move_vector
                break  # one subgraph move; re-evaluate the group

            if not candidate_found:
                break

    return updated_tgt


#%% For OP-FT


def translate_to_origine(coords, node_mask):
    centroid = coords.mean(dim=1, keepdim=True)  
    translation_vector = -centroid
    translated_coords = coords + translation_vector * node_mask
    return translated_coords


