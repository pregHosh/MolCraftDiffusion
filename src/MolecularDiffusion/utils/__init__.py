"""Shared helpers, imported on first use (PEP 562).

Importing this package used to load every helper eagerly (about 5 s and 850 MB,
mostly rdkit); now a name is imported when it is first read. All public names,
``from MolecularDiffusion.utils import X`` and ``import *`` work as before, and any
submodule is reachable as an attribute (``utils.comm``, ``utils.file``, ...).
"""

import importlib

# public name -> (module, attribute); attribute None means the name IS the module
_EXPORTS = {
    # .geometry.geom_utils
    "remove_mean": (".geometry.geom_utils", "remove_mean"),
    "remove_mean_with_mask": (".geometry.geom_utils", "remove_mean_with_mask"),
    "remove_mean_with_mask_v2": (".geometry.geom_utils", "remove_mean_with_mask_v2"),
    "coord2cosine": (".geometry.geom_utils", "coord2cosine"),
    "coord2diff": (".geometry.geom_utils", "coord2diff"),
    "assert_correctly_masked": (".geometry.geom_utils", "assert_correctly_masked"),
    "assert_mean_zero": (".geometry.geom_utils", "assert_mean_zero"),
    "assert_mean_zero_with_mask": (".geometry.geom_utils", "assert_mean_zero_with_mask"),
    "translate_to_origine": (".geometry.geom_utils", "translate_to_origine"),
    "sample_center_gravity_zero_gaussian_with_mask": (".geometry.geom_utils", "sample_center_gravity_zero_gaussian_with_mask"),
    "sample_gaussian_with_mask": (".geometry.geom_utils", "sample_gaussian_with_mask"),
    "random_rotation": (".geometry.geom_utils", "random_rotation"),
    "sample_uniform_rotation_matrices": (".geometry.geom_utils", "sample_uniform_rotation_matrices"),
    "check_mask_correct": (".geometry.geom_utils", "check_mask_correct"),
    "apply_rotation_augmentation": (".geometry.geom_utils", "apply_rotation_augmentation"),
    "random_rotation_matrix": (".geometry.geom_utils", "random_rotation_matrix"),
    "MoleculeReconstructionEvaluator": (".geometry.geom_utils", "MoleculeReconstructionEvaluator"),
    # .chem.molgraph_utils
    "create_pyg_graph": (".chem.molgraph_utils", "create_pyg_graph"),
    "correct_edges": (".chem.molgraph_utils", "correct_edges"),
    "remove_mean_pyG": (".chem.molgraph_utils", "remove_mean_pyG"),
    # .chem.geom_analyzer
    "check_connected": (".chem.geom_analyzer", "check_connected"),
    "check_quality": (".chem.geom_analyzer", "check_quality"),
    "check_stability": (".chem.geom_analyzer", "check_stability"),
    # .geom_constraint
    "initialize_extra_nodes": (".geom_constraint", "initialize_extra_nodes"),
    "initialize_extra_nodes_seed": (".geom_constraint", "initialize_extra_nodes_seed"),
    "build_extra_node_template": (".geom_constraint", "build_extra_node_template"),
    "ensure_intact": (".geom_constraint", "ensure_intact"),
    "enforce_min_nodes_per_connector": (".geom_constraint", "enforce_min_nodes_per_connector"),
    "resolve_connectors": (".geom_constraint", "resolve_connectors"),
    "find_close_points_torch_and_push_op2": (".geom_constraint", "find_close_points_torch_and_push_op2"),
    # .diffusion_utils
    "prepare_context": (".diffusion_utils", "prepare_context"),
    "prepare_context_pyG": (".diffusion_utils", "prepare_context_pyG"),
    "compute_mean_mad_from_dataloader": (".diffusion_utils", "compute_mean_mad_from_dataloader"),
    # .runtime.torch
    "cpu": (".runtime.torch", "cpu"),
    "cuda": (".runtime.torch", "cuda"),
    "detach": (".runtime.torch", "detach"),
    "clone": (".runtime.torch", "clone"),
    "mean": (".runtime.torch", "mean"),
    "cat": (".runtime.torch", "cat"),
    "stack": (".runtime.torch", "stack"),
    "seed_everything": (".runtime.torch", "seed_everything"),
    "get_vram_size": (".runtime.torch", "get_vram_size"),
    "recursive_module_to_device": (".runtime.torch", "recursive_module_to_device"),
    "move_stray_tensor_attrs": (".runtime.torch", "move_stray_tensor_attrs"),
    "adjust_weights": (".runtime.torch", "adjust_weights"),
    "adjust_bias": (".runtime.torch", "adjust_bias"),
    # .file
    "download": (".file", "download"),
    "smart_open": (".file", "smart_open"),
    "extract": (".file", "extract"),
    "compute_md5": (".file", "compute_md5"),
    "get_line_count": (".file", "get_line_count"),
    # .io
    "input_choice": (".io", "input_choice"),
    "literal_eval": (".io", "literal_eval"),
    "no_rdkit_log": (".io", "no_rdkit_log"),
    "CaptureStdIO": (".io", "CaptureStdIO"),
    "capture_rdkit_log": (".io", "capture_rdkit_log"),
    "RankedLogger": (".io", "RankedLogger"),
    # .pretty
    "task_wrapper": (".pretty", "task_wrapper"),
    # .chem.sascore
    "sascore": (".chem.sascore", None),
    # .metrics.geom_metrics
    "geom_metrics": (".metrics.geom_metrics", None),
    # .geometry.geom_utils
    "geom_utils": (".geometry.geom_utils", None),
    # .chem.smilify
    "smilify_xyz2mol": (".chem.smilify", "smilify_xyz2mol"),
    "smilify_openbabel": (".chem.smilify", "smilify_openbabel"),
    # .metrics.geom_stability
    "compute_molecules_stability": (".metrics.geom_stability", "compute_molecules_stability"),
    "compute_molecules_stability_from_graph": (".metrics.geom_stability", "compute_molecules_stability_from_graph"),
    "load_valency_table": (".metrics.geom_stability", "load_valency_table"),
    "is_valid_mol": (".metrics.geom_stability", "is_valid"),
    "compute_rmsd": (".metrics.geom_stability", "compute_rmsd"),
    "compute_mmff_energy_drop": (".metrics.geom_stability", "compute_mmff_energy_drop"),
    "compute_bond_lengths_diff": (".metrics.geom_stability", "compute_bond_lengths_diff"),
    "compute_bond_angles_diff": (".metrics.geom_stability", "compute_bond_angles_diff"),
    "compute_torsion_angles_diff": (".metrics.geom_stability", "compute_torsion_angles_diff"),
    "compute_differences": (".metrics.geom_stability", "compute_differences"),
    "bond_type_to_symbol": (".metrics.geom_stability", "bond_type_to_symbol"),
}

__all__ = [
    "coord2diff",
    "remove_mean",
    "remove_mean_with_mask",
    "remove_mean_with_mask_v2",
    "assert_correctly_masked",
    "assert_mean_zero",
    "assert_mean_zero_with_mask",
    "translate_to_origine",
    "sample_center_gravity_zero_gaussian_with_mask",
    "sample_gaussian_with_mask",
    "random_rotation",
    "sample_uniform_rotation_matrices",
    "check_mask_correct",
    "coord2cosine",
    "create_pyg_graph",
    "correct_edges",
    "remove_mean_pyG",
    "apply_rotation_augmentation",
    "random_rotation_matrix",
    "MoleculeReconstructionEvaluator",
    "initialize_extra_nodes",
    "initialize_extra_nodes_seed",
    "build_extra_node_template",
    "ensure_intact",
    "enforce_min_nodes_per_connector",
    "resolve_connectors",
    "find_close_points_torch_and_push_op2",
    "check_connected",
    "check_quality",
    "check_stability",
    "prepare_context",
    "prepare_context_pyG",
    "compute_mean_mad_from_dataloader",
    "cpu",
    "cuda",
    "detach",
    "clone",
    "mean",
    "cat",
    "stack",
    "seed_everything",
    "get_vram_size",
    "recursive_module_to_device",
    "move_stray_tensor_attrs",
    "adjust_weights",
    "adjust_bias",
    "download",
    "smart_open",
    "extract",
    "compute_md5",
    "get_line_count",
    "input_choice",
    "literal_eval",
    "no_rdkit_log",
    "CaptureStdIO",
    "capture_rdkit_log",
    "RankedLogger",
    "task_wrapper",
    "sascore",
    "geom_metrics",
    "geom_utils",
    "smilify_xyz2mol",
    "smilify_openbabel",
]


def __getattr__(name: str):
    if name in _EXPORTS:
        module, attr = _EXPORTS[name]
        mod = importlib.import_module(module, __name__)
        value = mod if attr is None else getattr(mod, attr)
    elif name.startswith("__"):
        raise AttributeError(name)
    else:  # any submodule, e.g. `utils.comm`, `utils.file`
        try:
            value = importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":  # a real missing dependency inside it
                raise
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *_EXPORTS})
