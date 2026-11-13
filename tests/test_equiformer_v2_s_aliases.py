"""``equiformer_v2_s`` modules that match ``equiformer_v2`` are aliases of it."""

import importlib
import pickle

import pytest

BASE = "MolecularDiffusion.modules.layers."
ALIASED = [
    "activation", "drop", "gaussian_rbf", "input_block", "layer_norm",
    "module_list", "radial_function", "so2_ops", "wigner",
]
# these differ in behaviour (guard, __repr__, constructor), so they stay forks
FORKS = ["edge_rot_mat", "so3", "transformer_block"]


@pytest.mark.parametrize("name", ALIASED)
def test_old_path_is_the_same_module_object(name: str) -> None:
    old = importlib.import_module(f"{BASE}equiformer_v2_s.{name}")
    new = importlib.import_module(f"{BASE}equiformer_v2.{name}")
    assert old is new


@pytest.mark.parametrize("name", FORKS)
def test_forks_stay_separate(name: str) -> None:
    old = importlib.import_module(f"{BASE}equiformer_v2_s.{name}")
    new = importlib.import_module(f"{BASE}equiformer_v2.{name}")
    assert old is not new


def test_old_checkpoint_style_pickle_global_resolves() -> None:
    # what a checkpoint saved before the merge stores for a layer class
    payload = b"cMolecularDiffusion.modules.layers.equiformer_v2_s.activation\nScaledSiLU\n."
    cls = pickle.loads(payload)  # noqa: S301
    assert cls.__module__ == f"{BASE}equiformer_v2.activation"
