"""Real network classes follow the calling conventions in ``models/_protocols.py``."""

import importlib
import inspect

import pytest

# Imported first on purpose: some model modules hit a known circular import otherwise.
import MolecularDiffusion.data.dataset  # noqa: F401
from MolecularDiffusion.modules.models import _protocols as proto

M = "MolecularDiffusion.modules.models."


def _params(fn) -> list[str]:
    return [p for p in inspect.signature(fn).parameters if p != "self"]


def _own_names(protocol) -> list[str]:
    return _params(protocol.__call__)


def _load(spec: str):
    mod, cls = spec.rsplit(":", 1)
    try:
        return getattr(importlib.import_module(M + mod), cls)
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"{spec} is not in this checkout ({type(exc).__name__})")


DENSE = [
    "dynamics.egcl:EGNN_dynamics", "dynamics.egt:EGT_dynamics", "gcdm.dynamics:GCDMDynamics",
    "dynamics.painn_dynamics:PaiNNDynamics", "dynamics.transformer_dynamics:TransformerDynamics",
    "gfmdiff.dynamics:GFMDiffDynamics", "dynamics.equiformer_v2_dynamics:EquiformerV2_dynamics",
    "geoldm.networks:EGNN_dynamics_QM9",
]
PYG = [
    "dynamics.egt_pyg:EGT_dynamics_PyG", "dynamics.egcl_dynamics_wrapper:EGNN_dynamics_PyGAdapter",
    "dynamics.esen:eSEN_dynamics",
]
PADDED = [
    "tabasco_egnn.egnn_backbone:EGNNBackbone",
    "tabasco_equiformer.equiformer_backbone:EquiformerV2TabascoBackbone",
    "tabasco_gvp.gvp_backbone:GVPBackbone",
]


def test_protocols_describe_what_the_tests_check() -> None:
    assert _own_names(proto.DenseDynamics) == ["t", "xh", "node_mask", "edge_mask", "context"]
    assert _own_names(proto.PaddedBackbone) == ["coords", "atomics", "padding_mask", "t"]
    assert _own_names(proto.PygDynamics)[0] == "mol_graph"


@pytest.mark.parametrize("spec", DENSE)
def test_dense_dynamics_calling_convention(spec: str) -> None:
    cls = _load(spec)
    # the wrappers keep the real signature on `_forward`; otherwise `forward` carries it
    fn = cls._forward if hasattr(cls, "_forward") else cls.forward
    names = _params(fn)[:5]
    first = {"t", "t_or_mol_graph"}  # EquiformerV2 accepts a graph in the same slot
    assert names[0] in first and names[1:] == ["xh", "node_mask", "edge_mask", "context"]


@pytest.mark.parametrize("spec", PYG)
def test_pyg_dynamics_calling_convention(spec: str) -> None:
    cls = _load(spec)
    assert _params(cls._forward)[0] == "mol_graph"


@pytest.mark.parametrize("spec", PADDED)
def test_padded_backbone_calling_convention(spec: str) -> None:
    cls = _load(spec)
    params = [p for n, p in inspect.signature(cls.forward).parameters.items() if n != "self"]
    assert [p.name for p in params[:4]] == ["coords", "atomics", "padding_mask", "t"]
    # anything after the required prefix (e.g. an optional `condition`) must have a default
    assert all(p.default is not inspect.Parameter.empty for p in params[4:])


def test_protocols_are_not_base_classes() -> None:
    # nothing inherits from them, so the existing class hierarchy is untouched
    for spec in DENSE + PYG + PADDED:
        try:
            cls = _load(spec)
        except pytest.skip.Exception:
            continue
        assert not any(b.__module__ == proto.__name__ for b in cls.__mro__)
