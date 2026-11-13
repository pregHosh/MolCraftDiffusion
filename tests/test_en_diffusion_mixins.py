"""``EnVariationalDiffusion`` is composed from loss / sampling / guidance mixins."""

import inspect

import pytest
import torch

# Imported first on purpose: some model modules hit a known circular import otherwise.
import MolecularDiffusion.data.dataset  # noqa: F401
from MolecularDiffusion.modules.models.dynamics.egcl import EGNN_dynamics
from MolecularDiffusion.modules.models.edm import _en_diffusion_math as math_mod
from MolecularDiffusion.modules.models.edm import en_diffusion as ed
from MolecularDiffusion.modules.models.edm.en_diffusion_guidance import EnDiffusionGuidanceMixin
from MolecularDiffusion.modules.models.edm.en_diffusion_loss import EnDiffusionLossMixin
from MolecularDiffusion.modules.models.edm.en_diffusion_sampling import EnDiffusionSamplingMixin

MIXINS = (EnDiffusionLossMixin, EnDiffusionSamplingMixin, EnDiffusionGuidanceMixin)
CLS = ed.EnVariationalDiffusion


def test_mro_puts_the_mixins_before_nn_module() -> None:
    assert CLS.__mro__[:5] == (CLS, *MIXINS, torch.nn.Module)


@pytest.mark.parametrize("mixin", MIXINS)
def test_mixins_hold_only_methods(mixin) -> None:
    # no __init__, no attributes: nothing that could add parameters or state
    own = {k: v for k, v in vars(mixin).items() if k not in {"__module__", "__doc__", "__dict__", "__weakref__"}}
    assert own and all(callable(v) or isinstance(v, (staticmethod, classmethod)) for v in own.values())
    assert "__init__" not in own


def test_no_method_is_defined_twice() -> None:
    seen: dict[str, str] = {}
    for owner in (CLS, *MIXINS):
        for name, value in vars(owner).items():
            if callable(value) and not name.startswith("__"):
                assert name not in seen, f"{name} defined in {seen[name]} and {owner.__name__}"
                seen[name] = owner.__name__
    assert len(seen) >= 55


@pytest.mark.parametrize(
    ("method", "mixin"),
    [("forward", EnDiffusionLossMixin), ("compute_loss", EnDiffusionLossMixin),
     ("sample", EnDiffusionSamplingMixin), ("sample_chain", EnDiffusionSamplingMixin),
     ("sample_guidance", EnDiffusionGuidanceMixin)],
)
def test_methods_resolve_through_the_mixins(method, mixin) -> None:
    assert inspect.getattr_static(CLS, method) is vars(mixin)[method]


def test_helpers_stay_importable_from_en_diffusion() -> None:
    for name in ("_resolve_jitter_scale", "_n_int", "split_feature_axis", "split_z", "split_frame",
                 "atomic_numbers", "gaussian_KL", "gaussian_KL_for_dimension", "sum_except_batch"):
        assert getattr(ed, name) is getattr(math_mod, name)


def _tiny(loss_type="vlb"):
    torch.manual_seed(0)
    dyn = EGNN_dynamics(in_node_nf=5, context_node_nf=0, n_dims=3, hidden_nf=16, n_layers=2,
                        inv_sublayers=1, normalization_factor=10)
    dyn.device = torch.device("cpu")
    model = CLS(dynamics=dyn, in_node_nf=4, n_dims=3, timesteps=10, noise_schedule="polynomial_2",
                loss_type=loss_type, include_charges=False)
    model.device = torch.device("cpu")
    return model


def _batch(b=2, n=5):
    g = torch.Generator().manual_seed(3)
    mask = torch.ones(b, n, 1)
    mask[1, 4] = 0
    x = torch.randn(b, n, 3, generator=g) * mask
    x = x - (x.sum(1, keepdim=True) / mask.sum(1, keepdim=True)) * mask
    onehot = torch.nn.functional.one_hot(torch.randint(0, 4, (b, n), generator=g), 4).float() * mask
    edge = (mask.unsqueeze(1) * mask.unsqueeze(2)) * (1 - torch.eye(n).unsqueeze(0).unsqueeze(-1))
    return x, {"categorical": onehot, "integer": torch.zeros(b, n, 0)}, mask, edge.reshape(b * n * n, 1)


@pytest.mark.parametrize("loss_type", ["vlb", "l2"])
def test_a_real_tiny_model_trains_and_samples(loss_type) -> None:
    model = _tiny(loss_type)
    x, h, mask, edge = _batch()
    out = model(x, h, mask, edge, None)
    assert all(torch.isfinite(t).all() for t in out if torch.is_tensor(t))
    model.eval()
    with torch.no_grad():
        xs, hs, _chain = model.sample(2, 5, mask, edge, None)
    assert xs.shape == (2, 5, 3) and torch.isfinite(xs).all()
