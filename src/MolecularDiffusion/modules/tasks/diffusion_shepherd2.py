"""ShEPhERD-2 platform task (``task_type: diffusion_shepherd2``).

A **thin delegator**, by design. Every number this task produces is computed by
upstream's own vendored ``LightningModule``
(``MolecularDiffusion.modules.models.shepherd2.lightning_module``): the
batch->input_dict mapping is its ``get_training_input_dict``, the forward is its
``forward_training``, and the four losses are its ``x{1,2,3,4}_denoising_loss``.
Nothing here re-derives any of them. Re-deriving them is specifically what
INTEGRATION_PLAN.md's directive D1 ("stay as close as possible to the shepherd2
git") forbids, because the EDM per-modality loss weighting
``(sigma^2 + sd^2)/(sigma^2 * sd^2)`` and the 50/50 real/non-bond balance are
easy to reproduce *almost* correctly and impossible to notice getting wrong.

How this differs from ShEPhERD-v1 (``modules/tasks/pharmacophore.py``), i.e. why
it is a new task rather than a config of the old one:

* **objective** -- v1 regresses eps against ``x_noise``/``pos_noise`` with unit
  weights; v2 regresses the *clean* structure with per-modality EDM weights.
* **noise process** -- v1 passes a discrete ``timestep`` indexed into
  ``alpha_ts``/``sigma_ts`` arrays; v2 passes a continuous per-molecule ``sigma``
  and does pre/post-conditioning inside ``Model.forward``.
* **backbone** -- EquiformerV3, not V2. Different modules, different state-dict
  keys, no weight compatibility.
* **feature widths** -- 13 atom types (the ``Dummy`` class is new in v2) -> 18
  node channels, against v1's 12 -> 17.

ShEPhERD-v1 and its ``shepherd-gdb17`` zoo checkpoint are untouched by this file.

Generation
----------
``sample()`` and ``sample_composition()`` are thin wrappers over the vendored
samplers. Note the hazard they exist to manage: ``inference/`` and
``comp_inference/`` read exactly three attributes off the object they are handed
-- ``.params``, ``.model`` and ``.device`` -- and the vendored ``LightningModule``
already exposes all three. So ``self.model`` is handed over directly and no
wrapper class exists. ``device`` and ``to()`` are overridden below because the
inner module's device is what the samplers actually use; see the ``.to()``
docstring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from MolecularDiffusion import core
from MolecularDiffusion.modules.tasks.task import Task

logger = logging.getLogger(__name__)


@core.Registry.register("Shepherd2Generative")
class Shepherd2Generative(Task, core.Configurable):
    """Multi-modal EDM diffusion task for ShEPhERD-2.

    Modalities: ``x1`` molecular structure (atoms + bonds + formal charges),
    ``x2`` shape point cloud (present but untrained in the release), ``x3``
    surface electrostatic potential, ``x4`` pharmacophore points + direction
    vectors.

    This class is both the Hydra factory and the task: ``build()`` returns
    ``self``, and ``self.task`` points at ``self``, matching the ShEPhERD-v1
    precedent and what ``cli/train.py`` (``task_module.build()`` then
    ``task_module.task``) and ``cli/generate.py``
    (``task = task_factory.build()``) each expect.
    """

    _option_members = {"task"}

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        modality_weights: Optional[Dict[str, float]] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.task_type = "diffusion_shepherd2"

        params = self._resolve_params(model_params, checkpoint_path)
        self.params = params

        from MolecularDiffusion.modules.models.shepherd2.lightning_module import (
            LightningModule as Shepherd2LightningModule,
        )

        self.model = Shepherd2LightningModule(params)

        # `inference/` and `comp_inference/` read `model_pl.params`, `.model` and
        # `.device` only -- all three already exist on the vendored module, so it
        # is handed to the samplers as-is. The one extra alias: the *inherited*
        # `PharmacophoreConditionGenerator._load_reference` reaches for
        # `task.model.dynamics.params` (the shape the v1 port produced). Our
        # generator subclass overrides `_load_reference`, so this alias is only a
        # safety net for anything else on that seam that still uses the v1 path.
        # object.__setattr__ is REQUIRED here, not stylistic: a plain
        # `self.model.dynamics = self.model` goes through nn.Module.__setattr__,
        # which registers the value as a submodule. Since the value is the module
        # itself, that makes `model._modules['dynamics'] is model` and
        # `state_dict()` recurses forever. Bypassing it puts the alias in
        # __dict__, where attribute lookup finds it and state_dict never walks it.
        if not hasattr(self.model, "dynamics"):
            object.__setattr__(self.model, "dynamics", self.model)

        self.modality_weights = dict(
            modality_weights or {"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0}
        )

        # ShEPhERD-2 has no learned size model: N_x1 is a user argument to the
        # sampler (`inference/sampler.py:53-54`), and the optional empirical
        # P(N_x4 | N_x1) is an .npz consumed by the generator's own
        # `distributions_path` knob, not by the task.
        self.node_dist_model = None
        self.n_node_dist = None

        # Width fixed by the checkpoint at 13 (`Dummy` is the 13th), NOT taken
        # from the data config -- a data config with a shorter vocabulary would
        # otherwise silently reshape the one-hot the released weights expect.
        self.atom_vocab = list(params["dataset"]["x1"]["atom_types"])

        # `cli/train.py:721` sets `task_module.task.<...>`; the v1 precedent.
        # object.__setattr__ bypasses BOTH Task.__setattr__ and
        # nn.Module.__setattr__, so `self` lands in __dict__ and is never
        # registered as a submodule (which would recurse in state_dict()).
        object.__setattr__(self, "task", self)

        logger.info(
            "Shepherd2Generative ready: modalities=%s, %d atom types, "
            "%d parameters, modality_weights=%s",
            params["explicit_diffusion_variables"],
            len(self.atom_vocab),
            sum(p.numel() for p in self.model.parameters()),
            self.modality_weights,
        )

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _resolve_params(
        model_params: Optional[Dict[str, Any]], checkpoint_path: Optional[str]
    ) -> Dict[str, Any]:
        """Return the ShEPhERD-2 ``params`` dict, checkpoint taking precedence."""
        if checkpoint_path:
            logger.info("Reading ShEPhERD-2 params from checkpoint: %s", checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            return ckpt["hyper_parameters"]["params"]

        if model_params is None:
            raise ValueError(
                "Shepherd2Generative needs the ShEPhERD-2 `params` dict. Use "
                "configs/tasks/diffusion_shepherd2.yaml (which carries it, "
                "generated from the released checkpoint) or set `checkpoint_path`."
            )

        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf

            if isinstance(model_params, (DictConfig, ListConfig)):
                return OmegaConf.to_container(model_params, resolve=True)
        except ImportError:  # pragma: no cover - omegaconf is a core dependency
            pass
        return dict(model_params)

    def build(self) -> "Shepherd2Generative":
        """Return the task. Both CLI seams call this; generate.py uses the result."""
        return self

    def preprocess(self, train_set=None, valid_set=None, test_set=None) -> None:
        """No dataset statistics are needed at construction.

        ShEPhERD-2 reads no marginals, no valency table and no size histogram --
        which is why the factory deliberately does not declare a ``train_set``
        parameter and so never opts into ``cli/train.py``'s train_set-injection
        seam.
        """
        if train_set is not None and len(train_set) == 0:
            raise ValueError("Training set is empty.")

    # ---------------------------------------------------------------- device
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Move the task, forwarding explicitly to the inner Lightning module.

        ``nn.Module.to`` already moves ``self.model``'s parameters, since it is a
        registered submodule. The explicit forward exists for the *non-parameter*
        state: a ``pl.LightningModule`` caches its own ``device``/dtype on
        ``_device``, and ``inference/edm_sampler.py`` allocates its noise tensors
        on ``model_pl.device``. Without this, parameters sit on the GPU while the
        sampler builds noise on the CPU and every sample dies on a device
        mismatch.
        """
        result = super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        return result

    # --------------------------------------------------------------- training
    def forward(self, batch):
        """One training step: upstream's input_dict -> forward -> four losses.

        Returns ``(total_loss, metrics)``, the shape ``cli/train.py``'s engines
        expect and the same shape ShEPhERD-v1's task returns.
        """
        model = self.model
        input_dict = model.get_training_input_dict(batch)
        output_dict = model.forward_training(input_dict)

        metrics: Dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor | float = 0.0

        def _w(name: str) -> float:
            return float(self.modality_weights.get(name, 1.0))

        if model.train_x1_denoising:
            loss_x1, feat, pos, bond = model.x1_denoising_loss(input_dict, output_dict)
            total_loss = total_loss + _w("x1") * loss_x1
            metrics["loss_x1"] = loss_x1.detach()
            metrics["feature_loss_x1"] = feat.detach()
            metrics["pos_loss_x1"] = pos.detach()
            metrics["bond_loss_x1"] = bond.detach()

        if model.train_x2_denoising:
            loss_x2 = model.x2_denoising_loss(input_dict, output_dict)
            total_loss = total_loss + _w("x2") * loss_x2
            metrics["loss_x2"] = loss_x2.detach()

        if model.train_x3_denoising:
            loss_x3, feat, pos = model.x3_denoising_loss(input_dict, output_dict)
            total_loss = total_loss + _w("x3") * loss_x3
            metrics["loss_x3"] = loss_x3.detach()
            metrics["feature_loss_x3"] = feat.detach()
            metrics["pos_loss_x3"] = pos.detach()

        if model.train_x4_denoising:
            loss_x4, feat, pos, direction = model.x4_denoising_loss(
                input_dict, output_dict
            )
            total_loss = total_loss + _w("x4") * loss_x4
            metrics["loss_x4"] = loss_x4.detach()
            metrics["feature_loss_x4"] = feat.detach()
            metrics["pos_loss_x4"] = pos.detach()
            metrics["direction_loss_x4"] = direction.detach()

        if not isinstance(total_loss, torch.Tensor):
            raise RuntimeError(
                "No modality was trained: train_x1/x2/x3/x4_denoising are all "
                "False in `model_params.training`. Check the task config."
            )

        metrics["loss_total"] = total_loss.detach()
        return total_loss, metrics

    # ------------------------------------------------------- platform metrics
    # ShEPhERD-2 is purely generative. The `"diffusion"` TaskEvalConfig that
    # `diffusion_shepherd2` resolves to by prefix is `needs_generative=True`;
    # returning empty tensors makes the in-training generative eval a no-op, and
    # real generation is driven by `MolCraftDiff generate`. Same choice, and the
    # same reason, as ShEPhERD-v1's task.
    def predict(self, batch, all_loss=None, metric=None):
        return torch.tensor([])

    def target(self, batch):
        return torch.tensor([])

    def evaluate(self, pred, target):
        return {}

    # ------------------------------------------------------------- generation
    def sample(self, **kwargs: Any) -> List[Any]:
        """Thin wrapper over the vendored ``inference.sampler.generate``.

        ``N_x1``/``N_x4`` accept ``int | list[int]``. Returns
        ``list[GeneratedSample]``, each a dict subclass carrying ``x1``
        (``atoms``, ``positions``, ``bonds``, ``formal_charges``), ``x2``, ``x3``
        and ``x4`` blocks -- plus ``trajectories`` / ``trajectories_x0`` when
        those were requested.
        """
        from MolecularDiffusion.modules.models.shepherd2.inference.sampler import generate

        return generate(model_pl=self.model, **kwargs)

    def sample_composition(self, **kwargs: Any) -> List[Any]:
        """Thin wrapper over ``comp_inference.sampler.generate_composition``.

        Multi-profile composition: logical AND over several interaction profiles
        (``composition_mode='default'``) or AND-NOT with negative weights
        (``composition_mode='conditional'``). Same ``model_pl``, so no extra
        plumbing beyond this delegation.
        """
        from MolecularDiffusion.modules.models.shepherd2.comp_inference.sampler import (
            generate_composition,
        )

        return generate_composition(model_pl=self.model, **kwargs)
