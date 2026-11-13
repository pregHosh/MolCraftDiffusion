"""GCDM property optimization -- refine molecules you already have towards a
property target.

This is an ADDITIVE sibling of ``GenerativeFactory``, not a modification of
it. ``cli/generate.py`` does
``hydra.utils.instantiate(cfg.interference, task=task)`` followed by
``generator.run()``; the ``GenerativeFactory`` annotation on that line is
cosmetic and there is no registry, so a new class in a new file plugs in
purely through its ``_target_``.

WHAT IT DOES (and what it is NOT)

    Port of GCDM's ``mol_gen_optimize``
    (``others/bio-diffusion/src/models/components/variational_diffusion.py:1416``),
    driven from ``src/mol_gen_eval_optimization_qm9.py:152``.

    It is a **time-compressed reverse loop seeded with a real molecule**:

    1. The clean molecule is normalized, then forward-diffused to
       ``t0 = noise_level`` via ``q(z_t0 | x)`` -- SDEdit-style. See
       DEVIATION below for why this step exists.
    2. The ordinary reverse loop then runs for ``num_optimization_timesteps``
       (K = 100) steps, with the **time axis rescaled by K rather than by the
       model's T = 1000** and truncated at ``t0`` -- so step K-1 is presented
       to the network as ``t = t0``. Upstream calls the rescaling
       ``norm_with_original_timesteps=False`` and its own comment marks it as
       "important": dividing by ``self.model.T`` instead would compress the
       trajectory into the last 10% of the schedule and produce a near-no-op.
    3. The context vector is the desired property value, so the reverse loop
       pulls the molecule towards it.
    4. The whole thing repeats ``iterations`` times, each round re-seeded
       with the previous round's output. Upstream applies **no filtering or
       selection between rounds** and neither does this.

    DEVIATION FROM UPSTREAM (deliberate, attempt 2)

    Upstream hands the *clean* normalized latent straight to
    ``sample_p_zs_given_zt`` at ``t = 1.0``
    (``variational_diffusion.py:1455-1470``); the caller
    (``mol_gen_eval_optimization_qm9.py:161-171``) does no forward noising
    either, so upstream really has no ``q(z_t | x)`` step. That is a
    signal/noise-scale mismatch: over the whole chain the reverse process
    multiplies the latent by ``1 / alpha(t=1) ~ 300``, which is only
    cancelled when the seed is drawn from ``N(0, I)``. Reproduced faithfully
    in attempt 1 it blew molecules apart (135 A median nearest-neighbour
    distance, 0/20 connected), while the *same* checkpoint through the
    ordinary conditional sampler from a Gaussian seed gave 8/8 connected at
    1.03 A. So the loop and the weights are fine and the seed is not.
    Forward-diffusing to ``t0`` puts the seed on the scale the network was
    trained to see at ``t0``, which is the standard way to edit an existing
    sample with a diffusion model. ``noise_level = 1.0`` restores upstream's
    literal behaviour (full noise, i.e. the seed only sets the atom count).

    **The default 0.5 remains UNVALIDATED.** It is a reasonable middle value
    chosen to make the mode usable (``alpha(0.5) = 0.75``, so the seed keeps
    most of its signal), not a tuned optimum. A 0.1/0.3/0.5/0.7 sweep has
    since been run (attempt-2 smoke test) -- its main result was that
    ``noise_level`` was NOT the cause of the failure being chased at the
    time, since every value failed identically. On the *corrected* code path
    a follow-up at K=100 measured 0.3 as marginally better than 0.5 (4/4 vs
    3/4 connected, comparable ~1.02 A median nearest-neighbour distance), but
    that is n=4 on one property with one checkpoint -- far too weak to call
    an optimum, so 0.5 stands as the default. Do not read either number as a
    result. Sweep it for your property before quoting anything that depends
    on it.

    No property classifier is loaded. GCDM's EGNN classifiers only *score*
    finished samples for the paper's table (``src/__init__.py:208``); no
    gradient ever flows through them into the loop. Scoring the result is out
    of scope here -- the platform already has a scoring seam at
    ``GenerativeFactory.property_prediction``.

REQUIREMENTS
    A property-conditional checkpoint (GCDM ships six for QM9: alpha, gap,
    homo, lumo, mu, Cv) and a directory of ``.xyz`` starting molecules.
    Nothing generates those here on purpose -- feed it the output of an
    ordinary ``MolCraftDiff generate`` run, exactly as GCDM's own README does
    in two stages (README:181 then :184).
"""

import glob
import logging
import os
from typing import Optional, Sequence

import torch

from MolecularDiffusion.utils.geometry.geom_utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    save_xyz_file,
)

logger = logging.getLogger(__name__)

# Minimal symbol -> Z table for the elements GCDM's two vocabularies use.
_ATOMIC_NUMBER = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Al": 13, "Si": 14,
    "P": 15, "S": 16, "Cl": 17, "As": 33, "Se": 34, "Br": 35, "I": 53,
    "Hg": 80, "Bi": 83,
}


def read_xyz(path: str):
    """Return ``(symbols, positions)`` from a plain ``.xyz`` file."""
    with open(path) as handle:
        lines = handle.read().splitlines()
    n_atoms = int(lines[0].split()[0])
    symbols, positions = [], []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(v) for v in parts[1:4]])
    return symbols, torch.tensor(positions, dtype=torch.float32)


class GCDMOptimizeFactory:
    """Interference-side entry point for GCDM property optimization.

    Parameters
    ----------
    task:
        The loaded ``GeomMolecularGenerative`` task (injected by
        ``cli/generate.py``).
    condition_configs:
        ``input_mols_dir`` (required), plus ``iterations`` and
        ``num_optimization_timesteps`` if not given at the top level.
    target_values / property_names:
        The property target, in the property's own units. Normalized through
        the task's existing ``prop_dist_model`` exactly as
        ``GenerativeFactory`` does.
    iterations:
        Optimization rounds (upstream default 1 --
        ``mol_gen_eval_optimization_qm9.yaml:27``).
    num_optimization_timesteps:
        K, the compressed trajectory length (upstream default 100 --
        ``mol_gen_eval_optimization_qm9.yaml:28``; the ``num_timesteps: 10``
        two lines above it belongs to the *initial unconditional generation*
        at ``mol_gen_eval_optimization_qm9.py:335``, not to this loop). Note
        K is deliberately NOT the model's T.
    noise_level:
        ``t0`` in ``(0, 1]`` -- how far the input is forward-diffused before
        the reverse walk starts. Low = stays close to the input molecule,
        1.0 = full noise (the input then only fixes the atom count). See the
        module docstring's DEVIATION note.
    """

    def __init__(
        self,
        task,
        condition_configs: Optional[dict] = None,
        target_values: Sequence[float] = (),
        property_names: Sequence[str] = (),
        iterations: int = 1,
        num_optimization_timesteps: int = 100,
        noise_level: float = 0.5,
        batch_size: int = 8,
        seed: int = 86,
        output_path: str = "gcdm_optimized",
        save_every_iteration: bool = True,
        **kwargs,
    ) -> None:
        self.task = task
        self.condition_configs = dict(condition_configs or {})
        self.target_values = list(target_values)
        self.property_names = list(property_names)
        self.iterations = int(
            self.condition_configs.get("iterations", iterations)
        )
        self.num_optimization_timesteps = int(
            self.condition_configs.get(
                "num_optimization_timesteps", num_optimization_timesteps
            )
        )
        self.noise_level = float(
            self.condition_configs.get("noise_level", noise_level)
        )
        if not 0.0 < self.noise_level <= 1.0:
            raise ValueError(
                f"noise_level must be in (0, 1]; got {self.noise_level}"
            )
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.output_path = output_path
        self.save_every_iteration = bool(save_every_iteration)
        self.input_mols_dir = self.condition_configs.get("input_mols_dir")
        if kwargs:
            logger.info(f"GCDMOptimizeFactory ignoring extra keys: {list(kwargs)}")

        # GeomMolecularGenerative does not always expose `.device` (it is a
        # plain nn.Module in some load paths), so read it off the weights.
        self.device = next(task.model.parameters()).device

        if not self.input_mols_dir:
            raise ValueError(
                "GCDMOptimizeFactory needs condition_configs.input_mols_dir -- "
                "a directory of .xyz molecules to optimize. Produce one with "
                "an ordinary `MolCraftDiff generate` run first."
            )

    # -- context -----------------------------------------------------------
    def _build_context_row(self) -> Optional[torch.Tensor]:
        """``(1, n_props)`` normalized context, or ``None``.

        Mirrors the task's own normalization block
        (``modules/tasks/diffusion.py:948-969``, inside ``sample_conditonal``)
        so a conditional GCDM checkpoint is steered with the identical
        arithmetic every other conditional model here uses -- including the
        ``value_n`` branch and the loud ``else``. (The similar-looking block
        in ``GenerativeFactory.structural_guidance`` is NOT the reference: it
        is missing ``value_n`` and silently passes the raw value through,
        which fed this loop a 10x out-of-distribution context.)
        """
        prop_dist = getattr(self.task, "prop_dist_model", None)
        if not self.target_values or prop_dist is None:
            return None

        method = self.task.normalize_condition
        values = []
        for i, key in enumerate(prop_dist.distributions):
            if method is None:
                val = self.target_values[i]
            elif method == "mad":
                mean = prop_dist.normalizer[key]["mean"]
                mad = prop_dist.normalizer[key]["mad"]
                val = (self.target_values[i] - mean) / mad
            elif method == "maxmin":
                low = prop_dist.normalizer[key]["min"]
                high = prop_dist.normalizer[key]["max"]
                val = 2 * (self.target_values[i] - low) / (high - low) - 1
            elif "value" in method:  # "value_n": divide by n
                val = self.target_values[i] / float(method.split("_")[1])
            else:
                raise ValueError(
                    f"Unknown normalization method: {method}"
                )
            values.append(torch.tensor([float(val)]).unsqueeze(1))
        return torch.cat(values, dim=1).float().to(self.device)

    # -- batching ----------------------------------------------------------
    def _load_inputs(self):
        paths = sorted(glob.glob(os.path.join(self.input_mols_dir, "*.xyz")))
        if not paths:
            raise FileNotFoundError(
                f"no .xyz files in {self.input_mols_dir}"
            )
        vocab = list(self.task.atom_vocab)
        index = {symbol: i for i, symbol in enumerate(vocab)}

        molecules = []
        for path in paths:
            symbols, positions = read_xyz(path)
            unknown = [s for s in symbols if s not in index]
            if unknown:
                logger.warning(
                    f"skipping {os.path.basename(path)}: element(s) "
                    f"{sorted(set(unknown))} not in the model's vocabulary"
                )
                continue
            one_hot = torch.zeros(len(symbols), len(vocab))
            one_hot[torch.arange(len(symbols)), [index[s] for s in symbols]] = 1
            charges = torch.tensor(
                [float(_ATOMIC_NUMBER[s]) for s in symbols]
            ).unsqueeze(-1)
            molecules.append((positions, one_hot, charges, path))
        if not molecules:
            raise ValueError(
                f"none of the {len(paths)} .xyz files in "
                f"{self.input_mols_dir} are encodable with the model's "
                f"vocabulary {vocab}"
            )
        return molecules

    def _pad(self, molecules):
        """Dense ``(B, N, ...)`` batch + node/edge masks."""
        device = self.device
        n_max = max(m[0].shape[0] for m in molecules)
        bsz = len(molecules)
        n_classes = molecules[0][1].shape[1]

        x = torch.zeros(bsz, n_max, 3)
        h_cat = torch.zeros(bsz, n_max, n_classes)
        h_int = torch.zeros(bsz, n_max, 1)
        node_mask = torch.zeros(bsz, n_max, 1)
        for i, (positions, one_hot, charges, _) in enumerate(molecules):
            n = positions.shape[0]
            x[i, :n] = positions
            h_cat[i, :n] = one_hot
            h_int[i, :n] = charges
            node_mask[i, :n] = 1.0

        edge_mask = node_mask.squeeze(-1).unsqueeze(1) * node_mask.squeeze(
            -1
        ).unsqueeze(2)
        edge_mask = edge_mask * ~torch.eye(n_max, dtype=torch.bool).unsqueeze(0)
        edge_mask = edge_mask.reshape(bsz * n_max * n_max, 1)

        return (
            x.to(device),
            h_cat.to(device),
            h_int.to(device),
            node_mask.to(device),
            edge_mask.to(device),
        )

    # -- the loop ----------------------------------------------------------
    @torch.no_grad()
    def _optimize_batch(self, molecules, context_row):
        model = self.task.model
        x, h_cat, h_int, node_mask, edge_mask = self._pad(molecules)
        bsz, n_max = node_mask.shape[0], node_mask.shape[1]

        # the reverse loop assumes a zero-CoG subspace
        x = remove_mean_with_mask(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        if not model.include_charges:
            # matches upstream, which passes torch.tensor([]) as h["integer"]
            # for the conditional checkpoints (variational_diffusion.py:1460)
            h_int = h_int[..., :0]

        x_n, h_n, _ = model.normalize(
            x, {"categorical": h_cat, "integer": h_int}, node_mask
        )
        z = torch.cat([x_n, h_n["categorical"], h_n["integer"]], dim=-1)
        z = z * node_mask

        context = None
        if context_row is not None:
            context = (
                context_row.unsqueeze(1).repeat(bsz, n_max, 1) * node_mask
            )

        # q(z_t0 | x): put the seed on the noise scale the network expects at
        # t0. Without this the reverse chain amplifies a data-scale latent by
        # 1 / alpha(t0) and the molecule explodes -- see the module docstring.
        t0 = self.noise_level
        t0_array = torch.full((bsz, 1), t0, device=z.device)
        gamma_t0 = model.gamma(t0_array)
        eps = model.sample_combined_position_feature_noise(
            bsz, n_max, node_mask
        )
        if eps.shape[-1] != z.shape[-1]:
            raise ValueError(
                f"noise width {eps.shape[-1]} != latent width {z.shape[-1]}; "
                "the checkpoint's include_charges does not match the input "
                "encoding"
            )
        z = (
            model.alpha(gamma_t0, z) * z + model.sigma(gamma_t0, z) * eps
        ) * node_mask
        assert_mean_zero_with_mask(z[..., :3], node_mask)

        # The time-axis rescaling. K, not model.T -- see the module
        # docstring. Truncated at t0: step K-1 is presented as t = t0.
        k = self.num_optimization_timesteps
        for s in reversed(range(k)):
            s_array = torch.full((bsz, 1), s * t0 / k, device=z.device)
            t_array = torch.full((bsz, 1), (s + 1) * t0 / k, device=z.device)
            z = model.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context
            )

        x_out, h_out = model.sample_p_xh_given_z0(
            z, node_mask, edge_mask, context
        )
        x_out = remove_mean_with_mask(x_out, node_mask)
        return x_out, h_out, node_mask

    def run(self) -> None:
        torch.manual_seed(self.seed)
        os.makedirs(self.output_path, exist_ok=True)

        molecules = self._load_inputs()
        logger.info(
            f"Optimizing {len(molecules)} molecule(s) from "
            f"{self.input_mols_dir} for {self.iterations} iteration(s) of "
            f"{self.num_optimization_timesteps} compressed reverse steps "
            f"from t0 = {self.noise_level} "
            f"(model T = {self.task.model.T})"
        )
        context_row = self._build_context_row()
        if context_row is None and self.target_values:
            logger.warning(
                "target_values were given but the task has no "
                "prop_dist_model; running UNCONDITIONALLY. With a "
                "conditional checkpoint this is almost certainly not what "
                "you want."
            )

        vocab = list(self.task.atom_vocab)
        for iteration in range(1, self.iterations + 1):
            next_round = []
            for start in range(0, len(molecules), self.batch_size):
                chunk = molecules[start : start + self.batch_size]
                x_out, h_out, node_mask = self._optimize_batch(
                    chunk, context_row
                )
                counts = node_mask.squeeze(-1).sum(dim=1).long()
                for i, (_, _, charges, path) in enumerate(chunk):
                    n = int(counts[i])
                    next_round.append(
                        (
                            x_out[i, :n].cpu(),
                            h_out["categorical"][i, :n].float().cpu(),
                            charges,
                            path,
                        )
                    )
            # No filtering or selection between rounds -- matches upstream
            # (mol_gen_eval_optimization_qm9.py:174-177).
            molecules = next_round

            if self.save_every_iteration or iteration == self.iterations:
                out_dir = os.path.join(
                    self.output_path, f"iteration_{iteration:02d}"
                )
                for i, (positions, one_hot, _, path) in enumerate(molecules):
                    save_xyz_file(
                        out_dir,
                        one_hot.unsqueeze(0),
                        positions.unsqueeze(0),
                        atom_decoder=vocab,
                        idxs=[i],
                        name=os.path.splitext(os.path.basename(path))[0],
                    )
                logger.info(f"iteration {iteration}: wrote {out_dir}")

        logger.info(f"Property optimization finished -> {self.output_path}")


__all__ = ["GCDMOptimizeFactory"]
