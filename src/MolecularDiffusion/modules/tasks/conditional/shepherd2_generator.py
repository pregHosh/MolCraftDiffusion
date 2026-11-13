"""ShEPhERD-2 generation: a thin subclass of the shared ShEPhERD seam.

This file contains **no ``run()`` loop**, and that is deliberate. The batching
loop, the ``N_x1`` range sampling, the ``P(N_x4 | N_x1)`` ``.npz`` path, the
SMARTS/ring scaffold selection and the xyz-figure rendering all already exist on
``runmodes.generate.tasks_generate.PharmacophoreConditionGenerator`` and are
inherited verbatim. Only four hooks are filled:

``__init__``          adds the v2-only and composition knobs, then defers.
``_load_reference``   returns the inherited dict plus an ``InteractionProfile``.
``_generate_batch``   maps ``task_type`` onto the vendored v2 samplers.
``_save_outputs``     writes the channels v1's writer has no slot for.

``_save_outputs`` is the hook the Generalizer added for exactly this purpose.
Before it existed, ``run()`` called the module-level ``save_shepherd_outputs``
inline, and that writer emits ``.xyz`` plus surface/ESP/pharmacophore ``.npz``
while **discarding ``x1['bonds']``** -- it also has no slot for
``x1['formal_charges']`` (v2-only) or for sampling trajectories. ShEPhERD is the
one model in the tree that *generates* bond orders rather than re-perceiving
them from geometry, so dropping them is a real loss of fidelity. The override
calls ``super()._save_outputs(...)`` first, so the v1 outputs are still written,
and then adds the rest. The structures handed to ``super()`` are **Dummy-cleaned**
(atomic number 0 dropped, with its bond entries), because those rows must not
reach any output and the inherited ``.xyz`` writer would otherwise render them
as the non-element symbol ``X`` -- see ``_save_outputs``.

Supported ``task_type`` values
------------------------------
``unconditional``, ``pharmacophore_condition``, ``shape_conditioned``,
``interaction_condition``, ``pharmacophore_inpaint`` / ``scaffold_condition``,
``pharmacophore_priority``, ``modality_hop``, ``composition``,
``composition_avoid``.

Known limitations of the RELEASED WEIGHTS (not of this port)
------------------------------------------------------------
* x2 / shape-only is not a trained channel: the checkpoint has
  ``compute_x2: False``. Shape still enters indirectly through the x3 surface.
* Property CFG is inert: no ``*prop*`` keys were released.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Sequence

import numpy as np

from MolecularDiffusion.runmodes.generate.tasks_generate import (
    PharmacophoreConditionGenerator,
)

logger = logging.getLogger(__name__)

# task_types routed to comp_inference.sampler.generate_composition
_COMPOSITION_TASKS = {"composition", "composition_avoid"}


class Shepherd2Generator(PharmacophoreConditionGenerator):
    """Generation seam for ShEPhERD-2 (``task_type: diffusion_shepherd2``)."""

    def __init__(
        self,
        task,
        task_type: str = "pharmacophore_condition",
        # --- v2 sampler knobs (upstream inference/sampler.py:50-87 defaults) ---
        early_stop_edm: float = 0.9,
        sigma_max: Optional[float] = 3.0,
        sigma_min: Optional[float] = 1e-3,
        rho: Optional[float] = 7.0,
        use_stochastic: bool = False,
        S_churn: float = 40.0,
        S_noise: float = 1.0,
        shepherd_pred: bool = True,
        condition_modalities: Any = "all",
        condition_com: Any = "origin",
        pharm_masks: Optional[Sequence[int]] = None,
        store_trajectories: bool = False,
        store_trajectories_x0: bool = False,
        # --- reference-profile extraction (interaction_profile.py:751-762) ---
        num_surf_points: int = 75,
        probe_radius: float = 0.6,
        # False, not upstream's True: tutorials 3 and 4 both pass
        # xtb_optimize=False when the reference pkl already ships xTB charges.
        # Relaxing a reference whose charges were computed for the UNrelaxed
        # geometry desynchronises the two, and the ESP is built from those
        # charges. See INTEGRATION_PLAN.md "Hyperparameter Provenance".
        xtb_profile_optimize: bool = False,
        solvent: str = "water",
        neutralize_esp: bool = False,
        # --- modality hopping: profile.subselection(...) ---
        subselect_atom_inds: Optional[Sequence[int]] = None,
        subselect_kwargs: Optional[dict] = None,
        # --- composition (comp_inference/sampler.py:583-624) ---
        reference_mols: Optional[Sequence[str]] = None,
        mol_idxs: Optional[Sequence[int]] = None,
        weights_conditions: Optional[Sequence[float]] = None,
        composition_mode: str = "default",
        align_to_idx: Optional[int] = None,
        align_condition: str = "esp",
        alignment_start_frac: float = 0.0,
        alignment_interval: int = 10,
        alignment_mode: str = "so3",
        alignment_ema_alpha: float = 1.0,
        atom_condition: Any = None,
        # --- output channels ---
        write_bonds: bool = True,
        write_sdf: bool = True,
        xtb_optimize: bool = True,
        xtb_timeout: int = 60,
        sdf_num_workers: int = 1,
        **kwargs: Any,
    ) -> None:
        # Every attribute _load_reference touches must exist BEFORE super(),
        # because the base __init__ calls self._load_reference(...) -- i.e. this
        # subclass's override -- as its last step.
        self.num_surf_points = num_surf_points
        self.probe_radius = probe_radius
        self.xtb_profile_optimize = xtb_profile_optimize
        self.solvent = solvent
        self.neutralize_esp = neutralize_esp
        self.subselect_atom_inds = list(subselect_atom_inds) if subselect_atom_inds else None
        self.subselect_kwargs = dict(subselect_kwargs or {})

        self.early_stop_edm = early_stop_edm
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.use_stochastic = use_stochastic
        self.S_churn = S_churn
        self.S_noise = S_noise
        self.shepherd_pred = shepherd_pred
        self.condition_modalities = condition_modalities
        self.condition_com = condition_com
        self.pharm_masks = list(pharm_masks) if pharm_masks else None
        self.store_trajectories = store_trajectories
        self.store_trajectories_x0 = store_trajectories_x0

        self.composition_mode = composition_mode
        self.weights_conditions = (
            np.asarray(weights_conditions, dtype=float)
            if weights_conditions is not None
            else None
        )
        self.align_to_idx = align_to_idx
        self.align_condition = align_condition
        self.alignment_start_frac = alignment_start_frac
        self.alignment_interval = alignment_interval
        self.alignment_mode = alignment_mode
        self.alignment_ema_alpha = alignment_ema_alpha
        self.atom_condition = atom_condition

        self.write_bonds = write_bonds
        self.write_sdf = write_sdf
        self.xtb_optimize = xtb_optimize
        self.xtb_timeout = xtb_timeout
        self.sdf_num_workers = sdf_num_workers

        # Composition needs several profiles, but the base __init__ loads exactly
        # one, guarded by `if task_type != "unconditional"`. Satisfy that guard
        # with the first entry (so self._ref is profile 0), then load the rest
        # below with the SAME override. This is why composition needs no second
        # core change: __init__'s signature is untouched.
        self.reference_mols = list(reference_mols) if reference_mols else None
        self.mol_idxs = list(mol_idxs) if mol_idxs else None
        if task_type in _COMPOSITION_TASKS:
            if not self.reference_mols:
                raise ValueError(
                    f"task_type='{task_type}' requires `reference_mols` "
                    "(a list of .pkl/.sdf/.mol/.pdb paths), one per condition"
                )
            if self.mol_idxs is None:
                self.mol_idxs = [0] * len(self.reference_mols)
            if len(self.mol_idxs) != len(self.reference_mols):
                raise ValueError(
                    f"mol_idxs has {len(self.mol_idxs)} entries but reference_mols "
                    f"has {len(self.reference_mols)}"
                )
            if self.weights_conditions is None:
                raise ValueError(
                    f"task_type='{task_type}' requires `weights_conditions`, one "
                    "weight per entry in reference_mols"
                )
            kwargs["reference_mol"] = self.reference_mols[0]
            kwargs["mol_idx"] = self.mol_idxs[0]

        super().__init__(task=task, task_type=task_type, **kwargs)

        self._profiles: List[Any] = []
        if task_type in _COMPOSITION_TASKS:
            self._profiles = [
                self._load_reference(path, idx)["profile"]
                for path, idx in zip(self.reference_mols, self.mol_idxs)
            ]
            if self.align_to_idx is not None:
                self._prealign_profiles()
            logger.info(
                "Shepherd2Generator: %d interaction profiles loaded for '%s' "
                "(mode=%s, weights=%s)",
                len(self._profiles), task_type, self.composition_mode,
                self.weights_conditions,
            )

    # ------------------------------------------------------------------ setup
    def _prealign_profiles(self) -> None:
        """Pre-align every profile onto ``align_to_idx`` before composing.

        Upstream's ``comp_inference.utils.align_two_ref_mol`` works on
        ``(molblock, charges)`` pairs, so the profiles are re-extracted from the
        aligned molecules rather than transformed in place.
        """
        from rdkit import Chem

        from MolecularDiffusion.modules.models.shepherd2.comp_inference.utils import (
            align_two_ref_mol,
        )

        anchor = self.align_to_idx
        pairs = [
            (Chem.MolToMolBlock(p.mol), p.partial_charges) for p in self._profiles
        ]
        for i in range(len(self._profiles)):
            if i == anchor:
                continue
            aligned = align_two_ref_mol(
                molblocks_and_charges=pairs,
                mol_to_align_i=i,
                ref_mol_i=anchor,
                condition=self.align_condition,
                num_surf_points=self.num_surf_points,
                probe_radius=self.probe_radius,
            )
            self._profiles[i] = self._profile_from_mol(
                aligned, self._profiles[i].partial_charges
            )
        logger.info("Pre-aligned %d profiles onto index %d (condition=%s)",
                    len(self._profiles) - 1, anchor, self.align_condition)

    def _profile_from_mol(self, mol, charges, condition_atom_inds=None):
        """Call upstream's own profile constructor. Nothing is re-derived here."""
        from MolecularDiffusion.modules.models.shepherd2.interaction_profile import (
            extract_interaction_profile,
        )

        return extract_interaction_profile(
            mol,
            xtb_optimize=self.xtb_profile_optimize,
            partial_charges=charges,
            center=True,
            solvent=self.solvent,
            condition_atom_inds=condition_atom_inds,
            num_surf_points=self.num_surf_points,
            probe_radius=self.probe_radius,
            neutralize_esp=self.neutralize_esp,
        )

    # --------------------------------------------------------------- hook 1/4
    def _load_reference(self, path: str, mol_idx: int) -> dict:
        """Load a reference and build its ``InteractionProfile``.

        Returns the same keys the base class returns (so anything inherited that
        reads ``self._ref`` keeps working) plus ``profile``, which is what the v2
        samplers actually consume.

        Adds a ``.mol`` and a ``.pdb`` branch to the base class's ``.pkl``/``.sdf``
        -- the modality-hopping fixtures are peptide PDBs and the
        fragment-merging fixtures are ``.mol`` files.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        path = str(path)
        charges = None

        if path.endswith(".pkl") or path.endswith(".pickle"):
            import pickle

            with open(path, "rb") as fh:
                data = pickle.load(fh)
            entry = data[mol_idx]
            mol = Chem.MolFromMolBlock(entry[0], removeHs=False)
            charges = np.asarray(entry[1], dtype=np.float32)
        elif path.endswith(".sdf"):
            mol = Chem.SDMolSupplier(path, removeHs=False)[mol_idx]
        elif path.endswith(".mol"):
            mol = Chem.MolFromMolFile(path, removeHs=False)
        elif path.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(path, removeHs=False)
        else:
            raise ValueError(
                f"Unsupported reference format (want .pkl/.pickle/.sdf/.mol/.pdb): {path}"
            )

        if mol is None:
            raise ValueError(f"Failed to parse molecule {mol_idx} from {path}")

        # No shipped charges (.sdf/.mol/.pdb): pass partial_charges=None and let
        # upstream's extractor compute them with xTB, rather than substituting
        # Gasteiger -- a different charge model from the one the ESP channel was
        # trained on, and the ESP is built directly from these numbers.
        if charges is None:
            logger.info(
                "%s ships no partial charges; extract_interaction_profile will "
                "compute them with xTB.", os.path.basename(path),
            )
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
                AllChem.MMFFOptimizeMolecule(mol)

        condition_atom_inds = None
        if self.task_type in ("pharmacophore_inpaint", "scaffold_condition"):
            condition_atom_inds, _ = self._get_scaffold_inds(mol)

        profile = self._profile_from_mol(mol, charges, condition_atom_inds)

        if profile is None:
            raise RuntimeError(
                f"extract_interaction_profile returned None for {path}[{mol_idx}] "
                "(xTB or surface extraction failed)"
            )

        if self.task_type == "modality_hop" and self.subselect_atom_inds:
            profile = profile.subselection(
                atom_inds=list(self.subselect_atom_inds), **self.subselect_kwargs
            )
            logger.info("modality_hop: subselected %d atoms -> %d pharmacophores",
                        len(self.subselect_atom_inds), profile.n_pharms)

        logger.info(
            "Reference [%d] from %s: %d atoms, %d pharmacophores, "
            "%d surface points",
            mol_idx, os.path.basename(path), profile.n_atoms, profile.n_pharms,
            0 if profile.surface is None else len(profile.surface),
        )

        return dict(
            profile=profile,
            mol=profile.mol if profile.mol is not None else mol,
            charges=profile.partial_charges,
            surface=profile.surface,
            electrostatics=profile.electrostatics,
            center_of_mass=profile.com_before_centering,
            pharm_types=profile.pharm_types,
            pharm_pos=profile.pharm_positions,
            pharm_direction=profile.pharm_directions,
        )

    # --------------------------------------------------------------- hook 2/4
    def _generate_batch(self, batch_size: int) -> list:
        """Map ``task_type`` onto the vendored v2 samplers.

        Unlike the base class this does NOT mutate
        ``params['explicit_diffusion_variables']`` from the ``compute_x*``
        toggles: ShEPhERD-2's ``Model`` builds its per-modality submodules from
        that list at construction time, so narrowing it after the fact would
        leave the forward reaching for branches whose inputs were never built.
        Modality selection is expressed through ``condition_modalities`` instead,
        which is the knob the v2 sampler actually reads.
        """
        task_type = self.task_type
        N_x1 = self._sample_N_x1()

        common = dict(
            batch_size=batch_size,
            num_steps=self.num_steps,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            rho=self.rho,
            use_stochastic=self.use_stochastic,
            S_churn=self.S_churn,
            S_noise=self.S_noise,
            shepherd_pred=self.shepherd_pred,
            verbose=self.verbose,
        )

        if task_type in _COMPOSITION_TASKS:
            # "default" prepends an implicit unconditional component weighted
            # 1 - sum(weights) (logical AND). "conditional" uses only the given
            # conditions and allows negative weights (AND-NOT / avoid).
            mode = (
                "conditional" if task_type == "composition_avoid"
                else self.composition_mode
            )
            N_x4 = self.N_x4 or max(p.n_pharms for p in self._profiles)
            return self.task.sample_composition(
                N_x1=N_x1,
                N_x4=N_x4,
                profiles=self._profiles,
                weights_conditions=self.weights_conditions,
                composition_mode=mode,
                atom_condition=self.atom_condition,
                condition_modalities=self.condition_modalities,
                condition_center_of_mass=self.condition_com,
                early_stop_edm=int(0.9 * self.num_steps),
                store_trajectories=self.store_trajectories,
                alignment_start_frac=self.alignment_start_frac,
                alignment_interval=self.alignment_interval,
                alignment_mode=self.alignment_mode,
                alignment_ema_alpha=self.alignment_ema_alpha,
                **common,
            )

        kw = dict(
            N_x1=N_x1,
            early_stop_edm=self.early_stop_edm,
            store_trajectories=self.store_trajectories,
            store_trajectories_x0=self.store_trajectories_x0,
            **common,
        )

        if task_type == "unconditional":
            kw["N_x4"] = (
                self._sample_N_x4(N_x1) if self._nx4_dist is not None else self.N_x4
            )
            # condition=None is what makes it unconditional (sampler.py:381).
            return self.task.sample(**kw)

        profile = self._ref["profile"]
        kw["condition"] = profile
        kw["condition_center_of_mass"] = self.condition_com
        kw["N_x4"] = profile.n_pharms if profile.n_pharms else self.N_x4

        if task_type == "pharmacophore_condition":
            kw["condition_modalities"] = "pharm"
        elif task_type == "shape_conditioned":
            kw["condition_modalities"] = "esp"
        elif task_type in ("interaction_condition", "modality_hop"):
            kw["condition_modalities"] = self.condition_modalities
        elif task_type in ("pharmacophore_inpaint", "scaffold_condition"):
            kw["condition_modalities"] = self.condition_modalities
            kw["scaffold_conditioning"] = True
        elif task_type == "pharmacophore_priority":
            kw["condition_modalities"] = self.condition_modalities
            kw["pharmacophore_conditioning"] = True
            if self.pharm_masks:
                kw["pharmacophore_prioritization"] = list(self.pharm_masks)
        else:
            raise ValueError(
                f"Unknown task_type for Shepherd2Generator: '{task_type}'. "
                "Supported: unconditional, pharmacophore_condition, "
                "shape_conditioned, interaction_condition, pharmacophore_inpaint, "
                "scaffold_condition, pharmacophore_priority, modality_hop, "
                "composition, composition_avoid."
            )

        return self.task.sample(**kw)

    # --------------------------------------------------------------- hook 3/4
    @staticmethod
    def _drop_dummies(atoms: np.ndarray, positions: np.ndarray, bonds):
        """Drop ``Dummy`` atoms and the bond entries that referenced them.

        The ``Dummy`` atom type decodes to atomic number **0**
        (``inference/steps.py:15-22``); those rows must not reach any output.

        Upstream's ``_clean_dummy_atom_arrays`` is used for the atoms and
        positions, but it takes only those two arrays -- it has no bond
        argument -- so the bond entries are filtered here with the *identical*
        predicate (``atomic_numbers != 0``). ``bonds`` is the flat
        ``N*(N-1)/2`` upper-triangular pair list built by
        ``np.triu(1 - I).nonzero()`` (``datasets.py:270-272``,
        ``inference/initialization.py:71-72``), i.e. row-major
        ``(0,1),(0,2),...,(1,2),...`` -- exactly ``np.triu_indices(N, k=1)``.
        """
        from shepherd_score.evaluations.evaluate.evals import _clean_dummy_atom_arrays

        atoms = np.asarray(atoms)
        n = len(atoms)
        keep = atoms != 0
        clean_atoms, clean_pos = _clean_dummy_atom_arrays(atoms, np.asarray(positions))

        clean_bonds = None
        if bonds is not None:
            bonds = np.asarray(bonds)
            rows, cols = np.triu_indices(n, k=1)
            if len(rows) != len(bonds):
                logger.warning(
                    "bond array length %d != N*(N-1)/2 = %d for N=%d; leaving "
                    "bonds unfiltered", len(bonds), len(rows), n,
                )
                clean_bonds = bonds
            else:
                clean_bonds = bonds[keep[rows] & keep[cols]]
        return clean_atoms, clean_pos, clean_bonds

    @staticmethod
    def _dummy_free_copy(structures: list) -> list:
        """Return ``structures`` with x1's ``Dummy`` rows removed.

        Copies rather than mutates: ``_write_sdf`` hands the batch to upstream's
        ``mol_charges_from_samples``, which does its own ``Dummy`` cleaning, and
        ``_write_bonds_and_charges`` filters with ``_drop_dummies`` itself. Both
        must keep seeing the sampler's own arrays, so nothing in place is
        touched. ``GeneratedSample`` is a plain ``dict`` subclass with no custom
        ``__init__``, so ``type(s)(s)`` preserves the class and its properties.

        Only x1 is cleaned. x2/x3/x4 are the surface, ESP and pharmacophore
        modalities, which carry no atom rows and so no atomic number 0; and the
        inherited writer reads nothing from x1 but ``atoms`` and ``positions``
        (``utils/geom_utils.py:514-523``).
        """
        cleaned = []
        for s in structures:
            x1 = s.get("x1") or {}
            if "atoms" not in x1 or "positions" not in x1:
                cleaned.append(s)
                continue
            raw_atoms = np.asarray(x1["atoms"])
            atoms, pos, bonds = Shepherd2Generator._drop_dummies(
                raw_atoms, x1["positions"], x1.get("bonds")
            )
            new_x1 = dict(x1)
            new_x1["atoms"] = atoms
            new_x1["positions"] = pos
            if x1.get("bonds") is not None:
                new_x1["bonds"] = bonds
            fc = x1.get("formal_charges")
            if fc is not None:
                new_x1["formal_charges"] = np.asarray(fc)[raw_atoms != 0]
            new_s = type(s)(s)
            new_s["x1"] = new_x1
            cleaned.append(new_s)
        return cleaned

    def _save_outputs(self, structures: list, idx_offset: int) -> None:
        """Write a batch: the inherited outputs, then v2's extra channels.

        ``super()`` runs first, and it receives **Dummy-cleaned** structures.
        The ``Dummy`` atom type decodes to atomic number 0
        (``inference/steps.py:15-22``), and the governing requirement is that
        those rows *and their bond entries* are dropped before **any** output is
        written -- so they must not reach the inherited ``.xyz`` writer either.
        That writer maps an unknown Z through ``ase.data.chemical_symbols``,
        where ``Z=0`` renders as the element symbol ``X``
        (``utils/geom_utils.py:523``), which is not a real atom: it was polluting
        the coordinate files, and because a virtual node sits wherever the model
        put it, it also produced spurious sub-angstrom "clashes" in the quality
        metrics computed over those files.

        This is deliberately **not** byte-equivalent to what ShEPhERD-v1's
        writer would emit for the same raw structures, and that is the point: v1
        has no ``Dummy`` type, so there was never a v1 behaviour to preserve
        here. Cleaning also makes the three output channels consistent -- the
        ``.sdf`` and ``_bonds.npz`` paths already dropped these rows, and the
        ``.xyz`` was the lone exception.

        The surface/ESP/pharmacophore ``.npz`` files are unaffected: they come
        from x2/x3/x4, which ``_dummy_free_copy`` does not touch.
        """
        super()._save_outputs(self._dummy_free_copy(structures), idx_offset)

        if self.write_bonds:
            self._write_bonds_and_charges(structures, idx_offset)
        if self.write_sdf:
            self._write_sdf(structures, idx_offset)
        if self.store_trajectories or self.store_trajectories_x0:
            self._write_trajectories(structures, idx_offset)

    def _write_bonds_and_charges(self, structures: list, idx_offset: int) -> None:
        """Emit the generated bond orders and formal charges v1's writer drops."""
        for i, s in enumerate(structures):
            x1 = s["x1"]
            atoms, pos, bonds = self._drop_dummies(
                x1["atoms"], x1["positions"], x1.get("bonds")
            )
            n = len(atoms)
            rows, cols = np.triu_indices(n, k=1)
            payload = dict(
                atoms=atoms,
                positions=pos,
                bond_index=np.stack([rows, cols], axis=0) if n > 1 else np.zeros((2, 0), int),
                # canonical classes: 0=none 1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC
                bond_type=bonds if bonds is not None else np.zeros(0, int),
            )
            fc = x1.get("formal_charges")
            if fc is not None:
                # raw signed charges from the 5-class lookup [0, 1, 2, -1, -2]
                payload["formal_charges"] = np.asarray(fc)[np.asarray(x1["atoms"]) != 0]
            np.savez_compressed(
                os.path.join(self.output_path, f"mol_{idx_offset + i:04d}_bonds.npz"),
                **payload,
            )

    def _write_sdf(self, structures: list, idx_offset: int) -> None:
        """Build real SDF molecules via upstream's xTB-backed extractor."""
        from rdkit import Chem

        from MolecularDiffusion.modules.models.shepherd2.extract import (
            mol_charges_from_samples,
        )

        try:
            results = mol_charges_from_samples(
                structures,
                num_workers=self.sdf_num_workers,
                solvent=self.solvent,
                xtb_optimize=self.xtb_optimize,
                xtb_timeout=self.xtb_timeout,
                verbose=False,
            )
        except Exception as exc:  # xtb is external; never lose the batch over it
            logger.warning("SDF extraction failed for this batch: %s", exc)
            return

        n_ok = 0
        for i, (mol, charges) in enumerate(results):
            if mol is None:
                continue
            out = os.path.join(self.output_path, f"mol_{idx_offset + i:04d}.sdf")
            if charges is not None:
                mol.SetProp(
                    "xtb_partial_charges",
                    " ".join(f"{c:.6f}" for c in np.asarray(charges).ravel()),
                )
            writer = Chem.SDWriter(out)
            writer.write(mol)
            writer.close()
            n_ok += 1
        logger.info("Wrote %d/%d SDF molecules for batch at offset %d",
                    n_ok, len(results), idx_offset)

    def _write_trajectories(self, structures: list, idx_offset: int) -> None:
        """Dump per-frame trajectories.

        These need no hook of their own: ``edm_sampler.py:738-744`` attaches each
        trajectory onto its own ``GeneratedSample`` before returning, so they
        arrive here inside ``structures``.
        """
        for i, s in enumerate(structures):
            for key in ("trajectories", "trajectories_x0"):
                frames = s.get(key)
                if not frames:
                    continue
                flat = {}
                for f_i, frame in enumerate(frames):
                    for mod, block in frame.items():
                        for name, arr in block.items():
                            if arr is not None:
                                flat[f"{f_i:04d}_{mod}_{name}"] = np.asarray(arr)
                np.savez_compressed(
                    os.path.join(
                        self.output_path, f"mol_{idx_offset + i:04d}_{key}.npz"
                    ),
                    **flat,
                )
