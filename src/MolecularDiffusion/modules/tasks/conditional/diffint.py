"""DiffInt task: DiffSBDD conditioned on explicit H-bond interaction nodes.

**No new model code.** DiffInt's ``equivariant_diffusion/*`` is functionally
identical to the already-ported ``modules/models/diffsbdd/``; the paper's
contribution is entirely data-level (two H-bond pseudo-atoms per protein-
ligand hydrogen bond, appended to the *pocket* node set with two extra
one-hot channels, widening ``residue_nf`` 20 -> 22). The pseudo-atoms are
conditioning and are never diffused, so :class:`DiffSBDDTask` runs them
unmodified.

What this module adds is therefore two one-method overrides plus a generator:

* :class:`ModelTaskFactory` -- ``diffusion_diffsbdd.ModelTaskFactory`` with a
  different :meth:`_size_histogram`. Upstream keys the size prior on the
  **residue** count, excluding the particles (``dataset.py:65``), and the
  released checkpoint's (107, 113) histogram is on that scale; keying it on
  ``len(pocket_coords)`` would mis-score every sample's ``log_pN``.
* :class:`DiffIntPocketGenerator` -- ``DiffSBDDPocketGenerator`` reading
  :class:`DiffIntDataset` rows, plus the novel-PDB path (``pocket_pdb`` +
  ``ref_sdf``), and passing the residue count as ``num_pocket_nodes``.

The task class itself is :class:`DiffSBDDTask`, reused verbatim: it already
reads ``pocket["size"]`` from ``num_pocket_nodes`` while ``pocket["mask"]``
spans every pocket node, which is exactly the residues/particles split
DiffInt needs.

Out of scope this pass (see the integration plan): the auxiliary interaction
loss (``loss_inter_xh`` / ``loss_inter_2`` -- commented out in the release and
never active for the shipped checkpoint), ``joint`` mode (DiffInt ships only
``pocket_conditioning`` weights and its forked ``inpaint()`` lost the
``pocket_fixed`` argument ``lightning_modules.py:967`` still passes),
inpainting, trajectories and virtual nodes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from MolecularDiffusion.data.integrations.diffint_data import (
    DiffIntDataset,
    make_item,
)
from MolecularDiffusion.data.integrations.diffint_prep import NUM_POCKET_CLASSES
from MolecularDiffusion.modules.tasks.conditional.diffsbdd import (
    INT_TYPE,
    DiffSBDDPocketGenerator,
    DiffSBDDTask,
    LigandPocketSizePrior,
)
from MolecularDiffusion.modules.tasks.conditional.diffsbdd import (
    ModelTaskFactory as DiffSBDDModelTaskFactory,
)

__all__ = [
    "DiffIntPocketGenerator",
    "DiffSBDDTask",
    "LigandPocketSizePrior",
    "ModelTaskFactory",
]


class ModelTaskFactory(DiffSBDDModelTaskFactory):
    """Hydra entry point for ``configs/tasks/diffusion_diffint.yaml``.

    Everything -- ``build()``, the EGNN/diffusion kwargs, the ``train_set``
    injection seam -- is inherited. Only the size histogram's pocket axis
    changes; see the module docstring.
    """

    def __init__(
        self,
        task_type: str = "diffusion_diffint",
        residue_nf: int = NUM_POCKET_CLASSES,
        max_n_pocket: int = 113,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task_type=task_type,
            residue_nf=residue_nf,
            max_n_pocket=max_n_pocket,
            **kwargs,
        )

    def _size_histogram(self) -> Optional[torch.Tensor]:
        """Keyed on the RESIDUE count, not the augmented pocket size."""
        if self.train_set is None:
            return None
        hist = torch.zeros(self.max_n_lig, self.max_n_pocket)
        for item in self.train_set:
            n_lig = len(item["lig_coords"])
            n_pocket = int(item["n_residues"])
            if n_lig < self.max_n_lig and n_pocket < self.max_n_pocket:
                hist[n_lig, n_pocket] += 1
        return hist


class DiffIntPocketGenerator(DiffSBDDPocketGenerator):
    """Pocket-conditioned generation behind ``interference/gen_diffint_pocket``.

    Two mutually exclusive pocket sources:

    * ``pocket_db`` + ``pocket_index`` -- a row of a converted ASE db
      (``scripts/convert_dataset.py``), read with ``center=False`` so samples
      come back in that pocket's own frame.
    * ``pocket_pdb`` + ``ref_sdf`` -- a novel protein + reference ligand pose,
      preprocessed on the fly by
      ``data.component.diffint_prep.complex_from_files``. The SDF is
      required, not optional: it is both the 8 A pocket-selection reference
      and the ligand ODDT detects the H-bonds against, since hydrogen bonds
      are protein<->ligand.

    **Do not "simplify" the novel-PDB path back onto upstream's
    ``generate_ligands()``**: ``lightning_modules.py:852-890`` zero-pads the
    20-class CA one-hot to 22 and never appends the particles, so the
    ``DD``/``AC`` columns stay all-zero and the model silently degrades to
    plain DiffSBDD with no error. This follows ``test_single.py:157-192``
    instead.
    """

    tag = "diffint"

    def __init__(
        self,
        task,
        pocket_db: Optional[str] = None,
        pocket_index: int = 0,
        pocket_pdb: Optional[str] = None,
        ref_sdf: Optional[str] = None,
        dist_cutoff: float = 8.0,
        output_path: str = "generated_diffint",
        **kwargs: Any,
    ) -> None:
        if bool(pocket_db) == bool(pocket_pdb):
            raise ValueError(
                "Set exactly one pocket source: interference.pocket_db "
                "(a converted ASE db) or interference.pocket_pdb (+ ref_sdf)."
            )
        if pocket_pdb and not ref_sdf:
            raise ValueError(
                "interference.ref_sdf is required with pocket_pdb: the "
                "reference ligand defines both the 8 A CA pocket and the "
                "protein-ligand hydrogen bonds that become the DD/AC "
                "particles."
            )
        if kwargs.get("fixed_atom_indices"):
            raise NotImplementedError(
                "inpainting is out of scope for the DiffInt port: DiffInt's "
                "forked inpaint() is broken upstream (lightning_modules.py:967 "
                "still passes a `pocket_fixed` argument the fork removed). Use "
                "the DiffSBDD task for inpainting."
            )
        # the base class demands pocket_db; satisfy it with a placeholder when
        # the novel-PDB path is in use, then overwrite.
        super().__init__(
            task,
            pocket_db=pocket_db or "<pocket_pdb>",
            pocket_index=pocket_index,
            output_path=output_path,
            **kwargs,
        )
        self.pocket_db = pocket_db
        self.pocket_pdb = pocket_pdb
        self.ref_sdf = ref_sdf
        self.dist_cutoff = dist_cutoff

    def _item(self) -> Dict[str, Any]:
        if self.pocket_db:
            return DiffIntDataset(self.pocket_db, center=False)[
                self.pocket_index
            ]

        from MolecularDiffusion.data.integrations.diffint_prep import (  # noqa: PLC0415
            complex_from_files,
        )

        raw = complex_from_files(
            self.pocket_pdb, self.ref_sdf, dist_cutoff=self.dist_cutoff
        )
        print(
            f"[diffint] built '{raw['name']}' from {self.pocket_pdb}: "
            f"{raw['n_residues']} CA residues + {raw['n_particles']} H-bond "
            "particles"
        )
        if raw["n_particles"] == 0:
            print(
                "[diffint] WARNING: ODDT found no protein-ligand hydrogen "
                "bonds, so the DD/AC channels are all zero and this run is "
                "equivalent to plain DiffSBDD. Check that the PDB carries "
                "hydrogens/valid residues and that the SDF pose is the one "
                "bound in it."
            )
        # center=False: samples come back in the input structure's frame.
        return make_item(
            raw["name"],
            raw["lig_coords"],
            raw["lig_z"],
            raw["pocket_coords"],
            raw["pocket_class"],
            center=False,
        )

    def _pocket_batch(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Tile one pocket ``n`` times.

        Overridden only for ``num_pocket_nodes``: the base class reports the
        full node count, but the size prior and ``log_pN`` are on the residue
        scale (``dataset.py:65``). ``pocket_mask`` still spans the particles.
        """
        batch = super()._pocket_batch(item, n)
        batch["num_pocket_nodes"] = torch.full(
            (n,), int(item["n_residues"]), dtype=INT_TYPE
        )
        return batch
