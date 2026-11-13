"""Drug-likeness descriptors for generated molecules.

Backs ``MolCraftDiff analyze metrics --metrics druglike``. Everything here is
computed from the molecule alone -- no reference molecule, no receptor -- which
is why it is a separate set from :mod:`similarity3d`.

Columns follow what the pocket-conditioned literature reports:

* RDKit descriptors -- QED, SA, LogP, fsp3, MW, HBD, HBA;
* ``lipinski`` -- how many of the five Lipinski rules a molecule obeys
  (``others/targetdiff/utils/evaluation/scoring_func.py:obey_lipinski``);
* ``pains_pass`` -- free of PAINS-A substructures;
* ``ring_filter_pass`` -- no ring larger than 6 that is not aromatic-fused,
  the 2D ring sanity filter from DiffLinker;
* ring statistics -- counts plus which ring sizes are present;
* ``rdkit_rmsd_*`` -- distance from the generated pose to UFF-optimised RDKit
  conformers. Expensive (embeds ``n_conf`` conformers per molecule), so it is
  opt-in via ``--rdkit-rmsd``.
"""

from __future__ import annotations

RING_SIZES = range(3, 10)


def _scscore(mol):
    """SCScore of an already-perceived molecule; NaN when the model is absent.

    Reuses the mol built for the other columns instead of re-perceiving the
    xyz. The model lives in ``assets/scscore`` (``get_scscore_model`` warns
    once when it is missing). Unlike SA_score, explicit H atoms are kept
    (matches the established procedure in
    ``compute_scscore_from_pdb.py``/``synthesis_and_novelty_metrics_procedure.md``:
    ``MolFromPDBFile(..., removeHs=False)`` -- stripping Hs here would silently
    diverge from previously published SCScore numbers for this project).
    """
    from rdkit import Chem  # noqa: PLC0415

    from MolecularDiffusion.runmodes.data.preparation import (  # noqa: PLC0415
        get_scscore_model,
    )

    model = get_scscore_model()
    if model is None:
        return float("nan")
    return float(model.get_score_from_smi(Chem.MolToSmiles(mol))[1])


def _lipinski(mol):
    """Number of Lipinski rules obeyed, 0-5."""
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors  # noqa: PLC0415

    logp = Crippen.MolLogP(mol)
    rules = [
        Descriptors.ExactMolWt(mol) < 500,
        Lipinski.NumHDonors(mol) <= 5,
        Lipinski.NumHAcceptors(mol) <= 10,
        -2 <= logp <= 5,
        rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10,
    ]
    return int(sum(rules))


def _pains_pass(mol):
    """True when the molecule matches no PAINS-A alert."""
    from rdkit.Chem import FilterCatalog  # noqa: PLC0415

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog = FilterCatalog.FilterCatalog(params)
    return catalog.GetFirstMatch(mol) is None


def _ring_filter_pass(mol):
    """DiffLinker's 2D ring filter: no non-aromatic ring bigger than 6."""
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) > 6 and not all(
            mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring
        ):
            return False
    return True


def _ring_stats(mol):
    """Ring counts and which of the 3..9-membered sizes are present."""
    from rdkit.Chem import rdMolDescriptors  # noqa: PLC0415

    sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
    stats = {
        "n_rings": len(sizes),
        "n_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "n_aliphatic_rings": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
    }
    # per-molecule booleans; the set-level number the papers print is the
    # fraction of molecules containing each ring size, i.e. the column mean
    for size in RING_SIZES:
        stats[f"ring_size_{size}"] = size in sizes
    return stats


def rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """RMSD from the generated pose to ``n_conf`` UFF-optimised conformers.

    Mirrors ``others/targetdiff/utils/evaluation/scoring_func.py:get_rdkit_rmsd``.
    Returns ``(min, median, max)``, or ``(None, None, None)`` when embedding
    fails.
    """
    from copy import deepcopy  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    try:
        probe = deepcopy(mol)
        Chem.SanitizeMol(probe)
        mol3d = Chem.AddHs(probe)
        conf_ids = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
        values = []
        for conf_id in conf_ids:
            AllChem.UFFOptimizeMolecule(mol3d, confId=conf_id)
            values.append(Chem.rdMolAlign.GetBestRMS(probe, mol3d, refId=conf_id))
        if not values:
            return None, None, None
        return float(np.min(values)), float(np.median(values)), float(np.max(values))
    except Exception:  # noqa: BLE001 -- an unembeddable molecule is data
        return None, None, None


def compute(mol, with_rdkit_rmsd=False, n_conf=20):
    """All drug-likeness columns for one molecule.

    ``mol`` must already be sanitized. Returns a flat dict ready for a
    DataFrame row.
    """
    from MolecularDiffusion.utils.metrics.geom_metrics import compute_drug_likeness  # noqa: PLC0415

    row = dict(compute_drug_likeness(mol))  # QED, SA_score, LogP, fsp3, MW, HBD, HBA
    row["SCScore"] = _scscore(mol)
    row["lipinski"] = _lipinski(mol)
    row["pains_pass"] = _pains_pass(mol)
    row["ring_filter_pass"] = _ring_filter_pass(mol)
    row.update(_ring_stats(mol))
    if with_rdkit_rmsd:
        low, mid, high = rdkit_rmsd(mol, n_conf=n_conf)
        row["rdkit_rmsd_min"] = low
        row["rdkit_rmsd_median"] = mid
        row["rdkit_rmsd_max"] = high
    return row
