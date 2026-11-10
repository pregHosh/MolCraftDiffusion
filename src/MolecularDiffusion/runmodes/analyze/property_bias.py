"""Check whether a docking score is really measuring molecule size.

Backs ``MolCraftDiff analyze bias``. A docking score such as Vina's rewards
big, oily molecules because they touch more of the pocket, whether or not they
bind. If the top-ranked molecules are simply the largest ones, the ranking
carries little information about the target. This reads the CSVs written by
``analyze dock`` and reports, for heavy atoms (non-hydrogen atoms, a size
measure), logP (how oily a molecule is, RDKit Crippen estimate) and formal
charge (the sum of the +/- charges in the SMILES):

* the correlation of score with each property (Spearman is rank-based and
  robust to outliers, Pearson is the plain linear one), with a bootstrap 95%
  interval;
* how the top-k molecules differ from the whole pool, with a permutation
  p-value (how often a random k-subset looks at least as extreme);
* optionally, the same comparison for known hits (``--labels``).

It is a diagnostic: it logs warnings and never fails the run.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

    import pandas as pd

PROPERTIES = ("heavy_atoms", "logp", "charge")

#: Below this many scored molecules a correlation is not worth reporting.
MIN_MOLECULES = 20

N_BOOT = 2000
N_PERM = 20000
SEED = 0

#: More than this share of charge-0 molecules makes charge uninformative.
CHARGE_CONSTANT_SHARE = 0.9
#: Charged-share gap (fraction, i.e. 20 percentage points) that triggers a
#: warning.
CHARGE_SHARE_GAP = 0.2
PERM_P_FLAG = 0.01
RHO_FLAG = 0.5

_LABELS = {
    "heavy_atoms": "heavy atoms",
    "logp": "logP",
    "charge": "charge",
}


def load_scores(
    paths: Sequence[str], score_col: str = "vina_dock"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read dock CSVs into one ``name, smiles, score`` frame, plus row counts.

    Drops rows with an empty or non-numeric score (a docking error), then
    keeps the lowest score for a repeated ``name``. Both counts are returned
    so the caller can log them.
    """
    import pandas as pd  # noqa: PLC0415

    frame = pd.concat(
        [pd.read_csv(p) for p in paths], ignore_index=True
    )
    for col in ("name", "smiles", score_col):
        if col not in frame.columns:
            msg = f"column {col!r} not found in {list(paths)}"
            raise ValueError(msg)
    frame = frame[["name", "smiles", score_col]].rename(
        columns={score_col: "score"}
    )
    frame["name"] = frame["name"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    n_rows = len(frame)
    frame = frame[np.isfinite(frame["score"])]
    n_error = n_rows - len(frame)
    frame = frame.sort_values("score", kind="stable")
    n_scored = len(frame)
    frame = frame.drop_duplicates("name", keep="first")
    counts = {
        "rows": n_rows,
        "errors": n_error,
        "duplicates": n_scored - len(frame),
        "scored": len(frame),
    }
    return frame.reset_index(drop=True), counts


def molecule_properties(
    smiles: str, *, largest_fragment: bool = True
) -> dict[str, float] | None:
    """Heavy atoms, logP and formal charge of one SMILES, or ``None``.

    With ``largest_fragment`` a counter-ion (a separate salt piece after a
    dot, such as ``[Na+]``) is ignored: the largest piece by heavy atoms is
    measured.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import Crippen  # noqa: PLC0415

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    if largest_fragment:
        mol = max(
            Chem.GetMolFrags(mol, asMols=True),
            key=lambda m: m.GetNumHeavyAtoms(),
        )
    return {
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "logp": float(Crippen.MolLogP(mol)),
        "charge": float(Chem.GetFormalCharge(mol)),
    }


def _with_properties(
    scores: pd.DataFrame, *, largest_fragment: bool = True
) -> tuple[pd.DataFrame, int]:
    """Add the property columns; drop unparseable SMILES and count them."""
    import pandas as pd  # noqa: PLC0415

    props = [
        molecule_properties(s, largest_fragment=largest_fragment)
        for s in scores["smiles"]
    ]
    keep = [p is not None for p in props]
    frame = scores[keep].reset_index(drop=True)
    table = pd.DataFrame([p for p in props if p is not None])
    return pd.concat([frame, table], axis=1), len(keep) - sum(keep)


def score_correlations(
    props: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED
) -> pd.DataFrame:
    """Spearman and Pearson of score against each property, with 95% CIs.

    One row per property. Correlations are NaN where a property is constant
    (the caller writes those as blanks). The bootstrap resamples molecules
    with replacement and reports the 2.5th and 97.5th percentiles.
    """
    from scipy.stats import rankdata  # noqa: PLC0415

    score = props["score"].to_numpy(float)
    n = len(score)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rows = []
    for name in PROPERTIES:
        values = props[name].to_numpy(float)
        row: dict[str, Any] = {"property": name, "n": n}
        for kind in ("pearson", "spearman"):
            if kind == "spearman":
                x, y = rankdata(score), rankdata(values)
            else:
                x, y = score, values
            point = _corr_pairs(x[None], y[None])[0]
            row[kind] = float(point)
            if math.isnan(point):
                row[f"{kind}_lo"] = row[f"{kind}_hi"] = float("nan")
                continue
            if kind == "spearman":
                # ranks are recomputed inside each resample
                boot = _corr_rows_ranked(score[idx], values[idx])
            else:
                boot = _corr_pairs(score[idx], values[idx])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            row[f"{kind}_lo"], row[f"{kind}_hi"] = float(lo), float(hi)
        rows.append(row)
    import pandas as pd  # noqa: PLC0415

    return pd.DataFrame(rows)


def _corr_pairs(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two ``(B, n)`` matrices."""
    xc = xs - xs.mean(axis=1, keepdims=True)
    yc = ys - ys.mean(axis=1, keepdims=True)
    den = np.sqrt((xc**2).sum(axis=1) * (yc**2).sum(axis=1))
    # a column with no spread has no correlation; test the data, not the
    # float arithmetic, which can leave a tiny nonzero residue
    flat = (np.ptp(xs, axis=1) == 0) | (np.ptp(ys, axis=1) == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(~flat & (den > 0), (xc * yc).sum(axis=1) / den, np.nan)


def _corr_rows_ranked(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Row-wise Spearman correlation of two ``(B, n)`` matrices."""
    from scipy.stats import rankdata  # noqa: PLC0415

    return _corr_pairs(rankdata(xs, axis=1), rankdata(ys, axis=1))


def _perm_p(
    values: np.ndarray,
    k: int,
    observed: float,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> float:
    """One-sided p that a random k-subset has a mean at or above ``observed``.

    ``p = (1 + count at or above observed) / (1 + draws)``. The columns of
    ``values`` (``(n, P)``) share the same random subsets.
    """
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    count = np.zeros(values.shape[1], dtype=int)
    done = 0
    while done < n_perm:
        chunk = min(1000, n_perm - done)
        order = np.argpartition(rng.random((chunk, n)), k - 1, axis=1)[:, :k]
        means = values[order].mean(axis=1)
        count += (means >= observed - 1e-9).sum(axis=0)
        done += chunk
    return (1 + count) / (1 + n_perm)


def subset_vs_pool(
    props: pd.DataFrame,
    picked: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Compare the rows at positions ``picked`` with the whole pool.

    One dict per property (plus ``nonzero_charge``, the charged share):
    subset mean, pool mean, pool median, pool SD, difference and one-sided
    permutation p (is the subset higher than chance?).
    """
    cols = [*PROPERTIES]
    values = props[cols].to_numpy(float)
    values = np.column_stack([values, (values[:, 2] != 0).astype(float)])
    names = [*cols, "nonzero_charge"]
    k = len(picked)
    observed = values[picked].mean(axis=0)
    p = _perm_p(values, k, observed, n_perm, seed)
    return [
        {
            "property": name,
            "k": k,
            "subset_mean": float(observed[i]),
            "pool_mean": float(values[:, i].mean()),
            "pool_median": float(np.median(values[:, i])),
            "pool_sd": float(values[:, i].std(ddof=1)),
            "diff": float(observed[i] - values[:, i].mean()),
            "p_perm": float(p[i]),
        }
        for i, name in enumerate(names)
    ]


def topk_vs_pool(
    props: pd.DataFrame,
    k: int,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Compare the k best-scored molecules (lowest score) with the pool.

    Ties in score are broken by a seeded random order.
    """
    rng = np.random.default_rng(seed)
    score = props["score"].to_numpy(float)
    order = np.lexsort((rng.random(len(score)), score))
    return subset_vs_pool(props, order[:k], n_perm, seed)


def bias_warnings(
    corr: pd.DataFrame | None,
    topk: list[dict[str, Any]],
    sd_flag: float = 0.5,
) -> list[str]:
    """Plain-English warnings from the correlation and top-k tables.

    ``topk`` is the output of ``topk_vs_pool`` for one k; ``corr`` may be
    ``None`` to skip the correlation lines. Empty list means nothing was
    flagged.
    """
    out = []
    k = topk[0]["k"] if topk else 0
    for row in topk:
        name = row["property"]
        if name in ("heavy_atoms", "logp"):
            big = row["diff"] > sd_flag * row["pool_sd"]
            if big and row["p_perm"] < PERM_P_FLAG:
                word, pick = (
                    ("larger", "size") if name == "heavy_atoms"
                    else ("oilier", "logP")
                )
                out.append(
                    f"WARNING: top {k} are {word} than the pool (mean "
                    f"{row['subset_mean']:.1f} vs median "
                    f"{row['pool_median']:.1f} {_LABELS[name]}). "
                    f"The score is picking {pick}."
                )
        elif name == "nonzero_charge":
            gap = row["subset_mean"] - row["pool_mean"]
            if abs(gap) > CHARGE_SHARE_GAP:
                out.append(
                    f"WARNING: {row['subset_mean']:.0%} of the top {k} are "
                    f"charged against {row['pool_mean']:.0%} of the pool."
                )
    rows = corr.iterrows() if corr is not None else ()
    for _, row in rows:
        rho = row["spearman"]
        if row["property"] != "charge" and abs(rho) > RHO_FLAG:
            out.append(
                f"WARNING: |Spearman| of score vs {_LABELS[row['property']]} "
                f"is {abs(rho):.2f} (above {RHO_FLAG})."
            )
    return out


def _default_ks(n: int) -> list[int]:
    return sorted({10, max(10, math.ceil(0.01 * n))})


def runner(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    """Read dock CSVs, report score-vs-property bias, optionally write CSVs."""
    import pandas as pd  # noqa: PLC0415

    log = logging.getLogger(__name__)
    scores, counts = load_scores(
        args.inputs, getattr(args, "score_col", "vina_dock")
    )
    props, n_bad = _with_properties(scores)
    n = len(props)
    log.info(
        "Rows: %d read, %d dropped (no score), %d duplicate names "
        "(best kept), %d unparseable SMILES; %d molecules scored",
        counts["rows"], counts["errors"], counts["duplicates"], n_bad, n,
    )
    if n < MIN_MOLECULES:
        log.warning(
            "Only %d scored molecules: too few for a correlation "
            "(need %d); skipping the tests.", n, MIN_MOLECULES,
        )
        return

    corr = score_correlations(props)
    for _, row in corr.iterrows():
        log.info(
            "score vs %-11s Spearman %+.3f [%+.3f, %+.3f]  "
            "Pearson %+.3f [%+.3f, %+.3f]",
            _LABELS[row["property"]],
            row["spearman"], row["spearman_lo"], row["spearman_hi"],
            row["pearson"], row["pearson_lo"], row["pearson_hi"],
        )
    if (props["charge"] == 0).mean() > CHARGE_CONSTANT_SHARE:
        log.info("charge is nearly constant, correlation uninformative")

    sd_flag = getattr(args, "sd_flag", 0.5)
    tops = getattr(args, "top", None) or _default_ks(n)
    topk_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for k in sorted(set(tops)):
        if k > n // 2:
            log.warning("top %d skipped: above half the pool (%d)", k, n)
            continue
        rows = topk_vs_pool(props, k)
        topk_rows += [{"subset": f"top{k}", **r} for r in rows]
        for r in rows:
            log.info(
                "top %-4d %-14s mean %8.2f  pool mean %8.2f median %8.2f"
                "  diff %+7.2f  p %.4f",
                k, r["property"], r["subset_mean"], r["pool_mean"],
                r["pool_median"], r["diff"], r["p_perm"],
            )
        warnings += bias_warnings(None, rows, sd_flag)

    labels_path = getattr(args, "labels", None)
    if labels_path:
        master = pd.read_csv(labels_path)
        id_col, label_col = args.id_col, args.label_col
        for col in (id_col, label_col):
            if col not in master.columns:
                msg = f"column {col!r} not found in {labels_path}"
                raise ValueError(msg)
        hit_ids = set(
            master.loc[
                pd.to_numeric(master[label_col], errors="coerce") > 0,
                id_col,
            ].astype(str)
        )
        picked = np.flatnonzero(props["name"].isin(hit_ids).to_numpy())
        log.info(
            "Labels: %d hits in %s, %d of them in the scored pool",
            len(hit_ids), labels_path, len(picked),
        )
        if 0 < len(picked) <= n // 2:
            rows = subset_vs_pool(props, picked)
            topk_rows += [{"subset": "hits", **r} for r in rows]
            for r in rows:
                log.info(
                    "hits %-4d %-14s mean %8.2f  pool mean %8.2f median "
                    "%8.2f  diff %+7.2f  p %.4f",
                    len(picked), r["property"], r["subset_mean"],
                    r["pool_mean"], r["pool_median"], r["diff"],
                    r["p_perm"],
                )
        else:
            log.warning("hits section skipped: no hits, or hits above half "
                        "the pool")

    warnings += bias_warnings(corr, [], sd_flag)
    for line in dict.fromkeys(warnings):
        log.warning(line)
    if not warnings:
        log.info("No property bias flagged.")

    out_dir = getattr(args, "out_dir", None)
    if out_dir:
        from pathlib import Path  # noqa: PLC0415

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        corr.to_csv(out / "property_bias_correlations.csv", index=False)
        pd.DataFrame(topk_rows).to_csv(
            out / "property_bias_topk.csv", index=False
        )
        props.to_csv(out / "property_bias_props.csv", index=False)
        log.info("Wrote property_bias_*.csv to %s", out)
