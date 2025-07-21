#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import shutil

def main() -> None:
    # -------------------------------------------------------
    # 0️⃣  Parse CLI argument
    # -------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Analyse pre/post‑optimised hit statistics "
                    "and save the resulting lists to text files."
    )
    parser.add_argument(
        "csv_path",
        metavar="CSV",
        help="Path to predictions.csv",
    )
    args = parser.parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()

    # -------------------------------------------------------
    # 1 Load dataframe and tag rows
    # -------------------------------------------------------
    df = pd.read_csv(csv_path)
    df["is_opt"] = df["xyz_file"].str.endswith("_opt.xyz")
    df["base"]   = df["xyz_file"].str.replace(r"_opt\.xyz$", ".xyz", regex=True)

    # -------------------------------------------------------
    # 2️ Pivot so each base molecule has pre/post hit flags
    # -------------------------------------------------------
    pivot = (
        df.pivot_table(
            index="base",
            columns="is_opt",          # False ⇒ pre, True ⇒ post
            values="hit",
            aggfunc="first",
        )
    )
    pivot.columns = ["pre_hit", "post_hit"]
    paired = pivot.dropna()            # only bases with both versions

    # -------------------------------------------------------
    # 3️ Categorise
    # -------------------------------------------------------
    pre_hit_post_miss = paired.query("pre_hit == 1 and post_hit == 0").index.tolist()
    both_hit          = paired.query("pre_hit == 1 and post_hit == 1").index.tolist()
    pre_miss_post_hit = paired.query("pre_hit == 0 and post_hit == 1").index.tolist()

    # -------------------------------------------------------
    # 4️ Summary stats (optional printing)
    # -------------------------------------------------------
    n_unpaired_pre = len(df[~df["is_opt"]]) - len(paired)
    n_base         = df["base"].nunique()
    print(f"Total bases           : {n_base}")
    print(f"pre-hit ➜ post-miss   : {len(pre_hit_post_miss)} ({len(pre_hit_post_miss)/n_base*100:.2f}%)")
    print(f"both-hit              : {len(both_hit)} ({len(both_hit)/n_base*100:.2f}%)")
    print(f"pre-miss ➜ post-hit   : {len(pre_miss_post_hit)} ({len(pre_miss_post_hit)/n_base*100:.2f}%)")
    print(f"skipped pre-only      : {n_unpaired_pre} ({n_unpaired_pre/n_base*100:.2f}%)")
    # -------------------------------------------------------
    # 5 Write the three lists to disk
    # -------------------------------------------------------
    out_dir = csv_path.parent
    (out_dir / "pre_hit_post_miss.txt").write_text("\n".join(pre_hit_post_miss), encoding="utf-8")
    (out_dir / "both_hit.txt").write_text("\n".join(both_hit), encoding="utf-8")
    (out_dir / "pre_miss_post_hit.txt").write_text("\n".join(pre_miss_post_hit), encoding="utf-8")

    print("Lists written to:")
    print(f"  {out_dir / 'pre_hit_post_miss.txt'}")
    print(f"  {out_dir / 'both_hit.txt'}")
    print(f"  {out_dir / 'pre_miss_post_hit.txt'}")

    # -------------------------------------------------------
    # 6 Collect post-hit xyz files in a sub-directory
    # -------------------------------------------------------
    post_hit_xyz = df.loc[df["is_opt"] & (df["hit"] == 1), "xyz_file"]
    dest_dir = out_dir / "post_hit_xyz"
    dest_dir.mkdir(exist_ok=True)

    for xyz in post_hit_xyz:
        src = Path(xyz).expanduser()
        # if not src.is_absolute():
        #     # treat relative paths as relative to the CSV directory
            # src = (csv_path.parent / src).resolve()

        if src.exists():
            shutil.copy2(src, dest_dir / src.name)
        else:
            print(f"⚠️  Warning: {src} not found, skipping.")

    print(f"\nCopied {len(list(dest_dir.iterdir()))} post-hit files to {dest_dir}")


if __name__ == "__main__":
    main()

