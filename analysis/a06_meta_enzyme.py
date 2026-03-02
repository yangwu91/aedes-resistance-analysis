#!/usr/bin/env python3
"""
a06_meta_enzyme.py -- Meta-analysis of metabolic enzyme activity.

For each enzyme system (MFO, NSE, GST, AChE) with >= MIN_STUDIES_FOR_META
studies, computes Hedges' g (standardised mean difference between field and
reference populations) and runs a DerSimonian-Laird random-effects
meta-analysis.  Generates forest plots and saves a summary table.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from utils import *


def _prepare_forest_df(sub: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame with Hedges' g effect sizes."""
    records = []
    for _, row in sub.iterrows():
        fm = row["field_mean"]
        fsd = row["field_sd"]
        fn = row["field_n"]
        rm = row["reference_mean"]
        rsd = row["reference_sd"]
        rn = row["reference_n"]

        # Validate inputs
        if any(pd.isna(v) for v in [fm, fsd, fn, rm, rsd, rn]):
            continue
        fn = int(fn)
        rn = int(rn)
        if fn < 2 or rn < 2 or fsd <= 0 or rsd <= 0:
            continue

        g, vg = hedges_g(fm, fsd, fn, rm, rsd, rn)

        label = row.get("study_id", "")
        if pd.notna(row.get("country")):
            label = f"{label} ({row['country']})"
        if pd.notna(row.get("year")):
            label = f"{label} {int(row['year'])}"

        records.append({
            "label": label.strip(),
            "yi": g,
            "vi": vg,
        })
    return pd.DataFrame(records)


def analyse_enzyme(enzyme: str, sub: pd.DataFrame,
                   forest_dir: Path) -> dict:
    """Run meta-analysis for a single enzyme system and generate forest plot.

    Returns a summary dict or None if analysis cannot be performed.
    """
    fdf = _prepare_forest_df(sub)
    if fdf.empty or len(fdf) < MIN_STUDIES_FOR_META:
        return None

    yi = fdf["yi"].values
    vi = fdf["vi"].values

    # DerSimonian-Laird random-effects
    res = meta_analysis_dl(yi, vi)

    # Forest plot
    safe_name = enzyme.replace("/", "-").replace(" ", "_")
    fp_path = forest_dir / f"forest_enzyme_{safe_name}.{FIGURE_FORMAT}"
    forest_plot(
        studies=fdf,
        yi_col="yi",
        vi_col="vi",
        label_col="label",
        title=f"Enzyme Activity -- {enzyme} (Hedges' g)",
        filepath=str(fp_path),
        xlabel="Hedges' g",
    )

    return {
        "enzyme_system": enzyme,
        "k": res["k"],
        "pooled_hedges_g": round(res["mu"], 3),
        "ci_lower": round(res["ci_lower"], 3),
        "ci_upper": round(res["ci_upper"], 3),
        "I2": round(res["I2"], 1),
        "tau2": round(res["tau2"], 6),
        "Q": round(res["Q"], 2),
        "Q_p": round(res["Q_p"], 4),
    }


def main():
    print("=" * 60)
    print("a06 -- Meta-analysis of metabolic enzyme activity")
    print("=" * 60)

    # Load data
    fp = DATA_PROCESSED / "enzyme_data.csv"
    if not fp.exists():
        print(f"[WARNING] Data file not found: {fp}")
        print("  Run a01_data_cleaning.py first.")
        return
    df = pd.read_csv(fp)
    print(f"  Loaded {len(df)} rows from {fp.name}")

    if df.empty:
        print("[WARNING] Enzyme data is empty. Nothing to analyse.")
        return

    # Ensure required columns exist
    required = [
        "enzyme_system",
        "field_mean", "field_sd", "field_n",
        "reference_mean", "reference_sd", "reference_n",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return

    # Drop rows where essential data is missing
    df = df.dropna(subset=required).copy()
    # Filter out invalid rows
    df = df[(df["field_n"] >= 2) & (df["reference_n"] >= 2)].copy()
    df = df[(df["field_sd"] > 0) & (df["reference_sd"] > 0)].copy()

    if df.empty:
        print("[WARNING] No valid rows after filtering. Nothing to analyse.")
        return

    # Output directories
    forest_dir = FIGURES_DIR / "forest_plots"
    forest_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Analyse each enzyme system
    results = []
    enzymes = df.groupby("enzyme_system")

    print(f"\n  Found {len(enzymes)} enzyme systems.\n")

    for enzyme, sub in enzymes:
        k = len(sub)
        if k < MIN_STUDIES_FOR_META:
            print(f"  Skipping {enzyme}: only {k} studies (< {MIN_STUDIES_FOR_META})")
            continue

        print(f"  Analysing {enzyme}: k = {k}")
        result = analyse_enzyme(enzyme, sub, forest_dir)
        if result is not None:
            results.append(result)

    if not results:
        print("\n[WARNING] No enzyme systems had enough studies for meta-analysis.")
        return

    # Summary table
    summary = pd.DataFrame(results)
    summary = summary.sort_values("enzyme_system").reset_index(drop=True)
    out_path = TABLES_DIR / "table5_enzyme_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Summary table saved: {out_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("ENZYME ACTIVITY META-ANALYSIS SUMMARY")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(
            f"  {row['enzyme_system']:8s} | "
            f"k={row['k']:3d} | "
            f"Pooled g={row['pooled_hedges_g']:+6.3f} "
            f"[{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}] | "
            f"I2={row['I2']:.1f}% | "
            f"Q={row['Q']:.1f} (p={row['Q_p']:.3f})"
        )
    print(f"\n  Total enzyme systems analysed: {len(summary)}")
    print("Done.")


if __name__ == "__main__":
    main()
