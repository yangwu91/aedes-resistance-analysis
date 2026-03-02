#!/usr/bin/env python3
"""
a03_meta_mortality.py -- Meta-analysis of bioassay mortality rates.

For each insecticide (grouped by insecticide_class) with >= MIN_STUDIES_FOR_META
studies, applies the Freeman-Tukey double arcsine transformation and runs a
DerSimonian-Laird random-effects meta-analysis.  Generates forest plots and
saves a summary table.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from utils import *


def _prepare_forest_df(sub: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame with transformed values for the forest plot."""
    records = []
    for _, row in sub.iterrows():
        x = int(round(row["n_dead"])) if pd.notna(row.get("n_dead")) else int(round(row["mortality_pct"] / 100 * row["n_tested"]))
        n = int(row["n_tested"])
        t, v = freeman_tukey_double_arcsine(x, n)
        label = row.get("study_id", "")
        if pd.notna(row.get("country")):
            label = f"{label} ({row['country']})"
        if pd.notna(row.get("year")):
            label = f"{label} {int(row['year'])}"
        records.append({
            "label": label.strip(),
            "yi": t,
            "vi": v,
            "n": n,
            "mortality_pct": row["mortality_pct"],
        })
    return pd.DataFrame(records)


def analyse_insecticide(name: str, insecticide_class: str, sub: pd.DataFrame,
                        forest_dir: Path) -> dict:
    """Run meta-analysis for a single insecticide and generate forest plot.

    Returns a summary dict or None if analysis cannot be performed.
    """
    k = len(sub)
    if k < MIN_STUDIES_FOR_META:
        return None

    # Build forest data
    fdf = _prepare_forest_df(sub)
    if fdf.empty:
        return None

    yi = fdf["yi"].values
    vi = fdf["vi"].values
    ns = fdf["n"].values

    # DerSimonian-Laird random-effects
    res = meta_analysis_dl(yi, vi)

    # Back-transform pooled estimate and CI
    n_harm = harmonic_mean(ns)
    pooled_mortality = back_transform_ft(res["mu"], n_harm)
    ci_lower = back_transform_ft(res["ci_lower"], n_harm)
    ci_upper = back_transform_ft(res["ci_upper"], n_harm)

    # Forest plot (skip if too many studies to avoid oversized image)
    if k <= 50:
        safe_name = name.replace("/", "-").replace(" ", "_")
        fp_path = forest_dir / f"forest_mortality_{safe_name}.{FIGURE_FORMAT}"
        forest_plot(
            studies=fdf,
            yi_col="yi",
            vi_col="vi",
            label_col="label",
            title=f"Mortality -- {name} ({insecticide_class})",
            filepath=str(fp_path),
            xlabel="Freeman-Tukey transformed mortality",
        )
    else:
        print(f"  Skipping forest plot for {name} (k={k} > 50)")

    return {
        "insecticide": name,
        "insecticide_class": insecticide_class,
        "k": res["k"],
        "pooled_mortality": round(pooled_mortality * 100, 2),
        "ci_lower": round(ci_lower * 100, 2),
        "ci_upper": round(ci_upper * 100, 2),
        "I2": round(res["I2"], 1),
        "tau2": round(res["tau2"], 6),
        "Q": round(res["Q"], 2),
        "Q_p": round(res["Q_p"], 4),
    }


def main():
    print("=" * 60)
    print("a03 -- Meta-analysis of bioassay mortality rates")
    print("=" * 60)

    # Load data
    fp = DATA_PROCESSED / "mortality_data.csv"
    if not fp.exists():
        print(f"[WARNING] Data file not found: {fp}")
        print("  Run a01_data_cleaning.py first.")
        return
    df = pd.read_csv(fp)
    print(f"  Loaded {len(df)} rows from {fp.name}")

    if df.empty:
        print("[WARNING] Mortality data is empty. Nothing to analyse.")
        return

    # Ensure required columns exist
    required = ["insecticide_name", "insecticide_class", "n_tested", "mortality_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return

    # Compute n_dead if absent
    if "n_dead" not in df.columns:
        df["n_dead"] = (df["mortality_pct"] / 100 * df["n_tested"]).round()

    # Drop rows where essential data is missing
    df = df.dropna(subset=["insecticide_name", "n_tested", "mortality_pct"]).copy()
    if df.empty:
        print("[WARNING] No valid rows after dropping NAs. Nothing to analyse.")
        return

    # Output directories
    forest_dir = FIGURES_DIR / "forest_plots"
    forest_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Analyse each insecticide grouped by class
    results = []
    insecticides = df.groupby(["insecticide_class", "insecticide_name"])

    print(f"\n  Found {len(insecticides)} insecticide groups.\n")

    for (cls, name), sub in insecticides:
        k = len(sub)
        if k < MIN_STUDIES_FOR_META:
            print(f"  Skipping {name} ({cls}): only {k} studies (< {MIN_STUDIES_FOR_META})")
            continue

        print(f"  Analysing {name} ({cls}): k = {k}")
        result = analyse_insecticide(name, cls, sub, forest_dir)
        if result is not None:
            results.append(result)

    if not results:
        print("\n[WARNING] No insecticides had enough studies for meta-analysis.")
        return

    # Summary table
    summary = pd.DataFrame(results)
    summary = summary.sort_values(["insecticide_class", "insecticide"]).reset_index(drop=True)
    out_path = TABLES_DIR / "table2_mortality_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Summary table saved: {out_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("MORTALITY META-ANALYSIS SUMMARY")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(
            f"  {row['insecticide']:25s} ({row['insecticide_class']:15s}) | "
            f"k={row['k']:3d} | "
            f"Pooled={row['pooled_mortality']:6.1f}% "
            f"[{row['ci_lower']:.1f}, {row['ci_upper']:.1f}] | "
            f"I2={row['I2']:.1f}% | "
            f"Q={row['Q']:.1f} (p={row['Q_p']:.3f})"
        )
    print(f"\n  Total insecticides analysed: {len(summary)}")
    print("Done.")


if __name__ == "__main__":
    main()
