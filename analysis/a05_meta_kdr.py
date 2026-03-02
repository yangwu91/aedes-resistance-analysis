#!/usr/bin/env python3
"""
a05_meta_kdr.py -- Meta-analysis of kdr mutation allele frequencies.

For each unique mutation (e.g., F1534C, V1016G, S989P) with
>= MIN_STUDIES_FOR_META studies, applies the Freeman-Tukey double arcsine
transformation (treating allele_frequency as a proportion with
n = n_genotyped * 2 for diploid organisms) and runs a DerSimonian-Laird
random-effects meta-analysis.  Generates forest plots and saves a summary
table.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from utils import *


def _prepare_forest_df(sub: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame with FT-transformed allele frequencies."""
    records = []
    for _, row in sub.iterrows():
        freq = row["allele_frequency"]
        n_geno = int(row["n_genotyped"])

        # Diploid: total alleles = 2 * n_genotyped
        n_alleles = n_geno * 2

        # Number of mutant alleles
        if pd.notna(row.get("n_mutant_alleles")) and row["n_mutant_alleles"] > 0:
            x = int(round(row["n_mutant_alleles"]))
        else:
            x = int(round(freq * n_alleles))

        # Ensure x is within valid range
        x = max(0, min(x, n_alleles))

        t, v = freeman_tukey_double_arcsine(x, n_alleles)

        label = row.get("study_id", "")
        if pd.notna(row.get("country")):
            label = f"{label} ({row['country']})"
        if pd.notna(row.get("year")):
            label = f"{label} {int(row['year'])}"

        records.append({
            "label": label.strip(),
            "yi": t,
            "vi": v,
            "n_alleles": n_alleles,
            "allele_frequency": freq,
        })
    return pd.DataFrame(records)


def analyse_mutation(mutation: str, sub: pd.DataFrame,
                     forest_dir: Path) -> dict:
    """Run meta-analysis for a single kdr mutation and generate forest plot.

    Returns a summary dict or None if analysis cannot be performed.
    """
    fdf = _prepare_forest_df(sub)
    if fdf.empty or len(fdf) < MIN_STUDIES_FOR_META:
        return None

    yi = fdf["yi"].values
    vi = fdf["vi"].values
    ns = fdf["n_alleles"].values

    # DerSimonian-Laird random-effects
    res = meta_analysis_dl(yi, vi)

    # Back-transform pooled estimate and CI
    n_harm = harmonic_mean(ns)
    pooled_freq = back_transform_ft(res["mu"], n_harm)
    ci_lower = back_transform_ft(res["ci_lower"], n_harm)
    ci_upper = back_transform_ft(res["ci_upper"], n_harm)

    # Forest plot
    safe_name = mutation.replace("/", "-").replace(" ", "_")
    fp_path = forest_dir / f"forest_kdr_{safe_name}.{FIGURE_FORMAT}"
    forest_plot(
        studies=fdf,
        yi_col="yi",
        vi_col="vi",
        label_col="label",
        title=f"kdr Allele Frequency -- {mutation}",
        filepath=str(fp_path),
        xlabel="Freeman-Tukey transformed frequency",
    )

    return {
        "mutation": mutation,
        "k": res["k"],
        "pooled_frequency": round(pooled_freq, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "I2": round(res["I2"], 1),
        "tau2": round(res["tau2"], 6),
        "Q": round(res["Q"], 2),
        "Q_p": round(res["Q_p"], 4),
    }


def main():
    print("=" * 60)
    print("a05 -- Meta-analysis of kdr mutation allele frequencies")
    print("=" * 60)

    # Load data
    fp = DATA_PROCESSED / "kdr_data.csv"
    if not fp.exists():
        print(f"[WARNING] Data file not found: {fp}")
        print("  Run a01_data_cleaning.py first.")
        return
    df = pd.read_csv(fp)
    print(f"  Loaded {len(df)} rows from {fp.name}")

    if df.empty:
        print("[WARNING] kdr data is empty. Nothing to analyse.")
        return

    # Ensure required columns exist
    required = ["mutation", "allele_frequency", "n_genotyped"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return

    # Drop rows where essential data is missing
    df = df.dropna(subset=["mutation", "allele_frequency", "n_genotyped"]).copy()
    df = df[df["n_genotyped"] > 0].copy()

    if df.empty:
        print("[WARNING] No valid rows after filtering. Nothing to analyse.")
        return

    # Compute n_mutant_alleles if absent
    if "n_mutant_alleles" not in df.columns:
        df["n_mutant_alleles"] = (df["allele_frequency"] * df["n_genotyped"] * 2).round()

    # Output directories
    forest_dir = FIGURES_DIR / "forest_plots"
    forest_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Analyse each mutation
    results = []
    mutations = df.groupby("mutation")

    print(f"\n  Found {len(mutations)} unique mutations.\n")

    for mutation, sub in mutations:
        k = len(sub)
        if k < MIN_STUDIES_FOR_META:
            print(f"  Skipping {mutation}: only {k} studies (< {MIN_STUDIES_FOR_META})")
            continue

        print(f"  Analysing {mutation}: k = {k}")
        result = analyse_mutation(mutation, sub, forest_dir)
        if result is not None:
            results.append(result)

    if not results:
        print("\n[WARNING] No mutations had enough studies for meta-analysis.")
        return

    # Summary table
    summary = pd.DataFrame(results)
    summary = summary.sort_values("mutation").reset_index(drop=True)
    out_path = TABLES_DIR / "table4_kdr_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Summary table saved: {out_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("kdr ALLELE FREQUENCY META-ANALYSIS SUMMARY")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(
            f"  {row['mutation']:12s} | "
            f"k={row['k']:3d} | "
            f"Pooled freq={row['pooled_frequency']:.4f} "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | "
            f"I2={row['I2']:.1f}% | "
            f"Q={row['Q']:.1f} (p={row['Q_p']:.3f})"
        )
    print(f"\n  Total mutations analysed: {len(summary)}")
    print("Done.")


if __name__ == "__main__":
    main()
