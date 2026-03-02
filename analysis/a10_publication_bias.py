#!/usr/bin/env python3
"""
a10_publication_bias.py -- Publication bias assessment.

For each insecticide class with >= 10 studies, generates funnel plots,
runs Egger's test, Begg's test, and trim-and-fill analysis.  Saves a
summary table of bias assessment results.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from utils import *


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

MIN_STUDIES_BIAS = 10  # minimum studies for publication bias assessment


def _ensure_dir(filepath: Path) -> None:
    """Create parent directories for *filepath* if they do not exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)


def _ft_transform_class(sub: pd.DataFrame) -> pd.DataFrame:
    """Apply Freeman-Tukey transformation to mortality data rows."""
    records = []
    for _, row in sub.iterrows():
        if pd.isna(row.get("n_tested")) or pd.isna(row.get("mortality_pct")):
            continue
        n = int(row["n_tested"])
        if n <= 0:
            continue
        if pd.notna(row.get("n_dead")):
            x = int(round(row["n_dead"]))
        else:
            x = int(round(row["mortality_pct"] / 100 * n))
        x = max(0, min(x, n))
        t, v = freeman_tukey_double_arcsine(x, n)
        records.append({"yi": t, "vi": v, "n": n})
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("a10 -- Publication bias assessment")
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

    # Ensure required columns
    required = ["insecticide_class", "n_tested", "mortality_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return

    # Compute n_dead if absent
    if "n_dead" not in df.columns:
        df["n_dead"] = (df["mortality_pct"] / 100 * df["n_tested"]).round()

    # Drop rows with missing essential data
    df = df.dropna(subset=["insecticide_class", "n_tested", "mortality_pct"]).copy()
    if df.empty:
        print("[WARNING] No valid rows after dropping NAs.")
        return

    # Output directories
    funnel_dir = FIGURES_DIR / "funnel_plots"
    funnel_dir.mkdir(parents=True, exist_ok=True)
    supp_dir = TABLES_DIR / "supplementary_tables"
    supp_dir.mkdir(parents=True, exist_ok=True)

    # Identify insecticide classes with >= MIN_STUDIES_BIAS studies
    class_counts = df.groupby("insecticide_class").size()
    eligible_classes = class_counts[class_counts >= MIN_STUDIES_BIAS].index.tolist()

    print(f"\n  Insecticide classes with >= {MIN_STUDIES_BIAS} studies: {eligible_classes}")
    if not eligible_classes:
        print(f"\n[WARNING] No insecticide classes have >= {MIN_STUDIES_BIAS} studies.")
        print("  Publication bias assessment requires at least 10 studies per class.")
        return

    results = []

    for cls in eligible_classes:
        print(f"\n  --- {cls} ---")
        sub = df[df["insecticide_class"] == cls].copy()
        fdf = _ft_transform_class(sub)

        if fdf.empty or len(fdf) < MIN_STUDIES_BIAS:
            print(f"    Only {len(fdf)} valid effect sizes -- skipping.")
            continue

        yi = fdf["yi"].values
        vi = fdf["vi"].values
        ns = fdf["n"].values
        k = len(yi)
        print(f"    k = {k} studies")

        # 1. Funnel plot
        safe_name = cls.replace("/", "-").replace(" ", "_")
        fp_path = funnel_dir / f"funnel_{safe_name}.{FIGURE_FORMAT}"
        funnel_plot(yi, vi, title=f"Funnel plot -- {cls}", filepath=str(fp_path))

        # 2. Egger's test
        try:
            egger = eggers_test(yi, vi)
            egger_intercept = round(egger["intercept"], 4)
            egger_p = round(egger["p_value"], 4)
            print(f"    Egger's test: intercept = {egger_intercept}, p = {egger_p}")
        except Exception as e:
            print(f"    Egger's test failed: {e}")
            egger_intercept = np.nan
            egger_p = np.nan

        # 3. Begg's test
        try:
            begg = beggs_test(yi, vi)
            begg_tau = round(begg["tau"], 4)
            begg_p = round(begg["p_value"], 4)
            print(f"    Begg's test: tau = {begg_tau}, p = {begg_p}")
        except Exception as e:
            print(f"    Begg's test failed: {e}")
            begg_tau = np.nan
            begg_p = np.nan

        # 4. Trim-and-fill
        try:
            tf = trim_and_fill(yi, vi)
            n_missing = tf.get("n_missing", 0)
            n_harm = harmonic_mean(ns)
            adjusted_mu = tf["mu"]
            adjusted_est = back_transform_ft(adjusted_mu, n_harm)
            adjusted_est = round(adjusted_est * 100, 2)
            print(f"    Trim-and-fill: {n_missing} missing studies, "
                  f"adjusted estimate = {adjusted_est}%")
        except Exception as e:
            print(f"    Trim-and-fill failed: {e}")
            n_missing = np.nan
            adjusted_est = np.nan

        # Original pooled estimate for comparison
        res = meta_analysis_dl(yi, vi)
        original_est = back_transform_ft(res["mu"], harmonic_mean(ns))
        original_est = round(original_est * 100, 2)

        # Compile result
        results.append({
            "insecticide_class": cls,
            "k": k,
            "original_estimate": original_est,
            "egger_intercept": egger_intercept,
            "egger_p": egger_p,
            "begg_tau": begg_tau,
            "begg_p": begg_p,
            "n_missing_trimfill": n_missing,
            "adjusted_estimate": adjusted_est,
        })

    if not results:
        print("\n[WARNING] No publication bias assessments could be completed.")
        return

    # Save summary table
    summary = pd.DataFrame(results)
    out_path = supp_dir / "publication_bias.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Bias assessment table saved: {out_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("PUBLICATION BIAS ASSESSMENT SUMMARY")
    print("=" * 60)
    print(f"  {'Insecticide Class':25s} | {'k':>4s} | {'Egger p':>8s} | "
          f"{'Begg p':>8s} | {'Missing':>7s} | {'Original':>8s} | {'Adjusted':>8s}")
    print("  " + "-" * 90)
    for _, row in summary.iterrows():
        egger_sig = "*" if pd.notna(row["egger_p"]) and row["egger_p"] < 0.05 else " "
        begg_sig = "*" if pd.notna(row["begg_p"]) and row["begg_p"] < 0.05 else " "
        n_miss = int(row["n_missing_trimfill"]) if pd.notna(row["n_missing_trimfill"]) else "N/A"
        print(
            f"  {row['insecticide_class']:25s} | "
            f"{row['k']:4d} | "
            f"{row['egger_p']:7.4f}{egger_sig} | "
            f"{row['begg_p']:7.4f}{begg_sig} | "
            f"{str(n_miss):>7s} | "
            f"{row['original_estimate']:7.1f}% | "
            f"{row['adjusted_estimate']:7.1f}%"
        )

    # Interpretation
    print("\n  Interpretation:")
    any_bias = False
    for _, row in summary.iterrows():
        flags = []
        if pd.notna(row["egger_p"]) and row["egger_p"] < 0.05:
            flags.append("Egger's test significant")
        if pd.notna(row["begg_p"]) and row["begg_p"] < 0.05:
            flags.append("Begg's test significant")
        if pd.notna(row["n_missing_trimfill"]) and row["n_missing_trimfill"] > 0:
            flags.append(f"{int(row['n_missing_trimfill'])} missing studies imputed")

        if flags:
            any_bias = True
            print(f"    {row['insecticide_class']}: " + "; ".join(flags))
        else:
            print(f"    {row['insecticide_class']}: No evidence of publication bias")

    if not any_bias:
        print("    No significant evidence of publication bias detected overall.")

    print(f"\n  Note: * indicates p < 0.05")
    print("Done.")


if __name__ == "__main__":
    main()
