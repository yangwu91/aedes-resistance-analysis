#!/usr/bin/env python3
"""
a04_meta_rr.py -- Meta-analysis of resistance ratios (RR50/RR95).

For each insecticide with >= MIN_STUDIES_FOR_META studies, applies log
transformation and runs a DerSimonian-Laird random-effects meta-analysis.
Back-transforms to the original RR scale.  Generates forest plots and saves a
summary table.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from utils import *


def _prepare_forest_df(sub: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame with log-transformed RR values for the forest plot."""
    records = []
    for _, row in sub.iterrows():
        rr = row["rr_value"]
        ci_lo = row["rr_ci_lower"]
        ci_hi = row["rr_ci_upper"]

        # Skip invalid RR values
        if pd.isna(rr) or rr <= 0:
            continue
        if pd.isna(ci_lo) or pd.isna(ci_hi) or ci_lo <= 0 or ci_hi <= 0:
            continue

        yi, vi = log_rr_transform(rr, ci_lo, ci_hi)

        label = row.get("study_id", "")
        if pd.notna(row.get("country")):
            label = f"{label} ({row['country']})"
        if pd.notna(row.get("year")):
            label = f"{label} {int(row['year'])}"

        records.append({
            "label": label.strip(),
            "yi": yi,
            "vi": vi,
            "rr_value": rr,
        })
    return pd.DataFrame(records)


def analyse_insecticide(name: str, insecticide_class: str, sub: pd.DataFrame,
                        forest_dir: Path) -> dict:
    """Run meta-analysis for a single insecticide and generate forest plot.

    Returns a summary dict or None if analysis cannot be performed.
    """
    fdf = _prepare_forest_df(sub)
    if fdf.empty or len(fdf) < MIN_STUDIES_FOR_META:
        return None

    yi = fdf["yi"].values
    vi = fdf["vi"].values

    # DerSimonian-Laird random-effects
    res = meta_analysis_dl(yi, vi)

    # Back-transform: pooled_RR = exp(mu), CI = exp(ci)
    pooled_rr = np.exp(res["mu"])
    ci_lower = np.exp(res["ci_lower"])
    ci_upper = np.exp(res["ci_upper"])

    # Forest plot (on log scale)
    safe_name = name.replace("/", "-").replace(" ", "_")
    fp_path = forest_dir / f"forest_rr_{safe_name}.{FIGURE_FORMAT}"
    forest_plot(
        studies=fdf,
        yi_col="yi",
        vi_col="vi",
        label_col="label",
        title=f"Resistance Ratio -- {name} ({insecticide_class})",
        filepath=str(fp_path),
        xlabel="ln(RR)",
    )

    return {
        "insecticide": name,
        "insecticide_class": insecticide_class,
        "k": res["k"],
        "pooled_RR": round(pooled_rr, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "I2": round(res["I2"], 1),
        "tau2": round(res["tau2"], 6),
        "Q": round(res["Q"], 2),
        "Q_p": round(res["Q_p"], 4),
    }


def main():
    print("=" * 60)
    print("a04 -- Meta-analysis of resistance ratios (RR)")
    print("=" * 60)

    # Load data
    fp = DATA_PROCESSED / "rr_data.csv"
    if not fp.exists():
        print(f"[WARNING] Data file not found: {fp}")
        print("  Run a01_data_cleaning.py first.")
        return
    df = pd.read_csv(fp)
    print(f"  Loaded {len(df)} rows from {fp.name}")

    if df.empty:
        print("[WARNING] RR data is empty. Nothing to analyse.")
        return

    # Ensure required columns exist
    required = ["insecticide_name", "rr_value", "rr_ci_lower", "rr_ci_upper"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return

    # Assign class if not present
    if "insecticide_class" not in df.columns:
        df["insecticide_class"] = df["insecticide_name"].map(INSECTICIDE_CLASS_MAP).fillna("Unknown")

    # Drop invalid rows
    df = df.dropna(subset=["insecticide_name", "rr_value", "rr_ci_lower", "rr_ci_upper"]).copy()
    df = df[(df["rr_value"] > 0) & (df["rr_ci_lower"] > 0) & (df["rr_ci_upper"] > 0)].copy()

    if df.empty:
        print("[WARNING] No valid rows after filtering. Nothing to analyse.")
        return

    # Output directories
    forest_dir = FIGURES_DIR / "forest_plots"
    forest_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Analyse each insecticide
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
    out_path = TABLES_DIR / "table3_rr_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Summary table saved: {out_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("RESISTANCE RATIO META-ANALYSIS SUMMARY")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(
            f"  {row['insecticide']:25s} ({row['insecticide_class']:15s}) | "
            f"k={row['k']:3d} | "
            f"Pooled RR={row['pooled_RR']:7.2f} "
            f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}] | "
            f"I2={row['I2']:.1f}% | "
            f"Q={row['Q']:.1f} (p={row['Q_p']:.3f})"
        )
    print(f"\n  Total insecticides analysed: {len(summary)}")
    print("Done.")


if __name__ == "__main__":
    main()
