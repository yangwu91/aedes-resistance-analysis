#!/usr/bin/env python3
"""
a08_subgroup_analysis.py -- Subgroup meta-analyses of mortality data.

Stratifies mortality data by WHO region, time period, species, life stage,
and bioassay method.  For each subgroup with >= MIN_STUDIES_FOR_META studies,
applies the Freeman-Tukey double arcsine transformation and runs a
DerSimonian-Laird random-effects meta-analysis.  Computes Q-between for
interaction between subgroups.  Generates grouped bar plots and forest plots.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from utils import *


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _ensure_dir(filepath: Path) -> None:
    """Create parent directories for *filepath* if they do not exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)


def _assign_time_period(year):
    """Categorize collection_year_start into time periods."""
    if pd.isna(year):
        return np.nan
    year = int(year)
    if year < 2010:
        return "Before 2010"
    elif year <= 2019:
        return "2010-2019"
    else:
        return "2020-present"


def _assign_species_group(species):
    """Categorize species into Ae. aegypti, Ae. albopictus, or Others."""
    if pd.isna(species):
        return np.nan
    s = str(species).strip().lower()
    if "aegypti" in s:
        return "Ae. aegypti"
    elif "albopictus" in s:
        return "Ae. albopictus"
    else:
        return "Others"


def _standardise_bioassay(method):
    """Standardise bioassay method names."""
    if pd.isna(method):
        return np.nan
    m = str(method).strip().lower()
    if "who" in m or "tube" in m:
        return "WHO tube"
    elif "cdc" in m or "bottle" in m:
        return "CDC bottle"
    else:
        return str(method).strip()


def _ft_transform_rows(sub: pd.DataFrame) -> pd.DataFrame:
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


def run_subgroup_meta(sub: pd.DataFrame) -> dict:
    """Run FT meta-analysis on a subgroup of mortality data.

    Returns a dict with pooled estimate, CI, heterogeneity stats,
    or None if insufficient data.
    """
    fdf = _ft_transform_rows(sub)
    if fdf.empty or len(fdf) < MIN_STUDIES_FOR_META:
        return None

    yi = fdf["yi"].values
    vi = fdf["vi"].values
    ns = fdf["n"].values

    res = meta_analysis_dl(yi, vi)
    n_harm = harmonic_mean(ns)
    pooled = back_transform_ft(res["mu"], n_harm)
    ci_lo = back_transform_ft(res["ci_lower"], n_harm)
    ci_hi = back_transform_ft(res["ci_upper"], n_harm)

    return {
        "k": res["k"],
        "pooled_mortality": round(pooled * 100, 2),
        "ci_lower": round(ci_lo * 100, 2),
        "ci_upper": round(ci_hi * 100, 2),
        "I2": round(res["I2"], 1),
        "tau2": res["tau2"],
        "Q": res["Q"],
        "Q_p": round(res["Q_p"], 4),
        "mu_transformed": res["mu"],
    }


def compute_q_between(subgroup_results: list[dict], total_Q: float) -> dict:
    """Compute Q-between statistic for interaction between subgroups.

    Q_between = Q_total - sum(Q_within_i)
    """
    Q_within_sum = sum(r["Q"] for r in subgroup_results)
    df_within = sum(r["k"] - 1 for r in subgroup_results)
    Q_between = max(0, total_Q - Q_within_sum)
    df_between = len(subgroup_results) - 1

    from scipy import stats as sp_stats
    if df_between > 0:
        p_between = 1 - sp_stats.chi2.cdf(Q_between, df_between)
    else:
        p_between = np.nan

    return {
        "Q_between": round(Q_between, 2),
        "df_between": df_between,
        "p_between": round(p_between, 4) if not np.isnan(p_between) else np.nan,
        "Q_within_sum": round(Q_within_sum, 2),
    }


# ──────────────────────────────────────────────────────────────────────
# Subgroup analysis for one stratification variable
# ──────────────────────────────────────────────────────────────────────

def analyse_stratification(
    df: pd.DataFrame,
    strat_col: str,
    strat_label: str,
    insecticide_class: str,
    forest_dir: Path,
) -> list[dict]:
    """Run subgroup meta-analysis for a stratification variable within
    a given insecticide class.

    Returns a list of result dicts (one per subgroup).
    """
    sub = df[df["insecticide_class"] == insecticide_class].copy()
    if sub.empty or strat_col not in sub.columns:
        return []

    sub = sub.dropna(subset=[strat_col])
    if sub.empty:
        return []

    groups = sub.groupby(strat_col)
    subgroup_results = []
    subgroup_meta_results = []

    for group_val, group_df in groups:
        res = run_subgroup_meta(group_df)
        if res is None:
            continue
        res["stratification"] = strat_label
        res["subgroup"] = str(group_val)
        res["insecticide_class"] = insecticide_class
        subgroup_results.append(res)
        subgroup_meta_results.append(res)

    # Compute Q_between if we have >= 2 subgroups
    if len(subgroup_meta_results) >= 2:
        # Compute total Q from the combined data
        all_fdf = _ft_transform_rows(sub)
        if not all_fdf.empty and len(all_fdf) >= 2:
            total_res = meta_analysis_dl(all_fdf["yi"].values, all_fdf["vi"].values)
            q_info = compute_q_between(subgroup_meta_results, total_res["Q"])
            for r in subgroup_results:
                r["Q_between"] = q_info["Q_between"]
                r["df_between"] = q_info["df_between"]
                r["p_interaction"] = q_info["p_between"]
    else:
        for r in subgroup_results:
            r["Q_between"] = np.nan
            r["df_between"] = np.nan
            r["p_interaction"] = np.nan

    # Generate forest plot for this stratification if there are results
    if subgroup_results:
        _generate_subgroup_forest(
            subgroup_results, strat_label, insecticide_class, forest_dir
        )

    return subgroup_results


def _generate_subgroup_forest(
    results: list[dict],
    strat_label: str,
    insecticide_class: str,
    forest_dir: Path,
) -> None:
    """Generate a grouped forest plot for subgroup results."""
    if not results:
        return

    rdf = pd.DataFrame(results)
    k = len(rdf)

    fig, ax = plt.subplots(figsize=(10, max(4, k * 0.6 + 2)))
    y_positions = np.arange(k, 0, -1)

    for i, (_, row) in enumerate(rdf.iterrows()):
        y = y_positions[i]
        est = row["pooled_mortality"]
        lo = row["ci_lower"]
        hi = row["ci_upper"]

        ax.errorbar(
            est, y,
            xerr=[[max(0, est - lo)], [max(0, hi - est)]],
            fmt="s", color="#333333",
            markersize=6, capsize=3, linewidth=1.0, zorder=3,
        )
        label_text = f"{row['subgroup']} (k={row['k']}, I2={row['I2']:.0f}%)"
        ax.text(
            ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -1, y,
            label_text, va="center", ha="right", fontsize=8,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{r['subgroup']} (k={r['k']})" for r in results],
        fontsize=8,
    )
    ax.set_xlabel("Pooled mortality (%)")
    ax.set_title(
        f"Subgroup analysis: {strat_label} -- {insecticide_class}",
        fontweight="bold",
    )
    ax.set_ylim(0, k + 1)
    ax.axvline(90, color="orange", linestyle="--", linewidth=0.5, label="WHO possible resistance")
    ax.axvline(98, color="green", linestyle="--", linewidth=0.5, label="WHO susceptible")
    ax.legend(fontsize=7, loc="lower right")

    safe_strat = strat_label.replace(" ", "_").lower()
    safe_class = insecticide_class.replace(" ", "_").replace("/", "-")
    fp = forest_dir / f"subgroup_{safe_strat}_{safe_class}.{FIGURE_FORMAT}"
    plt.tight_layout()
    plt.savefig(fp, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"    Forest plot saved: {fp}")


# ──────────────────────────────────────────────────────────────────────
# Grouped bar plot comparing subgroup estimates
# ──────────────────────────────────────────────────────────────────────

def generate_grouped_bar_plot(
    all_results: list[dict],
    strat_label: str,
    figures_dir: Path,
) -> None:
    """Generate grouped bar plot comparing pooled estimates across subgroups
    for different insecticide classes."""
    rdf = pd.DataFrame(all_results)
    if rdf.empty:
        return

    rdf_strat = rdf[rdf["stratification"] == strat_label].copy()
    if rdf_strat.empty:
        return

    classes = rdf_strat["insecticide_class"].unique()
    subgroups = rdf_strat["subgroup"].unique()
    n_classes = len(classes)
    n_subgroups = len(subgroups)

    if n_subgroups == 0 or n_classes == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, n_subgroups * n_classes * 0.5), 6))
    width = 0.8 / n_classes
    x = np.arange(n_subgroups)

    for i, cls in enumerate(classes):
        cls_data = rdf_strat[rdf_strat["insecticide_class"] == cls]
        vals = []
        errs_lo = []
        errs_hi = []
        for sg in subgroups:
            row = cls_data[cls_data["subgroup"] == sg]
            if not row.empty:
                v = row.iloc[0]["pooled_mortality"]
                lo = row.iloc[0]["ci_lower"]
                hi = row.iloc[0]["ci_upper"]
                vals.append(v)
                errs_lo.append(v - lo)
                errs_hi.append(hi - v)
            else:
                vals.append(0)
                errs_lo.append(0)
                errs_hi.append(0)

        color = INSECTICIDE_CLASS_COLORS.get(cls, "#999999")
        errs_lo = [max(0, e) for e in errs_lo]
        errs_hi = [max(0, e) for e in errs_hi]
        ax.bar(
            x + i * width,
            vals,
            width,
            yerr=[errs_lo, errs_hi],
            label=cls,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            capsize=2,
        )

    ax.set_xlabel(strat_label)
    ax.set_ylabel("Pooled mortality (%)")
    ax.set_title(f"Subgroup comparison: {strat_label}", fontweight="bold")
    ax.set_xticks(x + width * (n_classes - 1) / 2)
    ax.set_xticklabels(subgroups, rotation=45, ha="right", fontsize=8)
    ax.axhline(90, color="orange", linestyle="--", linewidth=0.5)
    ax.axhline(98, color="green", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=7, loc="best")
    ax.set_ylim(0, 105)

    safe_strat = strat_label.replace(" ", "_").lower()
    fp = figures_dir / f"subgroup_barplot_{safe_strat}.{FIGURE_FORMAT}"
    _ensure_dir(fp)
    plt.tight_layout()
    plt.savefig(fp, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Bar plot saved: {fp}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("a08 -- Subgroup meta-analyses of mortality data")
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

    # Create derived stratification columns
    if "collection_year_start" in df.columns:
        df["time_period"] = df["collection_year_start"].apply(_assign_time_period)
    else:
        df["time_period"] = np.nan

    if "species" in df.columns:
        df["species_group"] = df["species"].apply(_assign_species_group)
    else:
        df["species_group"] = np.nan

    if "bioassay_method" in df.columns:
        df["bioassay_std"] = df["bioassay_method"].apply(_standardise_bioassay)
    else:
        df["bioassay_std"] = np.nan

    # Output directories
    forest_dir = FIGURES_DIR / "forest_plots"
    forest_dir.mkdir(parents=True, exist_ok=True)
    supp_dir = TABLES_DIR / "supplementary_tables"
    supp_dir.mkdir(parents=True, exist_ok=True)
    figures_subgroup_dir = FIGURES_DIR / "forest_plots"
    figures_subgroup_dir.mkdir(parents=True, exist_ok=True)

    # Define stratification variables
    stratifications = [
        ("region", "WHO Region"),
        ("time_period", "Time Period"),
        ("species_group", "Species"),
        ("life_stage", "Life Stage"),
        ("bioassay_std", "Bioassay Method"),
    ]

    # Get insecticide classes with enough data
    class_counts = df.groupby("insecticide_class").size()
    eligible_classes = class_counts[class_counts >= MIN_STUDIES_FOR_META].index.tolist()
    print(f"\n  Insecticide classes with >= {MIN_STUDIES_FOR_META} studies: {eligible_classes}\n")

    all_results = []

    for strat_col, strat_label in stratifications:
        if strat_col not in df.columns:
            print(f"  Skipping {strat_label}: column '{strat_col}' not found.")
            continue

        n_non_null = df[strat_col].notna().sum()
        if n_non_null == 0:
            print(f"  Skipping {strat_label}: no non-null values.")
            continue

        print(f"\n  --- Stratification: {strat_label} ({strat_col}) ---")
        print(f"      Non-null values: {n_non_null}")

        for cls in eligible_classes:
            print(f"    Insecticide class: {cls}")
            results = analyse_stratification(
                df, strat_col, strat_label, cls, forest_dir
            )
            all_results.extend(results)
            if results:
                for r in results:
                    print(
                        f"      {r['subgroup']:25s} | k={r['k']:3d} | "
                        f"Pooled={r['pooled_mortality']:6.1f}% "
                        f"[{r['ci_lower']:.1f}, {r['ci_upper']:.1f}] | "
                        f"I2={r['I2']:.1f}%"
                    )
            else:
                print(f"      No subgroups with >= {MIN_STUDIES_FOR_META} studies.")

    if not all_results:
        print("\n[WARNING] No subgroup analyses could be performed.")
        return

    # Generate grouped bar plots for each stratification
    print("\n  Generating grouped bar plots...")
    for _, strat_label in stratifications:
        generate_grouped_bar_plot(all_results, strat_label, FIGURES_DIR / "forest_plots")

    # Save summary table
    summary = pd.DataFrame(all_results)
    # Clean up internal columns
    cols_to_save = [
        "stratification", "subgroup", "insecticide_class", "k",
        "pooled_mortality", "ci_lower", "ci_upper",
        "I2", "tau2", "Q", "Q_p",
        "Q_between", "df_between", "p_interaction",
    ]
    cols_avail = [c for c in cols_to_save if c in summary.columns]
    summary = summary[cols_avail].copy()
    summary["tau2"] = summary["tau2"].round(6)
    summary["Q"] = summary["Q"].round(2)

    out_path = supp_dir / "subgroup_analysis.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Summary table saved: {out_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("SUBGROUP ANALYSIS SUMMARY")
    print("=" * 60)
    for strat in summary["stratification"].unique():
        print(f"\n  {strat}:")
        strat_df = summary[summary["stratification"] == strat]
        for cls in strat_df["insecticide_class"].unique():
            cls_df = strat_df[strat_df["insecticide_class"] == cls]
            print(f"    {cls}:")
            for _, row in cls_df.iterrows():
                pint = row.get("p_interaction", np.nan)
                p_str = f", p_interaction={pint:.4f}" if pd.notna(pint) else ""
                print(
                    f"      {row['subgroup']:25s} | k={row['k']:3d} | "
                    f"Pooled={row['pooled_mortality']:6.1f}% "
                    f"[{row['ci_lower']:.1f}, {row['ci_upper']:.1f}]"
                    f"{p_str}"
                )

    print(f"\n  Total subgroup analyses: {len(summary)}")
    print("Done.")


if __name__ == "__main__":
    main()
