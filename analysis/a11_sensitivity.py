#!/usr/bin/env python3
"""
a11_sensitivity.py -- Sensitivity analyses to check robustness.

For the top 3 most studied insecticide classes, performs:
1. Leave-one-out analysis
2. Cumulative meta-analysis (sorted by year)
3. Method comparison (DL vs REML)
4. Quality restriction (high-quality studies only)

Generates diagnostic plots and saves summary tables.
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


def _ft_transform_rows(sub: pd.DataFrame) -> pd.DataFrame:
    """Apply Freeman-Tukey transformation to mortality data rows.

    Returns DataFrame with yi, vi, n, and a label column.
    """
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

        label = str(row.get("study_id", ""))
        if pd.notna(row.get("country")):
            label = f"{label} ({row['country']})"
        if pd.notna(row.get("year")):
            label = f"{label} {int(row['year'])}"

        year_val = row.get("collection_year_start", row.get("year", np.nan))
        quality = row.get("quality_score", np.nan)

        records.append({
            "label": label.strip(),
            "yi": t,
            "vi": v,
            "n": n,
            "year": year_val,
            "quality_score": quality,
        })
    return pd.DataFrame(records)


def _pooled_mortality(yi, vi, ns):
    """Run DL meta-analysis and back-transform to mortality percentage."""
    res = meta_analysis_dl(yi, vi)
    n_harm = harmonic_mean(ns)
    pooled = back_transform_ft(res["mu"], n_harm)
    ci_lo = back_transform_ft(res["ci_lower"], n_harm)
    ci_hi = back_transform_ft(res["ci_upper"], n_harm)
    return {
        "pooled_mortality": round(pooled * 100, 2),
        "ci_lower": round(ci_lo * 100, 2),
        "ci_upper": round(ci_hi * 100, 2),
        "I2": round(res["I2"], 1),
        "tau2": round(res["tau2"], 6),
        "k": res["k"],
    }


# ──────────────────────────────────────────────────────────────────────
# 1. Leave-one-out analysis
# ──────────────────────────────────────────────────────────────────────

def leave_one_out(fdf: pd.DataFrame, cls: str, forest_dir: Path) -> list[dict]:
    """Remove each study one at a time, recalculate pooled estimate.

    Returns list of results and generates a forest plot.
    For large datasets (k > 100), uses a random sample of 100 studies
    to keep computation and plot size manageable.
    """
    yi = fdf["yi"].values
    vi = fdf["vi"].values
    ns = fdf["n"].values
    labels = fdf["label"].values
    k = len(yi)

    # For large datasets, sample indices to keep LOO tractable
    if k > 100:
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(k, size=100, replace=False))
        print(f"    LOO: k={k} is large, sampling 100 studies for leave-one-out")
    else:
        indices = range(k)

    loo_results = []
    for i in indices:
        mask = np.ones(k, dtype=bool)
        mask[i] = False
        yi_loo = yi[mask]
        vi_loo = vi[mask]
        ns_loo = ns[mask]

        if len(yi_loo) < 2:
            continue

        res = _pooled_mortality(yi_loo, vi_loo, ns_loo)
        res["excluded_study"] = labels[i]
        res["analysis_type"] = "leave_one_out"
        res["insecticide_class"] = cls
        loo_results.append(res)

    # Generate leave-one-out forest plot (skip if too many studies)
    if loo_results and len(loo_results) <= 50:
        _plot_loo_forest(loo_results, cls, forest_dir)
    elif loo_results:
        print(f"    Skipping LOO forest plot for {cls} (k={len(loo_results)} > 50)")

    return loo_results


def _plot_loo_forest(results: list[dict], cls: str, forest_dir: Path) -> None:
    """Generate a leave-one-out forest plot."""
    rdf = pd.DataFrame(results)
    k = len(rdf)

    fig, ax = plt.subplots(figsize=(10, max(4, k * 0.35 + 2)))
    y_positions = np.arange(k, 0, -1)

    for i, (_, row) in enumerate(rdf.iterrows()):
        y = y_positions[i]
        est = row["pooled_mortality"]
        lo = row["ci_lower"]
        hi = row["ci_upper"]

        ax.errorbar(
            est, y,
            xerr=[[est - lo], [hi - est]],
            fmt="s", color="#333333",
            markersize=4, capsize=2, linewidth=0.8, zorder=3,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"Excl: {r['excluded_study']}" for r in results],
        fontsize=7,
    )
    ax.set_xlabel("Pooled mortality (%)")
    ax.set_title(f"Leave-one-out analysis -- {cls}", fontweight="bold")
    ax.set_ylim(0, k + 1)

    # Add reference line for overall estimate
    overall = _pooled_mortality(
        pd.DataFrame(results)["pooled_mortality"].values / 100,  # not used for line
        np.ones(len(results)),
        np.ones(len(results)),
    )
    # Instead use the mean of leave-one-out estimates as a rough guide
    mean_est = rdf["pooled_mortality"].mean()
    ax.axvline(mean_est, color="#0072B2", linestyle="--", linewidth=0.8,
               label=f"Mean LOO est: {mean_est:.1f}%")
    ax.legend(fontsize=7, loc="lower right")

    safe_name = cls.replace("/", "-").replace(" ", "_")
    fp = forest_dir / f"sensitivity_loo_{safe_name}.{FIGURE_FORMAT}"
    _ensure_dir(fp)
    plt.tight_layout()
    plt.savefig(fp, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"    LOO forest plot saved: {fp}")


# ──────────────────────────────────────────────────────────────────────
# 2. Cumulative meta-analysis
# ──────────────────────────────────────────────────────────────────────

def cumulative_meta_analysis(fdf: pd.DataFrame, cls: str, trend_dir: Path) -> list[dict]:
    """Sort studies by year, progressively add and recalculate pooled estimate.

    Returns list of results and generates a cumulative plot.
    """
    # Sort by year
    sorted_df = fdf.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)
    if len(sorted_df) < MIN_STUDIES_FOR_META:
        return []

    yi = sorted_df["yi"].values
    vi = sorted_df["vi"].values
    ns = sorted_df["n"].values
    years = sorted_df["year"].values

    cum_results = []
    for i in range(1, len(yi) + 1):
        if i < 2:
            continue  # need at least 2 studies
        yi_cum = yi[:i]
        vi_cum = vi[:i]
        ns_cum = ns[:i]

        res = _pooled_mortality(yi_cum, vi_cum, ns_cum)
        res["n_studies_cumulative"] = i
        res["up_to_year"] = int(years[i - 1])
        res["analysis_type"] = "cumulative"
        res["insecticide_class"] = cls
        cum_results.append(res)

    # Generate cumulative plot
    if cum_results:
        _plot_cumulative(cum_results, cls, trend_dir)

    return cum_results


def _plot_cumulative(results: list[dict], cls: str, trend_dir: Path) -> None:
    """Generate a cumulative meta-analysis plot."""
    rdf = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        rdf["up_to_year"], rdf["pooled_mortality"],
        "o-", color="#0072B2", markersize=4, linewidth=1.5, zorder=3,
    )
    ax.fill_between(
        rdf["up_to_year"],
        rdf["ci_lower"],
        rdf["ci_upper"],
        alpha=0.2, color="#0072B2",
    )

    ax.set_xlabel("Year of data collection")
    ax.set_ylabel("Cumulative pooled mortality (%)")
    ax.set_title(f"Cumulative meta-analysis -- {cls}", fontweight="bold")
    ax.axhline(90, color="orange", linestyle="--", linewidth=0.5, label="WHO possible resistance")
    ax.axhline(98, color="green", linestyle="--", linewidth=0.5, label="WHO susceptible")
    ax.legend(fontsize=7, loc="best")

    # Add secondary axis for number of studies
    ax2 = ax.twinx()
    ax2.bar(
        rdf["up_to_year"], rdf["n_studies_cumulative"],
        alpha=0.15, color="grey", width=0.8,
    )
    ax2.set_ylabel("Cumulative number of studies", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")

    safe_name = cls.replace("/", "-").replace(" ", "_")
    fp = trend_dir / f"cumulative_{safe_name}.{FIGURE_FORMAT}"
    _ensure_dir(fp)
    plt.tight_layout()
    plt.savefig(fp, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"    Cumulative plot saved: {fp}")


# ──────────────────────────────────────────────────────────────────────
# 3. Method comparison (DL vs REML)
# ──────────────────────────────────────────────────────────────────────

def method_comparison(fdf: pd.DataFrame, cls: str) -> dict:
    """Compare DerSimonian-Laird vs REML estimators."""
    yi = fdf["yi"].values
    vi = fdf["vi"].values
    ns = fdf["n"].values

    if len(yi) < MIN_STUDIES_FOR_META:
        return None

    n_harm = harmonic_mean(ns)

    # DL
    dl_res = meta_analysis_dl(yi, vi)
    dl_pooled = back_transform_ft(dl_res["mu"], n_harm)
    dl_ci_lo = back_transform_ft(dl_res["ci_lower"], n_harm)
    dl_ci_hi = back_transform_ft(dl_res["ci_upper"], n_harm)

    # REML
    reml_res = meta_analysis_reml(yi, vi)
    reml_pooled = back_transform_ft(reml_res["mu"], n_harm)
    reml_ci_lo = back_transform_ft(reml_res["ci_lower"], n_harm)
    reml_ci_hi = back_transform_ft(reml_res["ci_upper"], n_harm)

    return {
        "insecticide_class": cls,
        "analysis_type": "method_comparison",
        "k": len(yi),
        "DL_pooled_mortality": round(dl_pooled * 100, 2),
        "DL_ci_lower": round(dl_ci_lo * 100, 2),
        "DL_ci_upper": round(dl_ci_hi * 100, 2),
        "DL_tau2": round(dl_res["tau2"], 6),
        "DL_I2": round(dl_res["I2"], 1),
        "REML_pooled_mortality": round(reml_pooled * 100, 2),
        "REML_ci_lower": round(reml_ci_lo * 100, 2),
        "REML_ci_upper": round(reml_ci_hi * 100, 2),
        "REML_tau2": round(reml_res["tau2"], 6),
        "REML_I2": round(reml_res["I2"], 1),
        "difference_pct": round(abs(dl_pooled - reml_pooled) * 100, 2),
    }


# ──────────────────────────────────────────────────────────────────────
# 4. Quality restriction
# ──────────────────────────────────────────────────────────────────────

def quality_restriction(fdf: pd.DataFrame, cls: str, threshold: float = 7.0) -> dict:
    """Re-run meta-analysis restricted to high-quality studies."""
    if "quality_score" not in fdf.columns:
        return None

    high_q = fdf[fdf["quality_score"] >= threshold].copy()
    all_data = fdf.copy()

    if len(high_q) < MIN_STUDIES_FOR_META:
        return None

    yi_all = all_data["yi"].values
    vi_all = all_data["vi"].values
    ns_all = all_data["n"].values
    all_res = _pooled_mortality(yi_all, vi_all, ns_all)

    yi_hq = high_q["yi"].values
    vi_hq = high_q["vi"].values
    ns_hq = high_q["n"].values
    hq_res = _pooled_mortality(yi_hq, vi_hq, ns_hq)

    return {
        "insecticide_class": cls,
        "analysis_type": "quality_restriction",
        "k_all": len(yi_all),
        "k_high_quality": len(yi_hq),
        "quality_threshold": threshold,
        "all_pooled_mortality": all_res["pooled_mortality"],
        "all_ci_lower": all_res["ci_lower"],
        "all_ci_upper": all_res["ci_upper"],
        "hq_pooled_mortality": hq_res["pooled_mortality"],
        "hq_ci_lower": hq_res["ci_lower"],
        "hq_ci_upper": hq_res["ci_upper"],
        "difference_pct": round(abs(all_res["pooled_mortality"] - hq_res["pooled_mortality"]), 2),
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("a11 -- Sensitivity analyses")
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

    # Identify top 3 most studied insecticide classes
    class_counts = df.groupby("insecticide_class").size().sort_values(ascending=False)
    top_classes = class_counts.head(3).index.tolist()
    print(f"\n  Top 3 insecticide classes: {top_classes}")
    for cls in top_classes:
        print(f"    {cls}: {class_counts[cls]} studies")

    # Output directories
    forest_dir = FIGURES_DIR / "forest_plots"
    forest_dir.mkdir(parents=True, exist_ok=True)
    trend_dir = FIGURES_DIR / "temporal_trends"
    trend_dir.mkdir(parents=True, exist_ok=True)
    supp_dir = TABLES_DIR / "supplementary_tables"
    supp_dir.mkdir(parents=True, exist_ok=True)

    all_loo = []
    all_cum = []
    all_method = []
    all_quality = []

    for cls in top_classes:
        print(f"\n  {'=' * 50}")
        print(f"  Insecticide class: {cls}")
        print(f"  {'=' * 50}")

        sub = df[df["insecticide_class"] == cls].copy()
        fdf = _ft_transform_rows(sub)
        if fdf.empty or len(fdf) < MIN_STUDIES_FOR_META:
            print(f"  Not enough valid studies ({len(fdf)}) -- skipping.")
            continue

        print(f"  {len(fdf)} valid effect sizes")

        # 1. Leave-one-out
        print(f"\n  [1] Leave-one-out analysis...")
        loo = leave_one_out(fdf, cls, forest_dir)
        all_loo.extend(loo)
        if loo:
            loo_df = pd.DataFrame(loo)
            range_est = loo_df["pooled_mortality"].max() - loo_df["pooled_mortality"].min()
            print(f"    Range of pooled estimates: {loo_df['pooled_mortality'].min():.1f}% -- "
                  f"{loo_df['pooled_mortality'].max():.1f}% (spread: {range_est:.1f}%)")

            # Identify influential studies (those that change the estimate most)
            overall = _pooled_mortality(fdf["yi"].values, fdf["vi"].values, fdf["n"].values)
            overall_est = overall["pooled_mortality"]
            loo_df["deviation"] = abs(loo_df["pooled_mortality"] - overall_est)
            most_influential = loo_df.nlargest(3, "deviation")
            print(f"    Most influential studies (largest deviation from overall {overall_est:.1f}%):")
            for _, r in most_influential.iterrows():
                print(f"      {r['excluded_study']}: {r['pooled_mortality']:.1f}% "
                      f"(deviation: {r['deviation']:.1f}%)")
        else:
            print(f"    No leave-one-out results.")

        # 2. Cumulative meta-analysis
        print(f"\n  [2] Cumulative meta-analysis...")
        cum = cumulative_meta_analysis(fdf, cls, trend_dir)
        all_cum.extend(cum)
        if cum:
            print(f"    {len(cum)} cumulative steps computed.")
        else:
            print(f"    Could not perform cumulative analysis (missing year data).")

        # 3. Method comparison (DL vs REML)
        print(f"\n  [3] Method comparison (DL vs REML)...")
        mc = method_comparison(fdf, cls)
        if mc:
            all_method.append(mc)
            print(f"    DL:   {mc['DL_pooled_mortality']:.1f}% "
                  f"[{mc['DL_ci_lower']:.1f}, {mc['DL_ci_upper']:.1f}] "
                  f"(tau2={mc['DL_tau2']:.4f})")
            print(f"    REML: {mc['REML_pooled_mortality']:.1f}% "
                  f"[{mc['REML_ci_lower']:.1f}, {mc['REML_ci_upper']:.1f}] "
                  f"(tau2={mc['REML_tau2']:.4f})")
            print(f"    Difference: {mc['difference_pct']:.2f} percentage points")
        else:
            print(f"    Could not perform method comparison.")

        # 4. Quality restriction
        print(f"\n  [4] Quality restriction (score >= 7)...")
        qr = quality_restriction(fdf, cls, threshold=7.0)
        if qr:
            all_quality.append(qr)
            print(f"    All studies (k={qr['k_all']}): "
                  f"{qr['all_pooled_mortality']:.1f}% "
                  f"[{qr['all_ci_lower']:.1f}, {qr['all_ci_upper']:.1f}]")
            print(f"    High quality (k={qr['k_high_quality']}): "
                  f"{qr['hq_pooled_mortality']:.1f}% "
                  f"[{qr['hq_ci_lower']:.1f}, {qr['hq_ci_upper']:.1f}]")
            print(f"    Difference: {qr['difference_pct']:.2f} percentage points")
        else:
            has_qs = "quality_score" in fdf.columns and fdf["quality_score"].notna().any()
            if not has_qs:
                print(f"    Quality score not available -- skipping.")
            else:
                print(f"    Not enough high-quality studies -- skipping.")

    # ──────────────────────────────────────────────────────────────
    # Save combined sensitivity analysis table
    # ──────────────────────────────────────────────────────────────
    sensitivity_rows = []

    # LOO summary (per class)
    for cls in top_classes:
        loo_cls = [r for r in all_loo if r.get("insecticide_class") == cls]
        if loo_cls:
            loo_df = pd.DataFrame(loo_cls)
            sensitivity_rows.append({
                "insecticide_class": cls,
                "analysis": "Leave-one-out",
                "detail": f"Range: {loo_df['pooled_mortality'].min():.1f}% -- {loo_df['pooled_mortality'].max():.1f}%",
                "k": len(loo_cls),
                "estimate": f"{loo_df['pooled_mortality'].mean():.1f}%",
                "conclusion": "Robust" if (loo_df['pooled_mortality'].max() - loo_df['pooled_mortality'].min()) < 5 else "Sensitive to individual studies",
            })

    # Method comparison
    for mc in all_method:
        sensitivity_rows.append({
            "insecticide_class": mc["insecticide_class"],
            "analysis": "DL vs REML",
            "detail": f"DL: {mc['DL_pooled_mortality']:.1f}%, REML: {mc['REML_pooled_mortality']:.1f}%",
            "k": mc["k"],
            "estimate": f"Diff: {mc['difference_pct']:.2f}pp",
            "conclusion": "Consistent" if mc["difference_pct"] < 2 else "Methods diverge",
        })

    # Quality restriction
    for qr in all_quality:
        sensitivity_rows.append({
            "insecticide_class": qr["insecticide_class"],
            "analysis": "Quality restriction",
            "detail": f"All: {qr['all_pooled_mortality']:.1f}%, HQ: {qr['hq_pooled_mortality']:.1f}%",
            "k": f"{qr['k_all']} / {qr['k_high_quality']}",
            "estimate": f"Diff: {qr['difference_pct']:.2f}pp",
            "conclusion": "Robust" if qr["difference_pct"] < 5 else "Quality-dependent",
        })

    if sensitivity_rows:
        sens_df = pd.DataFrame(sensitivity_rows)
        out_path = supp_dir / "sensitivity_analysis.csv"
        sens_df.to_csv(out_path, index=False)
        print(f"\n  Sensitivity analysis table saved: {out_path}")
    else:
        print("\n  No sensitivity results to save.")

    # Console summary
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 60)
    if sensitivity_rows:
        for r in sensitivity_rows:
            print(
                f"  {r['insecticide_class']:20s} | {r['analysis']:20s} | "
                f"{r['detail']:45s} | {r['conclusion']}"
            )
    else:
        print("  No analyses completed.")
    print("Done.")


if __name__ == "__main__":
    main()
