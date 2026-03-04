#!/usr/bin/env python3
"""
a09_meta_regression.py -- Meta-regression to identify moderators of resistance.

Applies Freeman-Tukey transformation to mortality data, then fits weighted
least squares (WLS) meta-regression models with various moderators.
Generates bubble plots for continuous moderators and saves coefficient tables.
"""

import sys
import warnings
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


def prepare_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Freeman-Tukey transformation to each row and return
    a DataFrame with yi, vi, n columns appended.
    """
    records = []
    for idx, row in df.iterrows():
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
        records.append({"_orig_idx": idx, "yi": t, "vi": v, "n": n})

    if not records:
        return pd.DataFrame()

    es_df = pd.DataFrame(records).set_index("_orig_idx")
    result = df.join(es_df, how="inner")
    return result


def estimate_tau2_intercept(yi: np.ndarray, vi: np.ndarray) -> float:
    """Estimate tau2 from an intercept-only DL model."""
    res = meta_analysis_dl(yi, vi)
    return res["tau2"]


def run_wls_regression(
    data: pd.DataFrame,
    formula_vars: list[str],
    tau2: float,
    moderator_name: str,
) -> dict:
    """Fit a WLS meta-regression model.

    Parameters
    ----------
    data : DataFrame with yi, vi columns and moderator columns
    formula_vars : list of column names to use as predictors
    tau2 : between-study variance from intercept-only model
    moderator_name : label for the moderator

    Returns
    -------
    dict with coefficients, p-values, R2, etc. or None on failure
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  [ERROR] statsmodels not installed. Cannot run meta-regression.")
        return None

    # Prepare design matrix
    sub = data.dropna(subset=["yi", "vi"] + formula_vars).copy()
    if len(sub) < 5:
        return None

    yi = sub["yi"].values
    vi = sub["vi"].values

    # Weights = 1 / (vi + tau2)
    weights = 1.0 / (vi + tau2)

    # Build X matrix
    X = sub[formula_vars].copy()

    # Check if any column is categorical (object/string type)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)

    # Add constant
    X = sm.add_constant(X, has_constant="add")

    # Ensure all numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1)
    X = X[mask]
    yi_clean = yi[mask.values]
    weights_clean = weights[mask.values]

    if len(X) < X.shape[1] + 1:
        return None

    try:
        model = sm.WLS(yi_clean, X, weights=weights_clean)
        results = model.fit()
    except Exception as e:
        print(f"    WLS failed for {moderator_name}: {e}")
        return None

    # Compute pseudo R2 (proportion of tau2 explained)
    # R2 = 1 - tau2_model / tau2_intercept
    # Approximate tau2_model from residual Q
    resid = results.resid
    Q_resid = np.sum(weights_clean * resid ** 2)
    k = len(yi_clean)
    p = X.shape[1]
    C = np.sum(weights_clean) - np.sum(weights_clean ** 2) / np.sum(weights_clean)
    tau2_model = max(0, (Q_resid - (k - p)) / C) if C > 0 else 0
    R2 = max(0, 1 - tau2_model / tau2) if tau2 > 0 else 0.0

    # Extract coefficients
    coefs = []
    for i, name in enumerate(X.columns):
        coefs.append({
            "moderator": moderator_name,
            "term": name,
            "coefficient": np.round(float(results.params.iloc[i]), 6),
            "se": np.round(float(results.bse.iloc[i]), 6),
            "z_value": np.round(float(results.tvalues.iloc[i]), 3),
            "p_value": np.round(float(results.pvalues.iloc[i]), 4),
        })

    return {
        "moderator": moderator_name,
        "coefficients": coefs,
        "R2": np.round(float(R2 * 100), 1),
        "tau2_residual": np.round(float(tau2_model), 6),
        "k": k,
        "omnibus_p": np.round(float(results.f_pvalue), 4) if hasattr(results, "f_pvalue") and results.f_pvalue is not None else np.nan,
        "results_obj": results,
    }


# ──────────────────────────────────────────────────────────────────────
# Bubble plot for continuous moderators
# ──────────────────────────────────────────────────────────────────────

def bubble_plot(
    data: pd.DataFrame,
    moderator_col: str,
    moderator_label: str,
    reg_result: dict,
    filepath: Path,
) -> None:
    """Generate a bubble plot for a continuous moderator."""
    sub = data.dropna(subset=["yi", "vi", moderator_col]).copy()
    if sub.empty:
        return

    x = sub[moderator_col].values
    y = sub["yi"].values
    vi = sub["vi"].values
    sizes = 1.0 / vi
    # Normalise bubble sizes
    sizes = sizes / sizes.max() * 300

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        x, y,
        s=sizes,
        alpha=0.5,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )

    # Add regression line if available
    coefs = reg_result.get("coefficients", [])
    if len(coefs) >= 2:
        intercept = coefs[0]["coefficient"]
        slope = coefs[1]["coefficient"]
        x_range = np.linspace(x.min(), x.max(), 100)
        y_pred = intercept + slope * x_range
        ax.plot(x_range, y_pred, color="#D55E00", linewidth=2, zorder=4)

        # Add annotation
        p_val = coefs[1]["p_value"]
        R2 = reg_result.get("R2", 0)
        ax.text(
            0.02, 0.98,
            f"slope = {slope:.4f} (p = {p_val:.3f})\nR2 = {R2:.1f}%",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel(moderator_label)
    ax.set_ylabel("Freeman-Tukey transformed mortality")
    ax.set_title(f"Meta-regression: {moderator_label}", fontweight="bold")

    _ensure_dir(filepath)
    plt.tight_layout()
    plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"    Bubble plot saved: {filepath}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("a09 -- Meta-regression to identify moderators of resistance")
    print("=" * 60)

    # Check statsmodels availability
    try:
        import statsmodels.api as sm
    except ImportError:
        print("[ERROR] statsmodels is required but not installed.")
        print("  Install with: pip install statsmodels")
        return

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
    required = ["n_tested", "mortality_pct"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing columns: {missing_cols}")
        return

    # Compute n_dead if absent
    if "n_dead" not in df.columns:
        df["n_dead"] = (df["mortality_pct"] / 100 * df["n_tested"]).round()

    # Apply Freeman-Tukey transformation
    print("\n  Applying Freeman-Tukey transformation...")
    df = prepare_effect_sizes(df)
    if df.empty:
        print("[WARNING] No valid effect sizes could be computed.")
        return
    print(f"  Effect sizes computed for {len(df)} rows.")

    # Estimate tau2 from intercept-only model
    yi_all = df["yi"].values
    vi_all = df["vi"].values
    tau2 = estimate_tau2_intercept(yi_all, vi_all)
    print(f"  Intercept-only tau2 = {tau2:.6f}")

    # Output directories
    fig_dir = FIGURES_DIR
    supp_dir = TABLES_DIR / "supplementary_tables"
    supp_dir.mkdir(parents=True, exist_ok=True)

    # Define moderators
    moderators = [
        ("collection_year_start", "Collection Year", "continuous"),
        ("region", "WHO Region", "categorical"),
        ("insecticide_class", "Insecticide Class", "categorical"),
        ("species", "Species", "categorical"),
        ("life_stage", "Life Stage", "categorical"),
    ]

    all_coefs = []
    significant_moderators = []
    univariate_results = {}

    print("\n  --- Univariate meta-regression ---\n")

    for mod_col, mod_label, mod_type in moderators:
        if mod_col not in df.columns:
            print(f"  Skipping {mod_label}: column '{mod_col}' not found.")
            continue

        n_valid = df[mod_col].notna().sum()
        if n_valid < 5:
            print(f"  Skipping {mod_label}: only {n_valid} non-null values.")
            continue

        print(f"  Moderator: {mod_label} ({mod_type}, n={n_valid})")

        result = run_wls_regression(
            data=df,
            formula_vars=[mod_col],
            tau2=tau2,
            moderator_name=mod_label,
        )

        if result is None:
            print(f"    Could not fit model.")
            continue

        univariate_results[mod_col] = result
        all_coefs.extend(result["coefficients"])

        # Print results
        print(f"    R2 = {result['R2']:.1f}%, k = {result['k']}")
        for c in result["coefficients"]:
            sig_marker = "*" if c["p_value"] < 0.10 else ""
            print(
                f"    {c['term']:35s} | "
                f"coef={c['coefficient']:+.4f} | "
                f"SE={c['se']:.4f} | "
                f"p={c['p_value']:.4f}{sig_marker}"
            )

        # Check significance (p < 0.10 for any non-intercept term)
        non_intercept = [c for c in result["coefficients"] if c["term"] != "const"]
        if any(c["p_value"] < 0.10 for c in non_intercept):
            significant_moderators.append((mod_col, mod_label, mod_type))
            print(f"    ** Significant at p < 0.10 **")

        # Bubble plot for continuous moderators
        if mod_type == "continuous":
            bp_path = fig_dir / f"meta_regression_{mod_col}.{FIGURE_FORMAT}"
            bubble_plot(df, mod_col, mod_label, result, bp_path)

    # Multivariate model with significant moderators
    if len(significant_moderators) >= 2:
        print(f"\n  --- Multivariate meta-regression ---")
        print(f"  Significant moderators: {[m[1] for m in significant_moderators]}")

        multi_vars = [m[0] for m in significant_moderators]
        multi_result = run_wls_regression(
            data=df,
            formula_vars=multi_vars,
            tau2=tau2,
            moderator_name="Multivariate",
        )

        if multi_result is not None:
            all_coefs.extend(multi_result["coefficients"])
            print(f"  R2 = {multi_result['R2']:.1f}%, k = {multi_result['k']}")
            for c in multi_result["coefficients"]:
                sig_marker = "*" if c["p_value"] < 0.10 else ""
                print(
                    f"    {c['term']:35s} | "
                    f"coef={c['coefficient']:+.4f} | "
                    f"SE={c['se']:.4f} | "
                    f"p={c['p_value']:.4f}{sig_marker}"
                )
        else:
            print("  Multivariate model could not be fitted.")
    elif significant_moderators:
        print(f"\n  Only one significant moderator ({significant_moderators[0][1]}) -- "
              f"multivariate model not needed.")
    else:
        print("\n  No significant moderators (p < 0.10) -- "
              "multivariate model not fitted.")

    # Save coefficients table
    if all_coefs:
        coef_df = pd.DataFrame(all_coefs)
        out_path = supp_dir / "meta_regression.csv"
        coef_df.to_csv(out_path, index=False)
        print(f"\n  Coefficients table saved: {out_path}")
    else:
        print("\n  No coefficients to save.")

    # Console summary
    print("\n" + "=" * 60)
    print("META-REGRESSION SUMMARY")
    print("=" * 60)
    print(f"  Intercept-only tau2: {tau2:.6f}")
    print(f"  Total observations: {len(df)}")
    print(f"  Moderators tested: {len(moderators)}")
    print(f"  Significant (p<0.10): {len(significant_moderators)}")
    if significant_moderators:
        for _, label, _ in significant_moderators:
            r = [v for k, v in univariate_results.items()
                 if any(m[0] == k for m in significant_moderators if m[1] == label)]
            if r:
                print(f"    - {label}: R2 = {r[0]['R2']:.1f}%")
            else:
                print(f"    - {label}")
    print("Done.")


if __name__ == "__main__":
    main()
