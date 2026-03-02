"""
Shared utility functions for insecticide resistance meta-analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Effect-size transformations
# ──────────────────────────────────────────────────────────────────────

def freeman_tukey_double_arcsine(x: int, n: int):
    """Freeman-Tukey double arcsine transformation for proportions.

    Returns transformed value *t* and its sampling variance *v*.
    Suitable for mortality rates and kdr allele frequencies.
    """
    t = np.arcsin(np.sqrt(x / (n + 1))) + np.arcsin(np.sqrt((x + 1) / (n + 1)))
    v = 1.0 / (n + 1)
    return t, v


def back_transform_ft(t: float, n_harmonic: float):
    """Back-transform a pooled Freeman-Tukey estimate to a proportion.

    Parameters
    ----------
    t : pooled transformed estimate
    n_harmonic : harmonic mean of study sample sizes
    """
    z = 0.5 * (1 - np.sign(np.cos(t)) * np.sqrt(1 - (np.sin(t) + (np.sin(t) - 1.0 / np.sin(t)) / n_harmonic) ** 2))
    return np.clip(z, 0.0, 1.0)


def logit_transform(p: float, n: int, cc: float = 0.5):
    """Logit transformation for proportions.

    Parameters
    ----------
    p : proportion (0–1)
    n : sample size
    cc : continuity correction added when p == 0 or p == 1
    """
    if p <= 0:
        p = cc / n
    elif p >= 1:
        p = 1 - cc / n
    yi = np.log(p / (1 - p))
    vi = 1.0 / (n * p) + 1.0 / (n * (1 - p))
    return yi, vi


def log_rr_transform(rr: float, ci_lower: float, ci_upper: float):
    """Log-transform a resistance ratio and compute SE from CIs.

    Returns ln(RR) and its standard error.
    """
    yi = np.log(rr)
    se = (np.log(ci_upper) - np.log(ci_lower)) / (2 * 1.96)
    vi = se ** 2
    return yi, vi


def hedges_g(m1, sd1, n1, m2, sd2, n2):
    """Compute Hedges' g (bias-corrected standardised mean difference).

    Parameters
    ----------
    m1, sd1, n1 : mean, SD, n of group 1 (field)
    m2, sd2, n2 : mean, SD, n of group 2 (reference)
    """
    sp = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp
    # small-sample correction (Hedges & Olkin)
    df = n1 + n2 - 2
    j = 1 - 3.0 / (4 * df - 1)
    g = d * j
    vg = j ** 2 * (n1 + n2) / (n1 * n2) + g ** 2 / (2 * df)
    return g, vg


# ──────────────────────────────────────────────────────────────────────
# Random-effects meta-analysis (DerSimonian–Laird)
# ──────────────────────────────────────────────────────────────────────

def meta_analysis_dl(yi: np.ndarray, vi: np.ndarray):
    """DerSimonian-Laird random-effects meta-analysis.

    Parameters
    ----------
    yi : array of effect sizes
    vi : array of sampling variances

    Returns
    -------
    dict with keys: mu, se, ci_lower, ci_upper, tau2, I2, Q, Q_p,
                    k (number of studies), weights, prediction_interval
    """
    k = len(yi)
    if k < 2:
        return {
            "mu": yi[0] if k == 1 else np.nan,
            "se": np.sqrt(vi[0]) if k == 1 else np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "tau2": 0.0, "I2": 0.0,
            "Q": 0.0, "Q_p": 1.0, "k": k,
            "weights": np.ones(k) if k else np.array([]),
            "pred_lower": np.nan, "pred_upper": np.nan,
        }

    wi = 1.0 / vi
    mu_fe = np.sum(wi * yi) / np.sum(wi)

    Q = np.sum(wi * (yi - mu_fe) ** 2)
    df = k - 1
    Q_p = 1 - stats.chi2.cdf(Q, df)

    C = np.sum(wi) - np.sum(wi ** 2) / np.sum(wi)
    tau2 = max(0, (Q - df) / C)

    wi_re = 1.0 / (vi + tau2)
    mu = np.sum(wi_re * yi) / np.sum(wi_re)
    se = np.sqrt(1.0 / np.sum(wi_re))

    z = stats.norm.ppf(1 - 0.05 / 2)
    ci_lower = mu - z * se
    ci_upper = mu + z * se

    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0.0

    # Prediction interval
    pred_se = np.sqrt(se ** 2 + tau2)
    t_crit = stats.t.ppf(1 - 0.05 / 2, df=max(k - 2, 1))
    pred_lower = mu - t_crit * pred_se
    pred_upper = mu + t_crit * pred_se

    return {
        "mu": mu, "se": se,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "tau2": tau2, "I2": I2,
        "Q": Q, "Q_p": Q_p, "k": k,
        "weights": wi_re / np.sum(wi_re) * 100,
        "pred_lower": pred_lower, "pred_upper": pred_upper,
    }


def meta_analysis_reml(yi: np.ndarray, vi: np.ndarray, max_iter: int = 100, tol: float = 1e-6):
    """REML (Restricted Maximum Likelihood) random-effects meta-analysis.

    Iterative Fisher scoring to estimate tau2.
    """
    k = len(yi)
    if k < 2:
        return meta_analysis_dl(yi, vi)

    # Start from DL estimate
    dl = meta_analysis_dl(yi, vi)
    tau2 = dl["tau2"]

    for _ in range(max_iter):
        wi = 1.0 / (vi + tau2)
        mu = np.sum(wi * yi) / np.sum(wi)
        # REML update
        P_diag = wi - wi ** 2 / np.sum(wi)
        residuals = yi - mu
        tau2_new = max(0, tau2 + (np.sum(P_diag * residuals ** 2) - np.sum(P_diag * vi)) /
                       np.sum(P_diag ** 2))
        if abs(tau2_new - tau2) < tol:
            tau2 = tau2_new
            break
        tau2 = tau2_new

    wi = 1.0 / (vi + tau2)
    mu = np.sum(wi * yi) / np.sum(wi)
    se = np.sqrt(1.0 / np.sum(wi))

    Q = dl["Q"]
    df = k - 1
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0.0

    z = stats.norm.ppf(1 - 0.05 / 2)
    return {
        "mu": mu, "se": se,
        "ci_lower": mu - z * se, "ci_upper": mu + z * se,
        "tau2": tau2, "I2": I2,
        "Q": Q, "Q_p": dl["Q_p"], "k": k,
        "weights": wi / np.sum(wi) * 100,
        "pred_lower": mu - stats.t.ppf(1 - 0.05 / 2, max(k - 2, 1)) * np.sqrt(se ** 2 + tau2),
        "pred_upper": mu + stats.t.ppf(1 - 0.05 / 2, max(k - 2, 1)) * np.sqrt(se ** 2 + tau2),
    }


# ──────────────────────────────────────────────────────────────────────
# Heterogeneity helpers
# ──────────────────────────────────────────────────────────────────────

def i2_confidence_interval(Q: float, k: int, alpha: float = 0.05):
    """Compute 95 % CI for I² using the test-based method."""
    df = k - 1
    if df <= 0 or Q <= 0:
        return 0.0, 0.0

    B = 0.5 * (np.log(Q) - np.log(df))
    L = np.exp(B - stats.norm.ppf(1 - alpha / 2) * np.sqrt(2.0 / (df - 1)))
    U = np.exp(B + stats.norm.ppf(1 - alpha / 2) * np.sqrt(2.0 / (df - 1)))

    I2_lower = max(0, (L ** 2 * df - df) / (L ** 2 * df)) * 100
    I2_upper = max(0, (U ** 2 * df - df) / (U ** 2 * df)) * 100
    return I2_lower, I2_upper


# ──────────────────────────────────────────────────────────────────────
# Publication bias
# ──────────────────────────────────────────────────────────────────────

def eggers_test(yi: np.ndarray, vi: np.ndarray):
    """Egger's weighted regression test for funnel-plot asymmetry."""
    se = np.sqrt(vi)
    precision = 1.0 / se
    z = yi / se  # standardised effect
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(precision, z)
    return {
        "intercept": intercept,
        "slope": slope,
        "p_value": p_value,
        "r_value": r_value,
    }


def beggs_test(yi: np.ndarray, vi: np.ndarray):
    """Begg & Mazumdar rank correlation test."""
    se = np.sqrt(vi)
    tau, p_value = stats.kendalltau(yi, se)
    return {"tau": tau, "p_value": p_value}


def trim_and_fill(yi: np.ndarray, vi: np.ndarray, side: str = "left"):
    """Simple L0 trim-and-fill estimator (iterative)."""
    yi = yi.copy()
    vi = vi.copy()
    n = len(yi)

    for _ in range(50):
        res = meta_analysis_dl(yi, vi)
        mu = res["mu"]
        d = yi - mu
        ranks = stats.rankdata(np.abs(d))

        if side == "left":
            neg_ranks = ranks[d < 0]
        else:
            neg_ranks = ranks[d > 0]

        if len(neg_ranks) == 0:
            k0 = 0
        else:
            S = np.sum(neg_ranks)
            k0 = max(0, int(round((4 * S - n * (n + 1)) / (2 * n - 1))))

        if k0 == 0:
            break

        # Mirror the most extreme studies
        order = np.argsort(d if side == "left" else -d)
        imputed_yi = 2 * mu - yi[order[:k0]]
        imputed_vi = vi[order[:k0]]
        yi = np.concatenate([yi, imputed_yi])
        vi = np.concatenate([vi, imputed_vi])
        break  # single iteration for L0

    adjusted = meta_analysis_dl(yi, vi)
    adjusted["n_missing"] = k0
    return adjusted


# ──────────────────────────────────────────────────────────────────────
# Forest plot
# ──────────────────────────────────────────────────────────────────────

def forest_plot(
    studies: pd.DataFrame,
    yi_col: str,
    vi_col: str,
    label_col: str,
    title: str,
    filepath: str,
    transform_back=None,
    xlabel: str = "Effect size",
    figsize: tuple = (10, None),
):
    """Generate a publication-quality forest plot.

    Parameters
    ----------
    studies : DataFrame with columns for effect sizes, variances, labels
    yi_col, vi_col, label_col : column names
    title : plot title
    filepath : where to save figure
    transform_back : optional callable to back-transform estimates
    """
    import matplotlib.pyplot as plt

    yi = studies[yi_col].values
    vi = studies[vi_col].values
    labels = studies[label_col].values
    se = np.sqrt(vi)
    k = len(yi)

    res = meta_analysis_dl(yi, vi)

    if figsize[1] is None:
        figsize = (figsize[0], max(4, k * 0.35 + 2))

    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(k, 0, -1)

    # Individual studies
    ci_low = yi - 1.96 * se
    ci_high = yi + 1.96 * se
    ax.errorbar(yi, y_positions, xerr=1.96 * se, fmt="s", color="#333333",
                markersize=4, capsize=2, linewidth=0.8, zorder=3)

    # Summary diamond
    diamond_x = [res["ci_lower"], res["mu"], res["ci_upper"], res["mu"]]
    diamond_y = [-0.5, -1.0, -0.5, 0.0]
    ax.fill(diamond_x, diamond_y, color="#0072B2", alpha=0.7, zorder=4)

    # Reference line at null
    null_val = 0
    ax.axvline(null_val, color="gray", linestyle="--", linewidth=0.5, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(-2, k + 1)

    # Summary text
    I2_str = f"I² = {res['I2']:.1f}%"
    tau2_str = f"τ² = {res['tau2']:.4f}"
    ax.text(0.02, 0.02, f"k = {k}, {I2_str}, {tau2_str}",
            transform=ax.transAxes, fontsize=8, va="bottom")

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Forest plot saved: {filepath}")


# ──────────────────────────────────────────────────────────────────────
# Funnel plot
# ──────────────────────────────────────────────────────────────────────

def funnel_plot(yi: np.ndarray, vi: np.ndarray, title: str, filepath: str):
    """Generate a funnel plot."""
    import matplotlib.pyplot as plt

    se = np.sqrt(vi)
    res = meta_analysis_dl(yi, vi)
    mu = res["mu"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(yi, se, s=20, color="#333333", alpha=0.6, zorder=3)

    # Reference line at pooled estimate
    ax.axvline(mu, color="#0072B2", linewidth=1, zorder=2)

    # Pseudo-CI contours
    se_range = np.linspace(0.001, max(se) * 1.1, 100)
    ax.plot(mu - 1.96 * se_range, se_range, "k--", linewidth=0.5)
    ax.plot(mu + 1.96 * se_range, se_range, "k--", linewidth=0.5)

    ax.invert_yaxis()
    ax.set_xlabel("Effect size")
    ax.set_ylabel("Standard error")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Funnel plot saved: {filepath}")


# ──────────────────────────────────────────────────────────────────────
# Abbott's correction
# ──────────────────────────────────────────────────────────────────────

def abbotts_correction(test_mortality: float, control_mortality: float) -> float:
    """Apply Abbott's correction for control mortality between 5-20%.

    Parameters
    ----------
    test_mortality : observed mortality in test group (0-100)
    control_mortality : observed mortality in control group (0-100)

    Returns
    -------
    corrected mortality percentage
    """
    if control_mortality >= 100:
        return np.nan
    corrected = (test_mortality - control_mortality) / (100 - control_mortality) * 100
    return np.clip(corrected, 0, 100)


# ──────────────────────────────────────────────────────────────────────
# WHO resistance classification
# ──────────────────────────────────────────────────────────────────────

def classify_resistance(mortality_pct: float) -> str:
    """Classify resistance status based on WHO criteria."""
    if mortality_pct >= 98:
        return "Susceptible"
    elif mortality_pct >= 90:
        return "Possible resistance"
    else:
        return "Confirmed resistance"


# ──────────────────────────────────────────────────────────────────────
# Harmonic mean helper
# ──────────────────────────────────────────────────────────────────────

def harmonic_mean(x: np.ndarray) -> float:
    """Compute the harmonic mean (used for FT back-transform)."""
    x = x[x > 0]
    if len(x) == 0:
        return np.nan
    return len(x) / np.sum(1.0 / x)


# ──────────────────────────────────────────────────────────────────────
# Fisher's z transformation for correlations
# ──────────────────────────────────────────────────────────────────────

def fisher_z(r: float) -> float:
    """Fisher's z transformation of a correlation coefficient."""
    r = np.clip(r, -0.999, 0.999)
    return 0.5 * np.log((1 + r) / (1 - r))


def fisher_z_inv(z: float) -> float:
    """Inverse Fisher z transformation."""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def fisher_z_var(n: int) -> float:
    """Variance of Fisher's z."""
    return 1.0 / (n - 3) if n > 3 else np.inf
