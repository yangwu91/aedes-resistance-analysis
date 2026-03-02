#!/usr/bin/env python3
"""
a01_data_cleaning.py – Data cleaning and preprocessing.

Reads the raw extracted data, standardises names and values,
validates ranges, computes derived variables, and saves
indicator-specific processed datasets.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_RAW, DATA_PROCESSED,
    INSECTICIDE_NAME_MAP, INSECTICIDE_CLASS_MAP,
    WHO_REGION_MAP, KDR_MUTATION_MAP,
    MIN_SAMPLE_SIZE, CONTINUITY_CORRECTION,
    CONTROL_MORTALITY_MAX, CONTROL_MORTALITY_ABBOTT,
)
from utils import abbotts_correction, classify_resistance


def load_raw_data() -> pd.DataFrame:
    """Load the extracted data CSV."""
    fp = DATA_RAW / "extracted_data.csv"
    if not fp.exists():
        print(f"[ERROR] Raw data file not found: {fp}")
        print("  Please populate 03_data/raw/extracted_data.csv first.")
        sys.exit(1)
    df = pd.read_csv(fp, dtype=str)
    print(f"  Loaded {len(df)} rows from {fp.name}")
    return df


def standardise_insecticides(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise insecticide names and assign classes."""
    col = "insecticide_name"
    if col not in df.columns:
        return df
    df[col] = df[col].str.strip().str.lower()
    df[col] = df[col].replace(INSECTICIDE_NAME_MAP)
    # If still lower-case, title-case known names
    known = {v.lower(): v for v in INSECTICIDE_CLASS_MAP}
    df[col] = df[col].apply(lambda x: known.get(x, x) if pd.notna(x) else x)

    # Assign class
    df["insecticide_class"] = df[col].map(INSECTICIDE_CLASS_MAP).fillna("Unknown")
    n_unknown = (df["insecticide_class"] == "Unknown").sum()
    if n_unknown:
        unknown_names = df.loc[df["insecticide_class"] == "Unknown", col].unique()
        print(f"  WARNING: {n_unknown} rows with unknown insecticide class: {unknown_names}")
    return df


def standardise_geography(df: pd.DataFrame) -> pd.DataFrame:
    """Assign WHO region and continent from country."""
    if "country" in df.columns:
        df["country"] = df["country"].str.strip()
        df["region"] = df["country"].map(WHO_REGION_MAP).fillna(df.get("region", ""))
    if "region" in df.columns:
        from config import CONTINENT_MAP
        df["continent"] = df["region"].map(CONTINENT_MAP).fillna("")
    return df


def standardise_kdr(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise kdr mutation names."""
    if "mutation" in df.columns:
        df["mutation"] = df["mutation"].str.strip().replace(KDR_MUTATION_MAP)
    return df


def to_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns from string to float/int."""
    numeric_cols = [
        "year", "latitude", "longitude",
        "collection_year_start", "collection_year_end",
        "concentration", "exposure_time_min", "recovery_time_h",
        "n_tested", "n_dead", "mortality_pct",
        "mortality_ci_lower", "mortality_ci_upper",
        "control_mortality_pct", "quality_score",
        "rr_value", "rr_ci_lower", "rr_ci_upper",
        "lc_field", "lc_reference",
        "codon_position", "n_genotyped", "n_mutant_alleles",
        "allele_frequency", "freq_ci_lower", "freq_ci_upper",
        "genotype_RR", "genotype_RS", "genotype_SS",
        "field_mean", "field_sd", "field_n",
        "reference_mean", "reference_sd", "reference_n",
        "fold_change", "elevated_pct", "p_value",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def validate_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ranges and compute derived variables."""

    # --- Mortality ---
    mask_mort = df["n_tested"].notna() & df["n_dead"].notna()
    # Compute mortality if not provided
    missing_mort = mask_mort & df["mortality_pct"].isna()
    if missing_mort.any():
        df.loc[missing_mort, "mortality_pct"] = (
            df.loc[missing_mort, "n_dead"] / df.loc[missing_mort, "n_tested"] * 100
        )
        print(f"  Computed mortality_pct for {missing_mort.sum()} rows")

    # Clip mortality
    if "mortality_pct" in df.columns:
        df["mortality_pct"] = df["mortality_pct"].clip(0, 100)

    # Abbott's correction
    if "control_mortality_pct" in df.columns and "mortality_pct" in df.columns:
        need_abbott = (
            df["control_mortality_pct"].between(
                CONTROL_MORTALITY_ABBOTT * 100, CONTROL_MORTALITY_MAX * 100, inclusive="both"
            )
        )
        if need_abbott.any():
            df.loc[need_abbott, "mortality_pct"] = df.loc[need_abbott].apply(
                lambda r: abbotts_correction(r["mortality_pct"], r["control_mortality_pct"]),
                axis=1,
            )
            print(f"  Applied Abbott's correction to {need_abbott.sum()} rows")

        # Discard rows with control mortality > 20%
        bad_control = df["control_mortality_pct"] > CONTROL_MORTALITY_MAX * 100
        if bad_control.any():
            print(f"  Removing {bad_control.sum()} rows with control mortality > {CONTROL_MORTALITY_MAX*100}%")
            df = df[~bad_control].copy()

    # WHO classification
    if "mortality_pct" in df.columns:
        df["who_classification"] = df["mortality_pct"].apply(
            lambda x: classify_resistance(x) if pd.notna(x) else ""
        )

    # --- Sample size filter ---
    if "n_tested" in df.columns:
        small = df["n_tested"].notna() & (df["n_tested"] < MIN_SAMPLE_SIZE)
        if small.any():
            print(f"  Removing {small.sum()} rows with n_tested < {MIN_SAMPLE_SIZE}")
            df = df[~small].copy()

    # --- kdr allele frequency ---
    mask_kdr = df["n_genotyped"].notna() & df["n_mutant_alleles"].notna()
    missing_freq = mask_kdr & df["allele_frequency"].isna()
    if missing_freq.any():
        df.loc[missing_freq, "allele_frequency"] = (
            df.loc[missing_freq, "n_mutant_alleles"] /
            (df.loc[missing_freq, "n_genotyped"] * 2)  # diploid
        )
        print(f"  Computed allele_frequency for {missing_freq.sum()} rows")

    # Compute from genotype counts if available
    geno_cols = ["genotype_RR", "genotype_RS", "genotype_SS"]
    has_geno = df[geno_cols].notna().all(axis=1) & df["allele_frequency"].isna()
    if has_geno.any():
        total = df.loc[has_geno, "genotype_RR"] + df.loc[has_geno, "genotype_RS"] + df.loc[has_geno, "genotype_SS"]
        df.loc[has_geno, "n_genotyped"] = total
        df.loc[has_geno, "allele_frequency"] = (
            (2 * df.loc[has_geno, "genotype_RR"] + df.loc[has_geno, "genotype_RS"]) / (2 * total)
        )
        print(f"  Computed allele_frequency from genotypes for {has_geno.sum()} rows")

    if "allele_frequency" in df.columns:
        df["allele_frequency"] = df["allele_frequency"].clip(0, 1)

    # --- Fold change ---
    if "field_mean" in df.columns and "reference_mean" in df.columns:
        missing_fc = (
            df["field_mean"].notna() & df["reference_mean"].notna() &
            (df["reference_mean"] > 0) & df["fold_change"].isna()
        )
        if missing_fc.any():
            df.loc[missing_fc, "fold_change"] = (
                df.loc[missing_fc, "field_mean"] / df.loc[missing_fc, "reference_mean"]
            )

    return df


def split_and_save(df: pd.DataFrame):
    """Split the unified dataset into indicator-specific files and save."""

    # Mortality data
    mort_cols = [
        "study_id", "country", "region", "continent", "year",
        "collection_year_start", "collection_year_end",
        "species", "life_stage", "quality_score",
        "insecticide_name", "insecticide_class",
        "bioassay_method", "concentration", "concentration_unit",
        "n_tested", "n_dead", "mortality_pct",
        "mortality_ci_lower", "mortality_ci_upper",
        "who_classification",
    ]
    mort = df.dropna(subset=["mortality_pct", "n_tested"], how="any")
    avail_cols = [c for c in mort_cols if c in mort.columns]
    mort = mort[avail_cols].copy()
    mort.to_csv(DATA_PROCESSED / "mortality_data.csv", index=False)
    print(f"  Saved mortality_data.csv ({len(mort)} rows)")

    # Resistance ratio data
    rr_cols = [
        "study_id", "country", "region", "continent", "year",
        "species", "life_stage", "quality_score",
        "insecticide_name", "insecticide_class",
        "rr_type", "rr_value", "rr_ci_lower", "rr_ci_upper",
        "lc_field", "lc_reference", "reference_strain",
    ]
    rr = df.dropna(subset=["rr_value"], how="any")
    avail_cols = [c for c in rr_cols if c in rr.columns]
    rr = rr[avail_cols].copy()
    rr.to_csv(DATA_PROCESSED / "rr_data.csv", index=False)
    print(f"  Saved rr_data.csv ({len(rr)} rows)")

    # kdr data
    kdr_cols = [
        "study_id", "country", "region", "continent", "year",
        "species", "quality_score",
        "gene", "codon_position", "mutation",
        "n_genotyped", "n_mutant_alleles", "allele_frequency",
        "freq_ci_lower", "freq_ci_upper",
        "genotype_RR", "genotype_RS", "genotype_SS",
        "detection_method",
    ]
    kdr = df.dropna(subset=["allele_frequency", "n_genotyped"], how="any")
    avail_cols = [c for c in kdr_cols if c in kdr.columns]
    kdr = kdr[avail_cols].copy()
    kdr.to_csv(DATA_PROCESSED / "kdr_data.csv", index=False)
    print(f"  Saved kdr_data.csv ({len(kdr)} rows)")

    # Enzyme data
    enz_cols = [
        "study_id", "country", "region", "continent", "year",
        "species", "quality_score",
        "enzyme_system", "enzyme_full_name", "assay_method",
        "field_mean", "field_sd", "field_n",
        "reference_mean", "reference_sd", "reference_n",
        "fold_change", "elevated_pct", "p_value",
    ]
    enz = df.dropna(subset=["field_mean", "field_n"], how="any")
    avail_cols = [c for c in enz_cols if c in enz.columns]
    enz = enz[avail_cols].copy()
    enz.to_csv(DATA_PROCESSED / "enzyme_data.csv", index=False)
    print(f"  Saved enzyme_data.csv ({len(enz)} rows)")

    # Unified
    df.to_csv(DATA_PROCESSED / "unified_dataset.csv", index=False)
    print(f"  Saved unified_dataset.csv ({len(df)} rows)")


def main():
    print("=" * 60)
    print("a01 – Data Cleaning and Preprocessing")
    print("=" * 60)

    df = load_raw_data()
    df = to_numeric_columns(df)
    df = standardise_insecticides(df)
    df = standardise_geography(df)
    df = standardise_kdr(df)
    df = validate_and_derive(df)
    split_and_save(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
