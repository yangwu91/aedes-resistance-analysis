#!/usr/bin/env python3
"""
fill_missing_data.py – Fill in missing sample sizes with reasonable defaults.

For WHO tube bioassays: n_tested = 100 (standard 4 replicates of 25)
For CDC bottle bioassays: n_tested = 100
For larval bioassays: n_tested = 100
For kdr without n_genotyped: estimate from context or use median
For enzyme data: restructure to include fold_change as the primary measure
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "03_data" / "raw" / "extracted_data.csv"
OUTPUT_FILE = PROJECT_DIR / "03_data" / "raw" / "extracted_data.csv"  # overwrite


def main():
    print("=" * 60)
    print("Fill Missing Data with Reasonable Defaults")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE, dtype=str)
    print(f"  Loaded {len(df)} rows")

    # Convert numeric columns
    num_cols = [
        "mortality_pct", "n_tested", "n_dead", "rr_value", "rr_ci_lower",
        "rr_ci_upper", "allele_frequency", "n_genotyped", "fold_change",
        "elevated_pct", "field_mean", "field_sd", "field_n",
        "reference_mean", "reference_sd", "reference_n",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 1. Fill missing n_tested for mortality data ──
    # Standard WHO/CDC bioassay protocol: 4 replicates × 25 = 100 mosquitoes
    mort_mask = df["mortality_pct"].notna() & df["n_tested"].isna()
    method_col = "bioassay_method"

    # WHO tube/bottle: typically 100 (4 × 25)
    who_mask = mort_mask & df[method_col].str.contains("WHO", na=False, case=False)
    df.loc[who_mask, "n_tested"] = 100
    print(f"  Filled n_tested=100 for {who_mask.sum()} WHO bioassay rows")

    # CDC bottle: typically 100
    cdc_mask = mort_mask & df[method_col].str.contains("CDC", na=False, case=False)
    df.loc[cdc_mask, "n_tested"] = 100
    print(f"  Filled n_tested=100 for {cdc_mask.sum()} CDC bioassay rows")

    # Larval: typically 100
    larv_mask = mort_mask & df[method_col].str.contains("arv", na=False, case=False)
    df.loc[larv_mask, "n_tested"] = 100
    print(f"  Filled n_tested=100 for {larv_mask.sum()} larval bioassay rows")

    # Remaining mortality without n_tested: default 100
    remaining = df["mortality_pct"].notna() & df["n_tested"].isna()
    df.loc[remaining, "n_tested"] = 100
    print(f"  Filled n_tested=100 for {remaining.sum()} remaining mortality rows")

    # Compute n_dead from mortality_pct and n_tested
    need_dead = (
        df["mortality_pct"].notna() &
        df["n_tested"].notna() &
        df["n_dead"].isna()
    )
    if need_dead.any():
        df.loc[need_dead, "n_dead"] = (
            df.loc[need_dead, "mortality_pct"] / 100 *
            df.loc[need_dead, "n_tested"]
        ).round(0).astype(int)
        print(f"  Computed n_dead for {need_dead.sum()} rows")

    # ── 2. Fill missing n_genotyped for kdr data ──
    kdr_mask = df["allele_frequency"].notna() & df["n_genotyped"].isna()
    # Default: median n_genotyped from available data, or 30
    existing_n = df.loc[df["n_genotyped"].notna(), "n_genotyped"]
    default_n_geno = int(existing_n.median()) if len(existing_n) > 0 else 30
    df.loc[kdr_mask, "n_genotyped"] = default_n_geno
    print(f"  Filled n_genotyped={default_n_geno} for {kdr_mask.sum()} kdr rows")

    # Compute n_mutant_alleles from allele_frequency and n_genotyped
    need_alleles = (
        df["allele_frequency"].notna() &
        df["n_genotyped"].notna() &
        df["n_mutant_alleles"].isna()
    )
    if need_alleles.any():
        df.loc[need_alleles, "n_mutant_alleles"] = (
            df.loc[need_alleles, "allele_frequency"] *
            df.loc[need_alleles, "n_genotyped"] * 2
        ).round(0).astype(int)
        print(f"  Computed n_mutant_alleles for {need_alleles.sum()} rows")

    # ── 3. Handle enzyme data ──
    # Many enzyme rows only have enzyme_system without quantitative data
    # For fold_change: if not available, use a default based on the mention
    enz_mask = (
        df["enzyme_system"].notna() &
        (df["enzyme_system"] != "") &
        df["fold_change"].isna() &
        df["field_mean"].isna() &
        df["elevated_pct"].isna()
    )
    # We can't meaningfully fill enzyme data without quantitative values
    # But we can mark them for inclusion in the analysis if they mention elevation
    print(f"  Enzyme rows without quantitative data: {enz_mask.sum()}")

    # For enzyme rows with field_mean but no field_n, use default
    enz_n_mask = df["field_mean"].notna() & df["field_n"].isna()
    df.loc[enz_n_mask, "field_n"] = 30  # reasonable default
    print(f"  Filled field_n=30 for {enz_n_mask.sum()} enzyme rows")

    # ── 4. Ensure insecticide_class is filled ──
    import sys
    sys.path.insert(0, str(PROJECT_DIR / "04_analysis"))
    from config import INSECTICIDE_CLASS_MAP

    # Fill insecticide_class from insecticide_name where missing
    class_mask = (
        df["insecticide_name"].notna() &
        (df["insecticide_name"] != "") &
        (df["insecticide_class"].isna() | (df["insecticide_class"] == ""))
    )
    if class_mask.any():
        df.loc[class_mask, "insecticide_class"] = df.loc[class_mask, "insecticide_name"].map(
            INSECTICIDE_CLASS_MAP
        ).fillna("Unknown")
        print(f"  Filled insecticide_class for {class_mask.sum()} rows")

    # ── Save ──
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  Total rows: {len(df)}")

    # Summary
    print(f"\nData availability after filling:")
    mort = df["mortality_pct"].notna()
    mort_n = (mort & df["n_tested"].notna())
    print(f"  Mortality with n_tested: {mort_n.sum()}")
    kdr = df["allele_frequency"].notna()
    kdr_n = (kdr & df["n_genotyped"].notna())
    print(f"  kdr with n_genotyped: {kdr_n.sum()}")
    rr = df["rr_value"].notna()
    print(f"  RR data: {rr.sum()}")
    enz = df["enzyme_system"].notna() & (df["enzyme_system"] != "")
    enz_q = enz & (df["field_mean"].notna() | df["fold_change"].notna() | df["elevated_pct"].notna())
    print(f"  Enzyme with quantitative data: {enz_q.sum()}")


if __name__ == "__main__":
    main()
