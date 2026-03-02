#!/usr/bin/env python3
"""
a02_descriptive_stats.py -- Descriptive statistics and visualizations.

Generates summary tables, bar plots, heatmaps, and geographic maps
characterizing the included studies in the insecticide resistance
meta-analysis.

Outputs
-------
Figures:
    05_figures/temporal_trends/studies_by_year.png
    05_figures/maps/studies_by_region.png
    05_figures/descriptive/studies_by_insecticide_class.png
    05_figures/descriptive/studies_by_species.png
    05_figures/data_availability_matrix.png
    05_figures/maps/study_locations_map.png
Tables:
    06_tables/table1_study_characteristics.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from utils import *

# ---------------------------------------------------------------------------
# Helper: safe directory creation
# ---------------------------------------------------------------------------

def _ensure_dir(filepath: Path) -> None:
    """Create parent directories for *filepath* if they do not exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed_data() -> dict[str, pd.DataFrame]:
    """Load all processed CSV files.

    Returns a dict keyed by short name (mortality, rr, kdr, enzyme).
    Empty DataFrames are returned for files that do not exist or are empty.
    """
    files = {
        "mortality": DATA_PROCESSED / "mortality_data.csv",
        "rr": DATA_PROCESSED / "rr_data.csv",
        "kdr": DATA_PROCESSED / "kdr_data.csv",
        "enzyme": DATA_PROCESSED / "enzyme_data.csv",
    }
    data: dict[str, pd.DataFrame] = {}
    for key, fp in files.items():
        if fp.exists():
            try:
                df = pd.read_csv(fp)
                if df.empty:
                    print(f"  WARNING: {fp.name} is empty.")
                    data[key] = pd.DataFrame()
                else:
                    print(f"  Loaded {fp.name}: {len(df)} rows, {len(df.columns)} cols")
                    data[key] = df
            except Exception as exc:
                print(f"  WARNING: Could not read {fp.name}: {exc}")
                data[key] = pd.DataFrame()
        else:
            print(f"  WARNING: {fp.name} not found -- skipping.")
            data[key] = pd.DataFrame()
    return data


def build_combined(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all datasets into a single frame for study-level counts.

    Deduplicates on study_id so that each study is counted once.
    """
    frames = [df for df in data.values() if not df.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


# ---------------------------------------------------------------------------
# 1. Studies by year
# ---------------------------------------------------------------------------

def plot_studies_by_year(combined: pd.DataFrame) -> None:
    """Bar plot of unique study count per publication year."""
    if combined.empty or "year" not in combined.columns:
        print("  Skipping studies_by_year: no year data available.")
        return

    year_counts = (
        combined.dropna(subset=["year"])
        .drop_duplicates(subset=["study_id"])
        .groupby("year")
        .size()
        .reset_index(name="n_studies")
    )
    if year_counts.empty:
        print("  Skipping studies_by_year: no valid year entries.")
        return

    year_counts["year"] = year_counts["year"].astype(int)
    year_counts = year_counts.sort_values("year")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        year_counts["year"],
        year_counts["n_studies"],
        color="#0072B2",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of studies")
    ax.set_title("Included studies by publication year")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    outpath = FIGURES_DIR / "temporal_trends" / "studies_by_year.png"
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# 2. Studies by WHO region
# ---------------------------------------------------------------------------

def plot_studies_by_region(combined: pd.DataFrame) -> None:
    """Bar plot of unique study count per WHO region."""
    if combined.empty or "region" not in combined.columns:
        print("  Skipping studies_by_region: no region data available.")
        return

    region_counts = (
        combined.dropna(subset=["region"])
        .loc[lambda d: d["region"] != ""]
        .drop_duplicates(subset=["study_id"])
        .groupby("region")
        .size()
        .reset_index(name="n_studies")
        .sort_values("n_studies", ascending=False)
    )
    if region_counts.empty:
        print("  Skipping studies_by_region: no valid region entries.")
        return

    colors = [WHO_REGION_COLORS.get(r, "#999999") for r in region_counts["region"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        region_counts["region"],
        region_counts["n_studies"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Number of studies")
    ax.set_ylabel("WHO region")
    ax.set_title("Included studies by WHO region")
    ax.invert_yaxis()

    outpath = FIGURES_DIR / "maps" / "studies_by_region.png"
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# 3. Studies by insecticide class
# ---------------------------------------------------------------------------

def plot_studies_by_insecticide_class(combined: pd.DataFrame) -> None:
    """Bar plot of observation count per insecticide class."""
    if combined.empty or "insecticide_class" not in combined.columns:
        print("  Skipping studies_by_insecticide_class: no data available.")
        return

    class_counts = (
        combined.dropna(subset=["insecticide_class"])
        .loc[lambda d: d["insecticide_class"] != "Unknown"]
        .groupby("insecticide_class")
        .size()
        .reset_index(name="n_observations")
        .sort_values("n_observations", ascending=False)
    )
    if class_counts.empty:
        print("  Skipping studies_by_insecticide_class: no valid entries.")
        return

    colors = [
        INSECTICIDE_CLASS_COLORS.get(c, "#999999")
        for c in class_counts["insecticide_class"]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        class_counts["insecticide_class"],
        class_counts["n_observations"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Insecticide class")
    ax.set_title("Observations by insecticide class")
    ax.invert_yaxis()

    outpath = FIGURES_DIR / "descriptive" / "studies_by_insecticide_class.png"
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# 4. Studies by species
# ---------------------------------------------------------------------------

def plot_studies_by_species(combined: pd.DataFrame) -> None:
    """Bar plot of unique study count per species."""
    if combined.empty or "species" not in combined.columns:
        print("  Skipping studies_by_species: no species data available.")
        return

    species_counts = (
        combined.dropna(subset=["species"])
        .drop_duplicates(subset=["study_id", "species"])
        .groupby("species")
        .size()
        .reset_index(name="n_studies")
        .sort_values("n_studies", ascending=False)
    )
    if species_counts.empty:
        print("  Skipping studies_by_species: no valid species entries.")
        return

    # Show top 15 species to avoid clutter
    top = species_counts.head(15).copy()

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.4)))
    ax.barh(
        top["species"],
        top["n_studies"],
        color="#56B4E9",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Number of studies")
    ax.set_ylabel("Species")
    ax.set_title("Included studies by mosquito species (top 15)")
    ax.invert_yaxis()

    # Italicize species names
    for label in ax.get_yticklabels():
        label.set_fontstyle("italic")

    outpath = FIGURES_DIR / "descriptive" / "studies_by_species.png"
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# 5. Data availability matrix
# ---------------------------------------------------------------------------

def plot_data_availability_matrix(combined: pd.DataFrame) -> None:
    """Heatmap of observation counts by insecticide class x WHO region."""
    required = {"insecticide_class", "region"}
    if combined.empty or not required.issubset(combined.columns):
        print("  Skipping data_availability_matrix: missing columns.")
        return

    filtered = combined.dropna(subset=["insecticide_class", "region"])
    filtered = filtered.loc[
        (filtered["insecticide_class"] != "Unknown") & (filtered["region"] != "")
    ]
    if filtered.empty:
        print("  Skipping data_availability_matrix: no valid data.")
        return

    matrix = pd.crosstab(filtered["insecticide_class"], filtered["region"])

    # Order rows and columns consistently
    class_order = [
        c for c in INSECTICIDE_CLASS_COLORS
        if c in matrix.index
    ]
    remaining_classes = sorted(set(matrix.index) - set(class_order))
    class_order.extend(remaining_classes)
    matrix = matrix.reindex(index=class_order, fill_value=0)

    region_order = [r for r in WHO_REGION_COLORS if r in matrix.columns]
    remaining_regions = sorted(set(matrix.columns) - set(region_order))
    region_order.extend(remaining_regions)
    matrix = matrix.reindex(columns=region_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Number of observations"},
        ax=ax,
    )
    ax.set_title("Data availability: insecticide class by WHO region")
    ax.set_xlabel("WHO region")
    ax.set_ylabel("Insecticide class")

    outpath = FIGURES_DIR / "data_availability_matrix.png"
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# 6. Geographic distribution map
# ---------------------------------------------------------------------------

def _get_name_column(world_gdf) -> str:
    """Detect the country name column in Natural Earth shapefile.

    Different geopandas versions have different column names.
    Try common variations in order of preference.
    """
    possible_names = ["name", "NAME", "NAME_EN", "ADMIN", "NAME_LONG", "SOVEREIGNT"]
    for col in possible_names:
        if col in world_gdf.columns:
            return col
    # If none found, raise error with available columns
    raise KeyError(
        f"Could not find country name column. Available columns: {list(world_gdf.columns)}"
    )


def plot_study_locations_map(combined: pd.DataFrame) -> None:
    """World map showing study locations using geopandas."""
    try:
        import geopandas as gpd
    except ImportError:
        print("  Skipping study_locations_map: geopandas not installed.")
        return

    if combined.empty:
        print("  Skipping study_locations_map: no data available.")
        return

    has_coords = (
        "latitude" in combined.columns
        and "longitude" in combined.columns
    )
    has_country = "country" in combined.columns

    if not has_coords and not has_country:
        print("  Skipping study_locations_map: no geographic columns found.")
        return

    # --- Base world map ---
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        # geopandas >= 1.0 removed datasets module; try naturalearth_lowres
        # from geodatasets or bundled path
        try:
            import geodatasets
            world = gpd.read_file(geodatasets.data.naturalearth_lowres.url)
        except Exception:
            try:
                world = gpd.read_file(
                    "https://naciscdn.org/naturalearth/110m/cultural/"
                    "ne_110m_admin_0_countries.zip"
                )
            except Exception as exc:
                print(f"  Skipping study_locations_map: could not load world map ({exc}).")
                return

    # Detect the correct name column for merging
    try:
        name_col = _get_name_column(world)
    except KeyError as exc:
        print(f"  Skipping study_locations_map: {exc}")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    world.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.3)

    # --- Point-level data (if lat/lon available) ---
    if has_coords:
        pts = combined.dropna(subset=["latitude", "longitude"]).copy()
        if not pts.empty:
            # Deduplicate to study-level
            pts = pts.drop_duplicates(subset=["study_id", "latitude", "longitude"])
            gdf = gpd.GeoDataFrame(
                pts,
                geometry=gpd.points_from_xy(pts["longitude"], pts["latitude"]),
                crs="EPSG:4326",
            )
            # Color by region if available
            if "region" in gdf.columns:
                for region, color in WHO_REGION_COLORS.items():
                    subset = gdf[gdf["region"] == region]
                    if not subset.empty:
                        subset.plot(
                            ax=ax,
                            color=color,
                            markersize=18,
                            alpha=0.7,
                            edgecolor="black",
                            linewidth=0.3,
                            label=region,
                        )
                # Points with unknown region
                unknown = gdf[~gdf["region"].isin(WHO_REGION_COLORS)]
                if not unknown.empty:
                    unknown.plot(
                        ax=ax,
                        color="#999999",
                        markersize=18,
                        alpha=0.7,
                        edgecolor="black",
                        linewidth=0.3,
                        label="Other",
                    )
                ax.legend(
                    title="WHO region",
                    loc="lower left",
                    fontsize=8,
                    title_fontsize=9,
                    framealpha=0.9,
                )
            else:
                gdf.plot(
                    ax=ax,
                    color="#D55E00",
                    markersize=18,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.3,
                )

    # --- Country-level choropleth (if no coords, colour countries instead) ---
    elif has_country:
        study_countries = (
            combined.dropna(subset=["country"])
            .drop_duplicates(subset=["study_id"])
            .groupby("country")
            .size()
            .reset_index(name="n_studies")
        )
        if not study_countries.empty:
            merged = world.merge(
                study_countries,
                left_on=name_col,
                right_on="country",
                how="left",
            )
            merged["n_studies"] = merged["n_studies"].fillna(0)
            merged.plot(
                column="n_studies",
                ax=ax,
                cmap="YlOrRd",
                edgecolor="#cccccc",
                linewidth=0.3,
                legend=True,
                legend_kwds={"label": "Number of studies", "shrink": 0.5},
                missing_kwds={"color": "#f0f0f0"},
            )

    ax.set_title("Geographic distribution of included studies", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)

    outpath = FIGURES_DIR / "maps" / "study_locations_map.png"
    _ensure_dir(outpath)
    fig.savefig(outpath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# 7. Summary statistics table (Table 1)
# ---------------------------------------------------------------------------

def generate_table1(
    data: dict[str, pd.DataFrame],
    combined: pd.DataFrame,
) -> None:
    """Generate Table 1: Study characteristics summary."""
    if combined.empty:
        print("  Skipping table1: no data available.")
        return

    rows: list[dict[str, str]] = []

    # --- Overall counts ---
    studies_unique = combined.drop_duplicates(subset=["study_id"]) if "study_id" in combined.columns else combined
    n_studies = len(studies_unique)
    rows.append({"Characteristic": "Total included studies", "Value": str(n_studies)})

    # Per-dataset counts
    for label, key in [
        ("Studies with mortality data", "mortality"),
        ("Studies with resistance ratio data", "rr"),
        ("Studies with kdr data", "kdr"),
        ("Studies with enzyme data", "enzyme"),
    ]:
        df = data[key]
        if not df.empty and "study_id" in df.columns:
            n = df["study_id"].nunique()
            n_obs = len(df)
            rows.append({"Characteristic": label, "Value": f"{n} ({n_obs} observations)"})
        else:
            rows.append({"Characteristic": label, "Value": "0"})

    # --- Year range ---
    if "year" in combined.columns:
        years = combined["year"].dropna()
        if not years.empty:
            rows.append({
                "Characteristic": "Publication year range",
                "Value": f"{int(years.min())}--{int(years.max())}",
            })
            rows.append({
                "Characteristic": "Median publication year",
                "Value": str(int(years.median())),
            })

    # --- Countries ---
    if "country" in combined.columns:
        countries = combined["country"].dropna().unique()
        n_countries = len(countries)
        rows.append({"Characteristic": "Countries represented", "Value": str(n_countries)})

    # --- WHO regions ---
    if "region" in combined.columns:
        region_counts = (
            studies_unique["region"]
            .dropna()
            .loc[lambda s: s != ""]
            .value_counts()
        )
        for region, count in region_counts.items():
            rows.append({
                "Characteristic": f"  {region}",
                "Value": f"{count} studies",
            })

    # --- Species ---
    if "species" in combined.columns:
        species_list = combined["species"].dropna().unique()
        rows.append({"Characteristic": "Species reported", "Value": str(len(species_list))})
        species_counts = (
            studies_unique["species"]
            .dropna()
            .value_counts()
            .head(5)
        )
        for sp, count in species_counts.items():
            rows.append({
                "Characteristic": f"  {sp}",
                "Value": f"{count} studies",
            })

    # --- Insecticide classes ---
    if "insecticide_class" in combined.columns:
        ic_counts = (
            combined.loc[combined["insecticide_class"] != "Unknown", "insecticide_class"]
            .dropna()
            .value_counts()
        )
        rows.append({
            "Characteristic": "Insecticide classes tested",
            "Value": str(len(ic_counts)),
        })
        for ic, count in ic_counts.items():
            rows.append({
                "Characteristic": f"  {ic}",
                "Value": f"{count} observations",
            })

    # --- Insecticides ---
    if "insecticide_name" in combined.columns:
        insecticides = combined["insecticide_name"].dropna().unique()
        rows.append({
            "Characteristic": "Individual insecticides tested",
            "Value": str(len(insecticides)),
        })

    # --- Mortality summary (from mortality dataset) ---
    mort_df = data["mortality"]
    if not mort_df.empty and "mortality_pct" in mort_df.columns:
        mort_vals = mort_df["mortality_pct"].dropna()
        if not mort_vals.empty:
            rows.append({
                "Characteristic": "Mortality (%) -- median [IQR]",
                "Value": (
                    f"{mort_vals.median():.1f} "
                    f"[{mort_vals.quantile(0.25):.1f}--{mort_vals.quantile(0.75):.1f}]"
                ),
            })
        # WHO classification breakdown
        if "who_classification" in mort_df.columns:
            who_counts = mort_df["who_classification"].value_counts()
            for status, count in who_counts.items():
                pct = count / len(mort_df) * 100
                rows.append({
                    "Characteristic": f"  {status}",
                    "Value": f"{count} ({pct:.1f}%)",
                })

    # --- Sample size summary ---
    if "n_tested" in combined.columns:
        n_tested = combined["n_tested"].dropna()
        if not n_tested.empty:
            rows.append({
                "Characteristic": "Sample size (n tested) -- median [IQR]",
                "Value": (
                    f"{n_tested.median():.0f} "
                    f"[{n_tested.quantile(0.25):.0f}--{n_tested.quantile(0.75):.0f}]"
                ),
            })

    # --- kdr summary ---
    kdr_df = data["kdr"]
    if not kdr_df.empty and "mutation" in kdr_df.columns:
        mutations = kdr_df["mutation"].dropna().unique()
        rows.append({
            "Characteristic": "kdr mutations reported",
            "Value": str(len(mutations)),
        })
        for mut in sorted(mutations):
            n_obs = (kdr_df["mutation"] == mut).sum()
            rows.append({
                "Characteristic": f"  {mut}",
                "Value": f"{n_obs} observations",
            })

    # Save
    table1 = pd.DataFrame(rows)
    outpath = TABLES_DIR / "table1_study_characteristics.csv"
    _ensure_dir(outpath)
    table1.to_csv(outpath, index=False)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("a02 -- Descriptive Statistics and Visualizations")
    print("=" * 60)

    # Suppress minor warnings from matplotlib / geopandas
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

    # 0. Load data
    print("\n[1/8] Loading processed data...")
    data = load_processed_data()
    combined = build_combined(data)

    if combined.empty:
        print("\n  No data found in any processed file.")
        print("  Run a01_data_cleaning.py first to generate processed datasets.")
        print("  Exiting.\n")
        return

    n_datasets = sum(1 for df in data.values() if not df.empty)
    print(f"  Combined dataset: {len(combined)} total rows from {n_datasets} file(s)")

    # 1. Studies by year
    print("\n[2/8] Plotting studies by year...")
    plot_studies_by_year(combined)

    # 2. Studies by WHO region
    print("\n[3/8] Plotting studies by WHO region...")
    plot_studies_by_region(combined)

    # 3. Studies by insecticide class
    print("\n[4/8] Plotting studies by insecticide class...")
    plot_studies_by_insecticide_class(combined)

    # 4. Studies by species
    print("\n[5/8] Plotting studies by species...")
    plot_studies_by_species(combined)

    # 5. Data availability matrix
    print("\n[6/8] Plotting data availability matrix...")
    plot_data_availability_matrix(combined)

    # 6. Geographic map
    print("\n[7/8] Plotting study locations map...")
    plot_study_locations_map(combined)

    # 7. Table 1
    print("\n[8/8] Generating Table 1 (study characteristics)...")
    generate_table1(data, combined)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
