#!/usr/bin/env python3
"""
generate_figure6_map.py — Publication-quality world map (Figure 6).

Generates a choropleth of study counts by country.
Uses DataV GeoAtlas for China boundaries (includes Taiwan and 藏南).
"""

import json
import urllib.request
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE / "03_data" / "processed"
FIGURES_DIR = BASE / "05_figures"
MANUSCRIPT_DIR = BASE / "07_manuscript"


# ──────────────────────────────────────────────────────────────────────
# 1. Load study data
# ──────────────────────────────────────────────────────────────────────
def load_study_counts() -> pd.DataFrame:
    fp = DATA_PROCESSED / "mortality_data.csv"
    df = pd.read_csv(fp)
    counts = (
        df.drop_duplicates(subset=["study_id"])
        .groupby("country")
        .size()
        .reset_index(name="n_studies")
    )
    # Normalise country names to match Natural Earth
    name_map = {
        "USA": "United States of America",
        "United States": "United States of America",
        "UK": "United Kingdom",
        "United Kingdom": "United Kingdom",
        "Republic of Korea": "South Korea",
        "South Korea": "South Korea",
        "Ivory Coast": "Côte d'Ivoire",
        "Cote d'Ivoire": "Côte d'Ivoire",
        "DR Congo": "Dem. Rep. Congo",
        "Democratic Republic of the Congo": "Dem. Rep. Congo",
        "Papua New Guinea": "Papua New Guinea",
    }
    counts["country_std"] = counts["country"].replace(name_map)

    # Merge duplicate names (e.g. "USA" + "United States" → one row)
    counts = (
        counts.groupby("country_std")["n_studies"]
        .sum()
        .reset_index()
        .rename(columns={"country_std": "country"})
    )
    return counts


# ──────────────────────────────────────────────────────────────────────
# 2. Load world map + China official boundary
# ──────────────────────────────────────────────────────────────────────
def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def load_world_with_china() -> gpd.GeoDataFrame:
    """Load world countries, replacing China+Taiwan with official Chinese
    administrative boundary (DataV GeoAtlas) that includes 台湾 and 藏南."""

    # --- Natural Earth base ---
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        try:
            import geodatasets
            world = gpd.read_file(geodatasets.data.naturalearth_lowres.url)
        except Exception:
            world = gpd.read_file(
                "https://naciscdn.org/naturalearth/110m/cultural/"
                "ne_110m_admin_0_countries.zip"
            )

    # Detect name column
    for col in ("name", "NAME", "ADMIN", "NAME_EN"):
        if col in world.columns:
            name_col = col
            break
    else:
        raise KeyError("Cannot find name column in world shapefile")

    # --- Download China boundary from DataV GeoAtlas ---
    # 100000_full.json = all 34 provincial-level divisions + nine-dash line
    print("  Downloading China boundary from DataV GeoAtlas...")
    china_data = _fetch_json(
        "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"
    )
    china_gdf = gpd.GeoDataFrame.from_features(
        china_data["features"], crs="EPSG:4326"
    )

    # Separate nine-dash line feature
    nine_dash = china_gdf[china_gdf["name"] == ""]
    provinces = china_gdf[china_gdf["name"] != ""]

    # Fix invalid geometries before dissolving
    provinces = provinces.copy()
    provinces["geometry"] = provinces.geometry.buffer(0)
    china_geom = unary_union(provinces.geometry)

    # --- Remove China and Taiwan from Natural Earth ---
    remove_names = {"China", "Taiwan"}
    mask = ~world[name_col].isin(remove_names)
    world_no_china = world[mask].copy()

    # --- Clip India to remove 藏南 overlap with China ---
    india_mask = world_no_china[name_col] == "India"
    if india_mask.any():
        india_geom = world_no_china.loc[india_mask, "geometry"].iloc[0]
        india_clipped = india_geom.difference(china_geom)
        world_no_china.loc[india_mask, "geometry"] = india_clipped

    # --- Add China with official boundary ---
    china_row = gpd.GeoDataFrame(
        [{name_col: "China", "geometry": china_geom,
          "continent": "Asia", "pop_est": 1.4e9, "iso_a3": "CHN"}],
        crs="EPSG:4326",
    )
    world_fixed = pd.concat(
        [world_no_china, china_row], ignore_index=True
    )

    return gpd.GeoDataFrame(world_fixed, crs="EPSG:4326"), nine_dash


# ──────────────────────────────────────────────────────────────────────
# 3. Draw the map
# ──────────────────────────────────────────────────────────────────────
def draw_map(world: gpd.GeoDataFrame, nine_dash: gpd.GeoDataFrame,
             counts: pd.DataFrame, outpath: Path) -> None:

    # Detect name column
    for col in ("name", "NAME", "ADMIN", "NAME_EN"):
        if col in world.columns:
            name_col = col
            break

    # Merge study counts
    merged = world.merge(counts, left_on=name_col, right_on="country", how="left")
    merged["n_studies"] = merged["n_studies"].fillna(0).astype(int)

    # ── Colour scheme ──
    # Discrete bins for cleaner visualisation
    bins = [0, 1, 2, 4, 6, 10, 15, 30]
    labels_txt = ["0", "1", "2–3", "4–5", "6–9", "10–14", "15+"]
    colors = ["#F5F5F5", "#FFF5CC", "#FFE08A", "#FFBF47",
              "#FF8C42", "#E8553D", "#B71C3C"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, cmap.N)

    # Assign colour bin
    merged["bin"] = pd.cut(
        merged["n_studies"], bins=bins, labels=labels_txt,
        right=False, include_lowest=True,
    )

    # ── Figure layout ──
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    # Ocean background
    ax.set_facecolor("#EAF2F8")

    # Draw all countries
    merged.plot(
        column="n_studies",
        ax=ax,
        cmap=cmap,
        norm=norm,
        edgecolor="#999999",
        linewidth=0.3,
        missing_kwds={"color": "#F5F5F5", "edgecolor": "#999999", "linewidth": 0.3},
    )

    # Draw nine-dash line
    if not nine_dash.empty:
        nine_dash.plot(ax=ax, color="none", edgecolor="#333333",
                       linewidth=0.8, linestyle="--")

    # ── South China Sea inset ──
    inset_ax = fig.add_axes([0.715, 0.12, 0.12, 0.22])
    inset_ax.set_facecolor("#EAF2F8")

    # Draw countries in inset region
    merged.plot(
        column="n_studies", ax=inset_ax,
        cmap=cmap, norm=norm,
        edgecolor="#999999", linewidth=0.3,
        missing_kwds={"color": "#F5F5F5", "edgecolor": "#999999", "linewidth": 0.3},
    )
    if not nine_dash.empty:
        nine_dash.plot(ax=inset_ax, color="none", edgecolor="#333333",
                       linewidth=0.8, linestyle="--")

    inset_ax.set_xlim(104, 125)
    inset_ax.set_ylim(0, 26)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.8)

    # ── Main map limits & styling ──
    ax.set_xlim(-170, 180)
    ax.set_ylim(-58, 83)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Geographic distribution of included studies",
                 fontsize=13, fontweight="bold", pad=12)

    # ── Legend ──
    legend_patches = [
        mpatches.Patch(facecolor=c, edgecolor="#999999", linewidth=0.5, label=l)
        for c, l in zip(colors, labels_txt)
    ]
    leg = ax.legend(
        handles=legend_patches,
        title="Number of studies",
        loc="lower left",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        borderpad=0.8,
        labelspacing=0.4,
    )
    leg.get_frame().set_linewidth(0.5)

    # ── Annotate countries with many studies ──
    label_countries = {
        "China": (108, 35),
        "Brazil": (-52, -12),
        "Benin": (2.3, 9.3),
    }
    for cname, (lx, ly) in label_countries.items():
        row = counts[counts["country"] == cname]
        if not row.empty:
            n = row["n_studies"].iloc[0]
            ax.annotate(
                f"{cname}\n(n={n})",
                xy=(lx, ly),
                fontsize=7,
                ha="center", va="center",
                fontweight="bold",
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.7),
            )

    # ── Save ──
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Map saved: {outpath}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("Generating Figure 6 — Geographic distribution map")

    counts = load_study_counts()
    print(f"  {len(counts)} countries with study data")

    world, nine_dash = load_world_with_china()
    print(f"  World map loaded: {len(world)} features")

    # Save to figures dir and manuscript dir
    fig_path = FIGURES_DIR / "maps" / "study_locations_map.png"
    ms_path = MANUSCRIPT_DIR / "Figure_6.png"

    draw_map(world, nine_dash, counts, fig_path)
    draw_map(world, nine_dash, counts, ms_path)

    print("Done.")


if __name__ == "__main__":
    main()
