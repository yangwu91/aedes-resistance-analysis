#!/usr/bin/env python3
"""
generate_figure2_map.py — Publication-quality world map (Figure 2).

Generates a choropleth of study counts by country using Winkel Tripel projection.
"""

import json
import urllib.request
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from shapely.geometry import Polygon
from shapely.ops import unary_union
from pyproj import Transformer

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "03_data" / "processed" / "mortality_data.csv"
FIGURES_DIR = BASE / "05_figures"
SUBMISSION_DIR = BASE / "submission"

DPI = 300
WIDTH_MM = 180
WIDTH_IN = WIDTH_MM / 25.4 * 1.25
HEIGHT_IN = WIDTH_IN * 0.55

WINTRI = '+proj=wintri +lon_0=0 +datum=WGS84 +units=m'

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.linewidth': 0.8,
})


def load_study_counts():
    df = pd.read_csv(DATA)
    counts = (
        df.drop_duplicates(subset=["study_id"])
        .groupby("country").size()
        .reset_index(name="n_studies")
    )
    name_map = {
        "USA": "United States of America",
        "United States": "United States of America",
        "UK": "United Kingdom",
        "Republic of Korea": "South Korea",
        "Ivory Coast": "Côte d'Ivoire",
        "Cote d'Ivoire": "Côte d'Ivoire",
        "DR Congo": "Dem. Rep. Congo",
        "Democratic Republic of the Congo": "Dem. Rep. Congo",
    }
    counts["country"] = counts["country"].replace(name_map)
    counts = counts.groupby("country")["n_studies"].sum().reset_index()
    return counts


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def load_world():
    """Load world map with China boundary from DataV GeoAtlas."""
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        world = gpd.read_file(
            "https://naciscdn.org/naturalearth/110m/cultural/"
            "ne_110m_admin_0_countries.zip"
        )

    for col in ("name", "NAME", "ADMIN"):
        if col in world.columns:
            name_col = col
            break

    print("  Downloading China boundary from DataV GeoAtlas...")
    china_data = fetch_json(
        "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"
    )
    china_gdf = gpd.GeoDataFrame.from_features(
        china_data["features"], crs="EPSG:4326"
    )
    provinces = china_gdf[china_gdf["name"] != ""].copy()
    provinces["geometry"] = provinces.geometry.buffer(0)
    china_geom = unary_union(provinces.geometry)

    mask = ~world[name_col].isin({"China", "Taiwan"})
    world_clean = world[mask].copy()

    india_mask = world_clean[name_col] == "India"
    if india_mask.any():
        india_geom = world_clean.loc[india_mask, "geometry"].iloc[0]
        world_clean.loc[india_mask, "geometry"] = india_geom.difference(china_geom)

    china_row = gpd.GeoDataFrame(
        [{name_col: "China", "geometry": china_geom,
          "continent": "Asia", "iso_a3": "CHN"}],
        crs="EPSG:4326",
    )
    world_out = pd.concat([world_clean, china_row], ignore_index=True)
    return gpd.GeoDataFrame(world_out, crs="EPSG:4326"), name_col


def build_boundary():
    """Build Winkel Tripel earth boundary."""
    transformer = Transformer.from_crs("EPSG:4326", WINTRI, always_xy=True)
    n = 300
    lats = np.linspace(-90, 90, n)
    left_x, left_y = transformer.transform(np.full(n, -179.999), lats)
    right_x, right_y = transformer.transform(np.full(n, 179.999), lats[::-1])
    bx = np.concatenate([left_x, right_x])
    by = np.concatenate([left_y, right_y])
    return Polygon(zip(bx, by))


def draw_map(world_wt, counts, name_col, boundary, outpath):
    merged = world_wt.merge(counts, left_on=name_col, right_on="country", how="left")
    merged["n_studies"] = merged["n_studies"].fillna(0).astype(int)

    bins = [0, 1, 2, 4, 6, 10, 15, 30]
    labels_txt = ["0", "1", "2–3", "4–5", "6–9", "10–14", "15+"]
    colors = ["#F5F5F5", "#FFF5CC", "#FFE08A", "#FFBF47",
              "#FF8C42", "#E8553D", "#B71C3C"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, cmap.N)

    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    fig.patch.set_facecolor("white")

    bx, by = boundary.exterior.xy
    ax.fill(bx, by, facecolor="#EAF2F8", edgecolor="#888888", linewidth=0.8, zorder=0)

    merged.plot(
        column="n_studies", ax=ax, cmap=cmap, norm=norm,
        edgecolor="#999999", linewidth=0.3, zorder=1,
        missing_kwds={"color": "#F5F5F5", "edgecolor": "#999999", "linewidth": 0.3},
    )

    ax.set_xlim(boundary.bounds[0] * 1.02, boundary.bounds[2] * 1.02)
    ax.set_ylim(boundary.bounds[1] * 1.05, boundary.bounds[3] * 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    transformer = Transformer.from_crs("EPSG:4326", WINTRI, always_xy=True)
    label_countries = {
        "China": (108, 35),
        "Brazil": (-52, -12),
        "Benin": (2.3, 9.3),
    }
    for cname, (lon, lat) in label_countries.items():
        row = counts[counts["country"] == cname]
        if not row.empty:
            n = row["n_studies"].iloc[0]
            px, py = transformer.transform(lon, lat)
            ax.annotate(
                f"{cname}\n(n={n})",
                xy=(px, py), fontsize=8, ha="center", va="center",
                fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.7),
                zorder=3,
            )

    legend_patches = [
        mpatches.Patch(facecolor=c, edgecolor="#999999", linewidth=0.5, label=l)
        for c, l in zip(colors, labels_txt)
    ]
    leg = ax.legend(
        handles=legend_patches, title="Number of studies",
        loc="lower left", fontsize=8, title_fontsize=9,
        framealpha=0.95, edgecolor="#CCCCCC", borderpad=0.8, labelspacing=0.4,
    )
    leg.get_frame().set_linewidth(0.5)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fmt = "tiff" if outpath.suffix.lower() in (".tif", ".tiff") else outpath.suffix.lstrip(".")
    save_kw = dict(dpi=DPI, bbox_inches="tight", facecolor="white")
    if fmt == "tiff":
        save_kw.update(format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(outpath, **save_kw)
    plt.close(fig)
    print(f"  Map saved: {outpath}")


def main():
    print("Generating Figure 2 — Geographic distribution map")

    counts = load_study_counts()
    print(f"  {len(counts)} countries with study data")

    world, name_col = load_world()
    print(f"  World map loaded: {len(world)} features")

    world_wt = world.to_crs(WINTRI)
    boundary = build_boundary()

    draw_map(world_wt, counts, name_col, boundary,
             SUBMISSION_DIR / "Figure_2.tif")

    print("Done.")


if __name__ == "__main__":
    main()
