#!/usr/bin/env python3
"""
Generate a PRISMA 2020 flow diagram for the systematic review.
Produces a clean, grid-aligned flowchart with all final numbers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# PRISMA FLOW NUMBERS (finalised)
# ──────────────────────────────────────────────────────────────────────
N_PUBMED = 2002
N_OTHER = 763  # broad search + IR Mapper + citation tracking + grey lit

N_TOTAL = N_PUBMED + N_OTHER  # 2765
N_DUPLICATES = 172
N_AFTER_DEDUP = N_TOTAL - N_DUPLICATES  # 2593

N_SCREENED = N_AFTER_DEDUP  # 2593
N_EXCLUDED_SCREENING = 1439
N_FULLTEXT_ASSESSED = N_SCREENED - N_EXCLUDED_SCREENING  # 1154

N_INCLUDED = 132
N_EXCLUDED_FULLTEXT = N_FULLTEXT_ASSESSED - N_INCLUDED  # 1022

# Exclusion reasons at full-text stage (approximate breakdown)
EXCLUSION_REASONS = [
    ("Not Ae. albopictus specific", 676),
    ("No quantitative resistance data", 290),
    ("Insufficient data for pooling", 52),
    ("Non-English language", 4),
]

# Data indicator breakdown
N_MORTALITY_STUDIES = 116
N_MORTALITY_OBS = 5757
N_RR_STUDIES = 30
N_RR_OBS = 884
N_KDR_STUDIES = 19
N_KDR_OBS = 297
N_ENZYME_STUDIES = 4
N_ENZYME_OBS = 26

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "05_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────

def draw_box(ax, cx, cy, w, h, text, *,
             color="#E8F4FD", edge="#555555", fontsize=8.5,
             bold=False, align="center"):
    """Draw a rounded rectangle with centered multiline text."""
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=color, edgecolor=edge, linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text,
            ha=align, va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            linespacing=1.35,
            multialignment="center")


def arrow_down(ax, x, y1, y2):
    """Vertical downward arrow."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444444",
                                lw=1.0, shrinkA=0, shrinkB=0))


def arrow_right(ax, x1, y, x2):
    """Horizontal rightward arrow."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#444444",
                                lw=1.0, shrinkA=0, shrinkB=0))


def arrow_corner(ax, x1, y1, x2, y2):
    """Arrow that goes down then turns right (L-shaped)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444444",
                                lw=1.0, connectionstyle="arc3,rad=0",
                                shrinkA=0, shrinkB=0))


# ──────────────────────────────────────────────────────────────────────
# Layout constants — grid-based for precise alignment
# ──────────────────────────────────────────────────────────────────────
# Canvas: 12 wide × 18 tall
W = 12
H = 18

# Column centres
COL_LEFT = 2.0       # left ID box
COL_MAIN = 4.8       # main flow column
COL_RIGHT = 9.5      # exclusion / side boxes

# Box dimensions
BW_MAIN = 4.2        # main box width
BW_SIDE = 3.2        # side box width
BH = 0.85            # standard box height
BH_TALL = 1.1        # taller boxes

# Row centres (top to bottom)
ROW_ID = 16.2        # identification
ROW_DEDUP = 14.5     # after dedup
ROW_SCREEN = 12.8    # screening
ROW_ELIG = 11.0      # eligibility
ROW_INCL_Q = 8.8     # qualitative synthesis
ROW_INCL_M = 7.2     # meta-analysis
ROW_BREAK = 5.5      # data breakdown

# Phase label x
PHASE_X = 0.35

# Colours
C_ID = "#DAEAF6"
C_SCREEN = "#E8F4FD"
C_EXCL = "#FDEAEA"
C_INCL = "#D5F5D5"
C_DATA = "#FFF9E3"


def main():
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, W)
    ax.set_ylim(4.0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Title ──
    ax.text(W / 2, H - 0.3, "PRISMA 2020 Flow Diagram",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # ── Phase labels (left sidebar) ──
    phase_props = dict(fontsize=9.5, fontweight="bold", rotation=90,
                       va="center", ha="center", color="#0072B2")
    ax.text(PHASE_X, ROW_ID, "Identification", **phase_props)
    ax.text(PHASE_X, (ROW_DEDUP + ROW_SCREEN) / 2, "Screening", **phase_props)
    ax.text(PHASE_X, ROW_ELIG, "Eligibility", **phase_props)
    ax.text(PHASE_X, (ROW_INCL_Q + ROW_INCL_M) / 2, "Included", **phase_props)

    # Thin vertical separator line
    ax.plot([0.85, 0.85], [4.5, H - 0.7], color="#CCCCCC", lw=0.6, ls="--")

    # ══════════════════════════════════════════════════════════════
    # ROW 1 — Identification
    # ══════════════════════════════════════════════════════════════
    draw_box(ax, COL_LEFT + 1.0, ROW_ID, BW_MAIN, BH_TALL,
             f"Records identified through\ndatabase searching\n(n = {N_PUBMED})\n"
             f"PubMed/MEDLINE",
             color=C_ID)

    draw_box(ax, COL_RIGHT, ROW_ID, BW_SIDE, BH_TALL,
             f"Additional records from\nother sources\n(n = {N_OTHER})\n"
             f"IR Mapper, citation tracking",
             color=C_ID)

    # Arrows down to dedup — both go straight down to a merge point,
    # then a single arrow enters the dedup box.
    merge_y = ROW_DEDUP + BH / 2 + 0.6
    arrow_down(ax, COL_LEFT + 1.0, ROW_ID - BH_TALL / 2, merge_y)
    arrow_down(ax, COL_RIGHT, ROW_ID - BH_TALL / 2, merge_y)
    # Horizontal lines to merge
    ax.plot([COL_LEFT + 1.0, COL_MAIN], [merge_y, merge_y],
            color="#444444", lw=1.0)
    ax.plot([COL_RIGHT, COL_MAIN], [merge_y, merge_y],
            color="#444444", lw=1.0)
    # Single arrow down from merge to dedup box
    arrow_down(ax, COL_MAIN, merge_y, ROW_DEDUP + BH / 2)

    # ══════════════════════════════════════════════════════════════
    # ROW 2 — Deduplication
    # ══════════════════════════════════════════════════════════════
    draw_box(ax, COL_MAIN, ROW_DEDUP, BW_MAIN, BH,
             f"Records after duplicates removed\n(n = {N_AFTER_DEDUP:,})",
             color=C_SCREEN)

    draw_box(ax, COL_RIGHT, ROW_DEDUP, BW_SIDE, BH,
             f"Duplicates removed\n(n = {N_DUPLICATES})",
             color=C_EXCL)
    arrow_right(ax, COL_MAIN + BW_MAIN / 2, ROW_DEDUP,
                COL_RIGHT - BW_SIDE / 2)

    # Arrow down
    arrow_down(ax, COL_MAIN, ROW_DEDUP - BH / 2, ROW_SCREEN + BH / 2)

    # ══════════════════════════════════════════════════════════════
    # ROW 3 — Screening
    # ══════════════════════════════════════════════════════════════
    draw_box(ax, COL_MAIN, ROW_SCREEN, BW_MAIN, BH,
             f"Records screened\n(title & abstract)\n(n = {N_SCREENED:,})",
             color=C_SCREEN)

    draw_box(ax, COL_RIGHT, ROW_SCREEN, BW_SIDE, BH,
             f"Records excluded\n(n = {N_EXCLUDED_SCREENING:,})",
             color=C_EXCL)
    arrow_right(ax, COL_MAIN + BW_MAIN / 2, ROW_SCREEN,
                COL_RIGHT - BW_SIDE / 2)

    # Arrow down
    arrow_down(ax, COL_MAIN, ROW_SCREEN - BH / 2, ROW_ELIG + BH / 2)

    # ══════════════════════════════════════════════════════════════
    # ROW 4 — Eligibility
    # ══════════════════════════════════════════════════════════════
    draw_box(ax, COL_MAIN, ROW_ELIG, BW_MAIN, BH,
             f"Full-text articles assessed\nfor eligibility\n(n = {N_FULLTEXT_ASSESSED:,})",
             color=C_SCREEN)

    # Exclusion reasons box (taller)
    reasons_lines = [f"Full-text articles excluded\n(n = {N_EXCLUDED_FULLTEXT:,})\n"]
    for reason, count in EXCLUSION_REASONS:
        reasons_lines.append(f"{reason}: {count}")
    reasons_text = "\n".join(reasons_lines)

    excl_h = 1.5
    draw_box(ax, COL_RIGHT, ROW_ELIG - 0.3, BW_SIDE + 0.3, excl_h,
             reasons_text, color=C_EXCL, fontsize=7.5)
    arrow_right(ax, COL_MAIN + BW_MAIN / 2, ROW_ELIG,
                COL_RIGHT - (BW_SIDE + 0.3) / 2)

    # Arrow down
    arrow_down(ax, COL_MAIN, ROW_ELIG - BH / 2, ROW_INCL_Q + BH / 2)

    # ══════════════════════════════════════════════════════════════
    # ROW 5 — Included (qualitative)
    # ══════════════════════════════════════════════════════════════
    draw_box(ax, COL_MAIN, ROW_INCL_Q, BW_MAIN, BH,
             f"Studies included in\nqualitative synthesis\n(n = {N_INCLUDED})",
             color=C_INCL, bold=True)

    # Arrow down
    arrow_down(ax, COL_MAIN, ROW_INCL_Q - BH / 2, ROW_INCL_M + BH / 2)

    # ══════════════════════════════════════════════════════════════
    # ROW 6 — Included (meta-analysis)
    # ══════════════════════════════════════════════════════════════
    draw_box(ax, COL_MAIN, ROW_INCL_M, BW_MAIN, BH,
             f"Studies included in\nmeta-analysis\n(n = {N_INCLUDED})",
             color=C_INCL, bold=True)

    # Arrow down
    arrow_down(ax, COL_MAIN, ROW_INCL_M - BH / 2, ROW_BREAK + BH_TALL / 2)

    # ══════════════════════════════════════════════════════════════
    # ROW 7 — Data breakdown
    # ══════════════════════════════════════════════════════════════
    breakdown = (
        f"Data breakdown:\n"
        f"Mortality data: {N_MORTALITY_STUDIES} studies "
        f"({N_MORTALITY_OBS:,} observations)\n"
        f"Resistance ratio data: {N_RR_STUDIES} studies "
        f"({N_RR_OBS} observations)\n"
        f"kdr mutation data: {N_KDR_STUDIES} studies "
        f"({N_KDR_OBS} observations)\n"
        f"Enzyme activity data: {N_ENZYME_STUDIES} studies "
        f"({N_ENZYME_OBS} observations)"
    )
    draw_box(ax, COL_MAIN, ROW_BREAK, BW_MAIN + 1.0, BH_TALL + 0.3,
             breakdown, color=C_DATA, fontsize=7.5)

    # ── Save ──
    plt.tight_layout()
    outpath = OUTPUT_DIR / "prisma_flowchart.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"PRISMA flow diagram saved: {outpath}")


if __name__ == "__main__":
    main()
