#!/usr/bin/env python3
"""
Generate a PRISMA 2020 flow diagram for the systematic review.
Update the numbers below after completing each screening phase.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# UPDATE THESE NUMBERS after each screening phase
# ──────────────────────────────────────────────────────────────────────
N_PUBMED = 2002
N_SCOPUS = 0       # TODO: update after Scopus search
N_WOS = 0          # TODO: update after WoS search
N_EMBASE = 0       # TODO: update after Embase search
N_OTHER = 763      # broad search + IR Mapper + citation tracking

N_TOTAL = N_PUBMED + N_SCOPUS + N_WOS + N_EMBASE + N_OTHER
N_DUPLICATES = 172
N_AFTER_DEDUP = N_TOTAL - N_DUPLICATES

N_SCREENED = N_AFTER_DEDUP
N_EXCLUDED_TITLE_ABSTRACT = 1439  # automated title/abstract screening
N_FULLTEXT_ASSESSED = N_SCREENED - N_EXCLUDED_TITLE_ABSTRACT  # 1154
N_EXCLUDED_FULLTEXT = 0         # TODO: update after full-text screening
N_INCLUDED = 0                  # TODO: update after full-text screening

# Exclusion reasons at full-text stage
EXCLUSION_REASONS = {
    "Not original research (E1)": 0,
    "No quantitative data (E2)": 0,
    "Non-mosquito species (E3)": 0,
    "Lab strains only (E5)": 0,
    "Insufficient data (E7)": 0,
    "Sample size < 20 (E8)": 0,
    "Duplicate data (E9)": 0,
    "Other (E12)": 0,
}
# ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "05_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, text, color="#E8F4FD", fontsize=9, bold=False):
    """Draw a box with centered text."""
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="#333333", linewidth=1.2,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True,
            multialignment="center")


def draw_arrow(ax, x1, y1, x2, y2):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))


def main():
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # Title
    ax.text(7, 15.5, "PRISMA 2020 Flow Diagram", ha="center", va="center",
            fontsize=14, fontweight="bold")

    # ── Identification ──
    ax.text(0.5, 14.5, "Identification", fontsize=11, fontweight="bold",
            rotation=90, va="center", color="#0072B2")

    draw_box(ax, 4, 14.5, 5.5, 1.2,
             f"Records identified through\ndatabase searching\n(n = {N_TOTAL})\n\n"
             f"PubMed: {N_PUBMED}  |  Scopus: {N_SCOPUS}\n"
             f"WoS: {N_WOS}  |  Embase: {N_EMBASE}",
             color="#D4E8F7")

    draw_box(ax, 10.5, 14.5, 3.5, 1.2,
             f"Additional records from\nother sources\n(n = {N_OTHER})\n\n"
             f"IR Mapper, citation tracking,\ngrey literature",
             color="#D4E8F7")

    # Arrow down
    draw_arrow(ax, 4, 13.9, 4, 13.1)
    draw_arrow(ax, 10.5, 13.9, 7, 13.1)

    # ── Duplicates removed ──
    draw_box(ax, 7, 12.5, 5, 0.8,
             f"Records after duplicates removed\n(n = {N_AFTER_DEDUP})",
             color="#E8F4FD")

    draw_box(ax, 12, 12.5, 2.5, 0.8,
             f"Duplicates removed\n(n = {N_DUPLICATES})",
             color="#FDE8E8")
    draw_arrow(ax, 9.5, 12.5, 10.75, 12.5)

    # ── Screening ──
    draw_arrow(ax, 7, 12.1, 7, 11.4)

    ax.text(0.5, 11, "Screening", fontsize=11, fontweight="bold",
            rotation=90, va="center", color="#0072B2")

    draw_box(ax, 5, 11, 4.5, 0.8,
             f"Records screened\n(title & abstract)\n(n = {N_SCREENED})",
             color="#E8F4FD")

    draw_box(ax, 12, 11, 2.5, 0.8,
             f"Records excluded\n(n = {N_EXCLUDED_TITLE_ABSTRACT})",
             color="#FDE8E8")
    draw_arrow(ax, 7.25, 11, 10.75, 11)

    # Arrow down
    draw_arrow(ax, 5, 10.6, 5, 9.9)

    # ── Eligibility ──
    ax.text(0.5, 9, "Eligibility", fontsize=11, fontweight="bold",
            rotation=90, va="center", color="#0072B2")

    draw_box(ax, 5, 9.5, 4.5, 0.8,
             f"Full-text articles\nassessed for eligibility\n(n = {N_FULLTEXT_ASSESSED})",
             color="#E8F4FD")

    # Exclusion reasons box
    reasons_text = f"Full-text articles excluded\n(n = {N_EXCLUDED_FULLTEXT})\n\n"
    for reason, count in EXCLUSION_REASONS.items():
        if count > 0:
            reasons_text += f"  {reason}: {count}\n"
    if N_EXCLUDED_FULLTEXT == 0:
        reasons_text += "  [TO BE UPDATED]"

    draw_box(ax, 12, 9, 3, 1.8,
             reasons_text, color="#FDE8E8", fontsize=8)
    draw_arrow(ax, 7.25, 9.5, 10.5, 9.5)

    # Arrow down
    draw_arrow(ax, 5, 9.1, 5, 8.1)

    # ── Included ──
    ax.text(0.5, 7, "Included", fontsize=11, fontweight="bold",
            rotation=90, va="center", color="#0072B2")

    n_text = f"Studies included in\nqualitative synthesis\n(n = {N_INCLUDED})"
    draw_box(ax, 5, 7.5, 4.5, 0.8, n_text, color="#D4F7D4", bold=True)

    draw_arrow(ax, 5, 7.1, 5, 6.3)

    draw_box(ax, 5, 5.8, 4.5, 0.8,
             f"Studies included in\nmeta-analysis\n(n = {N_INCLUDED})",
             color="#D4F7D4", bold=True)

    # Data breakdown
    draw_arrow(ax, 5, 5.4, 5, 4.5)
    breakdown = (
        "Data breakdown:\n"
        "  Mortality data: [N] studies\n"
        "  Resistance ratio data: [N] studies\n"
        "  kdr mutation data: [N] studies\n"
        "  Enzyme activity data: [N] studies"
    )
    draw_box(ax, 5, 3.8, 5, 1.2, breakdown, color="#FFF8DC", fontsize=8)

    plt.tight_layout()
    outpath = OUTPUT_DIR / "prisma_flowchart.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"PRISMA flow diagram saved: {outpath}")


if __name__ == "__main__":
    main()
