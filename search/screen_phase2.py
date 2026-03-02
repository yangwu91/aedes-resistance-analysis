#!/usr/bin/env python3
"""
screen_phase2.py – Refined screening focused on Aedes albopictus.

Takes the initial screening_log.csv and applies more stringent criteria:
1. Must mention Ae. albopictus (or be a cross-resistance/management study
   explicitly relevant to Aedes mosquitoes)
2. Must show evidence of quantitative resistance data in abstract
3. Must be original research with field population data

Produces:
  - screening_phase2.csv  (refined decisions)
  - articles_for_fulltext.csv  (articles to retrieve full text)
"""

import re
import pandas as pd
from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
INPUT_LOG = SEARCH_DIR / "screening_log.csv"
INPUT_DATA = SEARCH_DIR / "search_records" / "pubmed_combined_deduplicated.csv"
OUTPUT_PHASE2 = SEARCH_DIR / "screening_phase2.csv"
OUTPUT_FULLTEXT = SEARCH_DIR / "articles_for_fulltext.csv"
OUTPUT_SUMMARY = SEARCH_DIR / "screening_phase2_summary.txt"


def has_pattern(text, patterns):
    """Check if any regex pattern matches in text."""
    if not text or pd.isna(text):
        return False
    text_str = str(text)
    for p in patterns:
        if re.search(p, text_str, re.IGNORECASE):
            return True
    return False


def main():
    print("=" * 60)
    print("Phase 2: Refined Screening (Ae. albopictus Focus)")
    print("=" * 60)

    # Load initial screening results
    log_df = pd.read_csv(INPUT_LOG, dtype=str)
    # Load full article data for abstract access
    data_df = pd.read_csv(INPUT_DATA, dtype=str)

    # Merge to get full text
    df = log_df.merge(
        data_df[["pmid", "abstract", "mesh_terms", "keywords", "pub_type"]],
        on="pmid", how="left", suffixes=("", "_full")
    )
    print(f"  Loaded {len(df)} articles")

    # ── Ae. albopictus patterns ──
    albo_patterns = [
        r"\balbopictus\b",
        r"\baedes\s+albopictus\b",
        r"\bae\.\s*albopictus\b",
        r"\basian\s+tiger\s+mosquito\b",
        r"\bstegomyia\s+albopictus\b",
    ]

    # Ae. aegypti patterns (include for comparison when cross-resistance is topic)
    aegypti_patterns = [
        r"\baegypti\b",
        r"\baedes\s+aegypti\b",
        r"\bae\.\s*aegypti\b",
    ]

    # Aedes genus general
    aedes_patterns = [
        r"\baedes\b", r"\bstegomyia\b",
    ]

    # Quantitative data indicators in abstract
    quant_patterns = [
        r"\d+\s*%\s*(mortality|knockdown|kill|dead|survival)",
        r"(mortality|knockdown|kill)\s*(rate|percent)?\s*[=:of]*\s*\d",
        r"\blc\s*[_]?\s*50\b", r"\blc\s*[_]?\s*95\b",
        r"\bld\s*50\b",
        r"\brr\s*[_]?\s*50\b", r"\brr\s*[_]?\s*95\b",
        r"\bresistance\s+ratio\s*[=:]*\s*\d",
        r"\ballele\s+frequen\w*\s*[=:]*\s*[\d.]",
        r"\bfrequen\w*\s*(of|=|:)\s*[\d.]",
        r"\bf1534c\b", r"\bv1016g\b", r"\bs989p\b", r"\bl1014\w?\b",
        r"\bn\s*=\s*\d{2,}",  # sample size ≥ 10
        r"\b\d+\s*(individuals|mosquito|specimens|females|males)\s*(were|was)?\s*(tested|assayed|genotyped|screened)",
        r"\bbioassay\b.*\d+",
        r"\bwho\s+(tube|bottle|standard)\b",
        r"\bcdc\s+bottle\b",
        r"\bdiagnostic\s+(dose|concentration)\b",
        r"\bfold\s*(increase|change|higher|elevation)\b",
        r"\bresistan\w+\s+to\s+\w+\s+was\b",
        r"\bconfirmed\s+resistan",
        r"\bsusceptib\w+\s+to\b",
    ]

    # Cross-resistance / resistance management (keep even if not albopictus-specific)
    cross_res_patterns = [
        r"\bcross[- ]resistan",
        r"\bresistance\s+management\b",
        r"\binsecticide\s+rotation\b",
        r"\bresistance\s+pattern\b",
        r"\bmosaic\s+(spray|treatment|deployment)\b",
        r"\bintegrated\s+(resistance|vector)\s+management\b",
    ]

    # ── Second pass screening ──
    decisions = []
    for _, row in df.iterrows():
        full_text = " ".join(str(row.get(c, "")) for c in
                            ["title", "abstract", "mesh_terms", "keywords"])
        abstract = str(row.get("abstract", ""))
        title = str(row.get("title", ""))

        # Skip already-excluded articles from Phase 1
        if row.get("decision") == "Exclude":
            decisions.append({
                "pmid": row["pmid"],
                "phase2_decision": "Exclude",
                "phase2_reason": f"Phase 1 exclusion: {row.get('reason', '')}",
                "category": "excluded_phase1",
            })
            continue

        # Check species
        is_albopictus = has_pattern(full_text, albo_patterns)
        is_aegypti = has_pattern(full_text, aegypti_patterns)
        is_aedes = has_pattern(full_text, aedes_patterns)
        is_cross_res = has_pattern(full_text, cross_res_patterns)
        has_quant = has_pattern(abstract, quant_patterns) or has_pattern(title, quant_patterns)

        # Category 1: Ae. albopictus with quantitative data → INCLUDE
        if is_albopictus and has_quant:
            decisions.append({
                "pmid": row["pmid"],
                "phase2_decision": "Include",
                "phase2_reason": "Ae. albopictus + quantitative resistance data",
                "category": "albopictus_quantitative",
            })
            continue

        # Category 2: Ae. albopictus without clear quantitative data → UNCERTAIN (need full text)
        if is_albopictus and not has_quant:
            # Check if abstract mentions resistance/insecticides at all
            if has_pattern(full_text, [r"\bresistan", r"\bsusceptib", r"\bbioassay", r"\bkdr\b", r"\binsecticid"]):
                decisions.append({
                    "pmid": row["pmid"],
                    "phase2_decision": "Uncertain",
                    "phase2_reason": "Ae. albopictus + resistance topic, but no clear quantitative data in abstract",
                    "category": "albopictus_uncertain",
                })
            else:
                decisions.append({
                    "pmid": row["pmid"],
                    "phase2_decision": "Exclude",
                    "phase2_reason": "Ae. albopictus but not about resistance",
                    "category": "albopictus_non_resistance",
                })
            continue

        # Category 3: Ae. aegypti cross-resistance or management study → UNCERTAIN
        if is_aegypti and is_cross_res:
            decisions.append({
                "pmid": row["pmid"],
                "phase2_decision": "Uncertain",
                "phase2_reason": "Ae. aegypti cross-resistance/management (potential comparison data)",
                "category": "aegypti_cross_resistance",
            })
            continue

        # Category 4: Aedes genus (species not specified) with resistance → UNCERTAIN
        if is_aedes and has_quant:
            decisions.append({
                "pmid": row["pmid"],
                "phase2_decision": "Uncertain",
                "phase2_reason": "Aedes spp. with quantitative data (species may include albopictus)",
                "category": "aedes_uncertain",
            })
            continue

        # Category 5: Cross-resistance/management review relevant to mosquitoes
        if is_cross_res and has_pattern(full_text, [r"\bmosquito", r"\baedes\b", r"\bvector\b"]):
            decisions.append({
                "pmid": row["pmid"],
                "phase2_decision": "Uncertain",
                "phase2_reason": "Cross-resistance/management topic relevant to mosquitoes",
                "category": "management_relevant",
            })
            continue

        # Category 6: Other mosquito species only → EXCLUDE
        decisions.append({
            "pmid": row["pmid"],
            "phase2_decision": "Exclude",
            "phase2_reason": "Not about Ae. albopictus and no cross-resistance/management relevance",
            "category": "other_species_only",
        })

    phase2_df = pd.DataFrame(decisions)

    # Merge with original data
    out_df = log_df[["pmid", "title", "first_author", "year", "journal", "doi",
                      "decision", "reason", "relevance_score", "has_albopictus"]].merge(
        phase2_df, on="pmid", how="left"
    )

    # Sort
    order = {"Include": 0, "Uncertain": 1, "Exclude": 2}
    out_df["_order"] = out_df["phase2_decision"].map(order)
    out_df = out_df.sort_values(
        ["_order", "relevance_score"], ascending=[True, False]
    ).drop(columns=["_order"])
    out_df["relevance_score"] = pd.to_numeric(out_df["relevance_score"], errors="coerce")

    out_df.to_csv(OUTPUT_PHASE2, index=False)
    print(f"\n  Saved: {OUTPUT_PHASE2}")

    # ── Articles for full-text screening ──
    fulltext_df = out_df[out_df["phase2_decision"].isin(["Include", "Uncertain"])].copy()
    fulltext_df = fulltext_df.sort_values("relevance_score", ascending=False)
    fulltext_df.to_csv(OUTPUT_FULLTEXT, index=False)
    print(f"  Saved: {OUTPUT_FULLTEXT}")

    # ── Summary ──
    n_include = (out_df["phase2_decision"] == "Include").sum()
    n_uncertain = (out_df["phase2_decision"] == "Uncertain").sum()
    n_exclude = (out_df["phase2_decision"] == "Exclude").sum()

    cat_counts = phase2_df["category"].value_counts()

    summary = []
    summary.append("=" * 60)
    summary.append("PHASE 2 SCREENING SUMMARY (Ae. albopictus Focus)")
    summary.append("=" * 60)
    summary.append(f"\nTotal articles: {len(out_df)}")
    summary.append(f"\n  Include (albopictus + quantitative data):  {n_include}")
    summary.append(f"  Uncertain (need full-text review):          {n_uncertain}")
    summary.append(f"  Exclude:                                    {n_exclude}")
    summary.append(f"\nArticles to retrieve full text: {n_include + n_uncertain}")
    summary.append(f"\nCategory breakdown:")
    for cat, cnt in cat_counts.sort_values(ascending=False).items():
        summary.append(f"  {cat}: {cnt}")

    # Year distribution of included
    incl = out_df[out_df["phase2_decision"] == "Include"]
    if len(incl) > 0:
        year_dist = incl["year"].value_counts().sort_index()
        summary.append(f"\nIncluded articles by year:")
        for yr, cnt in year_dist.items():
            summary.append(f"  {yr}: {cnt}")

    summary_text = "\n".join(summary)
    print(summary_text)

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(summary_text)
    print(f"\n  Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
