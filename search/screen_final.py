#!/usr/bin/env python3
"""
screen_final.py – Final screening to produce a manageable full-text review list.

Strategy:
1. INCLUDE: 344 articles clearly about Ae. albopictus with quantitative data
2. CHECK: 134 Ae. albopictus articles with unclear quantitative data
3. Among the 933 "aedes_uncertain": re-screen abstracts more carefully
   to find hidden albopictus mentions or multi-species studies
4. Among management/cross-resistance: keep only those with Aedes/vector focus
5. Produce a final list of articles for full-text retrieval

Goal: reduce to ~300-500 articles for full-text review.
"""

import re
import pandas as pd
from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
INPUT_PHASE2 = SEARCH_DIR / "screening_phase2.csv"
INPUT_DATA = SEARCH_DIR / "search_records" / "pubmed_combined_deduplicated.csv"
OUTPUT_FINAL = SEARCH_DIR / "screening_final.csv"
OUTPUT_FULLTEXT_LIST = SEARCH_DIR / "fulltext_retrieval_list.csv"
OUTPUT_SUMMARY = SEARCH_DIR / "screening_final_summary.txt"


def has_pattern(text, patterns):
    if not text or pd.isna(text):
        return False
    text_str = str(text)
    for p in patterns:
        if re.search(p, text_str, re.IGNORECASE):
            return True
    return False


def count_pattern(text, patterns):
    if not text or pd.isna(text):
        return 0
    text_str = str(text)
    return sum(1 for p in patterns if re.search(p, text_str, re.IGNORECASE))


def main():
    print("=" * 60)
    print("Final Screening: Focused Full-Text List")
    print("=" * 60)

    # Load data
    phase2 = pd.read_csv(INPUT_PHASE2, dtype=str)
    data_df = pd.read_csv(INPUT_DATA, dtype=str)

    # Merge abstract data
    df = phase2.merge(
        data_df[["pmid", "abstract", "mesh_terms", "keywords", "pub_type"]],
        on="pmid", how="left", suffixes=("", "_raw")
    )
    print(f"  Loaded {len(df)} articles")

    # ── Patterns ──
    albo_strong = [
        r"\balbopictus\b",
        r"\basian\s+tiger\s+mosquito\b",
    ]

    # Patterns that hint albopictus might be in the full text
    albo_hint = [
        r"\baedes\s+spp\b",
        r"\baedes\s+species\b",
        r"\bmultiple\s+(aedes|mosquito)\s+species\b",
        r"\btwo\s+(aedes|mosquito)\s+species\b",
        r"\bseveral\s+(aedes|mosquito)\s+species\b",
        r"\binvasive\s+(aedes|mosquito)\b",
        r"\bcontainer[- ]breeding\b",
        r"\burban\s+(mosquito|vector)\b",
        r"\bdengue\s+(vector|mosquito|control)\b",
        r"\bchikungunya\b.*\bvector\b",
        r"\bzika\b.*\bvector\b",
        r"\baedes[- ]borne\b",
        r"\baedes\b.*\band\b.*\baedes\b",
    ]

    # Strong quantitative data patterns
    quant_strong = [
        r"\d+\.?\d*\s*%\s*(mortality|knockdown)",
        r"(mortality|knockdown)\s*[:=]\s*\d",
        r"\blc\s*50\s*[:=]\s*\d", r"\blc\s*95\b",
        r"\brr\s*50\b", r"\brr\s*95\b",
        r"\bresistance\s+ratio\b",
        r"\bf1534c\b", r"\bv1016g\b", r"\bs989p\b",
        r"\ballele\s+frequen\w*\s*[:=]?\s*0\.\d",
        r"\bn\s*=\s*\d{2,}\b",
        r"\bwho\s+(tube|bottle|standard)\s+(test|bioassay|assay)\b",
        r"\bcdc\s+bottle\s+(bioassay|assay)\b",
    ]

    # Bioassay/resistance methodology patterns
    method_patterns = [
        r"\bbioassay\b", r"\bdiagnostic\s+(dose|concentration)\b",
        r"\bdose[- ]response\b", r"\bprobit\b", r"\blogistic\b",
        r"\bsynergist\b.*\b(PBO|DEF|DEM)\b",
        r"\bpbo\b", r"\bpiperonyl\s+butoxide\b",
    ]

    # Cross-resistance specific (high relevance for our review)
    cross_res_strong = [
        r"\bcross[- ]resistan",
        r"\binsecticide\s+resistance\s+management\b",
        r"\brotation\b.*\binsecticid",
        r"\bmosaic\b.*\b(spray|treatment|insecticid)",
        r"\bintegrated\s+resistance\s+management\b",
        r"\birm\b.*\bstrateg",
    ]

    # ── Final classification ──
    final_decisions = []

    for _, row in df.iterrows():
        pmid = row["pmid"]
        cat = row.get("category", "")
        p2_dec = row.get("phase2_decision", "")
        abstract = str(row.get("abstract", ""))
        title = str(row.get("title", ""))
        mesh = str(row.get("mesh_terms", ""))
        full = f"{title} {abstract} {mesh} {str(row.get('keywords', ''))}"

        # Already excluded → keep excluded
        if p2_dec == "Exclude":
            final_decisions.append({
                "pmid": pmid,
                "final_decision": "Exclude",
                "final_reason": row.get("phase2_reason", ""),
                "final_category": "excluded",
                "priority": 0,
            })
            continue

        # Category: albopictus_quantitative → INCLUDE (priority 1)
        if cat == "albopictus_quantitative":
            final_decisions.append({
                "pmid": pmid,
                "final_decision": "Include",
                "final_reason": "Ae. albopictus + quantitative resistance data",
                "final_category": "core_albopictus",
                "priority": 1,
            })
            continue

        # Category: albopictus_uncertain → need full-text (priority 2)
        if cat == "albopictus_uncertain":
            final_decisions.append({
                "pmid": pmid,
                "final_decision": "FullText",
                "final_reason": "Ae. albopictus mentioned but quantitative data unclear",
                "final_category": "albopictus_check",
                "priority": 2,
            })
            continue

        # Category: aedes_uncertain → refined re-screening
        if cat == "aedes_uncertain":
            # Check if abstract hints at multi-species including albopictus
            has_albo_hint = has_pattern(full, albo_hint)
            has_quant = count_pattern(abstract, quant_strong) >= 2
            has_method = has_pattern(full, method_patterns)
            has_cross = has_pattern(full, cross_res_strong)

            if has_albo_hint and (has_quant or has_method):
                final_decisions.append({
                    "pmid": pmid,
                    "final_decision": "FullText",
                    "final_reason": "Aedes spp. with albopictus hints + quantitative data",
                    "final_category": "aedes_possible_albopictus",
                    "priority": 3,
                })
            elif has_cross:
                final_decisions.append({
                    "pmid": pmid,
                    "final_decision": "FullText",
                    "final_reason": "Cross-resistance topic in Aedes context",
                    "final_category": "aedes_cross_resistance",
                    "priority": 4,
                })
            elif has_quant and has_method:
                final_decisions.append({
                    "pmid": pmid,
                    "final_decision": "FullText",
                    "final_reason": "Aedes with strong quantitative resistance data",
                    "final_category": "aedes_quantitative",
                    "priority": 5,
                })
            else:
                final_decisions.append({
                    "pmid": pmid,
                    "final_decision": "Exclude",
                    "final_reason": "Aedes spp. without albopictus hint or sufficient data indicators",
                    "final_category": "aedes_excluded",
                    "priority": 0,
                })
            continue

        # Category: aegypti_cross_resistance → keep if cross-resistance focused
        if cat == "aegypti_cross_resistance":
            if has_pattern(full, cross_res_strong):
                final_decisions.append({
                    "pmid": pmid,
                    "final_decision": "FullText",
                    "final_reason": "Ae. aegypti cross-resistance (comparison data)",
                    "final_category": "aegypti_cross_res",
                    "priority": 4,
                })
            else:
                final_decisions.append({
                    "pmid": pmid,
                    "final_decision": "Exclude",
                    "final_reason": "Ae. aegypti without strong cross-resistance focus",
                    "final_category": "aegypti_excluded",
                    "priority": 0,
                })
            continue

        # Category: management_relevant
        if cat == "management_relevant":
            # Keep only if specifically about Aedes or insecticide resistance management
            if has_pattern(full, cross_res_strong) and has_pattern(full, [r"\baedes\b", r"\bmosquito\b"]):
                final_decisions.append({
                    "pmid": pmid,
                    "final_decision": "FullText",
                    "final_reason": "Resistance management relevant to Aedes/mosquitoes",
                    "final_category": "management_keep",
                    "priority": 5,
                })
            else:
                final_decisions.append({
                    "pmid": pmid,
                    "final_decision": "Exclude",
                    "final_reason": "Management topic without sufficient Aedes/resistance specificity",
                    "final_category": "management_excluded",
                    "priority": 0,
                })
            continue

        # Catch-all
        final_decisions.append({
            "pmid": pmid,
            "final_decision": "Exclude",
            "final_reason": "Not matching refined criteria",
            "final_category": "other_excluded",
            "priority": 0,
        })

    final_df = pd.DataFrame(final_decisions)

    # Merge with article metadata
    out_df = phase2[["pmid", "title", "first_author", "year", "journal", "doi",
                      "has_albopictus", "relevance_score"]].merge(
        final_df, on="pmid", how="left"
    )
    out_df["relevance_score"] = pd.to_numeric(out_df["relevance_score"], errors="coerce")

    # Sort by priority then relevance
    out_df = out_df.sort_values(
        ["priority", "relevance_score"], ascending=[True, False]
    )

    out_df.to_csv(OUTPUT_FINAL, index=False)
    print(f"\n  Saved: {OUTPUT_FINAL}")

    # ── Full-text retrieval list ──
    ft_df = out_df[out_df["final_decision"].isin(["Include", "FullText"])].copy()
    ft_df = ft_df.sort_values(["priority", "relevance_score"], ascending=[True, False])
    ft_df.to_csv(OUTPUT_FULLTEXT_LIST, index=False)
    print(f"  Saved: {OUTPUT_FULLTEXT_LIST}")

    # ── Summary ──
    n_include = (out_df["final_decision"] == "Include").sum()
    n_fulltext = (out_df["final_decision"] == "FullText").sum()
    n_exclude = (out_df["final_decision"] == "Exclude").sum()

    cat_counts = final_df["final_category"].value_counts()

    summary = []
    summary.append("=" * 60)
    summary.append("FINAL SCREENING SUMMARY")
    summary.append("=" * 60)
    summary.append(f"\nTotal articles screened: {len(out_df)}")
    summary.append(f"\n  INCLUDE (core Ae. albopictus):     {n_include:>5}")
    summary.append(f"  FULL-TEXT REVIEW needed:            {n_fulltext:>5}")
    summary.append(f"  EXCLUDE:                            {n_exclude:>5}")
    summary.append(f"\n  Total for full-text retrieval: {n_include + n_fulltext}")
    summary.append(f"\n{'─' * 50}")
    summary.append("Category breakdown:")
    for cat, cnt in cat_counts.sort_values(ascending=False).items():
        summary.append(f"  {cat}: {cnt}")

    summary.append(f"\n{'─' * 50}")
    summary.append("Full-text retrieval priority:")
    for p in sorted(ft_df["priority"].unique()):
        n = (ft_df["priority"] == p).sum()
        cats = ft_df[ft_df["priority"] == p]["final_category"].unique()
        summary.append(f"  Priority {int(p)}: {n} articles ({', '.join(cats)})")

    # Year distribution
    incl = out_df[out_df["final_decision"].isin(["Include", "FullText"])]
    year_dist = incl["year"].value_counts().sort_index()
    summary.append(f"\n{'─' * 50}")
    summary.append("Articles for full-text review by year:")
    for yr, cnt in year_dist.items():
        summary.append(f"  {yr}: {cnt}")

    # PRISMA numbers
    summary.append(f"\n{'─' * 50}")
    summary.append("PRISMA Flow Numbers:")
    summary.append(f"  Records identified: 2765 (PubMed: 2002, Other: 763)")
    summary.append(f"  Duplicates removed: 172")
    summary.append(f"  Records screened (title/abstract): {len(out_df)}")
    summary.append(f"  Records excluded (title/abstract): {n_exclude}")
    summary.append(f"  Full-text articles to assess: {n_include + n_fulltext}")

    summary_text = "\n".join(summary)
    print(summary_text)

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(summary_text)
    print(f"\n  Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
