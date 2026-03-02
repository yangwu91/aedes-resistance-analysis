#!/usr/bin/env python3
"""
screen_articles.py – Automated title/abstract screening for the systematic review.

Applies inclusion/exclusion criteria to the deduplicated PubMed search results
using keyword matching on title, abstract, MeSH terms, and publication type.

Outputs:
  - screening_log.csv  (all articles with screening decisions)
  - screening_summary.txt (counts and statistics)
"""

import re
import csv
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──
SEARCH_DIR = Path(__file__).resolve().parent / "search_records"
INPUT_FILE = SEARCH_DIR / "pubmed_combined_deduplicated.csv"
OUTPUT_LOG = Path(__file__).resolve().parent / "screening_log.csv"
OUTPUT_SUMMARY = Path(__file__).resolve().parent / "screening_summary.txt"

# ── Keywords for screening ──

# Species keywords (case-insensitive)
SPECIES_MOSQUITO = [
    r"\baedes\b", r"\bae\.\s*albopictus\b", r"\bae\.\s*aegypti\b",
    r"\balbopictus\b", r"\baegypti\b",
    r"\bculex\b", r"\bcx\.\b", r"\banopheles\b", r"\ban\.\b",
    r"\bmosquito", r"\bculicid",
    r"\bstegomyia\b",  # old genus name for Aedes
    r"\bvector\s*(control|resistance|management|species|population|borne)",
]

# Primary target species
TARGET_SPECIES = [
    r"\balbopictus\b", r"\baedes\s+albopictus\b",
    r"\bae\.\s*albopictus\b", r"\basian\s+tiger\s+mosquito\b",
    r"\bstegomyia\s+albopictus\b",
]

# Resistance/susceptibility keywords
RESISTANCE_KEYWORDS = [
    r"\bresistan", r"\bsusceptib", r"\bbioassay",
    r"\bkdr\b", r"\bknockdown\s+resistance\b",
    r"\bmortality\s*(rate|data|percent)", r"\blethal\s+concentration",
    r"\blc\s*50\b", r"\blc\s*95\b", r"\bld\s*50\b",
    r"\brr\s*50\b", r"\brr\s*95\b", r"\bresistance\s+ratio",
    r"\bdiagnostic\s+(dose|concentration)",
    r"\bkt\s*50\b", r"\bkt\s*95\b",
    r"\bdose[- ]response\b", r"\bprobit\b",
    r"\ballele\s+frequen", r"\bmutation\s+frequen",
    r"\bgenotyp", r"\bf1534c\b", r"\bv1016g\b", r"\bs989p\b",
    r"\bvoltage[- ]gated\s+sodium\b", r"\bvgsc\b",
    r"\bmetabolic\s+resistance\b", r"\benzyme\s+activity\b",
    r"\bcytochrome\s+p450\b", r"\besterase\b",
    r"\bglutathione\s+s[- ]transferase\b", r"\bgst\b",
    r"\bacetylcholinesterase\b", r"\bache\b",
    r"\bmixed[- ]function\s+oxidase\b", r"\bmfo\b",
    r"\bcross[- ]resistan",
    r"\binsecticide\s+resistance\s+management\b",
    r"\bwho\s+(tube|bottle)\s+(test|bioassay|assay)",
    r"\bcdc\s+bottle\s+(bioassay|assay)",
]

# Insecticide keywords
INSECTICIDE_KEYWORDS = [
    r"\binsecticid", r"\bpesticid",
    r"\bpyrethroid", r"\borgano\s*phosph", r"\bcarbamate\b",
    r"\borgano\s*chlorine", r"\bneonicotinoid",
    r"\bdeltamethrin\b", r"\bpermethrin\b", r"\bcypermethrin\b",
    r"\blambda[- ]cyhalothrin\b", r"\balpha[- ]cypermethrin\b",
    r"\betofenprox\b", r"\bbifenthrin\b", r"\bcyfluthrin\b",
    r"\bmalathion\b", r"\btemephos\b", r"\bfenitrothion\b",
    r"\bpirimiphos[- ]methyl\b", r"\bchlorpyrifos\b",
    r"\bfenthion\b",
    r"\bpropoxur\b", r"\bbendiocarb\b", r"\bcarbaryl\b",
    r"\bddt\b", r"\bdieldrin\b", r"\blindane\b", r"\bhch\b",
    r"\bimidacloprid\b", r"\bclothianidin\b", r"\bthiamethoxam\b",
    r"\bacetamiprid\b",
    r"\bpyriproxyfen\b", r"\bmethoprene\b", r"\bdiflubenzuron\b",
    r"\bnovaluron\b",
    r"\bbti\b", r"\bbacillus\s+thuringiensis\b", r"\bspinosa[dt]\b",
]

# Exclusion: not original research
REVIEW_TYPES = [
    "review", "systematic review", "meta-analysis",
    "editorial", "comment", "letter", "news", "biography",
    "guideline", "practice guideline", "consensus development conference",
    "published erratum",
]

# Exclusion: non-mosquito focus
NON_MOSQUITO_KEYWORDS = [
    r"\bagricultur", r"\bcrop\s+pest",
    r"\btick\b", r"\bticks\b", r"\bixod",
    r"\bcockroach", r"\bblattella\b",
    r"\bfly\b", r"\bflies\b", r"\bmusca\b",
    r"\bbed\s*bug", r"\bcimex\b",
    r"\bhead\s*lice\b", r"\bpedicul",
    r"\btriatomin", r"\btriatoma\b",
    r"\bsandfl", r"\bphlebotom",
    r"\btsetse\b", r"\bglossina\b",
    r"\bsimulium\b", r"\bblackfl",
    r"\bplant\s+protection\b",
    r"\bhelicoverpa\b", r"\bspodoptera\b", r"\bplutella\b",
    r"\bbemisia\b", r"\bwhitefl", r"\baphid",
]

# Exclusion: GM/SIT focus
GM_SIT_KEYWORDS = [
    r"\bgenetically\s+modified\b", r"\btransgenic\s+mosquit",
    r"\bsterile\s+insect\s+technique\b", r"\bsit\b.*\brelease\b",
    r"\bgene\s+drive\b", r"\bincompatible\s+insect\b",
    r"\bwolbachia\b.*\b(release|suppression|replacement)\b",
]

# Exclusion: repellent-only
REPELLENT_KEYWORDS = [
    r"\brepellen(t|cy)\b(?!.*\b(insecticid|mortality|bioassay|resist))",
    r"\bdeet\b(?!.*\b(insecticid|mortality|bioassay|resist))",
    r"\bspatial\s+repellen",
]


def compile_patterns(keyword_list):
    """Compile a list of regex patterns."""
    return [re.compile(p, re.IGNORECASE) for p in keyword_list]


def any_match(text, patterns):
    """Check if any compiled pattern matches the text."""
    if not text or pd.isna(text):
        return False
    for p in patterns:
        if p.search(str(text)):
            return True
    return False


def count_matches(text, patterns):
    """Count how many patterns match the text."""
    if not text or pd.isna(text):
        return 0
    count = 0
    text_str = str(text)
    for p in patterns:
        if p.search(text_str):
            count += 1
    return count


def is_review_type(pub_type):
    """Check if the publication type indicates a review/non-original."""
    if not pub_type or pd.isna(pub_type):
        return False
    pt_lower = str(pub_type).lower()
    for rt in REVIEW_TYPES:
        if rt in pt_lower:
            # Exception: meta-analysis can contain original data
            if rt == "meta-analysis":
                continue
            return True
    return False


def screen_article(row, pat):
    """
    Screen a single article and return (decision, reason, score).

    decision: 'Include', 'Exclude', 'Uncertain'
    reason: explanation for decision
    score: relevance score (higher = more relevant)
    """
    title = str(row.get("title", ""))
    abstract = str(row.get("abstract", ""))
    mesh = str(row.get("mesh_terms", ""))
    keywords = str(row.get("keywords", ""))
    pub_type = str(row.get("pub_type", ""))

    # Combine searchable text
    full_text = f"{title} {abstract} {mesh} {keywords}"
    title_abstract = f"{title} {abstract}"

    # ── Exclusion checks ──

    # E1: Not original research (review, editorial, etc.)
    if is_review_type(pub_type):
        # Exception: keep if it's a review but contains original resistance data keywords
        res_count = count_matches(full_text, pat["resistance"])
        insect_count = count_matches(full_text, pat["insecticide"])
        if res_count < 3 or insect_count < 2:
            return "Exclude", "E1: Not original research (review/editorial)", 0

    # E3: Non-mosquito species
    if any_match(full_text, pat["non_mosquito"]):
        if not any_match(full_text, pat["mosquito"]):
            return "Exclude", "E3: Non-mosquito species", 0

    # E4: GM/SIT mosquitoes
    if any_match(full_text, pat["gm_sit"]):
        if not any_match(full_text, pat["resistance"]):
            return "Exclude", "E4: GM/SIT mosquitoes without resistance data", 0

    # ── Inclusion scoring ──

    score = 0

    # Check for mosquito species
    has_mosquito = any_match(full_text, pat["mosquito"])
    has_target = any_match(full_text, pat["target"])

    if has_target:
        score += 20  # strong signal for Ae. albopictus
    elif has_mosquito:
        score += 10  # other mosquito species

    # Check for resistance keywords
    res_count = count_matches(full_text, pat["resistance"])
    score += res_count * 3

    # Check for insecticide keywords
    insect_count = count_matches(full_text, pat["insecticide"])
    score += insect_count * 2

    # Bonus for quantitative indicators in abstract
    quant_patterns = [
        r"\bmortality\b.*\d+\s*%", r"\d+\s*%\s*mortality",
        r"\blc\s*50\b", r"\brr\s*=?\s*\d",
        r"\ballele\s+frequen\w*\s*=?\s*[\d.]",
        r"\bf1534c\b", r"\bv1016g\b", r"\bs989p\b",
        r"\bn\s*=\s*\d", r"\bp\s*[<=]\s*0\.\d",
    ]
    for qp in quant_patterns:
        if re.search(qp, title_abstract, re.IGNORECASE):
            score += 5

    # ── Decision logic ──

    if not has_mosquito and not has_target:
        if res_count < 2 and insect_count < 2:
            return "Exclude", "No mosquito species or insufficient resistance keywords", 0

    # Require at least some resistance-related content
    if res_count == 0 and insect_count == 0:
        return "Exclude", "No resistance or insecticide keywords found", score

    # High-confidence include
    if has_target and res_count >= 2 and insect_count >= 1:
        return "Include", "Ae. albopictus + resistance + insecticide data", score

    if has_mosquito and res_count >= 3 and insect_count >= 2:
        return "Include", "Mosquito + strong resistance + insecticide signals", score

    # Medium confidence
    if has_mosquito and (res_count >= 2 or insect_count >= 2):
        if score >= 25:
            return "Include", "Mosquito + adequate resistance signals", score
        return "Uncertain", "Mosquito species but moderate resistance signals", score

    if has_mosquito and (res_count >= 1 or insect_count >= 1):
        return "Uncertain", "Mosquito species with weak resistance signals", score

    # Low relevance
    if score >= 15:
        return "Uncertain", "Some relevant keywords but unclear focus", score

    return "Exclude", "Insufficient relevance signals", score


def main():
    print("=" * 60)
    print("Title/Abstract Screening")
    print("=" * 60)

    # Load data
    df = pd.read_csv(INPUT_FILE, dtype=str)
    print(f"  Loaded {len(df)} articles from {INPUT_FILE.name}")

    # Compile patterns
    pat = {
        "mosquito": compile_patterns(SPECIES_MOSQUITO),
        "target": compile_patterns(TARGET_SPECIES),
        "resistance": compile_patterns(RESISTANCE_KEYWORDS),
        "insecticide": compile_patterns(INSECTICIDE_KEYWORDS),
        "non_mosquito": compile_patterns(NON_MOSQUITO_KEYWORDS),
        "gm_sit": compile_patterns(GM_SIT_KEYWORDS),
        "repellent": compile_patterns(REPELLENT_KEYWORDS),
    }

    # Screen each article
    results = []
    for idx, row in df.iterrows():
        decision, reason, score = screen_article(row, pat)
        results.append({
            "pmid": row.get("pmid", ""),
            "title": row.get("title", ""),
            "first_author": row.get("first_author", ""),
            "year": row.get("year", ""),
            "journal": row.get("journal", ""),
            "doi": row.get("doi", ""),
            "decision": decision,
            "reason": reason,
            "relevance_score": score,
            "has_albopictus": 1 if any_match(
                f"{row.get('title', '')} {row.get('abstract', '')} {row.get('mesh_terms', '')}",
                pat["target"]
            ) else 0,
        })

    result_df = pd.DataFrame(results)

    # Sort: Include first, then Uncertain, then Exclude; within each, by score descending
    order = {"Include": 0, "Uncertain": 1, "Exclude": 2}
    result_df["_order"] = result_df["decision"].map(order)
    result_df = result_df.sort_values(
        ["_order", "relevance_score"], ascending=[True, False]
    ).drop(columns=["_order"])

    # Save screening log
    result_df.to_csv(OUTPUT_LOG, index=False)
    print(f"\n  Saved screening log: {OUTPUT_LOG}")

    # ── Summary statistics ──
    n_include = (result_df["decision"] == "Include").sum()
    n_uncertain = (result_df["decision"] == "Uncertain").sum()
    n_exclude = (result_df["decision"] == "Exclude").sum()
    n_albopictus = result_df["has_albopictus"].astype(int).sum()

    # Breakdown of exclusion reasons
    excl_reasons = result_df[result_df["decision"] == "Exclude"]["reason"].value_counts()

    # Breakdown of included by has_albopictus
    incl_albo = result_df[(result_df["decision"] == "Include") & (result_df["has_albopictus"] == 1)].shape[0]
    incl_other = result_df[(result_df["decision"] == "Include") & (result_df["has_albopictus"] == 0)].shape[0]

    summary = []
    summary.append("=" * 60)
    summary.append("SCREENING SUMMARY")
    summary.append("=" * 60)
    summary.append(f"\nTotal articles screened: {len(result_df)}")
    summary.append(f"\n  Include:   {n_include:>5}  ({n_include/len(result_df)*100:.1f}%)")
    summary.append(f"  Uncertain: {n_uncertain:>5}  ({n_uncertain/len(result_df)*100:.1f}%)")
    summary.append(f"  Exclude:   {n_exclude:>5}  ({n_exclude/len(result_df)*100:.1f}%)")
    summary.append(f"\nArticles mentioning Ae. albopictus: {n_albopictus}")
    summary.append(f"\nIncluded articles:")
    summary.append(f"  With Ae. albopictus:  {incl_albo}")
    summary.append(f"  Other mosquitoes:     {incl_other}")
    summary.append(f"\nTo advance to full-text screening: {n_include + n_uncertain} articles")
    summary.append(f"  (Include + Uncertain)")
    summary.append(f"\nExclusion reasons:")
    for reason, count in excl_reasons.items():
        summary.append(f"  {reason}: {count}")

    # Year distribution of included articles
    incl_df = result_df[result_df["decision"] == "Include"]
    if "year" in incl_df.columns:
        year_counts = incl_df["year"].value_counts().sort_index()
        summary.append(f"\nIncluded articles by year:")
        for yr, cnt in year_counts.items():
            summary.append(f"  {yr}: {cnt}")

    summary_text = "\n".join(summary)
    print(summary_text)

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(summary_text)
    print(f"\n  Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
