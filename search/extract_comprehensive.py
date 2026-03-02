#!/usr/bin/env python3
"""
extract_comprehensive.py – Comprehensive data extraction from full texts + abstracts.

For articles with PMC full text: parse the full text for quantitative data.
For articles without full text: use improved abstract parsing.

Produces extracted_data.csv with one row per test result.
"""

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SEARCH_DIR.parent
PMC_DIR = SEARCH_DIR / "pmc_texts"
INPUT_ARTICLES = SEARCH_DIR / "fulltext_retrieval_list.csv"
INPUT_DATA = SEARCH_DIR / "search_records" / "pubmed_combined_deduplicated.csv"
INPUT_PMC_MAP = SEARCH_DIR / "pmc_mapping.csv"
OUTPUT_FILE = PROJECT_DIR / "03_data" / "raw" / "extracted_data.csv"
OUTPUT_SUMMARY = SEARCH_DIR / "comprehensive_extraction_summary.txt"

sys.path.insert(0, str(PROJECT_DIR / "04_analysis"))
from config import INSECTICIDE_CLASS_MAP, WHO_REGION_MAP

# ── Insecticide patterns ──
INSECTICIDE_NAMES = list(INSECTICIDE_CLASS_MAP.keys())
INSECTICIDE_PATTERNS = {}
for name in INSECTICIDE_NAMES:
    escaped = re.escape(name.lower())
    INSECTICIDE_PATTERNS[name] = re.compile(
        r'\b' + escaped + r'\b', re.IGNORECASE
    )

# Additional name variants
NAME_VARIANTS = {
    "lambda-cyhalothrin": [r"lambda[\s-]?cyhalothrin", r"λ[\s-]?cyhalothrin"],
    "alpha-cypermethrin": [r"alpha[\s-]?cypermethrin", r"α[\s-]?cypermethrin"],
    "beta-cyfluthrin": [r"beta[\s-]?cyfluthrin", r"β[\s-]?cyfluthrin"],
    "pirimiphos-methyl": [r"pirimiphos[\s-]?methyl"],
    "DDT": [r"\bDDT\b"],
    "Bti": [r"\bBti\b", r"B\.?\s*t\.?\s*i\.?",
            r"Bacillus\s+thuringiensis\s+(?:var\.?\s+)?israelensis"],
    "HCH": [r"\bHCH\b", r"\bBHC\b", r"\blindane\b"],
}
for name, variants in NAME_VARIANTS.items():
    for v in variants:
        key = f"{name}_v"
        INSECTICIDE_PATTERNS[name] = re.compile(
            r'\b' + v + r'\b' if not v.startswith(r'\b') else v,
            re.IGNORECASE
        )

# ── Country patterns ──
COUNTRY_LIST = list(WHO_REGION_MAP.keys())
SUBNATIONAL = {
    "Hainan": "China", "Yunnan": "China", "Guangdong": "China",
    "Guangzhou": "China", "Zhejiang": "China", "Fujian": "China",
    "Jiangsu": "China", "Shandong": "China", "Sichuan": "China",
    "Hubei": "China", "Hunan": "China", "Beijing": "China",
    "Shanghai": "China", "Chongqing": "China", "Guizhou": "China",
    "Sabah": "Malaysia", "Sarawak": "Malaysia", "Penang": "Malaysia",
    "Selangor": "Malaysia", "Kuala Lumpur": "Malaysia", "Johor": "Malaysia",
    "Delhi": "India", "Mumbai": "India", "Kolkata": "India",
    "Chennai": "India", "Kerala": "India", "Karnataka": "India",
    "Tamil Nadu": "India", "Assam": "India", "Odisha": "India",
    "Java": "Indonesia", "Sumatra": "Indonesia", "Bali": "Indonesia",
    "Jakarta": "Indonesia",
    "Bangkok": "Thailand", "Chiang Mai": "Thailand",
    "Réunion": "Reunion", "La Réunion": "Reunion",
    "Viet Nam": "Vietnam",
    "Ho Chi Minh": "Vietnam", "Hanoi": "Vietnam",
    "Sardinia": "Italy", "Rome": "Italy",
    "Athens": "Greece", "Crete": "Greece", "Thessaloniki": "Greece",
}

# ── kdr mutations ──
KDR_MUTATIONS = {
    "F1534C": (1534, r"F1534C"), "F1534S": (1534, r"F1534S"),
    "F1534L": (1534, r"F1534L"),
    "V1016G": (1016, r"V1016G"), "V1016I": (1016, r"V1016I"),
    "S989P": (989, r"S989P"),
    "I1532T": (1532, r"I1532T"),
    "L1014F": (1014, r"L1014F"), "L1014S": (1014, r"L1014S"),
    "D1763Y": (1763, r"D1763Y"),
    "V410L": (410, r"V410L"),
}


def find_countries(text):
    found = set()
    for loc in COUNTRY_LIST + list(SUBNATIONAL.keys()):
        if re.search(r'\b' + re.escape(loc) + r'\b', text, re.IGNORECASE):
            country = SUBNATIONAL.get(loc, loc)
            if country in WHO_REGION_MAP:
                found.add(country)
    return sorted(found)


def find_insecticides(text):
    found = set()
    for name, pat in INSECTICIDE_PATTERNS.items():
        if pat.search(text):
            found.add(name)
    return sorted(found)


def find_kdr(text):
    found = set()
    for mut, (codon, pat_str) in KDR_MUTATIONS.items():
        if re.search(r'\b' + pat_str + r'\b', text, re.IGNORECASE):
            found.add(mut)
    return sorted(found)


def detect_life_stage(text):
    has_adult = bool(re.search(r'\badult\b', text, re.IGNORECASE))
    has_larva = bool(re.search(r'\blarv', text, re.IGNORECASE))
    if has_adult and has_larva:
        return "Adult"  # default to adult for combined
    elif has_adult:
        return "Adult"
    elif has_larva:
        return "Larva"
    return ""


def detect_method(text):
    if re.search(r'WHO\s+tube', text, re.IGNORECASE):
        return "WHO tube"
    if re.search(r'WHO\s+(?:susceptibility|diagnostic)\s+(?:test|assay)', text, re.IGNORECASE):
        return "WHO tube"
    if re.search(r'CDC\s+bottle', text, re.IGNORECASE):
        return "CDC bottle"
    if re.search(r'WHO\s+bottle', text, re.IGNORECASE):
        return "WHO bottle"
    if re.search(r'WHO\s+larv\w+\s+(?:susceptibility|bioassay)', text, re.IGNORECASE):
        return "WHO larval"
    if re.search(r'larv\w+\s+bioassay', text, re.IGNORECASE):
        return "Larval bioassay"
    return ""


def extract_mortality_from_text(text, insecticides):
    """
    Extract mortality-insecticide pairs from text using multiple strategies.
    """
    results = []

    # Strategy 1: Table-like patterns "insecticide ... XX.X% ... mortality"
    # or "insecticide ... mortality ... XX.X%"
    for name in insecticides:
        escaped = re.escape(name)
        # Pattern: insecticide name within 300 chars of a percentage
        for m in re.finditer(escaped, text, re.IGNORECASE):
            # Look forward up to 300 chars for a percentage
            after = text[m.end():m.end()+300]
            # Find percentages
            pct_matches = list(re.finditer(r'(\d+\.?\d*)\s*%', after))
            for pm in pct_matches:
                val = float(pm.group(1))
                # Check context around the percentage for mortality keywords
                context_start = max(0, pm.start() - 50)
                context = after[context_start:pm.end()+50]
                if (re.search(r'mortalit|knockdown|kill|dead|susceptib|surviv',
                             context, re.IGNORECASE) or
                    re.search(r'mortalit|knockdown|kill|dead',
                             after[:pm.start()], re.IGNORECASE)):
                    if 0 <= val <= 100:
                        results.append((name, val, None))
                        break  # take first matching percentage

    # Strategy 2: "XX% mortality for/to/against insecticide"
    for m in re.finditer(
        r'(\d+\.?\d*)\s*%\s*(?:mortality|knockdown)\s*'
        r'(?:rate\s+)?(?:for|to|with|against|of)\s+'
        r'(\w+(?:[-]\w+)?)',
        text, re.IGNORECASE
    ):
        val = float(m.group(1))
        insect_word = m.group(2).lower()
        if 0 <= val <= 100:
            for name in insecticides:
                if name.lower().startswith(insect_word[:5]):
                    results.append((name, val, None))

    # Strategy 3: "mortality (rate) of XX% for insecticide"
    for m in re.finditer(
        r'mortality\s*(?:rate\s*)?(?:of|was|=|:)\s*(\d+\.?\d*)\s*%\s*'
        r'(?:for|to|with|against)?\s*(\w+(?:[-]\w+)?)',
        text, re.IGNORECASE
    ):
        val = float(m.group(1))
        insect_word = m.group(2).lower()
        if 0 <= val <= 100:
            for name in insecticides:
                if name.lower().startswith(insect_word[:5]):
                    results.append((name, val, None))

    # Strategy 4: "insecticide (0.05%): XX.X ± Y.Y % mortality"
    for name in insecticides:
        escaped = re.escape(name)
        for m in re.finditer(
            escaped + r'\s*(?:\([^)]*\))?\s*[:,]\s*'
            r'(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?\s*%?\s*'
            r'(?:mortality|knockdown)',
            text, re.IGNORECASE
        ):
            val = float(m.group(1))
            if 0 <= val <= 100:
                results.append((name, val, None))

    # Strategy 5: WHO resistance status mentions
    for name in insecticides:
        escaped = re.escape(name)
        # "confirmed resistance to deltamethrin"
        if re.search(
            r'(?:confirmed|high)\s+resistance\s+(?:to|for|against)\s+' + escaped,
            text, re.IGNORECASE
        ):
            # Don't add a specific mortality, but flag
            pass
        # "susceptible to deltamethrin"
        if re.search(
            r'susceptib\w+\s+(?:to|for|against)\s+' + escaped,
            text, re.IGNORECASE
        ):
            pass

    # Extract sample sizes nearby
    n_tested = None
    for m in re.finditer(r'\bn\s*=\s*(\d+)', text, re.IGNORECASE):
        n = int(m.group(1))
        if 20 <= n <= 5000:
            n_tested = n
            break

    # Deduplicate: keep unique (insecticide, mortality) pairs
    seen = set()
    unique = []
    for name, val, n in results:
        key = (name, round(val, 1))
        if key not in seen:
            seen.add(key)
            unique.append((name, val, n if n else n_tested))

    return unique


def extract_rr_from_text(text, insecticides):
    """Extract resistance ratio data from text."""
    results = []

    # Pattern: "RR50 = X.XX" or "RR = X.XX"
    for m in re.finditer(
        r'(?:RR|resistance\s+ratio)\s*_?\s*(50|95)?\s*'
        r'(?:=|:|was|of)\s*'
        r'(\d+\.?\d*)\s*'
        r'(?:\(\s*(?:95\s*%?\s*(?:CI|CL|FL)\s*[:=]?\s*)?'
        r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*\))?',
        text, re.IGNORECASE
    ):
        rr_type = f"RR{m.group(1)}" if m.group(1) else "RR50"
        rr_val = float(m.group(2))
        ci_lo = float(m.group(3)) if m.group(3) else None
        ci_hi = float(m.group(4)) if m.group(4) else None

        if rr_val <= 0 or rr_val > 100000:
            continue

        # Find associated insecticide (look backward up to 300 chars)
        start = max(0, m.start() - 300)
        context = text[start:m.end()]
        matched = None
        for name in insecticides:
            if re.search(re.escape(name), context, re.IGNORECASE):
                matched = name
                break

        if matched:
            results.append((matched, rr_type, rr_val, ci_lo, ci_hi))

    # Pattern: "X.XX-fold resistance" near an insecticide
    for name in insecticides:
        escaped = re.escape(name)
        for m in re.finditer(
            escaped + r'[^.]{0,200}?(\d+\.?\d*)\s*[-–]?\s*fold\s*'
            r'(?:resistance|more\s+resistant|higher)',
            text, re.IGNORECASE
        ):
            rr_val = float(m.group(1))
            if 1 < rr_val < 100000:
                results.append((name, "RR50", rr_val, None, None))

    # Pattern: LC50 values that can be converted to RR
    lc_field_vals = {}
    lc_ref_vals = {}
    for name in insecticides:
        escaped = re.escape(name)
        # LC50 of field strain
        for m in re.finditer(
            r'(?:field|resistant|test)\s+(?:strain|population|colony)?\s*'
            r'(?:.*?)LC\s*_?\s*50\s*(?:=|:|was|of)\s*(\d+\.?\d*)',
            text, re.IGNORECASE
        ):
            lc_field_vals[name] = float(m.group(1))

        for m in re.finditer(
            r'(?:susceptible|reference|Bora|USDA|Rockefeller|New\s+Orleans)\s*'
            r'(?:.*?)LC\s*_?\s*50\s*(?:=|:|was|of)\s*(\d+\.?\d*)',
            text, re.IGNORECASE
        ):
            if name in lc_field_vals:
                lc_ref_vals[name] = float(m.group(1))

    # Calculate RR from LC50 pairs
    for name in lc_field_vals:
        if name in lc_ref_vals and lc_ref_vals[name] > 0:
            rr_val = lc_field_vals[name] / lc_ref_vals[name]
            if 0 < rr_val < 100000:
                results.append((name, "RR50", round(rr_val, 2), None, None))

    # Deduplicate
    seen = set()
    unique = []
    for item in results:
        key = (item[0], item[1], round(item[2], 2))
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def extract_kdr_from_text(text, mutations):
    """Extract kdr allele frequency data from text."""
    results = []

    for mut in mutations:
        codon = KDR_MUTATIONS[mut][0]

        # Collect all frequency mentions
        freq_values = []

        # Pattern 1: "F1534C ... frequency ... 0.XX"
        for m in re.finditer(
            re.escape(mut) + r'[^.]{0,150}?'
            r'(?:allele\s+)?(?:frequen\w*|prevalence|proportion)\s*'
            r'(?:of|=|:|was|were|ranged|between)?\s*'
            r'(\d+\.?\d*)\s*(%?)',
            text, re.IGNORECASE
        ):
            val = float(m.group(1))
            is_pct = m.group(2) == '%'
            freq = val / 100 if is_pct or val > 1 else val
            if 0 <= freq <= 1:
                # Look for n_genotyped
                ctx = text[max(0, m.start()-100):m.end()+100]
                n_match = re.search(r'n\s*=\s*(\d+)', ctx, re.IGNORECASE)
                n = int(n_match.group(1)) if n_match else None
                freq_values.append((freq, n))

        # Pattern 2: "frequency ... 0.XX ... F1534C"
        for m in re.finditer(
            r'(?:allele\s+)?(?:frequen\w*|prevalence)\s*'
            r'(?:of|=|:)?\s*(\d+\.?\d*)\s*(%?)\s*'
            r'(?:for\s+|of\s+)?' + re.escape(mut),
            text, re.IGNORECASE
        ):
            val = float(m.group(1))
            is_pct = m.group(2) == '%'
            freq = val / 100 if is_pct or val > 1 else val
            if 0 <= freq <= 1:
                freq_values.append((freq, None))

        # Pattern 3: "F1534C (X.XX)" or "F1534C (XX%)"
        for m in re.finditer(
            re.escape(mut) + r'\s*\(\s*(\d+\.?\d*)\s*(%?)\s*\)',
            text, re.IGNORECASE
        ):
            val = float(m.group(1))
            is_pct = m.group(2) == '%'
            freq = val / 100 if is_pct or val > 1 else val
            if 0 <= freq <= 1:
                freq_values.append((freq, None))

        # Pattern 4: "XX% F1534C"
        for m in re.finditer(
            r'(\d+\.?\d*)\s*%\s*(?:of\s+)?' + re.escape(mut),
            text, re.IGNORECASE
        ):
            val = float(m.group(1)) / 100
            if 0 <= val <= 1:
                freq_values.append((val, None))

        # Pattern 5: genotype counts
        for m in re.finditer(
            re.escape(mut) + r'[^.]{0,300}?'
            r'(?:RR|homozygous\s+mutant)\s*[:=]?\s*(\d+)\s*[,;]\s*'
            r'(?:RS|heterozygous)\s*[:=]?\s*(\d+)\s*[,;]\s*'
            r'(?:SS|homozygous\s+wild)\s*[:=]?\s*(\d+)',
            text, re.IGNORECASE
        ):
            rr = int(m.group(1))
            rs = int(m.group(2))
            ss = int(m.group(3))
            total = rr + rs + ss
            if total > 0:
                freq = (2 * rr + rs) / (2 * total)
                freq_values.append((freq, total))

        # Take unique frequencies
        seen = set()
        for freq, n in freq_values:
            key = round(freq, 3)
            if key not in seen:
                seen.add(key)
                results.append((mut, codon, freq, n))

    return results


def extract_enzyme_from_text(text):
    """Extract enzyme activity data from text."""
    results = []

    enzyme_keywords = {
        "MFO": [r'\bP450\b', r'\bCYP\w+', r'\bmonooxygenase\b', r'\bMFO\b',
                r'\bmixed[- ]function\s+oxidase\b', r'\boxidase\b'],
        "NSE": [r'\besterase\b', r'\bNSE\b', r'\bcarboxylesterase\b',
                r'\bnon[- ]specific\s+esterase\b', r'\bCCE\b', r'\bEST\b'],
        "GST": [r'\bGST\b', r'\bglutathione\s+S[- ]transferase\b',
                r'\bglutathione\b'],
        "AChE": [r'\bAChE\b', r'\bacetylcholinesterase\b',
                 r'\binsensitive\s+AChE\b'],
    }

    for enzyme, patterns in enzyme_keywords.items():
        found = False
        for pat_str in patterns:
            for m in re.finditer(pat_str, text, re.IGNORECASE):
                if found:
                    break
                context = text[m.start():min(m.end()+300, len(text))]

                fold_change = None
                elevated_pct = None
                field_mean = None
                field_sd = None
                field_n = None
                ref_mean = None
                ref_sd = None
                ref_n = None

                # Fold change
                fc_m = re.search(
                    r'(\d+\.?\d*)\s*[-–]?\s*fold\s*'
                    r'(?:increase|change|higher|elevation|overexpress|more)',
                    context, re.IGNORECASE
                )
                if fc_m:
                    fold_change = float(fc_m.group(1))

                # Elevated percentage
                elev_m = re.search(
                    r'(\d+\.?\d*)\s*%\s*(?:elevated|overexpress|above|increased|showed)',
                    context, re.IGNORECASE
                )
                if elev_m:
                    elevated_pct = float(elev_m.group(1))

                # Mean ± SD (n)
                mean_m = re.search(
                    r'(\d+\.?\d*)\s*[±+]\s*(\d+\.?\d*)\s*'
                    r'(?:\(n\s*=\s*(\d+)\))?',
                    context
                )
                if mean_m:
                    field_mean = float(mean_m.group(1))
                    field_sd = float(mean_m.group(2))
                    if mean_m.group(3):
                        field_n = int(mean_m.group(3))

                if fold_change or elevated_pct or field_mean:
                    results.append({
                        "enzyme_system": enzyme,
                        "fold_change": fold_change,
                        "elevated_pct": elevated_pct,
                        "field_mean": field_mean,
                        "field_sd": field_sd,
                        "field_n": field_n,
                        "reference_mean": ref_mean,
                        "reference_sd": ref_sd,
                        "reference_n": ref_n,
                    })
                    found = True
                    break
                elif re.search(r'\belevat|\boverexpress|\bincreas|\bhigh', context, re.IGNORECASE):
                    results.append({
                        "enzyme_system": enzyme,
                        "fold_change": None,
                        "elevated_pct": None,
                        "field_mean": None,
                        "field_sd": None,
                        "field_n": None,
                        "reference_mean": None,
                        "reference_sd": None,
                        "reference_n": None,
                    })
                    found = True
                    break

    return results


def extract_article(pmid, title, abstract, authors, first_author, year,
                    journal, doi, full_text=None):
    """Extract all data from one article."""
    # Use full text if available, otherwise abstract
    text = full_text if full_text and len(full_text) > 500 else f"{title} {abstract}"
    search_text = f"{title} {abstract} {text}" if full_text else f"{title} {abstract}"

    # Study ID
    author_part = first_author.split(",")[0].split()[0] if first_author else "Unknown"
    study_id = f"{author_part}_{year}"

    # Common elements
    countries = find_countries(search_text)
    insecticides = find_insecticides(search_text)
    mutations = find_kdr(search_text)
    life_stage = detect_life_stage(search_text)
    method = detect_method(search_text)

    country = countries[0] if countries else ""
    source = "PMC full text" if full_text and len(full_text) > 500 else "Abstract"

    base = {
        "study_id": study_id,
        "authors": str(authors)[:200] if authors else "",
        "year": year,
        "journal": journal,
        "doi": doi,
        "country": country,
        "species": "Aedes albopictus",
        "life_stage": life_stage,
        "strain_type": "Field",
        "bioassay_method": method,
    }

    rows = []

    # 1. Mortality data
    mortality_data = extract_mortality_from_text(text, insecticides)
    for insect_name, mort_pct, n_tested in mortality_data:
        r = base.copy()
        r["insecticide_name"] = insect_name
        r["insecticide_class"] = INSECTICIDE_CLASS_MAP.get(insect_name, "Unknown")
        r["mortality_pct"] = round(mort_pct, 1)
        if n_tested:
            r["n_tested"] = int(n_tested)
        r["notes"] = f"PMID:{pmid}; Source: {source}"
        rows.append(r)

    # 2. RR data
    rr_data = extract_rr_from_text(text, insecticides)
    for insect_name, rr_type, rr_val, ci_lo, ci_hi in rr_data:
        r = base.copy()
        r["insecticide_name"] = insect_name
        r["insecticide_class"] = INSECTICIDE_CLASS_MAP.get(insect_name, "Unknown")
        r["rr_type"] = rr_type
        r["rr_value"] = round(rr_val, 2)
        if ci_lo is not None:
            r["rr_ci_lower"] = round(ci_lo, 2)
        if ci_hi is not None:
            r["rr_ci_upper"] = round(ci_hi, 2)
        r["notes"] = f"PMID:{pmid}; Source: {source}"
        rows.append(r)

    # 3. kdr data
    kdr_data = extract_kdr_from_text(text, mutations)
    for mut, codon, freq, n_geno in kdr_data:
        r = base.copy()
        r["gene"] = "VGSC"
        r["codon_position"] = codon
        r["mutation"] = mut
        r["allele_frequency"] = round(freq, 4)
        if n_geno:
            r["n_genotyped"] = int(n_geno)
        r["notes"] = f"PMID:{pmid}; Source: {source}"
        rows.append(r)

    # 4. Enzyme data
    enzyme_data = extract_enzyme_from_text(text)
    for enz in enzyme_data:
        r = base.copy()
        r["enzyme_system"] = enz["enzyme_system"]
        if enz["fold_change"]:
            r["fold_change"] = round(enz["fold_change"], 2)
        if enz["elevated_pct"]:
            r["elevated_pct"] = round(enz["elevated_pct"], 1)
        if enz["field_mean"]:
            r["field_mean"] = enz["field_mean"]
        if enz["field_sd"]:
            r["field_sd"] = enz["field_sd"]
        if enz["field_n"]:
            r["field_n"] = int(enz["field_n"])
        if enz["reference_mean"]:
            r["reference_mean"] = enz["reference_mean"]
        if enz["reference_sd"]:
            r["reference_sd"] = enz["reference_sd"]
        if enz["reference_n"]:
            r["reference_n"] = int(enz["reference_n"])
        r["notes"] = f"PMID:{pmid}; Source: {source}"
        rows.append(r)

    # If no data extracted, create marker rows for mentioned insecticides
    if not rows and insecticides:
        for insect_name in insecticides[:5]:  # limit to top 5
            r = base.copy()
            r["insecticide_name"] = insect_name
            r["insecticide_class"] = INSECTICIDE_CLASS_MAP.get(insect_name, "Unknown")
            r["notes"] = f"PMID:{pmid}; Source: {source}; No quantitative data extracted"
            rows.append(r)

    # Expand for multiple countries
    if len(countries) > 1 and rows:
        expanded = []
        for c in countries:
            for r in rows:
                rc = r.copy()
                rc["country"] = c
                expanded.append(rc)
        rows = expanded

    return rows


def main():
    print("=" * 60)
    print("Comprehensive Data Extraction (Full Text + Abstracts)")
    print("=" * 60)

    # Load data
    ft_df = pd.read_csv(INPUT_ARTICLES, dtype=str)
    data_df = pd.read_csv(INPUT_DATA, dtype=str)
    core = ft_df[ft_df["final_category"].isin([
        "core_albopictus", "albopictus_check"
    ])]

    # Load PMC mapping
    pmc_map = {}
    if INPUT_PMC_MAP.exists():
        pmc_df = pd.read_csv(INPUT_PMC_MAP, dtype=str)
        pmc_map = dict(zip(pmc_df["pmid"], pmc_df["pmc_id"]))

    merged = core.merge(data_df, on="pmid", how="left", suffixes=("", "_raw"))
    print(f"  Processing {len(merged)} articles")
    print(f"  PMC full texts available: {len(pmc_map)}")

    # Extract
    all_rows = []
    n_with_fulltext = 0
    n_abstract_only = 0
    n_with_data = 0
    n_no_data = 0

    for idx, row in merged.iterrows():
        pmid = str(row.get("pmid", ""))
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        authors = str(row.get("authors", ""))
        first_author = str(row.get("first_author", ""))
        year = str(row.get("year", ""))
        journal = str(row.get("journal", ""))
        doi = str(row.get("doi", ""))

        # Check for full text
        full_text = None
        if pmid in pmc_map:
            pmc_id = pmc_map[pmid]
            ft_file = PMC_DIR / f"{pmc_id}.txt"
            if ft_file.exists():
                full_text = ft_file.read_text()
                n_with_fulltext += 1
            else:
                n_abstract_only += 1
        else:
            n_abstract_only += 1

        extracted = extract_article(
            pmid, title, abstract, authors, first_author,
            year, journal, doi, full_text
        )

        has_data = any(
            r.get("mortality_pct") or r.get("rr_value") or
            r.get("allele_frequency") or r.get("enzyme_system")
            for r in extracted
        )
        if has_data:
            n_with_data += 1
        else:
            n_no_data += 1

        all_rows.extend(extracted)

    # Build DataFrame
    template_cols = [
        "study_id", "authors", "year", "journal", "doi", "country", "region",
        "continent", "latitude", "longitude", "collection_year_start",
        "collection_year_end", "species", "life_stage", "strain_type",
        "quality_score", "insecticide_name", "insecticide_class",
        "bioassay_method", "concentration", "concentration_unit",
        "exposure_time_min", "recovery_time_h", "n_tested", "n_dead",
        "mortality_pct", "mortality_ci_lower", "mortality_ci_upper",
        "control_mortality_pct", "who_classification", "rr_type", "rr_value",
        "rr_ci_lower", "rr_ci_upper", "lc_field", "lc_reference", "lc_unit",
        "reference_strain", "gene", "codon_position", "mutation",
        "n_genotyped", "n_mutant_alleles", "allele_frequency",
        "freq_ci_lower", "freq_ci_upper", "genotype_RR", "genotype_RS",
        "genotype_SS", "detection_method", "enzyme_system",
        "enzyme_full_name", "assay_method", "field_mean", "field_sd",
        "field_n", "reference_mean", "reference_sd", "reference_n",
        "fold_change", "elevated_pct", "p_value", "notes",
    ]

    df = pd.DataFrame(all_rows)
    for col in template_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[template_cols]

    # Keep only rows with some data
    has_data = (
        (df["mortality_pct"].notna() & (df["mortality_pct"] != "")) |
        (df["rr_value"].notna() & (df["rr_value"] != "")) |
        (df["allele_frequency"].notna() & (df["allele_frequency"] != "")) |
        (df["enzyme_system"].notna() & (df["enzyme_system"] != "")) |
        (df["insecticide_name"].notna() & (df["insecticide_name"] != ""))
    )
    df = df[has_data].copy()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  Total rows: {len(df)}")

    # ── Summary ──
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("COMPREHENSIVE EXTRACTION SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"\nArticles processed: {len(merged)}")
    summary_lines.append(f"  With PMC full text: {n_with_fulltext}")
    summary_lines.append(f"  Abstract only: {n_abstract_only}")
    summary_lines.append(f"  With extractable data: {n_with_data}")
    summary_lines.append(f"  No specific data: {n_no_data}")
    summary_lines.append(f"\nTotal data rows: {len(df)}")

    # Data type counts
    n_mort = ((df["mortality_pct"].notna()) & (df["mortality_pct"] != "")).sum()
    n_rr = ((df["rr_value"].notna()) & (df["rr_value"] != "")).sum()
    n_kdr = ((df["allele_frequency"].notna()) & (df["allele_frequency"] != "")).sum()
    n_enz = ((df["enzyme_system"].notna()) & (df["enzyme_system"] != "")).sum()
    summary_lines.append(f"\n  Mortality data rows:     {n_mort}")
    summary_lines.append(f"  Resistance ratio rows:   {n_rr}")
    summary_lines.append(f"  kdr mutation rows:       {n_kdr}")
    summary_lines.append(f"  Enzyme activity rows:    {n_enz}")

    # Countries
    cdf = df[df["country"] != ""]
    countries = cdf["country"].value_counts()
    summary_lines.append(f"\nCountries ({len(countries)}):")
    for c, cnt in countries.head(25).items():
        summary_lines.append(f"  {c}: {cnt}")

    # Insecticides
    idf = df[df["insecticide_name"] != ""]
    insecticides = idf["insecticide_name"].value_counts()
    summary_lines.append(f"\nInsecticides ({len(insecticides)}):")
    for i, cnt in insecticides.head(20).items():
        summary_lines.append(f"  {i}: {cnt}")

    # Classes
    classes = idf["insecticide_class"].value_counts()
    summary_lines.append(f"\nInsecticide classes:")
    for c, cnt in classes.items():
        summary_lines.append(f"  {c}: {cnt}")

    # kdr
    kdf = df[df["mutation"] != ""]
    if len(kdf) > 0:
        mutations = kdf["mutation"].value_counts()
        summary_lines.append(f"\nkdr mutations ({len(mutations)}):")
        for m, cnt in mutations.items():
            summary_lines.append(f"  {m}: {cnt}")

    # Studies
    n_studies = df["study_id"].nunique()
    summary_lines.append(f"\nUnique studies: {n_studies}")

    years = pd.to_numeric(df["year"], errors="coerce")
    if years.notna().any():
        summary_lines.append(f"Year range: {int(years.min())}-{int(years.max())}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(summary_text)
    print(f"\n  Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
