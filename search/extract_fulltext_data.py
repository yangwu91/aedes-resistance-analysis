#!/usr/bin/env python3
"""
extract_fulltext_data.py – Comprehensive data extraction from article abstracts.

Parses abstracts of Ae. albopictus insecticide resistance articles to extract
structured data for the meta-analysis. Each output row = one test result
(one insecticide × one population × one indicator).

Extraction targets:
1. Mortality rates: insecticide name + mortality % (+ n_tested if available)
2. Resistance ratios: RR50/RR95, LC50/LC95 values
3. kdr mutations: mutation name + allele frequency (+ n_genotyped)
4. Enzyme activity: enzyme system + fold change or elevated %

Output: 03_data/raw/extracted_data.csv in the template format.
"""

import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──
SEARCH_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SEARCH_DIR.parent
INPUT_ARTICLES = SEARCH_DIR / "fulltext_retrieval_list.csv"
INPUT_DATA = SEARCH_DIR / "search_records" / "pubmed_combined_deduplicated.csv"
OUTPUT_FILE = PROJECT_DIR / "03_data" / "raw" / "extracted_data.csv"
OUTPUT_SUMMARY = SEARCH_DIR / "fulltext_extraction_summary.txt"

# Add analysis dir to path for config
sys.path.insert(0, str(PROJECT_DIR / "04_analysis"))
from config import INSECTICIDE_CLASS_MAP, WHO_REGION_MAP

# ── Insecticide name patterns (lowercased for matching) ──
INSECTICIDE_PATTERNS = {}
for name in INSECTICIDE_CLASS_MAP:
    # Create pattern that matches the insecticide name
    escaped = re.escape(name.lower())
    INSECTICIDE_PATTERNS[name] = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)

# Additional patterns for names with variants
EXTRA_INSECTICIDE_PATTERNS = {
    "lambda-cyhalothrin": re.compile(r'\blambda[\s-]?cyhalothrin\b', re.IGNORECASE),
    "alpha-cypermethrin": re.compile(r'\balpha[\s-]?cypermethrin\b', re.IGNORECASE),
    "beta-cyfluthrin": re.compile(r'\bbeta[\s-]?cyfluthrin\b', re.IGNORECASE),
    "pirimiphos-methyl": re.compile(r'\bpirimiphos[\s-]?methyl\b', re.IGNORECASE),
    "DDT": re.compile(r'\bDDT\b'),
    "Bti": re.compile(r'\b(?:Bti|B\.?t\.?i\.?|Bacillus\s+thuringiensis\s+(?:var\.\s+)?israelensis)\b', re.IGNORECASE),
    "HCH": re.compile(r'\b(?:HCH|BHC|lindane)\b', re.IGNORECASE),
}

# ── Country detection ──
COUNTRY_PATTERNS = {}
for country in WHO_REGION_MAP:
    COUNTRY_PATTERNS[country] = re.compile(r'\b' + re.escape(country) + r'\b', re.IGNORECASE)

# Sub-national to country mapping
SUBNATIONAL_TO_COUNTRY = {
    "Hainan": "China", "Yunnan": "China", "Guangdong": "China",
    "Guangzhou": "China", "Zhejiang": "China", "Fujian": "China",
    "Jiangsu": "China", "Shandong": "China", "Sichuan": "China",
    "Hubei": "China", "Hunan": "China", "Beijing": "China",
    "Shanghai": "China", "Chongqing": "China", "Guizhou": "China",
    "Henan": "China", "Anhui": "China",
    "Sabah": "Malaysia", "Sarawak": "Malaysia", "Penang": "Malaysia",
    "Selangor": "Malaysia", "Kuala Lumpur": "Malaysia", "Johor": "Malaysia",
    "Delhi": "India", "Mumbai": "India", "Kolkata": "India",
    "Chennai": "India", "Kerala": "India", "Karnataka": "India",
    "Pune": "India", "Assam": "India", "Tamil Nadu": "India",
    "Java": "Indonesia", "Sumatra": "Indonesia", "Bali": "Indonesia",
    "Sulawesi": "Indonesia", "Kalimantan": "Indonesia", "Jakarta": "Indonesia",
    "Bangkok": "Thailand", "Chiang Mai": "Thailand",
    "Réunion": "Reunion", "La Réunion": "Reunion",
    "Sardinia": "Italy", "Rome": "Italy",
    "Athens": "Greece", "Crete": "Greece",
    "Ho Chi Minh": "Vietnam", "Hanoi": "Vietnam",
    "Viet Nam": "Vietnam",
}
for loc, country in SUBNATIONAL_TO_COUNTRY.items():
    if loc not in COUNTRY_PATTERNS:
        COUNTRY_PATTERNS[loc] = re.compile(r'\b' + re.escape(loc) + r'\b', re.IGNORECASE)

# ── kdr mutation patterns ──
KDR_PATTERNS = {
    "F1534C": re.compile(r'\bF1534C\b', re.IGNORECASE),
    "F1534S": re.compile(r'\bF1534S\b', re.IGNORECASE),
    "F1534L": re.compile(r'\bF1534L\b', re.IGNORECASE),
    "V1016G": re.compile(r'\bV1016G\b', re.IGNORECASE),
    "V1016I": re.compile(r'\bV1016I\b', re.IGNORECASE),
    "S989P": re.compile(r'\bS989P\b', re.IGNORECASE),
    "I1532T": re.compile(r'\bI1532T\b', re.IGNORECASE),
    "L1014F": re.compile(r'\bL1014F\b', re.IGNORECASE),
    "L1014S": re.compile(r'\bL1014S\b', re.IGNORECASE),
    "D1763Y": re.compile(r'\bD1763Y\b', re.IGNORECASE),
    "V410L": re.compile(r'\bV410L\b', re.IGNORECASE),
}

KDR_CODON_MAP = {
    "F1534C": 1534, "F1534S": 1534, "F1534L": 1534,
    "V1016G": 1016, "V1016I": 1016,
    "S989P": 989,
    "I1532T": 1532,
    "L1014F": 1014, "L1014S": 1014,
    "D1763Y": 1763,
    "V410L": 410,
}


def find_countries(text):
    """Find all countries mentioned in text."""
    found = set()
    for loc, pat in COUNTRY_PATTERNS.items():
        if pat.search(text):
            country = SUBNATIONAL_TO_COUNTRY.get(loc, loc)
            if country in WHO_REGION_MAP:
                found.add(country)
    return sorted(found)


def find_insecticides(text):
    """Find all insecticides mentioned in text."""
    found = set()
    for name, pat in INSECTICIDE_PATTERNS.items():
        if pat.search(text):
            found.add(name)
    for name, pat in EXTRA_INSECTICIDE_PATTERNS.items():
        if pat.search(text):
            found.add(name)
    return sorted(found)


def find_kdr_mutations(text):
    """Find kdr mutations mentioned in text."""
    found = set()
    for mut, pat in KDR_PATTERNS.items():
        if pat.search(text):
            found.add(mut)
    return sorted(found)


def detect_life_stage(text):
    """Detect whether adult or larval bioassays were used."""
    has_adult = bool(re.search(r'\badult\b', text, re.IGNORECASE))
    has_larva = bool(re.search(r'\blarv', text, re.IGNORECASE))
    if has_adult and has_larva:
        return ["Adult", "Larva"]
    elif has_adult:
        return ["Adult"]
    elif has_larva:
        return ["Larva"]
    return [""]


def detect_bioassay_method(text):
    """Detect bioassay method from text."""
    methods = []
    if re.search(r'\bWHO\s+tube\b', text, re.IGNORECASE):
        methods.append("WHO tube")
    if re.search(r'\bWHO\s+bottle\b', text, re.IGNORECASE):
        methods.append("WHO bottle")
    if re.search(r'\bCDC\s+bottle\b', text, re.IGNORECASE):
        methods.append("CDC bottle")
    if not methods and re.search(r'\blarv\w*\s+bioassay\b', text, re.IGNORECASE):
        methods.append("Larval bioassay")
    if not methods and re.search(r'\bbioassay\b', text, re.IGNORECASE):
        methods.append("Bioassay")
    return methods if methods else [""]


def extract_mortality_data(text, insecticides_found):
    """
    Extract mortality rates associated with specific insecticides.
    Returns list of (insecticide, mortality_pct, n_tested) tuples.
    """
    results = []

    # Pattern 1: "X% mortality to/for/with insecticide"
    for m in re.finditer(
        r'(\d+\.?\d*)\s*%?\s*(?:mortality|knockdown|kill)\s*'
        r'(?:rate\s*)?(?:to|for|with|against|of)?\s*'
        r'(\w+(?:[-]\w+)?)',
        text, re.IGNORECASE
    ):
        mort = float(m.group(1))
        insect = m.group(2).lower()
        if 0 <= mort <= 100:
            for name in insecticides_found:
                if name.lower().startswith(insect[:4]):
                    results.append((name, mort, None))

    # Pattern 2: "insecticide (X% mortality)" or "insecticide: X%"
    for name in insecticides_found:
        escaped = re.escape(name)
        for m in re.finditer(
            escaped + r'\s*(?:\([^)]*\))?\s*(?:[:,]|\s+(?:showed?|exhibited?|resulted?\s+in|had|with))?\s*'
            r'(?:a\s+)?(?:mortality\s+(?:rate\s+)?(?:of\s+)?)?'
            r'(\d+\.?\d*)\s*%\s*(?:mortality|knockdown|kill)?',
            text, re.IGNORECASE
        ):
            mort = float(m.group(1))
            if 0 <= mort <= 100:
                results.append((name, mort, None))

    # Pattern 3: "mortality rates of X%, Y%, and Z% for insecticide A, B, and C"
    for m in re.finditer(
        r'mortality\s+(?:rates?\s+)?(?:of\s+|were?\s+|ranged?\s+from\s+)?'
        r'([\d.]+\s*%?\s*(?:[,;]\s*[\d.]+\s*%?\s*(?:and\s+)?)*[\d.]+\s*%?)\s*'
        r'(?:for|to|with|against)\s+'
        r'([\w-]+(?:\s*[,;]\s*(?:and\s+)?[\w-]+)*)',
        text, re.IGNORECASE
    ):
        mort_str = m.group(1)
        insect_str = m.group(2)
        morts = [float(x) for x in re.findall(r'(\d+\.?\d*)', mort_str)]
        insect_names = re.split(r'[,;]\s*(?:and\s+)?', insect_str)
        insect_names = [x.strip() for x in insect_names if x.strip()]

        if len(morts) == len(insect_names):
            for mort, iname in zip(morts, insect_names):
                if 0 <= mort <= 100:
                    for name in insecticides_found:
                        if name.lower().startswith(iname.lower()[:4]):
                            results.append((name, mort, None))

    # Pattern 4: "resistance to insecticide (X% mortality, n=Y)"
    for name in insecticides_found:
        escaped = re.escape(name)
        for m in re.finditer(
            r'(?:resistance|susceptib\w+)\s+(?:to|against)\s+' + escaped +
            r'.*?(\d+\.?\d*)\s*%.*?(?:n\s*=\s*(\d+))?',
            text, re.IGNORECASE
        ):
            mort = float(m.group(1))
            n = int(m.group(2)) if m.group(2) else None
            if 0 <= mort <= 100:
                results.append((name, mort, n))

    # Pattern 5: sample size extraction  "n = XX" or "XX mosquitoes"
    n_tested = None
    n_match = re.search(r'\bn\s*=\s*(\d+)', text, re.IGNORECASE)
    if n_match:
        n_tested = int(n_match.group(1))
    else:
        n_match = re.search(r'(\d+)\s*(?:female|mosquit|specimen|individual)', text, re.IGNORECASE)
        if n_match:
            n_val = int(n_match.group(1))
            if 20 <= n_val <= 10000:
                n_tested = n_val

    # Fill in n_tested where missing
    results = [(name, mort, n if n else n_tested) for name, mort, n in results]

    # Deduplicate
    seen = set()
    unique_results = []
    for name, mort, n in results:
        key = (name, round(mort, 1))
        if key not in seen:
            seen.add(key)
            unique_results.append((name, mort, n))

    return unique_results


def extract_rr_data(text, insecticides_found):
    """
    Extract resistance ratio data (RR50, RR95, LC50, LC95).
    Returns list of (insecticide, rr_type, rr_value, ci_lower, ci_upper) tuples.
    """
    results = []

    # Pattern: "RR50 = X.XX (CI: X.XX-X.XX)"
    for m in re.finditer(
        r'RR\s*_?\s*(50|95)\s*(?:=|:|\s+(?:of|was))?\s*'
        r'(\d+\.?\d*)\s*'
        r'(?:\((?:95\s*%?\s*CI:?\s*)?(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\))?',
        text, re.IGNORECASE
    ):
        rr_type = f"RR{m.group(1)}"
        rr_val = float(m.group(2))
        ci_lo = float(m.group(3)) if m.group(3) else None
        ci_hi = float(m.group(4)) if m.group(4) else None

        # Try to associate with nearest insecticide mention
        start = max(0, m.start() - 200)
        context = text[start:m.end()]
        matched_insect = None
        for name in insecticides_found:
            if re.search(re.escape(name), context, re.IGNORECASE):
                matched_insect = name
                break
        if matched_insect and rr_val > 0:
            results.append((matched_insect, rr_type, rr_val, ci_lo, ci_hi))

    # Pattern: "resistance ratio of X.XX for insecticide"
    for m in re.finditer(
        r'resistance\s+ratio\s*(?:of|=|:)?\s*(\d+\.?\d*)\s*'
        r'(?:\((?:95\s*%?\s*CI:?\s*)?(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\))?\s*'
        r'(?:for|to|against)\s+(\w+(?:[-]\w+)?)',
        text, re.IGNORECASE
    ):
        rr_val = float(m.group(1))
        ci_lo = float(m.group(2)) if m.group(2) else None
        ci_hi = float(m.group(3)) if m.group(3) else None
        insect = m.group(4).lower()
        for name in insecticides_found:
            if name.lower().startswith(insect[:4]):
                if rr_val > 0:
                    results.append((name, "RR50", rr_val, ci_lo, ci_hi))

    # Pattern: "LC50 = X.XX mg/L"
    for m in re.finditer(
        r'LC\s*_?\s*(50|95)\s*(?:=|:|\s+(?:of|was))?\s*'
        r'(\d+\.?\d*)\s*'
        r'(?:\((?:95\s*%?\s*CI:?\s*)?(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\))?\s*'
        r'(\w+/?(?:\w+)?)?',
        text, re.IGNORECASE
    ):
        lc_type = m.group(1)
        lc_val = float(m.group(2))
        # Store LC values as notes, not as RR
        start = max(0, m.start() - 200)
        context = text[start:m.end()]
        for name in insecticides_found:
            if re.search(re.escape(name), context, re.IGNORECASE):
                results.append((name, f"LC{lc_type}", lc_val,
                               float(m.group(3)) if m.group(3) else None,
                               float(m.group(4)) if m.group(4) else None))
                break

    # Deduplicate
    seen = set()
    unique = []
    for item in results:
        key = (item[0], item[1], round(item[2], 2))
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def extract_kdr_frequency(text, mutations_found):
    """
    Extract kdr allele frequencies.
    Returns list of (mutation, frequency, n_genotyped) tuples.
    """
    results = []

    for mut in mutations_found:
        # Pattern: "F1534C (frequency = 0.XX, n = XX)"
        # or "F1534C allele frequency of 0.XX"
        # or "F1534C (XX%)"
        for m in re.finditer(
            re.escape(mut) + r'[^.]*?'
            r'(?:allele\s+)?(?:frequen\w*|prevalence)\s*'
            r'(?:of|=|:|\s+was)?\s*'
            r'(\d+\.?\d*)\s*(%?)',
            text, re.IGNORECASE
        ):
            val = float(m.group(1))
            is_pct = m.group(2) == '%'
            if is_pct:
                freq = val / 100
            elif val > 1:
                freq = val / 100
            else:
                freq = val

            if 0 <= freq <= 1:
                # Find n_genotyped nearby
                n = None
                context = text[max(0, m.start()-100):m.end()+100]
                n_match = re.search(r'n\s*=\s*(\d+)', context, re.IGNORECASE)
                if n_match:
                    n = int(n_match.group(1))
                results.append((mut, freq, n))

        # Pattern: frequency followed by mutation
        for m in re.finditer(
            r'(?:allele\s+)?(?:frequen\w*|prevalence)\s*'
            r'(?:of|=|:)?\s*'
            r'(\d+\.?\d*)\s*(%?)\s*'
            r'(?:for|of)?\s*' + re.escape(mut),
            text, re.IGNORECASE
        ):
            val = float(m.group(1))
            is_pct = m.group(2) == '%'
            if is_pct:
                freq = val / 100
            elif val > 1:
                freq = val / 100
            else:
                freq = val
            if 0 <= freq <= 1:
                results.append((mut, freq, None))

        # Pattern: just percentage near mutation name
        for m in re.finditer(
            re.escape(mut) + r'[^.]{0,50}?(\d+\.?\d*)\s*%',
            text, re.IGNORECASE
        ):
            val = float(m.group(1))
            freq = val / 100
            if 0 <= freq <= 1:
                results.append((mut, freq, None))

    # Deduplicate (keep highest n_genotyped if duplicate)
    best = {}
    for mut, freq, n in results:
        key = (mut, round(freq, 3))
        if key not in best or (n is not None and (best[key][2] is None or n > best[key][2])):
            best[key] = (mut, freq, n)

    return list(best.values())


def extract_enzyme_data(text):
    """
    Extract enzyme activity information.
    Returns list of (enzyme_system, fold_change, elevated_pct) tuples.
    """
    results = []

    enzyme_keywords = {
        "MFO": [r'\bP450\b', r'\bCYP\w+', r'\bmonooxygenase\b', r'\bMFO\b',
                r'\bmixed[- ]function\s+oxidase\b'],
        "NSE": [r'\besterase\b', r'\bNSE\b', r'\bcarboxylesterase\b',
                r'\bnon[- ]specific\s+esterase\b', r'\bCCE\b'],
        "GST": [r'\bGST\b', r'\bglutathione\s+S[- ]transferase\b'],
        "AChE": [r'\bAChE\b', r'\bacetylcholinesterase\b'],
    }

    for enzyme, patterns in enzyme_keywords.items():
        for pat_str in patterns:
            pat = re.compile(pat_str, re.IGNORECASE)
            for m in pat.finditer(text):
                context = text[m.start():min(m.end()+200, len(text))]

                # Look for fold change
                fc_match = re.search(
                    r'(\d+\.?\d*)\s*[-]?\s*fold\s*(?:increase|change|higher|elevation|overexpress)',
                    context, re.IGNORECASE
                )
                fold_change = float(fc_match.group(1)) if fc_match else None

                # Look for elevated percentage
                elev_match = re.search(
                    r'(\d+\.?\d*)\s*%\s*(?:elevated|overexpress|above|increased)',
                    context, re.IGNORECASE
                )
                elevated_pct = float(elev_match.group(1)) if elev_match else None

                if fold_change or elevated_pct:
                    results.append((enzyme, fold_change, elevated_pct))
                elif bool(re.search(r'\belevat|\boverexpress|\bincreas|\bhigh', context, re.IGNORECASE)):
                    results.append((enzyme, None, None))

    # Deduplicate
    seen = set()
    unique = []
    for item in results:
        key = item[0]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def extract_article(row):
    """
    Extract all available data from one article. Returns list of row dicts.
    """
    pmid = str(row.get("pmid", ""))
    title = str(row.get("title", ""))
    abstract = str(row.get("abstract", ""))
    first_author = str(row.get("first_author", ""))
    authors = str(row.get("authors", ""))
    year = str(row.get("year", ""))
    journal = str(row.get("journal", ""))
    doi = str(row.get("doi", ""))

    full_text = f"{title} {abstract}"

    # Study ID
    author_part = first_author.split(",")[0].split()[0] if first_author else "Unknown"
    study_id = f"{author_part}_{year}"

    # Common fields
    countries = find_countries(full_text)
    insecticides = find_insecticides(full_text)
    mutations = find_kdr_mutations(full_text)
    life_stages = detect_life_stage(full_text)
    methods = detect_bioassay_method(full_text)

    country = countries[0] if countries else ""
    life_stage = life_stages[0] if life_stages else ""
    method = methods[0] if methods else ""

    base_row = {
        "study_id": study_id,
        "authors": authors,
        "year": year,
        "journal": journal,
        "doi": doi,
        "country": country,
        "species": "Aedes albopictus",
        "life_stage": life_stage,
        "strain_type": "Field",
        "insecticide_class": "",
        "bioassay_method": method,
        "notes": f"PMID:{pmid}; Extracted from abstract",
    }

    rows = []

    # 1. Mortality data
    mortality_data = extract_mortality_data(abstract, insecticides)
    for insect_name, mort_pct, n_tested in mortality_data:
        r = base_row.copy()
        r["insecticide_name"] = insect_name
        r["insecticide_class"] = INSECTICIDE_CLASS_MAP.get(insect_name, "Unknown")
        r["mortality_pct"] = round(mort_pct, 1)
        if n_tested:
            r["n_tested"] = n_tested
        rows.append(r)

    # 2. RR data
    rr_data = extract_rr_data(abstract, insecticides)
    for insect_name, rr_type, rr_val, ci_lo, ci_hi in rr_data:
        if rr_type.startswith("RR"):
            r = base_row.copy()
            r["insecticide_name"] = insect_name
            r["insecticide_class"] = INSECTICIDE_CLASS_MAP.get(insect_name, "Unknown")
            r["rr_type"] = rr_type
            r["rr_value"] = round(rr_val, 2)
            if ci_lo is not None:
                r["rr_ci_lower"] = round(ci_lo, 2)
            if ci_hi is not None:
                r["rr_ci_upper"] = round(ci_hi, 2)
            rows.append(r)

    # 3. kdr data
    kdr_data = extract_kdr_frequency(abstract, mutations)
    for mut, freq, n_geno in kdr_data:
        r = base_row.copy()
        r["gene"] = "VGSC"
        r["codon_position"] = KDR_CODON_MAP.get(mut, "")
        r["mutation"] = mut
        r["allele_frequency"] = round(freq, 4)
        if n_geno:
            r["n_genotyped"] = n_geno
        rows.append(r)

    # 4. Enzyme data
    enzyme_data = extract_enzyme_data(abstract)
    for enzyme_sys, fold_change, elevated_pct in enzyme_data:
        r = base_row.copy()
        r["enzyme_system"] = enzyme_sys
        if fold_change:
            r["fold_change"] = round(fold_change, 2)
        if elevated_pct:
            r["elevated_pct"] = round(elevated_pct, 1)
        rows.append(r)

    # If no specific data was extracted but insecticides are mentioned,
    # create one row per insecticide as a marker
    if not rows and insecticides:
        for insect_name in insecticides:
            r = base_row.copy()
            r["insecticide_name"] = insect_name
            r["insecticide_class"] = INSECTICIDE_CLASS_MAP.get(insect_name, "Unknown")
            r["notes"] = f"PMID:{pmid}; Insecticide mentioned but no specific data extracted from abstract"
            rows.append(r)

    # If still no rows, create a placeholder
    if not rows:
        r = base_row.copy()
        r["notes"] = f"PMID:{pmid}; No specific data extracted - needs full-text review"
        rows.append(r)

    # Add country to all rows if multi-country
    if len(countries) > 1:
        expanded_rows = []
        for c in countries:
            for r in rows:
                rc = r.copy()
                rc["country"] = c
                expanded_rows.append(rc)
        rows = expanded_rows

    return rows


def main():
    print("=" * 60)
    print("Full Data Extraction from Abstracts")
    print("=" * 60)

    # Load data
    ft_df = pd.read_csv(INPUT_ARTICLES, dtype=str)
    data_df = pd.read_csv(INPUT_DATA, dtype=str)

    # Focus on core albopictus articles (Priority 1 and 2)
    core = ft_df[ft_df["final_category"].isin([
        "core_albopictus", "albopictus_check"
    ])]
    print(f"  Processing {len(core)} Ae. albopictus articles")

    # Merge with full data
    merged = core.merge(data_df, on="pmid", how="left", suffixes=("", "_raw"))

    # Extract data from each article
    all_rows = []
    articles_with_data = 0
    articles_no_data = 0

    for idx, row in merged.iterrows():
        extracted = extract_article(row)
        has_real_data = any(
            r.get("mortality_pct") or r.get("rr_value") or
            r.get("allele_frequency") or r.get("enzyme_system")
            for r in extracted
        )
        if has_real_data:
            articles_with_data += 1
        else:
            articles_no_data += 1
        all_rows.extend(extracted)

    # Create DataFrame with template columns
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
    # Ensure all template columns exist
    for col in template_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[template_cols]

    # Remove rows that are just placeholders with no data
    has_data = (
        df["mortality_pct"].notna() & (df["mortality_pct"] != "") |
        df["rr_value"].notna() & (df["rr_value"] != "") |
        df["allele_frequency"].notna() & (df["allele_frequency"] != "") |
        df["enzyme_system"].notna() & (df["enzyme_system"] != "") |
        df["insecticide_name"].notna() & (df["insecticide_name"] != "")
    )
    df = df[has_data].copy()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  Total rows: {len(df)}")

    # ── Summary ──
    summary = []
    summary.append("=" * 60)
    summary.append("FULL DATA EXTRACTION SUMMARY")
    summary.append("=" * 60)
    summary.append(f"\nArticles processed: {len(merged)}")
    summary.append(f"  With extractable data: {articles_with_data}")
    summary.append(f"  No specific data (needs full-text): {articles_no_data}")
    summary.append(f"\nTotal data rows extracted: {len(df)}")

    # Data type counts
    n_mort = df["mortality_pct"].notna() & (df["mortality_pct"] != "")
    n_rr = df["rr_value"].notna() & (df["rr_value"] != "")
    n_kdr = df["allele_frequency"].notna() & (df["allele_frequency"] != "")
    n_enz = df["enzyme_system"].notna() & (df["enzyme_system"] != "")
    summary.append(f"\n  Mortality data rows:     {n_mort.sum()}")
    summary.append(f"  Resistance ratio rows:   {n_rr.sum()}")
    summary.append(f"  kdr mutation rows:       {n_kdr.sum()}")
    summary.append(f"  Enzyme activity rows:    {n_enz.sum()}")

    # Country distribution
    countries = df[df["country"] != ""]["country"].value_counts()
    summary.append(f"\nCountries ({len(countries)}):")
    for c, cnt in countries.head(20).items():
        summary.append(f"  {c}: {cnt} rows")

    # Insecticide distribution
    insecticides = df[df["insecticide_name"] != ""]["insecticide_name"].value_counts()
    summary.append(f"\nInsecticides ({len(insecticides)}):")
    for i, cnt in insecticides.head(15).items():
        summary.append(f"  {i}: {cnt} rows")

    # Insecticide class distribution
    classes = df[df["insecticide_class"] != ""]["insecticide_class"].value_counts()
    summary.append(f"\nInsecticide classes:")
    for c, cnt in classes.items():
        summary.append(f"  {c}: {cnt} rows")

    # kdr mutations
    mutations = df[df["mutation"] != ""]["mutation"].value_counts()
    if len(mutations) > 0:
        summary.append(f"\nkdr mutations:")
        for m, cnt in mutations.items():
            summary.append(f"  {m}: {cnt} rows")

    # Unique studies
    studies = df[df["study_id"] != ""]["study_id"].nunique()
    summary.append(f"\nUnique studies: {studies}")

    # Year range
    years = pd.to_numeric(df["year"], errors="coerce")
    if years.notna().any():
        summary.append(f"Year range: {int(years.min())}-{int(years.max())}")

    summary_text = "\n".join(summary)
    print(summary_text)

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(summary_text)
    print(f"\n  Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
