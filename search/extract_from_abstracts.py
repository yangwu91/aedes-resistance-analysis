#!/usr/bin/env python3
"""
extract_from_abstracts.py – Extract resistance data from abstracts.

For the core Ae. albopictus articles (Priority 1, N=344), parse the
abstracts to extract as much quantitative data as possible:
- Mortality rates with insecticide names
- Resistance ratios (RR50, RR95, LC50, LC95)
- kdr mutation names and frequencies
- Enzyme activity mentions
- Countries and WHO regions
- Bioassay methods

This produces a preliminary extracted dataset that can be verified
and supplemented with full-text data later.
"""

import re
import json
import pandas as pd
from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
INPUT_ARTICLES = SEARCH_DIR / "fulltext_retrieval_list.csv"
INPUT_DATA = SEARCH_DIR / "search_records" / "pubmed_combined_deduplicated.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "03_data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "abstract_extracted_data.csv"
OUTPUT_SUMMARY = SEARCH_DIR / "extraction_summary.txt"

# ── Country patterns ──
COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Armenia",
    "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh",
    "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
    "Bosnia", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso",
    "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde",
    "Central African Republic", "Chad", "Chile", "China", "Colombia",
    "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus",
    "Czech Republic", "Democratic Republic of the Congo", "Denmark",
    "Djibouti", "Dominica", "Dominican Republic", "East Timor", "Ecuador",
    "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
    "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia",
    "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala",
    "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary",
    "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya",
    "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",
    "Lesotho", "Liberia", "Libya", "Lithuania", "Luxembourg", "Madagascar",
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Mauritania",
    "Mauritius", "Mexico", "Micronesia", "Moldova", "Mongolia", "Montenegro",
    "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal", "Netherlands",
    "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea",
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine",
    "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
    "Poland", "Portugal", "Puerto Rico", "Qatar", "Romania", "Russia",
    "Rwanda", "Saudi Arabia", "Senegal", "Serbia", "Sierra Leone",
    "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
    "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka",
    "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan",
    "Tajikistan", "Tanzania", "Thailand", "Togo", "Trinidad",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Uganda",
    "Ukraine", "United Arab Emirates", "UAE", "United Kingdom", "UK",
    "United States", "USA", "US", "Uruguay", "Uzbekistan", "Vanuatu",
    "Venezuela", "Vietnam", "Viet Nam", "Yemen", "Zambia", "Zimbabwe",
    # Common sub-national references
    "Hainan", "Yunnan", "Guangdong", "Guangzhou", "Zhejiang", "Fujian",
    "Jiangsu", "Shandong", "Sichuan", "Hubei", "Hunan",
    "Sabah", "Sarawak", "Penang", "Selangor", "Kuala Lumpur",
    "Delhi", "Mumbai", "Kolkata", "Chennai", "Kerala", "Karnataka",
    "Java", "Sumatra", "Bali", "Sulawesi", "Kalimantan",
    "Bangkok", "Chiang Mai",
    "Réunion", "Reunion", "La Réunion", "Mayotte", "Guadeloupe",
    "Martinique", "French Guiana", "New Caledonia",
    "Sardinia", "Corsica", "Rome", "Milan",
    "Athens", "Crete",
    "Florida", "Texas", "California", "Hawaii",
]

# Map sub-national to country
SUBNATIONAL_MAP = {
    "Hainan": "China", "Yunnan": "China", "Guangdong": "China",
    "Guangzhou": "China", "Zhejiang": "China", "Fujian": "China",
    "Jiangsu": "China", "Shandong": "China", "Sichuan": "China",
    "Hubei": "China", "Hunan": "China",
    "Sabah": "Malaysia", "Sarawak": "Malaysia", "Penang": "Malaysia",
    "Selangor": "Malaysia", "Kuala Lumpur": "Malaysia",
    "Delhi": "India", "Mumbai": "India", "Kolkata": "India",
    "Chennai": "India", "Kerala": "India", "Karnataka": "India",
    "Java": "Indonesia", "Sumatra": "Indonesia", "Bali": "Indonesia",
    "Sulawesi": "Indonesia", "Kalimantan": "Indonesia",
    "Bangkok": "Thailand", "Chiang Mai": "Thailand",
    "Réunion": "France", "Reunion": "France", "La Réunion": "France",
    "Mayotte": "France", "Guadeloupe": "France", "Martinique": "France",
    "French Guiana": "France", "New Caledonia": "France",
    "Sardinia": "Italy", "Corsica": "France", "Rome": "Italy", "Milan": "Italy",
    "Athens": "Greece", "Crete": "Greece",
    "Florida": "United States", "Texas": "United States",
    "California": "United States", "Hawaii": "United States",
    "UAE": "United Arab Emirates", "UK": "United Kingdom",
    "USA": "United States", "US": "United States",
    "Viet Nam": "Vietnam",
}

# ── Insecticide patterns ──
INSECTICIDES = {
    "deltamethrin": "Pyrethroid", "permethrin": "Pyrethroid",
    "cypermethrin": "Pyrethroid", "alpha-cypermethrin": "Pyrethroid",
    "lambda-cyhalothrin": "Pyrethroid", "cyhalothrin": "Pyrethroid",
    "etofenprox": "Pyrethroid", "bifenthrin": "Pyrethroid",
    "cyfluthrin": "Pyrethroid", "beta-cyfluthrin": "Pyrethroid",
    "allethrin": "Pyrethroid", "d-allethrin": "Pyrethroid",
    "esbiothrin": "Pyrethroid", "prallethrin": "Pyrethroid",
    "transfluthrin": "Pyrethroid", "metofluthrin": "Pyrethroid",
    "resmethrin": "Pyrethroid",
    "malathion": "Organophosphate", "temephos": "Organophosphate",
    "fenitrothion": "Organophosphate", "pirimiphos-methyl": "Organophosphate",
    "chlorpyrifos": "Organophosphate", "fenthion": "Organophosphate",
    "naled": "Organophosphate",
    "propoxur": "Carbamate", "bendiocarb": "Carbamate",
    "carbaryl": "Carbamate",
    "DDT": "Organochlorine", "dieldrin": "Organochlorine",
    "lindane": "Organochlorine", "HCH": "Organochlorine",
    "imidacloprid": "Neonicotinoid", "clothianidin": "Neonicotinoid",
    "thiamethoxam": "Neonicotinoid", "acetamiprid": "Neonicotinoid",
    "dinotefuran": "Neonicotinoid",
    "pyriproxyfen": "IGR", "methoprene": "IGR",
    "diflubenzuron": "IGR", "novaluron": "IGR",
    "Bti": "Biological", "spinosad": "Biological",
}

# ── kdr mutations ──
KDR_MUTATIONS = [
    "F1534C", "F1534S", "F1534L",
    "V1016G", "V1016I",
    "S989P",
    "I1532T",
    "L1014F", "L1014S",
    "D1763Y",
]


def find_countries(text):
    """Find all country mentions in text."""
    found = set()
    if not text:
        return found
    for c in COUNTRIES:
        pattern = r'\b' + re.escape(c) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            country = SUBNATIONAL_MAP.get(c, c)
            found.add(country)
    return found


def find_insecticides(text):
    """Find all insecticide mentions in text."""
    found = set()
    if not text:
        return found
    for name in INSECTICIDES:
        if re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE):
            found.add(name)
    return found


def find_kdr_mutations(text):
    """Find kdr mutation mentions."""
    found = set()
    if not text:
        return found
    for mut in KDR_MUTATIONS:
        if re.search(r'\b' + re.escape(mut) + r'\b', text, re.IGNORECASE):
            found.add(mut)
    return found


def find_mortality_data(text):
    """Extract mortality percentages and associated insecticides."""
    data = []
    if not text:
        return data

    # Pattern: insecticide ... X% mortality / mortality X%
    patterns = [
        r'(\w+(?:[-]\w+)?)\s*(?:\([^)]*\))?\s*(?:showed?|exhibited?|had|resulted?\s+in|with|:)?\s*(\d+\.?\d*)\s*%\s*(?:mortality|kill|dead)',
        r'mortality\s*(?:rate|was|of|=|:)?\s*(\d+\.?\d*)\s*%\s*(?:for|with|against|to)?\s*(\w+)',
        r'(\d+\.?\d*)\s*%\s*(?:mortality|knockdown)\s*(?:for|with|to|against)\s*(\w+)',
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            groups = m.groups()
            data.append(groups)

    return data


def find_bioassay_method(text):
    """Identify bioassay method."""
    if not text:
        return ""
    if re.search(r'\bWHO\s+tube\b', text, re.IGNORECASE):
        return "WHO tube"
    if re.search(r'\bWHO\s+bottle\b', text, re.IGNORECASE):
        return "WHO bottle"
    if re.search(r'\bCDC\s+bottle\b', text, re.IGNORECASE):
        return "CDC bottle"
    if re.search(r'\blarv\w*\s+bioassay\b', text, re.IGNORECASE):
        return "Larval bioassay"
    if re.search(r'\bbioassay\b', text, re.IGNORECASE):
        return "Bioassay (unspecified)"
    return ""


def find_life_stage(text):
    """Identify life stage tested."""
    if not text:
        return ""
    if re.search(r'\badult\b', text, re.IGNORECASE):
        if re.search(r'\blarv', text, re.IGNORECASE):
            return "Adult,Larva"
        return "Adult"
    if re.search(r'\blarv', text, re.IGNORECASE):
        return "Larva"
    return ""


def find_enzyme_mentions(text):
    """Find metabolic enzyme mentions."""
    found = set()
    if not text:
        return found
    patterns = {
        "MFO": [r"\bP450\b", r"\bCYP\d", r"\bmonooxygenase\b", r"\bMFO\b",
                 r"\bmixed[- ]function\s+oxidase\b"],
        "NSE": [r"\besterase\b", r"\bNSE\b", r"\bcarboxylesterase\b",
                r"\bnon[- ]specific\s+esterase\b"],
        "GST": [r"\bGST\b", r"\bglutathione\b"],
        "AChE": [r"\bAChE\b", r"\bacetylcholinesterase\b"],
    }
    for enzyme, pats in patterns.items():
        for p in pats:
            if re.search(p, text, re.IGNORECASE):
                found.add(enzyme)
                break
    return found


def extract_study_data(row):
    """Extract all available data from an article's abstract."""
    pmid = row.get("pmid", "")
    title = str(row.get("title", ""))
    abstract = str(row.get("abstract", ""))
    first_author = str(row.get("first_author", ""))
    year = str(row.get("year", ""))
    journal = str(row.get("journal", ""))
    doi = str(row.get("doi", ""))
    mesh = str(row.get("mesh_terms", ""))
    full_text = f"{title} {abstract} {mesh}"

    # Construct study_id
    author_part = first_author.split()[0] if first_author else "Unknown"
    study_id = f"{author_part}_{year}"

    # Extract data elements
    countries = find_countries(full_text)
    insecticides = find_insecticides(full_text)
    kdr_mutations = find_kdr_mutations(full_text)
    enzymes = find_enzyme_mentions(full_text)
    bioassay = find_bioassay_method(full_text)
    life_stage = find_life_stage(full_text)

    # Determine data types available
    has_mortality = bool(re.search(
        r'mortality|knockdown|kill|dead|susceptib|resistan.*\d+\s*%',
        abstract, re.IGNORECASE
    ))
    has_rr = bool(re.search(
        r'\b(RR|resistance\s+ratio|LC\s*50|LC\s*95|LD\s*50)\b',
        abstract, re.IGNORECASE
    ))
    has_kdr = bool(kdr_mutations) or bool(re.search(
        r'\bkdr\b|\bknockdown\s+resistance\b|\ballele\s+frequen',
        abstract, re.IGNORECASE
    ))
    has_enzyme = bool(enzymes) or bool(re.search(
        r'\benzyme\s+activ|\bbiochemical\s+assay|\bsynergist\b',
        abstract, re.IGNORECASE
    ))

    return {
        "pmid": pmid,
        "study_id": study_id,
        "first_author": first_author,
        "year": year,
        "journal": journal,
        "doi": doi,
        "countries": "; ".join(sorted(countries)) if countries else "",
        "insecticides_mentioned": "; ".join(sorted(insecticides)) if insecticides else "",
        "insecticide_classes": "; ".join(sorted(set(
            INSECTICIDES[i] for i in insecticides if i in INSECTICIDES
        ))) if insecticides else "",
        "n_insecticides": len(insecticides),
        "kdr_mutations": "; ".join(sorted(kdr_mutations)) if kdr_mutations else "",
        "enzymes_mentioned": "; ".join(sorted(enzymes)) if enzymes else "",
        "bioassay_method": bioassay,
        "life_stage": life_stage,
        "has_mortality_data": 1 if has_mortality else 0,
        "has_rr_data": 1 if has_rr else 0,
        "has_kdr_data": 1 if has_kdr else 0,
        "has_enzyme_data": 1 if has_enzyme else 0,
        "data_types": "; ".join(filter(None, [
            "Mortality" if has_mortality else "",
            "RR" if has_rr else "",
            "kdr" if has_kdr else "",
            "Enzyme" if has_enzyme else "",
        ])),
        "title": title,
    }


def main():
    print("=" * 60)
    print("Abstract Data Extraction")
    print("=" * 60)

    # Load full-text list and article data
    ft_df = pd.read_csv(INPUT_ARTICLES, dtype=str)
    data_df = pd.read_csv(INPUT_DATA, dtype=str)

    # Focus on core albopictus and albopictus_check articles
    core = ft_df[ft_df["final_category"].isin([
        "core_albopictus", "albopictus_check"
    ])]
    print(f"  Processing {len(core)} Ae. albopictus articles")

    # Merge with full article data
    merged = core.merge(data_df, on="pmid", how="left", suffixes=("", "_raw"))

    # Extract data from each article
    extracted = []
    for _, row in merged.iterrows():
        study_data = extract_study_data(row)
        study_data["screening_category"] = row.get("final_category", "")
        extracted.append(study_data)

    ext_df = pd.DataFrame(extracted)
    ext_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved: {OUTPUT_FILE}")

    # ── Summary statistics ──
    summary = []
    summary.append("=" * 60)
    summary.append("ABSTRACT EXTRACTION SUMMARY")
    summary.append("=" * 60)
    summary.append(f"\nArticles processed: {len(ext_df)}")
    summary.append(f"\nData type availability:")
    summary.append(f"  Mortality data:  {ext_df['has_mortality_data'].astype(int).sum()}")
    summary.append(f"  RR data:         {ext_df['has_rr_data'].astype(int).sum()}")
    summary.append(f"  kdr data:        {ext_df['has_kdr_data'].astype(int).sum()}")
    summary.append(f"  Enzyme data:     {ext_df['has_enzyme_data'].astype(int).sum()}")

    # Countries
    all_countries = set()
    for c in ext_df["countries"]:
        if c:
            for cc in str(c).split("; "):
                if cc.strip():
                    all_countries.add(cc.strip())
    summary.append(f"\nCountries identified: {len(all_countries)}")
    country_counts = {}
    for c in ext_df["countries"]:
        if c:
            for cc in str(c).split("; "):
                cc = cc.strip()
                if cc:
                    country_counts[cc] = country_counts.get(cc, 0) + 1
    for cc, cnt in sorted(country_counts.items(), key=lambda x: -x[1])[:20]:
        summary.append(f"  {cc}: {cnt}")

    # Insecticides
    insect_counts = {}
    for i in ext_df["insecticides_mentioned"]:
        if i:
            for ii in str(i).split("; "):
                ii = ii.strip()
                if ii:
                    insect_counts[ii] = insect_counts.get(ii, 0) + 1
    summary.append(f"\nInsecticides mentioned: {len(insect_counts)}")
    for ii, cnt in sorted(insect_counts.items(), key=lambda x: -x[1])[:15]:
        summary.append(f"  {ii}: {cnt}")

    # kdr mutations
    kdr_counts = {}
    for k in ext_df["kdr_mutations"]:
        if k:
            for kk in str(k).split("; "):
                kk = kk.strip()
                if kk:
                    kdr_counts[kk] = kdr_counts.get(kk, 0) + 1
    summary.append(f"\nkdr mutations mentioned: {len(kdr_counts)}")
    for kk, cnt in sorted(kdr_counts.items(), key=lambda x: -x[1]):
        summary.append(f"  {kk}: {cnt}")

    # Insecticide class distribution
    class_counts = {}
    for c in ext_df["insecticide_classes"]:
        if c:
            for cc in str(c).split("; "):
                cc = cc.strip()
                if cc:
                    class_counts[cc] = class_counts.get(cc, 0) + 1
    summary.append(f"\nInsecticide classes:")
    for cc, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        summary.append(f"  {cc}: {cnt}")

    # Year distribution
    year_counts = ext_df["year"].value_counts().sort_index()
    summary.append(f"\nArticles by year:")
    for yr, cnt in year_counts.items():
        summary.append(f"  {yr}: {cnt}")

    summary_text = "\n".join(summary)
    print(summary_text)

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(summary_text)
    print(f"\n  Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
