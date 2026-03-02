#!/usr/bin/env python3
"""
fetch_pmc_fulltext.py – Fetch full-text articles from PubMed Central.

Checks which of our core articles are available in PMC (open access)
and downloads their full XML text for more detailed data extraction.
"""

import time
import json
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
INPUT_ARTICLES = SEARCH_DIR / "fulltext_retrieval_list.csv"
INPUT_DATA = SEARCH_DIR / "search_records" / "pubmed_combined_deduplicated.csv"
PMC_DIR = SEARCH_DIR / "pmc_texts"
PMC_DIR.mkdir(exist_ok=True)
OUTPUT_PMC_MAP = SEARCH_DIR / "pmc_mapping.csv"

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BATCH_SIZE = 100
SLEEP_TIME = 0.35  # stay under 3 req/sec

HEADERS = {
    "User-Agent": "InsecticideResistanceMetaAnalysis/1.0 (systematic review research)"
}


def convert_pmids_to_pmc(pmids):
    """Use ELink to find PMC IDs for a batch of PMIDs."""
    url = f"{BASE_URL}elink.fcgi"
    pmid_str = ",".join(str(p) for p in pmids)
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid_str,
        "retmode": "json",
        "linkname": "pubmed_pmc",
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        mapping = {}
        linksets = data.get("linksets", [])
        for ls in linksets:
            pmid = ls.get("ids", [None])[0]
            linksetdbs = ls.get("linksetdbs", [])
            for lsdb in linksetdbs:
                if lsdb.get("linkname") == "pubmed_pmc":
                    links = lsdb.get("links", [])
                    if links:
                        mapping[str(pmid)] = str(links[0])
        return mapping
    except Exception as e:
        print(f"    Error in ELink: {e}")
        return {}


def fetch_pmc_text(pmc_id):
    """Fetch full text from PMC in plain text format."""
    url = f"{BASE_URL}efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmc_id,
        "rettype": "xml",
        "retmode": "xml",
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"    Error fetching PMC{pmc_id}: {e}")
        return None


def extract_text_from_pmc_xml(xml_str):
    """Extract plain text from PMC XML."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return ""

    text_parts = []

    # Get article body
    for body in root.iter("body"):
        for elem in body.iter():
            if elem.text:
                text_parts.append(elem.text.strip())
            if elem.tail:
                text_parts.append(elem.tail.strip())

    # Also get tables (often contain key data)
    for table_wrap in root.iter("table-wrap"):
        for elem in table_wrap.iter():
            if elem.text:
                text_parts.append(elem.text.strip())
            if elem.tail:
                text_parts.append(elem.tail.strip())

    return " ".join(text_parts)


def main():
    print("=" * 60)
    print("Fetch Full Texts from PubMed Central")
    print("=" * 60)

    # Load our core articles
    ft_df = pd.read_csv(INPUT_ARTICLES, dtype=str)
    core = ft_df[ft_df["final_category"].isin([
        "core_albopictus", "albopictus_check"
    ])]
    pmids = core["pmid"].tolist()
    print(f"  Checking {len(pmids)} articles for PMC availability")

    # Batch check PMC availability
    pmc_mapping = {}
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i:i+BATCH_SIZE]
        print(f"  Batch {i//BATCH_SIZE + 1}/{(len(pmids)-1)//BATCH_SIZE + 1}...")
        result = convert_pmids_to_pmc(batch)
        pmc_mapping.update(result)
        time.sleep(SLEEP_TIME)

    print(f"\n  Found {len(pmc_mapping)} articles in PMC ({len(pmc_mapping)/len(pmids)*100:.1f}%)")

    # Save mapping
    mapping_df = pd.DataFrame([
        {"pmid": k, "pmc_id": v} for k, v in pmc_mapping.items()
    ])
    mapping_df.to_csv(OUTPUT_PMC_MAP, index=False)
    print(f"  Saved PMC mapping: {OUTPUT_PMC_MAP}")

    # Fetch full texts (limit to manageable number)
    n_to_fetch = len(pmc_mapping)
    print(f"\n  Fetching {n_to_fetch} full texts from PMC...")

    fetched = 0
    failed = 0
    for pmid, pmc_id in list(pmc_mapping.items()):
        outfile = PMC_DIR / f"PMC{pmc_id}.txt"
        if outfile.exists():
            fetched += 1
            continue

        xml_text = fetch_pmc_text(pmc_id)
        if xml_text:
            plain_text = extract_text_from_pmc_xml(xml_text)
            if plain_text:
                outfile.write_text(plain_text)
                fetched += 1
            else:
                failed += 1
        else:
            failed += 1

        time.sleep(SLEEP_TIME)
        if (fetched + failed) % 20 == 0:
            print(f"    Progress: {fetched} fetched, {failed} failed")

    print(f"\n  Fetched: {fetched}")
    print(f"  Failed: {failed}")
    print(f"  Texts saved in: {PMC_DIR}")


if __name__ == "__main__":
    main()
