#!/usr/bin/env python3
"""
fetch_pmc_v2.py – Fetch PMC full texts using the ID converter API.

Uses the NCBI PMC ID converter to find PMC IDs, then fetches
full text XML from PMC for data extraction.
"""

import time
import re
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
INPUT_ARTICLES = SEARCH_DIR / "fulltext_retrieval_list.csv"
PMC_DIR = SEARCH_DIR / "pmc_texts"
PMC_DIR.mkdir(exist_ok=True)
OUTPUT_PMC_MAP = SEARCH_DIR / "pmc_mapping.csv"

CONVERTER_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200
SLEEP_TIME = 0.4

HEADERS = {
    "User-Agent": "InsecticideResistanceMetaAnalysis/1.0"
}


def extract_text_from_pmc_xml(xml_str):
    """Extract plain text from PMC XML, preserving table data."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return ""

    text_parts = []

    # Get abstract
    for abstract in root.iter("abstract"):
        for elem in abstract.iter():
            if elem.text:
                text_parts.append(elem.text.strip())
            if elem.tail:
                text_parts.append(elem.tail.strip())

    # Get article body
    for body in root.iter("body"):
        for elem in body.iter():
            if elem.text:
                text_parts.append(elem.text.strip())
            if elem.tail:
                text_parts.append(elem.tail.strip())

    # Get tables (critical for data extraction)
    for table_wrap in root.iter("table-wrap"):
        for elem in table_wrap.iter():
            if elem.text:
                text_parts.append(elem.text.strip())
            if elem.tail:
                text_parts.append(elem.tail.strip())

    # Get supplementary data captions
    for supp in root.iter("supplementary-material"):
        for elem in supp.iter():
            if elem.text:
                text_parts.append(elem.text.strip())

    return " ".join(text_parts)


def main():
    print("=" * 60)
    print("Fetch PMC Full Texts (v2 - ID Converter)")
    print("=" * 60)

    # Load core articles
    ft_df = pd.read_csv(INPUT_ARTICLES, dtype=str)
    core = ft_df[ft_df["final_category"].isin([
        "core_albopictus", "albopictus_check"
    ])]
    pmids = core["pmid"].tolist()
    print(f"  Checking {len(pmids)} articles for PMC availability")

    # Step 1: Convert PMIDs to PMC IDs
    pmc_mapping = {}
    for i in range(0, len(pmids), BATCH_SIZE):
        batch = pmids[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(pmids) - 1) // BATCH_SIZE + 1
        print(f"  ID conversion batch {batch_num}/{total_batches}...")

        try:
            params = {
                "ids": ",".join(batch),
                "format": "json",
                "tool": "InsecticideResistanceMetaAnalysis",
                "email": "research@example.com",
            }
            resp = requests.get(CONVERTER_URL, params=params,
                              headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for rec in data.get("records", []):
                    pmid = rec.get("pmid", "")
                    pmcid = rec.get("pmcid", "")
                    if pmcid:
                        pmc_mapping[pmid] = pmcid
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(SLEEP_TIME)

    print(f"\n  PMC articles found: {len(pmc_mapping)} / {len(pmids)} ({len(pmc_mapping)/len(pmids)*100:.1f}%)")

    # Save mapping
    mapping_rows = [{"pmid": k, "pmc_id": v} for k, v in pmc_mapping.items()]
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(OUTPUT_PMC_MAP, index=False)

    # Step 2: Fetch full texts
    n_fetch = len(pmc_mapping)
    print(f"\n  Fetching {n_fetch} full texts...")

    fetched = 0
    skipped = 0
    failed = 0

    for pmid, pmc_id in pmc_mapping.items():
        # Remove PMC prefix for efetch
        pmc_num = pmc_id.replace("PMC", "")
        outfile = PMC_DIR / f"{pmc_id}.txt"

        if outfile.exists() and outfile.stat().st_size > 100:
            skipped += 1
            continue

        try:
            params = {
                "db": "pmc",
                "id": pmc_num,
                "rettype": "xml",
                "retmode": "xml",
            }
            resp = requests.get(EFETCH_URL, params=params,
                              headers=HEADERS, timeout=60)
            resp.raise_for_status()

            plain_text = extract_text_from_pmc_xml(resp.text)
            if plain_text and len(plain_text) > 500:
                outfile.write_text(plain_text)
                fetched += 1
            else:
                failed += 1

        except Exception as e:
            failed += 1
            if fetched + failed + skipped <= 5:
                print(f"    Error for {pmc_id}: {e}")

        time.sleep(SLEEP_TIME)

        total_done = fetched + failed + skipped
        if total_done % 50 == 0:
            print(f"    Progress: {total_done}/{n_fetch} "
                  f"(fetched: {fetched}, cached: {skipped}, failed: {failed})")

    print(f"\n  Results:")
    print(f"    New fetched: {fetched}")
    print(f"    Already cached: {skipped}")
    print(f"    Failed: {failed}")
    print(f"    Total available: {fetched + skipped}")
    print(f"    Texts in: {PMC_DIR}")

    # Quick stats on text sizes
    sizes = []
    for f in PMC_DIR.glob("*.txt"):
        sizes.append(f.stat().st_size)
    if sizes:
        print(f"\n  Text file stats:")
        print(f"    Files: {len(sizes)}")
        print(f"    Avg size: {sum(sizes)//len(sizes)//1024} KB")
        print(f"    Min size: {min(sizes)//1024} KB")
        print(f"    Max size: {max(sizes)//1024} KB")


if __name__ == "__main__":
    main()
