#!/usr/bin/env python3
"""
PubMed Systematic Search for Insecticide Resistance in Aedes albopictus
Uses NCBI E-utilities API to search, retrieve, and export results.
"""

import requests
import xml.etree.ElementTree as ET
import csv
import time
import json
import os
from datetime import datetime

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Search query - combining species, resistance, and insecticide concepts
SEARCH_QUERY = """
(("Aedes albopictus"[tiab] OR "Ae. albopictus"[tiab] OR "Asian tiger mosquito"[tiab]
  OR "Stegomyia albopicta"[tiab] OR "Aedes"[MeSH])
 AND
 ("insecticide resistance"[tiab] OR "pesticide resistance"[tiab] OR
  "susceptibility status"[tiab] OR "resistance status"[tiab] OR
  "resistance monitoring"[tiab] OR "resistance level"[tiab] OR
  "bioassay"[tiab] OR "kdr"[tiab] OR "knockdown resistance"[tiab] OR
  "cross-resistance"[tiab] OR "cross resistance"[tiab] OR
  "resistance ratio"[tiab] OR "LC50"[tiab] OR "LC95"[tiab] OR
  "RR50"[tiab] OR "RR95"[tiab] OR
  "mortality rate"[tiab] OR "percent mortality"[tiab] OR
  "KT50"[tiab] OR "KDT50"[tiab] OR
  "diagnostic dose"[tiab] OR "discriminating dose"[tiab] OR
  "metabolic resistance"[tiab] OR "target site mutation"[tiab] OR
  "Insecticide Resistance"[MeSH])
 AND
 ("pyrethroid"[tiab] OR "pyrethroids"[tiab] OR
  "organophosphate"[tiab] OR "organophosphates"[tiab] OR
  "carbamate"[tiab] OR "carbamates"[tiab] OR
  "organochlorine"[tiab] OR "organochlorines"[tiab] OR
  "neonicotinoid"[tiab] OR "neonicotinoids"[tiab] OR
  "insect growth regulator"[tiab] OR
  "deltamethrin"[tiab] OR "permethrin"[tiab] OR "cypermethrin"[tiab] OR
  "lambda-cyhalothrin"[tiab] OR "alpha-cypermethrin"[tiab] OR
  "etofenprox"[tiab] OR "bifenthrin"[tiab] OR
  "malathion"[tiab] OR "temephos"[tiab] OR "fenitrothion"[tiab] OR
  "pirimiphos-methyl"[tiab] OR "chlorpyrifos"[tiab] OR
  "propoxur"[tiab] OR "bendiocarb"[tiab] OR
  "DDT"[tiab] OR "dieldrin"[tiab] OR
  "imidacloprid"[tiab] OR "clothianidin"[tiab] OR
  "pyriproxyfen"[tiab] OR "methoprene"[tiab] OR
  "Bacillus thuringiensis"[tiab] OR "Bti"[tiab] OR
  "spinosad"[tiab] OR "chlorfenapyr"[tiab] OR
  "Insecticides"[MeSH] OR "Pyrethrins"[MeSH]))
""".strip().replace("\n", " ")

# Broader query to also capture Ae. aegypti and other mosquitoes for comparison
BROAD_QUERY = """
(("Aedes"[tiab] OR "Culex"[tiab] OR "Anopheles"[tiab] OR
  "mosquito"[tiab] OR "mosquitoes"[tiab] OR "Culicidae"[MeSH])
 AND
 ("insecticide resistance"[tiab] OR "Insecticide Resistance"[MeSH])
 AND
 ("cross-resistance"[tiab] OR "cross resistance"[tiab] OR
  "multiple resistance"[tiab] OR "resistance management"[tiab] OR
  "rotation"[tiab] OR "mosaic"[tiab] OR "mixture"[tiab]))
""".strip().replace("\n", " ")


def search_pubmed(query, retmax=10000):
    """Search PubMed and return list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "usehistory": "y",
    }
    response = requests.get(f"{BASE_URL}esearch.fcgi", params=params)
    response.raise_for_status()
    data = response.json()

    result = data["esearchresult"]
    count = int(result["count"])
    pmids = result["idlist"]
    webenv = result.get("webenv", "")
    query_key = result.get("querykey", "")

    print(f"  Found {count} results, retrieved {len(pmids)} PMIDs")
    return pmids, count, webenv, query_key


def fetch_details(pmids, batch_size=50):
    """Fetch article details for a list of PMIDs with retry logic."""
    all_articles = []
    max_retries = 3

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        print(f"  Fetching details for PMIDs {i+1}-{i+len(batch)} of {len(pmids)}...")

        for attempt in range(max_retries):
            try:
                params = {
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "xml",
                    "retmode": "xml",
                }
                response = requests.get(
                    f"{BASE_URL}efetch.fcgi",
                    params=params,
                    timeout=60,
                )
                response.raise_for_status()

                root = ET.fromstring(response.text)

                for article in root.findall(".//PubmedArticle"):
                    try:
                        record = parse_article(article)
                        all_articles.append(record)
                    except Exception as e:
                        pmid_elem = article.find(".//PMID")
                        pmid = pmid_elem.text if pmid_elem is not None else "unknown"
                        print(f"    Warning: Failed to parse PMID {pmid}: {e}")

                break  # Success, exit retry loop

            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"    Retry {attempt+1}/{max_retries} after {wait}s: {type(e).__name__}")
                    time.sleep(wait)
                else:
                    print(f"    Failed after {max_retries} attempts: {e}")

        # Respect rate limit (3 requests/sec without API key)
        time.sleep(0.5)

    return all_articles


def parse_article(article):
    """Parse a PubmedArticle XML element into a dictionary."""
    # PMID
    pmid = article.findtext(".//PMID", default="")

    # Title
    title_elem = article.find(".//ArticleTitle")
    title = "".join(title_elem.itertext()) if title_elem is not None else ""

    # Authors
    authors = []
    for author in article.findall(".//Author"):
        lastname = author.findtext("LastName", default="")
        forename = author.findtext("ForeName", default="")
        initials = author.findtext("Initials", default="")
        if lastname:
            authors.append(f"{lastname} {initials}" if initials else lastname)
    authors_str = "; ".join(authors)

    # First author
    first_author = authors[0] if authors else ""

    # Journal
    journal = article.findtext(".//Journal/Title", default="")
    journal_abbrev = article.findtext(".//Journal/ISOAbbreviation", default="")

    # Year
    year = article.findtext(".//Journal/JournalIssue/PubDate/Year", default="")
    if not year:
        medline_date = article.findtext(".//Journal/JournalIssue/PubDate/MedlineDate", default="")
        if medline_date:
            year = medline_date[:4]

    # Volume, Issue, Pages
    volume = article.findtext(".//Journal/JournalIssue/Volume", default="")
    issue = article.findtext(".//Journal/JournalIssue/Issue", default="")
    pages = article.findtext(".//Pagination/MedlinePgn", default="")

    # DOI
    doi = ""
    for eid in article.findall(".//ArticleIdList/ArticleId"):
        if eid.get("IdType") == "doi":
            doi = eid.text or ""
            break

    # Abstract
    abstract_parts = []
    for abs_text in article.findall(".//Abstract/AbstractText"):
        label = abs_text.get("Label", "")
        text = "".join(abs_text.itertext())
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    # MeSH terms
    mesh_terms = []
    for mesh in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
        mesh_terms.append(mesh.text or "")
    mesh_str = "; ".join(mesh_terms)

    # Keywords
    keywords = []
    for kw in article.findall(".//KeywordList/Keyword"):
        keywords.append(kw.text or "")
    keywords_str = "; ".join(keywords)

    # Publication type
    pub_types = []
    for pt in article.findall(".//PublicationTypeList/PublicationType"):
        pub_types.append(pt.text or "")
    pub_types_str = "; ".join(pub_types)

    return {
        "pmid": pmid,
        "title": title,
        "first_author": first_author,
        "authors": authors_str,
        "journal": journal,
        "journal_abbrev": journal_abbrev,
        "year": year,
        "volume": volume,
        "issue": issue,
        "pages": pages,
        "doi": doi,
        "abstract": abstract,
        "mesh_terms": mesh_str,
        "keywords": keywords_str,
        "pub_type": pub_types_str,
    }


def save_to_csv(articles, filename):
    """Save articles to CSV file."""
    filepath = os.path.join(OUTPUT_DIR, "search_records", filename)
    if not articles:
        print(f"  No articles to save to {filename}")
        return

    fieldnames = articles[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles)

    print(f"  Saved {len(articles)} articles to {filepath}")


def save_search_log(queries_info, filename="search_log.json"):
    """Save search execution log."""
    filepath = os.path.join(OUTPUT_DIR, "search_records", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(queries_info, f, indent=2, ensure_ascii=False)
    print(f"  Search log saved to {filepath}")


def main():
    print("=" * 70)
    print("PubMed Systematic Search: Insecticide Resistance in Aedes albopictus")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    search_log = {
        "search_date": datetime.now().isoformat(),
        "searches": [],
    }

    # Search 1: Primary query (Aedes albopictus focused)
    print("\n[1/2] Primary search: Aedes + resistance + insecticides...")
    pmids_primary, count_primary, webenv, qkey = search_pubmed(SEARCH_QUERY)

    search_log["searches"].append({
        "name": "primary_aedes_resistance",
        "query": SEARCH_QUERY,
        "total_results": count_primary,
        "pmids_retrieved": len(pmids_primary),
    })

    if pmids_primary:
        articles_primary = fetch_details(pmids_primary)
        save_to_csv(articles_primary, "pubmed_primary_results.csv")
    else:
        articles_primary = []

    # Search 2: Broad query (cross-resistance and management)
    print("\n[2/2] Broad search: mosquitoes + cross-resistance/management...")
    pmids_broad, count_broad, _, _ = search_pubmed(BROAD_QUERY)

    search_log["searches"].append({
        "name": "broad_cross_resistance_management",
        "query": BROAD_QUERY,
        "total_results": count_broad,
        "pmids_retrieved": len(pmids_broad),
    })

    if pmids_broad:
        articles_broad = fetch_details(pmids_broad)
        save_to_csv(articles_broad, "pubmed_broad_results.csv")
    else:
        articles_broad = []

    # Combine and deduplicate
    print("\n[Dedup] Combining and deduplicating results...")
    all_pmids = set(pmids_primary + pmids_broad)

    # Build combined dict keyed by PMID
    combined = {}
    for a in articles_primary + articles_broad:
        if a["pmid"] not in combined:
            combined[a["pmid"]] = a

    combined_list = sorted(combined.values(), key=lambda x: x.get("year", "0"), reverse=True)
    save_to_csv(combined_list, "pubmed_combined_deduplicated.csv")

    search_log["total_unique_results"] = len(combined_list)
    search_log["deduplication"] = {
        "primary_count": len(pmids_primary),
        "broad_count": len(pmids_broad),
        "duplicates_removed": len(pmids_primary) + len(pmids_broad) - len(all_pmids),
        "final_unique_count": len(all_pmids),
    }

    save_search_log(search_log)

    # Summary
    print("\n" + "=" * 70)
    print("SEARCH SUMMARY")
    print("=" * 70)
    print(f"Primary search results:         {count_primary}")
    print(f"Broad search results:           {count_broad}")
    print(f"Duplicates removed:             {len(pmids_primary) + len(pmids_broad) - len(all_pmids)}")
    print(f"Total unique articles:          {len(all_pmids)}")
    print(f"Articles with details fetched:  {len(combined_list)}")
    print("=" * 70)

    # Year distribution
    year_counts = {}
    for a in combined_list:
        y = a.get("year", "unknown")
        year_counts[y] = year_counts.get(y, 0) + 1

    print("\nYear distribution (top 15):")
    for year, count in sorted(year_counts.items(), reverse=True)[:15]:
        bar = "#" * min(count, 50)
        print(f"  {year}: {count:4d} {bar}")


if __name__ == "__main__":
    main()
