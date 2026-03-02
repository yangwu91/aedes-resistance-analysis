# Analysis Scripts: Resistance Management Strategies for *Aedes albopictus*

Scripts supporting the study:

> **Resistance Management Strategies for *Aedes albopictus* Informed by
> Global Resistance Profiles and Cross-Resistance Network Analysis**

---

## Repository Structure

```
scripts/
├── requirements.txt          Python dependencies
├── search/                   Literature search & data extraction
│   ├── pubmed_search.py      PubMed E-utilities search (returns raw records)
│   ├── screen_articles.py    Phase 1 title/abstract screening
│   ├── screen_phase2.py      Phase 2 screening with stricter criteria
│   ├── screen_final.py       Final inclusion/exclusion decision
│   ├── prisma_flowchart.py   PRISMA 2020 flow diagram generator
│   ├── fetch_pmc_fulltext.py Fetch full text from PubMed Central (v1)
│   ├── fetch_pmc_v2.py       Fetch full text from PubMed Central (v2)
│   ├── extract_from_abstracts.py  Extract resistance data from abstracts
│   ├── extract_fulltext_data.py   Extract resistance data from full texts
│   ├── extract_comprehensive.py   Comprehensive extraction (final pipeline)
│   └── fill_missing_data.py  Impute/fill missing values post-extraction
└── analysis/                 Statistical analysis pipeline
    ├── config.py             Global configuration (insecticide maps, thresholds)
    ├── utils.py              Shared functions (transforms, meta-analysis, plots)
    ├── a01_data_cleaning.py  Data cleaning and standardisation ← run first
    ├── a02_descriptive_stats.py  Descriptive statistics and geographic maps
    ├── a03_meta_mortality.py Meta-analysis: bioassay mortality rates
    ├── a04_meta_rr.py        Meta-analysis: resistance ratios
    ├── a05_meta_kdr.py       Meta-analysis: kdr mutation frequencies
    ├── a06_meta_enzyme.py    Meta-analysis: metabolic enzyme activity
    ├── a07_cross_resistance.py  ★ Cross-resistance network analysis (core)
    ├── a08_subgroup_analysis.py  Subgroup analyses (region, year, method)
    ├── a09_meta_regression.py    Meta-regression (temporal trends, moderators)
    ├── a10_publication_bias.py   Funnel plots, Egger's test, trim-and-fill
    ├── a11_sensitivity.py    Sensitivity analyses (leave-one-out, outliers)
    └── run_all.py            Master pipeline (runs a01–a11 in order)
```

---

## Quick Start

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare your data

Place your extracted data file at `data/raw/extracted_data.csv`.
The expected column structure follows the Newcastle-Ottawa Scale–aligned
extraction template used in this study (see `analysis/config.py` for
recognised field names and insecticide aliases).

### 3. Run the pipeline

```bash
# Run everything in order
python analysis/run_all.py

# Or step by step (a01 must run first)
python analysis/a01_data_cleaning.py
python analysis/a02_descriptive_stats.py
python analysis/a03_meta_mortality.py
python analysis/a04_meta_rr.py
python analysis/a05_meta_kdr.py
python analysis/a06_meta_enzyme.py
python analysis/a07_cross_resistance.py   # core innovation
python analysis/a08_subgroup_analysis.py
python analysis/a09_meta_regression.py
python analysis/a10_publication_bias.py
python analysis/a11_sensitivity.py
```

Outputs (figures and summary tables) are written to `figures/` and
`outputs/` respectively (created automatically on first run).

---

## Statistical Methods

| Outcome | Transformation | Model |
|---------|---------------|-------|
| Bioassay mortality | Freeman-Tukey double arcsine | DerSimonian-Laird random-effects |
| Resistance ratio (RR) | Natural log | Inverse-variance random-effects |
| kdr allele frequency | Freeman-Tukey double arcsine | Random-effects |
| Enzyme activity | Hedges' *g* (SMD) | Random-effects |

**Cross-resistance network analysis (`a07`) uses three complementary approaches:**

1. Study-level Spearman correlations pooled via Fisher's *z* transformation
2. Population-level co-occurrence odds ratios (Mantel-Haenszel)
3. Mechanism–phenotype linkage via weighted meta-regression (kdr frequency as moderator)

Resistance status thresholds follow WHO guidelines: ≥ 98% mortality =
susceptible; 90–97% = possible resistance; < 90% = confirmed resistance.

---

## Key Configuration

Edit `analysis/config.py` to adjust:

- `MIN_STUDIES_FOR_META` — minimum studies to pool (default: 3)
- `MIN_SAMPLE_SIZE` — minimum bioassay *n* (default: 20)
- `INSECTICIDE_NAME_MAP` — normalise spelling variants
- `INSECTICIDE_CLASS_MAP` — assign insecticides to classes
- `WHO_REGION_MAP` — country → WHO region mapping
- `KDR_MUTATION_MAP` — normalise mutation name aliases
- `FIGURE_DPI` — output resolution (default: 300)

---

## Citation

If you use these scripts, please cite the associated manuscript (details
to be added upon publication).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
