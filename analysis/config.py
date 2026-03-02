"""
Global configuration for the insecticide resistance meta-analysis.
Defines constants, mappings, paths, and analysis parameters.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "03_data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "03_data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "05_figures"
TABLES_DIR = PROJECT_ROOT / "06_tables"
QUALITY_DIR = PROJECT_ROOT / "08_quality"

for d in [DATA_RAW, DATA_PROCESSED, FIGURES_DIR, TABLES_DIR, QUALITY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Analysis parameters
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
MIN_STUDIES_FOR_META = 3          # minimum studies to run a meta-analysis
MIN_SAMPLE_SIZE = 20              # minimum n per bioassay test
CONTINUITY_CORRECTION = 0.5       # for zero-event / all-event studies
CONTROL_MORTALITY_MAX = 0.20      # discard if > 20 %
CONTROL_MORTALITY_ABBOTT = 0.05   # apply Abbott's if 5-20 %

# WHO resistance classification thresholds (mortality %)
WHO_SUSCEPTIBLE = 98
WHO_POSSIBLE_RESISTANCE = 90      # 90-97 % → "possible resistance"
# < 90 % → "confirmed resistance"

# ──────────────────────────────────────────────────────────────────────
# Insecticide name standardisation
# ──────────────────────────────────────────────────────────────────────
INSECTICIDE_NAME_MAP: dict[str, str] = {
    # Pyrethroids
    "lambda cyhalothrin": "lambda-cyhalothrin",
    "lambdacyhalothrin": "lambda-cyhalothrin",
    "λ-cyhalothrin": "lambda-cyhalothrin",
    "alpha cypermethrin": "alpha-cypermethrin",
    "alphacypermethrin": "alpha-cypermethrin",
    "α-cypermethrin": "alpha-cypermethrin",
    "beta cyfluthrin": "beta-cyfluthrin",
    "β-cyfluthrin": "beta-cyfluthrin",
    "d-allethrin": "d-allethrin",
    "d allethrin": "d-allethrin",
    "d.allethrin": "d-allethrin",
    "s-bioallethrin": "s-bioallethrin",
    # Organophosphates
    "pirimiphos methyl": "pirimiphos-methyl",
    "pirimiphos-methyl": "pirimiphos-methyl",
    "temefos": "temephos",
    # Carbamates
    # Organochlorines
    "ddt": "DDT",
    # Biologicals
    "bacillus thuringiensis israelensis": "Bti",
    "bacillus thuringiensis var. israelensis": "Bti",
    "b.t.i.": "Bti",
    "b.t.i": "Bti",
}

# ──────────────────────────────────────────────────────────────────────
# Insecticide → class mapping
# ──────────────────────────────────────────────────────────────────────
INSECTICIDE_CLASS_MAP: dict[str, str] = {
    # Pyrethroids (Type I – no α-cyano)
    "permethrin": "Pyrethroid",
    "etofenprox": "Pyrethroid",
    "bifenthrin": "Pyrethroid",
    "d-allethrin": "Pyrethroid",
    "s-bioallethrin": "Pyrethroid",
    "transfluthrin": "Pyrethroid",
    "metofluthrin": "Pyrethroid",
    "prallethrin": "Pyrethroid",
    "esbiothrin": "Pyrethroid",
    "resmethrin": "Pyrethroid",
    "tetramethrin": "Pyrethroid",
    # Pyrethroids (Type II – with α-cyano)
    "deltamethrin": "Pyrethroid",
    "cypermethrin": "Pyrethroid",
    "alpha-cypermethrin": "Pyrethroid",
    "lambda-cyhalothrin": "Pyrethroid",
    "cyfluthrin": "Pyrethroid",
    "beta-cyfluthrin": "Pyrethroid",
    "fenvalerate": "Pyrethroid",
    "esfenvalerate": "Pyrethroid",
    # Organophosphates
    "malathion": "Organophosphate",
    "temephos": "Organophosphate",
    "fenitrothion": "Organophosphate",
    "pirimiphos-methyl": "Organophosphate",
    "chlorpyrifos": "Organophosphate",
    "fenthion": "Organophosphate",
    "naled": "Organophosphate",
    "dichlorvos": "Organophosphate",
    # Carbamates
    "propoxur": "Carbamate",
    "bendiocarb": "Carbamate",
    "carbaryl": "Carbamate",
    # Organochlorines
    "DDT": "Organochlorine",
    "dieldrin": "Organochlorine",
    "lindane": "Organochlorine",
    "HCH": "Organochlorine",
    # Neonicotinoids
    "imidacloprid": "Neonicotinoid",
    "clothianidin": "Neonicotinoid",
    "thiamethoxam": "Neonicotinoid",
    "acetamiprid": "Neonicotinoid",
    "dinotefuran": "Neonicotinoid",
    # IGRs
    "pyriproxyfen": "IGR",
    "methoprene": "IGR",
    "diflubenzuron": "IGR",
    "novaluron": "IGR",
    # Biologicals
    "Bti": "Biological",
    "spinosad": "Biological",
    # Other
    "chlorfenapyr": "Other",
}

# ──────────────────────────────────────────────────────────────────────
# WHO region mapping  (country → WHO region)
# ──────────────────────────────────────────────────────────────────────
WHO_REGION_MAP: dict[str, str] = {
    # South-East Asia
    "India": "SEARO", "Indonesia": "SEARO", "Thailand": "SEARO",
    "Myanmar": "SEARO", "Sri Lanka": "SEARO", "Bangladesh": "SEARO",
    "Nepal": "SEARO", "Timor-Leste": "SEARO", "Maldives": "SEARO",
    # Western Pacific
    "China": "WPRO", "Malaysia": "WPRO", "Vietnam": "WPRO",
    "Philippines": "WPRO", "Cambodia": "WPRO", "Laos": "WPRO",
    "Singapore": "WPRO", "Japan": "WPRO", "South Korea": "WPRO",
    "Australia": "WPRO", "Papua New Guinea": "WPRO", "Fiji": "WPRO",
    "Taiwan": "WPRO",
    # Americas
    "Brazil": "AMRO", "Colombia": "AMRO", "Mexico": "AMRO",
    "Argentina": "AMRO", "Venezuela": "AMRO", "Peru": "AMRO",
    "Ecuador": "AMRO", "Cuba": "AMRO", "USA": "AMRO",
    "United States": "AMRO", "Costa Rica": "AMRO", "Panama": "AMRO",
    "Guatemala": "AMRO", "Honduras": "AMRO", "Puerto Rico": "AMRO",
    "Trinidad and Tobago": "AMRO", "Martinique": "AMRO",
    "Guadeloupe": "AMRO", "French Guiana": "AMRO",
    # African
    "Nigeria": "AFRO", "Cameroon": "AFRO", "Kenya": "AFRO",
    "Tanzania": "AFRO", "Ghana": "AFRO", "Senegal": "AFRO",
    "Burkina Faso": "AFRO", "Benin": "AFRO", "Gabon": "AFRO",
    "Congo": "AFRO", "DR Congo": "AFRO", "Ethiopia": "AFRO",
    "Mozambique": "AFRO", "Madagascar": "AFRO", "South Africa": "AFRO",
    "Uganda": "AFRO", "Côte d'Ivoire": "AFRO", "Mali": "AFRO",
    "Reunion": "AFRO", "Mayotte": "AFRO",
    # European
    "Italy": "EURO", "France": "EURO", "Spain": "EURO",
    "Germany": "EURO", "Greece": "EURO", "Turkey": "EURO",
    "Albania": "EURO", "Croatia": "EURO", "Montenegro": "EURO",
    "Serbia": "EURO", "Romania": "EURO", "Switzerland": "EURO",
    "Portugal": "EURO", "Belgium": "EURO", "Netherlands": "EURO",
    # Eastern Mediterranean
    "Pakistan": "EMRO", "Iran": "EMRO", "Saudi Arabia": "EMRO",
    "Egypt": "EMRO", "Sudan": "EMRO", "Yemen": "EMRO",
    "Oman": "EMRO", "Iraq": "EMRO", "Afghanistan": "EMRO",
    "Somalia": "EMRO",
}

CONTINENT_MAP: dict[str, str] = {
    "SEARO": "Asia", "WPRO": "Asia",
    "AMRO": "Americas",
    "AFRO": "Africa",
    "EURO": "Europe",
    "EMRO": "Eastern Mediterranean",
}

# ──────────────────────────────────────────────────────────────────────
# kdr mutation standardisation  (alias → canonical name)
# ──────────────────────────────────────────────────────────────────────
KDR_MUTATION_MAP: dict[str, str] = {
    "F1534C": "F1534C", "1534C": "F1534C", "F1534S": "F1534S",
    "1534S": "F1534S", "F1534L": "F1534L", "1534L": "F1534L",
    "V1016G": "V1016G", "1016G": "V1016G", "V1016I": "V1016I",
    "1016I": "V1016I",
    "S989P": "S989P", "989P": "S989P",
    "I1011M": "I1011M", "I1011V": "I1011V",
    "V410L": "V410L", "410L": "V410L",
    "L982W": "L982W",
    "D1763Y": "D1763Y",
}

# ──────────────────────────────────────────────────────────────────────
# Colour palette (colour-blind friendly)
# ──────────────────────────────────────────────────────────────────────
INSECTICIDE_CLASS_COLORS: dict[str, str] = {
    "Pyrethroid": "#E69F00",       # orange
    "Organophosphate": "#56B4E9",  # sky blue
    "Carbamate": "#009E73",        # green
    "Organochlorine": "#F0E442",   # yellow
    "Neonicotinoid": "#0072B2",    # blue
    "IGR": "#D55E00",              # vermillion
    "Biological": "#CC79A7",       # pink
    "Other": "#999999",            # grey
}

WHO_REGION_COLORS: dict[str, str] = {
    "SEARO": "#E69F00",
    "WPRO": "#56B4E9",
    "AMRO": "#009E73",
    "AFRO": "#D55E00",
    "EURO": "#0072B2",
    "EMRO": "#CC79A7",
}

# ──────────────────────────────────────────────────────────────────────
# Plot settings
# ──────────────────────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FONT_SIZE = 10
FONT_FAMILY = "Arial"

import matplotlib as mpl

mpl.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": FONT_FAMILY,
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
