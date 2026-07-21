"""Single source of truth: paths, panel window, thresholds, known quirks.
Every module and notebook reads its settings from here so nothing drifts."""
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parents[1]
DATA_RAW       = REPO_ROOT / "data"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
OUTPUTS        = REPO_ROOT / "outputs"

REGISTER_2021  = DATA_RAW / "housing-register-march-2021.xlsx"
REGISTER_2026  = DATA_RAW / "housing-register-march-2026.xlsx"
BONDS          = DATA_RAW / "detailed-monthly-tla-tenancy-v3.csv"
CONSENTS       = DATA_RAW / "building-consent.csv"
POPULATION     = DATA_RAW / "stat-data.csv"

N_TAS       = 66
PANEL_START = "2016Q1"   # Mar-2016
PANEL_END   = "2025Q3"   # Sep-2025 — provisional; confirm the bond-migration break in EDA

BOND_SENTINELS         = [-99, -1]        # Location Id ALL / NA — drop
BOND_MIGRATION_CUTOFF  = "2025-09-01"     # active bonds/closures unreliable after this
CONSENTS_DROP          = ["New Zealand", "Chatham Islands Territory",
                          "Area Outside Territorial Authority"]
POPULATION_DROP        = ["Chatham Islands Territory", "Chatham Islands District"]
REGISTER_JUNK_PREFIXES = ("Total", "Unknown", "Aggregated", "Note", "Please click")

MIN_ACTIVITY_BONDS = None   # small-TA threshold — decide during EDA
SEED = 42

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
