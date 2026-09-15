from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "database.sqlite"
MODELS_DIR = PROJECT_ROOT / "models"
INFORME_DIR = PROJECT_ROOT / "informe"

SEED = 42

FTR_CLASSES = [0, 1, 2]

BREAKEVEN_ODDS = 1.85
BREAKEVEN = 1 / BREAKEVEN_ODDS

BET_BOOKMAKERS = ["B365", "BW", "WH", "VC", "LB", "IW", "SJ", "GB", "BS", "PS"]
ODDS_SUFFIXES = ["H", "D", "A"]
ODDS_B365 = ["B365H", "B365D", "B365A"]
NULL_THRESHOLD = 0.30
STREAK_WINDOWS = (5, 10, 15)