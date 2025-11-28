# quant_pipeline/config.py
from pathlib import Path

# Project root is one level above the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data and results directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "Raw_Data"
PROCESSED_DIR = DATA_DIR / "Processed"

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

# Ensure they exist when the package is imported
for d in [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    METRICS_DIR,
    MODELS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
