import os

# Base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "student_data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure required directories exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Datasets
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
MACRO_FILE = os.path.join(DATA_DIR, "macro_indicators.csv")
SKU_MASTER_FILE = os.path.join(DATA_DIR, "sku_master.csv")
EVENTS_FILE = os.path.join(DATA_DIR, "events_calendar.csv")
STOCKOUTS_FILE = os.path.join(DATA_DIR, "inventory_stockouts.csv")
SAMPLE_SUBMISSION_FILE = os.path.join(DATA_DIR, "sample_submission.csv")

# Aliases for compatibility
TEST_CSV = TEST_FILE
TRAIN_CSV = TRAIN_FILE
SAMPLE_SUBMISSION_CSV = SAMPLE_SUBMISSION_FILE
MACRO_CSV = MACRO_FILE
SKU_MASTER_CSV = SKU_MASTER_FILE
EVENTS_CSV = EVENTS_FILE
STOCKOUTS_CSV = STOCKOUTS_FILE

# Cleaned outputs
CLEANED_TRAIN_FILE = os.path.join(OUTPUTS_DIR, "cleaned_train.csv")
CLEANED_TEST_FILE = os.path.join(OUTPUTS_DIR, "cleaned_test.csv")
DATA_QUALITY_REPORT_FILE = os.path.join(OUTPUTS_DIR, "data_quality_report.xlsx")
ANOMALY_REGISTER_FILE = os.path.join(OUTPUTS_DIR, "anomaly_register.csv")
SUBMISSION_FILE = os.path.join(OUTPUTS_DIR, "submission.csv")
SUBMISSION_CSV = SUBMISSION_FILE
BACKTEST_RESULTS_FILE = os.path.join(OUTPUTS_DIR, "backtest_results.csv")
MODEL_COMPARISON_FILE = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
SHAP_SUMMARY_FILE = os.path.join(OUTPUTS_DIR, "shap_summary.json")
INVENTORY_PLAN_FILE = os.path.join(OUTPUTS_DIR, "inventory_plan.csv")

# Project Constants
HISTORICAL_START = "2019-01"
HISTORICAL_END = "2026-10"
FORECAST_START = "2026-11"
FORECAST_END = "2028-05"
FORECAST_HORIZON_MONTHS = 19
RANDOM_STATE = 42

VALID_YEAR_MIN = 2019
VALID_YEAR_MAX = 2028

SUBMISSION_COLUMNS = ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]

# Hierarchy & Categorical Domains
REGIONS = [
    "North America",
    "Europe",
    "Asia Pacific",
    "India",
    "Latin America",
    "Middle East & Africa"
]
CANONICAL_REGIONS = REGIONS

PRODUCT_FAMILIES = [
    "Brake & Safety Systems",
    "Chassis & Structural",
    "Electrical Assemblies",
    "Hydraulic Systems",
    "Powertrain Components",
    "Precision Machined Parts",
    "Thermal Management"
]

SALES_CHANNELS = [
    "Direct OEM",
    "Tier-1 Distributor",
    "Aftermarket",
    "Export"
]
CANONICAL_CHANNELS = SALES_CHANNELS

CANONICAL_CUSTOMERS = [
    "Solaris Automotive",
    "Apex Tier One",
    "Vantara Auto",
    "Nexon Motor Group",
    "Helios Auto Parts",
    "Crestline Manufacturing",
    "Aura Dynamics",
    "Zephyr Industrial",
    "Kromatech Corp",
    "Titan Heavy Machinery",
    "Solstice Motors",
    "Pinnacle Power"
]

REGION_ALIASES = {
    "latam": "Latin America",
    "apac": "Asia Pacific",
    "na": "North America",
    "eu": "Europe",
    "mea": "Middle East & Africa",
    "LATAM": "Latin America",
    "APAC": "Asia Pacific",
    "NA": "North America",
    "EU": "Europe",
    "MEA": "Middle East & Africa"
}

CHANNEL_ALIASES = {
    "oem direct": "Direct OEM",
    "distributor": "Tier-1 Distributor",
    "tier 1 distributor": "Tier-1 Distributor",
    "export / other": "Export",
    "OEM Direct": "Direct OEM",
    "Distributor": "Tier-1 Distributor",
    "Tier 1 Distributor": "Tier-1 Distributor"
}

# Quantiles for uncertainty estimation
QUANTILES = [0.10, 0.50, 0.90]
