from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
DATA_PATH = Path(os.getenv("ONCOPATH_DATA", PROJECT_ROOT / "data" / "oncolody_dataset_final.csv"))
TARGET = "death_within_5yr"
LEAKAGE_COLUMNS = {"patient_id", TARGET, "survival_5yr_status", "follow_up_months"}
POST_BASELINE_COLUMNS = {
    "treatment_received", "surgical_margin_clear", "days_diagnosis_to_treatment",
    "treatment_completed", "resection_status", "chemo_cycles_completed",
    "dose_intensity_percent", "response_first_scan", "recurrence_score",
}
