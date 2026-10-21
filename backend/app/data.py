from functools import lru_cache

import pandas as pd

from .config import DATA_PATH, TARGET


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    required = {"patient_id", "cancer_type", "tumor_stage", TARGET}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    return frame


def cohort_frame(cancer: str | None = None, stage: str | None = None) -> pd.DataFrame:
    frame = load_data()
    if cancer and cancer != "All":
        frame = frame[frame["cancer_type"].astype(str) == cancer]
    if stage and stage != "All":
        frame = frame[frame["tumor_stage"].astype(str) == stage]
    return frame.copy()


def records_json(frame: pd.DataFrame) -> list[dict]:
    # Round-trip through JSON so pandas NaN values become JSON nulls.
    import json
    return json.loads(frame.to_json(orient="records", date_format="iso"))
