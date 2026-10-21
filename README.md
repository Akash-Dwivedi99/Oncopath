# OncoPath interactive dashboard

This Streamlit prototype explores the supplied cohort CSV, builds a feature-value co-occurrence graph, and trains a baseline Random Forest for the file's `death_within_5yr` target.

## Run locally

```powershell
cd oncopath_dashboard
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The CSV is included beside `app.py`, so no data download is needed.

## What is in the dashboard

- Cohort filters, cancer/stage summaries, outcome rates, and clinical/biomarker distributions.
- A NetworkX graph of categorical feature values that co-occur in patient rows, plus node degree ranking.
- An on-demand Random Forest using a stratified held-out split, accuracy, ROC AUC, confusion matrix, and feature importance.
- Filtered patient table and CSV download.

## Dataset and interpretation notes

The provided file has 3,000 rows, 46 columns, and three cancer labels (Breast, Lung, Colorectal). Its patient IDs use `SYN-` labels. The source, cohort accession, and whether the records are synthetic or clinically validated were not included with the file, so this dashboard reports the data as supplied and does not present performance as clinical evidence.

The CSV does not contain gene-expression profiles or gene/protein interactions. Its graph is therefore a feature co-occurrence graph, not a biological pathway or gene interaction network. The application does not connect to TCGA or GEO.

The model target is `death_within_5yr`. Identifier, duplicate survival status, follow-up duration, treatment, and response fields are excluded from predictors to reduce direct target leakage. Because the file's provenance and measurement timing are not documented, the metrics are exploratory and are not suitable for patient care or treatment decisions.
