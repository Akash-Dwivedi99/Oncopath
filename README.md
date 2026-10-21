# OncoPath

OncoPath is a local research and learning dashboard for exploring the supplied oncology cohort. It combines a Streamlit interface with a FastAPI service for cohort summaries, patient lookup, graph analysis, and an exploratory Random Forest evaluation.

## Start OncoPath in VS Code

1. Open the folder that contains `backend`, `frontend`, `data`, and `.vscode`.
2. Select **Terminal → Run Task… → OncoPath Full Stack**.
3. On first use, VS Code creates a Python environment at `%LOCALAPPDATA%\OncoPath\venv` and installs the required packages. This short location avoids Windows path-length errors.
4. Keep the **OncoPath API** and **OncoPath Dashboard** terminals open.
5. Open **http://localhost:8501**. The API documentation is at **http://127.0.0.1:8000/docs**.

You can also double-click **Launch OncoPath.bat** in File Explorer. It starts both services in separate terminal windows.

> Do not use VS Code’s **Run Python File** button for `frontend/app.py`. That starts the file outside Streamlit. Use **OncoPath Full Stack** or the launcher instead.

## If the dashboard says the backend is not connected

1. In VS Code, open **Terminal → Run Task… → OncoPath API**.
2. Check the API terminal. It should show `Uvicorn running on http://127.0.0.1:8000`.
3. If it shows an error, resolve that error first, then refresh the dashboard.

Both services must stay running while the dashboard is open. If the task list is missing, reopen the project root folder that contains `.vscode/tasks.json`.

## Using the dashboard

- **Cohort overview** shows record counts, stage composition, outcome-label rates, and clinical or biomarker distributions.
- **Graph & DSA** shows a feature co-occurrence network and supports BFS, DFS, weighted shortest paths, centrality, communities, and sparse adjacency summaries.
- **Model evaluation** trains a Random Forest on a stratified training split and reports metrics on a held-out split.
- **Patient lookup** searches the full dataset by exact or partial patient ID, regardless of the cohort filters.

Cancer and stage filters in the sidebar update cohort summaries and the graph. They do not restrict patient-ID lookup.

## Project structure

```text
OncoPath/
├── backend/app/       FastAPI routes, data loading, graph algorithms, model pipeline
├── frontend/app.py    Streamlit dashboard
├── data/              Supplied CSV
├── .vscode/tasks.json VS Code setup and run tasks
├── requirements.txt   Python dependencies
├── Launch OncoPath.bat
├── run_backend.bat
├── run_frontend.bat
└── README.md
```

## Data and interpretation

The supplied CSV contains 3,000 records and clinical and biomarker fields. Its provenance and clinical validation status were not supplied. The dashboard is an educational research prototype, not a clinical tool.

The graph represents categorical feature values that co-occur in patient rows. The CSV has no gene-expression matrix or gene/protein interaction data, so the graph does not represent biological pathways. No TCGA or GEO connection is configured.

The model target is `death_within_5yr`. Patient identifiers, duplicate survival status, follow-up duration, and post-baseline treatment or response fields are excluded from predictors. Metrics are exploratory and must not be used for diagnosis, prognosis, or treatment decisions.
