# OncoPath full-stack project

OncoPath is a local research and learning prototype with a Streamlit frontend and FastAPI backend. The backend serves cohort data, constructs a feature co-occurrence graph, runs graph algorithms, and evaluates an exploratory Random Forest baseline.

## Run on Windows

1. Extract this project folder and confirm `data/oncolody_dataset_final.csv` is present.
2. Double-click **Launch OncoPath.bat**. On the first run it creates `.venv` and installs the dependencies from `requirements.txt`.
3. Keep the **OncoPath API** and **OncoPath Dashboard** terminal windows open. The dashboard opens at `http://localhost:8501`.

If you prefer to start the services manually from this folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, from the same project folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend\app.py
```

The API health check is `http://127.0.0.1:8000/health`; interactive API docs are at `http://127.0.0.1:8000/docs`.

## Project layout

```text
oncopath_fullstack/
├── backend/app/
│   ├── main.py       FastAPI routes
│   ├── data.py       Dataset loading, validation, filtering
│   ├── graph.py      Co-occurrence graph, traversal, paths, centrality, communities, CSR
│   ├── model.py      Preprocessing and held-out Random Forest evaluation
│   └── config.py     Dataset location and target/leakage rules
├── frontend/app.py   Streamlit interface that calls the backend API
├── data/             Supplied cohort CSV
├── requirements.txt
├── Launch OncoPath.bat
└── README.md
```

## API routes

| Route | Purpose |
| --- | --- |
| `GET /health` | Backend and dataset readiness |
| `GET /api/metadata` | Dataset fields, labels, cancer types, stages |
| `GET /api/cohort` | Filtered cohort summaries and outcome breakdowns |
| `GET /api/patients` | Paginated records; `query` searches IDs across the full dataset |
| `GET /api/patients/export.csv` | Export filtered cohort records |
| `GET /api/graph` | Nodes, weighted edges, graph summary |
| `GET /api/graph/traverse` | BFS/DFS from a selected node |
| `GET /api/graph/path` | Weighted shortest path between nodes |
| `GET /api/graph/communities` | Greedy modularity communities |
| `GET /api/graph/adjacency` | CSR sparse adjacency storage summary |
| `GET /api/model/evaluate` | Stratified holdout metrics and feature importance |

## Pipeline and interpretation

The CSV supplied for the project has 3,000 records across Breast, Lung, and Colorectal labels. Nodes represent observed categorical feature values; links count co-occurrence in the same patient row. BFS/DFS, weighted shortest path, PageRank, degree centrality, community detection, and CSR adjacency are available through the backend and shown in the dashboard.

The target is `death_within_5yr`. The model withholds patient identifiers, the duplicate survival-status field, follow-up duration, and post-baseline treatment/response fields. It reports metrics from a stratified 75/25 train/test split. Results are exploratory; the supplied file's provenance and clinical validation are undocumented. The dashboard is not for diagnosis, prognosis, or treatment decisions.

The CSV has no gene-expression matrix or gene/protein interaction source. This build therefore does not claim to represent molecular pathways and does not connect to TCGA or GEO. Its graph is a cohort feature association graph.
