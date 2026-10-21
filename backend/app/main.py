from functools import lru_cache

import networkx as nx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io

from .config import TARGET
from .data import cohort_frame, load_data, records_json
from .graph import build_graph, communities, graph_payload, shortest_path, sparse_adjacency, traverse
from .model import evaluate_baseline

app = FastAPI(title="OncoPath API", version="1.0.0",
              description="Cohort exploration, feature co-occurrence graph, DSA, and baseline model APIs.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health():
    try:
        frame = load_data()
        return {"status": "ok", "dataset_rows": len(frame), "target": TARGET}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/metadata")
def metadata():
    frame = load_data()
    return {
        "records": len(frame), "columns": list(frame.columns),
        "cancer_types": sorted(frame["cancer_type"].dropna().astype(str).unique()),
        "stages": sorted(frame["tumor_stage"].dropna().astype(str).unique()),
        "outcome_counts": {str(key): int(value) for key, value in frame[TARGET].value_counts(dropna=False).items()},
        "target": TARGET,
    }


@app.get("/api/cohort")
def cohort(cancer: str | None = None, stage: str | None = None):
    frame = cohort_frame(cancer, stage)
    usable = frame[TARGET].dropna()
    cancer_counts = frame.groupby("cancer_type").size().rename("patients").reset_index()
    stage_counts = frame.groupby(["cancer_type", "tumor_stage"]).size().rename("patients").reset_index()
    outcome_rates = frame.groupby("cancer_type")[TARGET].mean().mul(100).rename("death_percent").reset_index()
    return {
        "records": len(frame), "cancer_types": int(frame["cancer_type"].nunique()),
        "death_percent": float(usable.mean() * 100) if len(usable) else None,
        "target_counts": {str(key): int(value) for key, value in usable.value_counts().items()},
        "cancer_counts": records_json(cancer_counts),
        "stage_counts": records_json(stage_counts),
        "outcome_rates": records_json(outcome_rates),
        "numeric_fields": frame.select_dtypes(include="number").columns.tolist(),
    }


@app.get("/api/patients")
def patients(query: str | None = None, cancer: str | None = None, stage: str | None = None,
             limit: int = Query(default=500, ge=1, le=3000), offset: int = Query(default=0, ge=0)):
    # A patient ID lookup intentionally searches the full file, independent of cohort filters.
    frame = load_data()
    if query and query.strip():
        ids = frame["patient_id"].astype(str).str.strip()
        text = query.strip()
        exact = ids.str.casefold().eq(text.casefold())
        frame = frame[exact] if exact.any() else frame[ids.str.contains(text, case=False, na=False, regex=False)]
    else:
        frame = cohort_frame(cancer, stage)
    count = len(frame)
    return {"total": count, "offset": offset, "limit": limit,
            "items": records_json(frame.iloc[offset:offset + limit])}


@app.get("/api/patients/export.csv")
def export_patients(cancer: str | None = None, stage: str | None = None):
    frame = cohort_frame(cancer, stage)
    stream = io.StringIO()
    frame.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=oncopath_filtered_cohort.csv"})


@app.get("/api/graph")
def graph(cancer: str | None = None, stage: str | None = None):
    return graph_payload(build_graph(cohort_frame(cancer, stage)))


@app.get("/api/graph/adjacency")
def graph_adjacency(cancer: str | None = None, stage: str | None = None):
    return sparse_adjacency(build_graph(cohort_frame(cancer, stage)))


@lru_cache(maxsize=12)
def cached_communities(cancer: str | None, stage: str | None):
    return communities(build_graph(cohort_frame(cancer, stage)))


@app.get("/api/graph/communities")
def graph_communities(cancer: str | None = None, stage: str | None = None):
    return {"communities": cached_communities(cancer, stage)}


@app.get("/api/graph/traverse")
def graph_traverse(start: str, method: str = "bfs", cancer: str | None = None, stage: str | None = None):
    graph = build_graph(cohort_frame(cancer, stage))
    try:
        order = traverse(graph, start, method)
        return {"method": method.lower(), "start": start, "visited_count": len(order), "order": order}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/graph/path")
def graph_path(source: str, target: str, cancer: str | None = None, stage: str | None = None):
    graph = build_graph(cohort_frame(cancer, stage))
    try:
        return shortest_path(graph, source, target)
    except (KeyError, nx.NetworkXNoPath) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@lru_cache(maxsize=12)
def cached_evaluation(cancer: str | None, stage: str | None):
    return evaluate_baseline(cohort_frame(cancer, stage))


@app.get("/api/model/evaluate")
def model_evaluation(cancer: str | None = None, stage: str | None = None):
    try:
        return cached_evaluation(cancer, stage)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
