import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
API_URL = os.getenv("ONCOPATH_API_URL", "http://127.0.0.1:8000").rstrip("/")
if get_script_run_ctx(suppress_warning=True) is None:
    raise SystemExit(
        "OncoPath is a Streamlit app. Start it with the VS Code task "
        "'OncoPath Full Stack' or double-click 'Launch OncoPath.bat'."
    )
st.set_page_config(page_title="OncoPath | Cohort Intelligence", page_icon="🧬", layout="wide")
st.markdown("""
<style>
:root { --ink:#15243a; --muted:#65758b; --teal:#087f8c; --line:#dce5ee; --paper:#f5f8fb; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { padding-top:1.25rem; padding-bottom:2rem; max-width:1480px; }
[data-testid="stSidebar"] { background:#edf3f7; border-right:1px solid var(--line); }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:16px 18px; border-radius:13px; box-shadow:0 2px 7px #18314b0a; }
[data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
[data-testid="stMetricValue"] { color:var(--ink); font-weight:720; }
div[data-testid="stTabs"] button { font-weight:650; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--teal); }
h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
.oncopath-hero { display:flex; justify-content:space-between; align-items:center; gap:20px; margin:0 0 1rem; padding:20px 24px; background:linear-gradient(115deg,#10283d,#164758); border-radius:16px; color:#fff; }
.oncopath-hero h1 { color:#fff; font-size:2rem; margin:0 0 5px; }
.oncopath-hero p { color:#c9dbe4; margin:0; }
.hero-tag { padding:7px 12px; border:1px solid #ffffff40; border-radius:999px; color:#d8f2f0; font-size:.78rem; }
</style>
<div class="oncopath-hero"><div><h1>OncoPath</h1><p>Clinical cohort intelligence · graph exploration · baseline risk modeling</p></div><span class="hero-tag">RESEARCH PROTOTYPE</span></div>
""", unsafe_allow_html=True)


def api_get(path, params=None, timeout=45):
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error("The OncoPath API is not responding. Start the **OncoPath API** task in VS Code, or double-click **Launch OncoPath.bat** in the project folder.")
        st.caption(f"API address: {API_URL} · Details: {exc}")
        st.stop()
        raise SystemExit("OncoPath API is not responding.")


@st.cache_data(ttl=60)
def get_metadata():
    response = requests.get(f"{API_URL}/api/metadata", timeout=15)
    response.raise_for_status()
    return response.json()


try:
    metadata = get_metadata()
except requests.RequestException as exc:
    st.error("The dashboard cannot reach the backend yet.")
    st.markdown("**To start it in VS Code:** choose **Terminal → Run Task → OncoPath API**. Wait for ‘Uvicorn running on http://127.0.0.1:8000’, then refresh this page.")
    st.markdown("Or double-click **Launch OncoPath.bat** in the project folder to start both parts.")
    st.caption(f"API address: {API_URL} · Details: {exc}")
    st.stop()
    raise SystemExit("OncoPath API is not responding.")

with st.sidebar:
    st.markdown("### Explore a cohort")
    st.success("Backend connected", icon="✅")
    cancer = st.selectbox("Cancer type", ["All", *metadata["cancer_types"]])
    stage = st.selectbox("Tumor stage", ["All", *metadata["stages"]])
    params = {"cancer": cancer, "stage": stage}
    st.divider()
    st.caption(f"Dataset size: {metadata['records']:,} records")
    st.caption(f"Outcome label: `{metadata['target']}`")
    st.caption("These filters update cohort summaries and the graph. Patient ID lookup searches the full dataset.")

with st.expander("How to use this dashboard", expanded=False):
    st.markdown("""
    1. Choose a cancer type or stage in the left panel.
    2. **Cohort overview** summarizes the selected records.
    3. **Graph & DSA** explores feature co-occurrence and graph algorithms.
    4. **Model evaluation** reports a held-out Random Forest baseline.
    5. **Patient lookup** searches by ID across all records.

    The outcome chart shows labels in this dataset; it is not an individual patient's medical risk.
    """)

cohort = api_get("/api/cohort", params)
metric_cols = st.columns(4)
metric_cols[0].metric("Patients", f"{cohort['records']:,}")
metric_cols[1].metric("Cancer types", cohort["cancer_types"])
metric_cols[2].metric("Records labeled death within 5 years", f"{cohort['death_percent']:.1f}%" if cohort["death_percent"] is not None else "—", help="Share of records with the supplied death_within_5yr label. This is not a personal risk prediction.")
metric_cols[3].metric("Clinical fields", len(metadata["columns"]) - 1)

overview_tab, graph_tab, model_tab, patient_tab = st.tabs(["Cohort overview", "Graph & DSA", "Model evaluation", "Patient lookup"])

with overview_tab:
    st.caption("Use the left-side filters to focus the summaries on a cancer type or stage.")
    left, right = st.columns([1.1, .9])
    with left:
        st.subheader("Cohort composition")
        stage_data = pd.DataFrame(cohort["stage_counts"])
        if not stage_data.empty:
            fig = px.bar(stage_data, x="cancer_type", y="patients", color="tumor_stage", barmode="stack",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         labels={"cancer_type":"Cancer type", "patients":"Patients", "tumor_stage":"Stage"})
            fig.update_layout(height=390, margin=dict(l=0, r=0, t=15, b=0), legend_title="Stage")
            st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Outcome label by cancer")
        outcome = pd.DataFrame(cohort["outcome_rates"])
        if not outcome.empty:
            fig = px.bar(outcome, x="cancer_type", y="death_percent", color="cancer_type", text_auto=".1f",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         labels={"cancer_type":"Cancer type", "death_percent":"Death label (%)"})
            fig.update_layout(height=390, showlegend=False, margin=dict(l=0, r=0, t=15, b=0))
            st.plotly_chart(fig, width="stretch")
    st.subheader("Clinical and biomarker distributions")
    choices = [field for field in ["age", "bmi", "tumor_size_cm", "ki67_percent", "crp_mg_l", "CEA", "CA 15-3", "CYFRA 21-1", "ca_19_9_u_ml"] if field in metadata["columns"]]
    if choices:
        measure = st.selectbox("Select a measure", choices)
        records = api_get("/api/patients", {**params, "limit": 3000})["items"]
        measure_df = pd.DataFrame(records)
        if not measure_df.empty:
            fig = px.box(measure_df, x="cancer_type", y=measure, color="cancer_type", points="outliers",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(height=360, showlegend=False, margin=dict(l=0, r=0, t=15, b=0))
            st.plotly_chart(fig, width="stretch")

with graph_tab:
    st.subheader("Feature-value co-occurrence network")
    st.write("Each node is a clinical feature value. A link means two values appeared in the same patient record; a stronger link means they co-occurred more often. This describes the supplied cohort, not biological gene interactions.")
    graph_data = api_get("/api/graph", params)
    summary = graph_data["summary"]
    g1, g2, g3 = st.columns(3)
    g1.metric("Nodes", summary["nodes"]); g2.metric("Links", summary["edges"]); g3.metric("Components", summary["components"])
    nodes = graph_data["nodes"]
    if nodes:
        import networkx as nx
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node["id"], **node)
        for edge in graph_data["edges"]:
            graph.add_edge(edge["source"], edge["target"], **edge)
        positions = nx.spring_layout(graph, seed=21, weight="weight", k=.8)
        edge_x, edge_y = [], []
        for left, right in graph.edges:
            edge_x += [positions[left][0], positions[right][0], None]
            edge_y += [positions[left][1], positions[right][1], None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#afbdca"), hoverinfo="none")
        pagerank = {row["node"]: row["pagerank"] for row in summary["top_centrality"]}
        node_trace = go.Scatter(x=[positions[node["id"]][0] for node in nodes], y=[positions[node["id"]][1] for node in nodes],
            mode="markers+text", text=[node["id"] for node in nodes], textposition="top center", textfont=dict(size=9),
            marker=dict(size=[10 + 3 * (node["count"] ** .5) for node in nodes],
                color=[pagerank.get(node["id"], 0) for node in nodes], colorscale="Teal", showscale=True,
                colorbar=dict(title="PageRank"), line=dict(width=1, color="white")),
            customdata=[[node["count"], graph.degree(node["id"])] for node in nodes],
            hovertemplate="%{text}<br>Rows: %{customdata[0]}<br>Connections: %{customdata[1]}<extra></extra>")
        fig = go.Figure([edge_trace, node_trace])
        fig.update_layout(height=600, margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
        with st.expander("Run graph algorithms"):
            node_ids = [item["id"] for item in nodes]
            source = st.selectbox("Starting node", node_ids)
            method = st.radio("Traversal", ["bfs", "dfs"], horizontal=True)
            if st.button("Run traversal"):
                result = api_get("/api/graph/traverse", {**params, "start": source, "method": method})
                st.write(f"Visited {result['visited_count']} connected nodes.")
                st.code(" → ".join(result["order"][:30]))
            target = st.selectbox("Path destination", node_ids, index=min(1, len(node_ids)-1))
            if st.button("Find weighted shortest path"):
                response = requests.get(f"{API_URL}/api/graph/path", params={**params, "source": source, "target": target}, timeout=45)
                if response.ok:
                    path_result = response.json()
                    st.success(f"{path_result['hops']} hops · weighted cost {path_result['cost']:.4f}")
                    st.code(" → ".join(path_result["path"]))
                else:
                    detail = response.json().get("detail", "No route exists between these nodes.")
                    st.warning(f"No path available for these nodes: {detail}")
        with st.expander("Centrality and communities"):
            st.markdown("**PageRank and degree centrality**")
            st.dataframe(pd.DataFrame(summary["top_centrality"]), hide_index=True, width="stretch")
            sparse = api_get("/api/graph/adjacency", params)
            st.caption(f"Sparse adjacency storage · {sparse['shape'][0]} × {sparse['shape'][1]} CSR · {sparse['nonzero_entries']:,} nonzero entries · {sparse['sparse_bytes']:,} bytes vs {sparse['dense_float32_bytes']:,} dense bytes")
            if st.button("Detect feature communities"):
                result = api_get("/api/graph/communities", params)
                st.dataframe(pd.DataFrame([{k:v for k,v in item.items() if k != "nodes"} for item in result["communities"]]), hide_index=True, width="stretch")

with model_tab:
    st.subheader("Random Forest baseline")
    st.write("Select **Evaluate model** to train on one part of the cohort and score on a held-out part. Identifiers, duplicate outcome fields, follow-up duration, and post-baseline treatment or response fields are excluded from predictors.")
    st.info("The supplied dataset's provenance and clinical validation status are undocumented. Results are for educational exploration and are not medical advice or patient-care predictions.")
    if st.button("Evaluate model", type="primary"):
        with st.spinner("Training and evaluating the baseline…"):
            result = api_get("/api/model/evaluate", params, timeout=120)
        st.session_state["model_evaluation"] = result
    result = st.session_state.get("model_evaluation")
    if result:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy", f"{result['accuracy']:.3f}"); m2.metric("ROC AUC", f"{result['roc_auc']:.3f}")
        m3.metric("Precision", f"{result['precision']:.3f}"); m4.metric("Recall", f"{result['recall']:.3f}")
        st.caption(f"Train: {result['train_records']:,} · Test: {result['test_records']:,} · Target: {result['target']}")
        cm = pd.DataFrame(result["confusion_matrix"], index=["Actual survived", "Actual death"], columns=["Predicted survived", "Predicted death"])
        st.markdown("**Confusion matrix**")
        st.dataframe(cm, width="stretch")
        importance = pd.DataFrame(result["feature_importance"])
        if not importance.empty:
            importance = importance.sort_values("importance")
            fig = px.bar(importance, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Teal")
            fig.update_layout(height=520, coloraxis_showscale=False, margin=dict(l=0,r=0,t=15,b=0))
            st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Run the evaluation to view held-out metrics and feature importance.")

with patient_tab:
    st.subheader("Patient ID lookup")
    st.caption("Enter a full or partial ID. Search covers the entire dataset, regardless of the cohort filters in the sidebar.")
    query = st.text_input("Patient ID", placeholder="SYN-00001").strip()
    if query:
        result = api_get("/api/patients", {"query": query, "limit": 100})
        if result["total"]:
            st.success(f"Found {result['total']} matching record(s).")
            st.dataframe(pd.DataFrame(result["items"]), hide_index=True, width="stretch")
        else:
            st.warning(f"No patient ID matching ‘{query}’ was found. Try an ID such as SYN-00001.")
    else:
        result = api_get("/api/patients", {**params, "limit": 100})
        st.caption(f"Showing {min(result['total'], result['limit'])} of {result['total']:,} filtered records.")
        st.dataframe(pd.DataFrame(result["items"]), hide_index=True, width="stretch", height=500)
        export = requests.get(f"{API_URL}/api/patients/export.csv", params=params, timeout=30)
        if export.ok:
            st.download_button("Download filtered CSV", export.content, "oncopath_filtered_cohort.csv", "text/csv")

st.divider()
st.caption("OncoPath research prototype · This file contains clinical and biomarker columns but no gene-expression matrix or gene-interaction source. No TCGA/GEO connection is configured.")
