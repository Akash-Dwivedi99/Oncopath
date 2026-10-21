from pathlib import Path
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="OncoPath | Clinical Cohort Intelligence", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
DATA_PATH = Path(__file__).with_name("oncolody_dataset_final.csv")
TARGET = "death_within_5yr"
LEAKAGE = {"patient_id", TARGET, "survival_5yr_status", "follow_up_months"}
POST_OUTCOME = {
    "treatment_received", "surgical_margin_clear", "days_diagnosis_to_treatment",
    "treatment_completed", "resection_status", "chemo_cycles_completed",
    "dose_intensity_percent", "response_first_scan", "recurrence_score",
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return df


@st.cache_resource
def fit_model(df, cancer_filter):
    # Only baseline clinical/biomarker fields are used; post-treatment fields are withheld.
    features = [c for c in df.columns if c not in LEAKAGE | POST_OUTCOME]
    data = df.dropna(subset=[TARGET]).copy()
    X, y = data[features], data[TARGET].astype(int)
    nums = X.select_dtypes(include=np.number).columns.tolist()
    cats = [c for c in features if c not in nums]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), cats),
    ])
    model = Pipeline([("prepare", prep), ("classifier", RandomForestClassifier(
        n_estimators=180, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1
    ))])
    train, test = train_test_split(np.arange(len(y)), test_size=.25, random_state=42, stratify=y)
    model.fit(X.iloc[train], y.iloc[train])
    prob = model.predict_proba(X.iloc[test])[:, 1]
    pred = (prob >= .5).astype(int)
    return model, features, X, {
        "accuracy": accuracy_score(y.iloc[test], pred),
        "auc": roc_auc_score(y.iloc[test], prob),
        "cm": confusion_matrix(y.iloc[test], pred),
        "n_test": len(test),
    }


def make_graph(df, limit=55):
    """Build a feature-value co-occurrence graph from observed patient rows."""
    columns = ["cancer_type", "tumor_stage", "molecular_subtype", "smoking_status",
               "m_stage", "er_status", "pr_status", "her2_status", "braf_v600e",
               "alk_rearranged", "lymphovascular_invasion", "histologic_grade"]
    columns = [c for c in columns if c in df]
    G = nx.Graph()
    pairs = {}
    counts = {}
    for row in df[columns].fillna("Unknown").astype(str).to_dict("records"):
        vals = [(c, v) for c, v in row.items() if v and v.lower() not in {"nan", "unknown", "none"}]
        for item in vals:
            counts[item] = counts.get(item, 0) + 1
        for a, b in combinations(vals, 2):
            if a[0] == b[0]:
                continue
            edge = tuple(sorted((a, b)))
            pairs[edge] = pairs.get(edge, 0) + 1
    keep = set(sorted(counts, key=counts.get, reverse=True)[:limit])
    for (field, value), n in counts.items():
        if (field, value) in keep:
            G.add_node(f"{field}: {value}", field=field, value=value, count=n)
    for (a, b), weight in pairs.items():
        if a in keep and b in keep and weight >= max(3, len(df) * .008):
            G.add_edge(f"{a[0]}: {a[1]}", f"{b[0]}: {b[1]}", weight=weight)
    return G


def graph_figure(G):
    if not G.nodes:
        return go.Figure().update_layout(title="No graph links in this filtered cohort")
    pos = nx.spring_layout(G, seed=17, weight="weight", k=0.8)
    edge_x, edge_y = [], []
    for u, v in G.edges:
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#334155"), hoverinfo="none")
    nodes = list(G.nodes)
    sizes = [9 + 3 * np.log1p(G.nodes[n]["count"]) for n in nodes]
    colors = [G.nodes[n]["field"] for n in nodes]
    node_trace = go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers+text", text=nodes, textposition="top center", textfont=dict(size=9),
        marker=dict(size=sizes, color=pd.Categorical(colors).codes, colorscale="Turbo", showscale=False,
                    line=dict(width=1, color="#e2e8f0")),
        customdata=[[G.nodes[n]["count"], G.degree(n)] for n in nodes],
        hovertemplate="%{text}<br>Rows: %{customdata[0]}<br>Connections: %{customdata[1]}<extra></extra>")
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(height=600, margin=dict(l=5,r=5,t=20,b=5), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False),
                      showlegend=False, dragmode="pan")
    return fig


st.markdown("""
<style>
:root { --ink:#15243a; --muted:#65758b; --teal:#087f8c; --line:#dce5ee; --paper:#f5f8fb; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { padding-top:1.4rem; padding-bottom:2.5rem; max-width:1480px; }
[data-testid="stHeader"] { background:rgba(245,248,251,.88); }
[data-testid="stSidebar"] { background:#edf3f7; border-right:1px solid var(--line); }
[data-testid="stSidebar"] h2 { color:var(--ink); letter-spacing:-.02em; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:17px 19px; border-radius:13px; box-shadow:0 2px 7px rgba(24,49,75,.04); }
[data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
[data-testid="stMetricValue"] { color:var(--ink); font-weight:720; }
div[data-testid="stTabs"] button { font-weight:650; color:#53657a; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--teal); }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:8px; }
div.stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); border-radius:9px; font-weight:650; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
.oncopath-hero { display:flex; justify-content:space-between; align-items:center; gap:24px; margin:0 0 1.1rem; padding:22px 26px; background:linear-gradient(115deg,#10283d,#164758); border-radius:16px; color:#fff; box-shadow:0 8px 24px rgba(18,50,69,.14); }
.oncopath-hero h1 { color:#fff; font-size:2rem; margin:0 0 5px; letter-spacing:-.04em; }
.oncopath-hero p { color:#c9dbe4; margin:0; font-size:.98rem; }
.hero-tag { padding:7px 12px; border:1px solid rgba(255,255,255,.25); border-radius:999px; color:#d8f2f0; font-size:.78rem; white-space:nowrap; }
@media(max-width:700px) { .oncopath-hero { padding:18px; align-items:flex-start; flex-direction:column; } }
</style>
<div class="oncopath-hero"><div><h1>OncoPath</h1><p>Clinical cohort intelligence · graph exploration · baseline risk modeling</p></div><span class="hero-tag">RESEARCH PROTOTYPE</span></div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}. Place the CSV beside app.py.")
    st.stop()
df = load_data()

with st.sidebar:
    st.header("Cohort filters")
    cancers = sorted(df["cancer_type"].dropna().unique())
    selected_cancers = st.multiselect("Cancer type", cancers, default=cancers)
    stages = sorted(df["tumor_stage"].dropna().unique(), key=str)
    selected_stages = st.multiselect("Tumor stage", stages, default=stages)
    st.divider()
    st.caption("Filters apply to cohort summaries and the network. The model evaluation uses the full eligible cohort.")

filtered = df[df.cancer_type.isin(selected_cancers) & df.tumor_stage.isin(selected_stages)].copy()
events = filtered[TARGET].mean() * 100 if len(filtered) else 0
c1,c2,c3,c4 = st.columns(4)
c1.metric("Patients", f"{len(filtered):,}")
c2.metric("Cancer types", filtered.cancer_type.nunique())
c3.metric("5-year deaths", f"{events:.1f}%")
c4.metric("Available fields", f"{df.shape[1]-1}")

overview, network, risk, records = st.tabs(["Cohort overview", "Feature network", "Risk model", "Patient records"])
with overview:
    left,right = st.columns([1.15, .85])
    with left:
        st.subheader("Cohort composition")
        counts = filtered.groupby(["cancer_type", "tumor_stage"], dropna=False).size().reset_index(name="Patients")
        fig = px.bar(counts, x="cancer_type", y="Patients", color="tumor_stage", barmode="stack",
                     color_discrete_sequence=px.colors.qualitative.Bold, labels={"cancer_type":"Cancer", "tumor_stage":"Stage"})
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0), legend_title="Stage")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Outcome by cancer type")
        outcome = filtered.groupby("cancer_type")[TARGET].mean().mul(100).rename("Death within 5 years (%)").reset_index()
        fig = px.bar(outcome, x="cancer_type", y="Death within 5 years (%)", color="cancer_type",
                     color_discrete_sequence=px.colors.qualitative.Bold, text_auto=".1f")
        fig.update_layout(height=380, showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Biomarker and clinical distributions")
    numeric_options = [c for c in ["age", "bmi", "tumor_size_cm", "ki67_percent", "crp_mg_l", "CEA", "CA 15-3", "CYFRA 21-1", "ca_19_9_u_ml"] if c in filtered]
    if numeric_options:
        measure = st.selectbox("Measure", numeric_options)
        box = px.box(filtered, x="cancer_type", y=measure, color="cancer_type", points="outliers", color_discrete_sequence=px.colors.qualitative.Bold)
        box.update_layout(height=360, showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(box, use_container_width=True)

with network:
    st.subheader("Observed feature co-occurrence network")
    st.write("Each node is a categorical feature value; an edge means two values appeared in the same patient record. Edge weight is co-occurrence count. This is a cohort association graph, not a gene or biological interaction network.")
    G = make_graph(filtered)
    a,b,c = st.columns(3)
    a.metric("Nodes", G.number_of_nodes())
    b.metric("Links", G.number_of_edges())
    c.metric("Connected components", nx.number_connected_components(G) if G else 0)
    st.plotly_chart(graph_figure(G), use_container_width=True, config={"scrollZoom": True})
    if G:
        degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
        st.markdown("**Most connected feature values**")
        st.dataframe(pd.DataFrame(degrees, columns=["Feature value", "Connections"]), hide_index=True, use_container_width=True)

with risk:
    st.subheader("Five-year outcome classification")
    st.write("A Random Forest estimates the dataset label `death_within_5yr`. This is a retrospective educational demonstration, not a clinical prediction tool.")
    st.info("The model excludes patient identifiers, follow-up duration, the duplicate survival label, and treatment or response fields that occur after baseline. The supplied data is labeled with synthetic IDs; dataset provenance and clinical validity were not provided.")
    if st.button("Fit and evaluate baseline model", type="primary"):
        with st.spinner("Training and evaluating on a stratified 75/25 split…"):
            model, features, X, metrics = fit_model(df, tuple(selected_cancers))
            st.session_state["model_result"] = (model, features, X, metrics)
    result = st.session_state.get("model_result")
    if result:
        model, features, X, metrics = result
        a,b,c = st.columns(3)
        a.metric("Held-out accuracy", f"{metrics['accuracy']:.3f}")
        b.metric("Held-out ROC AUC", f"{metrics['auc']:.3f}")
        c.metric("Test records", f"{metrics['n_test']:,}")
        cm = pd.DataFrame(metrics["cm"], index=["Actual survived", "Actual death"], columns=["Predicted survived", "Predicted death"])
        st.markdown("**Confusion matrix**")
        st.dataframe(cm, use_container_width=True)
        try:
            names = model.named_steps["prepare"].get_feature_names_out()
            importance = model.named_steps["classifier"].feature_importances_
            imp = pd.DataFrame({"Feature": names, "Importance": importance}).nlargest(18, "Importance").sort_values("Importance")
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Teal")
            fig.update_layout(height=540, margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
            st.subheader("Random Forest feature importance")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not display feature importances: {exc}")
    else:
        st.caption("Select the button to train the baseline model and view held-out metrics.")

with records:
    st.subheader("Filtered patient records")
    st.caption("Patient IDs are pseudonymous labels in the supplied file. Use this view for dataset exploration only.")
    query = st.text_input("Find patient ID", placeholder="SYN-00001").strip()
    if query:
        # Patient lookup spans the full file, regardless of sidebar cohort filters.
        patient_ids = df["patient_id"].astype(str).str.strip()
        exact = patient_ids.str.casefold().eq(query.casefold())
        view = df[exact] if exact.any() else df[patient_ids.str.contains(query, case=False, na=False, regex=False)]
        if view.empty:
            st.warning(f"No patient ID matching ‘{query}’ was found in the dataset. Try an ID such as SYN-00001.")
        else:
            st.caption(f"Showing {len(view)} matching record(s) from the full dataset, independent of the cohort filters.")
    else:
        view = filtered
    st.dataframe(view, use_container_width=True, hide_index=True, height=520)
    st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"), "oncopath_filtered_cohort.csv", "text/csv")

st.divider()
st.caption("OncoPath prototype · The CSV contains clinical and biomarker fields, but no gene-expression or molecular interaction data. No external TCGA/GEO data is connected.")
