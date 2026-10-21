import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import LEAKAGE_COLUMNS, POST_BASELINE_COLUMNS, TARGET


def evaluate_baseline(frame: pd.DataFrame, seed: int = 42) -> dict:
    """Evaluate an exploratory baseline on a stratified holdout split."""
    data = frame.dropna(subset=[TARGET]).copy()
    features = [column for column in data.columns if column not in LEAKAGE_COLUMNS | POST_BASELINE_COLUMNS]
    if not features:
        raise ValueError("No eligible baseline predictor columns were found.")
    X = data[features]
    y = data[TARGET].astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 4:
        raise ValueError("The selected cohort needs at least four examples in each outcome class for a stratified holdout evaluation.")

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]), categorical))

    pipeline = Pipeline([
        ("prepare", ColumnTransformer(transformers)),
        ("classifier", RandomForestClassifier(n_estimators=180, min_samples_leaf=3,
             class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=.25, random_state=seed, stratify=y)
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    probabilities = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
    predicted = (probabilities >= .5).astype(int)
    matrix = confusion_matrix(y.iloc[test_idx], predicted, labels=[0, 1])
    feature_names = pipeline.named_steps["prepare"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    top = sorted(zip(feature_names.tolist(), importances.tolist()), key=lambda item: item[1], reverse=True)[:20]
    return {
        "target": TARGET,
        "predictors": features,
        "train_records": int(len(train_idx)),
        "test_records": int(len(test_idx)),
        "accuracy": float(accuracy_score(y.iloc[test_idx], predicted)),
        "roc_auc": float(roc_auc_score(y.iloc[test_idx], probabilities)),
        "precision": float(precision_score(y.iloc[test_idx], predicted, zero_division=0)),
        "recall": float(recall_score(y.iloc[test_idx], predicted, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": ["Survived", "Death"],
        "feature_importance": [{"feature": str(name), "importance": float(value)} for name, value in top],
        "note": "Exploratory retrospective baseline; not clinically validated or intended for care decisions.",
    }
