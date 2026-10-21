"""FastAPI backend for the CardioScan AI replication dashboard.

Replaces the retired Streamlit `app.py`. It trains each of the six pipelines on the
**70/30 training split only** (protocol B, `random_state=42`), so live predictions never
leak from the test set, and it serves the committed `results/*.csv` tables as clean JSON.

Design notes
------------
* Live predictions come from the plain pipeline (so the hard votes and the reported held-out
  accuracy match the paper-audit numbers in `results/`).
* Probabilities come from a `CalibratedClassifierCV` (Platt / sigmoid) fitted on the training
  split, because a raw SVM margin or a Random-Forest vote fraction is not a probability. The
  UI labels these as calibrated estimates.
* Every results table is parsed server-side: the `"0.838 ± 0.064"` strings in the protocol-C
  CSV are split into numeric mean/std columns.

Run:  .venv/bin/python -m uvicorn backend.app:app --reload --port 8000
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (BINARY, CATEGORICAL, FEATURES, NUMERIC, RANDOM_STATE,  # noqa: E402
                      RESULTS_DIR, load_cleveland, split_xy)
from src.evaluation import compute_metrics, get_score  # noqa: E402
from src.models import MAIN_MODELS, build_pipeline  # noqa: E402

RESULTS = RESULTS_DIR
FIGURES = RESULTS


# --------------------------------------------------------------------------- state
class State:
    """Holds the fitted models and the pre-computed ROC data."""
    models: dict = {}
    roc: dict = {}
    df: pd.DataFrame | None = None
    X: pd.DataFrame | None = None
    y: pd.Series | None = None
    yte: pd.Series | None = None


STATE = State()


def _fit_all() -> None:
    """Fit every pipeline (and a calibrated twin) on the 70/30 TRAIN split only."""
    df = load_cleveland()
    X, y = split_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    STATE.df, STATE.X, STATE.y, STATE.yte = df, X, y, yte

    for name in MAIN_MODELS:
        pipe = build_pipeline(name).fit(Xtr, ytr)
        # Calibrated copy for honest P(disease); sigmoid/Platt is the safe choice on 212 rows.
        cal = CalibratedClassifierCV(build_pipeline(name), method="sigmoid", cv=5)
        cal.fit(Xtr, ytr)
        test_metrics = compute_metrics(yte, pipe.predict(Xte), get_score(pipe, Xte))
        n = len(yte)
        p = test_metrics["accuracy"]
        STATE.models[name] = {
            "pipeline": pipe,
            "calibrated": cal,
            "test": test_metrics,
            "held_out_accuracy": p,
            "ci_half": 1.96 * np.sqrt(p * (1 - p) / n),
            "n_test": n,
        }

    STATE.roc["A"] = _roc_holdout(0.4)
    STATE.roc["B"] = _roc_holdout(0.3)
    STATE.roc["C"] = _roc_pooled_cv()


def _roc_holdout(test_size: float) -> list[dict]:
    """ROC on the held-out test split. Protocol A uses a 40% test split, B a 30% one."""
    _, Xte, _, yte = train_test_split(STATE.X, STATE.y, test_size=test_size, stratify=STATE.y,
                                      random_state=RANDOM_STATE)
    out = []
    for name in MAIN_MODELS:
        pipe = STATE.models[name]["pipeline"]
        sc = get_score(pipe, Xte)
        fpr, tpr, _ = roc_curve(yte, sc)
        out.append({"model": name, "auc": float(roc_auc_score(yte, sc)),
                    "points": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]})
    return out


def _roc_pooled_cv() -> list[dict]:
    """Pooled out-of-fold ROC over 10-fold CV (protocol C)."""
    skf = StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE)
    out = []
    for name in MAIN_MODELS:
        pipe = build_pipeline(name)
        meth = "decision_function" if hasattr(pipe, "decision_function") else "predict_proba"
        sc = cross_val_predict(pipe, STATE.X, STATE.y, cv=skf, method=meth)
        sc = sc[:, 1] if getattr(sc, "ndim", 1) == 2 else sc
        fpr, tpr, _ = roc_curve(STATE.y, sc)
        out.append({"model": name, "auc": float(roc_auc_score(STATE.y, sc)),
                    "points": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]})
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    _fit_all()
    yield


app = FastAPI(title="CardioScan AI backend", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- helpers
def _read(name: str) -> pd.DataFrame:
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        raise HTTPException(404, f"results table '{name}' not found")
    return pd.read_csv(path)


def _split_pm(col: pd.Series) -> list[dict]:
    """Turn a `"0.838 ± 0.064"` column into numeric mean/std columns."""
    mean, std = [], []
    for v in col.astype(str):
        if "±" in v:
            m, s = v.split("±")
            mean.append(float(m.strip())); std.append(float(s.strip()))
        else:
            mean.append(float(v)); std.append(np.nan)
    return mean, std


def _records(df: pd.DataFrame, key: str = "label") -> dict:
    """DataFrame -> {columns, rows}, key column first, 'x ± y' cells split into numeric
    mean/std columns.

    The committed CSVs keep their key in the first column — sometimes named (`model`),
    sometimes with an empty header (pandas reads that as `Unnamed: 0`). We normalise both
    to a clean key column and never reset_index (the key is already a column)."""
    df = df.copy()
    first = df.columns[0]
    if str(first).startswith("Unnamed") or str(first).strip() == "":
        df = df.rename(columns={first: key})

    splits: dict[str, list] = {}
    for c in df.columns:
        # NB: pandas 3 defaults to the `str` dtype for text, not `object`, so test the
        # content rather than the dtype before deciding a column holds "mean ± std".
        try:
            is_pm = df[c].astype(str).str.contains("±", regex=False).any()
        except Exception:
            is_pm = False
        if is_pm:
            mean, std = _split_pm(df[c])
            splits[c] = mean
            splits[f"{c}_std"] = std

    out: dict[str, list] = {}
    ordered: list[str] = []
    for c in df.columns:
        if c in splits:
            out[c] = splits[c]
            out[f"{c}_std"] = splits[f"{c}_std"]
            ordered += [c, f"{c}_std"]
        else:
            out[c] = df[c].tolist()
            ordered.append(c)
    return {"columns": ordered, "rows": {c: out[c] for c in ordered}}


def _isolation_tidy() -> dict:
    """Reshape the wide isolation-forest table (multi-row header) into tidy records."""
    path = RESULTS / "22_isolation_forest_test_accuracy.csv"
    if not path.exists():
        raise HTTPException(404, "isolation-forest table not found")
    wide = pd.read_csv(path, header=[0, 1], index_col=0)
    wide = wide.dropna(how="all")
    records = []
    for model, row in wide.iterrows():
        for (protocol, outlier), acc in row.items():
            if pd.isna(acc):
                continue
            records.append({"model": model, "protocol": protocol,
                            "outlier": outlier, "accuracy": float(acc)})
    return {"records": records}


# --------------------------------------------------------------------------- schemas
class Patient(BaseModel):
    age: float = Field(..., ge=25, le=90)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=80, le=220)
    chol: float = Field(..., ge=100, le=600)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=60, le=220)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0, le=7.0)
    slope: int = Field(..., ge=1, le=3)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=3, le=7)


# --------------------------------------------------------------------------- routes
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models": list(STATE.models), "n_rows": len(STATE.X)}


@app.get("/api/models")
def models_meta() -> dict:
    """Per-model held-out accuracy + calibration note (drives the Diagnosis page)."""
    out = []
    for name, s in STATE.models.items():
        out.append({
            "model": name,
            "held_out_accuracy": round(s["held_out_accuracy"], 4),
            "ci_half": round(s["ci_half"], 4),
            "n_test": s["n_test"],
            "auc": round(s["test"]["auc"], 4),
        })
    return {"models": out, "note":
            "Held-out accuracy on the untouched 70/30 test split (n=91). "
            "Probabilities shown on the Diagnosis page are Platt-calibrated."}


@app.post("/api/predict")
def predict(p: Patient) -> dict:
    row = pd.DataFrame([{f: getattr(p, f) for f in FEATURES}])
    per_model = []
    for name, s in STATE.models.items():
        pipe, cal = s["pipeline"], s["calibrated"]
        hard = int(pipe.predict(row)[0])
        prob = float(cal.predict_proba(row)[0, 1])
        per_model.append({
            "model": name,
            "prediction": hard,
            "probability": round(prob, 4),
            "held_out_accuracy": round(s["held_out_accuracy"], 4),
            "ci_half": round(s["ci_half"], 4),
        })
    votes = sum(m["prediction"] for m in per_model)
    mean_prob = float(np.mean([m["probability"] for m in per_model]))
    label = ("disease" if votes >= 4 else "healthy" if votes <= 2 else "borderline")
    return {
        "models": per_model,
        "consensus": {
            "votes_for_disease": votes,
            "total": len(per_model),
            "mean_probability": round(mean_prob, 4),
            "label": label,
        },
    }


@app.get("/api/benchmarks/{protocol}")
def benchmarks(protocol: str) -> dict:
    protocol = protocol.upper()
    if protocol not in "ABC":
        raise HTTPException(400, "protocol must be A, B or C")
    return _records(_read(f"16_test_metrics_all13_protocol{protocol}"), "model")


@app.get("/api/confusion/{protocol}")
def confusion(protocol: str) -> dict:
    protocol = protocol.upper()
    if protocol == "B":
        return _records(_read("18_confusion_matrices_protocolB_all13"), "model")
    # protocols A and C confusion matrices live in the long results table
    raise HTTPException(404, "confusion matrices only precomputed for protocol B")


@app.get("/api/sampling-noise")
def sampling_noise() -> dict:
    return _records(_read("20_test_accuracy_sampling_noise"), "model")


@app.get("/api/roc/{protocol}")
def roc(protocol: str) -> dict:
    protocol = protocol.upper()
    if protocol not in STATE.roc:
        raise HTTPException(404, f"ROC for protocol {protocol} not available")
    return {"protocol": protocol, "curves": STATE.roc[protocol]}


# ---- audit ---------------------------------------------------------------
@app.get("/api/audit/{table}")
def audit(table: str) -> dict:
    mapping = {
        "claims": "32_claims_evidence",
        "comparison": "23_paper_vs_ours_all13",
        "comparison7": "24_paper_vs_ours_paper7features",
        "f1-check": "31_paper_f1_consistency",
        "table6": "27_paper_table6_recomputed",
        "table3-vs-table6": "28_table3_vs_table6_accuracy",
        "table3-vs-table6-prf": "29_table3_vs_table6_prf",
        "splits": "26_split_sizes",
        "rankings": "25_model_rankings",
        "train-vs-test": "17_train_vs_test_accuracy",
        "class-balance": "01_class_balance_check",
        "feature-sets": "19_feature_set_comparison_accuracy",
    }
    if table not in mapping:
        raise HTTPException(404, f"unknown audit table '{table}'")
    return _records(_read(mapping[table]), "model")


# ---- statistical extensions (Phase 3) ------------------------------------
@app.get("/api/stats/{table}")
def stats(table: str) -> dict:
    """Serve the run_extensions.py tables (33-38): repeated-CV stability, McNemar,
    clinical operating points, threshold sweep, permutation importance, HGB baseline."""
    mapping = {
        "stability": ("33_repeated_cv_stability", "model"),
        "mcnemar": ("34_mcnemar_pairwise", "model_a"),
        "operating-points": ("35_clinical_operating_points", "model"),
        "threshold-sweep": ("36_threshold_sweep", "model"),
        "permutation": ("37_permutation_importance", "model"),
        "hgb": ("38_hgb_across_protocols", "protocol"),
    }
    if table not in mapping:
        raise HTTPException(404, f"unknown stats table '{table}'")
    name, key = mapping[table]
    return _records(_read(name), key)


# ---- hyperparameter lab --------------------------------------------------
@app.get("/api/lab/{table}")
def lab(table: str) -> dict:
    mapping = {
        "knn-sweep": "11_knn_k_sweep",
        "knn-best": "12_knn_best_k_summary",
        "dt-criterion": "07_dt_criterion_comparison",
        "dt-sweep": "08_dt_feature_count_sweep",
        "svm-kernels": "13_svm_kernel_comparison",
        "rf-comparison": "10_rf_default_vs_tuned",
        "rf-importance": "09_rf_feature_importance",
        "feature-scores": "05_feature_selection_scores",
        "logreg-coef": "14_logreg_coefficients",
    }
    if table == "isolation":
        return _isolation_tidy()
    if table not in mapping:
        raise HTTPException(404, f"unknown lab table '{table}'")
    key = "feature" if table in ("feature-scores", "rf-importance", "logreg-coef") else "label"
    return _records(_read(mapping[table]), key)


# ---- EDA -----------------------------------------------------------------
@app.get("/api/eda/{table}")
def eda(table: str) -> dict:
    mapping = {
        "describe": "04_describe",
        "correlations": "03_correlation_with_target",
    }
    if table == "class-balance":
        vc = STATE.df["target"].value_counts().sort_index()
        return {"labels": ["No disease", "Disease"], "counts": [int(vc[0]), int(vc[1])],
                "total": int(vc.sum())}
    if table == "rows":
        return _records(STATE.df[FEATURES + ["target"]].reset_index(drop=True), "row")
    if table == "distributions":
        clean = pd.read_csv(RESULTS.parent / "data" / "heart_clean.csv")
        out = {}
        for c in FEATURES:
            g = clean.groupby("target")[c]
            out[c] = {
                "by_target": [
                    {"target": int(t), "mean": float(gr.mean()), "std": float(gr.std()),
                     "q25": float(gr.quantile(.25)), "q50": float(gr.quantile(.5)),
                     "q75": float(gr.quantile(.75)), "min": float(gr.min()), "max": float(gr.max()),
                     "values": [float(v) for v in gr.dropna()]}
                    for t, gr in g
                ]
            }
        return out
    if table not in mapping:
        raise HTTPException(404, f"unknown EDA table '{table}'")
    return _records(_read(mapping[table]), "feature")


@app.get("/api/figures/{name}")
def figure(name: str):
    path = (FIGURES / f"{name}.png").resolve()
    if not str(path).startswith(str(FIGURES.resolve())) or not path.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(path, media_type="image/png")
