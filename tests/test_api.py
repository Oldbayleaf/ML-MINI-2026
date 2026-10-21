"""Backend API tests: every endpoint answers, tables parse, validation bites.

These use FastAPI's TestClient, which runs the app (and its lifespan, so the models get
fitted on the training split) in-process — no server or port needed.
"""
import pytest

fastapi = pytest.importorskip("fastapi")  # skip cleanly if backend deps are absent
from fastapi.testclient import TestClient  # noqa: E402

from backend.app import app  # noqa: E402

VALID_PATIENT = {"age": 63, "sex": 1, "cp": 4, "trestbps": 152, "chol": 290, "fbs": 1,
                 "restecg": 2, "thalach": 112, "exang": 1, "oldpeak": 2.6, "slope": 2,
                 "ca": 2, "thal": 7}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["n_rows"] == 303


def test_all_get_endpoints_answer(client):
    paths = [
        "/api/models", "/api/benchmarks/A", "/api/benchmarks/B", "/api/benchmarks/C",
        "/api/confusion/B", "/api/sampling-noise",
        "/api/roc/A", "/api/roc/B", "/api/roc/C",
        "/api/audit/claims", "/api/audit/comparison", "/api/audit/f1-check",
        "/api/audit/table6", "/api/audit/table3-vs-table6", "/api/audit/splits",
        "/api/audit/rankings", "/api/audit/train-vs-test", "/api/audit/class-balance",
        "/api/audit/feature-sets",
        "/api/lab/knn-sweep", "/api/lab/knn-best", "/api/lab/dt-criterion", "/api/lab/dt-sweep",
        "/api/lab/svm-kernels", "/api/lab/rf-comparison", "/api/lab/rf-importance",
        "/api/lab/feature-scores", "/api/lab/logreg-coef", "/api/lab/isolation",
        "/api/eda/describe", "/api/eda/correlations", "/api/eda/class-balance",
        "/api/eda/rows", "/api/eda/distributions",
    ]
    for p in paths:
        assert client.get(p).status_code == 200, p


def test_unknown_tables_are_404_not_500(client):
    for p in ["/api/audit/nope", "/api/lab/nope", "/api/eda/nope", "/api/figures/nope"]:
        assert client.get(p).status_code == 404, p


def test_protocol_c_splits_mean_std(client):
    """The '0.838 ± 0.064' cells must come back as numeric mean and std."""
    data = client.get("/api/benchmarks/C").json()
    assert "accuracy" in data["columns"] and "accuracy_std" in data["columns"]
    nb = data["rows"]["model"].index("Naive Bayes")
    assert abs(data["rows"]["accuracy"][nb] - 0.838) < 1e-6
    assert abs(data["rows"]["accuracy_std"][nb] - 0.064) < 1e-6


def test_predict_returns_all_models_and_consensus(client):
    r = client.post("/api/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    d = r.json()
    assert len(d["models"]) == 6
    for m in d["models"]:
        assert m["prediction"] in (0, 1)
        assert 0.0 <= m["probability"] <= 1.0
        assert 0.0 < m["held_out_accuracy"] < 1.0
    cons = d["consensus"]
    assert cons["total"] == 6
    assert 0 <= cons["votes_for_disease"] <= 6
    assert 0.0 <= cons["mean_probability"] <= 1.0


def test_predict_rejects_out_of_range(client):
    bad = dict(VALID_PATIENT); bad["age"] = 200
    assert client.post("/api/predict", json=bad).status_code == 422
    missing = dict(VALID_PATIENT); missing.pop("thal")
    assert client.post("/api/predict", json=missing).status_code == 422


def test_roc_curves_are_monotone_and_bounded(client):
    for p in ["A", "B", "C"]:
        curves = client.get(f"/api/roc/{p}").json()["curves"]
        assert len(curves) == 6
        for cur in curves:
            assert 0.5 <= cur["auc"] <= 1.0
            fprs = [pt[0] for pt in cur["points"]]
            assert fprs == sorted(fprs), f"ROC FPR not monotone for {p}"


def test_isolation_records_are_tidy(client):
    recs = client.get("/api/lab/isolation").json()["records"]
    assert len(recs) > 0
    assert {"model", "protocol", "outlier", "accuracy"} <= set(recs[0])


def test_eda_rows_shape(client):
    data = client.get("/api/eda/rows").json()
    n = len(data["rows"]["age"])
    assert n == 303
    assert "target" in data["columns"]
