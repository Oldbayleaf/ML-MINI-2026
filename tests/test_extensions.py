"""Phase 3: run_extensions.py pure logic + /api/stats endpoints.

The heavy studies (repeated CV etc.) are run once and committed as results/33-38; here we
test the statistics that must be exactly right (McNemar) and that the API serves the tables.
"""
import numpy as np
import pytest

pytest.importorskip("fastapi")  # skip cleanly if backend deps are absent

from run_extensions import _mcnemar_exact  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.app import app

    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------ McNemar exact test
def test_mcnemar_symmetric_in_b_c():
    # swapping the two models swaps b and c; the p-value must not change
    assert _mcnemar_exact(5, 18) == _mcnemar_exact(18, 5)


def test_mcnemar_no_discordance_is_1():
    assert _mcnemar_exact(0, 0) == 1.0


def test_mcnemar_extreme_counts_are_significant():
    # 2 vs 18 discordant: heavily skewed -> small p
    assert _mcnemar_exact(2, 18) < 0.001


def test_mcnemar_balanced_counts_are_not_significant():
    # 10 vs 11: essentially a coin flip
    assert _mcnemar_exact(10, 11) > 0.9


def test_mcnemar_p_value_in_unit_interval():
    for b in range(0, 25, 3):
        for c in range(0, 25, 3):
            p = _mcnemar_exact(b, c)
            assert 0.0 <= p <= 1.0


def test_mcnemar_known_value():
    # exact two-sided binomial for b=5, c=18, n=23: 2*P(X<=5) = 0.01062...
    assert _mcnemar_exact(5, 18) == pytest.approx(0.0106225, abs=1e-5)


# ------------------------------------------------------------------ committed tables
def test_stability_table_ranks_all_models(client):
    data = client.get("/api/stats/stability").json()
    assert "model" in data["columns"] and "mean_accuracy" in data["columns"]
    accs = data["rows"]["mean_accuracy"]
    assert len(accs) == 7                     # six paper models + HistGradientBoosting
    assert accs == sorted(accs, reverse=True)  # served sorted best-first


def test_mcnemar_table_has_significance_flag(client):
    data = client.get("/api/stats/mcnemar").json()
    assert "mcnemar_p" in data["columns"]
    assert "significant_05" in data["columns"]
    ps = [p for p in data["rows"]["mcnemar_p"] if not np.isnan(p)]
    assert all(0 <= p <= 1 for p in ps)
    # every p flagged 'yes' really is below 0.05
    for p, sig in zip(data["rows"]["mcnemar_p"], data["rows"]["significant_05"]):
        if sig == "yes":
            assert p < 0.05


def test_operating_points_cover_both_settings(client):
    data = client.get("/api/stats/operating-points").json()
    assert set(data["rows"]["operating_point"]) == {"default_0.5", "sensitivity>=0.90"}
    # the sensitivity-first point must never miss more cases than the 0.5 default
    rows = list(zip(data["rows"]["model"], data["rows"]["operating_point"],
                    data["rows"]["missed_cases"]))
    by_model = {}
    for model, op, missed in rows:
        by_model.setdefault(model, {})[op] = missed
    for model, d in by_model.items():
        assert d["sensitivity>=0.90"] <= d["default_0.5"], model


def test_threshold_sweep_grid(client):
    data = client.get("/api/stats/threshold-sweep").json()
    ts = sorted(set(data["rows"]["threshold"]))
    assert ts[0] == 0.05 and ts[-1] == 0.95 and len(ts) == 19


def test_permutation_importance_ranked_per_model(client):
    data = client.get("/api/stats/permutation").json()
    models = set(data["rows"]["model"])
    assert "HistGradientBoosting" in models and "Logistic Regression" in models
    for m in models:
        ranks = [r for mm, r in zip(data["rows"]["model"], data["rows"]["rank"]) if mm == m]
        assert ranks == list(range(1, len(ranks) + 1)), f"{m} ranks not 1..n"


def test_hgb_served_and_never_beats_top_of_repeated_cv(client):
    hgb = client.get("/api/stats/hgb").json()
    assert set(hgb["rows"]["protocol"]) == {"A", "B", "C"}
    stab = client.get("/api/stats/stability").json()
    # honest yardstick: gradient boosting does not topsheet the paper's models on 303 rows
    top = max(stab["rows"]["mean_accuracy"])
    hgb_mean = max(a for m, a in zip(stab["rows"]["model"], stab["rows"]["mean_accuracy"])
                   if m == "HistGradientBoosting")
    assert hgb_mean < top


def test_unknown_stats_table_404(client):
    assert client.get("/api/stats/nope").status_code == 404
