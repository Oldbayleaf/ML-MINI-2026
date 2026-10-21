"""Every model must fit, predict, and stay deterministic under the fixed seed."""
import numpy as np

from src.models import MAIN_MODELS as MODELS, FeatureSelector, build_pipeline, build_tuned_rf


def test_all_models_fit_and_predict_binary(fitted_b):
    Xte, yte = fitted_b["Xte"], fitted_b["yte"]
    for name, model in fitted_b["models"].items():
        pred = model.predict(Xte)
        assert set(np.unique(pred)) <= {0, 1}, f"{name} predicted a non-binary label"
        assert pred.shape == yte.shape


def test_all_models_report_reasonable_accuracy(fitted_b):
    """Sanity floor only: even the weakest model (tree) beats a coin flip on this data."""
    Xte, yte = fitted_b["Xte"], fitted_b["yte"]
    for name, model in fitted_b["models"].items():
        acc = model.score(Xte, yte)
        assert 0.60 < acc < 1.0, f"{name} test accuracy {acc:.3f} is implausible"


def test_pipeline_is_deterministic():
    """Two fits with the same seed give identical predictions (reproducibility)."""
    from sklearn.model_selection import train_test_split

    from src.data import RANDOM_STATE, load_cleveland, split_xy

    X, y = split_xy(load_cleveland())
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    for name in MODELS:
        p1 = build_pipeline(name).fit(Xtr, ytr).predict(Xte)
        p2 = build_pipeline(name).fit(Xtr, ytr).predict(Xte)
        assert np.array_equal(p1, p2), f"{name} is not deterministic"


def test_feature_selectors(fitted_b):
    Xtr, Xte, ytr = fitted_b["Xtr"], fitted_b["Xte"], fitted_b["ytr"]
    all13 = FeatureSelector("all").fit(Xtr, ytr)
    assert len(all13.selected_) == 13
    mi7 = FeatureSelector("mi", k=7).fit(Xtr, ytr)
    assert len(mi7.selected_) == 7
    fixed = FeatureSelector("fixed", features=["age", "sex", "cp"]).fit(Xtr, ytr)
    assert fixed.selected_ == ["age", "sex", "cp"]
    assert fixed.transform(Xte).shape[1] == 3


def test_knn_respects_n_neighbors_override(fitted_b):
    Xtr, ytr = fitted_b["Xtr"], fitted_b["ytr"]
    m = build_pipeline("KNN", n_neighbors=3).fit(Xtr, ytr)
    assert m[-1].n_neighbors == 3


def test_tuned_rf_selects_params_from_grid(fitted_b):
    Xtr, ytr = fitted_b["Xtr"], fitted_b["ytr"]
    gs = build_tuned_rf().fit(Xtr, ytr)
    assert gs.best_params_, "GridSearchCV returned no best params"
    assert hasattr(gs, "best_score_") and 0 < gs.best_score_ <= 1


def test_pipelines_share_preprocessing_contract(fitted_b):
    """The final step of every pipeline must be the classifier (select -> prep -> clf)."""
    for name, model in fitted_b["models"].items():
        steps = [s for s, _ in model.steps]
        assert steps[-1] == "clf", f"{name} does not end in a classifier step"
        assert "prep" in steps and "select" in steps
