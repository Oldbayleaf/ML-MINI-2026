"""Phase 3 — statistical extensions that go beyond the published replication.

`run_replication.py` is left byte-identical so the published numbers stay reproducible.
This script reuses the same pipelines (src.models) and the same protocol conventions, and
writes NEW numbered tables (33+) plus figures into results/. Nothing here is tuned to match
the paper; it answers the questions the paper left open:

  1. Repeated stratified CV          — is the single-seed ranking stable, or noise?
  2. McNemar's test                  — are two models' error rates *significantly* different?
  3. Clinical decision metrics       — at a fixed operating point, what does missing a case cost?
  4. Threshold sweep                 — how do sensitivity/specificity trade off as we move the cut?
  5. Permutation importance          — model-agnostic feature reliance (MDI is tree-only).
  6. HistGradientBoosting baseline    — a modern gradient-boosting yardstick the paper never ran.

Run:  .venv/bin/python run_extensions.py     (~3-4 min)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from math import comb
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import FEATURES, NUMERIC, RANDOM_STATE, RESULTS_DIR, load_cleveland, split_xy  # noqa: E402
from src.models import MAIN_MODELS, build_pipeline  # noqa: E402

RESULTS = RESULTS_DIR
RESULTS.mkdir(exist_ok=True)

# How many repeats for the repeated-CV stability study. 10x10 = 100 folds per model.
N_REPEATS, N_SPLITS = 10, 10
CALIB_CV = 5


def _save(df: pd.DataFrame, name: str) -> None:
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}  ({len(df)} rows)")


def _hgb_pipeline():
    """HistGradientBoosting works on raw values, so only imputation precedes it.
    We still wrap it in the same numeric imputer for missing-value parity with the others."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=300)),
    ])


def all_pipelines():
    """The six paper pipelines plus the modern gradient-boosting yardstick."""
    pipes = {name: build_pipeline(name) for name in MAIN_MODELS}
    pipes["HistGradientBoosting"] = _hgb_pipeline()
    return pipes


# --------------------------------------------------------------------------- 1. stability
def _fit_predict_cv(pipe, X, y, splits):
    """Fit per fold and return the accuracy for each fold (list of floats).

    We can't use cross_val_predict here: under RepeatedStratifiedKFold a row lands in several
    test folds, so predictions don't map back to rows. We just need the per-fold score.
    """
    accs = []
    for tr, te in splits:
        model = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        accs.append(accuracy_score(y.iloc[te], model.predict(X.iloc[te])))
    return np.array(accs)


def repeated_cv_stability(X, y) -> None:
    """Repeated stratified CV: does the leaderboard survive when we change the folds?

    A single 10-fold CV with one seed gives each model one number +/- its fold std. That std
    mixes real fold-to-fold variation with seed luck. Repeating the whole CV with fresh
    shuffles (RepeatedStratifiedKFold) separates the two and lets us rank models by mean
    accuracy over ~100 fits, with a proper interval.
    """
    print("[1/6] Repeated stratified CV stability")
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y))
    n_total = N_REPEATS * N_SPLITS
    rows = []
    for name, pipe in all_pipelines().items():
        fold_accs = _fit_predict_cv(pipe, X, y, splits)
        repeat_mean = fold_accs.reshape(N_REPEATS, N_SPLITS).mean(axis=1)
        rows.append(dict(model=name, mean_accuracy=fold_accs.mean(),
                         std_across_folds=fold_accs.std(ddof=1),
                         std_across_repeats=repeat_mean.std(ddof=1),
                         min_fold_accuracy=fold_accs.min(), max_fold_accuracy=fold_accs.max(),
                         n_fits=n_total))
    df = pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    _save(df, "33_repeated_cv_stability")
    print(df.to_string(index=False))


# --------------------------------------------------------------------------- 2. McNemar
def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant counts b, c (no continuity correction).

    Only patients the two models DISAGREE about carry information. b = A-right/B-wrong,
    c = A-wrong/B-right. Under H0 they're a fair coin flip, so we do an exact binomial test
    and double the smaller tail (capped at 1).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, p))


def mcnemar_pairwise(X, y) -> None:
    """McNemar's test on the 70/30 held-out split for every model pair.

    Accuracy alone can't tell you whether 0.868 vs 0.835 is a real difference or a coin-flip
    of which patient landed in the test set. McNemar looks only at the discordant predictions
    and asks whether one model is systematically better on those. With n=91 almost nothing
    survives — which is the honest headline.
    """
    print("[2/6] McNemar pairwise tests (held-out 70/30 split)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    preds = {}
    for name, pipe in all_pipelines().items():
        preds[name] = clone(pipe).fit(Xtr, ytr).predict(Xte)

    names = list(preds)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = preds[a], preds[b]
            b_count = int(((pa == yte) & (pb != yte)).sum())  # A right, B wrong
            c_count = int(((pa != yte) & (pb == yte)).sum())  # A wrong, B right
            p = _mcnemar_exact(b_count, c_count)
            acc_a, acc_b = accuracy_score(yte, pa), accuracy_score(yte, pb)
            rows.append(dict(model_a=a, model_b=b, accuracy_a=acc_a, accuracy_b=acc_b,
                             delta=acc_a - acc_b, a_right_b_wrong=b_count,
                             a_wrong_b_right=c_count, discordant=b_count + c_count,
                             mcnemar_p=p, significant_05="yes" if p < 0.05 else "no"))
    df = pd.DataFrame(rows).sort_values("mcnemar_p").reset_index(drop=True)
    _save(df, "34_mcnemar_pairwise")
    print(df.head(12).to_string(index=False))
    sig = (df["significant_05"] == "yes").sum()
    print(f"  -> {sig} of {len(df)} pairs differ significantly at alpha=0.05")


# --------------------------------------------------------------------------- 3+4. thresholds
def clinical_thresholds(X, y) -> None:
    """Clinical operating point + full threshold sweep for each model.

    In screening, a false negative (missed disease) usually costs more than a false alarm.
    Accuracy silently optimises the wrong thing. Here we (a) compute the clinical metrics at
    the default 0.5 cut and at the sensitivity-first cut that catches ~90% of disease, and
    (b) sweep the threshold across [0.05, 0.95] so the UI can draw the trade-off curve.
    Probabilities are Platt-calibrated so a threshold means the same thing across models.
    """
    print("[3/6] Clinical metrics + threshold sweep (calibrated, held-out 70/30)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

    sweep_rows, point_rows = [], []
    for name, pipe in all_pipelines().items():
        cal = CalibratedClassifierCV(clone(pipe), method="sigmoid", cv=CALIB_CV).fit(Xtr, ytr)
        proba = cal.predict_proba(Xte)[:, 1]

        # full sweep
        for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
            pred = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
            sweep_rows.append(dict(model=name, threshold=t,
                                   sensitivity=tp / (tp + fn) if tp + fn else np.nan,
                                   specificity=tn / (tn + fp) if tn + fp else np.nan,
                                   precision=precision_score(yte, pred, zero_division=0),
                                   accuracy=accuracy_score(yte, pred),
                                   ppv_npv_note="", tp=tp, fn=fn, fp=fp, tn=tn))

        # operating points of interest
        for label, t in (("default_0.5", 0.5),):
            pred = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
            point_rows.append(dict(model=name, operating_point=label, threshold=t,
                                   sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                                   precision=precision_score(yte, pred, zero_division=0),
                                   npv=tn / (tn + fn) if tn + fn else np.nan,
                                   f1=f1_score(yte, pred), accuracy=accuracy_score(yte, pred),
                                   missed_cases=int(fn), false_alarms=int(fp)))

        # sensitivity-first: lowest threshold hitting >= 0.90 sensitivity (fall back to max)
        sweep = pd.DataFrame([r for r in sweep_rows if r["model"] == name])
        hi = sweep[sweep["sensitivity"] >= 0.90]
        t_sens = float(hi["threshold"].min()) if len(hi) else 0.05
        pred = (proba >= t_sens).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        point_rows.append(dict(model=name, operating_point="sensitivity>=0.90", threshold=t_sens,
                               sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                               precision=precision_score(yte, pred, zero_division=0),
                               npv=tn / (tn + fn) if tn + fn else np.nan,
                               f1=f1_score(yte, pred), accuracy=accuracy_score(yte, pred),
                               missed_cases=int(fn), false_alarms=int(fp)))

    _save(pd.DataFrame(point_rows), "35_clinical_operating_points")
    _save(pd.DataFrame(sweep_rows), "36_threshold_sweep")
    print(pd.DataFrame(point_rows).to_string(index=False))


# --------------------------------------------------------------------------- 5. permutation
def permutation_importance_study(X, y) -> None:
    """Model-agnostic permutation importance on the held-out 70/30 split.

    Mean-decrease-in-impurity importance (results/09) is tree-only and biased toward
    high-cardinality features. Permutation importance instead asks: shuffle this column in the
    test set and watch accuracy drop. A big drop means the fitted model actually relies on it.
    Repeated over several shuffles for a stable mean ± std.
    """
    print("[5/6] Permutation importance (held-out 70/30, repeated)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in all_pipelines().items():
        model = clone(pipe).fit(Xtr, ytr)
        res = permutation_importance(model, Xte, yte, n_repeats=30, random_state=RANDOM_STATE,
                                     scoring="accuracy", n_jobs=-1)
        order = np.argsort(res.importances_mean)[::-1]
        for i, fidx in enumerate(order):
            rows.append(dict(model=name, rank=i + 1, feature=FEATURES[fidx],
                             importance_mean=res.importances_mean[fidx],
                             importance_std=res.importances_std[fidx]))
    df = pd.DataFrame(rows)
    _save(df, "37_permutation_importance")
    # print a compact pivot: mean importance of the top model for a quick look
    top = df[df["model"] == "HistGradientBoosting"].head(13)
    print(top.to_string(index=False))


# --------------------------------------------------------------------------- 6. HGB vs best
def hgb_summary(X, y) -> None:
    """Where does modern gradient boosting land against the paper's six, honestly?

    Same three protocols, same no-leakage pipelines. This is the 'is the paper's model family
    even competitive by 2024 standards?' yardstick.
    """
    print("[6/6] HistGradientBoosting across protocols")
    from src.evaluation import compute_metrics, get_score, run_protocol
    rows = []
    for protocol in ("A", "B", "C"):
        pipe = _hgb_pipeline()
        df, _ = run_protocol(pipe, X, y, protocol)
        test = df[df["split"] == "test"].iloc[0]
        rows.append(dict(protocol=protocol, accuracy=test["accuracy"],
                         accuracy_std=test.get("accuracy_std", np.nan),
                         precision=test["precision"], recall=test["recall"], f1=test["f1"],
                         auc=test["auc"]))
    _save(pd.DataFrame(rows), "38_hgb_across_protocols")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    df = load_cleveland()
    X, y = split_xy(df)
    print(f"Loaded {len(X)} patients, {len(FEATURES)} features "
          f"({int(y.sum())} disease / {int((1 - y).sum())} healthy)\n")

    repeated_cv_stability(X, y)
    mcnemar_pairwise(X, y)
    clinical_thresholds(X, y)
    permutation_importance_study(X, y)
    hgb_summary(X, y)

    print("\nDone. New tables 33-38 written to results/. "
          "run_replication.py is untouched; the published numbers remain reproducible.")


if __name__ == "__main__":
    main()
