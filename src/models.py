"""Preprocessing + the six models from the paper, all as scikit-learn Pipelines.

Pipeline = [feature selector] -> [imputer / scaler / one-hot] -> classifier.
Because everything is inside one Pipeline, calling .fit() on the training split fits the
selector, imputer and scaler on training data only (no leakage).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import BINARY, CATEGORICAL, NUMERIC, PAPER_7, RANDOM_STATE


# ----------------------------------------------------------------------------- preprocessing
def _cols(group):
    """Column selector that tolerates a feature subset (keeps only columns present)."""
    return lambda X: [c for c in group if c in X.columns]


def make_preprocessor(kind):
    """kind='linear': impute + scale numerics + one-hot nominals  (KNN, SVM, LR, isolation forest)
       kind='tree'  : impute only, keep integer codes             (DT, RF, NB)
    Numeric -> median; binary and categorical -> most frequent (mode)."""
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    cat_steps = [("imp", SimpleImputer(strategy="most_frequent"))]
    if kind == "linear":
        num_steps.append(("scale", StandardScaler()))            # fitted on training data only
        cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), _cols(NUMERIC)),
        ("bin", Pipeline([("imp", SimpleImputer(strategy="most_frequent"))]), _cols(BINARY)),
        ("cat", Pipeline(cat_steps), _cols(CATEGORICAL)),
    ])


# ----------------------------------------------------------------------------- feature selection
def score_features(X, y, method):
    """Filter-method score for every column of X (higher = more related to the target).

    method='mi'   : mutual information (discrete_features flagged for binary/nominal columns)
    method='chi2' : chi-square on min-max scaled (non-negative) integer-coded data
    Missing values are imputed first (median numeric / mode otherwise).
    """
    Xi = make_preprocessor("tree").fit_transform(X)
    names = [n.split("__", 1)[1] for n in make_preprocessor("tree").fit(X).get_feature_names_out()]
    if method == "mi":
        discrete = np.array([n not in NUMERIC or n == "ca" for n in names])
        s = mutual_info_classif(Xi, y, discrete_features=discrete, random_state=RANDOM_STATE)
    elif method == "chi2":
        s, _ = chi2(MinMaxScaler().fit_transform(Xi), y)
    else:
        raise ValueError(method)
    return pd.Series(s, index=names).sort_values(ascending=False)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """mode='all' keeps everything; 'fixed' keeps `features`; 'mi'/'chi2' keeps the top-k scored
    on the training data passed to fit()."""

    def __init__(self, mode="all", features=None, k=7):
        self.mode, self.features, self.k = mode, features, k

    def fit(self, X, y=None):
        if self.mode == "all":
            self.selected_ = list(X.columns)
        elif self.mode == "fixed":
            self.selected_ = list(self.features)
        else:
            self.selected_ = list(score_features(X, y, self.mode).index[: self.k])
        return self

    def transform(self, X):
        return X[self.selected_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_)


def make_selector(name):
    return {"all13": FeatureSelector("all"),
            "paper7": FeatureSelector("fixed", features=PAPER_7),
            "MI7": FeatureSelector("mi", k=7)}[name]


# ----------------------------------------------------------------------------- models
# name -> (preprocessing kind, estimator class, default params)
SPECS = {
    "Decision Tree": ("tree", DecisionTreeClassifier, dict(criterion="entropy", random_state=RANDOM_STATE)),
    "Naive Bayes": ("tree", GaussianNB, {}),
    "Random Forest": ("tree", RandomForestClassifier, dict(random_state=RANDOM_STATE)),
    "KNN": ("linear", KNeighborsClassifier, dict(n_neighbors=8, metric="euclidean")),
    "SVM": ("linear", SVC, dict(kernel="linear", random_state=RANDOM_STATE)),
    "Logistic Regression": ("linear", LogisticRegression, dict(max_iter=1000, random_state=RANDOM_STATE)),
}
MAIN_MODELS = list(SPECS)

RF_GRID = {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 3, 5, 8],
           "clf__min_samples_leaf": [1, 3], "clf__max_features": ["sqrt", None]}


def build_pipeline(name, selector="all13", **params):
    """Pipeline for `name`. `params` override the estimator defaults, e.g. n_neighbors=5."""
    kind, cls, defaults = SPECS[name]
    sel = make_selector(selector) if isinstance(selector, str) else selector
    return Pipeline([("select", sel), ("prep", make_preprocessor(kind)),
                     ("clf", cls(**{**defaults, **params}))])


def build_tuned_rf(selector="all13"):
    """Random Forest whose hyper-parameters are chosen by 5-fold CV on the TRAINING data only
    (GridSearchCV is refit inside every outer split, so the test data is never used for tuning)."""
    return GridSearchCV(build_pipeline("Random Forest", selector), RF_GRID, cv=5,
                        scoring="accuracy", n_jobs=-1)


def clean_names(pipe):
    """Feature names after the preprocessing step, without the 'num__' style prefixes."""
    return [n.split("__", 1)[1] for n in pipe[:-1].get_feature_names_out()]
