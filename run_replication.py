"""
Replication of "Predictive Analysis of Heart Diseases with Machine Learning Approaches"
(Ramesh et al., Malaysian Journal of Computer Science, Special Issue 1/2022, pp. 132-148).

Run from the project folder:   .venv/bin/python run_replication.py      (about 8 minutes)

What it does: loads the UCI Cleveland data, preprocesses it, makes the EDA figures, runs the filter
feature selection, evaluates the six models under three protocols (A 60/40, B 70/30, C 10-fold CV),
compares with the paper's reported numbers, and checks the paper's known inconsistencies.
All tables go to results/*.csv and all figures to results/*.png; tables are also printed here.
Reusable code lives in src/ (data.py, models.py, evaluation.py). random_state=42 everywhere.
"""


# ====================================================================================================
# # Replication: *Predictive Analysis of Heart Diseases with Machine Learning Approaches*
# Ramesh et al., Malaysian Journal of Computer Science, Special Issue 1/2022, pp. 132-148.
#
# **Goal.** Re-run the paper's pipeline on the UCI Cleveland data (303 patients, 13 features) with six classifiers, and compare *honestly* with the numbers the paper reports. Nothing here is tuned to match the paper: every number below comes from a run of this script (`random_state=42` everywhere).
#
# **Reading guide for a viva.** The reusable code is in `src/`:
# * `src/data.py` - loading, feature groups
# * `src/models.py` - preprocessing, feature selection, the six models (as scikit-learn `Pipeline`s)
# * `src/evaluation.py` - protocols A / B / C, metrics, isolation-forest outlier removal
#
# A `Pipeline` bundles *imputer -> scaler -> classifier*. Calling `.fit` on the training split fits **all** of them on training data only, which is how we avoid data leakage.
# ====================================================================================================

import sys, warnings
import matplotlib
matplotlib.use("Agg")   # no pop-up windows: figures are saved to results/
from pathlib import Path
ROOT = Path(__file__).resolve().parent   # project folder (so `src` can be imported)
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import export_text, plot_tree

from src.data import *          # RANDOM_STATE, FEATURES, NUMERIC, BINARY, CATEGORICAL, PAPER_7, load_cleveland ...
from src.models import *        # build_pipeline, FeatureSelector, score_features, ...
from src.evaluation import *    # run_protocol, outlier_mask, METRICS ...

warnings.filterwarnings("ignore", category=FutureWarning)
RES = RESULTS_DIR; RES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
plt.rcParams["figure.dpi"] = 100
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

def save_table(df, name, index=True):
    df.to_csv(RES / f"{name}.csv", index=index); return df
def save_fig(name):
    plt.tight_layout(); plt.savefig(RES / f"{name}.png", dpi=150, bbox_inches="tight"); plt.close()

def display(*objs):
    """Terminal stand-in for the notebook's display(): print tables in full."""
    for o in objs:
        print(o.to_string() if hasattr(o, "to_string") else o); print()


# ====================================================================================================
# ## Step 1 - Load and verify the data
# **What:** read `processed.cleveland.data` (downloaded from the UCI archive), name the columns, turn `?` into `NaN`, and binarise the 0-4 `num` column into `target = (num > 0)` (1 = heart disease present).
# **Why:** the raw file has no header and uses `?` for missing values. We must check that what we loaded matches the paper's description (303 rows, ~6 missing, 165 positive / 138 negative) *before* modelling anything.
# ====================================================================================================

df = load_cleveland()
print("shape:", df.shape)
print("\ndtypes:"); display(df.dtypes.astype(str).to_frame("dtype").T)
miss = df.isna().sum(); print("missing values per column (only non-zero):"); display(miss[miss > 0].to_frame("n_missing").T)
print("total missing cells:", int(miss.sum()))
print("\noriginal 'num' (0 = no disease, 1-4 = disease severity):"); display(df["num"].value_counts().sort_index().to_frame("n").T)

counts = df["target"].value_counts()
check = pd.DataFrame({"paper": {"present (positive)": 165, "absent (negative)": 138},
                      "ours":  {"present (positive)": counts[1], "absent (negative)": counts[0]}})
check["difference"] = check["ours"] - check["paper"]
display(save_table(check, "01_class_balance_check"))
df.to_csv(DATA_DIR / "heart_raw.csv", index=False)   # raw (still contains NaN) + target
X, y = split_xy(df)


# ====================================================================================================
# **Finding.** Cleveland has **164 patients without** and **139 with** disease (`num > 0`). The paper states 165 positive / 138 negative. The numbers are almost the same but the labels look swapped (and off by one); we cannot tell whether that is a typo in the paper or a different labelling. Only the 6 missing values (4 in `ca`, 2 in `thal`) match expectations. We continue with the dataset as it is.
# ====================================================================================================


# ====================================================================================================
# ## Step 2 - Preprocessing
# **Imputation.** `ca` and `thal` have a few NaN. Numeric columns (`age, trestbps, chol, thalach, oldpeak, ca`) get the **median** (robust to outliers); binary/nominal columns (`sex, fbs, exang, cp, restecg, slope, thal`) get the **mode** (the median/mean of a category code is meaningless). `ca` is a count of vessels (0-3), so we treat it as numeric.
#
# **Encoding.** KNN, SVM and Logistic Regression get *one-hot* encoding of `cp, restecg, slope, thal` (otherwise they would treat code 4 as "twice" code 2) and *standardised* numeric columns (mean 0, sd 1 - essential for distance- and margin-based models). Tree models (DT, RF) and Naive Bayes keep the integer codes with no scaling (trees only compare against thresholds, so scaling changes nothing). *Choice of mine:* the paper does not say how Naive Bayes was fed; I use the integer-coded version.
#
# **No leakage.** The imputer and scaler live inside the `Pipeline`, so they are fitted on the training split only. The cell below proves it: the scaled training columns have mean exactly 0, the test columns do not.
# ====================================================================================================

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
prep = make_preprocessor("linear").fit(Xtr)               # fit on TRAIN only
print("learned medians (train):", dict(zip(NUMERIC, prep.named_transformers_["num"]["imp"].statistics_.round(2))))
print("learned modes   (train):", dict(zip(CATEGORICAL, prep.named_transformers_["cat"]["imp"].statistics_)))
Ztr, Zte = prep.transform(Xtr), prep.transform(Xte)
Ztr = Ztr.toarray() if hasattr(Ztr, "toarray") else Ztr; Zte = Zte.toarray() if hasattr(Zte, "toarray") else Zte
print("\ntransformed shape (train):", Ztr.shape, "<- 13 raw features became", Ztr.shape[1], "columns after one-hot")
print("mean of the 6 scaled numeric columns, TRAIN:", Ztr[:, :6].mean(0).round(3))
print("mean of the 6 scaled numeric columns, TEST :", Zte[:, :6].mean(0).round(3), "<- not 0: the test set did not influence the scaler")


# ====================================================================================================
# ### Anomaly removal with Isolation Forest
# The paper removes anomalies with `IsolationForest` but gives no `contamination` (the fraction of rows to flag). I therefore run **three variants** and report them all: none, `contamination=0.05`, `contamination=0.10`.
#
# *Where it is applied.* My main variant removes outliers **from the training split only** (the test set is never altered, so it stays comparable across variants). Because the paper most likely filtered the whole dataset first, I additionally run a *whole-dataset* variant in Step 6. Note that this changes the test set and therefore the numbers are not directly comparable. Isolation Forest is unsupervised (it never sees `target`), so filtering first does not leak labels, but it does change the evaluation population.
# ====================================================================================================

rows = []
for c in [None, 0.05, 0.10]:
    keep = outlier_mask(Xtr, c)
    removed = ytr[~keep]
    rows.append({"contamination": c, "train rows": len(Xtr), "removed": int((~keep).sum()),
                 "removed with disease": int(removed.sum()), "removed without disease": int((removed == 0).sum())})
display(save_table(pd.DataFrame(rows), "02_isolation_forest_removed_rows", index=False))

# 'Cleaned' CSV for inspection / EDA ONLY: whole-dataset median/mode imputation (models do NOT use this file).
clean = impute_whole_dataset(df); clean.to_csv(DATA_DIR / "heart_clean.csv", index=False)
print("remaining NaN in heart_clean.csv:", int(clean.isna().sum().sum()))


# ====================================================================================================
# ## Step 3 - Exploratory data analysis (the paper's figures)
# EDA is descriptive, so here I use the whole-dataset-imputed copy (`heart_clean.csv`). Fig. 8 is not requested; the exact contents of Figs. 9-11 are not specified in the task, so I split the 13 features into three groups (continuous / binary-nominal / ordinal) and plot each by target.
# ====================================================================================================

clean = pd.read_csv(DATA_DIR / "heart_clean.csv")
LAB = {0: "No disease", 1: "Disease"}

# Fig. 2 - correlation of each feature with the target (Pearson; for binary/ordinal codes this is a rough guide only)
corr = clean[FEATURES + ["target"]].corr()["target"].drop("target").sort_values()
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.barh(corr.index, corr.values, color=["#c0504d" if v < 0 else "#4f81bd" for v in corr.values])
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Pearson correlation with target"); ax.set_title("Fig. 2 - feature vs target correlation")
save_fig("fig02_correlation_with_target"); save_table(corr.to_frame("corr_with_target"), "03_correlation_with_target")

# Fig. 3 - class distribution
fig, ax = plt.subplots(figsize=(4.5, 3.5))
cnt = clean["target"].map(LAB).value_counts().reindex(["No disease", "Disease"])
ax.bar(cnt.index, cnt.values, color=["#4f81bd", "#c0504d"])
for i, v in enumerate(cnt.values): ax.text(i, v + 2, str(v), ha="center")
ax.set_title("Fig. 3 - class distribution"); ax.set_ylabel("patients")
save_fig("fig03_class_distribution")

# Figs. 4-6 - disease by sex, chest-pain type, fasting blood sugar
spec = [("sex", "Fig. 4 - by sex", {0: "female", 1: "male"}),
        ("cp", "Fig. 5 - by chest-pain type", {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"}),
        ("fbs", "Fig. 6 - by fasting blood sugar", {0: "<=120 mg/dl", 1: ">120 mg/dl"})]
for i, (col, title, names) in enumerate(spec, start=4):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    d = clean.assign(**{col: clean[col].map(names), "Target": clean["target"].map(LAB)})
    sns.countplot(data=d, x=col, hue="Target", order=list(names.values()), hue_order=["No disease", "Disease"], ax=ax)
    ax.set_title(title); ax.set_xlabel(col)
    save_fig(f"fig{i:02d}_{col}_vs_target")

# Fig. 7 - df.describe()
desc = clean[FEATURES + ["target"]].describe().T.round(2)
display(save_table(desc, "04_describe"))
fig, ax = plt.subplots(figsize=(9, 4.5)); ax.axis("off")
t = ax.table(cellText=desc.values, rowLabels=desc.index, colLabels=desc.columns, loc="center"); t.scale(1, 1.25)
ax.set_title("Fig. 7 - df.describe()"); save_fig("fig07_describe")

# Figs. 9-11 - feature distributions split by target
groups = {"fig09_continuous_features": ("Fig. 9 - continuous features", ["age", "trestbps", "chol", "thalach", "oldpeak"], "hist"),
          "fig10_binary_nominal_features": ("Fig. 10 - binary / nominal features", ["sex", "fbs", "exang", "restecg"], "count"),
          "fig11_ordinal_features": ("Fig. 11 - cp / slope / ca / thal", ["cp", "slope", "ca", "thal"], "count")}
for fname, (title, cols, kind) in groups.items():
    n = len(cols); fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.3))
    for ax, c in zip(np.atleast_1d(axes), cols):
        d = clean.assign(Target=clean["target"].map(LAB))
        if kind == "hist": sns.histplot(data=d, x=c, hue="Target", kde=True, element="step", stat="density", common_norm=False, ax=ax, legend=(c == cols[0]))
        else: sns.countplot(data=d, x=c, hue="Target", ax=ax, legend=(c == cols[0]))
    fig.suptitle(title, y=1.03); save_fig(fname)


# ====================================================================================================
# ## Step 4 - Feature selection (filter method)
# A *filter* method scores each feature on its own, independently of any classifier, then keeps the top *k*.
# * **Mutual information (MI):** how much knowing the feature reduces uncertainty about the target (any dependency, not only linear).
# * **Chi-square:** tests independence between a non-negative feature and the class. Needs non-negative inputs, so I min-max scale the integer-coded features; applying it to continuous columns and nominal codes is crude but is what a generic filter does.
#
# The paper selects 7 features (its Table 7): `age, sex, cp, fbs, chol, slope, ca` and does not say which filter or what parameters. Below I rank on the **whole** dataset *only to compare lists* (descriptive; this ranking is never used to evaluate a model). When models are evaluated with "my" 7 features (`MI7`), the selection is redone **inside each training split** (it is part of the `Pipeline`). I also check how stable the MI top-7 is across the 10 CV folds.
# ====================================================================================================

mi, chi = score_features(X, y, "mi"), score_features(X, y, "chi2")
tab = pd.DataFrame({"MI": mi, "MI_rank": mi.rank(ascending=False).astype(int),
                    "chi2": chi, "chi2_rank": chi.rank(ascending=False).astype(int)}).sort_values("MI_rank")
tab["in_paper_top7"] = tab.index.isin(PAPER_7)

# stability: how often is each feature in the MI top-7 across the 10 training folds?
skf = StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE)
freq = pd.Series(0, index=FEATURES)
for tr, _ in skf.split(X, y):
    freq[score_features(X.iloc[tr], y.iloc[tr], "mi").index[:7]] += 1
tab["MI_top7_in_folds(of 10)"] = freq
display(save_table(tab.round(4), "05_feature_selection_scores"))

top = {"paper": set(PAPER_7), "MI top7": set(mi.index[:7]), "chi2 top7": set(chi.index[:7])}
for k, v in top.items(): print(f"{k:10s}: {sorted(v)}")
print("\noverlap paper vs MI top7  :", sorted(top['paper'] & top['MI top7']), f"({len(top['paper'] & top['MI top7'])}/7)")
print("overlap paper vs chi2 top7:", sorted(top['paper'] & top['chi2 top7']), f"({len(top['paper'] & top['chi2 top7'])}/7)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, s, nm in zip(axes, [mi, chi], ["Mutual information", "Chi-square"]):
    s = s.sort_values()
    ax.barh(s.index, s.values, color=["#c0504d" if f in PAPER_7 else "#9aa5b1" for f in s.index]); ax.set_title(nm)
fig.suptitle("Filter scores (red = in the paper's 7 features)", y=1.02); save_fig("fig_feature_selection_scores")


# ====================================================================================================
# ## Step 5 - The six models
# Before the big comparison (Step 6), each model gets a short exploration of the hyper-parameter the paper discusses. Helper `acc_row` gives the **test accuracy** under the three protocols: A (60/40 split), B (70/30 split) and C (10-fold CV, mean and std). Keep in mind that a 91- or 122-patient test set is small: one patient is ~1 percentage point and the standard error of an accuracy near 85% is about 3-4 points. Differences of 1-3 points between models are **not** statistically meaningful.
#
# *A caveat on "sweeps".* Choosing the best *k* or kernel by looking at test accuracy (as the paper's presentation implies) is **selection on the test set** and makes the winner look better than it is. I show those curves because the paper does, but I also show a legitimate alternative: choose by cross-validation on the training split only.
# ====================================================================================================

def acc_row(pipe):
    # test accuracy under protocols A, B, C (C = mean and std over 10 folds)
    out = {}
    for p in "ABC":
        r, _ = run_protocol(pipe, X, y, p); t = r[r["split"] == "test"].iloc[0]
        out[f"acc_{p}"] = t["accuracy"]
        if p == "C": out["acc_C_std"] = t["accuracy_std"]
    return out

# the 70/30 split used for the 'look inside the model' demos below (identical to protocol B)
XtrB, XteB, ytrB, yteB = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)


# ====================================================================================================
# ### 5.1 Decision Tree - information gain (entropy) vs Gini, and number of features
# **Idea.** A tree repeatedly asks the yes/no question about one feature that makes the two child groups *purer*. Purity is measured by **entropy** $H=-\sum_c p_c\log_2 p_c$ (the split score is the *information gain* = parent entropy minus the size-weighted child entropies) or by **Gini** $G=1-\sum_c p_c^2$. Both are 0 for a pure node. They almost always pick the same splits.
#
# **Number of features.** The paper says its tree works with "4 or 18 features". The dataset only has 13 features, so "18" is impossible (even after one-hot there are 22 columns, but the tree there still uses the 13 original ones). I sweep the number of kept features k = 1..13 (top-k by mutual information, selected inside each training split) for both criteria.
# ====================================================================================================

# criterion comparison with all 13 features
crit = pd.DataFrame({c: acc_row(build_pipeline("Decision Tree", criterion=c)) for c in ["entropy", "gini"]}).T
display(save_table(crit.round(3), "07_dt_criterion_comparison"))

# number-of-features sweep (selection by mutual information, refit inside every split)
sweep = []
for c in ["entropy", "gini"]:
    for k in range(1, 14):
        sweep.append({"criterion": c, "k_features": k, **acc_row(build_pipeline("Decision Tree", FeatureSelector("mi", k=k), criterion=c))})
sweep = pd.DataFrame(sweep); save_table(sweep.round(3), "08_dt_feature_count_sweep", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, c in zip(axes, ["entropy", "gini"]):
    s = sweep[sweep.criterion == c]
    ax.plot(s.k_features, s.acc_C, "o-", label="10-fold CV (mean)"); ax.fill_between(s.k_features, s.acc_C - s.acc_C_std, s.acc_C + s.acc_C_std, alpha=.2)
    ax.plot(s.k_features, s.acc_A, "s--", label="test, 60/40"); ax.plot(s.k_features, s.acc_B, "^--", label="test, 70/30")
    ax.set_title(f"Decision tree ({c})"); ax.set_xlabel("number of features kept"); ax.set_xticks(range(1, 14))
axes[0].set_ylabel("test accuracy"); axes[0].legend(); save_fig("fig_dt_feature_count_sweep")
print("Best k by 10-fold CV:", sweep.loc[sweep.groupby("criterion").acc_C.idxmax(), ["criterion", "k_features", "acc_C"]].round(3).to_dict("records"))

# Look inside the tree (fitted on the 70/30 training part). Display is limited to depth 3; the real tree is deeper.
dt = build_pipeline("Decision Tree").fit(XtrB, ytrB); names = clean_names(dt); tree = dt[-1]
print("depth of the fully grown tree:", tree.get_depth(), "| leaves:", tree.get_n_leaves(), "| train accuracy:", round(dt.score(XtrB, ytrB), 3))
print(export_text(tree, feature_names=names, max_depth=3))
fig, ax = plt.subplots(figsize=(16, 7))
plot_tree(tree, feature_names=names, class_names=["absent", "present"], max_depth=3, filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("Decision tree (entropy), first 3 levels"); save_fig("fig_dt_tree_structure")

# Worked information-gain example for the ROOT split, computed by hand and compared with sklearn
def entropy(v):
    p = np.bincount(v, minlength=2) / len(v); p = p[p > 0]; return -(p * np.log2(p)).sum()
Ztr = dt[:-1].transform(XtrB); f, thr = tree.tree_.feature[0], tree.tree_.threshold[0]
left = Ztr[:, f] <= thr; yl, yr, yy = ytrB.values[left], ytrB.values[~left], ytrB.values
gain = entropy(yy) - (len(yl) / len(yy) * entropy(yl) + len(yr) / len(yy) * entropy(yr))
print(f"Root question: {names[f]} <= {thr:.1f}")
print(f"  parent entropy = {entropy(yy):.4f}  (sklearn: {tree.tree_.impurity[0]:.4f})")
print(f"  left  child: n={len(yl)}, entropy={entropy(yl):.4f}   right child: n={len(yr)}, entropy={entropy(yr):.4f}")
print(f"  information gain = {gain:.4f} bits")


# ====================================================================================================
# ### 5.2 Naive Bayes
# `GaussianNB` assumes the features are independent given the class and models each as a Gaussian per class; prediction = highest posterior $P(\text{class})\prod_j P(x_j|\text{class})$. It has no tuning knob that matters here (only `var_smoothing`), so it is evaluated as-is in Step 6.
#
# ### 5.3 Random Forest - default vs a small tuned grid, and feature importances
# The tuned version searches `n_estimators in {100,300}`, `max_depth in {None,3,5,8}`, `min_samples_leaf in {1,3}`, `max_features in {sqrt, all}` with 5-fold CV **on the training data only** (a `GridSearchCV` that is re-run inside every outer split, so the test data never influences the choice).
# ====================================================================================================

rf = build_pipeline("Random Forest").fit(XtrB, ytrB)
imp = pd.Series(rf[-1].feature_importances_, index=clean_names(rf)).sort_values()
fig, ax = plt.subplots(figsize=(6, 4.5)); ax.barh(imp.index, imp.values, color="#4f81bd")
ax.set_xlabel("impurity-based importance"); ax.set_title("Random Forest feature importances (70/30 train part)"); save_fig("fig_rf_feature_importance")
save_table(imp.sort_values(ascending=False).to_frame("importance"), "09_rf_feature_importance")

tuned = build_tuned_rf().fit(XtrB, ytrB)
print("best params (70/30 train part):", {k.split("__")[1]: v for k, v in tuned.best_params_.items()}, "| inner 5-fold CV acc:", round(tuned.best_score_, 3))
rf_cmp = pd.DataFrame({"default": acc_row(build_pipeline("Random Forest")), "tuned": acc_row(build_tuned_rf())}).T
display(save_table(rf_cmp.round(3), "10_rf_default_vs_tuned"))
print("Note: impurity-based importances favour high-cardinality/continuous features; treat as a rough guide.")


# ====================================================================================================
# ### 5.4 KNN - k = 1..20 and the distance metric
# **Idea.** Classify a patient by majority vote of the *k* nearest training patients. "Nearest" is measured by **Euclidean** distance $\sqrt{\sum(a_i-b_i)^2}$ or **Manhattan** distance $\sum|a_i-b_i|$ (sum of absolute differences, less dominated by one big gap). Features are standardised first, otherwise `chol` (hundreds) would swamp `oldpeak` (0-6). Small *k* = flexible/noisy, large *k* = smooth/biased. The paper reports k = 8 as best.
#
# Three curves: test accuracy under A and B (selection-on-test, as the paper does) and 10-fold CV accuracy over all 303 patients. Additionally, **k chosen by 10-fold CV inside the 70/30 training part only** is evaluated on the untouched 70/30 test part.
# ====================================================================================================

rows = []
for metric in ["euclidean", "manhattan"]:
    for k in range(1, 21):
        pipe = build_pipeline("KNN", n_neighbors=k, metric=metric)
        rows.append({"metric": metric, "k": k, **acc_row(pipe),
                     "innerCV_trainB": cross_val_score(pipe, XtrB, ytrB, cv=StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE)).mean()})
knn = pd.DataFrame(rows); save_table(knn.round(4), "11_knn_k_sweep", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, col, ttl in zip(axes, ["acc_A", "acc_B", "acc_C"], ["test acc, 60/40 split", "test acc, 70/30 split", "10-fold CV acc (all data)"]):
    for m, g in knn.groupby("metric"): ax.plot(g.k, g[col], "o-", label=m, ms=4)
    ax.axvline(8, color="grey", ls=":", label="paper's k=8"); ax.set_title(ttl); ax.set_xlabel("k"); ax.set_xticks(range(1, 21, 2))
axes[0].set_ylabel("accuracy"); axes[0].legend(); save_fig("fig_knn_k_sweep")

best = []
for m, g in knn.groupby("metric"):
    sel = g.loc[g.innerCV_trainB.idxmax()]
    best.append({"metric": m, "best k (test A)": int(g.loc[g.acc_A.idxmax(), "k"]), "best k (test B)": int(g.loc[g.acc_B.idxmax(), "k"]),
                 "best k (10-fold CV)": int(g.loc[g.acc_C.idxmax(), "k"]), "k chosen by inner CV on train-B": int(sel.k),
                 "its test-B acc": round(sel.acc_B, 3), "acc at k=8: A / B / CV": f"{g[g.k==8].acc_A.iloc[0]:.3f} / {g[g.k==8].acc_B.iloc[0]:.3f} / {g[g.k==8].acc_C.iloc[0]:.3f}"})
knn_best = save_table(pd.DataFrame(best), "12_knn_best_k_summary", index=False); display(knn_best)


# ====================================================================================================
# ### 5.5 SVM - kernels
# **Idea.** An SVM finds the separating boundary with the widest margin. The **kernel** defines the shape of the boundary: `linear` = a flat hyperplane; `poly` = polynomial curves (degree 3); `rbf` = smooth local bumps around points (gamma='scale'); `sigmoid` = tanh-shaped, often poorly behaved. All use C=1 and standardised inputs. The paper reports linear as best.
# ====================================================================================================

svm = pd.DataFrame({k: acc_row(build_pipeline("SVM", kernel=k)) for k in ["linear", "poly", "rbf", "sigmoid"]}).T
display(save_table(svm.round(3), "13_svm_kernel_comparison"))


# ====================================================================================================
# ### 5.6 Logistic Regression
# Models $P(\text{disease})=\sigma(w\cdot x+b)$, a linear decision boundary with probabilistic output. Because inputs are standardised, each coefficient shows how strongly a feature pushes the log-odds (positive = towards disease). `exp(coef)` is the odds ratio per +1 sd (or for being in a one-hot category). Default L2 regularisation, C=1.
# ====================================================================================================

lr = build_pipeline("Logistic Regression").fit(XtrB, ytrB)
coef = pd.Series(lr[-1].coef_[0], index=clean_names(lr)).sort_values()
display(save_table(pd.DataFrame({"coef": coef, "odds_ratio": np.exp(coef)}).round(3).iloc[::-1], "14_logreg_coefficients"))


# ====================================================================================================
# ## Step 6 - Evaluation: protocols A, B and C
# The paper's text says a 60/40 split, but its confusion matrices add up to 212 train + 91 test = 303, which is a 70/30 split. Since this is ambiguous I run **all three**:
# * **A** - 60/40 stratified split (`random_state=42`)
# * **B** - 70/30 stratified split (`random_state=42`)
# * **C** - 10-fold stratified CV (reported as mean ± std over folds)
#
# For every model: accuracy, precision, recall (sensitivity), F1, ROC-AUC (positive class = disease present), and the confusion matrix on **train and test**. For C the confusion matrix is summed over the 10 test folds. Everything is run for (a) all 13 features, (b) the paper's 7 features (`paper7`), (c) my own MI top-7 chosen inside each training split (`MI7`). I add the tuned Random Forest as a 7th row.
# ====================================================================================================

def run_grid(models, selectors, label, contamination=None, scope="train", protocols="ABC"):
    # scope='train': isolation forest filters the training split only. scope='full': filter all 303 rows BEFORE splitting (paper-style).
    Xin, yin = X, y
    if scope == "full" and contamination:
        m = outlier_mask(X, contamination); Xin, yin = X[m], y[m]
    out = []
    for sel in selectors:
        for name in models:
            pipe = build_tuned_rf(sel) if name == "Random Forest (tuned)" else build_pipeline(name, sel)
            for p in protocols:
                r, _ = run_protocol(pipe, Xin, yin, p, contamination if scope == "train" else None)
                r.insert(0, "model", name); r.insert(1, "features", sel); r.insert(2, "protocol", p); r.insert(3, "outlier", label)
                out.append(r)
    return pd.concat(out, ignore_index=True)

ALL_MODELS = MAIN_MODELS + ["Random Forest (tuned)"]
main = run_grid(ALL_MODELS, ["all13", "paper7", "MI7"], "none")
save_table(main, "15_results_main_long", index=False)
print(main.shape)

def pretty(res, features="all13", outlier="none", split="test"):
    # one table per protocol: A/B plain numbers, C as mean ± std
    d = res[(res.features == features) & (res.outlier == outlier) & (res["split"] == split)]
    tabs = {}
    for p in "ABC":
        t = d[d.protocol == p].set_index("model")
        if p == "C":
            tabs[p] = pd.DataFrame({m: t[m].map("{:.3f}".format) + " ± " + t[m + "_std"].map("{:.3f}".format) for m in METRICS})
        else:
            tabs[p] = t[METRICS].round(3)
    return tabs

for p, t in pretty(main).items():
    print(f"\n=== TEST metrics, all 13 features, protocol {p} ({PROTOCOLS[p]}) ==="); display(t)
    t.to_csv(RES / f"16_test_metrics_all13_protocol{p}.csv")

# Overfitting check: train vs test accuracy
d = main[(main.features == "all13")].pivot_table(index="model", columns=["protocol", "split"], values="accuracy").round(3)
gap = pd.DataFrame({f"gap {p} (train-test)": d[(p, "train")] - d[(p, "test")] for p in "ABC"}).round(3)
display(save_table(pd.concat([d, gap], axis=1), "17_train_vs_test_accuracy"))

# Confusion matrices (protocol B, all 13 features): counts as TP FN / FP TN
cm = main[(main.features == "all13") & (main.protocol == "B")][["model", "split", "tp", "fn", "fp", "tn"]]
display(save_table(cm, "18_confusion_matrices_protocolB_all13", index=False))

# Does feature selection help? mean test accuracy by feature set
fs = main[(main["split"] == "test")].pivot_table(index="model", columns=["protocol", "features"], values="accuracy").round(3)
display(save_table(fs, "19_feature_set_comparison_accuracy"))
print("mean over models (excluding tuned RF):"); display(fs.loc[MAIN_MODELS].mean().round(3).unstack())

# Sampling noise: 95% binomial interval for the test accuracy of protocol B (n=91) and A (n=122)
sub = main[(main.features == "all13") & (main["split"] == "test") & main.protocol.isin(["A", "B"])].copy()
yteA = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)[3]
sub["n_test"] = sub.protocol.map({"A": len(yteA), "B": len(yteB)}); sub["95% CI half-width"] = 1.96 * np.sqrt(sub.accuracy * (1 - sub.accuracy) / sub.n_test)
display(save_table(sub[["model", "protocol", "n_test", "accuracy", "95% CI half-width"]].round(3), "20_test_accuracy_sampling_noise", index=False))

# ROC curves, all six models on one chart per protocol (all 13 features). C = pooled out-of-fold scores.
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, p in zip(axes, "ABC"):
    for name in MAIN_MODELS:
        pipe = build_pipeline(name)
        if p in "AB":
            _, ex = run_protocol(pipe, X, y, p); yt, sc = ex["yte"], ex["score_test"]
        else:
            meth = "decision_function" if hasattr(pipe, "decision_function") else "predict_proba"
            sc = cross_val_predict(pipe, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE), method=meth)
            sc = sc[:, 1] if sc.ndim == 2 else sc; yt = y
        fpr, tpr, _ = roc_curve(yt, sc); ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yt, sc):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_title(f"Protocol {p}: {PROTOCOLS[p]}" + (" (pooled out-of-fold)" if p == "C" else " (test set)"))
    ax.set_xlabel("false positive rate"); ax.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel("true positive rate (recall)"); save_fig("fig_roc_curves_all_models")


# ====================================================================================================
# ### Isolation Forest variants (Step 2 follow-up)
# All six models, all 13 features, all three protocols, under: no filtering; filtering of the **training split only** at contamination 0.05 / 0.10; and filtering of the **whole dataset before splitting** (paper-style; the test set then also changes, so compare with care).
# ====================================================================================================

variants = [("none", None, "train"), ("IF 0.05 (train only)", 0.05, "train"), ("IF 0.10 (train only)", 0.10, "train"),
            ("IF 0.05 (whole data)", 0.05, "full"), ("IF 0.10 (whole data)", 0.10, "full")]
iso = pd.concat([main[(main.features == "all13") & (main.model.isin(MAIN_MODELS))]] +
                [run_grid(MAIN_MODELS, ["all13"], lab, c, sc) for lab, c, sc in variants[1:]], ignore_index=True)
save_table(iso, "21_results_isolation_forest_long", index=False)
tt = iso[iso["split"] == "test"].pivot_table(index="model", columns=["protocol", "outlier"], values="accuracy").round(3)
tt = tt.reindex(columns=[(p, v[0]) for p in "ABC" for v in variants])
display(save_table(tt, "22_isolation_forest_test_accuracy"))
print("rows kept in whole-data filtering:", {lab: int(outlier_mask(X, c).sum()) for lab, c, sc in variants[1:] if sc == "full"})


# ====================================================================================================
# ## Step 7 - Comparison with the paper
# Paper's Table 3 versus my **test** results (all 13 features, no outlier removal, default hyper-parameters; KNN k=8, SVM linear, entropy tree). The three "ours" columns are the three protocols; for C the std over folds is also given. Differences are `ours(B) - paper`.
# ====================================================================================================

paper3 = pd.DataFrame({
    "accuracy":  [0.7898, 0.8678, 0.879, 0.941, 0.891, 0.871],
    "recall":    [0.7548, 0.804, 0.832, 0.948, 0.878, 0.861],
    "precision": [0.778, 0.854, 0.887, 0.917, 0.907, 0.881],
    "f1":        [0.864, 0.865, 0.879, 0.908, 0.885, 0.865],
    "auc":       [0.812, 0.845, 0.898, 0.799, 0.809, 0.813]},
    index=["Decision Tree", "Naive Bayes", "Random Forest", "KNN", "SVM", "Logistic Regression"])
paper3.index.name = "model"

def compare(features):
    t = main[(main.features == features) & (main["split"] == "test") & main.model.isin(MAIN_MODELS)]
    rows = []
    for m in paper3.index:
        for met in METRICS:
            g = lambda p: t[(t.model == m) & (t.protocol == p)].iloc[0]
            rows.append({"model": m, "metric": met, "paper": paper3.loc[m, met], "ours A (60/40)": g("A")[met], "ours B (70/30)": g("B")[met],
                         "ours C mean": g("C")[met], "ours C std": g("C")[met + "_std"], "B - paper": g("B")[met] - paper3.loc[m, met]})
    return pd.DataFrame(rows)

cmp13, cmp7 = compare("all13"), compare("paper7")
save_table(cmp13.round(4), "23_paper_vs_ours_all13", index=False); save_table(cmp7.round(4), "24_paper_vs_ours_paper7features", index=False)
print("ALL 13 FEATURES"); display(cmp13.pivot(index="model", columns="metric", values="B - paper").round(3).rename_axis(columns="B - paper"))
display(cmp13.round(3).set_index(["model", "metric"]))

# Ranking of the six models by test accuracy (1 = best), and by AUC, under each protocol (all 13 features)
t = main[(main.features == "all13") & (main["split"] == "test") & main.model.isin(MAIN_MODELS)]
rk = pd.concat({f"{m} rank {p}": t[t.protocol == p].set_index("model")[m].rank(ascending=False, method="min").astype(int)
                for m in ["accuracy", "auc"] for p in "ABC"}, axis=1)
rk.insert(0, "paper acc rank", paper3.accuracy.rank(ascending=False, method="min").astype(int))
display(save_table(rk, "25_model_rankings"))


# ====================================================================================================
# ## Step 8 - Checking the paper's known problems
# Each item is verified with code below.
#
# ### 8.1 The split: text says 60/40, matrices imply 70/30
# ====================================================================================================

sizes = pd.DataFrame({
    "train": [len(train_test_split(X, y, test_size=ts, stratify=y, random_state=RANDOM_STATE)[0]) for ts in (0.4, 0.3)] + [212],
    "test":  [len(train_test_split(X, y, test_size=ts, stratify=y, random_state=RANDOM_STATE)[1]) for ts in (0.4, 0.3)] + [91]},
    index=["sklearn 60/40 (A)", "sklearn 70/30 (B)", "paper's confusion matrices"])
display(save_table(sizes, "26_split_sizes"))
print("-> 212/91 is exactly what a 70/30 split gives; a real 60/40 split would be 181/122.")


# ====================================================================================================
# ### 8.2 Paper's Table 6 (test confusion matrices) vs its own Table 3
# The task statement lists Table 6 as `TP/FP/FN/TN`. I test **two readings** of the four numbers: (i) as stated, TP/FP/FN/TN, and (ii) the usual matrix layout `[[TP, FN], [FP, TN]]` read row by row (TP/FN/FP/TN). All six models were supposedly evaluated on the *same* 91 test patients, so the number of truly-diseased patients (TP+FN) and truly-healthy patients (FP+TN) must be identical across models.
# ====================================================================================================

p6 = {"Logistic Regression": (34, 7, 5, 45), "SVM": (36, 5, 6, 44), "KNN": (35, 6, 6, 44),
      "Random Forest": (33, 8, 8, 42), "Decision Tree": (34, 7, 13, 37), "Naive Bayes": (33, 8, 8, 42)}   # as given in the task: a/b/c/d
def metrics_from(tp, fn, fp, tn):
    acc = (tp + tn) / (tp + fn + fp + tn); p = tp / (tp + fp); r = tp / (tp + fn)
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, actual_pos=tp + fn, actual_neg=fp + tn, accuracy=acc, precision=p, recall=r, f1=2 * p * r / (p + r))
rows = []
for m, (a, b, c, d) in p6.items():
    rows.append({"reading": "(i) as stated TP/FP/FN/TN", "model": m, **metrics_from(tp=a, fp=b, fn=c, tn=d)})
    rows.append({"reading": "(ii) TP/FN/FP/TN", "model": m, **metrics_from(tp=a, fn=b, fp=c, tn=d)})
t6 = pd.DataFrame(rows); save_table(t6.round(4), "27_paper_table6_recomputed", index=False)
display(t6.pivot(index="model", columns="reading", values=["actual_pos", "actual_neg"]))
print("Test class counts from my own stratified 70/30 split: ", yteB.value_counts().sort_index().to_dict(), "(0 = absent, 1 = present)")
print("Expected for a stratified 30% of the paper's claimed 165 pos / 138 neg: ~50 pos / ~41 neg.  Of the real 139 present / 164 absent: ~42 / ~49.")

# Table 3 accuracy vs the accuracy implied by Table 6 (accuracy only needs TP+TN, so it is the same under both readings)
imp = t6[t6.reading.str.startswith("(ii)")].set_index("model")
acc_cmp = pd.DataFrame({"Table 3 accuracy": paper3.accuracy, "Table 6 implied accuracy": imp.accuracy,
                        "Table 6 correct/91": (imp.TP + imp.TN).astype(int)})
acc_cmp["inflation (T3 - T6)"] = acc_cmp.iloc[:, 0] - acc_cmp.iloc[:, 1]
acc_cmp["my acc, B (70/30)"] = main[(main.features == "all13") & (main.protocol == "B") & (main["split"] == "test")].set_index("model").accuracy
display(save_table(acc_cmp.loc[paper3.index].round(4), "28_table3_vs_table6_accuracy"))

# Table 3 precision / recall vs Table 6 (reading ii)
pr = pd.DataFrame({"T3 precision": paper3.precision, "T6 precision": imp.precision, "T3 recall": paper3.recall, "T6 recall": imp.recall, "T3 F1": paper3.f1, "T6 F1": imp.f1})
display(save_table(pr.loc[paper3.index].round(3), "29_table3_vs_table6_prf"))

# Mine next to the paper's Table 6 (protocol B, my confusion matrices in the same TP/FN/FP/TN layout)
mine = main[(main.features == "all13") & (main.protocol == "B") & (main["split"] == "test") & main.model.isin(MAIN_MODELS)].set_index("model")
cmcmp = pd.DataFrame({"paper TP/FN/FP/TN": {m: "/".join(map(str, p6[m])) for m in p6},
                      "mine (B) TP/FN/FP/TN": {m: "/".join(str(int(mine.loc[m, c])) for c in ["tp", "fn", "fp", "tn"]) for m in p6}})
display(save_table(cmcmp.loc[paper3.index], "30_confusion_matrix_paper_vs_mine"))


# ====================================================================================================
# ### 8.3 Training confusion matrices: 100% accuracy = overfitting?
# The paper's training matrices show perfect accuracy for RF, DT and NB. If that holds for my models too, the models memorise the training set and the *training* score says nothing about generalisation; the train-test gap in `17_train_vs_test_accuracy` is the relevant figure.
# ====================================================================================================

tr = main[(main.features == "all13") & (main["split"] == "train") & main.model.isin(MAIN_MODELS)].pivot(index="model", columns="protocol", values="accuracy").round(3)
te = main[(main.features == "all13") & (main["split"] == "test") & main.model.isin(MAIN_MODELS)].pivot(index="model", columns="protocol", values="accuracy").round(3)
display(pd.concat({"TRAIN accuracy": tr, "TEST accuracy": te}, axis=1))
print("Models with 100% train accuracy in protocol B:", tr.index[tr["B"] == 1.0].tolist())


# ====================================================================================================
# ### 8.4 Is the paper's F1 consistent with its own precision and recall?
# F1 is the harmonic mean $F_1=2PR/(P+R)$, so it must lie between P and R. I recompute it from the paper's Table 3 P and R.
# ====================================================================================================

f1chk = paper3[["precision", "recall", "f1"]].rename(columns={"f1": "F1 reported"})
f1chk["F1 recomputed from P,R"] = 2 * f1chk.precision * f1chk.recall / (f1chk.precision + f1chk.recall)
f1chk["mismatch"] = f1chk["F1 reported"] - f1chk["F1 recomputed from P,R"]
f1chk["F1 outside [min(P,R), max(P,R)]?"] = (f1chk["F1 reported"] < f1chk[["precision", "recall"]].min(axis=1)) | (f1chk["F1 reported"] > f1chk[["precision", "recall"]].max(axis=1))
display(save_table(f1chk.round(4), "31_paper_f1_consistency"))


# ====================================================================================================
# ### 8.5 "Decision tree with 4 or 18 features" and other unspecified details
# ====================================================================================================

n_onehot = make_preprocessor("linear").fit_transform(X).shape[1]
print(f"Raw features: {len(FEATURES)}.  After one-hot encoding: {n_onehot} columns.  A tree on {len(FEATURES)} features cannot use 18 of them.")
print("(Even if one-hot columns were counted the total is", n_onehot, "- not 18 either.)")


# ====================================================================================================
# **Not specified by the paper (so my choices are assumptions):**
# * Isolation Forest: `contamination`, `n_estimators`, whether applied before or after splitting -> I ran 0.05 / 0.10 / none, train-only and whole-data.
# * Filter method for the 7 features: which score (MI? chi-square? ANOVA?) and which tie-breaking -> MI and chi-square compared in Step 4.
# * Train/test random seed and the split ratio (60/40 in text vs 70/30 implied) -> `random_state=42`, both ratios plus 10-fold CV.
# * Hyper-parameters of every model other than KNN's k=8 and SVM's linear kernel (C, tree depth, number of trees, ...), how Naive Bayes was fed, and what "positive" means in the 165/138 count.
# * The number of features in the Decision Tree experiment ("4 or 18").
# ====================================================================================================


# ====================================================================================================
# ## Step 9 - Evidence table for the paper's headline claims
# Computed from the results above. Interpretation and caveats are in `README.md`.
# ====================================================================================================

T = main[(main.features == "all13") & (main["split"] == "test") & main.model.isin(MAIN_MODELS)]
def best(p, met="accuracy"):
    s = T[T.protocol == p].set_index("model")[met]; return f"{s.idxmax()} ({s.max():.3f}); KNN={s['KNN']:.3f}, rank {int(s.rank(ascending=False)['KNN'])}/6"
eu = knn[knn.metric == "euclidean"]; ma = knn[knn.metric == "manhattan"]
claims = pd.DataFrame([
    ("KNN is the most accurate model", f"A: {best('A')} | B: {best('B')} | C: {best('C')}"),
    ("KNN has the highest recall", f"A: {best('A','recall')} | B: {best('B','recall')} | C: {best('C','recall')}"),
    ("RF has the highest AUC (0.898) and KNN the lowest (0.799)", f"A: {best('A','auc')} | B: {best('B','auc')} | C: {best('C','auc')}"),
    ("KNN best at k=8 (Euclidean)", f"best k Euclid. -> test A: {int(eu.loc[eu.acc_A.idxmax(),'k'])}, test B: {int(eu.loc[eu.acc_B.idxmax(),'k'])}, 10-fold CV: {int(eu.loc[eu.acc_C.idxmax(),'k'])}; acc at k=8 (10-fold CV): {eu[eu.k==8].acc_C.iloc[0]:.3f} vs best {eu.acc_C.max():.3f}"),
    ("Euclidean vs Manhattan (k=8, 10-fold CV acc)", f"euclidean {eu[eu.k==8].acc_C.iloc[0]:.3f} vs manhattan {ma[ma.k==8].acc_C.iloc[0]:.3f}; best-k CV acc: {eu.acc_C.max():.3f} vs {ma.acc_C.max():.3f}"),
    ("Linear kernel is the best SVM kernel", "best kernel -> " + ", ".join(f"{c.replace('acc_','')}: {svm[c].idxmax()} ({svm[c].max():.3f}; linear {svm.loc['linear', c]:.3f})" for c in ["acc_A", "acc_B", "acc_C"])),
    ("Entropy (information gain) tree vs Gini", f"acc A/B/C entropy {crit.loc['entropy','acc_A']:.3f}/{crit.loc['entropy','acc_B']:.3f}/{crit.loc['entropy','acc_C']:.3f}  gini {crit.loc['gini','acc_A']:.3f}/{crit.loc['gini','acc_B']:.3f}/{crit.loc['gini','acc_C']:.3f}"),
    ("RF/DT/NB reach 100% training accuracy", "train acc (B): " + ", ".join(f"{m} {tr.loc[m,'B']:.3f}" for m in ["Random Forest", "Decision Tree", "Naive Bayes"])),
    ("Selected 7 features = age, sex, cp, fbs, chol, slope, ca", f"overlap with MI top-7: {len(top['paper'] & top['MI top7'])}/7; with chi2 top-7: {len(top['paper'] & top['chi2 top7'])}/7"),
    ("Mean test accuracy over the 6 models, 13 vs paper's 7 features", "C: " + ", ".join(f"{f} {fs.loc[MAIN_MODELS, ('C', f)].mean():.3f}" for f in ["all13", "paper7", "MI7"])),
], columns=["paper claim", "evidence from my runs"])
pd.set_option("display.max_colwidth", 250); display(save_table(claims, "32_claims_evidence", index=False))
