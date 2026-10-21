# Replication: "Predictive Analysis of Heart Diseases with Machine Learning Approaches"

Ramesh et al., *Malaysian Journal of Computer Science*, Special Issue 1/2022, pp. 132-148 (DOI 10.22452/mjcs.sp2022no1.10).
UCI Cleveland data (303 patients, 13 features), six classifiers, `random_state=42` everywhere. Nothing was tuned to match the paper.
Every number below comes from `run_replication.py` (tables in `results/`, file names given in brackets).

## How to run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_replication.py   # ~8 min; prints all tables and writes results/
```
Data: `data/processed.cleveland.data` was downloaded from the UCI archive (`https://archive.ics.uci.edu/static/public/45/heart+disease.zip`).
`data/heart_raw.csv` (with NaN) and `data/heart_clean.csv` (whole-dataset imputation, used for EDA only) are written by the script.

```
src/data.py        loading, feature groups        src/models.py      preprocessing, feature selector, 6 pipelines
run_replication.py  main script (all steps, commented)
src/evaluation.py  protocols A/B/C, metrics, isolation forest        EXPLAINER.md  plain-language explanations
```

## Modern app (React + FastAPI)
On top of the untouched replication there is a live app — see `PLAN.md` for the full plan and status.

| Piece | Where | What it does |
|---|---|---|
| Backend | `backend/` | FastAPI: loads the six fitted pipelines once, serves inference, audit verdicts, and every `results/` table as JSON, including the Phase-3 stats tables (`/api/stats/{33..38}`). |
| Frontend | `frontend/` | React + Vite + TypeScript + Tailwind, dark "AI lab" glass UI: **Diagnosis** (live consensus prediction), **Benchmarks** (metrics/ROC/confusion matrices), **Audit** (paper claims ledger), **Hyperparameter Lab**, **EDA**, **Statistical Lens** (repeated CV, McNemar, permutation importance, thresholds). |
| Statistics | `run_extensions.py` | Tables 33–38: repeated 10×10 CV, exact McNemar pairwise tests, permutation importance, sensitivity/specificity trade-offs, HGB vs paper models. |
| Tests | `tests/` | 48 pytest checks: API contract, prediction stability, endpoint schemas, exact McNemar math. |

```bash
# backend (also what the frontend proxies to in dev)
.venv/bin/pip install -r requirements.txt
uvicorn backend.app:app --reload
# frontend
cd frontend && npm install && npm run dev   # http://localhost:5173
pytest tests/                               # 48 checks
npm run build                               # typecheck + production bundle
```

## Method in brief
* Target = `num > 0`. Imputation (median numeric / mode categorical), scaling, one-hot (KNN/SVM/LR) and feature selection all live **inside the sklearn Pipeline**, so they are fitted on training data only.
* Protocols: **A** 60/40 stratified, **B** 70/30 stratified, **C** 10-fold stratified CV (mean ± std). Metrics on train and test; positive class = disease.
* Isolation Forest (contamination unspecified in paper): none / 0.05 / 0.10, on the training split only and on the whole dataset before splitting.
* Feature sets: all 13, the paper's 7 (`paper7`), and my own mutual-information top-7 chosen inside each training split (`MI7`).
* Choices of mine that the paper does not specify: Naive Bayes uses the integer-coded features; default hyper-parameters except KNN k=8 and linear SVM (as in the paper); decision tree uses entropy.

## Headline results (all 13 features, no outlier removal, test data)
Accuracy [`16_test_metrics_*`, `20_test_accuracy_sampling_noise`]; paper = Table 3.

| Model | Paper | A (60/40, n=122) | B (70/30, n=91) | C (10-fold, mean ± std) |
|---|---|---|---|---|
| Decision Tree | 0.790 | 0.721 | 0.736 | 0.728 ± 0.078 |
| Naive Bayes | 0.868 | 0.885 | 0.879 | 0.838 ± 0.064 |
| Random Forest | 0.879 | 0.869 | 0.835 | 0.821 ± 0.064 |
| KNN (k=8) | 0.941 | 0.828 | 0.868 | 0.838 ± 0.064 |
| SVM (linear) | 0.891 | 0.844 | 0.824 | 0.835 ± 0.036 |
| Logistic Regression | 0.871 | 0.893 | 0.857 | 0.838 ± 0.058 |

**Noise matters.** The 95% binomial half-width on a single test accuracy is about ±5.5 to ±9 points (n=91/122). In 10-fold CV, NB, KNN and LR are tied to within 0.001 (0.8378 / 0.8382 / 0.8380) with std ≈ 0.06. The ranking of the top five models is **not statistically meaningful**; only the Decision Tree is clearly worse.

## Did the findings replicate?
| Paper claim | Verdict | Evidence |
|---|---|---|
| KNN (k=8) is the best model, accuracy 0.941 | **Not replicated** | KNN is 5th (A), 2nd (B), 1st (C, a three-way tie with NB and LR) [`25_model_rankings`]. Its accuracy is 0.83-0.87, not 0.94. Best model is LR (A), NB (B). |
| KNN has the highest recall (0.948) | **Not replicated** | KNN recall is 0.73-0.79; NB has the highest recall in A, B and C. |
| k=8 is the best k | **Not replicated** | Best Euclidean k is 5 (A), 6 (B), 5 (CV). k=8 is within about 0.01 of the best in CV (0.838 vs 0.845), i.e. it is a fine but not special choice. With Manhattan distance k=8 is best on A and B but not in CV (k=14) [`12_knn_best_k_summary`]. |
| Linear is the best SVM kernel | **Partly / inconclusive** | rbf wins on A (0.861 vs 0.844) and B (0.857 vs 0.824); linear wins in CV (0.835, all four kernels within 0.03) [`13_svm_kernel_comparison`]. |
| RF has the best AUC (0.898), KNN the lowest (0.799) | **Not replicated** | AUCs are 0.90-0.94 for all models except the tree (0.72-0.74). KNN AUC is 0.89-0.91 (not 0.799); LR/NB are top, RF 2nd-3rd. |
| Decision tree is the weakest model | **Replicated** | DT is last in every protocol (0.72-0.74); ROC AUC 0.72-0.74. |
| Naive Bayes, LR, RF, SVM roughly 0.87-0.89 | **Roughly**, in A/B for NB and LR (0.86-0.89); lower in CV (0.82-0.84). |
| Entropy (information gain) trees | Entropy ≥ Gini in all protocols (CV 0.728 vs 0.692; the gap is within one std 0.078) [`07_dt_criterion_comparison`]. |
| 7 selected features age, sex, cp, fbs, chol, slope, ca | **Not replicated** | MI top-7 = thal, cp, ca, oldpeak, exang, chol, slope; chi-square top-7 = thal, exang, ca, oldpeak, slope, sex, cp. Overlap with the paper: 4/7 each. `fbs` and `age` score ~0 in MI. Using the paper's 7 features lowered mean CV accuracy (0.790 vs 0.816 for all 13; my own MI7: 0.818) [`19_feature_set_comparison_accuracy`]. |
| Isolation-forest removal | **No consistent effect** | Changes of up to about ±0.1 in either direction depending on model, protocol and variant (the largest drops come from whole-data filtering on the 60/40 split, e.g. RF 0.869 → 0.774, where the test set also changes) [`22_isolation_forest_test_accuracy`]. Whole-dataset filtering changes the test set too. |

Also noted: in the decision-tree feature sweep a tree on 3-4 MI-selected features reaches 0.81-0.83 CV accuracy vs 0.728 with all 13 [`08_dt_feature_count_sweep`]. So "4 features" is plausible for a tree, but this k was picked using CV on all data, so it is somewhat optimistic. Random Forest tuning gave +0.004 CV accuracy (0.825 vs 0.821), within noise.

## Discrepancies and problems in the paper (verified in Step 8)
1. **Class counts.** Paper: 165 positive / 138 negative. Data: **139 disease / 164 no disease** [`01_class_balance_check`]. The paper's numbers look swapped and off by one. Missing values (4 `ca` + 2 `thal` = 6) match.
2. **Split.** Text says 60/40, but the confusion matrices total 212 + 91 = 303, which is exactly sklearn's 70/30 split (60/40 would be 181/122) [`26_split_sizes`].
3. **Table 6 layout.** As listed in the task (TP/FP/FN/TN) the six matrices disagree on how many patients truly have disease (39 to 47) although they share one test set. Read as TP/FN/FP/TN they all give 41 diseased / 50 healthy [`27_paper_table6_recomputed`]. I used the second reading. (Accuracy is the same either way.) 41/50 matches the real data (139/164 → ≈42/49) better than the paper's stated 165/138 (≈50/41).
4. **Table 3 accuracy vs Table 6.** Accuracy implied by Table 6 is below Table 3 for all models: KNN 0.868 (79/91) vs 0.941; RF 0.824 vs 0.879; NB 0.824 vs 0.868; DT 0.780 vs 0.790; SVM 0.879 vs 0.891; LR 0.868 vs 0.871 [`28_table3_vs_table6_accuracy`]. For KNN, Table 6 also gives precision = recall = 0.854, against 0.917 / 0.948 in Table 3 [`29_table3_vs_table6_prf`]. My KNN accuracy on the same split (B) is 0.868, equal to the Table 6 figure and not to Table 3.
5. **F1 inconsistent with P and R.** Recomputed 2PR/(P+R): DT 0.766 vs reported 0.864; NB 0.828 vs 0.865; RF 0.859 vs 0.879; KNN 0.932 vs 0.908. For DT, NB and KNN the reported F1 lies outside the range [min(P,R), max(P,R)], which is impossible for an F1 [`31_paper_f1_consistency`].
6. **Overfitting.** Paper's 100% training accuracy for RF, DT, NB: I reproduce it for **DT and RF** (train 1.000 vs test 0.72-0.74 and 0.82-0.87) but **not NB** (train 0.84-0.86, similar to test) [`17_train_vs_test_accuracy`]. KNN, SVM, LR show a small train-test gap (≤0.04). Tuned RF reduces the gap (train 0.87-0.92).
7. **"Decision tree with 4 or 18 features"** is impossible with 13 raw features (22 columns after one-hot, still not 18).
8. **Unspecified:** Isolation Forest hyper-parameters (and whether applied before splitting), the filter method for feature selection, the random seed, and all hyper-parameters except KNN k=8 and the linear kernel.

## Limitations of this replication
* 303 patients is small: single-split results move by several points with the seed. I report one seed (42) for A/B as required, plus 10-fold CV with std. Repeated 10×10 CV now exists as an extension (`run_extensions.py`, table 33): it reorders the leaderboard — LR leads on mean accuracy, and only 3 of 21 McNemar pairs are significant.
* k and kernel sweeps show test-set curves because the paper does; selection of k/kernel on test data is optimistic. A train-only alternative (inner CV) is also reported for KNN (`12_knn_best_k_summary`).
* My feature-ranking comparison with the paper uses the whole dataset (descriptive only); model evaluation always selects features inside the training split.
* The reading of Table 6 (TP/FN/FP/TN) is an inference from internal consistency, not something the paper confirms.
