# CardioScan AI — Full Project Document

Everything you need to understand what this project is, what data it uses, what we built,
what problem it solves, and what the results actually mean.

---

## 1. The problem we are solving

Heart disease is the leading cause of death globally, and one of the established ways to
study it computationally is **binary classification**: given a set of routine clinical
measurements for a patient, predict whether coronary heart disease is present.

This project tackles three layers of that problem:

1. **Can a model do it?** Train six classical ML pipelines on the UCI Cleveland heart-disease
   dataset and measure how well they distinguish diseased from healthy patients — with a
   methodological discipline (preprocessing inside the pipeline, held-out evaluation) that
   prevents inflated scores.
2. **Can we trust the numbers?** Reproduce a published paper's claims (Ramesh et al., 2022)
   from scratch and audit each claim against evidence. Most of the paper's headline claims
   **did not replicate** — the exercise shows how easily benchmark results mislead on small data.
3. **Can anyone use it?** Ship the analysis as a live, good-looking React + FastAPI app,
   deliberately kept to three focused screens: enter a patient's vitals and get a consensus
   prediction from six models, compare model benchmarks, and audit the paper's claims —
   instead of reading static tables.

**Real-world usefulness:** the app is a demonstration of leakage-safe ML and honest reporting
on a clinical dataset — a teaching/auditing tool and a template for how small-data ML results
*should* be reported (confidence intervals, significance tests, calibration, error analysis).
It is **not** a medical device; see the disclaimer in §10.

---

## 2. The dataset

### 2.1 Origin

| Property | Value |
|---|---|
| Name | UCI Cleveland Heart Disease (`processed.cleveland.data`) |
| Source | UCI Machine Learning Repository, archive 45 — `https://archive.ics.uci.edu/static/public/45/heart+disease.zip` |
| Size | **303 patients × 13 features** + target |
| Acquisition | Via `scripts/download_data.py`; integrity enforced at load time by `src/data.py` |
| License / use | Public research dataset (Cleveland Clinic Foundation, Robert Detrano et al.) |
| Paper under replication | Ramesh et al., *Malaysian Journal of Computer Science*, Special Issue 1/2022, pp. 132–148 (DOI 10.22452/mjcs.sp2022no1.10) |

The loader asserts the exact expected shape on every run — 303 rows, exactly 6 missing
values, `num` in 0–4, and class balance {0: 164, 1: 139} — so a truncated or silently altered
download fails loudly instead of quietly changing every downstream number.

### 2.2 The 13 features

| # | Feature | Type | Meaning |
|---|---|---|---|
| 1 | `age` | numeric | Age in years |
| 2 | `sex` | binary | 1 = male, 0 = female |
| 3 | `cp` | categorical | Chest-pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic) |
| 4 | `trestbps` | numeric | Resting blood pressure (mm Hg) on admission |
| 5 | `chol` | numeric | Serum cholesterol (mg/dl) |
| 6 | `fbs` | binary | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| 7 | `restecg` | categorical | Resting ECG result (0 normal, 1 ST-T abnormality, 2 LV hypertrophy) |
| 8 | `thalach` | numeric | Maximum heart rate achieved during exercise |
| 9 | `exang` | binary | Exercise-induced angina (1 = yes, 0 = no) |
| 10 | `oldpeak` | numeric | ST depression induced by exercise, relative to rest |
| 11 | `slope` | categorical | Slope of the peak-exercise ST segment (1 upsloping, 2 flat, 3 downsloping) |
| 12 | `ca` | numeric | Number of major vessels (0–3) seen by fluoroscopy |
| 13 | `thal` | categorical | Thallium stress test result (3 normal, 6 fixed defect, 7 reversible defect) |

Feature groups are declared in `src/data.py`:
- **Numeric** (median-imputed, scaled): `age, trestbps, chol, thalach, oldpeak, ca`
- **Binary** (already 0/1): `sex, fbs, exang`
- **Categorical** (mode-imputed, one-hot for linear/KNN/SVM models): `cp, restecg, slope, thal`

### 2.3 Target

`num` is the original ordinal diagnosis (0 = no disease, 1–4 = severity). We binarise it:

> **target = 1 if num > 0 else 0**

Class balance: **139 diseased / 164 healthy** (roughly balanced, so accuracy is a fair metric —
but note the paper claims 165/138, which is one of the discrepancies we document, §6.2).

### 2.4 Missing data

Exactly **6 missing values** (marked `?` → NaN): 4 in `ca`, 2 in `thal`.
- For **modelling**: imputation happens *inside* the sklearn `Pipeline`, fitted on training
  data only (median for numeric, mode for categorical).
- For **EDA only**: `data/heart_clean.csv` imputes on the whole dataset so plots cover all
  303 rows — that CSV is explicitly never used to train or score a model.

### 2.5 Why this dataset

- It is the canonical ML benchmark for heart-disease prediction — decades of comparable
  results, which makes replication meaningful.
- Small (303 rows) enough to run exhaustive experiments in minutes, yet rich enough
  (mixed numeric/categorical/binary features) to exercise real preprocessing pipelines.
- Its small size is exactly what makes it a good **case study in statistical honesty**:
  one test patient = 1.1 accuracy points, so most published rankings on it are noise.

---

## 3. What we have built (all work to date)

### Phase 0 — Foundation
Fixed dependencies, added a test suite, added data-checksum verification, retired the legacy
Streamlit UI. Baseline replication preserved untouched.

### Phase 1 — FastAPI backend (`backend/`)
Loads the six fitted pipelines **once** at startup and serves:
- `POST /api/predict` — consensus prediction (per-model probability, calibrated risk, vote count)
- `GET /api/health`, `/api/models`, `/api/audit/{table}`, `/api/benchmarks/{protocol}`,
  `/api/roc/{protocol}`, `/api/confusion/{protocol}`, `/api/eda/{table}`, `/api/lab/{table}`,
  `/api/figures/{name}`, `/api/sampling-noise`
- `GET /api/stats/{table}` — Phase-3 statistical tables via aliases:
  `stability`, `mcnemar`, `operating-points`, `threshold-sweep`, `permutation`, `hgb`

### Phase 2 — React frontend (`frontend/`)
React 18 + Vite + TypeScript + Tailwind, dark "AI lab" glass UI, framer-motion/recharts/lucide.
Three focused pages, all fed by live data (a first six-page version was later simplified —
see Phase 5):

| Page | What it shows |
|---|---|
| **Diagnosis** | Inputs for all 13 features (5 sliders + 8 dropdowns) → live consensus from six models: verdict, calibrated risk, per-model meters with held-out accuracies |
| **Benchmarks** | Test metrics for protocols A/B/C, ROC curves with AUCs, held-out confusion matrices |
| **Audit** | Four tabs: claims ledger, paper-vs-replication accuracy, F1 self-consistency, train-vs-test overfitting gap |

### Phase 3 — Statistical extensions (`run_extensions.py`, tables 33–38)
The paper reports only point estimates; we added the studies needed to judge whether gaps
are real: repeated 10×10 stratified CV, exact McNemar tests on the held-out split,
model-agnostic permutation importance, sensitivity/specificity threshold sweeps, clinical
operating points, and a HistGradientBoosting baseline under identical protocols.

### Phase 4 — Verification
48 pytest checks (API contract, prediction stability, endpoint schemas, exact McNemar math),
`tsc` typecheck, production build, and a full browser walkthrough of all three pages against
the live backend. Screenshots committed under `screenshots/`.

---

## 4. Methodology (how the numbers are produced)

### 4.1 Leakage-safe pipelines
Imputation, scaling, one-hot encoding and feature selection all live **inside**
`sklearn.Pipeline` (`src/models.py`), so those statistics are re-fitted on the training
partition every time and only *applied* to test data. Computing them on all 303 patients
would let test patients influence training — *leakage*, which inflates scores.

### 4.2 Evaluation protocols (`src/evaluation.py`)
| Protocol | Split |
|---|---|
| **A** | 60/40 stratified (n_test = 122) |
| **B** | 70/30 stratified (n_test = 91) |
| **C** | 10-fold stratified CV (mean ± std) |

`random_state = 42` everywhere; positive class = disease. Reported metrics always include
train **and** test, so overfitting is visible (the Decision Tree and Random Forest hit
train accuracy 1.000).

### 4.3 The six pipelines
| Model | Key hyper-parameters | Character here |
|---|---|---|
| Decision Tree | `criterion` (entropy/Gini), depth | Memorises train (1.000), worst test (≈0.73) |
| Naive Bayes (Gaussian) | none important | Fast, stable on small data (≈0.84–0.88) |
| Random Forest | trees, depth, `max_features` | Averaging cancels tree noise (≈0.82–0.87) |
| KNN | `k`, distance metric | Needs scaling; competitive but not special (≈0.83–0.87) |
| SVM (linear) | kernel, `C` | Solid on small standardised data (≈0.82–0.86) |
| Logistic Regression | regularisation `C` | Interpretable coefficients; ties for best |

An extra **HistGradientBoosting** pipeline (modern boosting baseline the paper never ran)
and an **Isolation Forest** outlier study (contamination none/0.05/0.10) round it out.
Plain-language explanations of every model, metric and formula are in `EXPLAINER.md`.

### 4.4 Metrics
Accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, NPV, confusion matrices,
train-vs-test gap, and **sampling-noise intervals** (95% binomial half-width ≈ ±7 points at
n = 91). In medicine, missed disease (FN) is usually worse than a false alarm, so the app
also surfaces sensitivity-first operating points.

---

## 5. Key results

### 5.1 Headline accuracy (all 13 features, test data; paper = its Table 3)

| Model | Paper | A (60/40) | B (70/30) | C (10-fold CV) |
|---|---|---|---|---|
| Decision Tree | 0.790 | 0.721 | 0.736 | 0.728 ± 0.078 |
| Naive Bayes | 0.868 | 0.885 | 0.879 | 0.838 ± 0.064 |
| Random Forest | 0.879 | 0.869 | 0.835 | 0.821 ± 0.064 |
| KNN (k=8) | 0.941 | 0.828 | 0.868 | 0.838 ± 0.064 |
| SVM (linear) | 0.891 | 0.844 | 0.824 | 0.835 ± 0.036 |
| Logistic Regression | 0.871 | 0.893 | 0.857 | 0.838 ± 0.058 |

### 5.2 Did the findings replicate? (full ledger in the **Audit** page / `results/32_claims_evidence.csv`)
- **Not replicated:** KNN as best model (0.941); KNN's highest recall (0.948); k=8 as best k;
  RF's best AUC (0.898); the paper's 7-feature set beating all 13.
- **Replicated:** Decision tree is the weakest model; entropy ≥ Gini (within noise).
- **Roughly / inconclusive:** NB-LR-RF-SVM band of 0.87–0.89; linear vs rbf SVM kernel.
- **No consistent effect:** isolation-forest outlier removal.

### 5.3 Discrepancies found in the paper (verified, `results/26`–`31`)
1. Class counts 165/138 vs. the real **139/164** (look swapped and off by one).
2. Text says 60/40 but the confusion matrices total 212 + 91 = exactly sklearn's **70/30** split.
3. Table 6 matrices only agree if read TP/FN/FP/TN (the listed order is impossible).
4. Table 6 accuracies are below Table 3 for all six models.
5. Reported F1 values violate the F1 identity — DT/NB/KNN lie outside [min(P,R), max(P,R)].
6. 100% train accuracy reproduced for DT and RF but **not** NB.
7. "4 or 18 features" is impossible with 13 raw features (22 after one-hot).
8. Unspecified: isolation-forest hyper-parameters, filter method, random seed.

### 5.4 Statistical extensions — the bottom line (`results/33–38`)
- Repeated 10×10 CV **reorders the leaderboard**: Logistic Regression leads at **0.844** mean
  accuracy; the top four models sit **inside each other's error bars**.
- Only **3 of 21** McNemar pairs are significant (p < 0.05) — all of them the Decision Tree
  losing. The ranking of the top five is not statistically meaningful on 91 test patients.
- Permutation importance puts `ca`, `thal`, `slope`, `cp` on top for Logistic Regression;
  near-zero bars mean the model barely uses that feature.
- The `sensitivity ≥ 0.90` operating point **degenerates**: calibrated probabilities cluster
  near 0.5, so reaching sensitivity 1.0 collapses specificity to 0 — a real, reportable
  finding about this dataset, not a bug.
- HistGradientBoosting (0.795–0.813) does **not** beat the paper's best models — modern
  boosting buys nothing at n = 303. The ceiling is the dataset size, not the model family.

---

## 6. What it is useful for

1. **Honest benchmarking** — a complete worked example of why single-split accuracy tables
   mislead: confidence intervals, repeated CV, and significance tests attached to every claim.
2. **Replication auditing** — a template for checking a published ML paper: recomputing every
   table, testing internal consistency (F1 identity, split sizes), and keeping a claims ledger.
3. **Clinical intuition building** — the threshold study (`results/36_threshold_sweep.csv`)
   shows the sensitivity/specificity trade-off a screening tool would actually face:
   catching more disease always costs more false alarms.
4. **Interpretability** — logistic-regression coefficients, permutation importance, decision
   trees, and per-model verdicts make *why* a prediction happened inspectable.
5. **A reusable full-stack template** — FastAPI serving sklearn models + a React analytics UI,
   with typed API clients, shared table/chart primitives, and end-to-end tests.

---

## 7. How to run

```bash
# 1. Python deps + replication (writes results/, ~8 min)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_replication.py
.venv/bin/python run_extensions.py          # tables 33–38 (statistical extensions)

# 2. Backend — loads models once, serves the API
.venv/bin/uvicorn backend.app:app --reload  # http://localhost:8000

# 3. Frontend — Vite dev server (proxies /api to the backend)
cd frontend && npm install && npm run dev   # http://localhost:5173

# 4. Checks
.venv/bin/python -m pytest tests/ -q        # 48 tests
cd frontend && npm run build                # tsc typecheck + production bundle
```

---

## 8. Repository structure

```
heart-replication-main/
├── DOCUMENTATION.md      ← this file
├── PLAN.md               ← agreed implementation plan (all phases ✅)
├── README.md             ← replication report: results, verdicts, discrepancies
├── EXPLAINER.md          ← plain-language explanations of models & metrics
├── run_replication.py    ← original experiment (steps, commented)
├── run_extensions.py     ← Phase-3 statistics → results/33–38
├── src/
│   ├── data.py           ← loading + shape/integrity verification
│   ├── models.py         ← preprocessing, feature selector, 6(+1) pipelines
│   └── evaluation.py     ← protocols A/B/C, metrics, isolation forest
├── backend/app.py        ← FastAPI: inference + all results tables as JSON
├── frontend/             ← React + Vite + TS + Tailwind (6 pages)
│   └── src/pages/        ← Diagnosis, Benchmarks, Audit (the whole UI)
├── tests/                ← 48 pytest checks
├── results/              ← 39 CSV tables + 15 figures (committed ground truth)
├── data/                 ← processed.cleveland.data + generated CSVs
├── scripts/download_data.py
└── screenshots/          ← verified walkthrough captures (incl. earlier 6-page UI)
```

### Verification status
- `pytest` — **48/48 passed**
- `npm run build` (`tsc -b && vite build`) — **clean**, 605 kB (173 kB gzip)
- Browser walkthrough — all three pages render live data, console clean
- Git — phase-by-phase commits from baseline through Phase 4; this document is the only
  uncommitted file in the working tree

---

## 9. Limitations

- **n = 303.** One test patient ≈ 1.1 accuracy points; 95% CI on a ~0.85 accuracy is about
  ±7 points. Most between-model gaps here are noise — that is a finding, not an excuse.
- **One dataset, one era.** Cleveland data comes from the 1988 era; results do not transfer
  to other populations, modern assays, or different prevalence settings.
- **Paper ambiguities.** Where the paper is silent (seeds, filter method, isolation-forest
  settings), we chose documented defaults and recorded the choice rather than guessing to fit.
- **Table 6 reading** (TP/FN/FP/TN) is an inference from internal consistency, not confirmed
  by the paper.
- **Optimistic sweeps.** KNN-k and SVM-kernel sweeps are plotted on test data because the
  paper does that; a train-only inner-CV alternative is also reported for KNN.

## 10. Disclaimer

This is a research and education project. It is **not** a medical device and must not be
used for diagnosis, screening, or treatment decisions. Predictions come from models trained
on 303 historical patients and will be wrong often enough to be dangerous in a clinical
setting; any real screening tool would require prospective validation, calibration on the
target population, regulatory review, and clinician oversight.
