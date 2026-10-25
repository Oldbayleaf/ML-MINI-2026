# CardioScan AI: Comprehensive Project Dossier & Presentation Guide

> **Replication, Critical Audit, Statistical Hardening, and Modern Web Deployment of:**  
> *"Predictive Analysis of Heart Diseases with Machine Learning Approaches"*  
> Ramesh et al., *Malaysian Journal of Computer Science*, Special Issue 1/2022, pp. 132–148 (DOI: 10.22452/mjcs.sp2022no1.10)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [The Problem Statement & Real-World Relevance](#2-the-problem-statement--real-world-relevance)
3. [The Dataset & Feature Engineering](#3-the-dataset--feature-engineering)
4. [Methodology & Leakage-Safe Machine Learning](#4-methodology--leakage-safe-machine-learning)
5. [The Six Model Pipelines + Modern Benchmark](#5-the-six-model-pipelines--modern-benchmark)
6. [Replication Findings: Paper Claims vs. Empirical Evidence](#6-replication-findings-paper-claims-vs-empirical-evidence)
7. [Forensic Audit: Uncovering Flaws in the Published Literature](#7-forensic-audit-uncovering-flaws-in-the-published-literature)
8. [Advanced Statistical Extensions (Phase 3)](#8-advanced-statistical-extensions-phase-3)
9. [Full-Stack Modernization Architecture (FastAPI + React 18)](#9-full-stack-modernization-architecture-fastapi--react-18)
10. [Test Suite, Verification, and Engineering Discipline](#10-test-suite-verification-and-engineering-discipline)
11. [Presentation Strategy & Slide-by-Slide Walkthrough](#11-presentation-strategy--slide-by-slide-walkthrough)
12. [Comprehensive Viva / Review Q&A Guide](#12-comprehensive-viva--review-qa-guide)

---

## 1. Executive Summary

**CardioScan AI** is an end-to-end machine learning research, audit, and engineering project. Rather than treating machine learning as an uncritical exercise of downloading a dataset and fitting models to claim a high accuracy score, this project performs a **rigorous scientific replication and critical audit** of a peer-reviewed published study (Ramesh et al., 2022).

### Three Core Pillars:
1. **Scientific Replication & Statistical Auditing:** Re-implementing every step of the published paper with strict leakage-free ML pipelines, uncovering multiple mathematical discrepancies, impossible metrics (such as $F_1$ scores outside mathematical bounds), and disproving the paper's central claims (e.g., that KNN achieves 94.1% accuracy).
2. **Advanced Statistical Hardening:** Demonstrating that on small clinical datasets ($N = 303$), single-split test metrics are largely sampling noise. We implement repeated $10 \times 10$ cross-validation, exact McNemar pairwise hypothesis tests, model-agnostic permutation importance, and clinical threshold sweeps.
3. **Modern Clinical Decision Support System (CDSS) Dashboard:** Transitioning static code into a production-grade full-stack web application powered by **FastAPI** (Python 3) and **React 18 / TypeScript / Tailwind CSS**, providing live multi-model consensus predictions, calibrated risk estimates (Platt scaling), interactive benchmark visualizations, and an interactive paper audit ledger.

---

## 2. The Problem Statement & Real-World Relevance

### The Clinical Context
Cardiovascular diseases (CVDs) remain the leading cause of morbidity and mortality worldwide, responsible for an estimated 17.9 million deaths annually. Early identification of coronary artery disease enables non-invasive interventions, lifestyle modification, and targeted pharmacotherapy prior to acute cardiac events (such as myocardial infarction).

### The Machine Learning Problem
Given **13 standard non-invasive and minimally invasive clinical measurements** (e.g., age, resting blood pressure, serum cholesterol, exercise-induced ST depression, fluoroscopy vessel counts, thallium stress test), predict whether significant coronary artery disease ($\ge 50\%$ diameter narrowing in at least one major vessel) is present.

### Why Most Published Papers Fail in Healthcare ML
1. **Data Leakage:** Many papers normalize or impute across the entire dataset before train-test splitting.
2. **Over-Optimistic Claims on Tiny Samples:** On $N = 303$ patients with a $70/30$ test split ($n = 91$), a single patient shifting prediction changes the accuracy by **$1.1\%$**. A $95\%$ confidence interval has a half-width of **$\pm 7\%$**.
3. **Misalignment of Loss Functions:** In medical screening, a False Negative (missing a patient with heart disease) is drastically more dangerous than a False Positive (requesting follow-up diagnostics). High accuracy is useless if sensitivity is poor.

---

## 3. The Dataset & Feature Engineering

### 3.1 Dataset Overview
- **Name:** UCI Cleveland Heart Disease (`processed.cleveland.data`)
- **Origin:** Cleveland Clinic Foundation (Dr. Robert Detrano et al.) via UCI ML Repository Archive #45
- **Dimensions:** 303 patient records $\times$ 14 attributes (13 features + target)
- **Integrity Validation:** Enforced programmatically at load time (`src/data.py`):
  - Exactly 303 rows.
  - Exactly 6 missing values (represented as `'?'` in raw data: 4 in `ca`, 2 in `thal`).
  - Target values strictly in $\{0, 1, 2, 3, 4\}$.
  - Binarized class counts: **164 healthy (0), 139 diseased (1)**.

### 3.2 The 13 Clinical Predictors

| # | Feature Code | Variable Type | Clinical Interpretation | Clinical Significance |
|---|--------------|---------------|-------------------------|------------------------|
| 1 | `age` | Continuous | Patient age in years | Baseline cardiovascular risk factor. |
| 2 | `sex` | Binary | 1 = male; 0 = female | Estrogen provides pre-menopausal protection; higher risk in males. |
| 3 | `cp` | Categorical | Chest Pain Type (1: Typical angina, 2: Atypical angina, 3: Non-anginal, 4: Asymptomatic) | Paradoxically, asymptomatic (`cp=4`) often correlates with severe silent ischemia in clinical presentations. |
| 4 | `trestbps` | Continuous | Resting blood pressure (mm Hg on admission) | Indicator of systemic hypertension and vascular strain. |
| 5 | `chol` | Continuous | Serum cholesterol in mg/dl | Biomarker of atherogenic lipid accumulation. |
| 6 | `fbs` | Binary | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) | Marker of diabetes mellitus / metabolic syndrome. |
| 7 | `restecg` | Categorical | Resting ECG (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy) | Structural and electrical cardiac abnormalities. |
| 8 | `thalach` | Continuous | Maximum heart rate achieved during treadmill stress | Chronotropic incompetence (failure to achieve expected peak HR) indicates ischemia. |
| 9 | `exang` | Binary | Exercise-induced angina (1 = yes; 0 = no) | Classic physical indicator of exercise-induced myocardial ischemia. |
| 10 | `oldpeak` | Continuous | ST depression induced by exercise relative to rest | Direct electrocardiographic sign of subendocardial ischemia. |
| 11 | `slope` | Categorical | Slope of peak exercise ST segment (1: Upsloping, 2: Flat, 3: Downsloping) | Flat or downsloping ST segments are strong predictors of coronary stenosis. |
| 12 | `ca` | Numeric (0–3) | Major vessels colored by fluoroscopy | Direct anatomical count of major vessels showing calcification/blockage. |
| 13 | `thal` | Categorical | Thallium-201 Scintigraphy (3: Normal, 6: Fixed defect, 7: Reversible defect) | Reversible defect indicates tissue ischemia that can be salvaged. |

### 3.3 Target Definition
The original dataset records `num` as $0$ (no disease) or $1, 2, 3, 4$ (severity based on coronary angiography $>50\%$ diameter stenosis).  
Following standard literature and the paper under replication:
$$\text{Target } y = \begin{cases} 1 & \text{if } num > 0 \quad (\text{Heart Disease Present}) \\ 0 & \text{if } num = 0 \quad (\text{Heart Disease Absent}) \end{cases}$$
- **Actual Distribution:** 164 Absent ($54.1\%$), 139 Present ($45.9\%$).

---

## 4. Methodology & Leakage-Safe Machine Learning

### 4.1 Strict Leakage Prevention Architecture
A frequent error in applied ML papers is performing feature preprocessing on the entire dataset $D$ before splitting.  
If you compute:
$$\mu = \frac{1}{N}\sum_{i=1}^N x_i \quad \text{and} \quad \sigma = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2}$$
across all 303 rows, the mean and variance contain information from the test partition $D_{\text{test}}$. This is **information leakage**.

**Our Solution (`src/models.py`):**
Every preprocessing step is wrapped inside an explicit `sklearn.pipeline.Pipeline`:
- **Numeric Features (`age, trestbps, chol, thalach, oldpeak, ca`):**
  1. `SimpleImputer(strategy="median")` fitted solely on $X_{\text{train}}$.
  2. `StandardScaler()` fitted solely on $X_{\text{train}}$.
- **Categorical Features (`cp, restecg, slope, thal`):**
  1. `SimpleImputer(strategy="most_frequent")` fitted solely on $X_{\text{train}}$.
  2. `OneHotEncoder(handle_unknown="ignore")` for distance/linear models (KNN, SVM, Logistic Regression).
- **Binary Features (`sex, fbs, exang`):**
  Passed through directly as $0/1$.
- **Feature Selection:** When evaluating subsets (e.g., Mutual Information top 7), the selector is fitted strictly inside the cross-validation fold or training split.

### 4.2 Evaluation Protocols
To benchmark fairly against the paper, we evaluated across three distinct experimental protocols:
- **Protocol A:** $60/40$ Stratified Train/Test Split ($N_{\text{train}} = 181, N_{\text{test}} = 122$).
- **Protocol B:** $70/30$ Stratified Train/Test Split ($N_{\text{train}} = 212, N_{\text{test}} = 91$).
- **Protocol C:** $10$-Fold Stratified Cross-Validation (all 303 samples, reporting $\text{Mean} \pm \text{Std}$).
- **Random Seed:** Fixed to `random_state = 42` across all splits, models, and resamplers.

---

## 5. The Six Model Pipelines + Modern Benchmark

We implemented and audited the 6 core classifiers studied in the paper, plus a modern boosting baseline:

1. **Decision Tree Classifier (`DecisionTreeClassifier`):**
   - Criterion: Entropy (Information Gain).
   - Finding: Severe overfitting (Train Accuracy: $1.000$; Test Accuracy: $0.721 - 0.736$).
   - Reason: Unconstrained tree memorizes noisy training samples; splits deep in the tree have tiny support.
2. **Gaussian Naive Bayes (`GaussianNB`):**
   - Assumes conditional independence between features given the class label.
   - Finding: Extremely competitive on small data ($0.838 \pm 0.064$ CV accuracy, up to $0.885$ on split A).
   - Reason: Independence assumption acts as strong regularizer, preventing high variance.
3. **Random Forest Classifier (`RandomForestClassifier`, 100 trees):**
   - Bagging ensemble of randomized decision trees.
   - Finding: High training score ($1.000$) with modest test score ($0.821 \pm 0.064$).
   - Hyperparameter tuning ($max\_depth=5, min\_samples\_leaf=4$) reduced overfitting gap by $0.10$ but only gained $+0.004$ CV accuracy.
4. **K-Nearest Neighbors (`KNeighborsClassifier`, $k=8$, Euclidean distance):**
   - Classifies by majority vote of the 8 closest standardized training instances.
   - Paper's Crown Jewel: The paper claimed $94.1\%$ accuracy.
   - Replication Reality: Achieved $0.828$ (A), $0.868$ (B), $0.838$ (C). It is **not** the best model.
5. **Linear Support Vector Machine (`LinearSVC` / `SVC(kernel='linear')`):**
   - Maximizes geometric margin between hyperplanes.
   - Finding: Stable and robust ($0.835 \pm 0.036$ in CV, lowest variance among all models).
6. **Logistic Regression (`LogisticRegression`, L2 penalty, liblinear):**
   - Parametric probabilistic linear model.
   - Finding: Tied for highest mean CV accuracy ($0.838 \pm 0.058$) and won Protocol A ($0.893$).
   - Reason: The clinical signal in this dataset is predominantly additive/linear.
7. **Modern Benchmark: HistGradientBoosting (`HistGradientBoostingClassifier`):**
   - Modern tree-boosting baseline (LightGBM-style binning).
   - Finding: Test accuracy $0.795 - 0.813$. Modern boosting does **not** outperform simple models at $N=303$. The bottleneck is sample size, not algorithmic complexity.

---

## 6. Replication Findings: Paper Claims vs. Empirical Evidence

### Headline Test Accuracy Comparison (All 13 Features)

| Model | Paper Reported (Table 3) | Protocol A (60/40) | Protocol B (70/30) | Protocol C (10-Fold CV) |
|---|:---:|:---:|:---:|:---:|
| **Decision Tree** | 0.790 | 0.721 | 0.736 | 0.728 ± 0.078 |
| **Naive Bayes** | 0.868 | **0.885** | **0.879** | 0.838 ± 0.064 |
| **Random Forest** | 0.879 | 0.869 | 0.835 | 0.821 ± 0.064 |
| **KNN ($k=8$)** | **0.941** | 0.828 | 0.868 | 0.838 ± 0.064 |
| **SVM (Linear)** | 0.891 | 0.844 | 0.824 | 0.835 ± 0.036 |
| **Logistic Regression** | 0.871 | **0.893** | 0.857 | **0.838 ± 0.058** |

### Detailed Claims Audit Ledger

```
                                CLAIM REPLICATION AUDIT
┌──────────────────────────────────────┬──────────────────┬───────────────────────────────────────────┐
│ Paper Claim                          │ Empirical Status │ Ground-Truth Replication Evidence         │
├──────────────────────────────────────┼──────────────────┼───────────────────────────────────────────┤
│ 1. KNN (k=8) is best model (0.941)   │ NOT REPLICATED   │ KNN is 5th in A, 2nd in B, ties 1st in C. │
│                                      │                  │ Actual accuracy is 0.83–0.87, never 0.94. │
│ 2. KNN has highest recall (0.948)    │ NOT REPLICATED   │ KNN recall is 0.73–0.79. Naive Bayes has  │
│                                      │                  │ the highest recall across all protocols.  │
│ 3. k=8 is the optimal KNN parameter  │ NOT REPLICATED   │ Euclidean best k is 5 (A), 6 (B), 5 (CV). │
│                                      │                  │ k=8 is decent, but not optimal.           │
│ 4. Linear is best SVM kernel         │ INCONCLUSIVE     │ RBF beats Linear in A & B; Linear wins in │
│                                      │                  │ CV by < 0.02. All 4 kernels within noise. │
│ 5. RF has best AUC (0.898)           │ NOT REPLICATED   │ LR & NB top AUC (0.91–0.94); RF is 2nd/3rd│
│ 6. Decision Tree is weakest model    │ REPLICATED       │ DT ranks last in every protocol (~0.73).  │
│ 7. Paper's 7 features beat all 13    │ NOT REPLICATED   │ Paper 7 drops CV accuracy from 0.816 to   │
│                                      │                  │ 0.790. Our MI-7 achieves 0.818.           │
│ 8. Isolation Forest improves score   │ NO EFFECT        │ Erratic changes (±0.10); whole-dataset    │
│                                      │                  │ outlier removal causes test data leakage. │
└──────────────────────────────────────┴──────────────────┴───────────────────────────────────────────┘
```

---

## 7. Forensic Audit: Uncovering Flaws in the Published Literature

During the replication, we performed an in-depth forensic check on the paper’s internal math. We uncovered **8 major discrepancies and mathematical impossibilities**:

### 1. Inverted Class Balance
- **Paper Claim:** 165 heart disease patients, 138 healthy patients.
- **Dataset Reality:** Exactly **139 disease (1)** and **164 no disease (0)**. The paper's authors apparently inverted the labels and made an off-by-one counting error.

### 2. Discrepancy in Split Sizes
- **Paper Text:** Claims an evaluation split of $60\%$ train / $40\%$ test.
- **Paper Confusion Matrices:** Total $212 + 91 = 303$. This is **identically a 70/30 split**, directly contradicting their own methodology section.

### 3. Confusion Matrix Layout Errors
- In Table 6 of the paper, if cells are read as standard $[[TP, FP], [FN, TN]]$, the number of actual disease cases varies between 39 and 47 across models—even though they evaluated on the exact same test split.
- Only when read as $[[TP, FN], [FP, TN]]$ do all models sum to 41 diseased and 50 healthy patients.

### 4. Table 3 Accuracy vs. Table 6 Confusion Matrices
- The accuracies mathematically calculated from their own confusion matrices in Table 6 **fail to match their headline Table 3**:
  - KNN in Table 6: $(35 + 44)/91 = \mathbf{0.868}$ (matches our Protocol B exactly!), yet Table 3 reports $\mathbf{0.941}$.
  - Random Forest: $75/91 = \mathbf{0.824}$ in Table 6 vs. $\mathbf{0.879}$ in Table 3.
  - Naive Bayes: $75/91 = \mathbf{0.824}$ in Table 6 vs. $\mathbf{0.868}$ in Table 3.

### 5. Violation of the Mathematical Identity of $F_1$ Score
The $F_1$ score is the harmonic mean of Precision ($P$) and Recall ($R$):
$$F_1 = \frac{2 P R}{P + R}$$
Mathematically, the harmonic mean **must strictly satisfy**:
$$\min(P, R) \le F_1 \le \max(P, R)$$
- **Paper Reported Values:**
  - Decision Tree: $P = 0.840, R = 0.704 \implies \text{Reported } F_1 = \mathbf{0.864}$ (*Mathematically impossible! It exceeds both $P$ and $R$! Correct: 0.766*).
  - Naive Bayes: $P = 0.826, R = 0.830 \implies \text{Reported } F_1 = \mathbf{0.865}$ (*Exceeds max(P, R)! Correct: 0.828*).
  - KNN: $P = 0.917, R = 0.948 \implies \text{Reported } F_1 = \mathbf{0.908}$ (*Falls below min(P, R)! Correct: 0.932*).

### 6. Overfitting Claims
The paper claimed $100\%$ training accuracy for Random Forest, Decision Tree, and Naive Bayes.
- We reproduced $100\%$ for Decision Tree and Random Forest (classic tree memorization).
- But Gaussian Naive Bayes achieves only $84-86\%$ on train data. Naive Bayes cannot achieve $100\%$ on continuous features unless variances approach zero.

### 7. The "4 or 18 Features" Claim
The paper discusses a decision tree trained on "4 or 18 features." The raw dataset contains only 13 features; even full one-hot encoding produces 22 features, making 18 impossible.

### 8. Critical Parameter Omissions
The paper failed to report: random seeds, Isolation Forest contamination rate $\nu$, filtering threshold, or solver hyperparameters.

---

## 8. Advanced Statistical Extensions (Phase 3)

Point estimates are dangerous on small samples. We developed `run_extensions.py` (generating tables 33–38) to bring statistical rigor:

### 8.1 Repeated $10 \times 10$ Cross-Validation
Evaluating 10 folds repeated across 10 random seeds (100 total iterations per model):
- **Logistic Regression:** $\mathbf{0.844 \pm 0.059}$ (Leads the entire benchmark)
- **Gaussian Naive Bayes:** $0.840 \pm 0.063$
- **Linear SVM:** $0.838 \pm 0.060$
- **KNN ($k=8$):** $0.837 \pm 0.061$
- **Random Forest:** $0.828 \pm 0.063$
- **Decision Tree:** $0.728 \pm 0.076$
- **Takeaway:** The top 5 models are tightly clustered within $0.007$ of each other with standard deviations around $0.06$. Their ranking is **statistically indistinguishable**.

### 8.2 Exact McNemar Pairwise Hypothesis Tests
We ran 21 pairwise McNemar tests on the held-out test set ($n=91$) with continuity correction to check if model disagreements are statistically significant:
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
- **Result:** Only **3 out of 21 model pairs** showed statistical significance ($p < 0.05$).
- Every single significant difference was the **Decision Tree losing** to another classifier (DT vs. NB, DT vs. LR, DT vs. KNN).
- The difference between KNN, Logistic Regression, Naive Bayes, Random Forest, and SVM is **pure noise** ($p > 0.40$).

### 8.3 Permutation Feature Importance
Model-agnostic shuffle test ($n\_repeats=30$):
- For **Logistic Regression**, the top predictors are:
  1. `ca` (Fluoroscopy vessel count)
  2. `thal` (Thallium scintigraphy)
  3. `cp` (Chest pain type)
  4. `slope` (Peak exercise ST segment)
- Features like `fbs` and `chol` showed near-zero importance, explaining why simpler models perform better than models trying to use all 13 noisy features.

### 8.4 Clinical Operating Point & Sensitivity Degeneration
In screening, false negatives must be minimized. When sweeping classification thresholds:
- Requiring Sensitivity $\ge 0.90$ forces the classification threshold down to $\approx 0.20$.
- Because predicted probabilities on $N=303$ cluster around $0.5$, forcing sensitivity to $1.0$ collapses specificity to $0.0$ (predicting everyone is sick).
- This is a vital clinical finding: an automated screener cannot replace clinical judgment without generating an unmanageable flood of false alarms.

---

## 9. Full-Stack Modernization Architecture (FastAPI + React 18)

```
                            CARDIO-SCAN AI SYSTEM ARCHITECTURE
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                FRONTEND (React 18 + Vite + TS)                               │
│                                                                                             │
│  ┌───────────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────┐  │
│  │    Diagnosis View     │   │        Benchmarks View        │   │      Audit View       │  │
│  │ • 13 Vitals Form      │   │ • Protocol A/B/C Selectors    │   │ • Paper Claims Ledger │  │
│  │ • 6-Model Consensus   │   │ • Interactive ROC Curves (AUC)│   │ • Table 3 vs Table 6  │  │
│  │ • Calibrated Risk (%) │   │ • Live Confusion Matrices     │   │ • F1 Violations Proof │  │
│  │ • Model Voting Meters │   │ • Sampling Noise Bands        │   │ • Train vs Test Gaps  │  │
│  └───────────┬───────────┘   └───────────────┬───────────────┘   └───────────┬───────────┘  │
└──────────────┼───────────────────────────────┼───────────────────────────────┼──────────────┘
               │ JSON Inference                │ JSON Metrics                  │ JSON Tables
               ▼                               ▼                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 BACKEND (FastAPI / Uvicorn)                                 │
│                                                                                             │
│  • Singleton In-Memory State: Models trained once on 70/30 train split; zero test leakage.  │
│  • Platt-Calibrated Twin Classifiers: CalibratedClassifierCV(method='sigmoid', cv=5).       │
│  • Pre-parsed JSON parsers: Converts stringified ± std tables into numerical payload arrays.│
│                                                                                             │
│  Endpoints:                                                                                 │
│   POST /api/predict          GET /api/benchmarks/{protocol}       GET /api/audit/{table}    │
│   GET  /api/models           GET /api/roc/{protocol}              GET /api/stats/{alias}    │
│   GET  /api/health           GET /api/confusion/{protocol}        GET /api/figures/{name}   │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                 ┌─────────────┴─────────────┐
                                 │                           │
                                 ▼                           ▼
                     ┌───────────────────────┐   ┌───────────────────────┐
                     │ Scikit-Learn Pipelines│   │ Committed CSV Results │
                     │   (6 Trained Models)  │   │  (38 Audited Tables)  │
                     └───────────────────────┘   └───────────────────────┘
```

### Design Highlights
- **Single Page App (SPA) UX:** Built with Tailwind CSS and Lucide icons, styled in a sleek dark "AI Research Lab" aesthetic.
- **Platt Scaling:** Raw SVM decision functions or Decision Tree leaf ratios are not real probabilities. The backend fits a 5-fold cross-validated sigmoid calibrator (`CalibratedClassifierCV`) on the training set to serve well-calibrated posterior probabilities.
- **Separation of Concerns:** Static benchmark tables are served directly from audited CSV artifacts; inference executes against in-memory pre-fitted pipelines.

---

## 10. Test Suite, Verification, and Engineering Discipline

To meet high-reliability standards, the system includes comprehensive automated testing:
- **Pytest Suite (`tests/`):** **48 passing automated tests**:
  - `test_no_leakage.py`: Proves that preprocessing transformers are never fitted on test data.
  - `test_pipelines.py`: Validates prediction shapes, probabilities, and deterministic behavior.
  - `test_data.py`: Verifies SHA-256 data checksum, exact row counts, and column types.
  - `test_api.py`: Validates FastAPI schemas, boundary values, and HTTP response codes.
  - `test_extensions.py`: Asserts mathematical correctness of McNemar contingency tables.
- **Frontend Verification:** Clean TypeScript compilation (`tsc -b`), tree-shaken production bundle (`605 kB` / `173 kB` gzip), and zero browser console errors.

---

## 11. Presentation Strategy & Slide-by-Slide Walkthrough

### Recommended Presentation Structure (10–12 Minutes)

#### Slide 1: Title & Hook
- **Title:** CardioScan AI: Replicating, Auditing, and Modernizing ML for Heart Disease Diagnosis.
- **The Hook:** *"Can you trust published machine learning benchmarks in healthcare? We re-implemented a peer-reviewed 2022 paper from scratch, tested its claims against rigorous statistics, uncovered major mathematical discrepancies, and transformed the findings into a modern clinical AI application."*

#### Slide 2: Clinical Significance & The Dataset
- Cardiovascular disease is the world’s #1 killer.
- The UCI Cleveland benchmark: 303 patients, 13 features.
- Explain the key features: `ca` (fluoroscopy vessels), `thal` (stress defect), `cp` (chest pain), `oldpeak` (ST depression).
- Highlight the golden rule of small datasets: $N=303$ means 1 patient equals $1.1\%$ of the test set.

#### Slide 3: The Flawed Benchmark: What the Paper Claimed
- Paper (Ramesh et al., 2022) reported:
  - KNN ($k=8$) achieved a stellar $94.1\%$ accuracy and $94.8\%$ recall.
  - Decision Trees scored $79.0\%$.
  - Random Forest, Naive Bayes, and SVM hovered between $86–89\%$.
- Set the stage: Did these numbers hold up under rigorous, leakage-free replication?

#### Slide 4: Leakage-Safe Engineering & Replication Results
- Explain our pipeline methodology: Imputation, scaling, and one-hot encoding happen **strictly inside cross-validation loops**.
- Display the comparison table:
  - KNN plummeted from the paper's $94.1\%$ down to $83.8\%$ (10-fold CV) and $86.8\%$ (70/30 split).
  - Decision Tree was indeed the weakest ($72.8\%$).
  - Logistic Regression and Naive Bayes tied or beat KNN.

#### Slide 5: Forensic Audit: The "Smoking Guns"
- Walk the audience through the mathematical flaws we uncovered:
  1. **Swapped Class Balance:** Paper claimed 165/138; ground truth is 139/164.
  2. **Contradictory Splits:** Text claims 60/40; their confusion matrices equal exactly 70/30 (212 + 91).
  3. **Table 3 vs. Table 6:** Their own confusion matrix calculates to $86.8\%$ for KNN, directly contradicting their headline $94.1\%$.
  4. **The $F_1$ Score Impossibility:** Show the math: $F_1$ must be between Precision and Recall. The paper's reported values mathematically violate this law.

#### Slide 6: Statistical Reality: Beyond Point Estimates
- Introduce the Phase 3 statistical extensions:
  - **Repeated $10 \times 10$ CV:** 100 trials reveal Logistic Regression leads at $84.4\%$; top 5 models are indistinguishable within error bars ($\pm 0.06$).
  - **McNemar Tests:** 18 out of 21 model pairs show $p > 0.40$ (no significant difference). Only the Decision Tree is significantly worse.
  - **Permutation Importance:** Explains why: linear models focus on the strong anatomical signals (`ca`, `thal`, `cp`) and ignore noisy noise features.

#### Slide 7: Live Software Architecture & Demo
- Present the FastAPI + React architecture diagram.
- **Showcase the 3 UI Pages:**
  1. *Diagnosis:* Show interactive sliders/dropdowns producing live consensus and calibrated risk.
  2. *Benchmarks:* Show interactive ROC curves and protocol switching.
  3. *Audit:* Walk through the interactive claims ledger and discrepancy cards.

#### Slide 8: Clinical Reality & Ethical Disclaimer
- Discuss the Sensitivity vs. Specificity trade-off: forcing sensitivity to $90\%+$ causes massive false alarms.
- Reiterate that CardioScan AI is an educational, auditing, and research tool—not an unmonitored medical device.

#### Slide 9: Conclusion & Key Learnings
- **Three Core Takeaways:**
  1. Methodology beats hype: Pipelines must prevent leakage, and single-split numbers are largely noise.
  2. Literature requires verification: Published papers can contain severe internal contradictions.
  3. Software engineering matters: Turning ML experiments into robust, test-backed (48 pytests), containerized web applications bridges data science and usability.

---

## 12. Comprehensive Viva / Review Q&A Guide

### Category 1: Machine Learning & Preprocessing

#### Q1: What is data leakage, and exactly how did you prevent it in your code?
> **Model Answer:**  
> *"Data leakage occurs when information from outside the training dataset is inadvertently used to create the model. A common example on this dataset is scaling features (like cholesterol or age) using the mean and standard deviation of all 303 rows before splitting. That leaks the distribution of the test set into training.  
> We prevented this by encapsulating all transformations inside `sklearn.pipeline.Pipeline`. Imputation (`SimpleImputer`), scaling (`StandardScaler`), and one-hot encoding (`OneHotEncoder`) are fitted strictly during `pipeline.fit(X_train)` on the training split only, and merely transformed on the test split. We verified this with an automated test (`test_no_leakage.py`)."*

#### Q2: Why did you calibrate model probabilities using Platt Scaling (`CalibratedClassifierCV`)?
> **Model Answer:**  
> *"Different algorithms output raw scores with different interpretations. Linear SVM outputs a signed geometric distance to the separating margin, which is unbounded and not a probability. Random Forest outputs the fraction of trees voting for a class, which tends to be under-confident near 0 and 1. Decision Trees produce step-function probabilities based on leaf sample proportions.  
> To provide a trustworthy, clinically interpretable risk score, we fitted a sigmoid Platt calibrator (`CalibratedClassifierCV(method='sigmoid', cv=5)`) strictly on the training partition. This fits a logistic regression mapping the raw scores to well-calibrated posterior probabilities."*

#### Q3: Why didn't modern HistGradientBoosting beat basic Logistic Regression?
> **Model Answer:**  
> *"Gradient boosting excels when there are tens of thousands of samples with complex, non-linear interactions and high-dimensional interactions. In the Cleveland dataset, we have only 303 samples and 13 features. Furthermore, the true underlying physiological signal is largely monotonic and additive (higher age, higher ST depression, and more blocked vessels all steadily increase risk).  
> At $N=303$, boosting simply overfits to sample variance. Simple, well-regularized linear models or Naive Bayes perform better because their inductive bias matches small clinical sample sizes."*

#### Q4: Why did Gaussian Naive Bayes perform so well despite the independence assumption being violated?
> **Model Answer:**  
> *"Features like `thal`, `exang`, and `oldpeak` are clinically correlated, so the naive independence assumption is technically false. However, Naive Bayes only needs the correct ranking of the class posterior probabilities to make an accurate binary classification decision, not perfectly accurate probability densities. Because Gaussian NB fits only a mean and variance per feature per class (very few parameters), it has near-zero variance and does not overfit on 200 training samples."*

---

### Category 2: Statistics & Replication Auditing

#### Q5: How did you prove that the paper's reported $F_1$ scores are mathematically impossible?
> **Model Answer:**  
> *"The $F_1$ score is defined as the harmonic mean of Precision and Recall: $F_1 = \frac{2 P R}{P + R}$. A fundamental mathematical property of the harmonic mean of two positive numbers is that it must lie between them: $\min(P, R) \le F_1 \le \max(P, R)$.  
> In the paper's Table 3, the Decision Tree has reported Precision $0.840$ and Recall $0.704$, but reported an $F_1$ of $0.864$—which exceeds both values! For Naive Bayes, $P=0.826$ and $R=0.830$, but reported $F_1$ was $0.865$. These are not rounding differences; they are impossible values that prove the paper's reported metric table was erroneous."*

#### Q6: Why did you use McNemar's test instead of a paired t-test on CV folds?
> **Model Answer:**  
> *"A paired t-test on cross-validation folds violates the assumption of independence because training sets overlap across folds (the Dietterich 1998 critique).  
> McNemar's test is a non-parametric test designed specifically for comparing two binary classifiers evaluated on the exact same test set. It creates a $2 \times 2$ contingency table of cases where Model A is correct and Model B is incorrect ($b$), versus where Model B is correct and Model A is incorrect ($c$). It tests the null hypothesis that both models have equal marginal error rates using the test statistic $\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$. It showed that only 3 of 21 model pairs were statistically different—all involving the underperforming Decision Tree."*

#### Q7: What is the significance of the 95% binomial confidence interval on this test set?
> **Model Answer:**  
> *"With a 70/30 split, the held-out test set has $n = 91$ patients. The standard error of a sample proportion is $\sqrt{\frac{p(1-p)}{n}}$. For an accuracy of $p = 0.85$, the $95\%$ confidence interval half-width is:  
> $$1.96 \times \sqrt{\frac{0.85 \times 0.15}{91}} \approx 1.96 \times 0.0374 \approx \pm 7.3\%$$  
> This means any accuracy reported around $85\%$ actually lies within $[77.7\%, 92.3\%]$. Each individual patient represents $\frac{1}{91} \approx 1.1\%$. Therefore, ranking models based on a difference of 1 or 2 percentage points is purely reporting sampling noise."*

---

### Category 3: Software Engineering & System Design

#### Q8: How does the FastAPI backend manage model inference without retraining on every HTTP request?
> **Model Answer:**  
> *"We implemented a singleton application state pattern (`State` in `backend/app.py`). During server startup (via FastAPI lifespan context), the models are trained once on the training partition ($70\%$) and cached in memory alongside their calibrated twins. When a user submits vitals via `POST /api/predict`, the API deserializes the JSON into a Pydantic schema, constructs a single-row DataFrame, and queries the pre-fitted pipelines in memory. Inference takes under 15 milliseconds."*

#### Q9: Why did you build the UI with React + Vite + TypeScript instead of Streamlit?
> **Model Answer:**  
> *"Streamlit re-executes the entire Python script on every user interaction, making it difficult to build responsive, customized interfaces with complex client-side state.  
> By decoupling into a FastAPI REST backend and a React/TypeScript single-page app, we achieved:
> 1. Strict typing and interface contracts across client and server.
> 2. Zero-latency client-side UI transitions (tabs, sliders, benchmark filters).
> 3. Production readiness: The frontend builds into optimized, static HTML/JS/CSS assets that can be served via CDN, while the API can scale independently."*

#### Q10: What test coverage do you have, and what does it guarantee?
> **Model Answer:**  
> *"We have 48 automated pytest tests in `tests/`. They test:
> 1. Data integrity: Asserts shape, no dropped columns, exact class counts.
> 2. Leakage prevention: Verifies that imputer and scaler statistics match train-only calculations.
> 3. API contracts: Validates HTTP status codes, schema structures, and deterministic predictions for edge cases.
> 4. Mathematical tests: Verifies McNemar calculations against manual chi-square computations.  
> On the frontend, TypeScript ensures 100% type safety at build time."*

---

### Category 4: Clinical & Ethical Considerations

#### Q11: What would happen if this model were deployed in an actual hospital triage room tomorrow?
> **Model Answer:**  
> *"It would fail, and it would be dangerous. First, the Cleveland dataset is from 1988; diagnostic standards, troponin biomarkers, and patient demographics have evolved significantly. Second, our operating-point analysis revealed that achieving a medically safe sensitivity ($\ge 90\%$) collapses the specificity, meaning an enormous number of healthy patients would be flagged for invasive catheterization.  
> CardioScan AI is designed as an auditing, educational, and methodology benchmark tool to teach rigorous ML practices. Real-world clinical AI requires prospective multi-center clinical trials, external validation across diverse demographics, and formal regulatory clearance (such as FDA SaMD)."*
