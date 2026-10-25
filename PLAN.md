# PLAN — CardioScan AI: modernization of the Cleveland heart-disease replication

This is the plan we agreed on before implementation started, kept up to date as
phases land. The original replication (paper code, `run_replication.py`, static
`results/`) stays intact; everything new is additive.

## What exists

- `run_replication.py` + `src/` — the original experiment: 303 patients, six
  pipelines, strict 70/30 split, tables/figures written to `results/`.
- `app.py` — the original legacy UI, untouched.
- `results/` — committed CSV artifacts (tables 1–32), the ground truth every
  view reads from.

## What we are changing

| # | Area | Change |
|---|------|--------|
| 1 | Backend | FastAPI service (`backend/`) that loads the fitted models once and serves inference + `results/` tables as JSON. |
| 2 | Frontend | A React + Vite + TypeScript + Tailwind app (`frontend/`) with a dark "AI lab" glass UI: sidebar nav, live backend indicator, charts (recharts), shared card/table primitives. |
| 3 | Statistics | `run_extensions.py` — the studies the paper only reports as point estimates: repeated CV stability, exact McNemar pairwise tests, permutation importance, threshold trade-offs, HGB comparison. Written as tables 33–38. |
| 4 | API | `/api/stats/{table}` route exposing tables 33–38 alongside the existing model/results routes. |
| 5 | Tests | pytest suite covering model loading, prediction contract, endpoints, and the exact statistics (McNemar). |

## Phases

- [x] **Phase 1** — FastAPI backend + tests (models, prediction contract,
  results/audit endpoints).
- [x] **Phase 2** — React frontend: scaffold, API client, shared UI
  primitives, and five pages (Diagnosis, Benchmarks, Audit, Hyperparameter
  Lab, EDA).
- [x] **Phase 3** — Statistical extensions: `run_extensions.py` (tables 33–38),
  `/api/stats/*` route, tests, and the **Statistical Lens** page.
- [x] **Phase 4** — Verification: full pytest run, `tsc` typecheck + production
  build, browser walkthrough of every page against live data, README updates.
- [x] **Phase 5** — Simplification (user request): dashboard cut to three core
  pages — Diagnosis, Benchmarks, Audit. The Hyperparameter Lab, EDA and
  Statistical Lens pages were deleted from the app (their content stays
  available in `results/`, the API routes, README and DOCUMENTATION), page
  copy tightened, dead client/UI code removed, docs updated to match.

## Findings worth keeping

- Repeated 10×10 CV reorders the leaderboard: Logistic Regression leads on
  mean accuracy, and most pairwise gaps are **not** significant — only 3 of 21
  McNemar comparisons clear p < 0.05 (the Decision Tree is the genuinely weak
  model).
- At n = 91 the held-out accuracy moves ≈ 1.1 points per patient (95% CI ≈
  ±7 pts), so single-split rankings are noise.
- A `sensitivity ≥ 0.90` operating point degenerates on this dataset: with
  calibrated probabilities clustered near 0.5 the threshold sweep cannot reach
  high sensitivity without collapsing specificity.
- HistGradientBoosting does **not** beat the paper's best model on 303 rows —
  modern boosting doesn't buy anything at this sample size.

## Rules of the road

- Every metric shown comes from the held-out 30% split, never training scores.
- Static charts read committed CSVs; only inference needs the backend alive.
- Original paper pipeline and its outputs are never modified.
