# 06 — Project Phases (strictly sequential)

Mapping to report Table 4.3: Stage 1 = Phase 0-pre (done) · Stage 2 = Phases 1–3 · Stage 3 = Phases 4–6 · Stage 4 = Phases 7–12.

Each phase: **Goal → Entry criteria → Tasks → Outputs → Exit criteria**. Checklist IDs in `CHECKLIST.md` use the same numbering (e.g. `P3.2`). 🧑 = human gate (stop and wait for approval).

---

### Phase 0 — Project Setup & Environment
- **Goal:** runnable skeleton.
- **Entry:** context files present.
- **Tasks:** create layout from `05_ARCHITECTURE.md`; venv (Python 3.11); `requirements.txt` (pinned); `pyproject.toml` (src layout, package `foodadult`); `.gitignore`; `configs/default.yaml` skeleton; `logging_utils.py`; `config.py`; `cli.py` with stub subcommands; `README.md`; `git init`; copy report into `docs/`.
- **Outputs:** skeleton repo, `pytest` passes (smoke test), `python -m foodadult.cli --help` works.
- **Exit:** commands above succeed; PROGRESS.md updated.

### Phase 1 — Dataset Selection, Acquisition & Validation 🧑
- **Goal:** a real, verified dataset in `data/raw/` and a validated interim file.
- **Entry:** Phase 0 done. **Blocked until D-01, D-02, D-04 are LOCKED by human.**
- **Tasks:** human places file in `data/raw/`; agent inspects actual columns/dtypes (prints `df.info()`, `head()`); fill `column_rename`, `features`, `ranges` in config; implement `load.py` (+ `make_binary_label` per D-04), `validate.py`; write interim CSV; unit tests.
- **Outputs:** `data/interim/clean.csv`, validation report in log, `tests/test_validate.py`.
- **Exit:** validation passes or all violations documented and approved; human confirms columns/label mapping. 🧑

### Phase 2 — Exploratory Data Analysis
- **Goal:** understand data; produce data card.
- **Entry:** Phase 1 done.
- **Tasks:** `eda.py`: summary stats, missingness, class balance, duplicates, per-class distributions (hist/box/violin), correlation heatmap, outlier counts per feature (IQR) per class; notebook for narrative.
- **Outputs:** `reports/eda/summary.json`, `reports/eda/data_card.md`, `reports/figures/eda_*.png`.
- **Exit:** data card complete; class imbalance ratio known → decide D-07 🧑 if minority < 35 %.

### Phase 3 — Splitting & Leak-free Preprocessing Pipeline
- **Goal:** locked test set + reusable preprocessing.
- **Entry:** Phase 2 done.
- **Tasks:** `split.py` (stratified 80/20, seed 42, save indices); `IQRClipper`; `build_preprocessor`; tests: transformer behaviour, no-leakage test, split stratification test.
- **Outputs:** `data/processed/{train,test}.csv`, `split.json`, tests green.
- **Exit:** class ratio train≈test (±1 %), leakage test passes.

### Phase 4 — Feature Engineering & Selection
- **Goal:** ranked features + selected subset (Objective 4).
- **Entry:** Phase 3 done.
- **Tasks:** correlation report, RF importance, permutation importance, RFECV — training data only; compare all vs subset (D-09).
- **Outputs:** `reports/features/feature_ranking.csv`, `selected_features.json`, `reports/figures/feat_*.png`.
- **Exit:** `features.use_selected_subset` set in config with justification in decision log.

### Phase 5 — Baseline Models
- **Goal:** sanity check with default hyper-parameters.
- **Entry:** Phase 4 done.
- **Tasks:** `registry.py`; train Dummy + 5 models with defaults using 5-fold CV on train; record CV metrics.
- **Outputs:** `reports/metrics/baseline_cv.json/.csv`.
- **Exit:** all 5 models beat Dummy on CV F1. If not → investigate (label bug? leakage? scaling?) before continuing.

### Phase 6 — Hyper-parameter Tuning
- **Goal:** optimised models (report Table 4.2).
- **Entry:** Phase 5 done.
- **Tasks:** `train.py`: Grid/RandomizedSearchCV per §6 of `04_ML_METHODOLOGY.md`; save `cv_results_` per model, best params, timing.
- **Outputs:** `reports/metrics/tuning_<model>.csv`, `reports/metrics/best_params.json`, `reports/metrics/tuned_cv.json`.
- **Exit:** best params for all 5 models saved; best model chosen by CV rule (D-10) **before** looking at test.

### Phase 7 — Final Evaluation on Test Set (one shot)
- **Goal:** unbiased comparison.
- **Entry:** Phase 6 done; model-selection decision recorded.
- **Tasks:** refit each tuned pipeline on full train; predict test once; compute metrics; confusion matrices; overlaid ROC + PR curves; comparison bar chart; save `best_model.joblib` + meta.
- **Outputs:** `reports/metrics/test_metrics.json/.csv/.md`, `reports/figures/eval_*.png`, `models/*.joblib`.
- **Exit:** metrics files exist; no re-tuning after viewing test results (if needed, log as D-xx and declare it in the report).

### Phase 8 — Interpretability & Feature Ranking Analysis
- **Goal:** "which lab tests matter most" (Outcome 5).
- **Entry:** Phase 7 done.
- **Tasks:** permutation importance of best model on test; compare with Phase 4 ranking; DT visualisation (top levels) for interpretability.
- **Outputs:** `reports/features/final_importance.csv`, figures.
- **Exit:** ranked list with consistency notes.

### Phase 9 — Robustness & Ablation Studies (recommended)
- **Goal:** defend design choices.
- **Entry:** Phase 8 done.
- **Tasks:** ablations from `04_ML_METHODOLOGY.md` §9 using CV on train; optional McNemar between top-2.
- **Outputs:** `reports/metrics/ablations.csv`.
- **Exit:** results summarised; human decides whether anything changes 🧑.

### Phase 10 — Prediction Interface
- **Goal:** report stage 6 "Final Prediction Generation".
- **Entry:** Phase 7 done.
- **Tasks:** `predict.py` + CLI; input schema validation; `tests/test_predict.py`; example input `docs/example_input.csv`.
- **Outputs:** working `predict` command.
- **Exit:** test passes; example runs. (Optional Streamlit demo only if D-11 approves.)

### Phase 11 — Results Report & Documentation
- **Goal:** content for the final project report.
- **Entry:** Phases 7–10 done.
- **Tasks:** `report.py` auto-builds `reports/RESULTS.md` (tables + figure links) from metric files; README usage; draft "Results & Discussion", "Conclusion", "Limitations", "Future Work" sections in `docs/final_report_draft.md` — numbers only from files; literature citations only those provided by the team.
- **Outputs:** `RESULTS.md`, `docs/final_report_draft.md`.
- **Exit:** every number traceable to a file 🧑 human review.

### Phase 12 — Final QA, Reproducibility & Submission
- **Goal:** clean, reproducible hand-in.
- **Entry:** Phase 11 approved.
- **Tasks:** fresh venv → `pip install -r requirements.txt` → `python -m foodadult.cli all` → compare metrics with previous run (must match); `pytest` green; lint; remove dead code; final checklist review; tag release.
- **Outputs:** reproducibility log `reports/runs/<ts>/`, final zip/git tag.
- **Exit:** all CHECKLIST items ticked; guide sign-off 🧑.

