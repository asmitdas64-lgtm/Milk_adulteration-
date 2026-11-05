# CHECKLIST.md — Food Adulteration ML Project

Tick `[x]` only after the item is implemented **and verified** (test/command run). 🧑 = human must approve.
Phases are strictly sequential — see `context/06_PHASES.md`.

**TDD rule for every code item:** 🔴 write test(s) from `context/10_TDD_WORKFLOW.md` §5 first and see them fail → 🟢 write code until they pass → 🔵 refactor → full `pytest -q` green → tick. Each phase ends with a **Tests** item. Beginner explanations for every step: `context/09_BEGINNER_GUIDE.md`.

## Pre-work — Human decisions (before Phase 1)
- [x] H.1 🧑 Choose food category (D-01: Milk chosen)
- [x] H.2 🧑 Choose & verify dataset: name, URL, licence, columns (D-02: Milk Quality Dataset `milknew.csv`, 1059 rows)
- [x] H.3 🧑 Synthetic data allowed? (D-03: No, real data confirmed)
- [x] H.4 🧑 Label mapping rule if label not binary (D-04: low/medium -> 1, high -> 0)
- [x] H.5 🧑 Units for viscosity/density (D-06: features match milk columns)
- [x] H.6 🧑 Minimum dataset size (D-12: 1059 rows > 500)
- [ ] H.7 🧑 Confirm PROPOSED decisions D-05, D-07, D-09, D-10, D-13, D-14 with guide
- [x] H.8 Copy original report into `docs/` (done 2026-11-04: `.docx` + extracted `report_text.txt`)

## Phase 0 — Setup & Environment
- [x] P0.0 🔴 Tests first: test_smoke.py (imports, load_config, CLI --help)
- [x] P0.1 Folder layout per `context/05_ARCHITECTURE.md`
- [x] P0.2 Python 3.12 venv created
- [x] P0.3 `requirements.txt` pinned + installed
- [x] P0.4 `pyproject.toml` (src layout, package `foodadult`), `pip install -e .`
- [x] P0.5 `.gitignore` (venv, data/raw large files, models/*.joblib, __pycache__)
- [x] P0.6 `configs/default.yaml` skeleton
- [x] P0.7 `config.py` + `logging_utils.py` (run dir + config snapshot)
- [x] P0.8 `cli.py` with stub subcommands; `python -m foodadult.cli --help` works
- [x] P0.9 Smoke test; `pytest -q` green
- [ ] P0.10 `README.md`, `git init`, first commit

## Phase 1 — Dataset Acquisition & Validation 🧑
- [ ] P1.0 🔴 Tests first: test_load.py, test_validate.py (see 10_TDD §5)
- [ ] P1.1 Raw file placed in `data/raw/` (by human); path set in config
- [ ] P1.2 Actual columns/dtypes inspected and printed (no assumptions)
- [ ] P1.3 `column_rename`, `features`, `ranges` filled in config
- [ ] P1.4 `load.py` incl. `make_binary_label` (per D-04) + tests
- [ ] P1.5 `validate.py` (schema, dtypes, ranges, label set, duplicates) + tests
- [ ] P1.6 `cli validate` writes `data/interim/clean.csv`
- [ ] P1.7 🧑 Human confirms columns + label mapping

## Phase 2 — EDA
- [ ] P2.0 🔴 Tests first: test_eda.py
- [ ] P2.1 `eda.py`: summary stats, missingness, duplicates
- [ ] P2.2 Class balance computed and plotted
- [ ] P2.3 Per-class feature distributions (hist/box)
- [ ] P2.4 Correlation heatmap
- [ ] P2.5 IQR outlier counts per feature per class
- [ ] P2.6 `reports/eda/summary.json` + `data_card.md` complete
- [ ] P2.7 `notebooks/01_eda.ipynb` narrative (uses src functions only)
- [ ] P2.8 🧑 Imbalance decision D-07 confirmed

## Phase 3 — Split & Preprocessing Pipeline
- [ ] P3.0 🔴 Tests first: test_split.py, test_transformers.py, test_preprocess.py, test_leakage.py
- [ ] P3.1 `split.py`: stratified 80/20, seed 42, `split.json` saved
- [ ] P3.2 `IQRClipper` transformer + tests
- [ ] P3.3 `build_preprocessor` (median → IQR clip → StandardScaler) in ColumnTransformer
- [ ] P3.4 Leakage test (`tests/test_leakage.py`) passes
- [ ] P3.5 Stratification test: class ratio train ≈ test (±1 %)

## Phase 4 — Feature Engineering & Selection
- [ ] P4.0 🔴 Tests first: test_selection.py
- [ ] P4.1 Pearson correlation report (|r| ≥ 0.90 pairs)
- [ ] P4.2 RF impurity importance
- [ ] P4.3 Permutation importance
- [ ] P4.4 RFECV
- [ ] P4.5 `feature_ranking.csv` + `selected_features.json`
- [ ] P4.6 All-features vs subset CV comparison; D-09 applied; config updated

## Phase 5 — Baseline Models
- [ ] P5.0 🔴 Tests first: test_registry.py, test_train.py (beats dummy on easy fixture)
- [ ] P5.1 `registry.py` builds Dummy + RF, SVM, k-NN, DT, GB (+ test)
- [ ] P5.2 Default-param 5-fold CV for all models
- [ ] P5.3 `baseline_cv.json/.csv` saved
- [ ] P5.4 All 5 models beat Dummy on CV F1 (else investigate)

## Phase 6 — Hyper-parameter Tuning
- [ ] P6.0 🔴 Tests first: tuning tests in test_train.py (params within grid, reproducible)
- [ ] P6.1 RF GridSearchCV
- [ ] P6.2 SVM (RBF) GridSearchCV
- [ ] P6.3 k-NN GridSearchCV over k & metric
- [ ] P6.4 DT GridSearchCV
- [ ] P6.5 GB RandomizedSearchCV (n_iter=30)
- [ ] P6.6 `best_params.json`, `tuned_cv.json`, per-model `tuning_*.csv` saved
- [ ] P6.7 Best model selected by CV rule (D-10) and recorded BEFORE test evaluation

## Phase 7 — Final Test Evaluation (one shot)
- [ ] P7.0 🔴 Tests first: test_metrics.py (hand-computed), test_evaluate.py
- [ ] P7.1 Refit tuned pipelines on full train
- [ ] P7.2 Single test-set prediction; metrics (Acc, P, R, F1, ROC-AUC, PR-AUC, CM)
- [ ] P7.3 `test_metrics.json/.csv/.md`
- [ ] P7.4 Confusion matrices, ROC overlay, PR curves, comparison bar chart
- [ ] P7.5 `models/best_model.joblib` + `best_model_meta.json`

## Phase 8 — Interpretability
- [ ] P8.0 🔴 Tests first: test_interpret.py
- [ ] P8.1 Permutation importance of best model on test
- [ ] P8.2 Compare with Phase 4 ranking
- [ ] P8.3 Decision-tree visualisation (top levels)
- [ ] P8.4 Final ranked "most useful lab tests" list

## Phase 9 — Robustness & Ablations
- [ ] P9.0 🔴 Tests first: test_ablation.py
- [ ] P9.1 Outliers: none vs clip vs drop_train
- [ ] P9.2 All features vs selected subset
- [ ] P9.3 Class weighting on/off (if relevant)
- [ ] P9.4 (Optional) McNemar top-2 models
- [ ] P9.5 `ablations.csv` + 🧑 review

## Phase 10 — Prediction Interface
- [ ] P10.0 🔴 Tests first: test_predict.py (schema error, save/load identical)
- [ ] P10.1 `predict.py` with input schema validation
- [ ] P10.2 `cli predict` command
- [ ] P10.3 `docs/example_input.csv` + `test_predict.py`
- [ ] P10.4 (Only if D-11 = yes) Streamlit demo

## Phase 11 — Results & Documentation
- [ ] P11.0 🔴 Tests first: test_report.py (RESULTS.md numbers == metric JSON)
- [ ] P11.1 `report.py` auto-generates `reports/RESULTS.md`
- [ ] P11.2 README usage section complete
- [ ] P11.3 `docs/final_report_draft.md` (Results, Discussion, Limitations, Future Work)
- [ ] P11.4 Fix report issues listed in the review (typos, units, leakage order, figure/text mismatch)
- [ ] P11.5 Domain literature references added (provided/verified by team — no invented citations)
- [ ] P11.6 🧑 Every number traced to a file in `reports/`

## Phase 12 — Final QA & Submission
- [ ] P12.0 🔴 Tests first: test_end_to_end.py (runs twice → identical metrics); coverage ≥ 80 %
- [ ] P12.1 Fresh venv install from `requirements.txt`
- [ ] P12.2 `python -m foodadult.cli all` end-to-end succeeds
- [ ] P12.3 Metrics identical to previous run (reproducibility)
- [ ] P12.4 `pytest -q` green, `ruff check` clean
- [ ] P12.5 All checklist items ticked; PROGRESS.md final entry
- [ ] P12.6 🧑 Guide sign-off; git tag `v1.0`
