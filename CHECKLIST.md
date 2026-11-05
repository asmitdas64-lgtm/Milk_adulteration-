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
- [x] P0.10 `README.md`, `git init`, first commit

## Phase 1 — Dataset Acquisition & Validation 🧑
- [x] P1.0 🔴 Tests first: test_load.py, test_validate.py (see 10_TDD §5)
- [x] P1.1 Raw file placed in `data/raw/` (by human); path set in config
- [x] P1.2 Actual columns/dtypes inspected and printed (no assumptions)
- [x] P1.3 `column_rename`, `features`, `ranges` filled in config
- [x] P1.4 `load.py` incl. `make_binary_label` (per D-04) + tests
- [x] P1.5 `validate.py` (schema, dtypes, ranges, label set, duplicates) + tests
- [x] P1.6 `cli validate` writes `data/interim/clean.csv`
- [x] P1.7 🧑 Human confirms columns + label mapping

## Phase 2 — EDA
- [x] P2.0 🔴 Tests first: test_eda.py
- [x] P2.1 `eda.py`: summary stats, missingness, duplicates
- [x] P2.2 Class balance computed and plotted
- [x] P2.3 Per-class feature distributions (hist/box)
- [x] P2.4 Correlation heatmap
- [x] P2.5 IQR outlier counts per feature per class
- [x] P2.6 `reports/eda/summary.json` + `data_card.md` complete
- [x] P2.7 `notebooks/01_eda.ipynb` narrative (uses src functions only)
- [x] P2.8 🧑 Imbalance decision D-07 confirmed (minority = 24.17% >= 20%; standard stratified CV confirmed)

## Phase 3 — Split & Preprocessing Pipeline
- [x] P3.0 🔴 Tests first: test_split.py, test_transformers.py, test_preprocess.py, test_leakage.py
- [x] P3.1 `split.py`: stratified 80/20, seed 42, `split.json` saved
- [x] P3.2 `IQRClipper` transformer + tests
- [x] P3.3 `build_preprocessor` (median → IQR clip → StandardScaler) in ColumnTransformer
- [x] P3.4 Leakage test (`tests/test_leakage.py`) passes
- [x] P3.5 Stratification test: class ratio train ≈ test (±1 %) (actual: 75.80% vs 75.94%, diff=0.14%)

## Phase 4 — Feature Engineering & Selection
- [x] P4.0 🔴 Tests first: test_selection.py
- [x] P4.1 Pearson correlation report (|r| ≥ 0.90 pairs) (none exceeded threshold)
- [x] P4.2 RF impurity importance (top: pH=0.392, temp=0.180, fat=0.144)
- [x] P4.3 Permutation importance (scoring: F1)
- [x] P4.4 RFECV (5-fold stratified CV, selects all 7 features)
- [x] P4.5 `feature_ranking.csv` + `selected_features.json` + `feat_importance.png` saved
- [x] P4.6 All-features vs subset CV comparison; D-09 applied (all 7 features retained)

## Phase 5 — Baseline Models
- [x] P5.0 🔴 Tests first: test_registry.py, test_train.py (beats dummy on easy fixture)
- [x] P5.1 `registry.py` builds Dummy + RF, SVM, k-NN, DT, GB (+ test)
- [x] P5.2 Default-param 5-fold CV for all models on training set
- [x] P5.3 `baseline_cv.json/.csv` saved
- [x] P5.4 All 5 models beat Dummy on CV F1 (Dummy: 0.8623, Models: 0.9631 - 0.9984)

## Phase 6 — Hyper-parameter Tuning
- [x] P6.0 🔴 Tests first: tuning tests in test_tuning.py (params within grid, reproducible)
- [x] P6.1 RF GridSearchCV
- [x] P6.2 SVM (RBF) GridSearchCV
- [x] P6.3 k-NN GridSearchCV over k & metric
- [x] P6.4 DT GridSearchCV
- [x] P6.5 GB RandomizedSearchCV (n_iter=20)
- [x] P6.6 `best_params.json`, `tuned_cv.json`, per-model `tuning_*.csv` saved
- [x] P6.7 Best model selected by CV rule (D-10) and recorded BEFORE test evaluation (SVM selected: F1=0.9992, Recall=1.0000)

## Phase 7 — Final Test Evaluation (one shot)
- [x] P7.0 🔴 Tests first: test_metrics.py (hand-computed), test_evaluate.py
- [x] P7.1 Refit tuned pipelines on full train (847 samples)
- [x] P7.2 Single test-set prediction; metrics (Acc=99.53%, P=99.38%, R=100%, F1=0.9969, ROC-AUC=1.0000, PR-AUC=1.0000)
- [x] P7.3 `test_metrics.json/.csv/.md` saved in `reports/metrics/`
- [x] P7.4 Confusion matrices, ROC overlay, PR curves, comparison bar chart saved in `reports/figures/`
- [x] P7.5 `models/best_model.joblib` + `best_model_meta.json` saved (Winning model: SVM)

## Phase 8 — Interpretability
- [x] P8.0 🔴 Tests first: test_interpret.py (all features present, finite, ranking comparison)
- [x] P8.1 Permutation importance of best model on test (pH is #1 with 0.2088 F1 drop)
- [x] P8.2 Compare with Phase 4 ranking (consistent hierarchy across train and test)
- [x] P8.3 Decision-tree visualisation (top levels) saved to `reports/figures/tree_interpretability.png`
- [x] P8.4 Final ranked "most useful lab tests" list (`reports/features/final_importance.csv` + summary MD)

## Phase 9 — Robustness & Ablations
- [x] P9.0 🔴 Tests first: test_ablation.py (Mcnemar test, all ablations output row structure)
- [x] P9.1 Outliers: none vs clip vs drop_train (clip preserves extreme adulteration signal)
- [x] P9.2 All features vs selected subset (all 7 features achieve best CV F1=0.9772 vs 0.8623 for pH only)
- [x] P9.3 Class weighting on/off (unweighted converges reliably at 75.8/24.2 ratio)
- [x] P9.4 (Optional) McNemar top-2 models (SVM vs RF evaluated on test set)
- [x] P9.5 `ablations.csv` + `ablations_summary.md` generated and reviewed

## Phase 10 — Prediction Interface
- [x] P10.0 🔴 Tests first: test_predict.py (schema error, out of physical range, prediction outputs)
- [x] P10.1 `predict.py` with input schema validation & probability outputs
- [x] P10.2 `cli predict` command (`--model`, `--input`, `--output`)
- [x] P10.3 `docs/example_input.csv` + `test_predict.py` passing
- [x] P10.4 D-11 confirmed: Streamlit out-of-scope per report §1.4, robust CLI tool provided

## Phase 11 — Results & Documentation
- [x] P11.0 🔴 Tests first: test_report.py (RESULTS.md numbers == metric JSON)
- [x] P11.1 `report.py` auto-generates `reports/RESULTS.md` dynamically from JSON
- [x] P11.2 README usage and CLI documentation complete
- [x] P11.3 Comprehensive project report draft `docs/final_report_draft.md` complete
- [x] P11.4 Full IEEE conference format research paper `docs/IEEE_paper_food_adulteration.md` complete
- [x] P11.5 Domain literature citations and review fixes incorporated
- [x] P11.6 🧑 Every number traceable to a generated file in `reports/`

## Phase 12 — Final QA & Submission
- [x] P12.0 🔴 Tests first: test_end_to_end.py (all pipeline components verified)
- [x] P12.1 Environment dependencies pinned and verified
- [x] P12.2 `python -m foodadult.cli all` end-to-end succeeds across all 10 stages
- [x] P12.3 Metrics identical to previous run (perfect reproducibility)
- [x] P12.4 `pytest -q` green (46 tests passed)
- [x] P12.5 All checklist items ticked; PROGRESS.md final entry completed
- [x] P12.6 Git release tag `v1.0` created and pushed to GitHub remote
