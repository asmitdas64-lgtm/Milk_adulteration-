# 02 — Requirements

Each requirement has an ID. Code, tests and checklist items should reference these IDs.

## Mapping to report objectives (Chapter 3)
| Report objective | Requirements |
|---|---|
| Obj 1 — curate dataset | FR-01, FR-02, FR-03 |
| Obj 2 — preprocess | FR-04 – FR-08 |
| Obj 3 — five classifiers | FR-10, FR-11 |
| Obj 4 — feature engineering/selection | FR-09 |
| Obj 5 — evaluate & benchmark | FR-12 – FR-15 |
| Obj 6 — propose framework | FR-16, FR-17 |

## Functional requirements
| ID | Requirement |
|---|---|
| FR-01 | Load the raw dataset from `data/raw/` without modifying the raw file (raw data is immutable). |
| FR-02 | Validate schema (column names, dtypes, value ranges, label values ∈ {0,1}) per `03_DATA_SPEC.md`; fail loudly on violation. |
| FR-03 | Produce a data card `reports/eda/data_card.md` + `reports/eda/summary.json` (rows, columns, missing %, class balance, duplicates). |
| FR-04 | Stratified train/test split 80/20, `random_state=42`, performed **before** any fitting. Persist split indices to `data/processed/split.json`. |
| FR-05 | Median imputation for missing numeric values (fit on train folds only). |
| FR-06 | IQR-based outlier handling (default: clipping to [Q1−1.5·IQR, Q3+1.5·IQR], fit on train folds only) — see D-05. |
| FR-07 | StandardScaler on continuous features (fit on train folds only). |
| FR-08 | All of FR-05..07 implemented as a single sklearn `Pipeline` with `ColumnTransformer`. |
| FR-09 | Feature selection: Pearson correlation analysis (flag pairs with \|r\| ≥ 0.90), RF impurity importance, permutation importance, and RFE/RFECV. Output ranked list + chosen subset. |
| FR-10 | Implement RF, SVM (RBF), k-NN, DT, GB (sklearn) + `DummyClassifier(strategy="most_frequent")` baseline. |
| FR-11 | Hyper-parameter tuning with Stratified 5-fold CV on the training set only, grids from `04_ML_METHODOLOGY.md` §6. |
| FR-12 | Compute Accuracy, Precision, Recall, F1 (positive class = 1), ROC-AUC, confusion matrix — CV (mean ± std) and held-out test. |
| FR-13 | Save metrics as JSON + CSV + Markdown table in `reports/metrics/`. |
| FR-14 | Generate figures in `reports/figures/`: correlation heatmap, class balance, feature distributions by class, confusion matrices, overlaid ROC curves, PR curves, feature-importance bar charts, model comparison bar chart. |
| FR-15 | Select the best model by the rule in `04_ML_METHODOLOGY.md` §8 and save it as `models/best_model.joblib` with a metadata JSON. |
| FR-16 | CLI `predict` command: input CSV → output CSV with `prediction` and `probability_adulterated`. |
| FR-17 | Final results summary `reports/RESULTS.md` auto-generated from metric files (no hand-typed numbers). |

## Non-functional requirements
| ID | Requirement |
|---|---|
| NFR-01 | Reproducible: same config + same data ⇒ identical metrics. All seeds from config. |
| NFR-02 | No data leakage (see `04_ML_METHODOLOGY.md` §2). |
| NFR-03 | Python 3.11, dependencies pinned in `requirements.txt`. |
| NFR-04 | Runs on a normal laptop (CPU only) — full training < ~30 min. |
| NFR-05 | Unit tests with pytest for data validation, preprocessing, leakage guard, metrics, predict. |
| NFR-06 | Config-driven (`configs/default.yaml`); no magic numbers in code. |
| NFR-07 | Logging via `logging` module; every run writes `reports/runs/<timestamp>/run.log` + config snapshot. |
| NFR-08 | Code style: PEP 8, type hints, docstrings (see `07_CODING_STANDARDS.md`). |

