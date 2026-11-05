# 04 — ML Methodology (implementation spec)

Based on report Chapter 4, with corrections recorded in `08_DECISIONS_LOG.md`.

## 1. Pipeline stages (report §4.1 — six stages)
1. Data acquisition → 2. Preprocessing & cleaning → 3. Feature engineering & selection → 4. Model training & optimisation → 5. Evaluation → 6. Final prediction generation (CLI `predict`).

```
raw CSV ──validate──► interim ──stratified split (80/20, seed 42)──► train | test (locked)
train ──► Pipeline[ ColumnTransformer( MedianImputer → IQRClipper → StandardScaler ) → [Selector] → Classifier ]
       └─ StratifiedKFold(5) GridSearch/RandomizedSearch  ──► best params per model
best per model ──refit on full train──► evaluate ONCE on test ──► compare ──► best_model.joblib
```

## 2. Leakage guard (D-08 — corrects report §4.3 order)
The report lists split as step 4 *after* imputation/outliers/scaling. That leaks test statistics. **Correct order:** split first; imputer, IQR bounds, scaler and selectors are fitted inside the Pipeline on training folds only. A unit test (`tests/test_leakage.py`) must assert the transformers' learned statistics equal those computed on the training data only.

## 3. Preprocessing
| Step | Implementation | Notes |
|---|---|---|
| Imputation | `SimpleImputer(strategy="median")` | report §4.3.1 |
| Outliers | custom `IQRClipper(factor=1.5)` transformer (`fit` learns Q1/Q3 per column; `transform` clips) | D-05: clip instead of drop, because adulterated samples may *be* the outliers. Ablation: `outliers: none` vs `clip` |
| Scaling | `StandardScaler()` | needed for SVM/k-NN; harmless for trees |
| Categorical (only if `food_type` exists) | `OneHotEncoder(handle_unknown="ignore")` | only if D-01 = multi-food |

## 4. Feature engineering & selection (report §4.5 + Objective 4)
Run on **training data only**:
1. Pearson correlation matrix + heatmap; list pairs with |r| ≥ 0.90; for each pair, propose dropping the one with lower |corr| with target.
2. Random Forest impurity importance (default RF, seed 42).
3. Permutation importance (`sklearn.inspection.permutation_importance`, scoring F1, n_repeats=10) on a CV validation fold.
4. `RFECV(estimator=RandomForestClassifier, cv=StratifiedKFold(5), scoring="f1")`.
5. Output: `reports/features/feature_ranking.csv` (all methods side by side) and `selected_features.json`.
6. Decision: compare CV F1 of **all features vs selected subset** for every model; use the subset only if it does not reduce mean CV F1 by more than 0.01 (D-09).

No new engineered features (ratios, polynomials) unless approved in the decision log.

## 5. Models (report §4.6) — scikit-learn only
| Key | Class | Fixed params |
|---|---|---|
| `dummy` | `DummyClassifier` | `strategy="most_frequent"` (sanity baseline, added) |
| `rf` | `RandomForestClassifier` | `random_state=42, n_jobs=-1` |
| `svm` | `SVC` | `kernel="rbf", probability=True, random_state=42` |
| `knn` | `KNeighborsClassifier` | – |
| `dt` | `DecisionTreeClassifier` | `criterion="gini", random_state=42` |
| `gb` | `GradientBoostingClassifier` | `random_state=42` |

If D-07 = class weighting: add `class_weight="balanced"` to rf, svm, dt (GB/k-NN: none).

## 6. Hyper-parameter search (report Table 4.2 — only these params)
All searches: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, `scoring="f1"`, `refit="f1"`, also record accuracy/precision/recall/roc_auc via multi-metric scoring. Params are prefixed `clf__` in code.

| Model | Search | Grid |
|---|---|---|
| RF | GridSearchCV | `n_estimators: [100, 300, 500]`, `max_depth: [None, 5, 10, 20]`, `min_samples_leaf: [1, 2, 4]` |
| SVM | GridSearchCV | `C: [0.1, 1, 10, 100]`, `gamma: ["scale", 0.001, 0.01, 0.1, 1]` |
| k-NN | GridSearchCV (= K-fold CV over k) | `n_neighbors: [1,3,5,...,31]` (odd), `metric: ["euclidean", "manhattan"]` |
| DT | GridSearchCV | `max_depth: [None, 3, 5, 7, 10, 15]`, `min_samples_split: [2, 5, 10, 20]` |
| GB | RandomizedSearchCV, `n_iter=30`, `random_state=42` | `learning_rate: loguniform(0.01, 0.3)`, `n_estimators: randint(50, 500)` |

Grids live in `configs/default.yaml`, not hard-coded.

## 7. Metrics (report §4.7)
- Accuracy, Precision, Recall, F1 (positive = 1), ROC-AUC. Additionally recorded: confusion matrix, PR-AUC (average precision), balanced accuracy.
- Report **CV mean ± std** (from search best index) and **test** values.
- Domain note: in food safety a missed adulterated sample (FN) is worse than a false alarm (FP) → recall is highlighted in discussion.

## 8. Model-selection rule (D-10)
1. Highest mean CV F1 (NOT test F1 — test is for final reporting only).
2. Tie (difference < 0.01): higher mean CV recall.
3. Still tied: simpler/more interpretable model (DT < kNN < RF < GB < SVM for interpretability ordering is debatable — document reasoning).
Optional: McNemar's test between the top-2 models on test predictions (report as supporting evidence only).

## 9. Ablations (Phase 9, optional but recommended)
- Outliers: `none` vs `clip` vs `drop-train-rows`.
- Features: all vs selected.
- Class weighting on/off (if imbalanced).
- If synthetic data used: train real+synthetic, test **real only**.

## 10. Reproducibility
Seed 42 everywhere; save `config_snapshot.yaml`, `pip freeze`, git commit hash (if git) into `reports/runs/<timestamp>/`.

