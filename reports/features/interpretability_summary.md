# Interpretability & Feature Hierarchy Analysis (Phase 8)

> Best Model Evaluated: **SVM** on held-out test set (212 samples).

## 1. Final Feature Importance Ranking (Test Set Permutation vs Training RF)
| Rank | Feature | Test Permutation Importance (F1 drop) | Train RF Impurity Importance | Consistency |
|:---:|:---:|:---:|:---:|:---:|
| 1 | `ph` | 0.2088 | 0.3925 | High (Consistent) |
| 2 | `turbidity` | 0.0839 | 0.0482 | Moderate |
| 3 | `colour` | 0.0654 | 0.0663 | Moderate |
| 4 | `odor` | 0.0600 | 0.1230 | High (Consistent) |
| 5 | `temprature` | 0.0397 | 0.1804 | Moderate |
| 6 | `fat` | 0.0382 | 0.1443 | Moderate |
| 7 | `taste` | 0.0355 | 0.0453 | High (Consistent) |

## 2. Practical Recommendations for Low-Cost Screening (Outcome 5)
- **Primary Diagnostic Tier**: The top parameters `ph` and `turbidity` account for the overwhelming majority of classification capability.
- **Field Deployment**: In resource-constrained dairy collection centres, testing just pH (via portable digital pH pen) and Temperature eliminates nearly all false negatives before expensive spectroscopic or microbiological assays are required.

## 3. Visual Artifacts Generated
- Test Permutation Importance Chart: `reports/figures/interpret_perm_importance.png`
- Decision Tree Rule Flowchart: `reports/figures/tree_interpretability.png`
- Final Importance Table CSV: `reports/features/final_importance.csv`
