# Empirical Machine Learning Results Summary

> **B.Tech Minor Project**: Machine Learning for Predicting Food Adulteration  
> **Department of Computer Science & Engineering, MIT Bengaluru (AY 2025–26)**  
> **Authors**: Asmit Das (235891248), Suprathik Narendra (235891216), Nadia Cecil (235891048)  
> **Guide**: Dr. S. Vaishnavi  

---

## 1. Executive Summary
- **Selected Best Classifier**: **SVM** (Decision D-10: highest 5-Fold CV F1 with Recall tie-breaker, selected prior to test scoring).
- **Held-Out Test Set Size**: 212 unseen samples (80/20 stratified split, locked until Phase 7).
- **Test Set Performance**:
  - **F1 Score**: `0.9969`
  - **Accuracy**: `99.53%` (211 / 212 correct)
  - **Recall (Sensitivity)**: `100.00%` (161 / 161 adulterated batches caught)
  - **Precision**: `99.38%`
  - **Specificity (Pure Milk Recall)**: `98.04%` (50 / 51 pure batches caught)
  - **ROC-AUC**: `1.0000`
  - **False Negatives**: **`0`** (Zero adulterated samples missed!)

---

## 2. Model Performance on Held-Out Test Set (One-Shot Evaluation)
| Model | Accuracy | Precision | Recall | F1 Score | Specificity | ROC-AUC | True Positives | False Negatives |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RF** | 99.53% | 99.38% | 100.00% | 0.9969 | 98.04% | 0.9999 | 161 | 0 |
| **SVM** | 99.53% | 99.38% | 100.00% | 0.9969 | 98.04% | 1.0000 | 161 | 0 |
| **KNN** | 99.53% | 99.38% | 100.00% | 0.9969 | 98.04% | 0.9902 | 161 | 0 |
| **DT** | 99.53% | 99.38% | 100.00% | 0.9969 | 98.04% | 0.9902 | 161 | 0 |
| **GB** | 99.53% | 99.38% | 100.00% | 0.9969 | 98.04% | 1.0000 | 161 | 0 |

---

## 3. Diagnostic Parameter Hierarchy (Permutation Importance)
| Rank | Parameter | Test Permutation Importance (F1 drop) | Train RF Impurity Importance | Role in Milk Safety |
|:---:|:---|:---:|:---:|:---|
| 1 | `ph` | 0.2088 | 0.3925 | Primary physicochemical indicator |
| 2 | `turbidity` | 0.0839 | 0.0482 | Primary physicochemical indicator |
| 3 | `colour` | 0.0654 | 0.0663 | Primary physicochemical indicator |
| 4 | `odor` | 0.0600 | 0.1230 | Primary physicochemical indicator |
| 5 | `temprature` | 0.0397 | 0.1804 | Primary physicochemical indicator |
| 6 | `fat` | 0.0382 | 0.1443 | Primary physicochemical indicator |
| 7 | `taste` | 0.0355 | 0.0453 | Primary physicochemical indicator |

---

## 4. Robustness & Ablation Analysis
| Category | Configuration | Mean CV F1 | Mean CV Accuracy | Mean CV Recall |
|:---|:---|:---:|:---:|:---:|
| Outlier Treatment | clip (IQRClipper, factor=1.5) [Baseline] | 0.9772 | 96.57% | 0.9719 |
| Outlier Treatment | none (Raw scaled, no clipping) | 0.9757 | 96.34% | 0.9719 |
| Outlier Treatment | drop_rows (Removed 326 outlier rows) | 0.9953 | 99.42% | 0.9968 |
| Feature Subset | all_features (7 features) [Baseline] | 0.9772 | 96.57% | 0.9719 |
| Feature Subset | top_3 (ph, turbidity, colour) | 0.9179 | 87.25% | 0.9392 |
| Feature Subset | top_2 (ph, temprature) | 0.8656 | 76.74% | 0.9876 |
| Feature Subset | single_feature (ph only) | 0.8623 | 75.80% | 1.0000 |
| Class Weighting | class_weight=None [Baseline] | 0.9772 | 96.57% | 0.9719 |
| Class Weighting | class_weight='balanced' | 0.9701 | 95.63% | 0.9424 |

---

## 5. Visual Artifacts & Diagnostic Plots
- **ROC Curves Overlay**: [`reports/figures/eval_roc_curves.png`](reports/figures/eval_roc_curves.png)
- **Precision-Recall Curves**: [`reports/figures/eval_pr_curves.png`](reports/figures/eval_pr_curves.png)
- **Model Comparison Chart**: [`reports/figures/eval_comparison_bar.png`](reports/figures/eval_comparison_bar.png)
- **Confusion Matrix (Best Model)**: [`reports/figures/eval_cm_svm.png`](reports/figures/eval_cm_svm.png)
- **Feature Importance Chart**: [`reports/figures/interpret_perm_importance.png`](reports/figures/interpret_perm_importance.png)
- **White-Box Decision Tree Flowchart**: [`reports/figures/tree_interpretability.png`](reports/figures/tree_interpretability.png)

> *All numbers generated automatically from verified experimental artifacts.*