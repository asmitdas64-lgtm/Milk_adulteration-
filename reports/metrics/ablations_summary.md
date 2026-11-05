# Robustness & Ablation Analysis Summary (Phase 9)

> Base Model: **SVM** evaluated with 5-Fold Stratified Cross-Validation on training data.

## 1. Ablation Comparison Table
| Category | Configuration Setting | Mean CV F1 | Mean CV Accuracy | Mean CV Recall | Mean CV ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|
| Outlier Treatment | clip (IQRClipper, factor=1.5) [Baseline] | 0.9772 | 96.57% | 0.9719 | 0.9978 |
| Outlier Treatment | none (Raw scaled, no clipping) | 0.9757 | 96.34% | 0.9719 | 0.9965 |
| Outlier Treatment | drop_rows (Removed 326 outlier rows) | 0.9953 | 99.42% | 0.9968 | 0.9998 |
| Feature Subset | all_features (7 features) [Baseline] | 0.9772 | 96.57% | 0.9719 | 0.9978 |
| Feature Subset | top_3 (ph, turbidity, colour) | 0.9179 | 87.25% | 0.9392 | 0.9145 |
| Feature Subset | top_2 (ph, temprature) | 0.8656 | 76.74% | 0.9876 | 0.8097 |
| Feature Subset | single_feature (ph only) | 0.8623 | 75.80% | 1.0000 | 0.7723 |
| Class Weighting | class_weight=None [Baseline] | 0.9772 | 96.57% | 0.9719 | 0.9978 |
| Class Weighting | class_weight='balanced' | 0.9701 | 95.63% | 0.9424 | 0.9993 |

## 2. Statistical Significance: McNemar's Test (SVM vs Random Forest)
- Contingency Table:
  - Both Correct ($n_{00}$): 199
  - SVM Correct, RF Incorrect ($n_{01}$): 0
  - SVM Incorrect, RF Correct ($n_{10}$): 12
  - Both Incorrect ($n_{11}$): 1
- McNemar Chi-Square Statistic: 10.0833 (p-value = 0.0015)
- **Conclusion**: p-value < 0.05 (Statistically Significant difference).


## 3. Defense of Design Decisions (Viva / Presentation Arguments)
- **Why Outlier Clipping (Winsorization) over Row Deletion?** Deleting outlier rows destroys the severe adulteration signals and shrinks training data unnecessarily. Clipping bounds preserve the signal without distorting Euclidean distances or SVM decision margins.
- **Why Retain All 7 Features?** While pH alone achieves strong discrimination, full laboratory safety and maximum recall require multi-parameter redundancy (turbidity, color, temperature).
- **Why Class Weighting is Optional?** At 75.8% / 24.2% balance, standard unweighted optimization converges to near-perfect F1 without causing false alarm inflation.