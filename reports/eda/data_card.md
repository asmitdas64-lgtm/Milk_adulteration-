# DATA CARD — Milk Quality & Adulteration Dataset

## 1. Overview & Provenance
- **Dataset Name:** Milk Quality Dataset (`clean.csv`)
- **Domain / Food Category:** Bovine Milk (Physicochemical & Organoleptic Screening)
- **Total Samples:** 1059
- **Total Columns:** 8 (7 features + 1 target)
- **Duplicate Rows (Discrete Combinations):** 976
- **Total Missing Values:** 0
- **Target Definition:** Binary classification where `0 = Pure / High Quality` and `1 = Adulterated / Substandard`.

## 2. Target Class Balance
- **Class 0 (Pure / High Quality):** 256 samples (24.17%)
- **Class 1 (Adulterated / Substandard):** 803 samples (75.83%)
- **Minority Class Proportion:** 24.17%
- **Decision D-07 Application:** Since the minority class is >= 24.2%, standard stratified CV is appropriate without mandatory synthetic oversampling (SMOTE).

## 3. Feature Descriptions & Statistics
| Feature | Type | Range / Values | Mean ± Std | Median | Outliers (IQR) | Description |
|---|---|---|---|---|---|---|
| `ph` | Continuous/Discrete | [3.0, 9.5] | 6.63 ± 1.40 | 6.70 | 379 | Physicochemical inspection parameter |
| `temprature` | Continuous/Discrete | [34.0, 90.0] | 44.23 ± 10.10 | 41.00 | 103 | Physicochemical inspection parameter |
| `taste` | Continuous/Discrete | [0.0, 1.0] | 0.55 ± 0.50 | 1.00 | 0 | Physicochemical inspection parameter |
| `odor` | Continuous/Discrete | [0.0, 1.0] | 0.43 ± 0.50 | 0.00 | 0 | Physicochemical inspection parameter |
| `fat` | Continuous/Discrete | [0.0, 1.0] | 0.67 ± 0.47 | 1.00 | 0 | Physicochemical inspection parameter |
| `turbidity` | Continuous/Discrete | [0.0, 1.0] | 0.49 ± 0.50 | 0.00 | 0 | Physicochemical inspection parameter |
| `colour` | Continuous/Discrete | [240.0, 255.0] | 251.84 ± 4.31 | 255.00 | 32 | Physicochemical inspection parameter |

## 4. IQR Outlier Analysis & Food Safety Note
In food safety and adulteration screening, extreme measurements (such as highly abnormal pH or temperature) are often genuine indicators of spoilage, chemical neutralizing agents, or watered-down milk.
Under decision **D-05**, outlier clipping (winsorization) is preferred over deleting rows, preserving the critical positive class signal.

## 5. Artifacts Generated
- `reports/eda/summary.json`: Complete numerical statistics snapshot.
- `reports/figures/eda_class_balance.png`: Target distribution chart.
- `reports/figures/eda_correlation_matrix.png`: Pearson correlation heatmap.
- `reports/figures/eda_distributions.png`: Per-class feature distributions.