# Final Test Set Evaluation Results

> Evaluated once on held-out `test.csv` (212 samples, 80/20 stratified split).

| model | accuracy | precision | recall | f1 | specificity | balanced_accuracy | roc_auc | pr_auc | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rf | 0.9953 | 0.9938 | 1.0000 | 0.9969 | 0.9804 | 0.9902 | 0.9999 | 1.0000 | 50 | 1 | 0 | 161 |
| svm | 0.9953 | 0.9938 | 1.0000 | 0.9969 | 0.9804 | 0.9902 | 1.0000 | 1.0000 | 50 | 1 | 0 | 161 |
| knn | 0.9953 | 0.9938 | 1.0000 | 0.9969 | 0.9804 | 0.9902 | 0.9902 | 0.9938 | 50 | 1 | 0 | 161 |
| dt | 0.9953 | 0.9938 | 1.0000 | 0.9969 | 0.9804 | 0.9902 | 0.9902 | 0.9938 | 50 | 1 | 0 | 161 |
| gb | 0.9953 | 0.9938 | 1.0000 | 0.9969 | 0.9804 | 0.9902 | 1.0000 | 1.0000 | 50 | 1 | 0 | 161 |

## Artifacts Generated
- Metrics JSON: `reports/metrics/test_metrics.json`
- Metrics CSV: `reports/metrics/test_metrics.csv`
- ROC Curves: `reports/figures/eval_roc_curves.png`
- Precision-Recall Curves: `reports/figures/eval_pr_curves.png`
- Comparison Bar Chart: `reports/figures/eval_comparison_bar.png`
- Confusion Matrices: `reports/figures/eval_cm_<model>.png`
