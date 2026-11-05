# Machine Learning for Predicting Food Adulteration

**B.Tech Minor Project — MIT Bengaluru (Department of Computer Science & Engineering, AY 2025–26)**  
- **Authors**: Asmit Das (235891248), Suprathik Narendra (235891216), Nadia Cecil (235891048)  
- **Project Guide**: Dr. S. Vaishnavi, Dept. of CSE, MIT Bengaluru  

An end-to-end, test-driven, leak-free machine learning pipeline for rapid, non-destructive screening of milk adulteration using low-cost physicochemical parameters ($\text{pH}$, Temperature, Taste, Odor, Fat, Turbidity, Color).

---

## Key Achievements & Empirical Highlights
- **Architecture**: 100% leak-free Scikit-Learn `Pipeline` with Stratified 80/20 train/test partition (seed=42).
- **Test-Driven Development (TDD)**: 45 automated unit, ML, and integration tests passing (`pytest -q`).
- **Best Model**: Tuned **Support Vector Machine (RBF kernel, $C=100.0, \gamma=\text{'scale'}$)** selected via 5-Fold Stratified Cross-Validation on training data.
- **Held-Out Test Set Performance (212 Unseen Samples)**:
  - **Accuracy**: `99.53%` (211 / 212 samples correct)
  - **Recall (Sensitivity)**: `100.00%` (**0 False Negatives** — zero adulterated samples missed!)
  - **Precision**: `99.38%`
  - **F1 Score**: `0.9969`
  - **ROC-AUC**: `1.0000`
- **Interpretability Hierarchy**: $\text{pH}$ and Turbidity identified as primary diagnostic parameters via permutation importance on unseen test data.

---

## Repository Structure
```
foodadult/
├── configs/
│   └── default.yaml             # All pipeline configurations & search grids
├── data/
│   ├── raw/milknew.csv          # 1,059 verified lab samples
│   ├── interim/clean.csv        # Validated interim dataset
│   └── processed/               # Stratified train.csv (847) & test.csv (212, locked)
├── docs/
│   ├── final_report_draft.md    # Comprehensive academic minor project report
│   ├── IEEE_paper_food_adulteration.md # IEEE format research paper
│   └── example_input.csv        # Demonstration samples for batch inference
├── models/
│   ├── best_model.joblib        # Fitted winning SVM pipeline
│   └── best_model_meta.json     # Full model provenance & metrics metadata
├── reports/
│   ├── RESULTS.md               # Empirical results dynamically compiled from JSON
│   ├── eda/data_card.md         # Comprehensive Data Card & summary stats
│   ├── features/final_importance.csv # Ranked diagnostic parameter hierarchy
│   ├── metrics/test_metrics.csv # Final one-shot test evaluation metrics
│   └── figures/                 # ROC curves, PR curves, CMs, & Decision Tree
├── src/foodadult/               # Modular Python package
│   ├── data/                    # load, validate, split
│   ├── features/                # IQRClipper, ColumnTransformer, selection
│   ├── models/                  # registry, train (tuning), persist
│   ├── evaluation/              # metrics, plots, interpretability, ablations, report
│   ├── predict.py               # Schema-validated batch prediction engine
│   └── cli.py                   # Master command-line interface
└── tests/                       # Complete TDD test suite (45 tests)
```

---

## Installation & Environment Setup
```powershell
# 1. Create and activate Python 3.12 virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies & editable package
pip install -r requirements.txt
pip install -e . --no-deps
```

---

## CLI Commands

### 1. Run Complete Automated Test Suite (TDD)
```powershell
pytest -v
```

### 2. Execute Pipeline Stages
```powershell
# Validate raw dataset
python -m foodadult.cli validate

# Run Exploratory Data Analysis & generate Data Card
python -m foodadult.cli eda

# Perform Stratified Train/Test split (80/20)
python -m foodadult.cli split

# Compute Feature Selection & Collinearity
python -m foodadult.cli select-features

# Train baseline models & execute hyperparameter tuning
python -m foodadult.cli train --tune

# Score held-out test set ONCE & serialize best model
python -m foodadult.cli evaluate

# Run Interpretability & Decision Tree induction
python -m foodadult.cli interpret

# Run Robustness & Ablation studies
python -m foodadult.cli ablate

# Auto-compile RESULTS.md from metric artifacts
python -m foodadult.cli report
```

### 3. Batch Prediction on New Samples
```powershell
python -m foodadult.cli predict `
  --model models/best_model.joblib `
  --input docs/example_input.csv `
  --output reports/example_predictions.csv
```
