# Machine Learning for Predicting Food Adulteration

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-46%20Passed-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Pipeline-Leak--Free-success.svg)](src/foodadult/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end, test-driven, leak-free machine learning framework for rapid, non-destructive screening of milk adulteration using routine physicochemical parameters ($\text{pH}$, Temperature, Taste, Odor, Fat, Turbidity, Color).

---

## Authors & Academic Credits
- **Project**: B.Tech Minor Project (Academic Year 2025–2026)
- **Institution**: Department of Computer Science & Engineering, Manipal Institute of Technology (MIT) Bengaluru, Manipal Academy of Higher Education (MAHE), Karnataka, India.
- **Student Researchers**:
  - **Asmit Das** (Registration No: `235891248`)
  - **Suprathik Narendra** (Registration No: `235891216`)
  - **Nadia Cecil** (Registration No: `235891048`)
- **Faculty Guide**: **Dr. S. Vaishnavi**, Assistant Professor, Department of Computer Science & Engineering, MIT Bengaluru.

### Dataset Provenance & Acknowledgement
The dataset used in this project is the **Milk Quality Benchmark Dataset** (`milknew.csv`), comprising 1,059 physical laboratory records.  
We gratefully acknowledge researcher **Subham Agarwal** for curating and making this dataset publicly available on Kaggle ([cpluzshrijayan/milknew](https://www.kaggle.com/datasets/cpluzshrijayan/milknew)) for food quality and machine learning research.

---

## Key Empirical Achievements
- **100% Leak-Free Pipeline**: Stratified 80/20 train/test partition executed prior to any scaling or transformation. All feature scaling, imputation, and IQR winsorization are strictly encapsulated inside Scikit-Learn `Pipeline` architectures.
- **Winning Classifier**: Support Vector Machine (RBF kernel, $C=100.0, \gamma=\text{'scale'}$), selected via 5-Fold Stratified Cross-Validation on the training set prior to test evaluation (Decision D-10).
- **Held-Out Test Set Performance (212 Unseen Samples)**:
  - **Accuracy**: `99.53%` (211 / 212 samples correct)
  - **Recall (Sensitivity)**: `100.00%` (**0 False Negatives** — zero adulterated samples missed!)
  - **Precision**: `99.38%`
  - **F1 Score**: `0.9969`
  - **ROC-AUC**: `1.0000`
- **Diagnostic Hierarchy**: Identified $\text{pH}$ ($\Delta F_1 = 0.2088$) and Turbidity ($\Delta F_1 = 0.0839$) as the primary diagnostic indicators, establishing that low-cost digital sensor screening eliminates missed adulteration incidents before costly secondary lab assays.

---

## Final Performance Comparison (Held-Out Test Cohort)

| Model | Accuracy | Precision | Recall (Sensitivity) | F1 Score | Specificity | ROC-AUC | True Positives | False Negatives |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SVM (Best)** | **99.53%** | **99.38%** | **100.00%** | **0.9969** | **98.04%** | **1.0000** | **161** | **0** |
| **Random Forest** | 99.53% | 99.38% | 100.00% | 0.9969 | 98.04% | 0.9999 | 161 | 0 |
| **Gradient Boosting** | 99.53% | 99.38% | 100.00% | 0.9969 | 98.04% | 1.0000 | 161 | 0 |
| **$k$-NN** | 99.53% | 99.38% | 100.00% | 0.9969 | 98.04% | 0.9902 | 161 | 0 |
| **Decision Tree** | 99.53% | 99.38% | 100.00% | 0.9969 | 98.04% | 0.9902 | 161 | 0 |

---

## Repository Structure
```
foodadult/
├── configs/
│   └── default.yaml             # Tunables: paths, features, seed, grids
├── data/
│   ├── raw/milknew.csv          # Verified raw dataset (Subham Agarwal / Kaggle)
│   ├── interim/clean.csv        # Validated interim dataset (1,059 samples)
│   └── processed/               # Stratified train.csv (847) & test.csv (212, locked)
├── docs/
│   ├── final_report_draft.md    # Comprehensive 13-chapter B.Tech Minor Project Report
│   ├── IEEE_paper_food_adulteration.md # Publication-ready IEEE conference format paper
│   └── example_input.csv        # Demonstration samples for batch inference
├── models/
│   ├── best_model.joblib        # Fitted winning SVM pipeline
│   └── best_model_meta.json     # Model metadata, parameters & provenance
├── reports/
│   ├── RESULTS.md               # Empirical results dynamically compiled from JSON
│   ├── eda/data_card.md         # Comprehensive Data Card & summary statistics
│   ├── features/final_importance.csv # Ranked diagnostic parameter hierarchy
│   ├── metrics/test_metrics.csv # Test evaluation metrics
│   └── figures/                 # ROC curves, PR curves, CMs, & Decision Tree
├── src/foodadult/               # Modular Python package
│   ├── data/                    # load, validate, split
│   ├── features/                # IQRClipper, ColumnTransformer, selection
│   ├── models/                  # registry, train (tuning), persist
│   ├── evaluation/              # metrics, plots, interpretability, ablations, report
│   ├── predict.py               # Schema-validated batch prediction engine
│   └── cli.py                   # Master command-line interface
└── tests/                       # Complete TDD test suite (46 tests)
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.12 (or 3.11+)
- Git

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/asmitdas64-lgtm/Milk_adulteration-.git
cd Milk_adulteration-

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies & local package
pip install -r requirements.txt
pip install -e . --no-deps
```

---

## Usage Guide (CLI Commands)

### Run the Full Pipeline End-to-End (Single Command)
```powershell
python -m foodadult.cli all
```

### Run Automated TDD Test Suite (46 Tests)
```powershell
pytest -v
```

### Run Individual Pipeline Stages
```powershell
# Step 1: Validate Raw Dataset
python -m foodadult.cli validate

# Step 2: Exploratory Data Analysis & Visualizations
python -m foodadult.cli eda

# Step 3: Stratified 80/20 Train/Test Partitioning
python -m foodadult.cli split

# Step 4: Feature Selection & Collinearity Analysis
python -m foodadult.cli select-features

# Step 5: Train Baseline Models & Hyperparameter Tuning
python -m foodadult.cli train --tune

# Step 6: Final One-Shot Evaluation on Held-Out Test Set
python -m foodadult.cli evaluate

# Step 7: Interpretability & White-Box Decision Tree Induction
python -m foodadult.cli interpret

# Step 8: Robustness & Ablation Studies (Outliers, Subsets, McNemar)
python -m foodadult.cli ablate

# Step 9: Recompile RESULTS.md from JSON metrics
python -m foodadult.cli report

# Step 10: Batch Prediction on New Samples
python -m foodadult.cli predict `
  --model models/best_model.joblib `
  --input docs/example_input.csv `
  --output reports/example_predictions.csv
```

---

## Primary Publications & Deliverables
- **Comprehensive Project Report**: [`docs/final_report_draft.md`](docs/final_report_draft.md)
  - 13 Exhaustive chapters detailing background, literature gap, data card, architecture, empirical metrics, difficulties faced in each phase, and answers to 15 critical viva/defense questions.
- **IEEE Research Paper**: [`docs/IEEE_paper_food_adulteration.md`](docs/IEEE_paper_food_adulteration.md)
  - Full IEEE conference paper format including mathematical equations, architecture diagrams, benchmark tables, ablation analysis, and peer-reviewed citations.

---

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
