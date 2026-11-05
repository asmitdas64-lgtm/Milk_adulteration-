# 08 — Decisions Log

Status: **LOCKED** (agents must follow) · **PROPOSED** (default the agent may use, human should confirm) · **OPEN** (blocks the listed phase; agent must ask).

Format: `D-xx | Status | Decision | Reason | Source | Blocks`

## OPEN — human must answer
| ID | Status | Question | Options / recommendation | Blocks |
|---|---|---|---|---|
| D-11 | OPEN | Build a Streamlit demo? | Report §1.4 says deployment is out of scope → default **no**. | Phase 10 |

## PROPOSED — defaults (agent may proceed, flag in summary)
| ID | Status | Decision | Reason | Source |
|---|---|---|---|---|
| D-05 | PROPOSED | IQR outlier handling = **clip (winsorise)** fitted on train folds, not row deletion; ablate vs none | Adulteration often *causes* extreme values; deleting outliers could delete the positive class signal | Report §4.3.2 says "detect and mitigate" — clipping satisfies this |
| D-07 | PROPOSED | Imbalance handling: if minority ≥ 35 % → none; 20–35 % → `class_weight="balanced"`; < 20 % → ask human (SMOTE via imblearn pipeline) | Report requires balance but gives no fallback | Report §4.2 |
| D-09 | PROPOSED | Use selected feature subset only if mean CV F1 drops ≤ 0.01 vs all features | Parsimony without hurting performance | Report §4.5 |
| D-10 | PROPOSED | Model selection by mean CV F1 → tie-break CV recall → interpretability. Test set never used for selection | Avoid test-set overfitting; recall matters in food safety | Report §4.7 (no rule given) |
| D-13 | PROPOSED | Primary optimisation metric = F1 (positive class) | Balanced for uneven classes (report §4.7 F1 definition) | Report §4.7 |
| D-14 | PROPOSED | Add `DummyClassifier` baseline (not in report) | Proves models learn something | Best practice |

## LOCKED
| ID | Status | Decision | Reason | Source |
|---|---|---|---|---|
| D-01 | LOCKED | Food category: **Milk** | Real public physicochemical dataset available matching FSSAI low-cost testing priority | Human choice (2026-11-05) |
| D-02 | LOCKED | Dataset: **Milk Quality Dataset** (`milknew.csv`, 1,059 rows) | Public tabular benchmark from Kaggle/data repositories containing real lab physicochemical measurements | Human uploaded |
| D-03 | LOCKED | Synthetic data allowed: **NO** (real dataset used) | 1,059 real records available | Human choice |
| D-04 | LOCKED | Binary label mapping: **low (and medium) -> 1 (Substandard/Adulterated), high -> 0 (Pure/Premium)** (or strict low=1, high=0 configurable) | Grade is categorical ('high', 'medium', 'low'); binary ML task requires mapping to {0,1} | Methodology spec |
| D-06 | LOCKED | Features: `ph`, `temprature`, `taste`, `odor`, `fat`, `turbidity`, `colour` | Direct columns in `milknew.csv` | Dataset inspection |
| D-12 | LOCKED | Dataset size: 1,059 rows (exceeds ≥ 500 requirement) | 1,059 samples provide stable 5-fold CV splits | Dataset inspection |
| D-08 | LOCKED | **Split before preprocessing**; all transforms inside sklearn Pipeline | Report §4.3 lists split last → would leak test statistics | Correction of report §4.3 |
| D-15 | LOCKED | Binary task, positive class 1 = adulterated | Report §1.4, Table 4.1, §5.3 | Report |
| D-16 | LOCKED | Models = RF, SVM-RBF, k-NN, DT, GB (sklearn). No XGBoost/DL | Report §4.6 | Report |
| D-17 | LOCKED | Stratified 80/20 split, 5-fold stratified CV, seed 42 | Report §4.3.4 + reproducibility | Report |
| D-18 | LOCKED | Feature selection includes Pearson, RF importance **and RFE** (report Objective 4 mentions RFE; §4.5 omits it) | Satisfy both sections | Report Ch.3 Obj 4 + §4.5 |
| D-19 | LOCKED | Deployment / consumer app out of scope; CLI predict only | Report §1.4 | Report |

## Change history
| Date | ID | Change | By |
|---|---|---|---|
| 2026-11-04 | all | Initial log created from report review | AI assistant |

