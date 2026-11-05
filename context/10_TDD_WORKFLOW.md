# 10 — Test-Driven Development (TDD) Workflow — MANDATORY

> Every piece of code in `src/` is written **test-first**. Agents must follow this cycle for every checklist item that produces code.

## 1. What is TDD? (simple)
Write the **test before the code**. A test is a small program that checks your real code does what it should.

Analogy: before cooking, write down "the cake must be sweet, round, and not burnt". Then bake. Then check against the list. If it fails, fix the recipe.

## 2. Why TDD for an ML project?
| Reason | Example |
|---|---|
| Catches bugs early, when they are cheap to fix | The label mapping flips 0 and 1, so the model learns the opposite. A test catches it on day 1. |
| ML bugs are **silent** | Leaky code doesn't crash; it just gives fake 99% accuracy. Only a test detects it. |
| Safe changes later | Change the outlier method, run `pytest`, and instantly know if anything broke. |
| Tests = living documentation | Reading a test shows exactly how a function should behave. |
| Proves "it works" | The agent may not claim success without green tests (anti-hallucination). |

## 3. The cycle: 🔴 RED → 🟢 GREEN → 🔵 REFACTOR
```
 ┌──────────────────────────────────────────────┐
 │ 1. 🔴 RED      Write ONE small test.          │
 │               Run it → it MUST FAIL.         │
 │               (proves the test really checks)│
 │ 2. 🟢 GREEN    Write the MINIMUM code to pass. │
 │               Run tests → all pass.          │
 │ 3. 🔵 REFACTOR Clean the code (names, dupes). │
 │               Run tests → still all pass.    │
 │ 4. ✅ COMMIT   git commit "P3.2: ..."          │
 └───────────── repeat for next behaviour ───────┘
```

### Agent rules (non-negotiable)
1. **No production code without a failing test first.** Exceptions: config files, `__init__.py`, plotting cosmetics (but plot functions still need a "file is created" test).
2. Show evidence in the response: the RED run output (failing) **and** the GREEN run output (passing).
3. One behaviour per test; test names describe behaviour: `test_iqr_clipper_clips_values_above_upper_bound`.
4. Never delete or weaken a test to make it pass. If a test is wrong, explain why and get human approval.
5. Run the **full** suite (`pytest -q`) before ticking a checklist item. All tests must be green.
6. Tests must be fast (< 1 s each, whole suite < 60 s): use tiny fixture data, `n_estimators=5`, `cv=2` in tests.
7. Fixture data in `tests/conftest.py` is fake test data, **never** used for real results.
8. Bug found? First write a test that reproduces it (RED), then fix it (GREEN). This is a regression test.

## 4. Kinds of tests in this project
| Type | Checks | Example |
|---|---|---|
| **Unit** | One function alone | `compute_metrics` on a hand-calculated example |
| **Data tests** | Data obeys rules | Labels ∈ {0,1}, pH within 0–14, required columns exist |
| **ML-specific** | ML correctness | No leakage, stratification, reproducibility, beats dummy |
| **Integration** | Steps work together | `validate → split → train(tiny) → evaluate → predict` on fixture data |
| **CLI / smoke** | Commands run | `python -m foodadult.cli --help` exits 0 |

## 5. Required tests per phase (write these FIRST)
| Phase | Test file | Tests (minimum) |
|---|---|---|
| 0 | `test_smoke.py` | package imports; `load_config` returns dict with `random_state`; CLI `--help` exit code 0 |
| 1 | `test_load.py`, `test_validate.py` | missing column → `SchemaError`; non-numeric feature → error; label outside {0,1} → error; out-of-range pH flagged; duplicates reported; `make_binary_label` maps per D-04 on hand examples; raw file never modified |
| 2 | `test_eda.py` | `summary.json` has keys rows/cols/missing/class_balance; class balance sums to 1; figures saved |
| 3 | `test_split.py`, `test_transformers.py`, `test_preprocess.py`, `test_leakage.py` | split sizes 80/20; class ratio within ±1 %; same seed → same split; no overlap train/test; `IQRClipper` learns Q1/Q3 on fit and clips on transform; doesn't change in-range values; handles NaN after imputer; `transform` before `fit` raises; **leakage:** pipeline fitted on train has median/mean equal to train-only statistics, not full data; output has zero NaN; scaled train mean ≈ 0, std ≈ 1 |
| 4 | `test_selection.py` | correlated pair (synthetic r≈1) detected; ranking contains all features; RFECV returns ≥ 1 feature; selection uses train only |
| 5 | `test_registry.py`, `test_train.py` | registry builds all 6 models; unknown key raises; each model fits & predicts on fixture; on an easy separable fixture, every model beats dummy F1 |
| 6 | `test_train.py` | search space keys match config; tuning returns best_params ⊂ grid; results saved to JSON; reproducible with same seed |
| 7 | `test_metrics.py`, `test_evaluate.py` | metrics equal hand-computed values for a known confusion matrix; positive label = 1; ROC-AUC = 1.0 on perfect scores; metric files written; best model file + meta saved |
| 8 | `test_interpret.py` | importance output has every feature; values finite |
| 9 | `test_ablation.py` | each ablation config runs and produces a row |
| 10 | `test_predict.py` | predict on fixture CSV → output has `prediction` ∈ {0,1} and `probability_adulterated` ∈ [0,1]; missing column → clear error; save→load model gives identical predictions |
| 11 | `test_report.py` | RESULTS.md numbers equal the numbers in the metric JSON (no hand-typed values) |
| 12 | `test_end_to_end.py` | full `all` command on fixture data runs; running twice gives identical metrics |

## 6. Worked example (Phase 3, `IQRClipper`)
🔴 **RED** — write the test first:
```python
# tests/test_transformers.py
import numpy as np
from foodadult.features.transformers import IQRClipper

def test_iqr_clipper_clips_values_above_upper_bound():
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])  # Q1=2, Q3=4, IQR=2 → upper=7
    clipper = IQRClipper(factor=1.5).fit(X_train)
    assert clipper.transform(np.array([[100.0]]))[0, 0] == 7.0
```
Run `pytest tests/test_transformers.py -q` → **fails** (`ModuleNotFoundError`). ✅ Expected.

🟢 **GREEN** — minimal code:
```python
# src/foodadult/features/transformers.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [Q1 - factor*IQR, Q3 + factor*IQR] learned on fit data."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.factor * iqr, q3 + self.factor * iqr
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
```
Run → **passes**. 🔵 Refactor (add type hints, a fitted check), re-run → passes. ✅ Commit.

## 7. Commands
```powershell
pytest -q                                   # all tests
pytest tests/test_transformers.py -q        # one file
pytest -k leakage -q                        # tests whose name contains "leakage"
pytest --cov=foodadult --cov-report=term    # coverage (target ≥ 80 % of src/)
```
(`pytest-cov` is an allowed dev dependency.)

## 8. Definition of Done (for every checklist item with code)
- [ ] Test(s) written first and seen failing (RED output shown)
- [ ] Code makes them pass (GREEN output shown)
- [ ] Refactored; full `pytest -q` green
- [ ] `ruff check .` clean
- [ ] Explained in simple words what & why (Teaching Mode)
- [ ] Checklist ticked, PROGRESS.md updated, committed

