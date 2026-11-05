# 07 — Coding Standards

## Stack (allowed dependencies)
| Package | Use |
|---|---|
| python 3.11 | runtime |
| numpy, pandas | data |
| scikit-learn (≥1.4) | preprocessing, models, CV, metrics |
| scipy | `loguniform`, `randint`, stats tests |
| matplotlib, seaborn | plots (save to PNG, dpi=150, no `plt.show()` in src) |
| pyyaml | config |
| joblib | model persistence |
| pytest | tests |
| jupyter | EDA notebook only |
| ruff (dev) | lint/format |
| imbalanced-learn | **only if D-07 = SMOTE** |
| streamlit | **only if D-11 approved** |

Anything else → decision log entry first.

## Code rules
- `src/` layout; import as `from foodadult.x import y`.
- Type hints on all public functions; Google-style docstrings stating inputs, outputs, raises.
- No hard-coded paths, seeds, thresholds, or grids — read from config.
- Functions small and pure where possible; I/O at the edges (CLI / thin wrappers).
- Use `logging`, not `print`, in `src/`.
- Raise explicit exceptions with helpful messages (e.g. `SchemaError`), never `except: pass`.
- Notebooks may only call functions from `src/`; no unique logic in notebooks.
- Every figure: title, axis labels, units, legend; filename prefix by phase (`eda_`, `feat_`, `eval_`).
- Output files: JSON for machine-readable, CSV for tables, MD for human summaries.

## Testing rules
- **TDD is mandatory** — follow `context/10_TDD_WORKFLOW.md` (RED → GREEN → REFACTOR; required tests per phase; Definition of Done).
- Allowed dev dependency: `pytest-cov` (coverage target ≥ 80 % of `src/`).
- Every module in `src/` has a test file. Use small deterministic fixtures in `tests/conftest.py` (fixtures are test-only synthetic data — never used for results).
- Mandatory tests: schema validation, label mapping, IQRClipper fit/transform, no-leakage, stratified split ratio, metrics on hand-computed example, model registry builds all 6 models, predict end-to-end on tiny model.
- Run `pytest -q` before marking any checklist item done.

## Git
- Commit per checklist item: `P<phase>.<n>: <summary>` (e.g. `P3.2: add IQRClipper with tests`).
- Never commit large/licensed raw data or `models/*.joblib` (use `.gitignore`).

## Windows notes
- Activate venv: `.\.venv\Scripts\Activate.ps1`. Use `pathlib.Path` everywhere; no hard-coded `\`.

