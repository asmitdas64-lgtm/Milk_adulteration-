# 03 — Data Specification

> ⚠️ The report does **not** name a concrete dataset. Until decision **D-02** is LOCKED, the schema below is the *target/proposed* schema from report Table 4.1. Agents must NOT assume any column exists until the real file is inspected in Phase 1.

## 1. Proposed feature set (report Table 4.1, corrected)
| Column (snake_case) | Description | Unit | Type | Notes |
|---|---|---|---|---|
| `ph` | Acidity / basicity | – | float | Plausible range 0–14 |
| `density` | Mass per unit volume | g/mL | float | Report says "specific density" — clarify vs specific gravity (dimensionless) |
| `viscosity` | Resistance to flow | cP (dynamic) | float | Report text says *kinematic* (cSt) but table says cP (dynamic) → resolve in D-06 |
| `refractive_index` | Optical density | – | float | Typically ~1.33–1.55 for foods |
| `colour_index` | Standardised colour value | dataset-defined | float | Report says "spectrophotometric" — conflicts with low-cost goal; accept whatever the dataset provides |
| `moisture_content` | Water content by mass | % | float | 0–100 |
| `label` | **Target** — 1 = adulterated, 0 = unadulterated | – | int {0,1} | Target, NOT a feature |

Optional/extra columns (allowed only if present in chosen dataset; record in data card): e.g. electrical conductivity, fat %, temperature, `food_type`.

## 2. Dataset requirements (report §4.2)
- Source: public repository (UCI / Kaggle) preferred; institutional lab data acceptable; synthetic only per D-03.
- Must contain both classes; target minority class ≥ 20 % preferred. If imbalanced, follow D-07.
- Must have a licence permitting academic use — record it in the data card.
- Minimum size: TODO(HUMAN) — recommended ≥ 500 rows for stable 5-fold CV.

## 3. Candidate datasets — ALL UNVERIFIED
Agents must not download or describe these as facts. The human verifies existence, licence and columns, then locks D-02.
| Candidate | Reported content (unverified) | Concern |
|---|---|---|
| Kaggle "honey purity" style dataset | Reportedly density, water content, pH, viscosity, colour score, purity | Purity may be continuous → needs threshold (D-04); may be synthetic |
| Kaggle "milk quality" style dataset | Reportedly pH, temperature, fat, turbidity, colour, grade | Label is *quality grade*, not adulteration → only usable with explicit relabelling decision |
| Institutional lab measurements | Real measurements by team | Small size; needs controlled adulteration protocol |

## 4. Label rules
- Positive class `1` = adulterated. Never invert.
- If the source label is continuous (e.g. purity %) or multi-class, the mapping to {0,1} must be LOCKED in D-04 before Phase 2. Implement it in `src/foodadult/data/load.py::make_binary_label` and unit-test it.

## 5. Validation rules (implemented in `data/validate.py`, Phase 1)
- Required columns present (from `configs/default.yaml: data.features` + `data.target`).
- Numeric dtypes for features; target ∈ {0,1}.
- Physical ranges (config-driven): `ph` 0–14, `moisture_content` 0–100, `density` > 0, `viscosity` > 0, `refractive_index` 1.0–2.0.
- Report (don't silently drop) duplicates and out-of-range values.
- Missing-value % per column logged.

## 6. Data directories
```
data/raw/        # original files, read-only, never edited (gitignored if large/licensed)
data/interim/    # renamed/typed/labelled data, pre-split
data/processed/  # split.json (indices), train.csv, test.csv
```

## 7. Data card (to fill in Phase 1 — `reports/eda/data_card.md`)
Name · source URL · licence · download date · #rows · #cols · column dictionary · label definition · class balance · missingness · known biases · synthetic? (Y/N).

