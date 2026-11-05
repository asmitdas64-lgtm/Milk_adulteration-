# 01 — Project Brief

## Identity
| Field | Value |
|---|---|
| Title | Machine Learning for Predicting Food Adulteration |
| Type | B.Tech Minor Project, CSE, Manipal Institute of Technology Bengaluru (AY 2025–26) |
| Team | Asmit Das (235891248), Suprathik Narendra (235891216), Nadia Cecil (235891048) |
| Guide | Dr. S. Vaishnavi, Assistant Professor, CSE |
| Source document | `ML_Food_Adulteration_Project_Miniproj_report.docx` (proposal stage, May 2026) |
| Project status | Stage 1 (proposal) DONE. Execution stages 2–4 pending → this repo. |

## Problem
Lab methods (HPLC, GC-MS, ELISA) are accurate but slow, costly and need experts. A cheap, fast **preliminary screening** tool is needed.

## Solution (what we build)
A supervised **binary classifier** that predicts whether a food sample is **adulterated (1)** or **unadulterated (0)** from **low-cost physicochemical measurements** (pH, density, viscosity, refractive index, colour, moisture — final set depends on dataset). It *complements*, not replaces, lab testing.

## Core deliverables
1. Cleaned, validated dataset + data card.
2. Leak-free preprocessing + feature-selection pipeline.
3. Five trained & tuned classifiers: Random Forest, SVM (RBF), k-NN, Decision Tree, Gradient Boosting (+ Dummy baseline for sanity).
4. Comparative evaluation: Accuracy, Precision, Recall, F1, ROC-AUC (+ confusion matrices, ROC curves, CV mean ± std).
5. Ranked feature importance ("which lab tests matter most").
6. Best-model selection with justification.
7. A simple CLI `predict` command that loads the saved best model and scores new samples.
8. Final project report content (results chapter, figures, tables).

## In scope
- Binary classification only.
- One food category (see decision D-01) — or multiple only if dataset supports it and D-01 says so.
- Offline Python/scikit-learn workflow, reproducible from the CLI.

## Out of scope (per report §1.4, §5.3)
- Identifying the *type* of adulterant (multi-class).
- Production/consumer app deployment, mobile app, cloud hosting.
- Spectroscopy / hyperspectral / image data.
- Deep learning.
- (Optional stretch only if human approves: a minimal Streamlit demo.)

## Success criteria
- Every model beats the Dummy baseline on test F1 and ROC-AUC.
- Entire pipeline re-runs end-to-end with one command and reproduces the same metrics (fixed seed).
- All numbers in the final report trace back to files in `reports/`.

