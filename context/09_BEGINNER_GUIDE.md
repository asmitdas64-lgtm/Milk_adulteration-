# 09 — Beginner's Guide: Every Step Explained Simply

> Written for someone doing their **first ML project**. Every step answers: **WHAT** is it, **WHY** do we do it, and **HOW** do we do it (in plain words).
> Agents: use the same simple language when explaining work to the team (see `SYSTEM_PROMPT.md` §7 Teaching Mode).

---

## Part 0 — The Big Picture

### What is Machine Learning (ML)?
Normal programming: **you** write the rules.
> "If pH < 4 and density > 1.5 → adulterated."

Machine learning: you give the computer **lots of examples with answers**, and it **finds the rules by itself**.
> "Here are 1000 honey samples. These 400 are adulterated, these 600 are pure. Figure out the pattern."

### What exactly are we building?
A program that looks at a few numbers measured from a food sample and answers one question:
**"Is this sample adulterated? Yes (1) or No (0)?"**

This is called **binary classification**: *binary* = 2 possible answers, *classification* = putting things into groups.

### Key words you'll see everywhere
| Word | Simple meaning | Food example |
|---|---|---|
| **Sample / row** | One thing we measured | One jar of honey |
| **Feature** | One measurement (an input) | pH, density, viscosity |
| **Label / target** | The correct answer | 1 = adulterated, 0 = pure |
| **Dataset** | A table of many samples | A CSV file with 1000 rows |
| **Model** | The "brain" that learns the pattern | Random Forest |
| **Training** | Showing the model examples so it learns | Model studies 800 jars |
| **Testing** | Checking the model on examples it has never seen | Model's "final exam" on 200 new jars |
| **Prediction** | The model's guess for a new sample | "This jar: adulterated, 87% sure" |

### The analogy we'll use throughout: a student preparing for an exam
- **Dataset** = the textbook with solved questions
- **Training set** = the questions the student practises on
- **Test set** = the final exam questions, kept secret until exam day
- **Model** = the student
- **Cross-validation** = mock tests during preparation
- **Overfitting** = memorising answers instead of understanding, so they fail on new questions
- **Data leakage** = the student secretly seeing the exam paper beforehand, so their score is fake

---

## Phase 0 — Setup

### What
Create the project folders, install Python libraries, and make a settings file.

### Why
- **Folders:** so data, code, results and models don't get mixed up. Anyone (including you in 3 months) can find things.
- **Virtual environment (venv):** a private box of Python libraries just for this project. Without it, installing something for another project could break this one.
- **`requirements.txt` with exact versions:** library updates can change results. Pinning versions means your teammate gets **the same results** on their laptop.
- **Config file (`configs/default.yaml`):** all settings (file paths, random seed, model settings) live in one place. To change something, you edit one file instead of hunting through code.
- **Git:** a time machine for your code. If something breaks, you can go back.

### How (simple)
```powershell
python -m venv .venv                 # create the private box
.\.venv\Scripts\Activate.ps1         # step into the box
pip install -r requirements.txt      # install the exact libraries
pytest -q                            # run tests (checks everything works)
```

### Libraries we use and why
| Library | What it does |
|---|---|
| **pandas** | Reads tables (CSV) like Excel in Python |
| **numpy** | Fast maths on numbers |
| **scikit-learn** | All ML: models, cleaning, testing, scores |
| **matplotlib / seaborn** | Draws charts |
| **pyyaml** | Reads the config file |
| **joblib** | Saves the trained model to a file and loads it later |
| **pytest** | Runs our automatic tests (see TDD, `10_TDD_WORKFLOW.md`) |

---

## Phase 1 — Getting the Data

### What
Find a real dataset, put it in `data/raw/`, and check it is correct.

### Why
**"Garbage in, garbage out."** The model can only be as good as the data. If labels are wrong or columns are mixed up, everything after this is useless, no matter how fancy the model.

### Why only ONE food type?
pH of milk ≈ 6.7, honey ≈ 3.9, lemon juice ≈ 2. If you mix foods, the model might just learn "low pH = lemon", not "this is adulterated". One food = a fair comparison.

### Why we never edit the raw file
If you accidentally delete or change something, you can always start again from the original. Cleaned versions are saved separately in `data/interim/`.

### What "validation" checks
| Check | Why |
|---|---|
| Are all expected columns there? | Code would crash or use the wrong column |
| Are numbers actually numbers? | "6.5" stored as text can't be used in maths |
| Are values physically possible? | pH = 25 is impossible, so it's a typo or a broken sensor |
| Labels only 0 or 1? | Anything else means the answer column is broken |
| Duplicate rows? | The same sample twice can make the model look better than it is |

### Why the label rule matters
Some datasets give "purity = 73%" instead of yes/no. Someone must decide the cutoff (e.g. below 80% = adulterated). This **must come from a standard or your guide**, not be guessed, because a different cutoff = a different project.

---

## Phase 2 — Exploratory Data Analysis (EDA)

### What
Look at the data with charts and simple statistics **before** building any model.

### Why
Like a doctor looking at a patient before prescribing medicine. You discover problems early:
- **Missing values:** some samples have no viscosity measured.
- **Class imbalance:** e.g. 950 pure and only 50 adulterated. A lazy model that always says "pure" would be 95% "accurate" but completely useless!
- **Outliers:** weird extreme values.
- **Which features look different between pure and adulterated:** a first hint of what will matter.

### Charts we make and what they tell us
| Chart | Tells you |
|---|---|
| **Bar chart of classes** | Is the data balanced? |
| **Histogram per feature, coloured by class** | Does e.g. density look different for adulterated samples? |
| **Box plot** | Middle value, spread, and outliers at a glance |
| **Correlation heatmap** | Which features move together (e.g. density and refractive index might be ~the same information) |

### Output
A **data card**: a one-page "ID card" of the dataset (source, size, columns, balance, problems).

---

## Phase 3 — Splitting & Cleaning the Data

### Step 3.1 — Train/Test Split (80/20)

**What:** Randomly put 80% of samples in the **training set** and 20% in the **test set**. Lock the test set away.

**Why:** If you test the student on questions they practised, a high score proves nothing. The test set shows how the model performs on **new, unseen samples**, which is what happens in real life.

**Stratified:** keeps the same pure/adulterated ratio in both parts. If 30% of all data is adulterated, then 30% of train AND 30% of test are adulterated. Otherwise, by bad luck, the test set might have almost no adulterated samples.

**Random seed = 42:** the "random" split is the same every time you run it, so results are reproducible. (42 is just a convention; any fixed number works.)

### ⚠️ Why we split BEFORE cleaning (the most important rule)
Cleaning uses statistics like the "middle value" (median). If you calculate it from ALL data (including test), information from the exam paper sneaks into preparation. This is **data leakage** and makes scores look better than reality.
**Rule:** learn cleaning settings from training data only, then apply them to test data.
*(The original report had this in the wrong order; we fixed it — decision D-08.)*

### Step 3.2 — Filling Missing Values (Median Imputation)
**What:** If a sample's viscosity is missing, fill it with the **median** (middle value) of viscosity from the training data.
**Why:** Most models crash on empty cells. Throwing away rows wastes data.
**Why median and not mean (average)?** Values: 2, 3, 3, 4, 100. Mean = 22.4 (pulled up by the 100). Median = 3 (not affected). The median is safer with outliers.

### Step 3.3 — Handling Outliers (IQR Clipping)
**What:** Find a "normal range" for each feature and pull extreme values back to its edge.
- Sort the values. **Q1** = value at 25%, **Q3** = value at 75%.
- **IQR** = Q3 − Q1 (the spread of the middle half).
- Normal range = from Q1 − 1.5×IQR to Q3 + 1.5×IQR.
- Anything outside gets **clipped** to the edge (not deleted).

**Why clip instead of delete?** In our problem, **adulterated samples are often the "weird" ones!** Deleting outliers could delete the exact cases we want to catch. Clipping reduces their extreme pull while keeping the sample. *(Decision D-05. We'll also test "no outlier handling" to see which works better.)*

### Step 3.4 — Scaling (StandardScaler)
**What:** Convert every feature so its average = 0 and spread = 1.
**Why:** pH ranges 0–14, viscosity might range 0–10,000. Models that measure **distance** (k-NN, SVM) would think viscosity matters 1000× more just because its numbers are bigger. Scaling makes every feature play fair.
Formula (just for understanding): `scaled = (value − mean) / standard_deviation`.

### Step 3.5 — Pipeline
**What:** Chain the steps into one object: `fill missing → clip outliers → scale → model`.
**Why:**
1. Guarantees cleaning is always learned on training data only (no leakage).
2. When predicting new samples later, they get **exactly** the same cleaning automatically.
3. One object to save and load.

---

## Phase 4 — Feature Selection

### What
Find out which measurements actually help predict adulteration, and possibly drop useless ones.

### Why
- **Fewer tests = cheaper in real life** (the whole point of this project is low cost!).
- Useless features add noise and can confuse the model.
- Simpler models are easier to explain.

### Methods (all on training data only)
| Method | Simple idea |
|---|---|
| **Pearson correlation** | If two features carry almost the same information (correlation ≥ 0.90), keep only one. Like having two thermometers in the same room. |
| **Random Forest importance** | The forest tells us how much each feature helped it make decisions. |
| **Permutation importance** | Shuffle one feature randomly. If the score drops a lot, that feature was important. If nothing changes, it was useless. |
| **RFE (Recursive Feature Elimination)** | Train → remove the weakest feature → retrain → repeat. Find the smallest set that still works well. |

**Rule:** use the smaller feature set only if the score drops by less than 0.01 (D-09).

---

## Phase 5 — Baseline Models

### What
Train all 5 models with their **default settings**, plus a **"dummy" model** that always guesses the most common answer.

### Why
- The **dummy** is the "lazy student". If 70% of samples are pure, it says "pure" every time and gets 70% accuracy. **Any real model must beat it**, or it has learned nothing.
- Defaults give a starting point so we can see whether tuning (Phase 6) actually helps.

### The 5 models in simple words
| Model | How it thinks (analogy) | Strength | Weakness |
|---|---|---|---|
| **Decision Tree** | A flowchart of yes/no questions: "Is density > 1.4? → Is pH < 4? → Adulterated" | Very easy to understand and draw | Memorises easily (overfits) |
| **Random Forest** | Hundreds of different decision trees **vote**; the majority wins | Accurate, robust, gives feature importance | Harder to explain than one tree |
| **Gradient Boosting** | Trees built one after another; each new tree **fixes the mistakes** of the previous ones | Often the most accurate | Slower; needs careful tuning |
| **k-NN (k-Nearest Neighbours)** | "Tell me who your neighbours are." Find the k most similar samples and copy the majority answer | Very simple idea | Needs scaling; slow on big data |
| **SVM (RBF kernel)** | Draws the best possible "boundary" between pure and adulterated, with as much gap as possible. RBF lets the boundary be curved | Strong on small/medium data | Needs scaling; less interpretable |

### What is Cross-Validation (5-fold CV)? = mock tests
Split the **training** data into 5 parts. Train on 4 parts, test on the 5th. Repeat 5 times so each part is the "mock test" once. Average the 5 scores.
**Why:** one mock test could be lucky or unlucky; 5 gives a reliable average **and** shows how stable the model is (the ± value). The real test set stays locked the whole time.

---

## Phase 6 — Hyperparameter Tuning

### What are hyperparameters?
Settings **you choose before training**; the model doesn't learn them. Like knobs on a radio.
| Model | Knob examples | Meaning |
|---|---|---|
| Random Forest | `n_estimators` | How many trees vote |
| | `max_depth` | How many questions each tree may ask |
| SVM | `C` | How strict about mistakes on training data |
| | `gamma` | How curvy the boundary can be |
| k-NN | `k` | How many neighbours to ask |
| Decision Tree | `max_depth`, `min_samples_split` | Tree size limits (prevent memorising) |
| Gradient Boosting | `learning_rate`, `n_estimators` | Step size of each correction, number of trees |

### How we tune
- **Grid Search:** try **every** combination from a list (e.g. 3 × 4 × 3 = 36 combinations), each checked with 5-fold CV.
- **Randomized Search:** try 30 **random** combinations (used for Gradient Boosting because it has too many possibilities).

### Why
Too simple (e.g. tree depth 1) = **underfitting** (the student didn't study enough). Too complex = **overfitting** (memorised the textbook). Tuning finds the sweet spot.

### Picking the winner — BEFORE the final exam
We choose the best model using the **mock-test (CV) scores**, not the test set. If we peeked at the test set to choose, the test would no longer be a fair exam (D-10).

---

## Phase 7 — Final Evaluation (the exam, done ONCE)

### What
Retrain each tuned model on all the training data, then predict the locked test set **one time**.

### Why only once?
If you keep re-tuning after seeing test scores, you are slowly "studying the exam paper", and the score becomes fake.

### The confusion matrix (the base of all scores)
|  | Model says Pure | Model says Adulterated |
|---|---|---|
| **Actually Pure** | ✅ True Negative (TN) | ❌ False Positive (FP): false alarm |
| **Actually Adulterated** | ❌ False Negative (FN): **missed danger!** | ✅ True Positive (TP) |

### The scores, in simple words
| Metric | Question it answers | Formula |
|---|---|---|
| **Accuracy** | Out of all samples, how many did we get right? | (TP+TN) / all |
| **Precision** | When the model says "adulterated", how often is it right? (few false alarms) | TP / (TP+FP) |
| **Recall** | Out of all truly adulterated samples, how many did we catch? (few missed dangers) | TP / (TP+FN) |
| **F1-score** | One number balancing precision and recall | 2·P·R / (P+R) |
| **ROC-AUC** | How well does the model separate the two classes overall? 0.5 = coin toss, 1.0 = perfect | Area under the ROC curve |

### Why recall matters most in food safety
- False alarm (FP) → a pure sample gets an extra lab test. **Annoying, but safe.**
- Missed danger (FN) → adulterated food reaches people. **Harmful.**
So among equally good models, we prefer the one with higher recall.

### Why accuracy alone is misleading
With 95 pure and 5 adulterated samples, the lazy "always pure" model gets 95% accuracy but catches **zero** adulterated samples (recall = 0). That's why we use F1 as the main score.

---

## Phase 8 — Interpretability: "Which lab tests matter most?"

### What
Measure how much the final model relies on each feature (permutation importance on the test set), and draw the decision tree.

### Why
- This is a promised outcome of the report (Outcome 5).
- Practical value: a small lab could do **only the 2–3 most useful tests** and save money.
- It builds trust: if the model relies on something that makes scientific sense (e.g. density for diluted honey), we believe it more.

---

## Phase 9 — Robustness / Ablation Studies

### What
Change **one design choice at a time** and see if the score changes. For example:
- Outlier clipping ON vs OFF
- All features vs selected features
- Class weighting ON vs OFF

### Why
When your guide asks "Why did you clip outliers?", you can answer **with evidence**, not opinion. "Ablation" just means removing one part to see how much it mattered.

---

## Phase 10 — Prediction Tool

### What
A command that takes a CSV of new samples and outputs a prediction and a probability for each.
```powershell
python -m foodadult.cli predict --model models/best_model.joblib --input new_samples.csv --output predictions.csv
```

### Why
A model sitting in a notebook is useless to anyone else. This is the "Final Prediction Generation" stage from the report. It also checks the input columns are correct before predicting.

### What "probability" means
"0.87" = the model is 87% confident the sample is adulterated. By default, ≥ 0.5 → predicted adulterated.

---

## Phase 11 — Results & Report

### What
The code automatically generates `reports/RESULTS.md` with all tables and charts. You then write the Results/Discussion chapter.

### Why auto-generate?
Typing numbers by hand causes mistakes (e.g. writing 0.93 when it was 0.89). Auto-generation guarantees **report = actual results**.

---

## Phase 12 — Final Check

### What
Delete the environment, reinstall from scratch, run everything with one command, and confirm the numbers are identical.

### Why
Proves your project is **reproducible**, a basic requirement of science. If it only works on your laptop, it doesn't really work.

---

## Common Beginner Mistakes (and how this project prevents them)
| Mistake | What goes wrong | Our protection |
|---|---|---|
| Cleaning before splitting | Fake high scores (leakage) | Split first + Pipeline + leakage test |
| Using the test set many times | Fake high scores | Test set used once, in Phase 7 |
| Trusting accuracy only | Lazy model looks good | F1, recall, dummy baseline |
| Not fixing the random seed | Different results every run | seed = 42 in config |
| Logic hidden in notebooks | Can't reproduce or test | All logic in `src/`, tested |
| Deleting outliers blindly | Deletes adulterated samples | Clip + ablation |
| Making up numbers in the report | Academic dishonesty | Auto-generated RESULTS.md |
| Code "works" but nobody checked | Hidden bugs | TDD: tests written first (`10_TDD_WORKFLOW.md`) |

## Glossary (quick lookup)
**Algorithm:** the learning method (e.g. Random Forest). **Baseline:** simplest reference model. **Class:** a possible answer (0 or 1). **CV:** cross-validation, mock tests on training data. **Epoch:** not used here (deep-learning term). **Fit:** train / learn from data. **Generalisation:** doing well on new data. **Hyperparameter:** a setting you choose. **Imbalance:** one class much rarer. **Inference:** making predictions. **Leakage:** test info sneaking into training. **Overfitting:** memorising, failing on new data. **Pipeline:** chained steps. **Seed:** number making randomness repeatable. **Transform:** apply learned cleaning to data. **Underfitting:** too simple to learn the pattern.

