# 🏀 NCAA Tournament Seed Prediction — XGBoost Regression

Predicts NCAA Men's Basketball Tournament seeds (1–68) for all 68 tournament teams using team performance metrics, NET rankings, strength of schedule, and quadrant win-loss records.

Built for the **Purdue FFAC Data Competition**.

---

## Problem

Each March, the NCAA selection committee assigns seeds (1–68) to 68 teams for the tournament. Predicting these seeds from team performance data is a regression problem with real stakes — accurate seeding prediction has implications for bracket forecasting, sports analytics, and team evaluation.

**Key challenge:** The dataset contains all Division I teams, but only 68 make the tournament. Non-tournament teams have no seed (0), while tournament teams have seeds 1–68. Naively predicting seeds for all 451 teams inflates RMSE significantly.

---

## Solution

An XGBoost regression pipeline that:
1. Recovers corrupted win-loss strings caused by Excel date auto-formatting (e.g. `8-May` → `8-5`, `Jun-00` → `0-6`)
2. Engineers win percentage features from raw win-loss records across 8 categories
3. One-hot encodes Conference and Bid Type
4. Trains on tournament teams only, predicts 0 for non-tournament teams
5. Tunes hyperparameters via GridSearchCV with KFold cross-validation

---

## Results

| Metric | Baseline XGBoost | Tuned XGBoost |
|---|---|---|
| MAE | — | ~2.1 |
| RMSE | — | ~3.4 |
| R² | — | ~0.88 |

Predictions clipped to valid seed range [1, 68].

---

## Key Engineering Decisions

**Why XGBRegressor instead of XGBClassifier?**
Seeds 1–68 are ordinal/continuous — not independent categories. XGBClassifier requires contiguous 0-indexed classes, which breaks during cross-validation when rare seed values are missing from a fold.

**Why KFold instead of StratifiedKFold?**
StratifiedKFold stratifies by class, which fails for regression targets. KFold preserves the correct train/test split structure.

**Why predict 0 for non-tournament teams?**
Only 91 of 451 test rows are tournament teams. Predicting non-zero seeds for the other 360 teams causes massive RMSE inflation. Filtering by `Bid Type` cleanly separates the two groups.

**Why use training mean for imputation (not test mean)?**
Using test set statistics for imputation leaks future information into the model. The training mean for `NETNonConfSOS` is computed on the training set and applied to both.

---

## Features Used

| Feature | Description |
|---|---|
| NET Rank / PrevNET | NCAA Evaluation Tool ranking (current + prior) |
| AvgOppNETRank | Average opponent NET rank |
| NETSOS | NET Strength of Schedule (1 = hardest) |
| NETNonConfSOS | Non-conference strength of schedule |
| WL_WinPct | Overall win percentage |
| Conf.Record_WinPct | Conference win percentage |
| Non-ConferenceRecord_WinPct | Non-conference win percentage |
| RoadWL_WinPct | Road game win percentage |
| Quadrant1–4_WinPct | Win % vs. top / mid / lower-tier opponents |
| Conference (OHE) | One-hot encoded conference membership |
| Bid Type (OHE) | Automatic Qualifier vs. At-Large bid |

---

## Data Dictionary (Key Columns)

| Column | Description |
|---|---|
| `Overall Seed` | Target — tournament seed 1–68 |
| `NET Rank` | Primary NCAA metric for tournament selection |
| `Bid Type` | AQ = automatic qualifier (conference winner); AL = at-large selection |
| `Quadrant1` | Win-Loss vs. home top-30 / neutral top-50 / away top-75 opponents |
| `NETSOS` | NET Strength of Schedule (1 = toughest, 364 = easiest) |

---

## Quick Start

```bash
pip install pandas numpy scikit-learn xgboost

# Place data files in the same directory as the notebook
# NCAA_Seed_Training_Set2_0.csv
# NCAA_Seed_Test_Set2_0.csv
# submission_template2_0.csv

jupyter lab NCAA_Final_4_competition_fixed.ipynb
```

---

## Project Structure

```
├── NCAA_Final_4_competition_fixed.ipynb   # Full pipeline notebook
├── NCAA_Seed_Training_Set2_0.csv          # Training data
├── NCAA_Seed_Test_Set2_0.csv              # Test data
├── submission_template2_0.csv             # Submission format
└── FFAC_Data_Dictionary.xlsx              # Feature descriptions
```

---

## Notebook Structure

| Step | Description |
|---|---|
| Load & inspect | Load training data, check duplicates |
| WL recovery | Fix Excel date-corrupted win-loss strings |
| Feature engineering | Parse WL strings → win %, losses, wins per category |
| Imputation | Fill NETNonConfSOS with training mean |
| Encoding | One-hot encode Conference and Bid Type |
| Train/test split | 80/20 split, seed=42 |
| Baseline XGBoost | Train XGBRegressor, evaluate MAE/RMSE/R² |
| Hyperparameter tuning | GridSearchCV over n_estimators, max_depth, learning_rate, subsample |
| Test prediction | Predict tournament teams only, assign 0 to non-tournament |
| Submission | Map predictions to RecordID, export submission.csv |

---

## Tech Stack

- **Model:** XGBoost (XGBRegressor)
- **Tuning:** GridSearchCV + KFold
- **Data:** pandas, numpy
- **Evaluation:** scikit-learn (MAE, RMSE, R²)
