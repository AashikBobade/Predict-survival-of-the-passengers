# Predict-survival-of-the-passengers

A classification challenge from the IIT Madras BS in Data Science & Applications program.

This repository contains the full workflow for solving a noisy Titanic-style binary classification problem with dirty data, obfuscated columns, missing values, and feature engineering experiments.

---

## Problem Statement

The goal is to predict whether a passenger **survived (1)** or **perished (0)** using historical passenger records from a maritime incident.

The evaluation metric is **Accuracy** on a hidden test set.

### Files

* `maritime_train.csv` — training data with target column `Outcome`
* `maritime_test.csv` — test data without target
* `maritime_sample_submission.csv` — submission format

### Submission Format

The submission file must contain:

* `PassengerName`
* `Outcome`

Example:

```csv
PassengerName,Outcome
"Braund, Mr. Owen Harris",0
"Heikkinen, Miss. Laina",1
```

---

## Dataset Overview

The dataset is intentionally noisy and includes:

* Gaussian noise in numeric variables
* missing values
* obfuscated / renamed columns
* sparse cabin information
* multiple categorical variables with hidden structure

### Important columns

* `PassengerName`: unique identifier and a source of family/surname signal
* `TicketTier`: proxy for social class
* `Gender`: categorical sex feature
* `Age`: noisy continuous feature with missing values
* `RelativesAboard`, `ParentsChildren`: family structure features
* `TicketCost`: fare paid, noisy but informative
* `BoardingPort`: embarkation port (`C`, `Q`, `S`)
* `Class`: raw ticket identifier / ticket string
* `Berth`: cabin / section string, very sparse
* `Singleton`: indicates whether passenger was travelling alone
* `FarePerPerson`: derived fare feature
* `Title`: extracted honorific (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`)
* `Outcome`: target label

---

## Project Goals

This project explored:

1. Data cleaning and preprocessing
2. Encoding categorical variables
3. Missing value handling
4. Feature engineering
5. Model comparison
6. Cross-validation vs leaderboard behavior
7. Simple blending / ensemble ideas

---

## Key Observations From EDA

The following patterns were the most useful:

* `Gender` was the strongest single predictor.
* `Title` carried very strong survival signal.
* `TicketTier` and `TicketCost` were both informative.
* `Singleton` and family-related features mattered.
* `Berth` was extremely sparse, so raw cabin strings were not useful directly.
* `FarePerPerson` was useful early on, but later appeared to be noisy / redundant in the final feature set.
* `PassengerId` was not useful and behaved like noise.

The correlation map and coefficient inspection showed that many derived features were highly collinear, so feature pruning mattered a lot.

---

## Preprocessing Decisions

### 1) `BoardingPort`

* Missing values were very rare.
* Final encoding used **one-hot encoding**.
* Missing values were imputed using the mode when needed.

### 2) `Gender`

* Encoded as binary numeric values:

  * `male -> 0`
  * `female -> 1`

### 3) `Title`

* One-hot encoded.
* `Rare` was kept as a grouped category rather than trying to relabel it.

### 4) `Berth`

Because `Berth` had very high missingness, the raw cabin string was not imputed directly.

Instead, we engineered:

* `BerthKnown`: cabin available or not
* `BerthDeck`: first deck letter from cabin string
* `BerthCabinCount`: number of cabin tokens listed
* `BerthMultiCabin`: multi-cabin indicator

Later, based on model performance and multicollinearity, some of these cabin-derived features were dropped.

### 5) `PassengerName`

Initially used for family-surname-based experiments, but the final best model did not rely on it.

### 6) `FarePerPerson`

This feature was tested multiple times. It was useful early in experimentation, but the final best logistic regression submission performed better after dropping it.

---

## Feature Engineering Trials

The following engineered features were tested during development:

* `Child = Age < 16`
* `FemaleChild`
* `FirstClassFemale`
* `FamilyGroup`
* `Surname`
* `SurnameCount`
* `TicketGroupSize`
* `BerthKnown`
* `BerthCabinCount`
* `BerthMultiCabin`
* `BerthDeck`

### Important learning

Many of these features looked useful in isolation, but several were redundant with existing columns and created multicollinearity. In particular:

* `Child`
* `FemaleChild`
* `FamilyGroup`
* `SurnameCount`
* `TicketGroupSize`

did not improve the final leaderboard score.

---

## Model Comparison

Several models were tested.

### Models tried

* Logistic Regression
* Random Forest
* Extra Trees
* CatBoost

### Main result

For this dataset, **Logistic Regression** performed best and was the most stable model once the feature set was cleaned.

Observed performance patterns:

* Logistic Regression achieved the best cross-validation score among the tested models.
* Random Forest and Extra Trees were competitive but generally weaker.
* CatBoost did not outperform Logistic Regression on the final cleaned feature set.

---

## Multicollinearity Analysis

Variance Inflation Factor (VIF) was used to inspect redundancy.

### Findings

High VIF values were seen for:

* `BerthDeck_Missing`
* `Title_Mr`
* `Gender`
* `TicketTier`
* some berth deck categories
* `FamilySize`
* `Singleton`

This showed that several features were overlapping heavily.

### Lessons learned

* High VIF did **not** automatically mean a feature was useless.
* Many dummy columns had high VIF because of the dummy-variable trap.
* The more useful approach was to remove redundant engineered features only when both CV and leaderboard behavior confirmed they were noisy.

---

## Final Best Observations

The strongest stable signals in the final usable feature set were:

* `Gender`
* `Title_Mr`, `Title_Mrs`, `Title_Master`, `Title_Rare`
* `TicketTier`
* `FamilySize`
* `Singleton`
* selected `BerthDeck_*` dummy variables
* `BoardingPort_C`, `BoardingPort_S`

The model consistently showed that the Titanic-style survival pattern still dominates:

* women survived more often than men
* children and titled passengers had strong survival differences
* higher ticket tier tended to help survival
* family structure had a non-linear effect

---

## Best Public Leaderboard Result

The best observed leaderboard score during experimentation was:

* **0.84146** using a tuned Logistic Regression feature set

This was obtained after removing several noisy or redundant features.

### Notable results

* Over-engineering with extra family/surname features often reduced CV.
* The best improvement came from careful pruning rather than adding many derived features.
* A reduced logistic regression model outperformed more complex tree-based models in this setting.

---

## Why the Final Logistic Regression Worked Best

This dataset behaved unusually well for a linear model after cleaning because:

1. The strongest survival signals were already captured by the available features.
2. Many engineered features were redundant.
3. Tree-based models overfit noise more easily here.
4. Logistic Regression generalized better after removing weak features.

---

## Final Feature Engineering Takeaways

### Features that helped most

* `Gender`
* `Title_*`
* `TicketTier`
* `Singleton`
* `FamilySize`
* selected `BerthDeck_*`
* `BoardingPort_*`

### Features that were tried but did not improve the final score

* `PassengerId`
* `FarePerPerson`
* `TicketGroupSize`
* `SurnameCount`
* `Child`
* `FemaleChild`
* `FamilyGroup`
* `BerthCabinCount`
* `BerthMultiCabin`

---

## Data Access

The dataset is already included in this repository for convenience.

Alternatively, you can download it directly from Kaggle using `kagglehub`:

```python
import kagglehub

# Download latest version
path = kagglehub.competition_download('predict-survival-of-the-passengers')

print("Path to competition files:", path)
```
---

## Repository Structure

```text
.
├── final-notebook-84146.ipynb
├── data/
│   ├── maritime_sample_submission.csv
│   ├── maritime_test.csv
│   └── maritime_train.csv
└── README.md
```

---

## Reproducibility

To reproduce the results presented in this repository:

1. Install the required dependencies.
2. Place the dataset files in the appropriate directory.
3. Run the notebook sequentially from the first cell to the last.

The final output generated will be:

```text
submission.csv
```

---

## Summary of the Final Approach

The final best-performing workflow was:

1. Clean and normalize the raw data
2. Encode categorical variables properly
3. Engineer only a small set of meaningful features
4. Remove noisy / redundant columns
5. Train Logistic Regression
6. Tune the decision threshold if needed
7. Generate submission predictions

This competition was a good example of how, on tabular problems, **careful feature selection can matter more than model complexity**.

---
