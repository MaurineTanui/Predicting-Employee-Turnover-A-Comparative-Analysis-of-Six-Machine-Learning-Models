# Predicting Employee Turnover: A Comparative Analysis of Six Machine Learning Models

A group predictive-modeling project comparing parametric and non-parametric machine learning approaches to forecast employee attrition, using a dataset of 7,590 employees across 20+ HR, job satisfaction, and organizational variables.

![Dashboard Screenshot](assets/turnover-analysis-screenshot.png)
*(Add a screenshot of a key visual — e.g., the model performance table or feature importance chart — here)*

## Overview

Employee turnover is one of the most costly and disruptive challenges in workforce management. This project builds and compares six predictive models to estimate the likelihood of an employee leaving an organization, based on demographic, behavioral, and performance-related factors — with the goal of helping HR teams move from reactive to proactive retention strategy.

## Dataset

- **7,590 employee observations**, 20+ original variables (expanded to 28 after preprocessing)
- **Target variable:** `left` — binary indicator of whether an employee left (1) or stayed (0)
- **Predictors include:** satisfaction level, last performance evaluation, number of projects, average monthly hours, tenure, work accidents, promotions, department, and salary level
- **Class distribution:** ~85% stayed / ~15% left (imbalanced classification problem)

## Methodology

### Preprocessing
- Ordinal encoding for salary (Low/Medium/High); one-hot encoding for nominal categorical variables
- Z-score standardization applied to numeric predictors for scale-sensitive models (logistic regression, SVM, neural network); tree-based models used unscaled data
- Verified no near-zero-variance predictors or identifier columns requiring exclusion

### Models Built
**Parametric:**
- Logistic Regression
- LASSO Logistic Regression

**Non-parametric:**
- Generalized Additive Model (GAM)
- Random Forest
- Support Vector Machine (RBF kernel)
- Deep Learning Neural Network (1 hidden layer)

### Validation
- 5-fold cross-validation with fixed random seed (42) for reproducibility
- `GridSearchCV` used for hyperparameter tuning across all models
- Train/test performance compared to check for overfitting and data drift

## Key Results

| Model | Accuracy (Test) | AUC (Test) | Recall (Test) |
|---|---|---|---|
| **SVM (RBF)** | 0.54 | **0.55** | 0.36 |
| Random Forest | 0.85 | 0.48 | 0.33 |
| GAM (Nonlinear) | 0.85 | 0.48 | 0.38 |
| Deep Learning | 0.84 | 0.46 | 0.32 |
| Logistic Regression | 0.49 | 0.48 | 0.36 |
| LASSO Logistic Regression | 0.49 | 0.47 | 0.35 |

**Model Ranking (by AUC — the most meaningful metric given class imbalance):**
1. **SVM (RBF)** — best discrimination between leavers and stayers
2. **Random Forest** — strong accuracy and interpretability, mild overfitting
3. **GAM** — smooth nonlinear effects, highly interpretable
4. **Deep Learning** — captures complex patterns, higher variance
5. **LASSO / Logistic Regression** — fast, interpretable baselines with high bias

## Top Predictors of Turnover

Aggregated across Random Forest, LASSO, and SVM feature importance rankings:

1. **Job Satisfaction** — the single strongest predictor of attrition
2. **Emotional Exhaustion** — burnout sharply raises exit probability
3. **Supervisor Satisfaction** — supportive leadership reduces turnover
4. **Work–Life Balance** — moderate but consistent influence
5. **Career Advancement** — lack of growth opportunities predicts leaving

## Business Implications

Psychological and managerial factors — not salary or tenure — are the dominant drivers of turnover. This has direct implications for HR strategy:
- **SVM (RBF)** is best suited for precision-driven risk flagging in retention dashboards
- **Random Forest** offers the best balance of accuracy and explainability for leadership reporting
- **GAM** is ideal for transparent, executive-facing presentations of nonlinear trends
- Retention initiatives should prioritize **job satisfaction, supervisor relationships, and workload management** over compensation-focused interventions

## Limitations & Future Work

- Class imbalance (~85/15 split) limited recall across all models; future work should explore class-balancing techniques such as SMOTE
- AUC values across most models remained modest (0.46–0.55), suggesting room for additional behavioral or engagement-based features
- Results should be validated against a rebalanced or expanded dataset before deployment in a live HR system

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn
- **pygam** — Generalized Additive Models
- **TensorFlow** — Deep Learning model
- **GridSearchCV** — hyperparameter tuning and cross-validation

## Team

- Maurine Tanui
- Poornima Anamaneni Sayeeswaran

*Group 2 — Predictive Modeling Project, AA5300 Advanced Analytics, Saint Louis University*

## Author (this repo)

**Maurine Tanui** — MSc Analytics, Saint Louis University
[LinkedIn](https://www.linkedin.com/in/maurine-cherono-cpa-016a29196/) · [Portfolio](https://maurinetanui.github.io/)
