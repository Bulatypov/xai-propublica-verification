# COMPAS Recidivism Risk Score Analysis with Custom Random Forest

This notebook implements a custom Random Forest regressor to analyze and predict COMPAS recidivism risk scores, with additional functionality for fairness analysis and counterfactual explanations.

## Overview

The project includes:
1. A custom implementation of Decision Tree and Random Forest regressors from scratch
2. Analysis of the COMPAS dataset (recidivism risk scores)
3. Bias/fairness evaluation across demographic groups
4. Implementation of FACE (Fairness-Aware Counterfactual Explanations) to generate actionable recourse

## Dataset

The COMPAS dataset contains criminal defendant information from Broward County, including:
- Demographic features (age, race, gender)
- Criminal history features (prior counts, juvenile offenses)
- Charge degree (felony/misdemeanor)
- Target: `decile_score` (1-10 risk score)

Preprocessing steps:
- Filters invalid records (-30 < days_b_screening_arrest < 30)
- Handles missing data (median imputation for numerical, mode for categorical)
- One-hot encodes categorical variables

## Model Architecture

### Custom Decision Tree Regressor
- Recursive binary splitting based on MSE reduction
- Configurable max depth and minimum samples per split
- Handles both numerical and categorical features

### Custom Random Forest Regressor
- Ensemble of decision trees with:
  - Bootstrap aggregation (bagging)
  - Random feature subsets (√n features per split)
- Parallel training of independent trees
- Mean prediction aggregation

## Key Features

1. **Model Evaluation**:
   - Mean Absolute Error (MAE) on test set
   - Feature importance analysis

2. **Bias Analysis**:
   - Compares MAE and average scores across racial groups
   - Visualizes score distributions by race

3. **Counterfactual Explanations (FACE)**:
   - Generates actionable recourse to achieve desired score
   - Respects immutable attributes (e.g., race, age)
   - Shows minimal required feature changes

