# Notebook Folder

## Overview

This folder contains Jupyter notebooks that guide the BrightPath-Grade-Predictor project through its workflow, from data preparation to modeling. Each notebook corresponds to specific steps in the project instructions, ensuring a structured approach to predicting student performance (GradeClass) at BrightPath Academy.

## Contents

The notebooks are organized to follow the project steps as outlined:

### data_preparation.ipynb:

- Step 3: Getting the System Ready and Loading the Data - Sets up the environment (e.g., imports libraries) and loads the raw dataset - ----(student_data.csv) from data/raw/.
- Step 4: Understanding the Data - Performs initial exploration, such as checking data types, summary statistics, and missing values.
- Step 6: Missing Value and Outlier Treatment - Handles missing values (e.g., imputation) and outliers (e.g., capping), saving the cleaned data to data/processed/.

### eda.ipynb:

- Step 5: Exploratory Data Analysis (Univariate and Bivariate): Conducts detailed exploratory data analysis, including univariate analysis (e.g., histograms of Age or GPA) and bivariate analysis (e.g., scatter plots of StudyTimeWeekly vs. GradeClass). This notebook also tests hypotheses listed in the README.md (e.g., "Higher study time improves grades") using visualizations and statistical methods.

### feature_engineering.ipynb:

- Step 8: Feature Engineering: Creates and transforms features to improve model performance, such as encoding categorical variables (e.g., Gender, Ethnicity) or creating new features (e.g., combining extracurricular activities into a single metric). Reusable functions are stored in src/feature_engineering.py.

### modeling.ipynb:

- Step 7: Evaluation Metrics for Classification Problem - Defines metrics (e.g., accuracy, precision, recall, F1-score) for evaluating model performance, implemented via src/evaluate.py.
- Step 9: Model Building: Part 1 (Baseline ML Models) - Builds and evaluates baseline machine learning models (Logistic Regression, Random Forest, XGBoost) using src/model.py.
- Step 10: Model Building: Part 2 (Deep Learning Model) - Implements a deep learning classification model (e.g., a neural network) to predict GradeClass, also using src/model.py.

## Notes

- Use these notebooks for interactive analysis, visualization, and documenting findings in a step-by-step manner.
- Outputs such as plots (e.g., grade_dist.png) or trained models (e.g., trained_model.pkl) should be saved to the artifacts/ folder.
- Ensure notebooks are well-documented with markdown cells to explain each step, making the workflow clear for team members and reviewers.
