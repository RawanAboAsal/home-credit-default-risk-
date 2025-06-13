# Home Credit Default Risk – A Comprehensive Machine Learning Approach

## Project Overview
This project predicts the likelihood of loan applicants failing to repay debts for [Home Credit](https://www.homecredit.net/), an international consumer finance provider. The solution combines **multi-table feature engineering**, **smart imputation strategies**, and **ensemble modeling** to help lenders make data-driven decisions while mitigating risk.

This project addresses the challenge of predicting loan repayment ability for individuals with limited or no credit history, supporting Home Credit’s mission to expand financial inclusion. By leveraging alternative data and machine learning, the goal is to improve credit risk assessment and ensure fair access to responsible lending [Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk).
## Project Structure

### 1. `EDA_and_Preprocessing.ipynb`
**Exploratory Data Analysis & Data Preprocessing**  
- Performs comprehensive data exploration across 8 datasets
- Handles missing values, outliers, and skewness
- Implements feature encoding and scaling
- Merges and aggregates multiple data sources

### 2. `Feature_Selection_and_Modeling.ipynb`
**Feature Engineering & Model Training**  
- Implements 7 different feature selection methods
- Trains and evaluates LightGBM models with cross-validation
- Optimizes model performance through hyperparameter tuning
- Generates submission files for Kaggle competition

## Key Findings

### Data Insights
- Significant class imbalance (8% default rate)
- Strong predictors: EXT_SOURCE features, DAYS_BIRTH, DAYS_EMPLOYED
- Demographic trends: Younger applicants and those with unstable employment history more likely to default

### Model Performance
- Best cross-validation AUC: 0.7749
- Top Kaggle scores: 
  - Public leaderboard: 0.772
  - Private leaderboard: 0.774
- Most effective feature selection: Correlation-based subset (196 features)

## Technical Approach

### Data Processing
- Missing value imputation using correlation-aware methods
- Outlier detection and capping with IQR
- Skewness correction using Yeo-Johnson and Box-Cox transforms
- Categorical encoding (Label, One-Hot, Binary)

### Feature Engineering
- Created meaningful ratios (Annuity/Income, Credit/Income)
- Aggregated bureau and previous application data
- Time-based feature engineering from payment histories

### Modeling
- LightGBM with class weighting for imbalance
- Stratified 5-10 fold cross-validation
- Feature importance analysis and pruning
- Multiple feature selection approaches tested

