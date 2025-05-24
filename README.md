# Credit Risk Prediction App

A Streamlit-based interactive web app to predict credit risk (loan approval) using machine learning models.  
This app allows users to upload their own credit dataset, train multiple models, evaluate performance, and test predictions on sample inputs.

---

## Project Description

This project is designed to help financial institutions and lenders evaluate credit risk by predicting the likelihood of loan approval or default using historical credit data. It leverages several popular classification algorithms to build predictive models and provides an easy-to-use web interface for data upload, model training, evaluation, and sample predictions.

---

## Features

- Upload any CSV dataset containing credit-related data
- Select the target variable (loan approval or credit risk label)
- Automatic data preprocessing including:
  - Handling missing values
  - Encoding categorical variables
- Train multiple models:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- Evaluate models with metrics:
  - Accuracy
  - ROC-AUC Score
  - Classification Report (precision, recall, f1-score)
- Interactive sample prediction on user-provided input values
- Visualization support using seaborn and matplotlib (extendable)

---

## Benefits

- Helps lenders automate credit risk assessment
- Facilitates data-driven decision making to reduce loan defaults
- Modular and extensible for other binary or multiclass classification tasks
- Provides transparency with detailed evaluation reports
- Easy to use via web interface without coding skills

---

## Models Used

| Model               | Description                                       |
|---------------------|-------------------------------------------------|
| Logistic Regression | Baseline linear model for binary classification |
| Random Forest       | Ensemble method using multiple decision trees   |
| XGBoost             | Gradient boosting tree-based model, powerful for tabular data |

---


