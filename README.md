# HeartPredict-AI

*HeartPredict-AI* is a machine learning-based mini-project designed to predict the presence of heart disease based on patient medical data. It uses a variety of ML algorithms and presents the results through a user-friendly web application.

## Features

- *Prediction Models*:
  - Random Forest
  - Decision Tree
  - Logistic Regression
  - K-Nearest Neighbors (KNN)

- *Techniques Used*:
  - Data preprocessing and feature selection
  - SMOTE (Synthetic Minority Over-sampling Technique) for class balancing
  - Hyperparameter tuning
  - Evaluation metrics: Accuracy, F1 Score, Precision, ROC AUC
  - Confusion matrix and model comparison visualization

- *Web Application*:
  - Built with Python (Flask) for the backend
  - HTML,CSS, Javascript and Bootstrap for the frontend
  - Accepts user inputs for key health parameters (age, cholesterol, blood pressure, etc.)
  - Provides instant predictions on heart disease risk
  
## Objective

The goal of this project is to demonstrate how machine learning can be applied in the healthcare domain to assist with early detection of heart disease, which can help in taking preventive actions.

## Dataset

The dataset used is sourced from Kaggle and contains over 1000 records with features like:
- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Cholesterol level (chol)
- Fasting blood sugar (fbs)
- Resting ECG (restecg)
- Maximum heart rate achieved (thalach)
- Exercise-induced angina (exang)
- ST depression (oldpeak)
- Slope of the peak exercise ST segment (slope)
- Number of major vessels (ca)
- Thalassemia (thal)

## Tools & Libraries

- Python
- Flask
- Pandas
- Scikit-learn
- Matplotlib / Seaborn (for visualization)

