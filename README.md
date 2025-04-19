Credit Risk Assessment Using Machine Learning

📝 Project Overview

A machine learning-based solution to assess the credit risk of bank loan applicants using classification models. The goal is to predict whether a loan applicant is likely to default, enabling financial institutions to make informed lending decisions.

📁 Dataset

Source: bankloans.csv
Features: Includes financial, demographic, and credit-related variables of loan applicants.
Target Variable: Indicates whether a customer is a credit risk (binary classification).
🔍 Objectives

Predict whether a loan applicant is a credit risk.
Compare the performance of different classification models.
Fine-tune hyperparameters using GridSearchCV.
Evaluate models using various performance metrics.
🧠 Models Used

Logistic Regression
Support Vector Classifier (SVC)
Random Forest Classifier
🛠️ Techniques & Tools

Data Preprocessing (handling missing values, encoding categorical data, feature scaling)
Model Training and Evaluation
Hyperparameter Tuning with GridSearchCV
Performance Metric Analysis
🧪 Evaluation Metrics

accuracy_score
r2_score
confusion_matrix
mean_squared_error
📊 Results

All models were evaluated on test data using the above metrics.
Random Forest Classifier showed the highest accuracy.
GridSearchCV improved model performance by tuning hyperparameters effectively.
🧾 Requirements

Python
NumPy
Pandas
Scikit-learn
Matplotlib / Seaborn (for visualization, if used)
Jupyter Notebook
📂 File Structure

credit_risk_assessment.ipynb: Jupyter notebook containing the full analysis and modeling.
bankloans.csv: Dataset used for training and testing the models.
