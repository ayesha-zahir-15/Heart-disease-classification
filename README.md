# Heart Disease Risk Classifier

A logistic regression model that predicts the 10-year risk of coronary heart disease based on patient health data.

## Overview
Early identification of heart disease risk can help guide preventive care. This project builds a binary classification model using the Framingham Heart Study dataset to predict whether a patient is at risk of developing coronary heart disease within 10 years, based on clinical and lifestyle features.

## Dataset
- **Source:** Framingham Heart Study dataset
- Includes features such as age, blood pressure, cholesterol, smoking status, diabetes, and other cardiovascular risk factors

## Approach
1. **Data Cleaning** — handled missing values
2. **Exploratory Data Analysis** — examined feature distributions and correlations with heart disease risk
3. **Feature Engineering** — encoded categorical variables, applied **StandardScaler** to normalize feature ranges
4. **Model Training** — reshaped target labels using `.ravel()`, used **GridSearchCV** to tune hyperparameters, then fit a logistic regression model for binary classification
5. **Evaluation** — assessed performance using accuracy, confusion matrix, and classification report

## Results
- **Accuracy:** 86.3%
- **Precision (no disease / class 0):** 0.87 | **Recall:** 0.99
- **Precision (disease / class 1):** 0.71 | **Recall:** 0.11

While overall accuracy is high, the dataset is imbalanced (1,194 negative cases vs. 205 positive cases). 
This means the model is very good at identifying patients *without* heart disease risk but currently misses most patients who *are* at risk (recall of only 11% for the positive class). 
Accuracy alone is misleading here — recall on the positive class is the more clinically important metric, and it highlights a key limitation to address in future work.

## Insights
Logistic regression coefficients identified age as the strongest risk predictor at 29.2% of model influence followed by systolic blood pressure and daily cigarette usage 

## Tools Used
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## How to Run
1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open and run `LogisticRegression_HeartDisease.ipynb` in Jupyter Notebook

## Files
- `LogisticRegression_HeartDisease.ipynb` — main analysis and model

## Future Improvements
- Address class imbalance using techniques like SMOTE, class weighting, or resampling
- Try alternative models (Random Forest, XGBoost) better suited to imbalanced data
- Adjust classification threshold to improve recall on the positive class, given the clinical cost of missing at-risk patients
