# Fraud Detection App

A machine learning web app that predicts whether a credit card transaction is fraudulent, built on the IEEE-CIS Fraud Detection dataset.

**Live Demo:** https://fraud-detection-app-hccr.onrender.com

*(Note: hosted on a free tier - first load may take 30-50 seconds if the app has been inactive.)*

## Overview

This project detects fraudulent transactions using a Random Forest model trained on real-world, anonymized e-commerce transaction data. The dataset is highly imbalanced (only 3.5% of transactions are fraud), which made handling class imbalance a core part of the modeling process.

## Dataset

- **Source:** [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)
- **Size:** ~590,000 transactions, 434 features
- **Target:** `isFraud` (0 = legit, 1 = fraud)
- **Class distribution:** 96.5% legit, 3.5% fraud

## Approach

1. Merged transaction and identity data on `TransactionID`
2. Dropped columns with over 90% missing values
3. Imputed remaining missing values (median for numeric, "unknown" for categorical)
4. Label-encoded categorical features
5. Trained and compared Logistic Regression vs Random Forest (class-weighted, to handle imbalance)
6. Selected Random Forest as the final model based on performance

## Model Performance

| Metric | Logistic Regression | Random Forest |
|--------|---------------------|----------------|
| Fraud Precision | 0.07 | 0.27 |
| Fraud Recall | 0.70 | 0.74 |
| ROC-AUC | 0.75 | 0.92 |

**Confusion Matrix (Random Forest):**
- True Negatives: 105,861
- False Positives: 8,114
- False Negatives: 1,090
- True Positives: 3,043

## Visualizations

Fraud vs Legit distribution, feature importance, and confusion matrix are in the `visualizations/` folder.

## Tech Stack

- **Language:** Python
- **Data & ML:** Pandas, NumPy, Scikit-learn
- **Web App:** Flask
- **Deployment:** Render

## How the App Works

The model needs 420 input features to make a prediction, but a user can't realistically fill in all of them. The app takes 5 key inputs from the user (transaction amount, product code, card type, card category, email domain) and automatically fills the remaining features with their median/default values from training data.

## Limitations

- Most features (`C` and `V` columns) are anonymized by Kaggle for privacy reasons, so their real-world meaning is unknown - the model can show *which* features matter, but not *why* in business terms.
- Precision (0.27) is moderate — in a production system, this would need threshold tuning to reduce false positives before deployment.
- The model was trained with a recall-first tradeoff (catching more fraud, at the cost of more false alarms), which is a common and defensible choice in fraud detection but would need to be validated against actual business costs.

## Run Locally

```bash
git clone https://github.com/vaibhavanalytix/fraud-detection-app.git
cd fraud-detection-app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd app
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

## Author

**Vaibhav** - [GitHub](https://github.com/vaibhavanalytix)
