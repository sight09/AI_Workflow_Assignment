"""
hospital_readmission_prediction.py

Author: Amanuel Alemu Zewdu
Purpose: Demonstration of a logistic regression workflow for predicting 30-day hospital readmission risk.
Language: Formal academic English in comments and docstrings.
Requirements: pandas, numpy, scikit-learn
Install: pip install pandas numpy scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def generate_synthetic_ehr(n=600, random_state=0):
    """
    Generate a synthetic EHR-like dataset.
    Features include:
      - age: integer
      - num_prev_adm: number of previous admissions
      - length_of_stay: days for current stay
      - comorbidity_score: aggregated score for chronic conditions
      - lab_abnormal: binary indicator of an abnormal lab result
    Target:
      - readmitted_30: binary (1 indicates readmission within 30 days)
    """
    rng = np.random.RandomState(random_state)
    age = rng.randint(18, 90, size=n)
    num_prev_adm = rng.poisson(1.2, size=n)
    length_of_stay = np.clip(rng.normal(5, 3, size=n), 1, 30).astype(int)
    comorbidity_score = np.clip(rng.poisson(2, size=n), 0, 10)
    lab_abnormal = rng.binomial(1, 0.2, size=n)
    risk = 0.02*(age-50) + 0.3*num_prev_adm + 0.08*length_of_stay + 0.25*comorbidity_score + 0.6*lab_abnormal
    prob_readmit = 1 / (1 + np.exp(-risk/3.0))
    readmitted_30 = (rng.rand(n) < prob_readmit).astype(int)
    df = pd.DataFrame({
        'age': age,
        'num_prev_adm': num_prev_adm,
        'length_of_stay': length_of_stay,
        'comorbidity_score': comorbidity_score,
        'lab_abnormal': lab_abnormal,
        'readmitted_30': readmitted_30
    })
    # introduce some missingness
    mask = rng.rand(n) < 0.04
    df.loc[mask, 'comorbidity_score'] = np.nan
    return df

def train_and_evaluate(df):
    X = df.drop(columns=['readmitted_30'])
    y = df['readmitted_30']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=1, stratify=y_temp)

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    return clf, cm, prec, rec, acc

def main():
    df = generate_synthetic_ehr(n=800)
    model, cm, prec, rec, acc = train_and_evaluate(df)
    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
