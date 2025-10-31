"""
student_dropout_prediction.py

Author: Amanuel Alemu Zewdu
Purpose: Demonstration of an end-to-end AI workflow for predicting student dropout risk.
Language: Formal academic English in comments and docstrings.
Requirements: pandas, numpy, scikit-learn
Install: pip install pandas numpy scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(n=500, random_state=42):
    """
    Generate a synthetic dataset emulating student records.
    Features:
      - gpa: Grade point average (0.0 - 4.0)
      - attendance_rate: Fraction of attended classes (0.0 - 1.0)
      - assignments_submitted: integer count
      - major: categorical
      - socioeconomic_status: categorical (low, medium, high)
    Target:
      - dropped_out: binary (1 indicates dropout)
    """
    rng = np.random.RandomState(random_state)
    gpa = np.clip(rng.normal(2.8, 0.6, size=n), 0.0, 4.0)
    attendance = np.clip(rng.beta(5,2, size=n), 0.0, 1.0)
    assignments = rng.poisson(8, size=n)
    majors = rng.choice(['CS','Business','Arts','Engineering'], size=n, p=[0.3,0.25,0.2,0.25])
    ses = rng.choice(['low','medium','high'], size=n, p=[0.25,0.5,0.25])
    # construct risk score and generate label
    risk_score = (2.5 - gpa) + (0.5 - attendance) + (5 - assignments)/10 + (ses == 'low')*0.5
    prob_dropout = 1 / (1 + np.exp(-risk_score))
    dropped_out = (rng.rand(n) < prob_dropout).astype(int)
    df = pd.DataFrame({
        'gpa': gpa,
        'attendance_rate': attendance,
        'assignments_submitted': assignments,
        'major': majors,
        'socioeconomic_status': ses,
        'dropped_out': dropped_out
    })
    # Introduce some missing values for preprocessing demo
    for col in ['gpa','attendance_rate']:
        mask = rng.rand(n) < 0.03
        df.loc[mask, col] = np.nan
    return df

def build_and_evaluate(df):
    """
    Build a Random Forest classifier within a preprocessing pipeline, then evaluate.
    """
    X = df.drop(columns=['dropped_out'])
    y = df['dropped_out']

    # split the dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp)

    numeric_features = ['gpa','attendance_rate','assignments_submitted']
    categorical_features = ['major','socioeconomic_status']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0))
    ])

    clf.fit(X_train, y_train)

    # Evaluate on validation and test sets
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)

    metrics = {
        'val_accuracy': accuracy_score(y_val, y_pred_val),
        'val_precision': precision_score(y_val, y_pred_val, zero_division=0),
        'val_recall': recall_score(y_val, y_pred_val, zero_division=0),
        'val_f1': f1_score(y_val, y_pred_val, zero_division=0),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
        'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
        'test_f1': f1_score(y_test, y_pred_test, zero_division=0)
    }

    return clf, metrics

def main():
    df = generate_synthetic_data(n=800)
    model, metrics = build_and_evaluate(df)
    print("Evaluation metrics (validation and test):")
    for k,v in metrics.items():
        print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    main()
