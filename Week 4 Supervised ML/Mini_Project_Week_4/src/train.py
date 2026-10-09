import numpy as np
import pandas as pd
from preprocessing import load_and_preprocess_data
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def execute_experiments():
    reg_data, cls_data = load_and_preprocess_data("data/student_performance.csv")
    X_tr_r, X_te_r, y_tr_r, y_te_r = reg_data
    X_tr_c, X_te_c, y_tr_c, y_te_c = cls_data

    print("=== PART A: REGRESSION RESULTS ===")
    reg_models = {
        "Dummy Regressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    for name, model in reg_models.items():
        model.fit(X_tr_r, y_tr_r)
        preds = model.predict(X_te_r)
        mae = mean_absolute_error(y_te_r, preds)
        rmse = np.sqrt(mean_squared_error(y_te_r, preds))
        r2 = r2_score(y_te_r, preds)
        print(f"[{name}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.2f}")

    print("\n=== PART B: CLASSIFICATION RESULTS ===")
    cls_models = {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for name, model in cls_models.items():
        model.fit(X_tr_c, y_tr_c)
        preds = model.predict(X_te_c)
        acc = accuracy_score(y_te_c, preds)
        prec = precision_score(y_te_c, preds, zero_division=0)
        rec = recall_score(y_te_c, preds, zero_division=0)
        f1 = f1_score(y_te_c, preds, zero_division=0)
        print(f"[{name}] Acc: {acc:.2f} | Prec: {prec:.2f} | Rec: {rec:.2f} | F1: {f1:.2f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_te_c, preds)}\n")

if __name__ == "__main__":
    execute_experiments()