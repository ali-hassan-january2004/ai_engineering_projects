import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath: str):
    """Loads CSV dataset, handles missing values, and returns train/test splits."""
    df = pd.read_csv(filepath)
    
    # 1. Clean missing values (NaNs) across features and targets
    df = df.dropna().copy()
    
    # 2. Define binary pass target without target leakage
    df["pass"] = (df["final_score"] >= 50).astype(int)
    
    features = ["study_hours", "attendance", "previous_score"]
    X = df[features]
    y_reg = df["final_score"]
    y_cls = df["pass"]
    
    # Verify dataset size after cleaning
    print(f"[Data Check] Cleaned dataset row count: {len(df)}")
    if len(df) < 5:
        raise ValueError("Dataset has fewer than 5 rows after dropna(). Add more sample rows to student_performance.csv.")
    
    # 3. Split Regression data (80% train, 20% test)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.20, random_state=42
    )
    
    # 4. Split Classification data
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cls, test_size=0.20, random_state=42
    )
    
    return (X_train_r, X_test_r, y_train_r, y_test_r), (X_train_c, X_test_c, y_train_c, y_test_c)