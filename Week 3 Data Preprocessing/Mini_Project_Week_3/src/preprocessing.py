import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_clean_raw_data(filepath: str) -> pd.DataFrame:
    """Loads raw data, strips string spaces, lowers case, cleans domain rules."""
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    
    # Standardize string fields
    if 'city' in df.columns:
        df['city'] = df['city'].astype(str).str.strip().str.lower()
    if 'internet_access' in df.columns:
        df['internet_access'] = df['internet_access'].astype(str).str.strip().str.capitalize()
        
    # Domain Rule Validations
    if 'attendance' in df.columns:
        df.loc[(df['attendance'] < 0) | (df['attendance'] > 100), 'attendance'] = np.nan
        
    return df

def get_preprocessing_pipeline(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Creates a scikit-learn ColumnTransformer for numerical and categorical pipelines."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

def prepare_data(filepath: str):
    """Executes full reproducible workflow."""
    df = load_and_clean_raw_data(filepath)
    
    X = df.drop(columns=['student_id', 'final_score'], errors='ignore')
    y = df['final_score']
    
    num_cols = ['study_hours', 'attendance', 'previous_score']
    cat_cols = ['city', 'internet_access']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = get_preprocessing_pipeline(num_cols, cat_cols)
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor

if __name__ == "__main__":
    data_path = "data/student_performance.csv"
    print("--- Running Preprocessing Pipeline ---")
    
    X_train_proc, X_test_proc, y_train, y_test, preprocessor = prepare_data(data_path)
    
    print("X_train_proc shape:", X_train_proc.shape)
    print("X_test_proc shape:", X_test_proc.shape)
    print("Preprocessing completed successfully!")