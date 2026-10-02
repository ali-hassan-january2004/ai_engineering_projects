# Data Preprocessing Decision Log

## Summary
- **Data Ingestion**: Processed 1,000 raw student records.
- **Duplicates**: Removed 12 duplicate entries.
- **Domain Corrections**: Set negative attendance values (`< 0%`) to `NaN`.

## Imputation Strategy
- **Numeric Features (`study_hours`, `attendance`, `previous_score`)**: Replaced missing values using **Median Imputation** to remain robust against skewness/outliers.
- **Categorical Features (`city`, `internet_access`)**: Replaced missing values using **Most Frequent (Mode) Imputation**.

## Feature Scaling & Encoding
- **Categoricals**: One-Hot Encoded with `handle_unknown='ignore'` to handle unseen cities in production without throwing errors.
- **Numerics**: Standardized using `StandardScaler` (Mean = 0, Std = 1).

## Leakage Prevention
- All transformers (`StandardScaler`, `SimpleImputer`, `OneHotEncoder`) were fitted **exclusively on training data (`X_train`)** and then applied to test data (`X_test`).