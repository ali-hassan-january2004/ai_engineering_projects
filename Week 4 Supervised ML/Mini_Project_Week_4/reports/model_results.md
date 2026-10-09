# Supervised ML Experiment Results

## Data Summary
- Total cleaned records: 10
- Train split: 8 samples
- Test split: 2 samples

## Part A: Regression Performance
| Model | MAE | RMSE | R2 | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 23.50 | 24.42 | -0.08 | Predicts mean target value |
| **Linear Regression** | 1.11 | 1.50 | 1.00 | Top regression performer |
| **Random Forest Regressor** | 6.85 | 7.01 | 0.91 | Effective, limited by dataset size |

## Part B: Classification Performance
| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.50 | 0.50 | 1.00 | 0.67 |
| **Logistic Regression** | 1.00 | 1.00 | 1.00 | 1.00 |
| **Random Forest Classifier** | 1.00 | 1.00 | 1.00 | 1.00 |

## Observations & Key Findings
1. **Baseline Comparison**: Both Linear Regression and Logistic Regression beat their dummy baselines[cite: 3, 4].
2. **Data Scale Note**: The test set contains 2 samples due to the 10-row dataset size. Evaluating on a larger dataset (e.g., 100+ rows) will yield more reliable cross-validation metrics[cite: 5, 6].