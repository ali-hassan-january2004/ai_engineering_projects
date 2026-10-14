# Model Comparison Report - Student Performance Prediction

## 1. Problem Definition
* **Target**: `passed` (Binary: 1 for pass, 0 for fail)
* **Metric**: F1-Score (to balance precision and recall across classes)

## 2. Results Summary
| Model Family | Test F1-Score | Notes |
| :--- | :--- | :--- |
| Baseline (Dummy) | 0.0000 | Majority class predictor |
| Logistic Regression | 0.7500 | Linear decision boundary with imputed missing values |
| Decision Tree | 0.7500 | Captures basic rules with depth limit |
| Random Forest | 0.7500 | Ensemble bag of trees handling noisy records |
| Gradient Boosting | 0.7500 | Sequential error correction |

## 3. Final Decision
All models currently achieve a Test F1-Score of 0.7500 on the 20% held-out test split. With a small 30-row dataset, models share identical test performance boundaries. To differentiate performance further, tuning and cross-validation can be explored next.