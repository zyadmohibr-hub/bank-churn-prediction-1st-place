# 🏆 1st Place Solution — Bank Customer Churn Prediction

Achieved **1st place** (Public LB: **0.94048** AUC) out of 55 teams in a Kaggle-style competition.

## 📊 Overview
Binary classification task predicting whether a bank customer will churn (`Exited`), evaluated using ROC-AUC.

## 🔑 Key Techniques
- **Base model:** CatBoost, hyperparameter-tuned with Optuna
- **Feature engineering:** interaction features, KMeans clustering, ratio-based features
- **K-Fold Target Encoding** on the `Surname` column — a feature typically dropped, but turned out to carry strong signal
- **Multi-seed averaging** (5 seeds) for prediction stability
- **Weighted model blending**, optimized via `scipy.optimize` on out-of-fold predictions
- **Data augmentation:** merged the original Bank Customer Churn dataset (10K extra rows) as permitted by competition rules — the single biggest score improvement

## 📈 Score Progression
| Stage | Public LB AUC |
|---|---|
| Baseline CatBoost | 0.9385 |
| + Optimized blend weights | 0.9387 |
| + Surname target encoding | 0.9386 |
| + External data augmentation | **0.9405** |

## 🛠️ Tools
Python, CatBoost, XGBoost, LightGBM, Optuna, scikit-learn, pandas, NumPy
