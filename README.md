# Retail Customer Spend Prediction
## Complete MLOps Pipeline

**By Siddharth Shrivastava | AI/ML Engineer**

End-to-end MLOps pipeline predicting
total customer spend using Random Forest
Regression with complete CI/CD automation.

## Model Performance
| Metric | Value  |
|--------|--------|
| MAE    | 370 GBP |
| RMSE   | 2399 GBP |
| R2     | 0.85   |
| MAPE   | 9.6%   |

## Dataset
- Source: UCI Online Retail II
- Raw records: 1,067,371 transactions
- After cleaning: 724,560 rows
- Customers: 5,340 UK customers
- Features: 16 engineered features

## Tech Stack
- ML: Python, Pandas, Scikit-learn
- Model: Random Forest Regressor
- Testing: pytest 44 tests
- CI/CD: GitHub Actions
- Cloud: AWS S3 + SageMaker
- API: FastAPI + Docker

## Customer Segments
| Segment      | Spend Range  | Action               |
|--------------|--------------|----------------------|
| VIP          | 5000+ GBP    | Dedicated manager    |
| High Value   | 2000-5000    | Premium offer        |
| Medium Value | 500-2000     | Standard offer       |
| Standard     | below 500    | Regular email        |

## Run Locally
pip install -r requirements.txt
python src/data_cleaning.py
python src/feature_engineering.py
python src/train_model.py
python src/validate_metrics.py
pytest tests/ -v
