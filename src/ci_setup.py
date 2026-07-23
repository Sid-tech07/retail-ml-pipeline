import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib, json, os

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

np.random.seed(42)
nc = 300

customers = pd.DataFrame({
    "Customer ID"             : range(10000, 10000+nc),
    "total_spend"             : np.random.lognormal(7, 1, nc),
    "num_transactions"        : np.random.randint(1, 100, nc),
    "num_unique_invoices"     : np.random.randint(1, 50, nc),
    "avg_transaction_amount"  : np.random.uniform(10, 500, nc),
    "max_transaction_amount"  : np.random.uniform(100, 2000, nc),
    "min_transaction_amount"  : np.random.uniform(1, 50, nc),
    "std_transaction_amount"  : np.random.uniform(5, 200, nc),
    "median_transaction"      : np.random.uniform(10, 300, nc),
    "max_price"               : np.random.uniform(10, 100, nc),
    "days_since_last_purchase": np.random.randint(1, 365, nc),
    "customer_lifetime_days"  : np.random.randint(30, 700, nc),
    "purchase_frequency_days" : np.random.uniform(5, 60, nc),
    "num_unique_products"     : np.random.randint(1, 50, nc),
    "num_unique_descriptions" : np.random.randint(1, 50, nc),
    "avg_items_per_invoice"   : np.random.uniform(1, 20, nc),
    "high_value_ratio"        : np.random.uniform(0, 1, nc),
    "r_score"                 : np.random.randint(1, 6, nc),
    "f_score"                 : np.random.randint(1, 6, nc),
    "m_score"                 : np.random.randint(1, 6, nc),
    "peak_shopping_month"     : np.random.randint(1, 13, nc),
    "q4_spend"                : np.random.uniform(0, 5000, nc),
    "q4_spend_ratio"          : np.random.uniform(0, 1, nc),
    "weekend_purchases"       : np.random.randint(0, 20, nc),
    "high_value_txn_count"    : np.random.randint(0, 20, nc),
    "frequency"               : np.random.randint(1, 50, nc),
    "monetary"                : np.random.lognormal(7, 1, nc),
    "rfm_score"               : np.random.randint(3, 16, nc),
    "weekend_ratio"           : np.random.uniform(0, 1, nc),
    "avg_day_of_week"         : np.random.uniform(0, 6, nc),
})

n = 2000
transactions = pd.DataFrame({
    "Invoice"     : [f"INV{i}" for i in range(n)],
    "StockCode"   : np.random.choice(["A1","B2","C3"], n),
    "Description" : np.random.choice(["Prod A","Prod B"], n),
    "Quantity"    : np.random.randint(1, 50, n),
    "InvoiceDate" : pd.date_range("2010-01-01", periods=n, freq="h").astype(str),
    "Price"       : np.random.uniform(1, 50, n),
    "Customer ID" : np.random.randint(10000, 18000, n),
    "Country"     : "United Kingdom",
    "amount"      : np.random.uniform(10, 500, n),
    "year"        : 2010,
    "month"       : np.random.randint(1, 13, n),
    "day_of_week" : np.random.randint(0, 7, n),
    "hour"        : np.random.randint(8, 18, n),
    "is_weekend"  : np.random.randint(0, 2, n),
    "quarter"     : np.random.randint(1, 5, n),
})

transactions.to_csv("data/clean_transactions.csv", index=False)
customers.to_csv("data/customer_features.csv", index=False)
print(f"Data generated: {nc} customers {n} transactions")

feature_cols = [
    "num_transactions", "num_unique_invoices",
    "avg_transaction_amount", "max_transaction_amount",
    "min_transaction_amount", "std_transaction_amount",
    "median_transaction", "max_price",
    "days_since_last_purchase", "customer_lifetime_days",
    "purchase_frequency_days", "num_unique_products",
    "num_unique_descriptions", "avg_items_per_invoice",
    "high_value_ratio", "m_score"
]

with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

stats = {}
for col in feature_cols:
    stats[col] = {
        "mean"  : float(customers[col].mean()),
        "std"   : float(customers[col].std()),
        "min"   : float(customers[col].min()),
        "max"   : float(customers[col].max()),
        "median": float(customers[col].median())
    }
with open("models/training_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("Training model...")
X     = customers[feature_cols]
y     = customers["total_spend"]
y_log = np.log1p(y)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_tr_s, y_tr)

y_pred = np.expm1(model.predict(X_te_s))
y_act  = np.expm1(y_te)
mae    = mean_absolute_error(y_act, y_pred)
rmse   = np.sqrt(mean_squared_error(y_act, y_pred))
r2     = r2_score(y_act, y_pred)
mape   = np.mean(np.abs((y_act - y_pred) / y_act)) * 100

joblib.dump(model,  "models/rf_regressor.pkl")
joblib.dump(scaler, "models/scaler.pkl")

metrics = {
    "mae"          : round(mae, 2),
    "rmse"         : round(rmse, 2),
    "r2"           : round(r2, 4),
    "mape"         : round(mape, 2),
    "log_transform": True,
    "features_used": feature_cols,
    "model_name"   : "RF CI/CD Pipeline"
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"MAE={mae:.0f} R2={r2:.3f} MAPE={mape:.1f}%")
print("Setup complete!")