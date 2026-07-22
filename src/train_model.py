"""
Step 4b: Improved Model Training
Better hyperparameters + feature selection
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── Load Features ─────────────────────────────────
print("Loading features...")
df = pd.read_csv('data/customer_features.csv')
print(f"Customers: {len(df):,}\n")


# ═══════════════════════════════════════════════════
# IMPROVEMENT 1 — Better Feature Selection
# Keep only features with importance > 0.01
# ═══════════════════════════════════════════════════

print("="*55)
print("IMPROVEMENT 1 — Better Feature Selection")
print("="*55)

# Based on feature importance from last run
# Remove very low importance features
final_feature_cols = [
    'num_transactions',         # 0.1419
    'num_unique_invoices',      # 0.1526
    'avg_transaction_amount',   # 0.0267
    'max_transaction_amount',   # 0.0468
    'min_transaction_amount',   # keep
    'std_transaction_amount',   # keep
    'median_transaction',       # keep
    'max_price',                # keep
    'days_since_last_purchase', # 0.0191
    'customer_lifetime_days',   # 0.0653
    'purchase_frequency_days',  # 0.0453
    'num_unique_products',      # 0.0670
    'num_unique_descriptions',  # 0.0969
    'avg_items_per_invoice',    # 0.0327
    'high_value_ratio',         # keep
    'm_score',                  # 0.2471 ← BEST!

    # Remove these (importance < 0.01):
    # r_score          : 0.0052
    # weekend_ratio    : 0.0038
    # avg_day_of_week  : 0.0018
    # f_score          : 0.0017
    # peak_month       : 0.0009
]

print(f"Features selected: {len(final_feature_cols)}")
for i, f in enumerate(final_feature_cols, 1):
    print(f"  {i:2d}. {f}")

X = df[final_feature_cols]
y = df['total_spend']

# Log transform
y_log = np.log1p(y)
print(f"\nLog transform applied ✅")
print(f"Skewness: {y.skew():.2f} → "
      f"{y_log.skew():.2f}")


# ═══════════════════════════════════════════════════
# IMPROVEMENT 2 — Better Train Test Split
# Stratified by spend level
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("IMPROVEMENT 2 — Stratified Split")
print("="*55)

# Create spend buckets for stratification
# This ensures all spend levels in both sets
spend_buckets = pd.qcut(
    y_log,
    q=5,
    labels=False,
    duplicates='drop'
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X, y_log,
        test_size=0.2,
        random_state=42,
        stratify=spend_buckets
    )
)

print(f"Train: {len(X_train):,}")
print(f"Test : {len(X_test):,}")
print(f"Stratified by spend bucket ✅")

# Scale
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# ═══════════════════════════════════════════════════
# IMPROVEMENT 3 — Compare Multiple Models
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("IMPROVEMENT 3 — Model Comparison")
print("="*55)

models = {
    'Random Forest (baseline)': RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    'Random Forest (tuned)': RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
}

results = {}
print(f"\n{'Model':<30} {'MAE':>10} "
      f"{'RMSE':>10} {'R²':>8} {'MAPE':>8}")
print("-"*70)

best_model   = None
best_r2      = 0
best_model_name = ""

for name, model in models.items():
    # Train
    model.fit(X_train_s, y_train)

    # Predict and reverse transform
    y_pred_log    = model.predict(X_test_s)
    y_pred_actual = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    # Metrics
    mae  = mean_absolute_error(
        y_test_actual, y_pred_actual
    )
    rmse = np.sqrt(mean_squared_error(
        y_test_actual, y_pred_actual
    ))
    r2   = r2_score(
        y_test_actual, y_pred_actual
    )
    mape = np.mean(
        np.abs(
            (y_test_actual - y_pred_actual) /
            y_test_actual
        )
    ) * 100

    results[name] = {
        'model': model,
        'mae'  : mae,
        'rmse' : rmse,
        'r2'   : r2,
        'mape' : mape
    }

    print(f"{name:<30} £{mae:>8,.0f} "
          f"£{rmse:>8,.0f} {r2:>8.4f} "
          f"{mape:>7.1f}%")

    if r2 > best_r2:
        best_r2         = r2
        best_model      = model
        best_model_name = name

print(f"\n✅ Best model: {best_model_name}")
print(f"   R² = {best_r2:.4f}")


# ═══════════════════════════════════════════════════
# IMPROVEMENT 4 — Segment Analysis
# Check performance per customer segment
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("IMPROVEMENT 4 — Segment Analysis")
print("="*55)

y_pred_best = np.expm1(
    best_model.predict(X_test_s)
)
y_test_orig = np.expm1(y_test)

# Create segments
test_df = pd.DataFrame({
    'actual'   : y_test_orig,
    'predicted': y_pred_best
})

def get_segment(spend):
    if spend >= 5000:   return 'VIP'
    elif spend >= 2000: return 'High Value'
    elif spend >= 500:  return 'Medium Value'
    else:               return 'Standard'

test_df['segment'] = test_df['actual'].apply(
    get_segment
)

print(f"\nPerformance by segment:")
print(f"{'Segment':<15} {'Count':>6} "
      f"{'MAE':>10} {'MAPE':>8} "
      f"{'Accuracy':>10}")
print("-"*55)

for segment in ['Standard', 'Medium Value',
                'High Value', 'VIP']:
    seg_df = test_df[
        test_df['segment'] == segment
    ]
    if len(seg_df) == 0:
        continue

    seg_mae  = mean_absolute_error(
        seg_df['actual'], seg_df['predicted']
    )
    seg_mape = np.mean(
        np.abs(
            (seg_df['actual'] -
             seg_df['predicted']) /
            seg_df['actual']
        )
    ) * 100
    accuracy = 100 - seg_mape

    print(
        f"{segment:<15} {len(seg_df):>6} "
        f"£{seg_mae:>8,.0f} {seg_mape:>7.1f}% "
        f"{accuracy:>9.1f}%"
    )


# ═══════════════════════════════════════════════════
# FINAL MODEL EVALUATION
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("FINAL MODEL EVALUATION")
print("="*55)

best_results = results[best_model_name]
mae  = best_results['mae']
rmse = best_results['rmse']
r2   = best_results['r2']
mape = best_results['mape']

print(f"\nBest Model: {best_model_name}")
print(f"{'='*45}")
print(f"  MAE   : £{mae:,.2f}")
print(f"  RMSE  : £{rmse:,.2f}")
print(f"  R²    : {r2:.4f}")
print(f"  MAPE  : {mape:.1f}%")
print(f"{'='*45}")

# CV on best model
cv = cross_val_score(
    best_model,
    X_train_s, y_train,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
print(f"\n5-Fold CV MAE: {-cv.mean():.4f} "
      f"(±{cv.std():.4f})")


# ═══════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(16, 12))
fig.suptitle(
    f'Best Model: {best_model_name}\n'
    f'MAE=£{mae:,.0f} | R²={r2:.4f} | '
    f'MAPE={mape:.1f}%',
    fontsize=13, fontweight='bold'
)

y_pred_plot = np.expm1(
    best_model.predict(X_test_s)
)
y_test_plot = np.expm1(y_test)

# Plot 1: Actual vs Predicted
axes[0][0].scatter(
    y_test_plot, y_pred_plot,
    alpha=0.4, color='steelblue', s=20
)
max_val = max(y_test_plot.max(),
              y_pred_plot.max())
axes[0][0].plot(
    [0, max_val], [0, max_val],
    'r--', linewidth=2,
    label='Perfect'
)
axes[0][0].set_title(
    f'Actual vs Predicted\nR²={r2:.4f}'
)
axes[0][0].set_xlabel('Actual (£)')
axes[0][0].set_ylabel('Predicted (£)')
axes[0][0].legend()

# Plot 2: Residuals
residuals = y_test_plot - y_pred_plot
axes[0][1].scatter(
    y_pred_plot, residuals,
    alpha=0.4, color='coral', s=20
)
axes[0][1].axhline(
    y=0, color='black',
    linestyle='--', linewidth=2
)
axes[0][1].set_title('Residual Plot')
axes[0][1].set_xlabel('Predicted (£)')
axes[0][1].set_ylabel('Residual (£)')

# Plot 3: Error distribution
axes[1][0].hist(
    residuals.clip(
        lower=np.percentile(residuals, 5),
        upper=np.percentile(residuals, 95)
    ),
    bins=40, color='steelblue',
    edgecolor='white'
)
axes[1][0].axvline(
    x=0, color='red',
    linestyle='--', linewidth=2
)
axes[1][0].set_title('Error Distribution')
axes[1][0].set_xlabel('Error (£)')

# Plot 4: Model comparison
model_names = list(results.keys())
r2_values   = [results[m]['r2']
               for m in model_names]
colors      = [
    'gold' if m == best_model_name
    else 'steelblue'
    for m in model_names
]
axes[0][2].barh(
    model_names, r2_values, color=colors
)
axes[0][2].set_title('Model Comparison (R²)')
axes[0][2].set_xlabel('R² Score')
axes[0][2].axvline(
    x=0.7, color='red',
    linestyle='--', label='Target R²=0.7'
)
axes[0][2].legend()

# Plot 5: Feature importance
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(
        best_model.feature_importances_,
        index=final_feature_cols
    ).sort_values().tail(10)
    axes[1][1].barh(
        imp.index, imp.values,
        color='steelblue'
    )
    axes[1][1].set_title('Top 10 Feature Importance')
    axes[1][1].set_xlabel('Importance')

# Plot 6: Segment performance
segments   = ['Standard', 'Medium Value',
              'High Value', 'VIP']
seg_maes   = []
seg_counts = []
for seg in segments:
    seg_df = test_df[test_df['segment'] == seg]
    if len(seg_df) > 0:
        seg_maes.append(
            mean_absolute_error(
                seg_df['actual'],
                seg_df['predicted']
            )
        )
        seg_counts.append(len(seg_df))
    else:
        seg_maes.append(0)
        seg_counts.append(0)

axes[1][2].bar(
    segments, seg_maes,
    color=sns.color_palette('husl', 4)
)
axes[1][2].set_title('MAE by Customer Segment')
axes[1][2].set_xlabel('Segment')
axes[1][2].set_ylabel('MAE (£)')
axes[1][2].tick_params(
    axis='x', rotation=15
)

plt.tight_layout()
plt.savefig(
    'outputs/improved_model_results.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("\nPlot saved → "
      "outputs/improved_model_results.png ✅")


# ═══════════════════════════════════════════════════
# SAVE BEST MODEL
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("SAVING BEST MODEL")
print("="*55)

joblib.dump(best_model, 'models/rf_regressor.pkl')
joblib.dump(scaler,     'models/scaler.pkl')

with open('models/feature_cols.json', 'w') as f:
    json.dump(final_feature_cols, f, indent=2)

metrics = {
    'model_name'   : best_model_name,
    'mae'          : round(mae, 2),
    'rmse'         : round(rmse, 2),
    'r2'           : round(r2, 4),
    'mape'         : round(mape, 2),
    'n_features'   : len(final_feature_cols),
    'n_customers'  : len(df),
    'log_transform': True,
    'features_used': final_feature_cols
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Training stats for drift detection
training_stats = {}
for col in final_feature_cols:
    training_stats[col] = {
        'mean'  : float(X[col].mean()),
        'std'   : float(X[col].std()),
        'min'   : float(X[col].min()),
        'max'   : float(X[col].max()),
        'median': float(X[col].median())
    }

with open('models/training_stats.json', 'w') as f:
    json.dump(training_stats, f, indent=2)

print(f"Best model saved ✅")
print(f"Model: {best_model_name}")
print(f"R²   : {r2:.4f}")
print(f"MAE  : £{mae:,.0f}")

print("\n" + "="*55)
print("IMPROVED MODEL TRAINING COMPLETE ✅")
print("Next step: validate_metrics.py")
print("="*55)