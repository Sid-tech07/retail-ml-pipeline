"""
Step 3: Feature Engineering
Create customer level features
for regression model
Target: Total spend per customer
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── Load Clean Data ───────────────────────────────
print("Loading clean data...")
df = pd.read_csv(
    'data/clean_transactions.csv',
    parse_dates=['InvoiceDate']
)
print(f"Loaded {len(df):,} rows ✅")
print(f"Unique customers: "
      f"{df['Customer ID'].nunique():,}\n")

# Reference date for recency calculation
reference_date = df['InvoiceDate'].max()
print(f"Reference date: {reference_date}\n")


# ═══════════════════════════════════════════════════
# FEATURE GROUP 1 — Transaction Behavior
# ═══════════════════════════════════════════════════

print("="*55)
print("FEATURE GROUP 1 — Transaction Behavior")
print("="*55)

txn_features = df.groupby('Customer ID').agg(

    # TARGET VARIABLE
    total_spend = ('amount', 'sum'),

    # Transaction count features
    num_transactions = ('Invoice', 'count'),
    num_unique_invoices = ('Invoice', 'nunique'),

    # Amount features
    avg_transaction_amount = ('amount', 'mean'),
    max_transaction_amount = ('amount', 'max'),
    min_transaction_amount = ('amount', 'min'),
    std_transaction_amount = ('amount', 'std'),
    median_transaction     = ('amount', 'median'),

    # Quantity features
    total_quantity   = ('Quantity', 'sum'),
    avg_quantity     = ('Quantity', 'mean'),
    max_quantity     = ('Quantity', 'max'),

    # Price features
    avg_price        = ('Price', 'mean'),
    max_price        = ('Price', 'max'),

    # Date features
    first_purchase   = ('InvoiceDate', 'min'),
    last_purchase    = ('InvoiceDate', 'max'),

    # Time features
    avg_month        = ('month', 'mean'),
    avg_day_of_week  = ('day_of_week', 'mean'),
    avg_hour         = ('hour', 'mean'),
    weekend_purchases= ('is_weekend', 'sum'),

).reset_index()

print(f"Transaction features created ✅")
print(f"Shape: {txn_features.shape}")


# ═══════════════════════════════════════════════════
# FEATURE GROUP 2 — Product Behavior
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("FEATURE GROUP 2 — Product Behavior")
print("="*55)

product_features = df.groupby('Customer ID').agg(
    num_unique_products = ('StockCode', 'nunique'),
    num_unique_descriptions = (
        'Description', 'nunique'
    ),
).reset_index()

print(f"Product features created ✅")


# ═══════════════════════════════════════════════════
# FEATURE GROUP 3 — Time Based Features
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("FEATURE GROUP 3 — Time Based Features")
print("="*55)

# Merge to get date columns
customer_df = txn_features.merge(
    product_features, on='Customer ID'
)

# Days since last purchase (Recency)
customer_df['days_since_last_purchase'] = (
    reference_date -
    pd.to_datetime(customer_df['last_purchase'])
).dt.days

# Customer lifetime in days
customer_df['customer_lifetime_days'] = (
    pd.to_datetime(customer_df['last_purchase']) -
    pd.to_datetime(customer_df['first_purchase'])
).dt.days

# Purchase frequency (avg days between purchases)
customer_df['purchase_frequency_days'] = np.where(
    customer_df['num_unique_invoices'] > 1,
    customer_df['customer_lifetime_days'] /
    customer_df['num_unique_invoices'],
    customer_df['customer_lifetime_days']
)

# Weekend shopping ratio
customer_df['weekend_ratio'] = (
    customer_df['weekend_purchases'] /
    customer_df['num_transactions']
)

# Average items per invoice
customer_df['avg_items_per_invoice'] = (
    customer_df['total_quantity'] /
    customer_df['num_unique_invoices']
)

# Amount per quantity (avg unit price paid)
customer_df['amount_per_unit'] = (
    customer_df['total_spend'] /
    customer_df['total_quantity']
)

# High value transaction ratio
# (transactions above median amount)
median_amount = df['amount'].median()
high_value_txns = df[
    df['amount'] > median_amount
].groupby('Customer ID')['Invoice'].count()

customer_df['high_value_txn_count'] = (
    customer_df['Customer ID']
    .map(high_value_txns)
    .fillna(0)
)
customer_df['high_value_ratio'] = (
    customer_df['high_value_txn_count'] /
    customer_df['num_transactions']
)

print(f"Time based features created ✅")


# ═══════════════════════════════════════════════════
# FEATURE GROUP 4 — RFM Features
# (Recency, Frequency, Monetary)
# Most important in retail ML!
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("FEATURE GROUP 4 — RFM Features")
print("="*55)

# Recency — already calculated
# = days_since_last_purchase

# Frequency
customer_df['frequency'] = (
    customer_df['num_unique_invoices']
)

# Monetary
customer_df['monetary'] = (
    customer_df['total_spend']
)

# RFM Scores (1-5 scale)
# Higher is better for F and M
# Lower recency days = better

def rfm_score(series, reverse=False):
    """Score 1-5 based on quartiles"""
    labels = [1, 2, 3, 4, 5]
    if reverse:
        labels = [5, 4, 3, 2, 1]
    try:
        return pd.qcut(
            series,
            q=5,
            labels=labels,
            duplicates='drop'
        ).astype(int)
    except:
        return pd.cut(
            series,
            bins=5,
            labels=labels,
            duplicates='drop'
        ).astype(int)

customer_df['r_score'] = rfm_score(
    customer_df['days_since_last_purchase'],
    reverse=True  # lower days = higher score
)
customer_df['f_score'] = rfm_score(
    customer_df['frequency']
)
customer_df['m_score'] = rfm_score(
    customer_df['monetary']
)

# Combined RFM score
customer_df['rfm_score'] = (
    customer_df['r_score'] +
    customer_df['f_score'] +
    customer_df['m_score']
)

print(f"RFM features created ✅")
print(f"\nRFM Score Distribution:")
print(customer_df['rfm_score'].describe().round(2))


# ═══════════════════════════════════════════════════
# FEATURE GROUP 5 — Monthly Behavior
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("FEATURE GROUP 5 — Monthly Behavior")
print("="*55)

# Peak shopping month
peak_month = df.groupby(
    ['Customer ID', 'month']
)['amount'].sum().reset_index()
peak_month = peak_month.loc[
    peak_month.groupby('Customer ID')
    ['amount'].idxmax()
][['Customer ID', 'month']].rename(
    columns={'month': 'peak_shopping_month'}
)

customer_df = customer_df.merge(
    peak_month, on='Customer ID', how='left'
)

# Q4 (holiday season) purchases
q4_purchases = df[df['quarter'] == 4].groupby(
    'Customer ID'
)['amount'].sum().reset_index().rename(
    columns={'amount': 'q4_spend'}
)

customer_df = customer_df.merge(
    q4_purchases, on='Customer ID', how='left'
)
customer_df['q4_spend'] = (
    customer_df['q4_spend'].fillna(0)
)

# Q4 ratio
customer_df['q4_spend_ratio'] = np.where(
    customer_df['total_spend'] > 0,
    customer_df['q4_spend'] /
    customer_df['total_spend'],
    0
)

print(f"Monthly behavior features created ✅")


# ═══════════════════════════════════════════════════
# CLEAN UP FEATURES
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("CLEANING UP FEATURES")
print("="*55)

# Fill NaN in std (customers with 1 transaction)
customer_df['std_transaction_amount'] = (
    customer_df['std_transaction_amount'].fillna(0)
)

# Drop date columns (already extracted)
customer_df = customer_df.drop(
    columns=['first_purchase', 'last_purchase']
)

# Final feature list
feature_cols = [
    # Transaction behavior
    'num_transactions',
    'num_unique_invoices',
    'avg_transaction_amount',
    'max_transaction_amount',
    'min_transaction_amount',
    'std_transaction_amount',
    'median_transaction',

    # Quantity features
    'total_quantity',
    'avg_quantity',
    'max_quantity',

    # Price features
    'avg_price',
    'max_price',

    # Time features
    'days_since_last_purchase',
    'customer_lifetime_days',
    'purchase_frequency_days',
    'weekend_ratio',
    'avg_month',
    'avg_day_of_week',
    'avg_hour',

    # Product features
    'num_unique_products',
    'num_unique_descriptions',

    # Derived features
    'avg_items_per_invoice',
    'amount_per_unit',
    'high_value_ratio',

    # RFM scores
    'r_score',
    'f_score',
    'm_score',
    'rfm_score',

    # Monthly behavior
    'peak_shopping_month',
    'q4_spend_ratio',
]

print(f"Total features: {len(feature_cols)}")
print(f"\nAll features:")
for i, f in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {f}")


# ═══════════════════════════════════════════════════
# VALIDATE FEATURES
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("VALIDATING FEATURES")
print("="*55)

# Check nulls
null_check = customer_df[
    feature_cols + ['total_spend']
].isnull().sum()
print(f"\nNull values in features:")
if null_check.sum() == 0:
    print("  Zero nulls ✅ Perfect!")
else:
    print(null_check[null_check > 0])

# Check target
print(f"\nTarget (total_spend) stats:")
print(f"  Customers: {len(customer_df):,}")
print(f"  Min      : £{customer_df['total_spend'].min():,.2f}")
print(f"  Max      : £{customer_df['total_spend'].max():,.2f}")
print(f"  Mean     : £{customer_df['total_spend'].mean():,.2f}")
print(f"  Median   : £{customer_df['total_spend'].median():,.2f}")
print(f"  Skewness : {customer_df['total_spend'].skew():.2f}")

# Check if log transform needed
if customer_df['total_spend'].skew() > 1:
    print(f"\n  ⚠️ RIGHT SKEWED!")
    print(f"  Action: Apply log1p in training ✅")


# ═══════════════════════════════════════════════════
# VISUALIZE TOP FEATURES
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("VISUALIZING KEY FEATURES")
print("="*55)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    'Customer Feature Distributions',
    fontsize=14, fontweight='bold'
)

# Plot 1: Target distribution
customer_df['total_spend'].clip(
    upper=customer_df['total_spend'].quantile(0.95)
).hist(bins=40, ax=axes[0][0], color='coral')
axes[0][0].set_title(
    f"Total Spend (Target)\n"
    f"Skew: {customer_df['total_spend'].skew():.2f}"
)
axes[0][0].set_xlabel('£')

# Plot 2: Log transformed target
np.log1p(customer_df['total_spend']).hist(
    bins=40, ax=axes[0][1], color='steelblue'
)
axes[0][1].set_title('Log(Total Spend)\nMore Normal')
axes[0][1].set_xlabel('Log(£)')

# Plot 3: Number of transactions
customer_df['num_transactions'].clip(
    upper=customer_df['num_transactions'].quantile(0.95)
).hist(bins=40, ax=axes[0][2], color='green')
axes[0][2].set_title('Number of Transactions')
axes[0][2].set_xlabel('Count')

# Plot 4: Days since last purchase
customer_df['days_since_last_purchase'].hist(
    bins=40, ax=axes[1][0], color='purple'
)
axes[1][0].set_title('Days Since Last Purchase\n(Recency)')
axes[1][0].set_xlabel('Days')

# Plot 5: RFM Score distribution
customer_df['rfm_score'].hist(
    bins=12, ax=axes[1][1], color='orange'
)
axes[1][1].set_title('RFM Score Distribution')
axes[1][1].set_xlabel('RFM Score (3-15)')

# Plot 6: Avg transaction amount
customer_df['avg_transaction_amount'].clip(
    upper=customer_df['avg_transaction_amount']
    .quantile(0.95)
).hist(bins=40, ax=axes[1][2], color='brown')
axes[1][2].set_title('Avg Transaction Amount')
axes[1][2].set_xlabel('£')

plt.tight_layout()
plt.savefig(
    'outputs/feature_distributions.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Plot saved → outputs/feature_distributions.png")


# ═══════════════════════════════════════════════════
# CORRELATION WITH TARGET
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("CORRELATION WITH TARGET")
print("="*55)

corr_with_target = customer_df[
    feature_cols + ['total_spend']
].corr()['total_spend'].drop('total_spend')
corr_sorted = corr_with_target.abs().sort_values(
    ascending=False
)

print("\nTop 15 features by correlation with target:")
for feat, corr in corr_sorted.head(15).items():
    direction = "+" if corr_with_target[feat] > 0 else "-"
    bar = "█" * int(abs(corr_with_target[feat]) * 20)
    print(f"  {feat:35s}: "
          f"{direction}{abs(corr_with_target[feat]):.3f} "
          f"{bar}")

print("\nBottom 5 features (lowest correlation):")
for feat, corr in corr_sorted.tail(5).items():
    print(f"  {feat:35s}: {corr:.3f}")


# ═══════════════════════════════════════════════════
# SAVE CUSTOMER FEATURES
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("SAVING CUSTOMER FEATURES")
print("="*55)

# Save complete customer dataframe
customer_df.to_csv(
    'data/customer_features.csv',
    index=False
)
print(f"Saved: data/customer_features.csv ✅")
print(f"Shape: {customer_df.shape}")
print(f"Customers: {len(customer_df):,}")
print(f"Features : {len(feature_cols)}")

# Save feature list
import json
with open('data/feature_cols.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)
print(f"Feature list saved: data/feature_cols.json ✅")

print("\n" + "="*55)
print("FEATURE ENGINEERING COMPLETE ✅")
print("Next step: Model Training")
print("="*55)