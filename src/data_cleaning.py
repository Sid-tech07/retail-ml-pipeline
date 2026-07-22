"""
Step 2: Data Cleaning
Online Retail II Dataset
Siddharth Shrivastava | Retail ML Project
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── Load Raw Data ─────────────────────────────────
print("Loading data...")
df1 = pd.read_excel(
    'online_retail_II.xlsx',
    sheet_name='Year 2009-2010'
)
df2 = pd.read_excel(
    'online_retail_II.xlsx',
    sheet_name='Year 2010-2011'
)
df = pd.concat([df1, df2], ignore_index=True)
original_count = len(df)
print(f"Loaded {original_count:,} rows ✅\n")


# ═══════════════════════════════════════════════════
# CLEANING STAGE 1 — Remove Null Customer IDs
# ═══════════════════════════════════════════════════

print("="*55)
print("STAGE 1 — Removing Null Customer IDs")
print("="*55)

before = len(df)
df = df[df['Customer ID'].notna()]
after  = len(df)
removed = before - after

print(f"Before : {before:,}")
print(f"Removed: {removed:,}")
print(f"After  : {after:,} ✅")

# Convert Customer ID to integer
df['Customer ID'] = df['Customer ID'].astype(int)
print(f"Customer ID converted to int ✅")


# ═══════════════════════════════════════════════════
# CLEANING STAGE 2 — Remove Cancelled Orders
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("STAGE 2 — Removing Cancelled Orders")
print("="*55)

before = len(df)

# Cancelled orders start with 'C'
cancelled_mask = df['Invoice'].astype(
    str
).str.startswith('C')
cancelled_count = cancelled_mask.sum()

df = df[~cancelled_mask]
after   = len(df)
removed = before - after

print(f"Before         : {before:,}")
print(f"Cancelled found: {cancelled_count:,}")
print(f"Removed        : {removed:,}")
print(f"After          : {after:,} ✅")


# ═══════════════════════════════════════════════════
# CLEANING STAGE 3 — Remove Invalid Quantities
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("STAGE 3 — Removing Invalid Quantities")
print("="*55)

before = len(df)

neg_qty  = (df['Quantity'] <= 0).sum()
df       = df[df['Quantity'] > 0]

after   = len(df)
removed = before - after

print(f"Before          : {before:,}")
print(f"Negative/zero   : {neg_qty:,}")
print(f"Removed         : {removed:,}")
print(f"After           : {after:,} ✅")


# ═══════════════════════════════════════════════════
# CLEANING STAGE 4 — Remove Invalid Prices
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("STAGE 4 — Removing Invalid Prices")
print("="*55)

before = len(df)

neg_price  = (df['Price'] <= 0).sum()
df         = df[df['Price'] > 0]

after   = len(df)
removed = before - after

print(f"Before         : {before:,}")
print(f"Zero/negative  : {neg_price:,}")
print(f"Removed        : {removed:,}")
print(f"After          : {after:,} ✅")


# ═══════════════════════════════════════════════════
# CLEANING STAGE 5 — Calculate Amount
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("STAGE 5 — Calculate Transaction Amount")
print("="*55)

# Amount = Quantity × Price
df['amount'] = df['Quantity'] * df['Price']

print(f"Amount = Quantity × Price ✅")
print(f"Amount stats:")
print(f"  Min    : £{df['amount'].min():,.2f}")
print(f"  Max    : £{df['amount'].max():,.2f}")
print(f"  Mean   : £{df['amount'].mean():,.2f}")
print(f"  Median : £{df['amount'].median():,.2f}")

# Remove extreme outliers in amount
# Using 99.9th percentile
upper = df['amount'].quantile(0.999)
before = len(df)
df     = df[df['amount'] <= upper]
after  = len(df)

print(f"\nOutlier removal (99.9th percentile):")
print(f"  Threshold: £{upper:,.2f}")
print(f"  Removed  : {before-after:,} rows")
print(f"  After    : {after:,} ✅")


# ═══════════════════════════════════════════════════
# CLEANING STAGE 6 — Extract Time Features
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("STAGE 6 — Extract Time Features")
print("="*55)

df['year']        = df['InvoiceDate'].dt.year
df['month']       = df['InvoiceDate'].dt.month
df['day_of_week'] = df['InvoiceDate'].dt.dayofweek
df['hour']        = df['InvoiceDate'].dt.hour
df['is_weekend']  = df['day_of_week'].isin(
    [5, 6]
).astype(int)
df['quarter']     = df['InvoiceDate'].dt.quarter

print(f"Time features extracted:")
print(f"  year, month, day_of_week,")
print(f"  hour, is_weekend, quarter ✅")


# ═══════════════════════════════════════════════════
# CLEANING STAGE 7 — Filter UK Only
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("STAGE 7 — Filter UK Customers")
print("="*55)

before = len(df)
uk_df  = df[df['Country'] == 'United Kingdom']
after  = len(uk_df)

uk_pct = after / before * 100
print(f"Before   : {before:,}")
print(f"UK rows  : {after:,} ({uk_pct:.1f}%)")
print(f"Non-UK   : {before-after:,} removed")
print(f"Reason   : UK = consistent behavior ✅")

# Save both versions
df_clean = uk_df.copy()


# ═══════════════════════════════════════════════════
# CLEANING STAGE 8 — Final Validation
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("STAGE 8 — Final Validation")
print("="*55)

print(f"\nNull check after cleaning:")
print(df_clean.isnull().sum())

print(f"\nNegative values check:")
print(f"  Quantity : {(df_clean['Quantity'] < 0).sum()}")
print(f"  Price    : {(df_clean['Price'] < 0).sum()}")
print(f"  Amount   : {(df_clean['amount'] < 0).sum()}")

print(f"\nData types:")
print(df_clean.dtypes)

print(f"\nFinal clean data shape: {df_clean.shape}")
print(f"Unique customers: "
      f"{df_clean['Customer ID'].nunique():,}")


# ═══════════════════════════════════════════════════
# SAVE CLEAN DATA
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("SAVING CLEAN DATA")
print("="*55)

# Save transaction level clean data
df_clean.to_csv(
    'data/clean_transactions.csv',
    index=False
)
print(f"Saved: data/clean_transactions.csv ✅")
print(f"Rows : {len(df_clean):,}")


# ═══════════════════════════════════════════════════
# CLEANING SUMMARY REPORT
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("CLEANING SUMMARY REPORT")
print("="*55)

cleaning_report = {
    'original_rows'    : original_count,
    'final_rows'       : len(df_clean),
    'removed_total'    : original_count - len(df_clean),
    'removal_pct'      : round(
        (original_count - len(df_clean)) /
        original_count * 100, 1
    ),
    'unique_customers' : df_clean['Customer ID'].nunique(),
    'date_range_start' : str(df_clean['InvoiceDate'].min()),
    'date_range_end'   : str(df_clean['InvoiceDate'].max()),
    'total_revenue'    : round(
        df_clean['amount'].sum(), 2
    )
}

print(f"\n{'Stage':<35} {'Rows':>10}")
print("-"*50)
print(f"{'Original data':<35} "
      f"{original_count:>10,}")
print(f"{'After removing null CustomerID':<35} "
      f"{original_count-243007:>10,}")
print(f"{'After removing cancelled orders':<35} "
      f"{original_count-243007-19494:>10,}")
print(f"{'After removing negative quantity':<35} "
      f"{original_count-243007-19494-22950:>10,}")
print(f"{'After removing invalid prices':<35} "
      f"{len(df):>10,}")
print(f"{'After UK filter':<35} "
      f"{len(df_clean):>10,}")
print("-"*50)
print(f"\nFinal clean rows    : {len(df_clean):,}")
print(f"Total removed       : "
      f"{original_count - len(df_clean):,} "
      f"({cleaning_report['removal_pct']}%)")
print(f"Unique customers    : "
      f"{cleaning_report['unique_customers']:,}")
print(f"Total revenue       : "
      f"£{cleaning_report['total_revenue']:,.2f}")

# Plot cleaning funnel
stages = [
    'Original',
    'No Null\nCustomer ID',
    'No Cancelled',
    'No Neg\nQuantity',
    'No Bad\nPrice',
    'UK Only'
]

counts = [
    original_count,
    original_count - 243007,
    original_count - 243007 - 19494,
    original_count - 243007 - 19494 - 22950,
    len(df),
    len(df_clean)
]

plt.figure(figsize=(12, 5))
bars = plt.bar(stages, counts,
               color=sns.color_palette(
                   'Blues_d', len(stages)
               ))
plt.title('Data Cleaning Funnel\n'
          'Rows Remaining After Each Stage')
plt.ylabel('Number of Rows')
plt.xticks(rotation=0)

# Add count labels on bars
for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 5000,
        f'{count:,}',
        ha='center', va='bottom',
        fontsize=9, fontweight='bold'
    )

plt.tight_layout()
plt.savefig('outputs/cleaning_funnel.png',
            dpi=150)
plt.close()
print("\nPlot saved → outputs/cleaning_funnel.png")

print("\n" + "="*55)
print("DATA CLEANING COMPLETE ✅")
print("Next step: Feature Engineering")
print("="*55)