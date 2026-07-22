"""
Step 2: Complete EDA
Online Retail II Dataset
Siddharth Shrivastava | Retail ML Project
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('eda_outputs', exist_ok=True)

# ── Load Data ─────────────────────────────────────
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
print(f"Loaded {len(df):,} rows ✅\n")


# ═══════════════════════════════════════════════════
# EDA STEP 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════

print("="*55)
print("EDA STEP 1 — DATA OVERVIEW")
print("="*55)

print(f"\nShape          : {df.shape}")
print(f"Date range     : {df['InvoiceDate'].min()} "
      f"to {df['InvoiceDate'].max()}")
print(f"Unique customers: "
      f"{df['Customer ID'].nunique():,}")
print(f"Unique products : "
      f"{df['StockCode'].nunique():,}")
print(f"Unique countries: "
      f"{df['Country'].nunique()}")
print(f"Unique invoices : "
      f"{df['Invoice'].nunique():,}")


# ═══════════════════════════════════════════════════
# EDA STEP 2 — MISSING VALUES
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("EDA STEP 2 — MISSING VALUES")
print("="*55)

null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(2)

for col in df.columns:
    if null_counts[col] > 0:
        print(f"\n  {col}:")
        print(f"    Count  : {null_counts[col]:,}")
        print(f"    Percent: {null_pct[col]}%")
        print(f"    Action : ", end="")
        if col == 'Customer ID':
            print("REMOVE rows — cannot predict "
                  "without customer ID")
        elif col == 'Description':
            print("KEEP — not needed for modeling")

# Plot missing values
plt.figure(figsize=(8, 4))
null_counts[null_counts > 0].plot(
    kind='bar',
    color='coral',
    edgecolor='black'
)
plt.title('Missing Values per Column')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_outputs/missing_values.png',
            dpi=150)
plt.close()
print("\nPlot saved → eda_outputs/missing_values.png")


# ═══════════════════════════════════════════════════
# EDA STEP 3 — QUANTITY ANALYSIS
# ═══════════════════════════════════════════════════

print("\n" + "="*55)
print("EDA STEP 3 — QUANTITY ANALYSIS")
print("="*55)

neg_qty = (df['Quantity'] < 0).sum()
zero_qty = (df['Quantity'] == 0).sum()
pos_qty = (df['Quantity'] > 0).sum()

print(f"\n  Positive quantity: {pos_qty:,} "
      f"({pos_qty/len(df)*100:.1f}%) ✅")
print(f"  Zero quantity    : {zero_qty:,} "
      f"({zero_qty/len(df)*100:.1f}%) ⚠️")
print(f"  Negative quantity: {neg_qty:,} "
      f"({neg_qty/len(df)*100:.1f}%) ❌")

# Check cancelled orders
cancelled = df[df['Invoice'].str.startswith('C',
              na=False)]
print(f"\n  Cancelled orders (Invoice starts C):")
print(f"  Count: {len(cancelled):,}")
print(f"  These have negative quantity")
print(f"  Action: REMOVE these rows")

# Quantity distribution (positive only)
pos_data = df[df['Quantity'] > 0]['Quantity']
print(f"\n  Positive Quantity Stats:")
print(f"  Mean   : {pos_data.mean():.1f}")
print(f"  Median : {pos_data.median():.0f}")
print(f"  Max    : {pos_data.max():,}")
print(f"  95th % : {pos_data.quantile(0.95):.0f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
pos_data.clip(upper=100).hist(
    bins=50, color='steelblue',
    edgecolor='white'
)
plt.title('Quantity Distribution (clipped at 100)')
plt.xlabel('Quantity')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.boxplot(pos_data.clip(upper=200))
plt.title('Quantity Boxplot')
plt.ylabel('Quantity')

plt.tight_layout()
plt.savefig('eda_outputs/quantity_analysis.png')