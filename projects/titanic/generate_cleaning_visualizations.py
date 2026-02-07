#!/usr/bin/env python3
"""
Script to generate data cleaning visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
np.random.seed(42)

os.makedirs('images', exist_ok=True)

print("Loading Titanic dataset...")
raw_data = pd.read_csv('data/titanic.csv')

# Age distribution before/after imputation
print("Generating visualization: Age Distribution Before/After Imputation...")
data = raw_data.copy()
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
    'Capt': 'Rare', 'Sir': 'Rare'
}
data['Title'] = data['Title'].map(title_mapping).fillna('Rare')

# Before imputation
raw_age = raw_data['Age'].dropna()

# After imputation
data['Age'] = data.groupby(['Pclass', 'Title'])['Age'].transform(
    lambda x: x.fillna(x.median())
).fillna(data['Age'].median())
imputed_age = data['Age']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(raw_age, bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')
axes[0].set_xlabel('Age (years)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Age Distribution (Before Imputation)', fontweight='bold', fontsize=14)
axes[0].axvline(raw_age.median(), color='red', linestyle='--', 
               label=f'Median: {raw_age.median():.1f}', linewidth=2)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

axes[1].hist(imputed_age, bins=30, edgecolor='black', alpha=0.7, color='#2ca02c')
axes[1].set_xlabel('Age (years)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Age Distribution (After Imputation)', fontweight='bold', fontsize=14)
axes[1].axvline(imputed_age.median(), color='red', linestyle='--', 
               label=f'Median: {imputed_age.median():.1f}', linewidth=2)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/age_imputation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/age_imputation_comparison.png")

# Outlier detection visualization
print("Generating visualization: Fare Outlier Detection...")
fig, ax = plt.subplots(figsize=(10, 6))
fare_data = data['Fare']
bp = ax.boxplot(fare_data, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('#1f77b4')
bp['boxes'][0].set_alpha(0.7)

# Add IQR lines
Q1 = fare_data.quantile(0.25)
Q3 = fare_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

ax.axhline(y=lower_bound, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Lower bound: {lower_bound:.2f}')
ax.axhline(y=upper_bound, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Upper bound: {upper_bound:.2f}')

ax.set_ylabel('Fare', fontsize=12)
ax.set_title('Fare Distribution with Outliers (IQR Method)', fontweight='bold', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('images/fare_outliers.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/fare_outliers.png")

print("\n✅ All cleaning visualizations generated successfully!")
