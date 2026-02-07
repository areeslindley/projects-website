#!/usr/bin/env python3
"""
Script to generate visualizations from the Titanic analysis notebooks
and save them as images to be included in the documentation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import seaborn, but continue without it if it fails
try:
    import seaborn as sns
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except:
    HAS_SEABORN = False
    print("Note: seaborn not available, using matplotlib only")

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
np.random.seed(42)

# Create images directory
os.makedirs('images', exist_ok=True)

# Load data
print("Loading Titanic dataset...")
try:
    raw_data = pd.read_csv('data/titanic.csv')
    print(f"Loaded {len(raw_data)} rows")
except FileNotFoundError:
    print("Error: titanic.csv not found. Please download it first.")
    exit(1)

# 1. Survival Rate by Sex
print("Generating visualization 1: Survival by Sex...")
fig, ax = plt.subplots(figsize=(8, 6))
sex_survival = raw_data.groupby('Sex')['Survived'].agg(['mean', 'count'])
bars = ax.bar(sex_survival.index, sex_survival['mean'], color=['#1f77b4', '#ff7f0e'])
ax.set_ylabel('Survival Rate', fontsize=12)
ax.set_xlabel('Sex', fontsize=12)
ax.set_title('Survival Rate by Sex', fontweight='bold', fontsize=14)
ax.set_ylim([0, 1])
for i, (idx, row) in enumerate(sex_survival.iterrows()):
    ax.text(i, row['mean'] + 0.02, f"{row['mean']:.2%}", 
            ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('images/survival_by_sex.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/survival_by_sex.png")

# 2. Survival Rate by Passenger Class
print("Generating visualization 2: Survival by Passenger Class...")
fig, ax = plt.subplots(figsize=(8, 6))
pclass_survival = raw_data.groupby('Pclass')['Survived'].agg(['mean', 'count'])
bars = ax.bar(pclass_survival.index, pclass_survival['mean'], 
              color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_xlabel('Passenger Class', fontsize=12)
ax.set_ylabel('Survival Rate', fontsize=12)
ax.set_title('Survival Rate by Passenger Class', fontweight='bold', fontsize=14)
ax.set_xticks([1, 2, 3])
ax.set_ylim([0, 1])
for idx, row in pclass_survival.iterrows():
    ax.text(idx-1, row['mean'] + 0.02, f"{row['mean']:.2%}", 
            ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('images/survival_by_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/survival_by_class.png")

# 3. Age Distribution by Survival
print("Generating visualization 3: Age Distribution by Survival...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
survived_ages = raw_data[raw_data['Survived'] == 1]['Age'].dropna()
not_survived_ages = raw_data[raw_data['Survived'] == 0]['Age'].dropna()

axes[0].hist(not_survived_ages, bins=30, alpha=0.6, label='Did Not Survive', 
             color='#d62728', edgecolor='black')
axes[0].hist(survived_ages, bins=30, alpha=0.6, label='Survived', 
             color='#2ca02c', edgecolor='black')
axes[0].set_xlabel('Age (years)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Age Distribution by Survival', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=11)

survival_data = [not_survived_ages, survived_ages]
axes[1].boxplot(survival_data, labels=['Did Not Survive', 'Survived'])
axes[1].set_ylabel('Age (years)', fontsize=12)
axes[1].set_title('Age Distribution by Survival (Box Plot)', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('images/age_by_survival.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/age_by_survival.png")

# 4. Missing Data Visualization
print("Generating visualization 4: Missing Data Patterns...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calculate missing data
missing_counts = raw_data.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
missing_pct = (missing_counts / len(raw_data)) * 100

# Left plot: Missing data counts (bar chart)
bars = axes[0].bar(range(len(missing_counts)), missing_counts.values, 
                   color=['#d62728' if pct > 50 else '#ff7f0e' if pct > 20 else '#2ca02c' 
                          for pct in missing_pct])
axes[0].set_xticks(range(len(missing_counts)))
axes[0].set_xticklabels(missing_counts.index, rotation=45, ha='right', fontsize=11)
axes[0].set_title('Missing Data Counts', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, count) in enumerate(missing_counts.items()):
    axes[0].text(i, count + 10, f'{int(count)}\n({missing_pct[idx]:.1f}%)', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Right plot: Missing data percentage (horizontal bar)
axes[1].barh(range(len(missing_pct)), missing_pct.values, 
            color=['#d62728' if pct > 50 else '#ff7f0e' if pct > 20 else '#2ca02c' 
                   for pct in missing_pct.values])
axes[1].set_yticks(range(len(missing_pct)))
axes[1].set_yticklabels(missing_pct.index, fontsize=11)
axes[1].set_xlabel('Missing Percentage (%)', fontsize=12)
axes[1].set_title('Missing Data Percentage', fontsize=14, fontweight='bold')
axes[1].set_xlim([0, max(missing_pct.values) * 1.1])
axes[1].grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (idx, pct) in enumerate(missing_pct.items()):
    axes[1].text(pct + 1, i, f'{pct:.1f}%', 
                ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('images/missing_data.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/missing_data.png")

# 5. Correlation Heatmap
print("Generating visualization 5: Correlation Heatmap...")
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_matrix = raw_data[numeric_cols].corr()
plt.figure(figsize=(10, 8))
if HAS_SEABORN:
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
else:
    # Fallback using matplotlib
    im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, shrink=0.8)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=10)
plt.title('Correlation Matrix of Numeric Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/correlation_heatmap.png")

# 6. Survival by Title (after feature engineering)
print("Generating visualization 6: Survival by Title...")
raw_data['Title'] = raw_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
    'Capt': 'Rare', 'Sir': 'Rare'
}
raw_data['Title'] = raw_data['Title'].map(title_mapping).fillna('Rare')
title_survival = raw_data.groupby('Title')['Survived'].agg(['mean', 'count']).sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(title_survival.index, title_survival['mean'], color='#1f77b4')
ax.set_ylabel('Survival Rate', fontsize=12)
ax.set_xlabel('Title', fontsize=12)
ax.set_title('Survival Rate by Title', fontweight='bold', fontsize=14)
ax.set_ylim([0, 1])
for i, (idx, row) in enumerate(title_survival.iterrows()):
    ax.text(i, row['mean'] + 0.02, f"{row['mean']:.2%}\n(n={int(row['count'])})", 
            ha='center', fontsize=9)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('images/survival_by_title.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/survival_by_title.png")

# 7. Fare Distribution
print("Generating visualization 7: Fare Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(raw_data['Fare'], bins=50, edgecolor='black', alpha=0.7, color='#1f77b4')
axes[0].set_xlabel('Fare', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Fare Distribution', fontweight='bold', fontsize=14)

axes[1].hist(np.log1p(raw_data['Fare']), bins=50, edgecolor='black', alpha=0.7, color='#2ca02c')
axes[1].set_xlabel('Log(Fare + 1)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Fare Distribution (Log Scale)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('images/fare_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/fare_distribution.png")

print("\n✅ All visualizations generated successfully!")
print(f"Images saved in: {os.path.abspath('images')}")
