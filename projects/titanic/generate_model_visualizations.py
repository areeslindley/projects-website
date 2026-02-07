#!/usr/bin/env python3
"""
Script to generate model evaluation visualizations for the Titanic analysis.
This includes ROC curves, confusion matrices, and model comparison charts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import seaborn
try:
    import seaborn as sns
    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except:
    HAS_SEABORN = False

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
np.random.seed(42)

# Create images directory
os.makedirs('images', exist_ok=True)

# Load data and prepare it
print("Loading Titanic dataset...")
try:
    raw_data = pd.read_csv('data/titanic.csv')
    print(f"Loaded {len(raw_data)} rows")
except FileNotFoundError:
    print("Error: titanic.csv not found.")
    exit(1)

# Prepare data (same as in notebooks)
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
data['Age'] = data.groupby(['Pclass', 'Title'])['Age'].transform(
    lambda x: x.fillna(x.median())
).fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)
data['HasCabin'] = data['Cabin'].notna().astype(int)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100],
                         labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
data['FarePerPerson'] = data['Fare'] / data['FamilySize']
data['FareLog'] = np.log1p(data['Fare'])
data['Sex'] = (data['Sex'] == 'female').astype(int)
embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
title_dummies = pd.get_dummies(data['Title'], prefix='Title')
agegroup_dummies = pd.get_dummies(data['AgeGroup'], prefix='AgeGroup')
data = pd.concat([data, embarked_dummies, title_dummies, agegroup_dummies], axis=1)

feature_cols = [col for col in data.columns 
               if col not in ['PassengerId', 'Name', 'Ticket', 'Cabin', 
                             'Embarked', 'Title', 'AgeGroup', 'Survived']]

X = data[feature_cols]
y = data['Survived']

# Split data manually to avoid sklearn import issues
np.random.seed(42)
indices = np.random.permutation(len(X))
split_idx = int(0.8 * len(X))
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Simple scaling
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training models for visualization...")

# Train simple models for visualization
models = {}
models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
models['Logistic Regression'].fit(X_train_scaled, y_train)

models['Decision Tree'] = DecisionTreeClassifier(random_state=42, max_depth=5)
models['Decision Tree'].fit(X_train, y_train)

models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
models['Random Forest'].fit(X_train, y_train)

# 1. ROC Curves
print("Generating visualization 1: ROC Curves...")
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (name, model) in enumerate(models.items()):
    if name in ['Logistic Regression']:
        X_test_model = X_test_scaled
    else:
        X_test_model = X_test
    
    y_pred_proba = model.predict_proba(X_test_model)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})",
            linewidth=2, color=colors[i % len(colors)])

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves: Model Comparison', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/roc_curves.png")

# 2. Precision-Recall Curves
print("Generating visualization 2: Precision-Recall Curves...")
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (name, model) in enumerate(models.items()):
    if name in ['Logistic Regression']:
        X_test_model = X_test_scaled
    else:
        X_test_model = X_test
    
    y_pred_proba = model.predict_proba(X_test_model)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.3f})",
            linewidth=2, color=colors[i % len(colors)])

baseline_precision = y_test.mean()
plt.axhline(y=baseline_precision, color='k', linestyle='--', 
           label=f'Baseline (P = {baseline_precision:.3f})', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves: Model Comparison', fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower left", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/precision_recall_curves.png")

# 3. Confusion Matrices
print("Generating visualization 3: Confusion Matrices...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, model) in enumerate(models.items()):
    if name in ['Logistic Regression']:
        X_test_model = X_test_scaled
    else:
        X_test_model = X_test
    
    y_pred = model.predict(X_test_model)
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if HAS_SEABORN:
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=['Did Not Survive', 'Survived'],
                   yticklabels=['Did Not Survive', 'Survived'],
                   ax=axes[idx], cbar_kws={'label': 'Proportion'})
    else:
        im = axes[idx].imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['Did Not Survive', 'Survived'])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_yticklabels(['Did Not Survive', 'Survived'])
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, f'{cm_normalized[i, j]:.2f}', 
                              ha='center', va='center', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[idx], label='Proportion')
    
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)
    axes[idx].set_title(f'{name}', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('images/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/confusion_matrices.png")

# 4. Model Comparison Bar Chart
print("Generating visualization 4: Model Comparison...")
# Calculate metrics for each model
model_metrics = {}
for name, model in models.items():
    if name in ['Logistic Regression']:
        X_test_model = X_test_scaled
    else:
        X_test_model = X_test
    
    y_pred = model.predict(X_test_model)
    y_pred_proba = model.predict_proba(X_test_model)[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    model_metrics[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }

# Create comparison chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.15
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors_metrics = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, metric in enumerate(metrics):
    values = [model_metrics[name][metric] for name in models.keys()]
    ax.bar(x + i*width, values, width, label=metric, color=colors_metrics[i], alpha=0.8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison: Performance Metrics', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(list(models.keys()), fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('images/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/model_comparison.png")

# 5. Feature Importance (Random Forest)
print("Generating visualization 5: Feature Importance...")
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # Top 15

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(range(len(indices)), importances[indices], color='#1f77b4')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([X.columns[i] for i in indices], fontsize=10)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Random Forest: Top 15 Feature Importance', fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
    ax.text(imp + 0.001, i, f'{imp:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: images/feature_importance.png")

print("\n✅ All model visualizations generated successfully!")
print(f"Images saved in: {os.path.abspath('images')}")
