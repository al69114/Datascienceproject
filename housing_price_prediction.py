"""
Utrecht Housing Price Prediction - Intro to Machine Learning Workshop

Compares 5 ML algorithms on the Utrecht housing dataset:
  1. Decision Tree
  2. Naive Bayes
  3. KNN (K-Nearest Neighbors)
  4. Linear Regression
  5. Random Forest

Part A — Regression  (Linear, Decision Tree, KNN, Random Forest)
Part B — Classification  (all 5 including Naive Bayes)
Part C — Grand comparison dashboard

All plots are saved to output_plots/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150

os.makedirs('output_plots', exist_ok=True)

# ============================================================================
# 1. LOAD & EXPLORE
# ============================================================================
print("=" * 80)
print("UTRECHT HOUSING — INTRO TO MACHINE LEARNING WORKSHOP")
print("Comparing: Decision Tree | Naive Bayes | KNN | Linear | Random Forest")
print("=" * 80)

df = pd.read_csv('utrechthousingsmall.csv', encoding='utf-8-sig')
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.describe().round(2))

# ============================================================================
# 2. PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("PREPROCESSING")
print("=" * 80)

df_proc = df.drop('id', axis=1)
df_proc['house_age'] = df_proc['buildyear'].max() - df_proc['buildyear'] + 1
df_proc['lot_utilization'] = df_proc['house-area'] / df_proc['lot-area']

# --- Targets ---
y_reg = df_proc['retailvalue']

# Classification target: bin price into 3 categories
price_labels = ['Low', 'Medium', 'High']
df_proc['price_category'] = pd.qcut(y_reg, q=3, labels=price_labels)
y_cls = LabelEncoder().fit_transform(df_proc['price_category'])

feature_cols = [c for c in df_proc.columns
                if c not in ['retailvalue', 'price_category', 'select']]
X = df_proc[feature_cols]

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Regression target: retailvalue (continuous)")
print(f"Classification target: price_category — {dict(zip(price_labels, np.bincount(y_cls)))}")

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
_, _, y_cls_train, y_cls_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ============================================================================
# PART A — REGRESSION (4 models — Naive Bayes cannot do regression)
# ============================================================================
print("\n" + "=" * 80)
print("PART A: REGRESSION  (predicting exact housing price)")
print("Note: Naive Bayes is classification-only, so 4 models are compared here.")
print("=" * 80)

reg_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree':     DecisionTreeRegressor(random_state=42),
    'KNN':               KNeighborsRegressor(n_neighbors=5),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
}

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train_sc, y_reg_train)
    y_pred = model.predict(X_test_sc)
    cv = cross_val_score(model, X_train_sc, y_reg_train, cv=5, scoring='r2')

    r2   = r2_score(y_reg_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    mae  = mean_absolute_error(y_reg_test, y_pred)

    reg_results[name] = dict(r2=r2, rmse=rmse, mae=mae,
                             cv_mean=cv.mean(), cv_std=cv.std(), y_pred=y_pred)
    print(f"\n  {name}")
    print(f"    R²   = {r2:.4f}   RMSE = {rmse:,.0f}   MAE = {mae:,.0f}")
    print(f"    CV R² = {cv.mean():.4f} +/- {cv.std():.4f}")

# ---- Plot 1: Actual vs Predicted scatter for each regression model ----
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
colors_reg = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
for ax, (name, res), col in zip(axes, reg_results.items(), colors_reg):
    ax.scatter(y_reg_test, res['y_pred'], alpha=0.7, color=col,
               edgecolors='k', linewidths=0.3, s=60)
    lims = [min(y_reg_test.min(), res['y_pred'].min()),
            max(y_reg_test.max(), res['y_pred'].max())]
    ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title(f"{name}\nR² = {res['r2']:.4f}  |  RMSE = {res['rmse']:,.0f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
plt.suptitle('Regression: Actual vs Predicted', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output_plots/01_regression_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: output_plots/01_regression_actual_vs_predicted.png")

# ---- Plot 2: R² and RMSE bar comparison ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
rnames = list(reg_results.keys())
r2s   = [reg_results[n]['r2'] for n in rnames]
rmses = [reg_results[n]['rmse'] for n in rnames]

bars1 = axes[0].bar(rnames, r2s, color=colors_reg, edgecolor='black', alpha=0.85)
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Score  (higher = better)')
axes[0].set_ylim(0, max(r2s) * 1.25)
for bar, val in zip(bars1, r2s):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)
axes[0].grid(axis='y', alpha=0.3)

bars2 = axes[1].bar(rnames, rmses, color=colors_reg, edgecolor='black', alpha=0.85)
axes[1].set_ylabel('RMSE')
axes[1].set_title('RMSE  (lower = better)')
for bar, val in zip(bars2, rmses):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'{val:,.0f}', ha='center', fontweight='bold', fontsize=9)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Regression Model Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output_plots/02_regression_r2_rmse.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_plots/02_regression_r2_rmse.png")

# ---- Plot 3: Cross-validation boxplot (regression) ----
fig, ax = plt.subplots(figsize=(10, 6))
cv_data = []
for name, model in reg_models.items():
    cv_data.append(cross_val_score(model, X_train_sc, y_reg_train, cv=10, scoring='r2'))
bp = ax.boxplot(cv_data, labels=rnames, patch_artist=True)
for patch, col in zip(bp['boxes'], colors_reg):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
ax.set_ylabel('R² Score')
ax.set_title('Regression: 10-Fold Cross-Validation R²')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('output_plots/03_regression_cv_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_plots/03_regression_cv_boxplot.png")

# ============================================================================
# PART B — CLASSIFICATION (all 5 models head-to-head)
# ============================================================================
print("\n" + "=" * 80)
print("PART B: CLASSIFICATION  (predicting price category: Low / Medium / High)")
print("All 5 algorithms compared directly.")
print("=" * 80)

cls_models = {
    'Decision Tree':     DecisionTreeClassifier(random_state=42),
    'Naive Bayes':       GaussianNB(),
    'KNN':               KNeighborsClassifier(n_neighbors=5),
    'Linear (Logistic)': __import__('sklearn.linear_model', fromlist=['LogisticRegression']).LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42),
}

cls_results = {}
for name, model in cls_models.items():
    model.fit(X_train_sc, y_cls_train)
    y_pred = model.predict(X_test_sc)
    cv = cross_val_score(model, X_train_sc, y_cls_train, cv=5, scoring='accuracy')

    acc = accuracy_score(y_cls_test, y_pred)
    f1  = f1_score(y_cls_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_cls_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_cls_test, y_pred, average='weighted', zero_division=0)

    cls_results[name] = dict(
        accuracy=acc, f1=f1, precision=prec, recall=rec,
        cv_mean=cv.mean(), cv_std=cv.std(), y_pred=y_pred,
        report=classification_report(y_cls_test, y_pred,
                                     target_names=price_labels, zero_division=0)
    )
    print(f"\n  {name}")
    print(f"    Accuracy  = {acc:.4f}   F1 = {f1:.4f}")
    print(f"    Precision = {prec:.4f}   Recall = {rec:.4f}")
    print(f"    CV Accuracy = {cv.mean():.4f} +/- {cv.std():.4f}")

# ---- Plot 4: Confusion matrices for all 5 classifiers ----
fig, axes = plt.subplots(1, 5, figsize=(26, 5))
for ax, (name, res) in zip(axes, cls_results.items()):
    cm = confusion_matrix(y_cls_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=price_labels, yticklabels=price_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f"{name}\nAcc = {res['accuracy']:.2%}", fontsize=10)
plt.suptitle('Classification: Confusion Matrices', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output_plots/04_classification_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: output_plots/04_classification_confusion_matrices.png")

# ---- Plot 5: Accuracy + F1 bar comparison ----
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
cnames = list(cls_results.keys())
accs    = [cls_results[n]['accuracy'] for n in cnames]
f1s     = [cls_results[n]['f1'] for n in cnames]
colors5 = ['#e74c3c', '#9b59b6', '#f39c12', '#3498db', '#2ecc71']

bars = axes[0].bar(cnames, accs, color=colors5, edgecolor='black', alpha=0.85)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Test Accuracy  (higher = better)')
axes[0].set_ylim(0, 1.15)
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)
axes[0].grid(axis='y', alpha=0.3)
axes[0].tick_params(axis='x', rotation=15)

bars = axes[1].bar(cnames, f1s, color=colors5, edgecolor='black', alpha=0.85)
axes[1].set_ylabel('F1 Score (weighted)')
axes[1].set_title('F1 Score  (higher = better)')
axes[1].set_ylim(0, 1.15)
for bar, val in zip(bars, f1s):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', rotation=15)

plt.suptitle('Classification Model Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output_plots/05_classification_accuracy_f1.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_plots/05_classification_accuracy_f1.png")

# ---- Plot 6: CV Accuracy with error bars ----
fig, ax = plt.subplots(figsize=(10, 6))
cv_means = [cls_results[n]['cv_mean'] for n in cnames]
cv_stds  = [cls_results[n]['cv_std'] for n in cnames]
bars = ax.bar(cnames, cv_means, yerr=cv_stds, color=colors5,
              edgecolor='black', alpha=0.85, capsize=8)
ax.set_ylabel('CV Accuracy')
ax.set_title('5-Fold Cross-Validation Accuracy  (higher & more stable = better)')
ax.set_ylim(0, 1.15)
for bar, val, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.03,
            f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig('output_plots/06_classification_cv_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_plots/06_classification_cv_accuracy.png")

# ---- Plot 7: Classification CV boxplot ----
fig, ax = plt.subplots(figsize=(10, 6))
cv_cls_data = []
for name, model in cls_models.items():
    cv_cls_data.append(cross_val_score(model, X_train_sc, y_cls_train, cv=10, scoring='accuracy'))
bp = ax.boxplot(cv_cls_data, labels=cnames, patch_artist=True)
for patch, col in zip(bp['boxes'], colors5):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
ax.set_ylabel('Accuracy')
ax.set_title('Classification: 10-Fold Cross-Validation Accuracy')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig('output_plots/07_classification_cv_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_plots/07_classification_cv_boxplot.png")

# ---- Plot 8: Precision / Recall / F1 grouped bar chart ----
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cnames))
width = 0.25
precs = [cls_results[n]['precision'] for n in cnames]
recs  = [cls_results[n]['recall'] for n in cnames]

ax.bar(x - width, precs, width, label='Precision', color='#3498db', edgecolor='black', alpha=0.85)
ax.bar(x,         recs,  width, label='Recall',    color='#e74c3c', edgecolor='black', alpha=0.85)
ax.bar(x + width, f1s,   width, label='F1 Score',  color='#2ecc71', edgecolor='black', alpha=0.85)

for i in range(len(cnames)):
    ax.text(x[i] - width, precs[i] + 0.02, f'{precs[i]:.2f}', ha='center', fontsize=8)
    ax.text(x[i],         recs[i]  + 0.02, f'{recs[i]:.2f}',  ha='center', fontsize=8)
    ax.text(x[i] + width, f1s[i]   + 0.02, f'{f1s[i]:.2f}',   ha='center', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(cnames, rotation=15)
ax.set_ylabel('Score')
ax.set_title('Classification: Precision / Recall / F1 per Model')
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('output_plots/08_classification_precision_recall_f1.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_plots/08_classification_precision_recall_f1.png")

# ============================================================================
# PART C — GRAND COMPARISON DASHBOARD
# ============================================================================
print("\n" + "=" * 80)
print("GRAND COMPARISON — WHICH ALGORITHM IS BEST?")
print("=" * 80)

# ---- Plot 9: Grand dashboard ----
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Top-left: Regression R²
ax = axes[0, 0]
bars = ax.barh(rnames, r2s, color=colors_reg, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, r2s):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontweight='bold')
ax.set_xlabel('R² Score')
ax.set_title('REGRESSION: R² Score  (higher = better)')
ax.set_xlim(0, max(r2s) * 1.2)
ax.grid(axis='x', alpha=0.3)

# Top-right: Regression RMSE
ax = axes[0, 1]
bars = ax.barh(rnames, rmses, color=colors_reg, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, rmses):
    ax.text(val + 200, bar.get_y() + bar.get_height()/2,
            f'{val:,.0f}', va='center', fontweight='bold')
ax.set_xlabel('RMSE')
ax.set_title('REGRESSION: RMSE  (lower = better)')
ax.grid(axis='x', alpha=0.3)

# Bottom-left: Classification Accuracy (all 5)
ax = axes[1, 0]
bars = ax.barh(cnames, accs, color=colors5, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, accs):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontweight='bold')
ax.set_xlabel('Accuracy')
ax.set_title('CLASSIFICATION: Accuracy  (higher = better)')
ax.set_xlim(0, 1.15)
ax.grid(axis='x', alpha=0.3)

# Bottom-right: Classification F1 (all 5)
ax = axes[1, 1]
bars = ax.barh(cnames, f1s, color=colors5, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, f1s):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontweight='bold')
ax.set_xlabel('F1 Score (weighted)')
ax.set_title('CLASSIFICATION: F1 Score  (higher = better)')
ax.set_xlim(0, 1.15)
ax.grid(axis='x', alpha=0.3)

plt.suptitle('ALGORITHM COMPARISON DASHBOARD', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('output_plots/09_grand_comparison_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: output_plots/09_grand_comparison_dashboard.png")

# ---- Plot 10: Winner summary ----
best_reg = max(reg_results, key=lambda n: reg_results[n]['r2'])
best_cls = max(cls_results, key=lambda n: cls_results[n]['accuracy'])
# tie-break on CV if accuracy is tied
tied_cls = [n for n in cls_results if cls_results[n]['accuracy'] == cls_results[best_cls]['accuracy']]
if len(tied_cls) > 1:
    best_cls = max(tied_cls, key=lambda n: cls_results[n]['cv_mean'])

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

all_models = ['Decision Tree', 'Naive Bayes', 'KNN', 'Linear', 'Random Forest']

summary_text = "FINAL RESULTS SUMMARY\n"
summary_text += "=" * 60 + "\n\n"
summary_text += "REGRESSION (predicting exact price)\n"
summary_text += "-" * 60 + "\n"
for n in rnames:
    r = reg_results[n]
    tag = "  << BEST" if n == best_reg else ""
    summary_text += f"  {n:20s}  R²={r['r2']:.4f}  RMSE={r['rmse']:>8,.0f}{tag}\n"

summary_text += "\nCLASSIFICATION (predicting Low/Medium/High)\n"
summary_text += "-" * 60 + "\n"
for n in cnames:
    r = cls_results[n]
    tag = "  << BEST" if n == best_cls else ""
    summary_text += f"  {n:20s}  Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}  CV={r['cv_mean']:.4f}{tag}\n"

summary_text += "\n" + "=" * 60 + "\n"
summary_text += f"  Best Regression model:      {best_reg}\n"
summary_text += f"  Best Classification model:   {best_cls}\n"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=12, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
plt.savefig('output_plots/10_winner_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: output_plots/10_winner_summary.png")

# ============================================================================
# PRINT FINAL VERDICT
# ============================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print("\n--- REGRESSION (predicting exact price) ---")
print(f"  {'Model':20s}  {'R²':>8s}  {'RMSE':>10s}  {'MAE':>10s}")
print("  " + "-" * 55)
for n in rnames:
    r = reg_results[n]
    tag = " << BEST" if n == best_reg else ""
    print(f"  {n:20s}  {r['r2']:8.4f}  {r['rmse']:>10,.0f}  {r['mae']:>10,.0f}{tag}")

print("\n--- CLASSIFICATION (predicting Low / Medium / High price) ---")
print(f"  {'Model':20s}  {'Acc':>8s}  {'F1':>8s}  {'CV Acc':>8s}")
print("  " + "-" * 50)
for n in cnames:
    r = cls_results[n]
    tag = " << BEST" if n == best_cls else ""
    print(f"  {n:20s}  {r['accuracy']:8.4f}  {r['f1']:8.4f}  {r['cv_mean']:8.4f}{tag}")

print(f"\nBest regression model:      {best_reg}")
print(f"Best classification model:   {best_cls}")

print("\n" + "=" * 80)
print("KEY TAKEAWAY")
print("=" * 80)
print("""
For this Utrecht housing dataset the task is predicting house prices.

  REGRESSION (exact price):
    - 4 of the 5 algorithms can do regression.
    - Naive Bayes is classification-only and cannot predict exact prices.

  CLASSIFICATION (Low / Medium / High price bucket):
    - All 5 algorithms are compared head-to-head here.
    - "Linear" uses Logistic Regression (the classification counterpart
      of Linear Regression).

Check the plots in output_plots/ for visual proof.
""")

print("OUTPUT FILES:")
import glob as _glob
for f in sorted(_glob.glob('output_plots/*.png')):
    print(f"  {f}")

print("\nDone!")
