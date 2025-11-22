"""
Utrecht Housing Price Prediction - Data Science Project

This script implements a predictive model for housing prices in Utrecht using multiple regression techniques.

Objectives:
1. Implement a main predictive model
2. Compare against 3 baseline models
3. Demonstrate model improvement techniques
4. Explain statistical significance of results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Statistical Testing
from scipy.stats import f_oneway, ttest_rel

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory for plots
os.makedirs('output_plots', exist_ok=True)

print("="*80)
print("UTRECHT HOUSING PRICE PREDICTION")
print("="*80)
print()

# ============================================================================
# 1. LOAD AND EXPLORE DATASET
# ============================================================================
print("1. LOADING DATASET...")
print("-" * 80)

df = pd.read_csv('utrechthousingsmall.csv', encoding='utf-8-sig')

print(f"Dataset Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\n\nDataset Info:")
df.info()
print(f"\n\nBasic Statistics:")
print(df.describe())

# Check for missing values
print(f"\n\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found!")

print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("2. EXPLORATORY DATA ANALYSIS")
print("-" * 80)

# Distribution of target variable (retailvalue)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['retailvalue'], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Retail Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Housing Prices')

axes[1].boxplot(df['retailvalue'])
axes[1].set_ylabel('Retail Value')
axes[1].set_title('Boxplot of Housing Prices')

plt.tight_layout()
plt.savefig('output_plots/01_price_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: output_plots/01_price_distribution.png")

print(f"\nPrice Statistics:")
print(f"Mean Price: ${df['retailvalue'].mean():,.2f}")
print(f"Median Price: ${df['retailvalue'].median():,.2f}")
print(f"Std Dev: ${df['retailvalue'].std():,.2f}")

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.savefig('output_plots/02_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: output_plots/02_correlation_matrix.png")

print("\nCorrelation with retailvalue (sorted):")
print(correlation_matrix['retailvalue'].sort_values(ascending=False))

# Scatter plots of key features vs target
key_features = ['house-area', 'lot-area', 'buildyear', 'bathrooms', 'taxvalue']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    if feature in df.columns:
        axes[idx].scatter(df[feature], df['retailvalue'], alpha=0.5)
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Retail Value')
        axes[idx].set_title(f'{feature} vs Retail Value')

# Hide the last subplot if not used
if len(key_features) < 6:
    axes[5].set_visible(False)

plt.tight_layout()
plt.savefig('output_plots/03_feature_relationships.png', dpi=300, bbox_inches='tight')
print("Saved: output_plots/03_feature_relationships.png")

# ============================================================================
# 3. DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("3. DATA PREPROCESSING AND FEATURE ENGINEERING")
print("-" * 80)

# Create a copy for preprocessing
df_processed = df.copy()

# Drop id column (not useful for prediction)
if 'id' in df_processed.columns:
    df_processed = df_processed.drop('id', axis=1)

# Feature Engineering: Create new features
df_processed['house_age'] = 2024 - df_processed['buildyear']
df_processed['price_per_sqm'] = df_processed['retailvalue'] / df_processed['house-area']
df_processed['lot_utilization'] = df_processed['house-area'] / df_processed['lot-area']
df_processed['has_garden'] = (df_processed['garden-size'] > 0).astype(int)

print("New features created:")
print("  - house_age: Age of the house (2024 - buildyear)")
print("  - price_per_sqm: Price per square meter")
print("  - lot_utilization: House area / lot area ratio")
print("  - has_garden: Binary indicator for garden presence")

print(f"\nProcessed dataset shape: {df_processed.shape}")

# Prepare features and target
X = df_processed.drop(['retailvalue', 'select'], axis=1, errors='ignore')
y = df_processed['retailvalue']

print(f"\nFeatures ({len(X.columns)}): {X.columns.tolist()}")
print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Train/Test split ratio: {X_train.shape[0]/X_test.shape[0]:.1f}:1")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeatures scaled using StandardScaler")
print(f"Mean of scaled features: {X_train_scaled.mean():.6f}")
print(f"Std of scaled features: {X_train_scaled.std():.6f}")

# ============================================================================
# 4. BASELINE MODELS
# ============================================================================
print("\n" + "="*80)
print("4. TRAINING BASELINE MODELS")
print("-" * 80)

models = {}
results = {}

# 1. Linear Regression (Baseline)
print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
models['Linear Regression'] = lr

# 2. Decision Tree Regressor
print("Training Decision Tree...")
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)
models['Decision Tree'] = dt

# 3. K-Nearest Neighbors
print("Training KNN...")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
models['KNN'] = knn

print("\nAll baseline models trained successfully!")

# Evaluate all baseline models
print("\n" + "="*80)
print("BASELINE MODELS PERFORMANCE")
print("="*80)

for name, model in models.items():
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

    # Store results
    results[name] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred_test
    }

    print(f"\n{name}:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  RMSE:     ${test_rmse:,.2f}")
    print(f"  MAE:      ${test_mae:,.2f}")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n" + "="*80)

# ============================================================================
# 5. MAIN MODEL: RANDOM FOREST WITH HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("5. MAIN MODEL: RANDOM FOREST WITH HYPERPARAMETER TUNING")
print("-" * 80)

# Random Forest with default parameters first
print("\nTraining Random Forest (default parameters)...")
rf_default = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_default.fit(X_train_scaled, y_train)

y_pred_rf_default = rf_default.predict(X_test_scaled)
rf_default_r2 = r2_score(y_test, y_pred_rf_default)
rf_default_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_default))

print(f"Random Forest (default) - Test R²: {rf_default_r2:.4f}")
print(f"Random Forest (default) - RMSE: ${rf_default_rmse:,.2f}")

# Hyperparameter tuning with GridSearchCV
print("\nPerforming hyperparameter tuning with GridSearchCV...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV R² score: {grid_search.best_score_:.4f}")

# Use the best model
rf_best = grid_search.best_estimator_

# Evaluate the optimized model
y_pred_rf_train = rf_best.predict(X_train_scaled)
y_pred_rf_test = rf_best.predict(X_test_scaled)

rf_train_r2 = r2_score(y_train, y_pred_rf_train)
rf_test_r2 = r2_score(y_test, y_pred_rf_test)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
rf_test_mae = mean_absolute_error(y_test, y_pred_rf_test)

# Cross-validation
rf_cv_scores = cross_val_score(rf_best, X_train_scaled, y_train, cv=5, scoring='r2')

# Store results
results['Random Forest (Optimized)'] = {
    'train_r2': rf_train_r2,
    'test_r2': rf_test_r2,
    'test_rmse': rf_test_rmse,
    'test_mae': rf_test_mae,
    'cv_mean': rf_cv_scores.mean(),
    'cv_std': rf_cv_scores.std(),
    'predictions': y_pred_rf_test
}

print("\n" + "="*80)
print("MAIN MODEL: Random Forest (Optimized)")
print("="*80)
print(f"Train R²: {rf_train_r2:.4f}")
print(f"Test R²:  {rf_test_r2:.4f}")
print(f"RMSE:     ${rf_test_rmse:,.2f}")
print(f"MAE:      ${rf_test_mae:,.2f}")
print(f"CV R² (mean ± std): {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
print("="*80)

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('output_plots/04_feature_importance.png', dpi=300, bbox_inches='tight')
print("\nSaved: output_plots/04_feature_importance.png")

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("6. MODEL COMPARISON")
print("-" * 80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train R²': [results[m]['train_r2'] for m in results.keys()],
    'Test R²': [results[m]['test_r2'] for m in results.keys()],
    'RMSE': [results[m]['test_rmse'] for m in results.keys()],
    'MAE': [results[m]['test_mae'] for m in results.keys()],
    'CV R² Mean': [results[m]['cv_mean'] for m in results.keys()],
    'CV R² Std': [results[m]['cv_std'] for m in results.keys()]
}).sort_values('Test R²', ascending=False)

print("\nMODEL COMPARISON:")
print("="*100)
print(comparison_df.to_string(index=False))
print("="*100)

# Visualize model comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# R² comparison
x_pos = np.arange(len(comparison_df))
axes[0].bar(x_pos - 0.2, comparison_df['Train R²'], 0.4, label='Train R²', alpha=0.8)
axes[0].bar(x_pos + 0.2, comparison_df['Test R²'], 0.4, label='Test R²', alpha=0.8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Score Comparison')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# RMSE comparison
axes[1].bar(comparison_df['Model'], comparison_df['RMSE'], alpha=0.8, color='coral')
axes[1].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
axes[1].set_ylabel('RMSE ($)')
axes[1].set_title('RMSE Comparison (Lower is Better)')
axes[1].grid(axis='y', alpha=0.3)

# CV Score with error bars
axes[2].bar(comparison_df['Model'], comparison_df['CV R² Mean'],
           yerr=comparison_df['CV R² Std'], alpha=0.8, color='lightgreen',
           capsize=5)
axes[2].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
axes[2].set_ylabel('CV R² Score')
axes[2].set_title('Cross-Validation R² (with std dev)')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output_plots/05_model_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: output_plots/05_model_comparison.png")

# ============================================================================
# 7. STATISTICAL SIGNIFICANCE TESTING
# ============================================================================
print("\n" + "="*80)
print("7. STATISTICAL SIGNIFICANCE TESTING")
print("-" * 80)

# Perform cross-validation for all models to get multiple samples
print("\nPerforming 10-fold cross-validation for statistical testing...\n")

cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='r2')
    cv_results[name] = cv_scores
    print(f"{name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Add Random Forest
rf_cv = cross_val_score(rf_best, X_train_scaled, y_train, cv=10, scoring='r2')
cv_results['Random Forest (Optimized)'] = rf_cv
print(f"Random Forest (Optimized): {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# ANOVA test - compare all models
print("\n" + "="*80)
print("ANOVA TEST - Comparing all models")
print("="*80)

f_statistic, p_value = f_oneway(*cv_results.values())

print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("\nConclusion: There is a statistically significant difference between models (p < 0.05)")
    print("This means the performance differences we observe are unlikely due to random chance.")
else:
    print("\nConclusion: No statistically significant difference between models (p >= 0.05)")

print("="*80)

# Paired t-tests: Random Forest vs each baseline
print("\n" + "="*80)
print("PAIRED T-TESTS - Random Forest vs Baseline Models")
print("="*80)

rf_scores = cv_results['Random Forest (Optimized)']

for model_name in ['Linear Regression', 'Decision Tree', 'KNN']:
    baseline_scores = cv_results[model_name]

    t_stat, p_val = ttest_rel(rf_scores, baseline_scores)

    print(f"\nRandom Forest vs {model_name}:")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_val:.6f}")
    print(f"  Mean difference: {(rf_scores.mean() - baseline_scores.mean()):.4f}")

    if p_val < 0.05:
        if t_stat > 0:
            print(f"  Conclusion: Random Forest is significantly BETTER (p < 0.05)")
        else:
            print(f"  Conclusion: Random Forest is significantly WORSE (p < 0.05)")
    else:
        print(f"  Conclusion: No significant difference (p >= 0.05)")

print("\n" + "="*80)

# Confidence intervals for model performance
print("\n" + "="*80)
print("95% CONFIDENCE INTERVALS for R² Scores")
print("="*80)

for model_name, scores in cv_results.items():
    mean = scores.mean()
    std_err = scores.std() / np.sqrt(len(scores))
    ci_lower = mean - 1.96 * std_err
    ci_upper = mean + 1.96 * std_err

    print(f"\n{model_name}:")
    print(f"  Mean R²: {mean:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

print("\n" + "="*80)

# ============================================================================
# 8. MODEL PREDICTIONS VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("8. PREDICTION VISUALIZATIONS")
print("-" * 80)

# Actual vs Predicted for Random Forest
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
axes[0].scatter(y_test, y_pred_rf_test, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title('Random Forest: Actual vs Predicted Prices')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Residuals plot
residuals = y_test - y_pred_rf_test
axes[1].scatter(y_pred_rf_test, residuals, alpha=0.6)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals Plot')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('output_plots/06_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("Saved: output_plots/06_actual_vs_predicted.png")

# Residuals distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Residuals')
axes[0].axvline(x=0, color='r', linestyle='--', lw=2)

stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.savefig('output_plots/07_residuals_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: output_plots/07_residuals_analysis.png")

print(f"\nResidual Statistics:")
print(f"Mean of residuals: ${residuals.mean():,.2f}")
print(f"Std of residuals: ${residuals.std():,.2f}")

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PROJECT SUMMARY")
print("="*80)

print("\n1. DATASET:")
print(f"   - Total samples: {len(df)}")
print(f"   - Features used: {len(X.columns)}")
print(f"   - Target variable: retailvalue (housing prices)")

print("\n2. MODELS IMPLEMENTED:")
for i, model_name in enumerate(results.keys(), 1):
    print(f"   {i}. {model_name}")

print("\n3. BEST MODEL:")
best_model_name = comparison_df.iloc[0]['Model']
best_r2 = comparison_df.iloc[0]['Test R²']
best_rmse = comparison_df.iloc[0]['RMSE']
print(f"   Model: {best_model_name}")
print(f"   Test R²: {best_r2:.4f}")
print(f"   RMSE: ${best_rmse:,.2f}")
print(f"   This model explains {best_r2*100:.2f}% of the variance in housing prices")

print("\n4. MODEL IMPROVEMENTS APPLIED:")
print("   - Feature engineering (house_age, lot_utilization, price_per_sqm, has_garden)")
print("   - Feature scaling (StandardScaler)")
print("   - Hyperparameter tuning (GridSearchCV)")
print("   - Cross-validation for robust evaluation (5-fold and 10-fold)")

print("\n5. STATISTICAL SIGNIFICANCE:")
print(f"   - ANOVA F-statistic: {f_statistic:.4f}")
print(f"   - ANOVA p-value: {p_value:.6f}")
if p_value < 0.05:
    print("   - Models show statistically significant differences (p < 0.05)")
    print("   - The performance differences are unlikely due to random chance")
else:
    print("   - No statistically significant differences between models")

print("\n6. TOP 5 IMPORTANT FEATURES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")

print("\n7. OUTPUT FILES GENERATED:")
print("   - output_plots/01_price_distribution.png")
print("   - output_plots/02_correlation_matrix.png")
print("   - output_plots/03_feature_relationships.png")
print("   - output_plots/04_feature_importance.png")
print("   - output_plots/05_model_comparison.png")
print("   - output_plots/06_actual_vs_predicted.png")
print("   - output_plots/07_residuals_analysis.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nAll results have been printed above and plots saved to 'output_plots/' directory.")
print("You can use these results for your project submission.")
print()
