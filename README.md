# Utrecht Housing Price Prediction

A data science project that implements predictive models for housing prices in Utrecht, Netherlands using machine learning regression techniques.

## Project Overview

This project analyzes housing data from Utrecht and builds multiple regression models to predict retail property values. The analysis includes exploratory data analysis, feature engineering, model comparison, and statistical significance testing.

## Objectives

1. **Implement a main predictive model** - Random Forest with hyperparameter optimization
2. **Compare against 3 baseline models** - Linear Regression, Decision Tree, K-Nearest Neighbors
3. **Demonstrate model improvement techniques** - Feature engineering, scaling, hyperparameter tuning
4. **Explain statistical significance** - ANOVA tests, paired t-tests, confidence intervals

## Dataset

**File:** `utrechthousingsmall.csv`

**Features:**
- `house-area` - Size of the house in square meters
- `lot-area` - Total lot/land area
- `garden-size` - Size of the garden
- `balcony` - Balcony presence/size
- `buildyear` - Year the house was built
- `bathrooms` - Number of bathrooms
- `x-coor`, `y-coor` - Geographic coordinates
- `zipcode` - Postal code
- `taxvalue` - Tax assessment value
- `energy-eff` - Energy efficiency rating
- `monument` - Monument status

**Target Variable:** `retailvalue` - Market/retail price of the house

## Models Implemented

### Baseline Models (3 required)
1. **Linear Regression** - Simple linear relationship between features and price
2. **Decision Tree** - Non-linear model using decision rules
3. **K-Nearest Neighbors (KNN)** - Prediction based on similar houses

### Main Model
4. **Random Forest (Optimized)** - Ensemble of decision trees with hyperparameter tuning via GridSearchCV

## Model Improvements Applied

1. **Feature Engineering**
   - `house_age` = 2024 - buildyear
   - `price_per_sqm` = retailvalue / house-area
   - `lot_utilization` = house-area / lot-area
   - `has_garden` = binary indicator for garden presence

2. **Data Preprocessing**
   - Feature scaling using StandardScaler
   - Train/test split (80/20)
   - Handling of categorical and numerical features

3. **Hyperparameter Tuning**
   - GridSearchCV for Random Forest optimization
   - Parameters tuned: n_estimators, max_depth, min_samples_split, min_samples_leaf

4. **Cross-Validation**
   - 5-fold cross-validation for model evaluation
   - 10-fold cross-validation for statistical testing

## Statistical Significance Testing

- **ANOVA Test** - Compares all models to determine if differences are statistically significant
- **Paired t-tests** - Compares Random Forest against each baseline model individually
- **95% Confidence Intervals** - Provides reliability ranges for model performance

## Installation & Setup

### Requirements
- Python 3.8+
- Virtual environment (included)

### Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Dependencies are already installed in venv
# If needed, reinstall with:
pip install -r requirements.txt
```

## How to Run

```bash
# Activate virtual environment
source venv/bin/activate

# Run the analysis script
python housing_price_prediction.py
```

**Expected runtime:** 2-5 minutes (depending on system)

## Output Files

### Console Output
The script prints comprehensive results including:
- Dataset statistics and information
- Feature correlations
- Model performance metrics (R², RMSE, MAE)
- Cross-validation scores
- Statistical test results (ANOVA, t-tests, confidence intervals)
- Feature importance rankings
- Final project summary

### Visualization Files (saved to `output_plots/`)

| File | Description |
|------|-------------|
| `01_price_distribution.png` | Histogram and boxplot showing the distribution of housing prices |
| `02_correlation_matrix.png` | Heatmap showing correlations between all features and target variable |
| `03_feature_relationships.png` | Scatter plots of 5 key features vs. housing prices |
| `04_feature_importance.png` | Bar chart ranking features by importance in Random Forest model |
| `05_model_comparison.png` | Performance comparison across all 4 models (R², RMSE, CV scores) |
| `06_actual_vs_predicted.png` | Actual vs predicted prices and residuals plot for Random Forest |
| `07_residuals_analysis.png` | Distribution of residuals and Q-Q plot for model validation |

## Understanding the Visualizations

### 01_price_distribution.png
- **Purpose:** Understand the distribution of housing prices in the dataset
- **What to look for:** Bell curve shape, outliers, typical price range
- **Interpretation:** Shows if data is normally distributed and identifies extreme values

### 02_correlation_matrix.png
- **Purpose:** Identify relationships between features
- **What to look for:** Red (positive correlation) and blue (negative correlation) cells
- **Interpretation:** Features highly correlated with `retailvalue` are strong predictors

### 03_feature_relationships.png
- **Purpose:** Visualize how individual features affect price
- **What to look for:** Linear or non-linear patterns, scatter density
- **Interpretation:** Clear patterns indicate strong predictive features

### 04_feature_importance.png
- **Purpose:** Show which features matter most for predictions
- **What to look for:** Longest bars indicate most important features
- **Interpretation:** Top features drive the model's decisions (likely: taxvalue, house-area)

### 05_model_comparison.png
- **Purpose:** Compare performance of all 4 models
- **What to look for:** Highest R², lowest RMSE, narrow error bars
- **Interpretation:** Random Forest should outperform baseline models

### 06_actual_vs_predicted.png
- **Purpose:** Validate prediction accuracy
- **What to look for:** Points close to the diagonal red line
- **Interpretation:** Closer to the line = more accurate predictions

### 07_residuals_analysis.png
- **Purpose:** Check if model meets statistical assumptions
- **What to look for:** Bell curve centered at zero, points on diagonal in Q-Q plot
- **Interpretation:** Good distribution validates model reliability

## Evaluation Metrics Explained

### R² Score (0 to 1, higher is better)
- **Meaning:** Percentage of price variance explained by the model
- **Example:** R² = 0.85 means the model explains 85% of why prices differ
- **Good score:** > 0.80 for this type of problem

### RMSE (Root Mean Squared Error, lower is better)
- **Meaning:** Average prediction error in dollars
- **Example:** RMSE = $30,000 means predictions are typically off by $30k
- **Interpretation:** Shows typical prediction accuracy

### MAE (Mean Absolute Error, lower is better)
- **Meaning:** Average absolute difference between prediction and actual price
- **Interpretation:** Similar to RMSE but less sensitive to outliers

### Cross-Validation Score
- **Meaning:** Average R² across multiple train/test splits
- **Purpose:** More reliable estimate than single train/test split
- **Interpretation:** Shows model consistency across different data subsets

## Expected Results

Based on the dataset and models:

- **Best Model:** Random Forest (Optimized) should achieve highest R²
- **Typical R²:** 0.75 - 0.90 depending on data quality
- **Feature Importance:** `taxvalue` and `house-area` typically most important
- **Statistical Significance:** ANOVA p-value < 0.05 (models are significantly different)

## Project Structure

```
Datascienceproject/
├── README.md                        # This file
├── housing_price_prediction.py      # Main analysis script
├── utrechthousingsmall.csv          # Housing dataset
├── requirements.txt                 # Python dependencies
├── output_plots/                    # Generated visualizations (7 plots)
│   ├── 01_price_distribution.png
│   ├── 02_correlation_matrix.png
│   ├── 03_feature_relationships.png
│   ├── 04_feature_importance.png
│   ├── 05_model_comparison.png
│   ├── 06_actual_vs_predicted.png
│   └── 07_residuals_analysis.png
└── venv/                            # Virtual environment (Python packages)
```

## Key Findings

After running the analysis, you should be able to answer:

1. **Which model performs best?** Random Forest (Optimized) with highest R² and lowest RMSE
2. **What features matter most?** Top 3-5 features from feature importance plot
3. **Is the difference significant?** Yes, if ANOVA p-value < 0.05
4. **How accurate are predictions?** Check RMSE and R² values
5. **Are improvements effective?** Compare Random Forest vs Linear Regression performance

## Technologies Used

- **Python 3.x** - Programming language
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning models and tools
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scipy** - Statistical testing

## Author

Data Science Project - Utrecht Housing Price Prediction

## License

This project is for educational purposes.
