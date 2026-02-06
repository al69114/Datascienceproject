# Utrecht Housing Price Prediction

Intro to Machine Learning workshop comparing **5 algorithms** on the Utrecht housing dataset to determine which performs best for predicting house prices.

## Algorithms Compared

| # | Algorithm | Regression | Classification |
|---|-----------|:----------:|:--------------:|
| 1 | Decision Tree | Yes | Yes |
| 2 | Naive Bayes | No (classification-only) | Yes |
| 3 | KNN (K-Nearest Neighbors) | Yes | Yes |
| 4 | Linear Regression / Logistic Regression | Yes | Yes |
| 5 | Random Forest | Yes | Yes |

## Dataset

**File:** `utrechthousingsmall.csv` — 100 houses from Utrecht, Netherlands with 17 columns.

**Features used:** zipcode, lot-len, lot-width, lot-area, house-area, garden-size, balcony, x-coor, y-coor, buildyear, bathrooms, taxvalue, energy-eff, monument, house_age (engineered), lot_utilization (engineered)

**Target variable:** `retailvalue` (market price)

## Results

### Regression — Predicting Exact Housing Price

4 of the 5 algorithms are compared (Naive Bayes cannot do regression).

| Model | R² | RMSE | MAE | CV R² |
|-------|---:|-----:|----:|------:|
| **Linear Regression** | **0.9891** | **23,019** | **18,089** | **0.9794** |
| Random Forest | 0.9382 | 54,945 | 41,808 | 0.9167 |
| Decision Tree | 0.8447 | 87,094 | 69,900 | 0.8859 |
| KNN | 0.7861 | 102,192 | 76,350 | 0.6985 |

**Winner: Linear Regression** with R² = 0.9891 (explains 98.9% of price variance) and the lowest error (RMSE = 23,019).

### Classification — Predicting Price Category (Low / Medium / High)

All 5 algorithms are compared head-to-head. The target is created by binning `retailvalue` into 3 equal-frequency categories.

| Model | Accuracy | F1 Score | Precision | Recall | CV Accuracy |
|-------|:--------:|:--------:|:---------:|:------:|:-----------:|
| **Random Forest** | **0.9500** | **0.9500** | **0.9571** | **0.9500** | **0.8500** |
| Decision Tree | 0.9500 | 0.9500 | 0.9571 | 0.9500 | 0.8125 |
| Linear (Logistic) | 0.9500 | 0.9494 | 0.9563 | 0.9500 | 0.7875 |
| Naive Bayes | 0.9000 | 0.8962 | 0.9222 | 0.9000 | 0.8000 |
| KNN | 0.8000 | 0.7955 | 0.8438 | 0.8000 | 0.6125 |

**Winner: Random Forest** — Decision Tree and Logistic Regression tied at 0.95 accuracy, but Random Forest has the highest cross-validation accuracy (0.8500), proving it generalizes best.

### Final Verdict

| Task | Best Algorithm | Key Metric |
|------|---------------|------------|
| **Regression** (exact price) | **Linear Regression** | R² = 0.9891 |
| **Classification** (Low/Medium/High) | **Random Forest** | Accuracy = 0.95, CV = 0.85 |
| Worst performer (both tasks) | KNN | Lowest R² and lowest accuracy |

**Why Linear Regression wins for regression:** The relationship between features and price is largely linear in this dataset. The strong correlation with `taxvalue` gives Linear Regression a near-perfect fit. More complex models (Decision Tree, KNN) actually overfit or underfit, producing worse results.

**Why Random Forest wins for classification:** While three models tied at 95% test accuracy, Random Forest's cross-validation score (0.85) is the highest, meaning it is the most consistent and generalizable across different data splits.

**Why KNN performs worst:** With only 100 samples and 16 features, KNN struggles with the curse of dimensionality — there aren't enough neighbors to make reliable predictions in such a high-dimensional space.

## Output Plots

All visualizations are saved to the `output_plots/` folder:

| File | Description |
|------|-------------|
| `01_regression_actual_vs_predicted.png` | Scatter plots showing predicted vs actual prices for all 4 regression models |
| `02_regression_r2_rmse.png` | Bar chart comparing R² and RMSE across regression models |
| `03_regression_cv_boxplot.png` | 10-fold cross-validation boxplot for regression models |
| `04_classification_confusion_matrices.png` | Confusion matrices for all 5 classifiers |
| `05_classification_accuracy_f1.png` | Accuracy and F1 score bar comparison |
| `06_classification_cv_accuracy.png` | Cross-validation accuracy with error bars |
| `07_classification_cv_boxplot.png` | 10-fold cross-validation boxplot for classifiers |
| `08_classification_precision_recall_f1.png` | Precision, Recall, and F1 grouped bar chart |
| `09_grand_comparison_dashboard.png` | 4-panel dashboard comparing all models |
| `10_winner_summary.png` | Final results summary image |

## How to Run

```bash
# Activate virtual environment
source venv/bin/activate

# Run the analysis
python housing_price_prediction.py
```

## Project Structure

```
Datascienceproject/
├── README.md                        # This file
├── housing_price_prediction.py      # Main analysis script
├── utrechthousingsmall.csv          # Housing dataset (100 rows)
├── requirements.txt                 # Python dependencies
├── output_plots/                    # Generated visualizations (10 plots)
└── venv/                            # Virtual environment
```

## Dependencies

- pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

Install with: `pip install -r requirements.txt`
