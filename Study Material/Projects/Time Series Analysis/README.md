# AAPL Stock Price Time Series Analysis

A comprehensive time series analysis project focused on Apple (AAPL) stock price data, implementing classical decomposition methods, stationarity testing, and various forecasting models.

## Project Overview

This project performs an in-depth time series analysis of Apple stock prices, covering data preprocessing, decomposition, stationarity testing, transformation techniques, and forecasting using AR, MA, ARMA, and ARIMA models.

## Dataset

- **Source**: Apple Inc. (AAPL) historical stock data
- **Data URL**: [CSV File on GitHub](https://raw.githubusercontent.com/rahulinchal/CSV-Files-/refs/heads/main/Data/AAPL.csv)
- **Analysis Period**: Focused on data from 2015-01-01 to 2018-12-31
- **Features**: Date, Open, High, Low, Close, Volume

## Requirements

```python
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
yfinance
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn yfinance
```

## Project Structure

### 1. Data Preprocessing
- Loading and cleaning the dataset
- Removing extra spaces from column names
- Converting date strings to datetime objects
- Removing dollar signs and converting price columns to float
- Handling missing values and duplicates

### 2. Time Series Decomposition

#### Classical Decomposition
- **Model Type**: Additive decomposition
- **Period**: 30 days
- **Components**: Trend, Seasonal, Residual

#### STL Decomposition (Seasonal and Trend decomposition using Loess)
- **Period**: 30 days
- Provides robust decomposition with comparison to classical method

### 3. Stationarity Testing

The project implements multiple stationarity tests:

#### Augmented Dickey-Fuller (ADF) Test
- **Null Hypothesis**: Series has a unit root (non-stationary)
- **Decision Rule**: p-value < 0.05 indicates stationarity

#### KPSS Test
- **Null Hypothesis**: Series is stationary
- **Decision Rule**: p-value > 0.05 indicates stationarity

#### Kolmogorov-Smirnov (K-S) Test
- Demonstrates strict vs non-strict stationarity with synthetic data

### 4. Transformation Techniques

Three transformation methods are explored to achieve stationarity:

1. **Differencing**
   - First-order differencing
   - Second-order differencing
   - Result: Data becomes stationary after first-order differencing

2. **Logarithmic Transformation**
   - Applied using `np.log()`
   - Result: Non-stationary

3. **Power Transformation**
   - Square root transformation using `np.sqrt()`
   - Result: Non-stationary

4. **Box-Cox Transformation**
   - Automatic lambda parameter selection
   - Result: Non-stationary

### 5. Forecasting Models

All models are evaluated using the last 30 data points as the test set.

#### AR (Autoregressive) Model
- **Lags**: 30
- Captures linear dependencies on past values

#### MA (Moving Average) Model
- **Order**: (0, 0, 30)
- Models the series as a linear combination of past error terms

#### ARMA Model
- **Order**: (7, 0, 7)
- Combines both AR and MA components

#### ARIMA Model
- **Order**: (7, 1, 7)
- Includes differencing component for non-stationary data
- Most comprehensive model tested

### 6. Model Evaluation

- **Metric**: Root Mean Squared Error (RMSE)
- Visual comparison of predictions vs actual test data
- All models plotted for comparison

## Key Findings

1. **Stationarity**: The original AAPL close price series is non-stationary
2. **Best Transformation**: First-order differencing successfully achieves stationarity
3. **Decomposition**: The series exhibits additive characteristics with consistent seasonal patterns
4. **Model Performance**: RMSE values calculated for each forecasting model

## Usage

1. Clone the repository and ensure all dependencies are installed
2. Run the notebook/script sequentially
3. The code will:
   - Download and preprocess the data
   - Perform decomposition analysis
   - Test for stationarity
   - Apply transformations
   - Train and evaluate forecasting models

## Functions Defined

```python
def remove(x)
    # Removes dollar signs and converts to float

def adf(df)
    # Performs ADF test and prints results

def kpss_res(df)
    # Performs KPSS test and prints results

def ks_test_stationarity(series)
    # Performs K-S test by splitting series
```

## Visualizations

The project includes numerous visualizations:
- Line plots of original and transformed data
- Decomposition plots (trend, seasonal, residual)
- Comparison plots of different decomposition methods
- Stationarity demonstration plots
- Forecast vs actual comparison plots

## Additional Trial Section

The project includes a trial section using `yfinance` to:
- Download recent AAPL data (from 2023-01-01)
- Apply the same modeling techniques
- Compare performance on more recent data

## Notes

- The analysis assumes an additive model based on consistent seasonal patterns
- First-order differencing is the most effective method for achieving stationarity
- Model hyperparameters (lags, order) can be tuned for better performance
- RMSE values should be compared relative to the scale of the data

## Future Improvements

- Implement automatic parameter selection (e.g., auto_arima)
- Add SARIMA for better seasonal modeling
- Include additional evaluation metrics (MAE, MAPE)
- Implement cross-validation for robust model comparison
- Add prediction intervals for uncertainty quantification

## Author

Time Series Analysis Project for AAPL Stock Prediction

## License

This project is for educational purposes.