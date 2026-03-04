# Forecasting Project Report

## 1) General Idea (Simple Explanation)
Time series forecasting means we use past values of a variable to estimate future values.
In this project, the variable is **daily female births**.

We compare three approaches:
- **ARIMA**: combines trend/differencing and past errors.
- **AutoReg**: uses previous values (lags).
- **Feature Engineering + Linear Regression**: creates useful predictors from time and past values, then learns a regression model.

## 2) Dataset
- Source file: `C:\Users\Yassine\OneDrive\Documents\daily-total-female-births.csv`
- Total observations: 365
- Train set: 335 days
- Test set: 30 days (the most recent days)

## 3) Engineered Features
The feature-engineering model uses:
- Lag features: previous 14 day values
- Rolling features: 7-day mean, 7-day std, 30-day mean
- Calendar features: day-of-week and month cyclic encodings (sin/cos)

## 4) How We Evaluated Models
We trained on historical data, predicted the test period, then compared predictions with real values.
Lower values are better for all metrics:
- **MAE**: average absolute error
- **RMSE**: penalizes larger errors more strongly
- **MAPE**: average percentage error

## 5) Model Settings Found by the Script
- ARIMA order: `(1, 1, 1)`
- AutoReg lag: `30`
- Feature model max lag: `14`

## 6) Results
| Model | MAE | RMSE | MAPE(%) |
|---|---:|---:|---:|
| ARIMA | 6.3013 | 7.1592 | 15.9687 |
| AutoReg | 6.0080 | 6.7955 | 14.9505 |
| FeatEng-LinearReg | 5.2537 | 6.2983 | 12.4876 |

## 7) Conclusion
**Best model on this split: FeatEng-LinearReg** (based on lowest RMSE).

This does not mean it is always the best model for all datasets.
For stronger confidence, run walk-forward validation and test on multiple periods.

## 8) Output Files
- `outputs/metrics.csv`
- `outputs/predictions.csv`
- `outputs/forecast_plot.png`
- `outputs/project_report.md`
