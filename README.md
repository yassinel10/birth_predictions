# Time Series Forecasting Project (Beginner-Friendly)

This project explains the general idea of time series forecasting using a real dataset of daily female births.

## What Is Time Series Forecasting?
A time series is data recorded over time (day by day, month by month, etc.).
Forecasting means predicting future values based on historical patterns.

In this project, we answer:
"Can we predict future daily births from past daily births?"

## Models Used
- ARIMA
- AutoReg (Autoregression)
- Feature Engineering + Linear Regression

## Why Feature Engineering?
Feature engineering turns raw time series into additional useful signals so a model can learn better patterns.

Added engineered features:
- Lag features (`lag_1` to `lag_14` by default)
- Rolling window features (`roll_mean_7`, `roll_std_7`, `roll_mean_30`)
- Calendar cyclical features (`dow_sin`, `dow_cos`, `month_sin`, `month_cos`)

## Project Pipeline
1. Load and validate dataset (`Date`, `Births`)
2. Sort by date and set daily index
3. Train/test split (last 30 days as test)
4. Train ARIMA and AutoReg
5. Train engineered-feature regression model
6. Forecast test horizon
7. Compare with MAE, RMSE, MAPE
8. Save outputs and plain-language report

## Run
```powershell
cd C:\Users\Yassine\data
python .\time_series_forecasting.py
```

Optional:
```powershell
python .\time_series_forecasting.py --csv "C:\path\to\file.csv" --test-size 30 --output-dir outputs --fe-max-lag 14
```

## Output Files
- `outputs/metrics.csv`
- `outputs/predictions.csv`
- `outputs/forecast_plot.png`
- `outputs/project_report.md`

## Interpretation Tip
Lower MAE/RMSE/MAPE means better predictions on the test set.
A winner on one split is not always the best model for every period.
