"""Time Series Forecasting Demo: ARIMA vs AutoReg vs Feature Engineering model.

This script is intentionally written as a learning-friendly project:
- Clear pipeline steps
- Explainable metrics
- Reproducible outputs for sharing
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def load_series(csv_path: Path) -> pd.Series:
    """Load CSV and return a daily indexed numeric series."""
    df = pd.read_csv(csv_path)
    required_cols = {"Date", "Births"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain 'Date' and 'Births' columns.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    series = df["Births"].astype(float).asfreq("D")
    return series


def split_train_test(series: pd.Series, test_size: int = 30) -> tuple[pd.Series, pd.Series]:
    """Hold out the last N days as test data."""
    if len(series) <= test_size:
        raise ValueError("test_size must be smaller than total number of rows.")
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test


def select_arima_order(train: pd.Series) -> tuple[int, int, int]:
    """Pick ARIMA(p,d,q) using lowest AIC over a small grid."""
    best_order = None
    best_aic = np.inf

    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 4):
                if (p, d, q) == (0, 0, 0):
                    continue
                try:
                    fit = ARIMA(train, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    if best_order is None:
        raise RuntimeError("Could not fit any ARIMA order.")
    return best_order


def forecast_with_arima(train: pd.Series, horizon: int) -> tuple[pd.Series, tuple[int, int, int]]:
    order = select_arima_order(train)
    fit = ARIMA(train, order=order).fit()
    preds = fit.forecast(steps=horizon)
    return preds, order


def select_autoreg_lag(train: pd.Series, max_lag: int = 30) -> int:
    """Pick the best lag value for AutoReg using AIC."""
    best_lag = None
    best_aic = np.inf

    upper = min(max_lag, len(train) // 2)
    for lag in range(1, upper + 1):
        try:
            fit = AutoReg(train, lags=lag, old_names=False).fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_lag = lag
        except Exception:
            continue

    if best_lag is None:
        raise RuntimeError("Could not fit AutoReg with any lag.")
    return best_lag


def forecast_with_autoreg(train: pd.Series, horizon: int) -> tuple[pd.Series, int]:
    lag = select_autoreg_lag(train)
    fit = AutoReg(train, lags=lag, old_names=False).fit()
    preds = fit.predict(start=len(train), end=len(train) + horizon - 1, dynamic=False)
    return preds, lag


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features known in advance for future dates."""
    dow = index.dayofweek
    month = index.month
    return pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
        },
        index=index,
    )


def build_training_features(train: pd.Series, max_lag: int = 14) -> tuple[pd.DataFrame, pd.Series]:
    """Feature engineering on training data only (no leakage)."""
    frame = pd.DataFrame({"y": train})
    for lag in range(1, max_lag + 1):
        frame[f"lag_{lag}"] = train.shift(lag)

    frame["roll_mean_7"] = train.shift(1).rolling(7).mean()
    frame["roll_std_7"] = train.shift(1).rolling(7).std()
    frame["roll_mean_30"] = train.shift(1).rolling(30).mean()

    frame = frame.join(add_calendar_features(frame.index))
    frame = frame.dropna()

    X = frame.drop(columns=["y"])
    y = frame["y"]
    return X, y


def make_next_feature_row(history: pd.Series, next_date: pd.Timestamp, max_lag: int = 14) -> pd.DataFrame:
    """Create one feature row for recursive forecasting."""
    if len(history) < max_lag:
        raise ValueError("History is too short for selected max_lag.")

    row = {}
    for lag in range(1, max_lag + 1):
        row[f"lag_{lag}"] = float(history.iloc[-lag])

    row["roll_mean_7"] = float(history.iloc[-7:].mean())
    row["roll_std_7"] = float(history.iloc[-7:].std())
    row["roll_mean_30"] = float(history.iloc[-30:].mean()) if len(history) >= 30 else float(history.mean())

    cal = add_calendar_features(pd.DatetimeIndex([next_date])).iloc[0]
    row["dow_sin"] = float(cal["dow_sin"])
    row["dow_cos"] = float(cal["dow_cos"])
    row["month_sin"] = float(cal["month_sin"])
    row["month_cos"] = float(cal["month_cos"])

    return pd.DataFrame([row], index=[next_date])


def forecast_with_feature_engineering(
    train: pd.Series,
    forecast_index: pd.DatetimeIndex,
    max_lag: int = 14,
) -> tuple[pd.Series, int]:
    """Train a linear model on engineered features and forecast recursively."""
    X_train, y_train = build_training_features(train, max_lag=max_lag)

    model = LinearRegression()
    model.fit(X_train, y_train)

    history = train.copy()
    preds = []

    for dt in forecast_index:
        row = make_next_feature_row(history, dt, max_lag=max_lag)
        pred = float(model.predict(row)[0])
        preds.append(pred)
        history.loc[dt] = pred

    return pd.Series(preds, index=forecast_index), max_lag


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float, float]:
    """Return MAE, RMSE, MAPE(%) where lower is better."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


def pick_winner(metrics: pd.DataFrame) -> str:
    """Select winner by RMSE, then MAE as tie-breaker."""
    ranked = metrics.sort_values(["RMSE", "MAE"], ascending=True)
    return str(ranked.iloc[0]["Model"])


def build_summary_markdown(
    csv_path: Path,
    train_size: int,
    test_size: int,
    arima_order: tuple[int, int, int],
    ar_lag: int,
    fe_lag: int,
    metrics: pd.DataFrame,
    winner: str,
) -> str:
    """Create a plain-language report for non-technical readers."""

    arima_row = metrics.loc[metrics["Model"] == "ARIMA"].iloc[0]
    ar_row = metrics.loc[metrics["Model"] == "AutoReg"].iloc[0]
    fe_row = metrics.loc[metrics["Model"] == "FeatEng-LinearReg"].iloc[0]

    return f"""# Forecasting Project Report

## 1) General Idea (Simple Explanation)
Time series forecasting means we use past values of a variable to estimate future values.
In this project, the variable is **daily female births**.

We compare three approaches:
- **ARIMA**: combines trend/differencing and past errors.
- **AutoReg**: uses previous values (lags).
- **Feature Engineering + Linear Regression**: creates useful predictors from time and past values, then learns a regression model.

## 2) Dataset
- Source file: `{csv_path}`
- Total observations: {train_size + test_size}
- Train set: {train_size} days
- Test set: {test_size} days (the most recent days)

## 3) Engineered Features
The feature-engineering model uses:
- Lag features: previous {fe_lag} day values
- Rolling features: 7-day mean, 7-day std, 30-day mean
- Calendar features: day-of-week and month cyclic encodings (sin/cos)

## 4) How We Evaluated Models
We trained on historical data, predicted the test period, then compared predictions with real values.
Lower values are better for all metrics:
- **MAE**: average absolute error
- **RMSE**: penalizes larger errors more strongly
- **MAPE**: average percentage error

## 5) Model Settings Found by the Script
- ARIMA order: `{arima_order}`
- AutoReg lag: `{ar_lag}`
- Feature model max lag: `{fe_lag}`

## 6) Results
| Model | MAE | RMSE | MAPE(%) |
|---|---:|---:|---:|
| ARIMA | {arima_row['MAE']:.4f} | {arima_row['RMSE']:.4f} | {arima_row['MAPE(%)']:.4f} |
| AutoReg | {ar_row['MAE']:.4f} | {ar_row['RMSE']:.4f} | {ar_row['MAPE(%)']:.4f} |
| FeatEng-LinearReg | {fe_row['MAE']:.4f} | {fe_row['RMSE']:.4f} | {fe_row['MAPE(%)']:.4f} |

## 7) Conclusion
**Best model on this split: {winner}** (based on lowest RMSE).

This does not mean it is always the best model for all datasets.
For stronger confidence, run walk-forward validation and test on multiple periods.

## 8) Output Files
- `outputs/metrics.csv`
- `outputs/predictions.csv`
- `outputs/forecast_plot.png`
- `outputs/project_report.md`
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learning-friendly time series forecasting with ARIMA, AutoReg, and feature engineering"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=r"C:\Users\Yassine\OneDrive\Documents\daily-total-female-births.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument("--test-size", type=int, default=30, help="Number of points for test set")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--fe-max-lag", type=int, default=14, help="Max lag for engineered feature model")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = load_series(csv_path)
    train, test = split_train_test(series, test_size=args.test_size)

    arima_pred, arima_order = forecast_with_arima(train, len(test))
    ar_pred, ar_lag = forecast_with_autoreg(train, len(test))
    fe_pred, fe_lag = forecast_with_feature_engineering(train, test.index, max_lag=args.fe_max_lag)

    arima_mae, arima_rmse, arima_mape = evaluate(test, arima_pred)
    ar_mae, ar_rmse, ar_mape = evaluate(test, ar_pred)
    fe_mae, fe_rmse, fe_mape = evaluate(test, fe_pred)

    metrics = pd.DataFrame(
        {
            "Model": ["ARIMA", "AutoReg", "FeatEng-LinearReg"],
            "Config": [f"order={arima_order}", f"lag={ar_lag}", f"max_lag={fe_lag}"],
            "MAE": [arima_mae, ar_mae, fe_mae],
            "RMSE": [arima_rmse, ar_rmse, fe_rmse],
            "MAPE(%)": [arima_mape, ar_mape, fe_mape],
        }
    )

    winner = pick_winner(metrics)

    preds = pd.DataFrame(
        {
            "Actual": test.values,
            "ARIMA_Pred": np.array(arima_pred),
            "AutoReg_Pred": np.array(ar_pred),
            "FeatEng_Pred": np.array(fe_pred),
        },
        index=test.index,
    )

    metrics_path = out_dir / "metrics.csv"
    preds_path = out_dir / "predictions.csv"
    plot_path = out_dir / "forecast_plot.png"
    report_path = out_dir / "project_report.md"

    metrics.to_csv(metrics_path, index=False)
    preds.to_csv(preds_path)

    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train.values, label="Train", color="black", alpha=0.5)
    plt.plot(test.index, test.values, label="Actual Test", color="tab:blue", linewidth=2)
    plt.plot(test.index, arima_pred, label=f"ARIMA {arima_order}", color="tab:red")
    plt.plot(test.index, ar_pred, label=f"AutoReg lag={ar_lag}", color="tab:green")
    plt.plot(test.index, fe_pred, label=f"FeatEng lag={fe_lag}", color="tab:orange")
    plt.title("Daily Female Births Forecast: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Births")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    report_md = build_summary_markdown(
        csv_path=csv_path,
        train_size=len(train),
        test_size=len(test),
        arima_order=arima_order,
        ar_lag=ar_lag,
        fe_lag=fe_lag,
        metrics=metrics,
        winner=winner,
    )
    report_path.write_text(report_md, encoding="utf-8")

    print("Dataset:", csv_path)
    print("Total rows:", len(series))
    print("Train/Test:", len(train), "/", len(test))
    print("ARIMA order:", arima_order)
    print("AutoReg lag:", ar_lag)
    print("Feature model max lag:", fe_lag)
    print("\nMetrics (lower is better):")
    print(metrics.to_string(index=False))
    print("\nWinner by RMSE:", winner)
    print("\nSaved:")
    print("-", metrics_path.resolve())
    print("-", preds_path.resolve())
    print("-", plot_path.resolve())
    print("-", report_path.resolve())


if __name__ == "__main__":
    main()
