import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.config_loader import load_config

from src.data.download_yfinance import download_ticker
from src.features.feature_pipeline import build_features
from src.regimes.clustering_regime import add_cluster_regimes
from src.regimes.regime_features import build_regime_features

from src.models.lightgbm_model import LightGBMModel
from src.models.lstm_model import LSTMModel
from src.models.arima_model import ARIMAModel
from src.models.stacking_model import StackingModel

from src.backtesting.backtest_engine import BacktestEngine
from src.evaluation.compare_models import build_comparison_table
from src.evaluation.regression_metrics import rmse, mae
from src.evaluation.direction_metrics import direction_accuracy


def clean_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


def run_pipeline():
    data_cfg = load_config("data")
    train_cfg = load_config("train")

    ticker = data_cfg["data"]["tickers"][0]

    df = download_ticker(ticker)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna().reset_index(drop=True)

    features = build_features(df)

    if "rsi" not in features.columns:
        raise ValueError("Feature 'rsi' is missing. Fix feature_pipeline.py")

    features = add_cluster_regimes(features)
    features = build_regime_features(features)
    features = clean_ml_features(features)

    features["target"] = features["log_return"].shift(-1)
    features = features.dropna().reset_index(drop=True)

    X = features.drop(columns=["log_return", "target"])
    y = features["target"]

    split = int(len(X) * train_cfg["training"]["train_ratio"])

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print("Training models...")
    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)

    X_train_lstm = X_train.values.reshape(len(X_train), X_train.shape[1], 1)
    X_test_lstm = X_test.values.reshape(len(X_test), X_test.shape[1], 1)

    lstm_model = LSTMModel(input_shape=(X_train.shape[1], 1))
    lstm_model.fit(X_train_lstm, y_train.values)
    lstm_pred = lstm_model.predict(X_test_lstm)

    arima_model = ARIMAModel()
    arima_model.fit(y_train)
    arima_pred = np.array(arima_model.predict(len(y_test))).reshape(-1)

    min_len = min(len(lgb_pred), len(lstm_pred), len(arima_pred), len(y_test))

    lgb_pred = lgb_pred[:min_len]
    lstm_pred = lstm_pred[:min_len]
    arima_pred = arima_pred[:min_len]
    y_test_aligned = y_test.values[:min_len]

    print("Training Stacking model...")
    meta_X = np.column_stack([lgb_pred, lstm_pred, arima_pred])

    split_meta = int(len(meta_X) * 0.8)

    meta_X_train, meta_X_test = meta_X[:split_meta], meta_X[split_meta:]
    y_train_meta, y_test_meta = y_test_aligned[:split_meta], y_test_aligned[split_meta:]

    stacker = StackingModel()
    stacker.fit(meta_X_train, y_train_meta)

    final_pred = stacker.predict(meta_X_test)

    model_rmse = rmse(y_test_meta, final_pred)
    model_mae = mae(y_test_meta, final_pred)
    model_dir_acc = direction_accuracy(y_test_meta, final_pred)

    print(f"\nPrediction Quality Metrics:")
    print(f"RMSE: {model_rmse:.6f}")
    print(f"MAE: {model_mae:.6f}")
    print(f"Direction Accuracy: {model_dir_acc:.4f}")

    print("\nRunning backtest...")
    engine = BacktestEngine()

    results_meta = engine.run(final_pred, y_test_meta)

    results_meta["rmse"] = model_rmse
    results_meta["mae"] = model_mae
    results_meta["direction_acc"] = model_dir_acc

    price_start_idx = split + split_meta
    price_end_idx = price_start_idx + len(y_test_meta) + 1

    bh_prices = df["close"].iloc[price_start_idx:price_end_idx].values

    if len(bh_prices) > 1:
        bh_returns = np.diff(np.log(bh_prices))
        bh_equity = bh_prices / bh_prices[0]

        bh_total_return = bh_prices[-1] / bh_prices[0] - 1
        bh_max_dd = np.min(
            (bh_equity - np.maximum.accumulate(bh_equity)) /
            (np.maximum.accumulate(bh_equity) + 1e-9)
        )
        bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-9) * np.sqrt(252)
    else:
        bh_total_return = 0
        bh_max_dd = 0
        bh_sharpe = np.nan

    results = {
        "MetaModel": results_meta,
        "BuyHold": {
            "sharpe": bh_sharpe,
            "max_drawdown": bh_max_dd,
            "cumulative_return": bh_total_return,
            "rmse": None,
            "mae": None,
            "direction_acc": None
        }
    }

    table = build_comparison_table(results)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(table.to_string())

    return table


if __name__ == "__main__":
    run_pipeline()
