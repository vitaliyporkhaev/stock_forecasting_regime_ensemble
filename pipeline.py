
import numpy as np
import pandas as pd

from configs.config_loader import load_config

from src.data.download_yfinance import download_ticker
from src.features.feature_pipeline import build_features
from src.regimes.clustering_regime import add_cluster_regimes
from src.regimes.regime_features import build_regime_features

from src.models.lightgbm_model import LightGBMModel
from src.models.lstm_model import LSTMModel
from src.models.arima_model import ARIMAModel
from src.models.stacking_model import StackingModel

from src.inference.signals import prediction_to_signal
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.buy_hold import buy_and_hold_returns

from src.evaluation.compare_models import build_comparison_table


def run_pipeline():

    data_cfg = load_config("data")
    train_cfg = load_config("train")

    ticker = data_cfg["data"]["tickers"][0]

    df = download_ticker(ticker)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna().reset_index(drop=True)

    features = build_features(df)

    if "rsi" not in features.columns:
        raise ValueError(
            "Feature 'rsi' is missing. Fix feature_pipeline.py (RSI must be created before regimes)."
        )

    features = add_cluster_regimes(features)
    features = build_regime_features(features)

    X = features.drop(columns=["log_return"])
    y = features["log_return"]

    split = int(len(X) * train_cfg["training"]["train_ratio"])

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]


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
    arima_pred = arima_model.predict(len(y_test))

    arima_pred = np.array(arima_pred).reshape(-1)

    min_len = min(len(lgb_pred), len(lstm_pred), len(arima_pred), len(y_test))

    lgb_pred = lgb_pred[:min_len]
    lstm_pred = lstm_pred[:min_len]
    arima_pred = arima_pred[:min_len]
    y_test = y_test.values[:min_len]

    meta_X = np.column_stack([lgb_pred, lstm_pred, arima_pred])

    stacker = StackingModel()
    stacker.fit(meta_X, y_test)

    final_pred = stacker.predict(meta_X)

    engine = BacktestEngine()

    results_meta = engine.run(final_pred, y_test)

    bh = buy_and_hold_returns(df["close"].iloc[split:split + min_len])

    results = {
        "MetaModel": results_meta,
        "BuyHold": {
            "sharpe": np.nan,
            "max_drawdown": np.min(bh),
            "return": bh[-1]
        }
    }

    table = build_comparison_table(results)

    print("\nRESULTS")
    print(table)

    return table


if __name__ == "__main__":
    run_pipeline()
