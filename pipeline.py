import numpy as np
import pandas as pd

from configs.config_loader import load_config

from src.data.download_yfinance import download_ticker_safe
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

    # ======================
    # CONFIG
    # ======================
    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/model.yaml")
    train_cfg = load_config("configs/train.yaml")
    bt_cfg = load_config("configs/backtest.yaml")

    ticker = data_cfg["data"]["tickers"][0]

    # ======================
    # DATA
    # ======================
    df = download_ticker_safe(ticker)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna()

    # ======================
    # FEATURES
    # ======================
    features = build_features(df)
    features = add_cluster_regimes(features)
    features = build_regime_features(features)

    X = features.drop(columns=["log_return"])
    y = features["log_return"]

    split = int(len(X) * train_cfg["training"]["train_ratio"])

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # ======================
    # MODELS
    # ======================

    # LightGBM
    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)

    # LSTM (simplified placeholder)
    lstm_model = LSTMModel(input_shape=(X_train.shape[1], 1))
    lstm_model.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train.values)
    lstm_pred = lstm_model.predict(X_test.values.reshape(-1, X_test.shape[1], 1))

    # ARIMA baseline
    arima_model = ARIMAModel()
    arima_model.fit(y_train)
    arima_pred = arima_model.predict(len(y_test))

    # ======================
    # STACKING
    # ======================
    meta_X = np.vstack([lgb_pred, lstm_pred, arima_pred]).T

    stacker = StackingModel()
    stacker.fit(meta_X, y_test.values)

    final_pred = stacker.predict(meta_X)

    # ======================
    # BACKTEST
    # ======================
    engine = BacktestEngine()

    results_meta = engine.run(final_pred, y_test.values)

    # Buy & Hold benchmark
    bh = buy_and_hold_returns(df["close"].iloc[split:])

    # ======================
    # METRICS TABLE
    # ======================
    results = {
        "MetaModel": results_meta,
        "BuyHold": {
            "sharpe": np.nan,
            "max_drawdown": np.min(bh),
            "return": bh[-1]
        }
    }

    table = build_comparison_table(results)

    print("\n=== RESULTS ===")
    print(table)

    return table


if __name__ == "__main__":
    run_pipeline()
