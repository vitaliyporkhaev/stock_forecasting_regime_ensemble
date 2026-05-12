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

from src.backtesting.backtest_engine import BacktestEngine
from src.evaluation.compare_models import build_comparison_table


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

    X = features.drop(columns=["log_return"])
    y = features["log_return"]

    split = int(len(X) * train_cfg["training"]["train_ratio"])

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]


    print("Training LightGBM...")
    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)

    print("Training LSTM...")
    X_train_lstm = X_train.values.reshape(len(X_train), X_train.shape[1], 1)
    X_test_lstm = X_test.values.reshape(len(X_test), X_test.shape[1], 1)

    lstm_model = LSTMModel(input_shape=(X_train.shape[1], 1))
    lstm_model.fit(X_train_lstm, y_train.values)
    lstm_pred = lstm_model.predict(X_test_lstm)

    print("Training ARIMA...")
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

    print("Running backtest...")
    engine = BacktestEngine()


    results_meta = engine.run(final_pred, y_test_meta)


    price_start_idx = split + split_meta
    price_end_idx = price_start_idx + len(y_test_meta)

    bh_prices = df["close"].iloc[price_start_idx:price_end_idx].values

    bh_returns = np.diff(np.log(bh_prices))
    bh_equity = bh_prices / bh_prices[0]

    bh_total_return = bh_prices[-1] / bh_prices[0] - 1
    bh_max_dd = np.min(
        (bh_equity - np.maximum.accumulate(bh_equity)) /
        (np.maximum.accumulate(bh_equity) + 1e-9)
    )
    bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-9) * np.sqrt(252) if len(bh_returns) > 0 else np.nan

    results = {
        "MetaModel": results_meta,
        "BuyHold": {
            "sharpe": bh_sharpe,
            "max_drawdown": bh_max_dd,
            "cumulative_return": bh_total_return
        }
    }

    table = build_comparison_table(results)

    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(table)

    print("\nDebug info:")
    print(f"MetaModel predictions shape: {final_pred.shape}")
    print(f"MetaModel predictions range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
    print(f"Test returns shape: {y_test_meta.shape}")
    print(f"MetaModel Sharpe: {results_meta.get('sharpe', 'N/A')}")
    print(f"MetaModel returns mean: {np.mean(results_meta['returns']):.6f}")
    print(f"BuyHold prices: {len(bh_prices)} points")
    print(f"BuyHold first price: {bh_prices[0]:.2f}, last price: {bh_prices[-1]:.2f}")

    return table


if __name__ == "__main__":
    run_pipeline()
