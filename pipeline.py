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

    print("\n" + "="*50)
    print("DIAGNOSTICS")
    print("="*50)
    print(f"Predictions mean: {np.mean(final_pred):.6f}")
    print(f"Predictions std: {np.std(final_pred):.6f}")
    print(f"Real returns mean: {np.mean(y_test_meta):.6f}")
    print(f"Real returns std: {np.std(y_test_meta):.6f}")

    correlation = np.corrcoef(final_pred, y_test_meta)[0, 1]
    print(f"Correlation(pred, actual): {correlation:.4f}")

    direction_match = np.mean(np.sign(final_pred) == np.sign(y_test_meta))
    print(f"Direction accuracy: {direction_match:.4f}")

    print("\nRunning backtest...")
    engine = BacktestEngine()

    results_meta = engine.run(final_pred, y_test_meta)

    print(f"\nStrategy returns stats:")
    print(f"Mean: {np.mean(results_meta['returns']):.6f}")
    print(f"Std: {np.std(results_meta['returns']):.6f}")
    print(f"Min: {np.min(results_meta['returns']):.6f}")
    print(f"Max: {np.max(results_meta['returns']):.6f}")
    print(f"Number of trades: {np.sum(np.diff(results_meta['positions']) != 0)}")

    price_start_idx = split + split_meta
    price_end_idx = price_start_idx + len(y_test_meta) + 1  # +1 потому что нужны цены для расчета всех доходностей

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
            "cumulative_return": bh_total_return
        }
    }

    table = build_comparison_table(results)

    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(table)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(final_pred[:100], label='Predictions', alpha=0.7)
    axes[0].plot(y_test_meta[:100], label='Actual returns', alpha=0.7)
    axes[0].set_title('Predictions vs Actual (first 100 points)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(results_meta['equity_curve'], label='Strategy Equity')
    axes[1].plot(bh_equity[:len(results_meta['equity_curve'])], label='Buy&Hold Equity', alpha=0.7)
    axes[1].set_title('Equity Curves')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(results_meta['returns'][:100], label='Strategy Returns', alpha=0.7)
    axes[2].set_title('Strategy Returns (first 100 points)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    return table


if __name__ == "__main__":
    run_pipeline()
