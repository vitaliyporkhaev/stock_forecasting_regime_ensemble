# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from configs.config_loader import load_config

# from src.data.download_yfinance import download_ticker
# from src.features.feature_pipeline import build_features
# from src.regimes.clustering_regime import add_cluster_regimes
# from src.regimes.regime_features import build_regime_features

# from src.models.lightgbm_model import LightGBMModel
# from src.models.lstm_model import LSTMModel
# from src.models.arima_model import ARIMAModel
# from src.models.stacking_model import StackingModel

# from src.backtesting.backtest_engine import BacktestEngine
# from src.evaluation.compare_models import build_comparison_table
# from src.evaluation.regression_metrics import rmse, mae
# from src.evaluation.direction_metrics import direction_accuracy


# def clean_ml_features(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df = df.select_dtypes(include=[np.number])
#     df = df.replace([np.inf, -np.inf], np.nan)
#     df = df.dropna().reset_index(drop=True)
#     return df


# def run_pipeline():
#     data_cfg = load_config("data")
#     train_cfg = load_config("train")

#     ticker = data_cfg["data"]["tickers"][0]

#     df = download_ticker(ticker)

#     df["log_return"] = np.log(df["close"] / df["close"].shift(1))
#     df = df.dropna().reset_index(drop=True)

#     features = build_features(df)

#     if "rsi" not in features.columns:
#         raise ValueError("Feature 'rsi' is missing. Fix feature_pipeline.py")

#     features = add_cluster_regimes(features)
#     features = build_regime_features(features)
#     features = clean_ml_features(features)

#     features["target"] = features["log_return"].shift(-1)
#     features = features.dropna().reset_index(drop=True)

#     X = features.drop(columns=["log_return", "target"])
#     y = features["target"]

#     split = int(len(X) * train_cfg["training"]["train_ratio"])

#     X_train, X_test = X.iloc[:split], X.iloc[split:]
#     y_train, y_test = y.iloc[:split], y.iloc[split:]

#     print("Training models...")
#     lgb_model = LightGBMModel()
#     lgb_model.fit(X_train, y_train)
#     lgb_pred = lgb_model.predict(X_test)

#     X_train_lstm = X_train.values.reshape(len(X_train), X_train.shape[1], 1)
#     X_test_lstm = X_test.values.reshape(len(X_test), X_test.shape[1], 1)

#     lstm_model = LSTMModel(input_shape=(X_train.shape[1], 1))
#     lstm_model.fit(X_train_lstm, y_train.values)
#     lstm_pred = lstm_model.predict(X_test_lstm)

#     arima_model = ARIMAModel()
#     arima_model.fit(y_train)
#     arima_pred = np.array(arima_model.predict(len(y_test))).reshape(-1)

#     min_len = min(len(lgb_pred), len(lstm_pred), len(arima_pred), len(y_test))

#     lgb_pred = lgb_pred[:min_len]
#     lstm_pred = lstm_pred[:min_len]
#     arima_pred = arima_pred[:min_len]
#     y_test_aligned = y_test.values[:min_len]

#     print("Training Stacking model...")
#     meta_X = np.column_stack([lgb_pred, lstm_pred, arima_pred])

#     split_meta = int(len(meta_X) * 0.8)

#     meta_X_train, meta_X_test = meta_X[:split_meta], meta_X[split_meta:]
#     y_train_meta, y_test_meta = y_test_aligned[:split_meta], y_test_aligned[split_meta:]

#     stacker = StackingModel()
#     stacker.fit(meta_X_train, y_train_meta)

#     final_pred = stacker.predict(meta_X_test)

#     model_rmse = rmse(y_test_meta, final_pred)
#     model_mae = mae(y_test_meta, final_pred)
#     model_dir_acc = direction_accuracy(y_test_meta, final_pred)

#     print(f"\nPrediction Quality Metrics:")
#     print(f"RMSE: {model_rmse:.6f}")
#     print(f"MAE: {model_mae:.6f}")
#     print(f"Direction Accuracy: {model_dir_acc:.4f}")

#     print("\nRunning backtest...")
#     engine = BacktestEngine()

#     results_meta = engine.run(final_pred, y_test_meta)

#     results_meta["rmse"] = model_rmse
#     results_meta["mae"] = model_mae
#     results_meta["direction_acc"] = model_dir_acc

#     price_start_idx = split + split_meta
#     price_end_idx = price_start_idx + len(y_test_meta) + 1

#     bh_prices = df["close"].iloc[price_start_idx:price_end_idx].values

#     if len(bh_prices) > 1:
#         bh_returns = np.diff(np.log(bh_prices))
#         bh_equity = bh_prices / bh_prices[0]

#         bh_total_return = bh_prices[-1] / bh_prices[0] - 1
#         bh_max_dd = np.min(
#             (bh_equity - np.maximum.accumulate(bh_equity)) /
#             (np.maximum.accumulate(bh_equity) + 1e-9)
#         )
#         bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-9) * np.sqrt(252)
#     else:
#         bh_total_return = 0
#         bh_max_dd = 0
#         bh_sharpe = np.nan

#     results = {
#         "MetaModel": {
#             **results_meta,
#             "return": results_meta.get("cumulative_return"),
#             "rmse": model_rmse,
#             "mae": model_mae,
#             "direction_acc": model_dir_acc
#         },
#         "BuyHold": {
#             "sharpe": bh_sharpe,
#             "max_drawdown": bh_max_dd,
#             "cumulative_return": bh_total_return,
#             "return": bh_total_return,
#             "rmse": None,
#             "mae": None,
#             "direction_acc": None
#         }
#     }

#     table = build_comparison_table(results)

#     print("\n" + "="*50)
#     print("FINAL RESULTS")
#     print("="*50)
#     print(table.to_string())

#     return table


# if __name__ == "__main__":
#     run_pipeline()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge

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

    for col in df.columns:
        if col not in ['target', 'log_return']:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df.loc[abs(df[col] - mean) > 5*std, col] = np.nan

    df = df.dropna().reset_index(drop=True)
    return df


def add_market_regime_features(df):
    df = df.copy()

    if 'volatility' not in df.columns:
        if 'log_return' in df.columns:
            df['volatility'] = df['log_return'].rolling(20).std()
        else:
            df['log_return_temp'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_return_temp'].rolling(20).std()
            df.drop('log_return_temp', axis=1, inplace=True)

    df['volatility'] = df['volatility'].fillna(df['volatility'].mean())

    try:
        df['volatility_regime'] = pd.qcut(
            df['volatility'].rank(method='first'),
            q=3,
            labels=['low', 'medium', 'high']
        )
    except:
        median_vol = df['volatility'].median()
        df['volatility_regime'] = 'medium'

    if 'close' in df.columns:
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['sma_50'] = df['sma_50'].fillna(df['close'])
        df['sma_200'] = df['sma_200'].fillna(df['close'])
        df['trend_regime'] = (df['sma_50'] > df['sma_200']).astype(int)

    if 'close' in df.columns:
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        df['momentum_60'] = df['close'] / df['close'].shift(60) - 1
        df['momentum_20'] = df['momentum_20'].fillna(0)
        df['momentum_60'] = df['momentum_60'].fillna(0)

    return df


def add_alternative_data_features(df):
    df = df.copy()

    if 'high' in df.columns and 'low' in df.columns:
        df['high_low_ratio'] = df['high'] / df['low']
    else:
        df['high_low_ratio'] = 1.0

    if 'close' in df.columns and 'open' in df.columns:
        df['close_open_ratio'] = df['close'] / df['open']
    else:
        df['close_open_ratio'] = 1.0

    if 'volume' in df.columns:
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_20'] = df['volume_ma_20'].fillna(df['volume'])
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-9)
        df['volume_trend'] = df['volume'].pct_change(5).fillna(0)
    else:
        df['volume_ratio'] = 1.0
        df['volume_trend'] = 0.0

    if 'log_return' in df.columns:
        returns = df['log_return']
    else:
        returns = np.log(df['close'] / df['close'].shift(1))

    for period in [5, 10, 20, 60]:
        df[f'volatility_{period}'] = returns.rolling(period).std()
        df[f'volatility_{period}'] = df[f'volatility_{period}'].fillna(0)

    df['vol_ratio_5_20'] = df['volatility_5'] / (df['volatility_20'] + 1e-9)
    df['vol_ratio_5_20'] = df['vol_ratio_5_20'].fillna(1.0)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def filter_trades_by_confidence(predictions, threshold=0.0005):
    signals = np.zeros_like(predictions)
    confident_idx = np.abs(predictions) > threshold
    signals[confident_idx] = np.sign(predictions[confident_idx])
    return signals


def calculate_position_size(predictions, max_position=1.0):
    confidence = np.abs(predictions)
    if np.std(confidence) > 0:
        position_size = np.clip(confidence / np.std(confidence) * 0.5, 0, max_position)
    else:
        position_size = np.ones_like(predictions) * 0.5
    return np.sign(predictions) * position_size


def run_pipeline():
    data_cfg = load_config("data")
    train_cfg = load_config("train")

    ticker = data_cfg["data"]["tickers"][0]

    df = download_ticker(ticker)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna().reset_index(drop=True)

    df = add_market_regime_features(df)
    features = build_features(df)
    features = add_alternative_data_features(features)

    if "rsi" not in features.columns:
        raise ValueError("Feature 'rsi' is missing")

    features = add_cluster_regimes(features)
    features = build_regime_features(features)
    features = clean_ml_features(features)

    features["target"] = features["log_return"].shift(-1)
    features = features.dropna().reset_index(drop=True)

    exclude_cols = ["log_return", "target", "date", "volatility_regime"]
    feature_cols = [col for col in features.columns if col not in exclude_cols]

    X = features[feature_cols]
    y = features["target"]

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    split = int(len(X) * train_cfg["training"]["train_ratio"])

    X_train, X_test = X_scaled.iloc[:split], X_scaled.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    lgb_model = LightGBMModel()
    lgb_model.model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=7,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)

    X_train_lstm = X_train.values.reshape(len(X_train), X_train.shape[1], 1)
    X_test_lstm = X_test.values.reshape(len(X_test), X_test.shape[1], 1)

    lstm_model = LSTMModel(input_shape=(X_train.shape[1], 1))
    lstm_model.fit(X_train_lstm, y_train.values, epochs=20, batch_size=64)
    lstm_pred = lstm_model.predict(X_test_lstm)

    arima_model = ARIMAModel(order=(2, 1, 2))
    arima_model.fit(y_train)
    arima_pred = np.array(arima_model.predict(len(y_test))).reshape(-1)

    min_len = min(len(lgb_pred), len(lstm_pred), len(arima_pred), len(y_test))

    lgb_pred = lgb_pred[:min_len]
    lstm_pred = lstm_pred[:min_len]
    arima_pred = arima_pred[:min_len]
    y_test_aligned = y_test.values[:min_len]

    meta_X = np.column_stack([lgb_pred, lstm_pred, arima_pred])

    split_meta = int(len(meta_X) * 0.8)

    meta_X_train, meta_X_test = meta_X[:split_meta], meta_X[split_meta:]
    y_train_meta, y_test_meta = y_test_aligned[:split_meta], y_test_aligned[split_meta:]

    stacker = StackingModel()
    stacker.meta_model = Ridge(alpha=1.0)
    stacker.fit(meta_X_train, y_train_meta)

    final_pred = stacker.predict(meta_X_test)


    signals = filter_trades_by_confidence(final_pred, threshold=0.0003)
    position_sizes = calculate_position_size(final_pred)
    combined_signals = signals * position_sizes / (np.abs(signals) + 1e-9)

    strategy_returns = combined_signals * y_test_meta

    print("⚙️ Бэктестинг...")
    engine = BacktestEngine()
    results_meta = engine.run(final_pred, y_test_meta)

    equity_curve = np.cumprod(1 + strategy_returns + 1e-9)
    results_meta["cumulative_return"] = equity_curve[-1] - 1 if len(equity_curve) > 0 else 0
    results_meta["sharpe"] = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)

    model_rmse = rmse(y_test_meta, final_pred)
    model_mae = mae(y_test_meta, final_pred)
    model_dir_acc = direction_accuracy(y_test_meta, final_pred)

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
        bh_total_return = bh_max_dd = bh_sharpe = 0

    results = {
        "MetaModel": {**results_meta, "return": results_meta["cumulative_return"]},
        "BuyHold": {
            "sharpe": bh_sharpe,
            "max_drawdown": bh_max_dd,
            "cumulative_return": bh_total_return,
            "return": bh_total_return,
            "rmse": None,
            "mae": None,
            "direction_acc": None
        }
    }

    table = build_comparison_table(results)

    print("\n" )
    print("Финальные результаты")
    print(table.to_string())

    print("\nСоздание графиков")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axes[0, 0]
    ax1.plot(equity_curve, label='Strategy', linewidth=1.5)
    ax1.plot(bh_equity, label='Buy & Hold', alpha=0.7, linewidth=1.5)
    ax1.set_title('Equity Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Equity')

    ax2 = axes[0, 1]
    ax2.hist(strategy_returns, bins=50, alpha=0.7, label='Strategy')
    ax2.hist(bh_returns, bins=50, alpha=0.5, label='Buy&Hold')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Returns Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / (running_max + 1e-9)
    ax3.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
    ax3.set_title('Strategy Drawdown')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Drawdown')

    ax4 = axes[1, 1]
    sample_size = min(200, len(final_pred))
    ax4.plot(final_pred[:sample_size], label='Predictions', alpha=0.7)
    ax4.plot(y_test_meta[:sample_size], label='Actual', alpha=0.7)
    ax4.set_title(f'Predictions vs Actual (first {sample_size} points)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('strategy_analysis.png', dpi=150, bbox_inches='tight')
    print("Графики сохранены в strategy_analysis.png")
    plt.close()

    # Дополнительная аналитика
    print("\nДоп. аналитика:")
    print(f"Модель переиграла рынок на: {(results_meta['cumulative_return'] - bh_total_return)*100:.2f}%")
    print(f"Количество сделок: {np.sum(np.diff(signals) != 0)}")
    print(f"Процент времени в рынке: {np.mean(np.abs(signals) > 0)*100:.1f}%")
    print(f"Средняя доходность на сделку: {np.mean(strategy_returns[signals != 0])*100:.3f}%")

    return table


if __name__ == "__main__":
    run_pipeline()
