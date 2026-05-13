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
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

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


def percentile_strategy(predictions, long_pct=60, short_pct=40):
    long_threshold = np.percentile(predictions, long_pct)
    short_threshold = np.percentile(predictions, short_pct)

    signals = np.zeros(len(predictions))
    signals[predictions > long_threshold] = 1
    signals[predictions < short_threshold] = -1

    return signals


def momentum_strategy(predictions, window=5):
    pred_series = pd.Series(predictions)
    pred_ma = pred_series.rolling(window).mean()
    pred_ma = pred_ma.fillna(pred_series.median())

    signals = np.zeros(len(predictions))
    signals[predictions > pred_ma.values] = 1
    signals[predictions <= pred_ma.values] = -1

    return signals

def calculate_dynamic_position_size(confidence, volatility, max_position=1.0, risk_target=0.02):

    norm_confidence = np.abs(confidence) / (np.percentile(np.abs(confidence), 90) + 1e-9)
    norm_confidence = np.clip(norm_confidence, 0.1, 1.0)

    vol_scalar = risk_target / (volatility * 100 + 1e-9)
    vol_scalar = np.clip(vol_scalar, 0.1, 1.0)

    position_size = norm_confidence * vol_scalar
    position_size = np.clip(position_size, 0, max_position)

    return position_size


def apply_stop_loss(signals, returns, stop_loss_pct=0.02):

    modified_signals = signals.copy()
    cumulative_pnl = 0
    entry_price = 0
    in_position = False

    for i in range(len(signals)):
        if signals[i] != 0 and not in_position:
            in_position = True
            entry_price = 1
            cumulative_pnl = 0
        elif signals[i] == 0 and in_position:
            in_position = False
            cumulative_pnl = 0

        if in_position:
            cumulative_pnl += returns[i]

            if cumulative_pnl < -stop_loss_pct:
                modified_signals[i] = 0
                in_position = False
                cumulative_pnl = 0

    return modified_signals


def apply_take_profit(signals, returns, take_profit_pct=0.05):

    modified_signals = signals.copy()
    cumulative_pnl = 0
    in_position = False

    for i in range(len(signals)):
        if signals[i] != 0 and not in_position:
            in_position = True
            cumulative_pnl = 0
        elif signals[i] == 0 and in_position:
            in_position = False
            cumulative_pnl = 0

        if in_position:
            cumulative_pnl += returns[i]

            if cumulative_pnl > take_profit_pct:
                modified_signals[i] = 0
                in_position = False
                cumulative_pnl = 0

    return modified_signals


def apply_trailing_stop(signals, returns, trailing_pct=0.03):

    modified_signals = signals.copy()
    cumulative_pnl = 0
    max_pnl = 0
    in_position = False

    for i in range(len(signals)):
        if signals[i] != 0 and not in_position:
            in_position = True
            cumulative_pnl = 0
            max_pnl = 0
        elif signals[i] == 0 and in_position:
            in_position = False
            cumulative_pnl = 0
            max_pnl = 0

        if in_position:
            cumulative_pnl += returns[i]
            max_pnl = max(max_pnl, cumulative_pnl)

            if cumulative_pnl < max_pnl - trailing_pct:
                modified_signals[i] = 0
                in_position = False
                cumulative_pnl = 0
                max_pnl = 0

    return modified_signals


def calculate_var(returns, confidence=0.95, window=20):

    rolling_var = pd.Series(returns).rolling(window).quantile(1 - confidence)
    return rolling_var.fillna(0).values


def calculate_expected_shortfall(returns, confidence=0.95, window=20):

    es = []
    for i in range(window, len(returns) + 1):
        window_returns = returns[max(0, i-window):i]
        var_threshold = np.percentile(window_returns, (1 - confidence) * 100)
        tail_losses = window_returns[window_returns <= var_threshold]
        es.append(np.mean(tail_losses) if len(tail_losses) > 0 else 0)

    return np.array(es)


def apply_risk_filters(predictions, returns, volatility,
                       max_drawdown_limit=0.15, var_limit=0.03):

    signals = np.sign(predictions)
    modified_signals = signals.copy()

    cumulative = np.cumprod(1 + returns + 1e-9)
    running_max = np.maximum.accumulate(cumulative)
    current_drawdown = (cumulative - running_max) / (running_max + 1e-9)

    var_95 = calculate_var(returns, confidence=0.95, window=20)

    for i in range(1, len(signals)):
        if current_drawdown[i] < -max_drawdown_limit:
            modified_signals[i] = 0

        if i < len(var_95) and abs(var_95[i]) > var_limit:
            modified_signals[i] = 0

        if volatility[i] > np.percentile(volatility[:i+1], 90):
            modified_signals[i] = 0

    return modified_signals


def apply_kelly_criterion(win_rate, avg_win, avg_loss):

    if avg_loss == 0:
        return 0

    b = abs(avg_win / avg_loss) if avg_loss != 0 else 1
    p = win_rate

    kelly_fraction = (p * b - (1 - p)) / b

    return np.clip(kelly_fraction * 0.5, 0, 0.25)


def apply_risk_management(final_pred, y_test_meta, base_signals,
                          volatility_window=20):

    returns = base_signals * y_test_meta

    volatility = pd.Series(y_test_meta).rolling(volatility_window).std().fillna(0).values

    win_rate = np.mean(returns > 0)
    avg_win = np.mean(returns[returns > 0]) if np.sum(returns > 0) > 0 else 0.01
    avg_loss = abs(np.mean(returns[returns < 0])) if np.sum(returns < 0) > 0 else 0.01
    kelly_size = apply_kelly_criterion(win_rate, avg_win, avg_loss)

    position_sizes = calculate_dynamic_position_size(
        final_pred, volatility, max_position=kelly_size * 2
    )

    signals_with_stops = apply_stop_loss(base_signals, returns, stop_loss_pct=0.02)
    signals_with_stops = apply_take_profit(signals_with_stops, returns, take_profit_pct=0.05)
    signals_with_stops = apply_trailing_stop(signals_with_stops, returns, trailing_pct=0.03)

    signals_filtered = apply_risk_filters(
        final_pred, returns, volatility,
        max_drawdown_limit=0.15, var_limit=0.03
    )

    final_signals = signals_filtered * signals_with_stops / (np.abs(signals_with_stops) + 1e-9)
    final_signals = final_signals * position_sizes

    final_signals = np.clip(final_signals, -1.0, 1.0)

    return final_signals, {
        'kelly_size': kelly_size,
        'avg_position': np.mean(np.abs(final_signals)),
        'stop_loss_triggers': np.sum(signals_with_stops != base_signals),
        'filter_triggers': np.sum(signals_filtered != base_signals),
        'max_position': np.max(np.abs(final_signals))
    }

def run_pipeline_for_ticker_with_risk_management(ticker, data_cfg, train_cfg):
    print(f"\n")
    print(f"Обработка тикера: {ticker}")

    df = download_ticker(ticker)
    print(f"   Загружено {len(df)} строк")

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
        n_estimators=300, learning_rate=0.03, num_leaves=63, max_depth=7,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    print(f"      LightGBM: pred std={np.std(lgb_pred):.6f}")

    X_train_lstm = X_train.values.reshape(len(X_train), X_train.shape[1], 1)
    X_test_lstm = X_test.values.reshape(len(X_test), X_test.shape[1], 1)

    lstm_model = LSTMModel(input_shape=(X_train.shape[1], 1))
    lstm_model.fit(X_train_lstm, y_train.values, epochs=10, batch_size=64)
    lstm_pred = lstm_model.predict(X_test_lstm)
    print(f"      LSTM: pred std={np.std(lstm_pred):.6f}")

    arima_model = ARIMAModel(order=(2, 1, 2))
    arima_model.fit(y_train)
    arima_pred = np.array(arima_model.predict(len(y_test))).reshape(-1)
    print(f"      ARIMA: pred std={np.std(arima_pred):.6f}")

    min_len = min(len(lgb_pred), len(lstm_pred), len(arima_pred), len(y_test))
    lgb_pred = lgb_pred[:min_len]
    lstm_pred = lstm_pred[:min_len]
    arima_pred = arima_pred[:min_len]
    y_test_aligned = y_test.values[:min_len]

    meta_X = np.column_stack([lgb_pred, lstm_pred, arima_pred])
    meta_scaler = StandardScaler()
    meta_X_scaled = meta_scaler.fit_transform(meta_X)

    split_meta = int(len(meta_X_scaled) * 0.8)
    meta_X_train = meta_X_scaled[:split_meta]
    meta_X_test = meta_X_scaled[split_meta:]
    y_train_meta = y_test_aligned[:split_meta]
    y_test_meta = y_test_aligned[split_meta:]

    stacker = StackingModel()
    stacker.meta_model = RandomForestRegressor(
        n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=-1
    )
    stacker.fit(meta_X_train, y_train_meta)

    final_pred = stacker.predict(meta_X_test)
    final_pred = (final_pred - np.mean(final_pred)) / (np.std(final_pred) + 1e-9)
    print(f"      Stack: pred std={np.std(final_pred):.6f}, positive={np.mean(final_pred > 0)*100:.1f}%")

    train_cumret = np.prod(1 + y_train_meta) - 1
    test_cumret = np.prod(1 + y_test_meta) - 1

    price_start_idx = split + split_meta
    price_end_idx = price_start_idx + len(y_test_meta) + 1
    bh_prices = df["close"].iloc[price_start_idx:price_end_idx].values

    if len(bh_prices) > 1:
        bh_returns = np.diff(np.log(bh_prices))
        bh_equity = bh_prices / bh_prices[0]
        bh_total_return = bh_prices[-1] / bh_prices[0] - 1
        bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-9) * np.sqrt(252)
        bh_max_dd = np.min(
            (bh_equity - np.maximum.accumulate(bh_equity)) /
            (np.maximum.accumulate(bh_equity) + 1e-9)
        )
    else:
        bh_total_return = bh_sharpe = bh_max_dd = 0

    market_trend = "UP" if bh_total_return > 0 else "DOWN"

    train_avg = np.mean(y_train_meta) * 100
    test_avg = np.mean(y_test_meta) * 100

    print(f"Анализ рынка:")
    print(f"      Train: avg={train_avg:+.3f}%/день, B&H={train_cumret*100:+.2f}%")
    print(f"      Test:  avg={test_avg:+.3f}%/день, B&H={test_cumret*100:+.2f}%")
    print(f"      B&H:   {bh_total_return*100:+.2f}% → {market_trend}")

    strategies = {
        'Percentile L/S (70/30)': lambda p: percentile_strategy(p, 70, 30),
        'Percentile L/S (60/40)': lambda p: percentile_strategy(p, 60, 40),
        'Long only (60/0)': lambda p: np.where(p > np.percentile(p, 40), 1, 0),
        'Long only (70/0)': lambda p: np.where(p > np.percentile(p, 30), 1, 0),
        'Long-biased (80/20)': lambda p: np.where(p > np.percentile(p, 20), 1, 0),
        'Short only (50)': lambda p: np.where(p < np.percentile(p, 50), -1, 0),
        'Short only (30)': lambda p: np.where(p < np.percentile(p, 30), -1, 0),
        'Short-biased': lambda p: np.where(p < np.percentile(p, 70), -1, 1),
        'Momentum L/S (5)': lambda p: momentum_strategy(p, 5),
        'Momentum L/S (10)': lambda p: momentum_strategy(p, 10),
        'Always Long': lambda p: np.ones_like(p),
        'Always Short': lambda p: -np.ones_like(p),
    }

    print(f"\nТестирование {len(strategies)} стратегий:")

    best_sharpe = -np.inf
    best_signals = None
    best_name = ""
    best_strategy_returns = None

    for name, strategy_func in strategies.items():
        try:
            signals = strategy_func(final_pred)
            ret = signals * y_test_meta

            if len(ret) == 0 or np.std(ret) == 0:
                continue

            sharpe = np.mean(ret) / (np.std(ret) + 1e-9) * np.sqrt(252)
            trades = np.sum(np.diff(signals) != 0) // 2

            adjusted_sharpe = sharpe - (0.5 if trades == 0 else 0)

            if adjusted_sharpe > best_sharpe:
                best_sharpe = adjusted_sharpe
                best_signals = signals.copy()
                best_name = name
                best_strategy_returns = ret.copy()
        except:
            continue

    if best_signals is None:
        best_signals = np.ones_like(final_pred)
        best_name = 'Always Long (fallback)'
        best_strategy_returns = best_signals * y_test_meta

    print(f"Базовая стратегия: {best_name}")

    print(f"\nПрименение риск-менеджмента")

    base_ret = best_strategy_returns
    win_rate = np.mean(base_ret > 0)
    avg_win = np.mean(base_ret[base_ret > 0]) if np.sum(base_ret > 0) > 0 else 0.01
    avg_loss = abs(np.mean(base_ret[base_ret < 0])) if np.sum(base_ret < 0) > 0 else 0.01

    kelly_size = apply_kelly_criterion(win_rate, avg_win, avg_loss)

    volatility = pd.Series(y_test_meta).rolling(20).std().fillna(np.std(y_test_meta)).values

    position_sizes = calculate_dynamic_position_size(
        final_pred, volatility, max_position=kelly_size * 2, risk_target=0.02
    )

    signals_sl = apply_stop_loss(best_signals, base_ret, stop_loss_pct=0.02)

    signals_tp = apply_take_profit(signals_sl, base_ret, take_profit_pct=0.05)

    signals_ts = apply_trailing_stop(signals_tp, base_ret, trailing_pct=0.03)

    signals_filtered = apply_risk_filters(
        final_pred, base_ret, volatility,
        max_drawdown_limit=0.15, var_limit=0.03
    )

    managed_signals = signals_filtered.copy()
    stop_days = (signals_ts != best_signals)
    managed_signals[stop_days] = signals_ts[stop_days]

    managed_signals = managed_signals * position_sizes

    managed_signals = np.clip(managed_signals, -1.0, 1.0)

    managed_returns = managed_signals * y_test_meta

    transaction_cost = 0.001
    trades_mask = np.diff(managed_signals, prepend=0) != 0
    managed_returns[trades_mask] -= transaction_cost * np.abs(managed_signals[trades_mask])

    stop_loss_count = np.sum(signals_sl != best_signals)
    take_profit_count = np.sum(signals_tp != signals_sl)
    trailing_stop_count = np.sum(signals_ts != signals_tp)
    filter_count = np.sum(signals_filtered != best_signals)

    base_cumret = np.prod(1 + base_ret + 1e-9) - 1
    managed_cumret = np.prod(1 + managed_returns + 1e-9) - 1

    base_sharpe = np.mean(base_ret) / (np.std(base_ret) + 1e-9) * np.sqrt(252)
    managed_sharpe = np.mean(managed_returns) / (np.std(managed_returns) + 1e-9) * np.sqrt(252)

    base_equity = np.cumprod(1 + base_ret + 1e-9)
    managed_equity = np.cumprod(1 + managed_returns + 1e-9)

    base_max_dd = np.min((base_equity - np.maximum.accumulate(base_equity)) /
                         (np.maximum.accumulate(base_equity) + 1e-9))
    managed_max_dd = np.min((managed_equity - np.maximum.accumulate(managed_equity)) /
                            (np.maximum.accumulate(managed_equity) + 1e-9))

    base_trades = np.sum(np.diff(best_signals) != 0) // 2
    managed_trades = np.sum(np.diff(managed_signals) != 0) // 2 + stop_loss_count + take_profit_count + trailing_stop_count

    managed_vol = np.std(managed_returns) * np.sqrt(252)
    base_vol = np.std(base_ret) * np.sqrt(252)

    print(f"\n Сравнение риск-метрик:")
    print(f"   {'Метрика':<30s} {'Базовая':>12s} {'С Risk Mgmt':>12s} {'Изменение':>12s}")
    print(f"   {'-'*70}")
    print(f"   {'Доходность':<30s} {base_cumret*100:>11.2f}% {managed_cumret*100:>11.2f}% {(managed_cumret-base_cumret)*100:>+11.2f}%")
    print(f"   {'Sharpe Ratio':<30s} {base_sharpe:>12.3f} {managed_sharpe:>12.3f} {managed_sharpe-base_sharpe:>+12.3f}")
    print(f"   {'Волатильность (год)':<30s} {base_vol*100:>11.1f}% {managed_vol*100:>11.1f}% {(managed_vol-base_vol)*100:>+11.1f}%")
    print(f"   {'Max Drawdown':<30s} {base_max_dd*100:>11.2f}% {managed_max_dd*100:>11.2f}% {(managed_max_dd-base_max_dd)*100:>+11.2f}%")
    print(f"   {'Количество сделок':<30s} {base_trades:>12d} {managed_trades:>12d} {managed_trades-base_trades:>+12d}")
    print(f"   {'Средняя позиция':<30s} {np.mean(np.abs(best_signals))*100:>11.1f}% {np.mean(np.abs(managed_signals))*100:>11.1f}%")
    print(f"   {'Келли размер':<30s} {'-':>12s} {kelly_size*100:>11.1f}%")
    print(f"   {'Стоп-лоссы':<30s} {'-':>12s} {stop_loss_count:>12d}")
    print(f"   {'Тейк-профиты':<30s} {'-':>12s} {take_profit_count:>12d}")
    print(f"   {'Трейлинг-стопы':<30s} {'-':>12s} {trailing_stop_count:>12d}")
    print(f"   {'Риск-фильтры':<30s} {'-':>12s} {filter_count:>12d}")

    long_exposure = np.mean(managed_signals > 0) * 100
    short_exposure = np.mean(managed_signals < 0) * 100
    flat_exposure = np.mean(managed_signals == 0) * 100

    long_returns = managed_returns[managed_signals > 0]
    short_returns = managed_returns[managed_signals < 0]

    long_avg = np.mean(long_returns) * 100 if len(long_returns) > 0 else 0
    short_avg = np.mean(short_returns) * 100 if len(short_returns) > 0 else 0
    long_wr = np.mean(long_returns > 0) * 100 if len(long_returns) > 0 else 0
    short_wr = np.mean(short_returns > 0) * 100 if len(short_returns) > 0 else 0

    print(f"\n   Статистика позиций с риск-менеджментом:")
    print(f"      Long:  {long_exposure:.0f}% (avg={long_avg:+.3f}%/день, WR={long_wr:.0f}%)")
    print(f"      Short: {short_exposure:.0f}% (avg={short_avg:+.3f}%/день, WR={short_wr:.0f}%)")
    print(f"      Flat:  {flat_exposure:.0f}%")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax1 = axes[0, 0]
    ax1.plot(base_equity, label='Базовая стратегия', alpha=0.7, linewidth=1.5)
    ax1.plot(managed_equity, label='С Risk Mgmt', linewidth=2, color='green')
    ax1.plot(bh_equity, label='Buy & Hold', alpha=0.5, linewidth=1, color='gray')
    ax1.set_title('Кривые капитала', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='black', linestyle='-', alpha=0.2)

    ax2 = axes[0, 1]
    base_dd_curve = (base_equity - np.maximum.accumulate(base_equity)) / (np.maximum.accumulate(base_equity) + 1e-9)
    managed_dd_curve = (managed_equity - np.maximum.accumulate(managed_equity)) / (np.maximum.accumulate(managed_equity) + 1e-9)
    ax2.fill_between(range(len(base_dd_curve)), base_dd_curve, 0, alpha=0.3, color='blue', label='Базовая')
    ax2.fill_between(range(len(managed_dd_curve)), managed_dd_curve, 0, alpha=0.4, color='green', label='С Risk Mgmt')
    ax2.set_title('Просадки', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(position_sizes, label='Размер позиции', alpha=0.7, color='purple')
    ax3.axhline(y=np.mean(position_sizes), color='red', linestyle='--', label=f'Средняя: {np.mean(position_sizes)*100:.1f}%')
    ax3.set_title('Динамический размер позиции', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('% капитала')

    ax4 = axes[1, 1]
    ax4.hist(base_ret, bins=50, alpha=0.5, label='Базовая', color='blue')
    ax4.hist(managed_returns, bins=50, alpha=0.5, label='С Risk Mgmt', color='green')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_title('Распределение доходностей', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    info_text = (
        f"{ticker} | Стратегия: {best_name}\n"
        f"Базовая: {base_cumret*100:+.2f}%, Sharpe: {base_sharpe:.2f}, DD: {base_max_dd*100:.2f}%\n"
        f"С RM:    {managed_cumret*100:+.2f}%, Sharpe: {managed_sharpe:.2f}, DD: {managed_max_dd*100:.2f}%\n"
        f"Келли: {kelly_size*100:.1f}% | Стопы: {stop_loss_count} | TP: {take_profit_count}"
    )

    plt.figtext(0.02, 0.02, info_text, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'risk_management_{ticker}.png', dpi=150, bbox_inches='tight')
    print(f"Графики сохранены в risk_management_{ticker}.png")
    plt.close()

    return {
        'ticker': ticker,
        'base_return': base_cumret,
        'base_sharpe': base_sharpe,
        'base_max_dd': base_max_dd,
        'base_trades': base_trades,
        'managed_return': managed_cumret,
        'managed_sharpe': managed_sharpe,
        'managed_max_dd': managed_max_dd,
        'managed_trades': managed_trades,
        'managed_vol': managed_vol,
        'alpha': managed_cumret - bh_total_return,
        'sharpe_improvement': managed_sharpe - base_sharpe,
        'dd_improvement': base_max_dd - managed_max_dd,
        'kelly_size': kelly_size,
        'stop_loss_count': stop_loss_count,
        'take_profit_count': take_profit_count,
        'trailing_stop_count': trailing_stop_count,
        'avg_position': np.mean(np.abs(managed_signals)),
        'bh_return': bh_total_return,
        'bh_sharpe': bh_sharpe,
        'market': market_trend,
        'strategy': best_name,
        'long_pct': long_exposure,
        'short_pct': short_exposure,
        'flat_pct': flat_exposure
    }

def run_pipeline():
    print("Запуск торгового пайплайна с риск-менеджментом")

    data_cfg = load_config("data")
    train_cfg = load_config("train")

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    print(f"\nТестируемые тикеры: {', '.join(tickers)}")

    all_results = []

    for ticker in tickers:
        try:
            result = run_pipeline_for_ticker_with_risk_management(ticker, data_cfg, train_cfg)
            all_results.append(result)
        except Exception as e:
            print(f"Ошибка для {ticker}: {str(e)[:150]}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\nНет результатов для отображения")
        return None

    results_df = pd.DataFrame(all_results)

    print("\n")
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ: БАЗОВАЯ СТРАТЕГИЯ vs РИСК-МЕНЕДЖМЕНТ")

    print("\nБАЗОВАЯ СТРАТЕГИЯ (без риск-менеджмента):")
    print(f"   {'Ticker':<8s} {'Return':>10s} {'Sharpe':>8s} {'Max DD':>8s} {'Trades':>8s} {'B&H Ret':>10s} {'Alpha':>10s} {'Market':>8s} {'Strategy':<20s}")
    print(f"   {'-'*95}")

    for _, row in results_df.iterrows():
        print(f"   {row['ticker']:<8s} {row['base_return']*100:>9.2f}% {row['base_sharpe']:>8.3f} "
              f"{row['base_max_dd']*100:>7.2f}% {row['base_trades']:>8d} "
              f"{row['bh_return']*100:>9.2f}% {(row['base_return']-row['bh_return'])*100:>9.2f}% "
              f"{row['market']:>8s} {row['strategy']:<20s}")

    print(f"\n РИСК-МЕНЕДЖМЕНТОМ:")
    print(f"   {'Ticker':<8s} {'Return':>10s} {'Sharpe':>8s} {'Max DD':>8s} {'Trades':>8s} {'Vol':>8s} {'StopLoss':>10s} {'TakeProf':>10s} {'Avg Pos':>8s} {'Kelly':>8s}")
    print(f"   {'-'*100}")

    for _, row in results_df.iterrows():
        print(f"   {row['ticker']:<8s} {row['managed_return']*100:>9.2f}% {row['managed_sharpe']:>8.3f} "
              f"{row['managed_max_dd']*100:>7.2f}% {row['managed_trades']:>8d} "
              f"{row['managed_vol']*100:>7.1f}% {row['stop_loss_count']:>10d} "
              f"{row['take_profit_count']:>10d} {row['avg_position']*100:>7.1f}% {row['kelly_size']*100:>7.1f}%")

    print(f"\nУлучшения от риск-менеджмента:")
    print(f"   {'Ticker':<8s} {'Δ Return':>10s} {'Δ Sharpe':>10s} {'Δ Max DD':>10s} {'Δ Trades':>10s} {'Alpha (vs B&H)':>15s}")
    print(f"   {'-'*70}")

    for _, row in results_df.iterrows():
        delta_return = (row['managed_return'] - row['base_return']) * 100
        delta_sharpe = row['sharpe_improvement']
        delta_dd = row['dd_improvement'] * 100
        delta_trades = row['managed_trades'] - row['base_trades']
        alpha = row['alpha'] * 100

        print(f"   {row['ticker']:<8s} {delta_return:>+9.2f}% {delta_sharpe:>+10.3f} "
              f"{delta_dd:>+9.2f}% {delta_trades:>+10d} {alpha:>+14.2f}%")

    print("\n")
    print("Агрегированная статистика")

    base_win_rate = np.mean((results_df['base_return'] - results_df['bh_return']) > 0) * 100
    base_avg_alpha = np.mean(results_df['base_return'] - results_df['bh_return']) * 100
    base_avg_sharpe = np.mean(results_df['base_sharpe'])
    base_avg_dd = np.mean(results_df['base_max_dd']) * 100

    managed_win_rate = np.mean(results_df['alpha'] > 0) * 100
    managed_avg_alpha = np.mean(results_df['alpha']) * 100
    managed_avg_sharpe = np.mean(results_df['managed_sharpe'])
    managed_avg_dd = np.mean(results_df['managed_max_dd']) * 100

    avg_sharpe_improvement = np.mean(results_df['sharpe_improvement'])
    avg_dd_improvement = np.mean(results_df['dd_improvement']) * 100
    avg_return_change = np.mean(results_df['managed_return'] - results_df['base_return']) * 100

    sharpe_improved_pct = np.mean(results_df['sharpe_improvement'] > 0) * 100
    dd_improved_pct = np.mean(results_df['dd_improvement'] > 0) * 100

    print(f"\nСРАВНЕНИЕ БАЗОВАЯ vs РИСК-МЕНЕДЖМЕНТ:")
    print(f"   {'Метрика':<35s} {'Базовая':>12s} {'С Risk Mgmt':>12s} {'Изменение':>12s}")
    print(f"   {'-'*75}")
    print(f"   {'Win Rate vs Buy&Hold':<35s} {base_win_rate:>11.1f}% {managed_win_rate:>11.1f}% {managed_win_rate-base_win_rate:>+11.1f}%")
    print(f"   {'Средняя Альфа':<35s} {base_avg_alpha:>11.2f}% {managed_avg_alpha:>11.2f}% {avg_return_change:>+11.2f}%")
    print(f"   {'Средний Sharpe':<35s} {base_avg_sharpe:>12.3f} {managed_avg_sharpe:>12.3f} {avg_sharpe_improvement:>+12.3f}")
    print(f"   {'Средняя просадка':<35s} {base_avg_dd:>11.2f}% {managed_avg_dd:>11.2f}% {-avg_dd_improvement:>+11.2f}%")

    print(f"\nЭФФЕКТИВНОСТЬ РИСК-МЕНЕДЖМЕНТА:")
    print(f"   Улучшил Sharpe: {sharpe_improved_pct:.0f}% тикеров")
    print(f"   Снизил просадку: {dd_improved_pct:.0f}% тикеров")
    print(f"   Среднее улучшение Sharpe: {avg_sharpe_improvement:+.3f}")
    print(f"   Среднее снижение просадки: {avg_dd_improvement:+.2f}%")

    best_alpha_idx = results_df['alpha'].idxmax()
    worst_alpha_idx = results_df['alpha'].idxmin()
    best_sharpe_idx = results_df['managed_sharpe'].idxmax()

    print(f"\nЛУЧШИЕ РЕЗУЛЬТАТЫ:")
    print(f"   Альфа: {results_df.loc[best_alpha_idx, 'ticker']} "
          f"({results_df.loc[best_alpha_idx, 'alpha']*100:.2f}%)")
    print(f"   Sharpe: {results_df.loc[best_sharpe_idx, 'ticker']} "
          f"({results_df.loc[best_sharpe_idx, 'managed_sharpe']:.3f})")
    print(f"   Худшая альфа: {results_df.loc[worst_alpha_idx, 'ticker']} "
          f"({results_df.loc[worst_alpha_idx, 'alpha']*100:.2f}%)")

    avg_kelly = np.mean(results_df['kelly_size']) * 100
    avg_position = np.mean(results_df['avg_position']) * 100
    avg_stops = np.mean(results_df['stop_loss_count'])
    avg_tp = np.mean(results_df['take_profit_count'])

    print(f"\nСРЕДНИЕ ПАРАМЕТРЫ РИСК-МЕНЕДЖМЕНТА:")
    print(f"   Келли размер: {avg_kelly:.1f}%")
    print(f"   Средняя позиция: {avg_position:.1f}%")
    print(f"   Стоп-лоссов: {avg_stops:.0f}")
    print(f"   Тейк-профитов: {avg_tp:.0f}")

    results_df.to_csv('risk_management_results.csv', index=False)
    print("\nРезультаты сохранены в risk_management_results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    x = np.arange(len(results_df))
    width = 0.3

    bars1 = ax1.bar(x - width, results_df['base_return']*100, width,
                    label='Базовая', color='blue', alpha=0.7)
    bars2 = ax1.bar(x, results_df['managed_return']*100, width,
                    label='С Risk Mgmt', color='green', alpha=0.7)
    bars3 = ax1.bar(x + width, results_df['bh_return']*100, width,
                    label='Buy&Hold', color='gray', alpha=0.5)

    ax1.set_ylabel('Доходность (%)')
    ax1.set_title('Сравнение доходностей стратегий')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['ticker'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%', ha='center',
                va='bottom' if height > 0 else 'top', fontsize=7)

    ax2 = axes[0, 1]
    bars1 = ax2.bar(x - width/2, results_df['base_sharpe'], width,
                    label='Базовая', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, results_df['managed_sharpe'], width,
                    label='С Risk Mgmt', color='green', alpha=0.7)

    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Сравнение Sharpe Ratio')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['ticker'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax3 = axes[1, 0]
    bars1 = ax3.bar(x - width/2, results_df['base_max_dd']*100, width,
                    label='Базовая', color='red', alpha=0.5)
    bars2 = ax3.bar(x + width/2, results_df['managed_max_dd']*100, width,
                    label='С Risk Mgmt', color='orange', alpha=0.7)

    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_title('Сравнение максимальных просадок')
    ax3.set_xticks(x)
    ax3.set_xticklabels(results_df['ticker'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]

    sharpe_improvements = results_df['sharpe_improvement'].values
    dd_improvements = results_df['dd_improvement'].values * 100

    colors_sharpe = ['green' if v > 0 else 'red' for v in sharpe_improvements]
    colors_dd = ['green' if v > 0 else 'red' for v in dd_improvements]

    ax4.barh(x + 0.2, sharpe_improvements, 0.4, color=colors_sharpe, alpha=0.7, label='Δ Sharpe')
    ax4.barh(x - 0.2, dd_improvements, 0.4, color=colors_dd, alpha=0.7, label='Δ Drawdown (%)')

    ax4.set_yticks(x)
    ax4.set_yticklabels(results_df['ticker'])
    ax4.set_xlabel('Изменение')
    ax4.set_title('Эффект риск-менеджмента')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle(f'Анализ риск-менеджмента ({len(results_df)} тикеров)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('risk_management_comparison.png', dpi=150, bbox_inches='tight')
    print("Графики сохранены в risk_management_comparison.png")
    plt.close()

    return results_df


if __name__ == "__main__":
    run_pipeline()
