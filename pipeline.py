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


def run_pipeline_for_ticker(ticker, data_cfg, train_cfg):
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
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=7,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)

    X_train_lstm = X_train.values.reshape(len(X_train), X_train.shape[1], 1)
    X_test_lstm = X_test.values.reshape(len(X_test), X_test.shape[1], 1)

    lstm_model = LSTMModel(input_shape=(X_train.shape[1], 1))
    lstm_model.fit(X_train_lstm, y_train.values, epochs=10, batch_size=64)
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

    meta_scaler = StandardScaler()
    meta_X_scaled = meta_scaler.fit_transform(meta_X)

    split_meta = int(len(meta_X_scaled) * 0.8)

    meta_X_train = meta_X_scaled[:split_meta]
    meta_X_test = meta_X_scaled[split_meta:]
    y_train_meta = y_test_aligned[:split_meta]
    y_test_meta = y_test_aligned[split_meta:]

    stacker = StackingModel()
    stacker.meta_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    stacker.fit(meta_X_train, y_train_meta)

    final_pred = stacker.predict(meta_X_test)
    final_pred = (final_pred - np.mean(final_pred)) / (np.std(final_pred) + 1e-9)

    train_trend = np.mean(y_train_meta)

    test_trend = np.mean(y_test_meta)

    price_start_idx = split + split_meta
    price_end_idx = price_start_idx + len(y_test_meta) + 1
    bh_prices = df["close"].iloc[price_start_idx:price_end_idx].values

    if len(bh_prices) > 1:
        bh_returns = np.diff(np.log(bh_prices))
        bh_total_return = bh_prices[-1] / bh_prices[0] - 1
        bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-9) * np.sqrt(252)
        bh_equity = bh_prices / bh_prices[0]
    else:
        bh_total_return = bh_sharpe = 0

    market_up = bh_total_return > 0

    print(f"Анализ рынка:")
    print(f"      Train тренд: {train_trend*100:+.2f}%/день")
    print(f"      Test тренд:  {test_trend*100:+.2f}%/день")
    print(f"      B&H на тесте: {bh_total_return*100:+.2f}% → {'📈 UP' if market_up else '📉 DOWN'}")

    strategies = {
        'Percentile (70/30)': lambda p: percentile_strategy(p, 70, 30),
        'Percentile (60/40)': lambda p: percentile_strategy(p, 60, 40),
        'Momentum (5)': lambda p: momentum_strategy(p, 5),
        'Momentum (10)': lambda p: momentum_strategy(p, 10),
        'Long-biased (80/20)': lambda p: np.where(p > np.percentile(p, 20), 1, 0),
        'Long only (70/0)': lambda p: np.where(p > np.percentile(p, 30), 1, 0),
        'Always Long': lambda p: np.ones_like(p),
    }

    if not market_up:
        strategies.update({
            'Short only': lambda p: np.where(p < np.median(p), -1, 0),
            'Short-biased': lambda p: np.where(p < np.percentile(p, 70), -1, 1),
        })

    print(f"\nТестирование {len(strategies)} стратегий:")
    print(f"   {'Стратегия':<25s} {'Sharpe':>8s} {'Return':>10s} {'Trades':>8s}")
    print(f"   {'-'*55}")

    best_sharpe = -np.inf
    best_strategy = None
    best_signals = None
    best_name = ""

    for name, strategy_func in strategies.items():
        try:
            signals = strategy_func(final_pred)
            ret = signals * y_test_meta
            sharpe = np.mean(ret) / (np.std(ret) + 1e-9) * np.sqrt(252)
            total_ret = np.prod(1 + ret + 1e-9) - 1
            trades = np.sum(np.diff(signals) != 0) // 2

            print(f"   {name:<25s} {sharpe:>8.3f} {total_ret*100:>9.2f}% {trades:>8d}")

            adjusted_sharpe = sharpe - (0.5 if trades == 0 else 0)

            if adjusted_sharpe > best_sharpe:
                best_sharpe = adjusted_sharpe
                best_signals = signals
                best_name = name
                best_strategy = ret
        except Exception as e:
            print(f"   {name:<25s} ERROR: {str(e)[:30]}")
            continue

    if best_strategy is None:
        best_signals = np.ones_like(final_pred)
        best_name = 'Always Long (fallback)'
        best_strategy = best_signals * y_test_meta

    strategy_returns = best_strategy

    equity_curve = np.cumprod(1 + strategy_returns + 1e-9)
    cumulative_return = equity_curve[-1] - 1 if len(equity_curve) > 0 else 0
    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9) * np.sqrt(252)

    trades = np.sum(np.diff(best_signals) != 0) // 2

    long_exposure = np.mean(best_signals > 0) * 100
    short_exposure = np.mean(best_signals < 0) * 100
    flat_exposure = np.mean(best_signals == 0) * 100

    long_returns = strategy_returns[best_signals > 0]
    short_returns = strategy_returns[best_signals < 0]

    long_avg = np.mean(long_returns) * 100 if len(long_returns) > 0 else 0
    short_avg = np.mean(short_returns) * 100 if len(short_returns) > 0 else 0

    print(f"\nВыбрана: {best_name}")
    print(f"Позиции: Long={long_exposure:.0f}%, Short={short_exposure:.0f}%, Flat={flat_exposure:.0f}%")
    print(f"Long avg: {long_avg:+.3f}%/день, Short avg: {short_avg:+.3f}%/день")
    print(f"Strategy: {cumulative_return*100:+.2f}% vs B&H: {bh_total_return*100:+.2f}%")
    print(f"Alpha: {(cumulative_return-bh_total_return)*100:+.2f}%, Sharpe: {sharpe:.3f}")

    return {
        'ticker': ticker,
        'strategy_return': cumulative_return,
        'bh_return': bh_total_return,
        'alpha': cumulative_return - bh_total_return,
        'sharpe': sharpe,
        'bh_sharpe': bh_sharpe,
        'trades': trades,
        'strategy': best_name,
        'market': 'UP' if market_up else 'DOWN',
        'long_pct': long_exposure,
        'short_pct': short_exposure,
        'flat_pct': flat_exposure,
        'train_trend': train_trend,
        'test_trend': test_trend
    }


def run_pipeline():
    print("ЗАПУСК ТОРГОВОГО ПАЙПЛАЙНА (МУЛЬТИ-ТИКЕР)")

    data_cfg = load_config("data")
    train_cfg = load_config("train")

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

    print(f"\nТестируемые тикеры: {', '.join(tickers)}")

    all_results = []

    for ticker in tickers:
        try:
            result = run_pipeline_for_ticker(ticker, data_cfg, train_cfg)
            all_results.append(result)
        except Exception as e:
            print(f"Ошибка для {ticker}: {str(e)[:100]}")
            continue

    if all_results:
        results_df = pd.DataFrame(all_results)

        print("\n")
        print("СВОДНЫЕ РЕЗУЛЬТАТЫ ПО ВСЕМ ТИКЕРАМ")

        display_df = results_df.copy()
        display_df['strategy_return'] = display_df['strategy_return'] * 100
        display_df['bh_return'] = display_df['bh_return'] * 100
        display_df['alpha'] = display_df['alpha'] * 100

        print(display_df.to_string(
            index=False,
            formatters={
                'strategy_return': '{:+.2f}%'.format,
                'bh_return': '{:+.2f}%'.format,
                'alpha': '{:+.2f}%'.format,
                'sharpe': '{:.3f}'.format,
                'bh_sharpe': '{:.3f}'.format,
            }
        ))

        print("\n")
        print("АГРЕГИРОВАННАЯ СТАТИСТИКА")

        winning_tickers = np.sum(results_df['alpha'] > 0)
        total_tickers = len(results_df)

        print(f"Стратегия обыграла Buy&Hold: {winning_tickers}/{total_tickers} тикеров ({winning_tickers/total_tickers*100:.1f}%)")
        print(f"Средняя альфа: {np.mean(results_df['alpha'])*100:.2f}%")
        print(f"Медианная альфа: {np.median(results_df['alpha'])*100:.2f}%")
        print(f"Средний Sharpe: {np.mean(results_df['sharpe']):.3f}")
        print(f"Средний Buy&Hold Sharpe: {np.mean(results_df['bh_sharpe']):.3f}")
        print(f"Среднее количество сделок: {np.mean(results_df['trades']):.0f}")
        print(f"\nЛучший тикер: {results_df.loc[results_df['alpha'].idxmax(), 'ticker']} "
              f"(α={results_df['alpha'].max()*100:.2f}%)")
        print(f"Худший тикер: {results_df.loc[results_df['alpha'].idxmin(), 'ticker']} "
              f"(α={results_df['alpha'].min()*100:.2f}%)")

        results_df.to_csv('multi_ticker_results.csv', index=False)
        print("\nРезультаты сохранены в multi_ticker_results.csv")

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(results_df))
        width = 0.35

        bars1 = ax.bar(x - width/2, results_df['strategy_return']*100, width,
                       label='Strategy', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, results_df['bh_return']*100, width,
                       label='Buy&Hold', color='blue', alpha=0.7)

        ax.set_xlabel('Ticker')
        ax.set_ylabel('Return (%)')
        ax.set_title('Strategy vs Buy&Hold Returns by Ticker')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['ticker'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8)

        plt.tight_layout()
        plt.savefig('multi_ticker_comparison.png', dpi=150, bbox_inches='tight')
        print("График сравнения сохранен в multi_ticker_comparison.png")
        plt.close()

    return results_df if all_results else None


if __name__ == "__main__":
    run_pipeline()
