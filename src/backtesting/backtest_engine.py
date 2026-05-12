import numpy as np

from src.backtesting.metrics import (
    sharpe_ratio,
    max_drawdown,
    cumulative_return
)

from src.backtesting.strategy import strategy_returns
from src.backtesting.portfolio import equity_curve


class BacktestEngine:
    def __init__(self, rf=0.0):
        self.rf = rf

    def run(self, predictions, real_returns):
        print(f"DEBUG: predictions shape: {predictions.shape}")
        print(f"DEBUG: real_returns shape: {real_returns.shape}")
        print(f"DEBUG: predictions mean: {np.mean(predictions):.4f}")
        print(f"DEBUG: real_returns mean: {np.mean(real_returns):.4f}")

        strat_ret, positions = strategy_returns(predictions, real_returns)

        equity = equity_curve(strat_ret)

        results = {
            "returns": strat_ret,
            "positions": positions,
            "equity_curve": equity,
            "sharpe": sharpe_ratio(strat_ret, self.rf),
            "max_drawdown": max_drawdown(equity),
            "cumulative_return": cumulative_return(strat_ret),
        }

        return results
