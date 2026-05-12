import numpy as np


def sharpe_ratio(returns, rf=0.0):
    returns = np.nan_to_num(returns)

    if np.std(returns) == 0:
        return 0.0

    return (np.mean(returns) - rf) / np.std(returns) * np.sqrt(252)


def max_drawdown(equity_curve):
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-9)
    return np.min(drawdown)


def cumulative_return(returns):
    returns = np.nan_to_num(returns)

    return np.prod(1 + returns) - 1


def sortino_ratio(returns, rf=0.0):
    returns = np.array(returns)
    downside = returns[returns < 0]
    downside_std = np.std(downside) + 1e-9
    return np.mean(returns - rf) / downside_std
