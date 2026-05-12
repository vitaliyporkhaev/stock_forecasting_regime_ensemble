import numpy as np


def sharpe_ratio(returns):
    returns = np.array(returns)
    return np.mean(returns) / (np.std(returns) + 1e-9)


def sortino_ratio(returns):
    returns = np.array(returns)
    downside = returns[returns < 0]

    return np.mean(returns) / (np.std(downside) + 1e-9)


def max_drawdown(equity_curve):
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / (peak + 1e-9)
    return np.min(dd)


def calmar_ratio(total_return, max_dd):
    return total_return / (abs(max_dd) + 1e-9)
