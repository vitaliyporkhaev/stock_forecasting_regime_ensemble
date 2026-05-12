import numpy as np


def equity_curve(strategy_returns):
    returns = np.array(strategy_returns)
    return np.cumprod(1 + returns)


def portfolio_performance(strategy_returns):
    equity = equity_curve(strategy_returns)

    return {
        "equity_curve": equity,
        "total_return": equity[-1] - 1,
    }
