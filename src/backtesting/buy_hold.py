import numpy as np


def buy_and_hold_returns(prices):
    prices = np.array(prices)
    return prices / prices[0] - 1


def buy_and_hold_equity(prices):
    prices = np.array(prices)
    return prices / prices[0]
