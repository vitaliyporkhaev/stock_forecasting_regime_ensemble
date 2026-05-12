import numpy as np


def strategy_returns(predictions, real_returns, mode="sign"):

    if mode == "sign":
        positions = np.sign(predictions)

    elif mode == "tanh":
        positions = np.tanh(predictions)

    else:
        positions = predictions

    returns = positions * real_returns
    return returns, positions
