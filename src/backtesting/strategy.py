import numpy as np

def strategy_returns(predictions, real_returns):
    predictions = np.asarray(predictions)
    real_returns = np.asarray(real_returns)

    predictions = np.nan_to_num(predictions)
    real_returns = np.nan_to_num(real_returns)

    positions = np.sign(predictions)

    strat_returns = positions * real_returns

    return strat_returns, positions
