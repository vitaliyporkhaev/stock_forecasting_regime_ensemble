import pandas as pd
import numpy as np

def realized_volatility(returns: pd.Series, window: int = 20):
    return returns.rolling(window).std() * np.sqrt(window)

def parkinson_volatility(high, low, window=20):
    hl_ratio = np.log(high / low)
    return hl_ratio.rolling(window).std()
