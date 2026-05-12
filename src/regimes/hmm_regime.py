import pandas as pd
from hmmlearn.hmm import GaussianHMM


def prepare_features(df):
    return pd.DataFrame({
        "return": df["log_return"],
        "vol": df["log_return"].rolling(10).std(),
        "volume": df["volume"].pct_change()
    }).dropna()


class HMMRegime:
    def __init__(self, n_states=3):
        self.model = GaussianHMM(n_components=n_states, covariance_type="full")

    def fit_predict(self, X):
        self.model.fit(X)
        return self.model.predict(X)


def add_regimes(df):
    X = prepare_features(df)

    hmm = HMMRegime()
    regimes = hmm.fit_predict(X)

    df = df.loc[X.index]
    df["regime"] = regimes

    return df
