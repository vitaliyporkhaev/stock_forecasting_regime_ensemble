import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def prepare(df):
    return pd.DataFrame({
        "return": df["log_return"],
        "vol": df["log_return"].rolling(10).std(),
        "rsi": df["rsi"]
    }).dropna()


def add_cluster_regimes(df, k=3):
    X = prepare(df)

    X_scaled = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)

    df = df.loc[X.index]
    df["regime"] = labels

    return df
