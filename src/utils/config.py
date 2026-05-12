from dataclasses import dataclass


@dataclass
class Config:
    # data
    tickers: list = None
    start_date: str = "2010-01-01"
    end_date: str = None

    # features
    feature_window: int = 20

    # regimes
    n_regimes: int = 3

    # training
    train_ratio: float = 0.8
    n_splits: int = 5

    # models
    lstm_seq_len: int = 20

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = ["AAPL", "MSFT", "NVDA", "SPY"]
