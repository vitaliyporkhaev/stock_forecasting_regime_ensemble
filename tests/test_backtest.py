import numpy as np

from src.backtesting.backtest_engine import BacktestEngine


def test_backtest_engine():
    preds = np.random.randn(100)
    returns = np.random.randn(100) * 0.01

    engine = BacktestEngine()
    result = engine.run(preds, returns)

    assert "sharpe" in result
    assert "max_drawdown" in result
    assert len(result["equity_curve"]) == 100
