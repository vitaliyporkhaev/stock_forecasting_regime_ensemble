import matplotlib.pyplot as plt


def plot_equity_curve(equity_curve, title="Equity Curve"):
    plt.figure()
    plt.plot(equity_curve)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.show()


def plot_returns(returns, title="Returns"):
    plt.figure()
    plt.plot(returns)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.show()
