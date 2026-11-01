# tesla backtest for the momentum + mean reversion model
#
# each day t (after the close) we predict tomorrow's log return with
#   r_hat = -kappa * ln(close / gma) + beta * tanh(gamma * trend)
# where gma is the geometric moving average of the last n closes and
# trend is a fading memory of daily log returns (90% old, 10% new)
#
# kappa and beta are fitted on 2011-2021 with nonnegative least squares,
# then we test on 2022 to oct 2026 without changing anything.
# trading rule: if r_hat > 0 hold shares from the next open, otherwise hold cash,
# and pay a fee on every buy and sell.
#
# data is yahoo finance adjusted daily prices (downloaded with yfinance on 28 oct 2026)
# run from the repo folder with: python src/backtest.py

import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls

os.makedirs("results", exist_ok=True)

# settings, all picked before looking at the test years
gamma = 100.0                  # sensitivity to the trend (days)
rho = 0.1                      # weight on today's return in the trend memory
windows = [20, 60, 120]        # gma lengths in trading days, 60 is the main one
fees = [0.0, 0.001, 0.0025]    # fee on every buy or sell (0%, 0.10%, 0.25%)
train_end = "2021-12-31"
test_start = "2022-01-01"
test_end = "2026-10-27"
tol = 1e-12                    # anything smaller than this counts as zero

prices = pd.read_csv("data/tsla_daily_2011_2026.csv", index_col=0, parse_dates=True)
prices.columns = prices.columns.str.lower()
print(f"{len(prices)} trading days from {prices.index[0].date()} to {prices.index[-1].date()}")


def make_features(prices, window):
    df = pd.DataFrame(index=prices.index)
    df["close"] = prices.close
    df["open"] = prices.open
    df["logreturn"] = np.log(prices.close).diff()
    # gma = exp(average of the log prices), first window-1 days are nan
    df["gma"] = np.exp(np.log(prices.close).rolling(window, min_periods=window).mean())
    # trend memory m_t = 0.9*m_{t-1} + 0.1*r_t, starting at 0
    trend = np.zeros(len(df))
    for i in range(1, len(df)):
        trend[i] = (1 - rho) * trend[i - 1] + rho * df.logreturn.iloc[i]
    df["trend"] = trend
    # the two things we multiply by kappa and beta
    df["reversion_feature"] = -np.log(df.close / df.gma)
    df["momentum_feature"] = np.tanh(gamma * df.trend)
    # what we're trying to predict is tomorrow's return
    df["target"] = df.logreturn.shift(-1)
    df["target_date"] = pd.Series(df.index, index=df.index).shift(-1)
    return df


def sign(x):
    return np.where(x > tol, 1, np.where(x < -tol, -1, 0))


def forecast_scores(actual, predicted):
    err = predicted - actual
    mse = np.mean(err ** 2)
    return {
        "rmse": np.sqrt(mse),
        "mae": np.mean(np.abs(err)),
        "mse vs guessing zero": mse / np.mean(actual ** 2),
        "direction right": np.mean(sign(predicted) == sign(actual)),
    }


def simulate(opens, signals, fee):
    # start with $1 of cash, trade at each open
    cash, shares = 1.0, 0.0
    path = [1.0]        # wealth before and after every trade, used for the drawdown
    wealth = []
    invested = []
    trades = 0
    total_fees = 0.0
    for i, price in enumerate(opens):
        before = cash + shares * price
        path.append(before)
        last_day = i == len(opens) - 1
        want_shares = signals[i] > tol and not last_day   # always sell out on the last day
        if want_shares and shares == 0:
            shares = cash / (price * (1 + fee))
            total_fees += shares * price * fee
            cash = 0.0
            trades += 1
        elif not want_shares and shares > 0:
            cost = shares * price * fee
            total_fees += cost
            cash = shares * price - cost
            shares = 0.0
            trades += 1
        after = cash + shares * price
        path.append(after)
        wealth.append(after)
        invested.append(shares > 0)
    path = np.array(path)
    drawdown = 1 - path / np.maximum.accumulate(path)
    return {
        "total return": wealth[-1] - 1,
        "max drawdown": drawdown.max(),
        "trades": trades,
        "days invested": np.mean(invested[:-1]),
        "fees paid": total_fees,
    }, np.array(wealth)


all_trading = []
for window in windows:
    df = make_features(prices, window)
    df = df.dropna(subset=["reversion_feature", "momentum_feature", "target", "target_date"])
    train = df[df.target_date <= pd.Timestamp(train_end)]
    test = df[(df.target_date >= pd.Timestamp(test_start)) & (df.target_date <= pd.Timestamp(test_end))]

    # fit kappa and beta >= 0 by least squares (no constant term)
    x_train = train[["reversion_feature", "momentum_feature"]].to_numpy()
    y_train = train.target.to_numpy()
    (kappa, beta), _ = nnls(x_train, y_train)
    reversion_only, _ = nnls(x_train[:, [0]], y_train)
    momentum_only, _ = nnls(x_train[:, [1]], y_train)

    x_test = test[["reversion_feature", "momentum_feature"]].to_numpy()
    y_test = test.target.to_numpy()
    predictions = {
        "combined": kappa * x_test[:, 0] + beta * x_test[:, 1],
        "momentum only": momentum_only[0] * x_test[:, 1],
        "reversion only": reversion_only[0] * x_test[:, 0],
        "guess zero": np.zeros(len(test)),
        "training mean": np.full(len(test), y_train.mean()),
    }

    print()
    print(f"===== gma window {window} days =====")
    print(f"train: {len(train)} days, test: {len(test)} days")
    print(f"fitted kappa = {kappa:.6f}, beta = {beta:.6f}")
    for name, pred in predictions.items():
        s = forecast_scores(y_test, pred)
        print(f"  {name:<15} rmse {100 * s['rmse']:.4f}%  mae {100 * s['mae']:.4f}%  "
              f"mse/zero {s['mse vs guessing zero']:.6f}  direction {100 * s['direction right']:.1f}%")

    # keep the 60 day forecasts for the bootstrap check
    if window == 60:
        out = pd.DataFrame({"realised_logreturn": y_test, "combined": predictions["combined"]},
                           index=pd.DatetimeIndex(test.target_date, name="target_date"))
        out.to_csv("results/test_forecasts_60.csv", float_format="%.17g")

    # trading: forecast made after day t's close is used at day t+1's open
    opens = prices.open.loc[test.target_date].to_numpy()
    strategies = {
        "combined": predictions["combined"],
        "momentum only": predictions["momentum only"],
        "reversion only": predictions["reversion only"],
        "buy and hold": np.ones(len(test)),
        "cash": np.zeros(len(test)),
    }
    for fee in fees:
        print(f"  fee {100 * fee:.2f}% per trade")
        for name, signals in strategies.items():
            stats, wealth = simulate(opens, signals, fee)
            print(f"    {name:<15} return {100 * stats['total return']:7.2f}%  "
                  f"max drawdown {100 * stats['max drawdown']:6.2f}%  trades {stats['trades']:3d}  "
                  f"invested {100 * stats['days invested']:5.1f}%")
            all_trading.append({"window": window, "fee": fee, "strategy": name, **stats})
            if window == 60 and fee == 0.001 and name in ("combined", "buy and hold"):
                pd.Series(wealth, index=test.target_date.to_numpy(), name="wealth").to_csv(
                    f"results/wealth_{name.replace(' ', '_')}_60.csv")

pd.DataFrame(all_trading).to_csv("results/trading_results.csv", index=False)
print()
print("saved results/trading_results.csv and results/test_forecasts_60.csv")
