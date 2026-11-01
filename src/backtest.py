# tesla backtest for the momentum + mean reversion model
#
# each day t (after the close) we predict tomorrow's log return with
#   predicted = -kappa * ln(close / gma) + beta * tanh(gamma * trend)
# where gma is the geometric moving average of the last n closes and
# trend is a fading memory of daily log returns (90% old, 10% new)
#
# kappa and beta are fitted on 2011-2021 with nonnegative least squares,
# then we test on 2022 to oct 2026 without changing anything.
# trading rule: if predicted > 0 hold shares from the next open, otherwise hold cash,
# and pay a fee on every buy and sell.

import os
from typing import Final # ensures variables cannot reassigned

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import nnls

FloatArray = npt.NDArray[np.float64]

os.makedirs("results", exist_ok=True)

TREND_SENSITIVITY: Final[float] = 100.0 # (gamma)
TREND_NEW_RETURN_WEIGHT: Final[float] = 0.1   # (rho)
GMA_WINDOWS: Final[list[int]] = [20, 60, 120]  # (gma)
FEE_RATES: Final[list[float]] = [0.0, 0.001, 0.0025]  # fee on every buy or sell (proportion)
TRAIN_END_DATE: Final[str] = "2021-12-31"
TEST_START_DATE: Final[str] = "2022-01-01"
TEST_END_DATE: Final[str] = "2026-10-27"
SIGN_TOLERANCE: Final[float] = 1e-12

PRICES_CSV_PATH: Final[str] = "data/tesla_prices_clean.csv"
DATE_COLUMN: Final[str] = "date"


def create_features(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    features = pd.DataFrame(index=prices.index)
    features["close"] = prices.close
    features["open"] = prices.open
    log_close = np.log(prices.close)
    features["log_return"] = log_close.diff()
    features["gma"] = np.exp(log_close.rolling(window, min_periods=window).mean())
    trend = np.zeros(len(features))
    for i in range(1, len(features)):
        trend[i] = (
            (1 - TREND_NEW_RETURN_WEIGHT) * trend[i - 1]
            + TREND_NEW_RETURN_WEIGHT * features.log_return.iloc[i]
        )
    features["trend"] = trend
    features["reversion_feature"] = -np.log(features.close / features.gma)
    features["momentum_feature"] = np.tanh(TREND_SENSITIVITY * features.trend)
    features["next_day_log_return"] = features.log_return.shift(-1)
    features["next_day_date"] = pd.Series(features.index, index=features.index).shift(-1)
    return features


def sign_with_tolerance(values: FloatArray) -> npt.NDArray[np.int_]:
    return np.where(values > SIGN_TOLERANCE, 1, np.where(values < -SIGN_TOLERANCE, -1, 0))


def compute_forcast_metrics(actual: FloatArray, predicted: FloatArray) -> dict[str, float]:
    error = predicted - actual
    mean_squared_error = float(np.mean(error ** 2))
    
    return {
        "rmse": float(np.sqrt(mean_squared_error)),
        "mae": float(np.mean(np.abs(error))),
        "mse vs guessing zero": mean_squared_error / float(np.mean(actual ** 2)),
        "direction right": float(np.mean(sign_with_tolerance(predicted) == sign_with_tolerance(actual))),
    }


def simulate_strategy(
    open_prices: FloatArray,
    signals: FloatArray,
    fee_rate: float ) -> tuple[dict[str, float], FloatArray]:
    
    cash, shares = 1.0, 0.0
    wealth_checkpoints: list[float] = [1.0]  # wealth before and after trading each day
    end_of_day_wealth: list[float] = []
    is_invested_each_day: list[bool] = []
    trade_count = 0
    total_fees_paid = 0.0

    for day, open_price in enumerate(open_prices):
        wealth_before_trading = cash + shares * open_price
        wealth_checkpoints.append(wealth_before_trading)
        is_last_day = day == len(open_prices) - 1
        want_to_hold_shares = signals[day] > SIGN_TOLERANCE and not is_last_day
        
        if want_to_hold_shares and shares == 0:
            shares = cash / (open_price * (1 + fee_rate))
            total_fees_paid += shares * open_price * fee_rate
            cash = 0.0
            trade_count += 1
            
        elif not want_to_hold_shares and shares > 0:
            sell_fee = shares * open_price * fee_rate
            total_fees_paid += sell_fee
            cash = shares * open_price - sell_fee
            shares = 0.0
            trade_count += 1

        wealth_after_trading = cash + shares * open_price
        wealth_checkpoints.append(wealth_after_trading)
        end_of_day_wealth.append(wealth_after_trading)
        is_invested_each_day.append(shares > 0)

    checkpoint_array = np.array(wealth_checkpoints)
    drawdown = 1 - checkpoint_array / np.maximum.accumulate(checkpoint_array)
    stats = {
        "total return": end_of_day_wealth[-1] - 1,
        "max drawdown": float(drawdown.max()),
        "trades": trade_count,
        "days invested": float(np.mean(is_invested_each_day[:-1])),
        "fees paid": total_fees_paid
    }

    return stats, np.array(end_of_day_wealth)




# (main script)

tesla_prices = pd.read_csv(PRICES_CSV_PATH)
tesla_prices = tesla_prices.drop(columns = "Unnamed: 0")
tesla_prices[DATE_COLUMN] = pd.to_datetime(
    tesla_prices[DATE_COLUMN].astype(str).str.slice(0, 10),
    format="%Y-%m-%d"
)
print(tesla_prices.info())
print(tesla_prices.describe())

prices = tesla_prices.set_index(DATE_COLUMN).sort_index()
prices.index.name = "date"
    
assert isinstance(prices.index, pd.DatetimeIndex), "date index must be datetime64"
assert prices.index.is_unique, "duplicate dates in the price file"

all_trading_results: list[dict[str, float | str]] = []

for window in GMA_WINDOWS:
    features = create_features(tesla_prices, window)
    features = features.dropna(
        subset = ["reversion_feature", "momentum_feature", "next_day_log_return", "next_day_date"]
    )

    train_rows = features[features.next_day_date <= pd.Timestamp(TRAIN_END_DATE)]
    test_rows = features[
        (features.next_day_date >= pd.Timestamp(TEST_START_DATE))
        & (features.next_day_date <= pd.Timestamp(TEST_END_DATE))
    ]

    feature_columns = ["reversion_feature", "momentum_feature"]
    train_features: FloatArray = train_rows[feature_columns].to_numpy()
    train_target: FloatArray = train_rows.next_day_log_return.to_numpy()
    (reversion_coef, momentum_coef), _ = nnls(train_features, train_target)  # kappa, beta
    reversion_only_coef, _ = nnls(train_features[:, [0]], train_target)
    momentum_only_coef, _ = nnls(train_features[:, [1]], train_target)

    test_features: FloatArray = test_rows[feature_columns].to_numpy()
    test_target: FloatArray = test_rows.next_day_log_return.to_numpy()
    predictions: dict[str, FloatArray] = {
        "combined": reversion_coef * test_features[:, 0] + momentum_coef * test_features[:, 1],
        "momentum only": momentum_only_coef[0] * test_features[:, 1],
        "reversion only": reversion_only_coef[0] * test_features[:, 0],
        "guess zero": np.zeros(len(test_rows)),
        "training mean": np.full(len(test_rows), train_target.mean()),
    }

    print(f"\n===== gma window {window} days =====")
    print(f"train: {len(train_rows)} days, test: {len(test_rows)} days")
    print(f"fitted kappa = {reversion_coef:.6f}, beta = {momentum_coef:.6f}")

    for model_name, predicted in predictions.items():
        scores = compute_forcast_metrics(test_target, predicted)
        print(f"  {model_name:<15} rmse {100 * scores['rmse']:.4f}%  mae {100 * scores['mae']:.4f}%  "
              f"mse/zero {scores['mse vs guessing zero']:.6f}  direction {100 * scores['direction right']:.1f}%")

    # keep the 60 day forecasts for the bootstrap check
    if window == 60:
        forecasts_for_bootstrap = pd.DataFrame(
            {"realised_logreturn": test_target, "combined": predictions["combined"]},
            index=pd.DatetimeIndex(test_rows.next_day_date, name="target_date"),
        )
        forecasts_for_bootstrap.to_csv("results/test_forecasts_60.csv", float_format="%.17g")

    open_prices: FloatArray = tesla_prices.open.loc[test_rows.next_day_date].to_numpy()
    strategies: dict[str, FloatArray] = {
        "combined": predictions["combined"],
        "momentum only": predictions["momentum only"],
        "reversion only": predictions["reversion only"],
        "buy and hold": np.ones(len(test_rows)),
        "cash": np.zeros(len(test_rows)),
    }

    for fee_rate in FEE_RATES:
        print(f"  fee {100 * fee_rate:.2f}% per trade")

        for strategy_name, signals in strategies.items():
            trading_stats, wealth_path = simulate_strategy(open_prices, signals, fee_rate)
            print(f"    {strategy_name:<15} return {100 * trading_stats['total return']:7.2f}%  "
                  f"max drawdown {100 * trading_stats['max drawdown']:6.2f}%  "
                  f"trades {int(trading_stats['trades']):3d}  "
                  f"invested {100 * trading_stats['days invested']:5.1f}%")
            all_trading_results.append(
                {"window": window, "fee": fee_rate, "strategy": strategy_name, **trading_stats}
            )

            if window == 60 and fee_rate == 0.001 and strategy_name in ("combined", "buy and hold"):
                pd.Series(
                    wealth_path, index=test_rows.next_day_date.to_numpy(), name="wealth"
                ).to_csv(f"results/wealth_{strategy_name.replace(' ', '_')}_60.csv")

pd.DataFrame(all_trading_results).to_csv("results/trading_results.csv", index=False)