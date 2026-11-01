# is the tiny forecast improvement real or just noise?
# we compare the daily squared error of our model against just guessing zero,
# then resample the test period in blocks of 20 days (moving block bootstrap)
# so we keep some of the day to day dependence.
# run src/backtest.py first, then: python src/bootstrap.py

import numpy as np
import pandas as pd

f = pd.read_csv("results/test_forecasts_60.csv")
y = f.realised_logreturn.to_numpy()
pred = f.combined.to_numpy()

# negative means our model had a smaller error than guessing zero
differences = (pred - y) ** 2 - y ** 2
n = len(differences)

block = 20
reps = 1000
rng = np.random.default_rng(20261028)
means = np.empty(reps)
for j in range(reps):
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    means[j] = differences[idx].mean()

low, high = np.quantile(means, [0.025, 0.975])
print(f"test days: {n}")
print(f"mean difference in squared error: {differences.mean():.3e}")
print(f"mse reduction vs guessing zero: {100 * -differences.mean() / np.mean(y ** 2):.4f}%")
print(f"95% bootstrap interval: [{low:.3e}, {high:.3e}]")
print("interval includes zero, so no real improvement" if low <= 0 <= high else "interval excludes zero")
