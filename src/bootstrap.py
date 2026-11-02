import typing
import numpy as np
import pandas as pd

forecast_results = pd.read_csv("results/test_forecasts_60.csv")
y_result = forecast_results.realised_logreturn.to_numpy()
y_pred = forecast_results.combined.to_numpy()

differences: np.ndarray = (y_pred - y_result) ** 2 - y_result ** 2
n = len(differences)

block = 20
reps = 1000
seed = np.random.default_rng(20261028)
means = np.empty(reps)

for j in range(reps):
    starts = seed.integers(0, n - block + 1, size = int(np.ceil(n / block)))
    index = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    means[j] = differences[index].mean()

low, high = np.quantile(means, [0.025, 0.975])
print(f"Sample Size: {n}")
print(f"Mean difference in squared error: {differences.mean():.3e}")
print(f"MSE reduction vs guessing zero: {100 * -differences.mean() / np.mean(y ** 2):.4f}%")
print(f"95% bootstrap interval: [{low:.3e}, {high:.3e}]")