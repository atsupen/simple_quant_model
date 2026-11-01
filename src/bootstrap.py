import numpy as np
import pandas as pd

f = pd.read_csv("results/test_forecasts_60.csv")
y = f.realised_logreturn.to_numpy()
pred = f.combined.to_numpy()

# negative means we had a smaller error than guessing no change
differences = (pred - y) ** 2 - y ** 2
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
print(f"test days: {n}")
print(f"mean difference in squared error: {differences.mean():.3e}")
print(f"mse reduction vs guessing zero: {100 * -differences.mean() / np.mean(y ** 2):.4f}%")
print(f"95% bootstrap interval: [{low:.3e}, {high:.3e}]")
print("interval includes zero, so no real improvement" if low <= 0 <= high else "interval excludes zero")
# what the fuck.... 🧧🧧🧧🧧