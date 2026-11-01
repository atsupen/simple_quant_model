Python implementation of a simple quantitative trading model that combines momentum and mean reversion to predict changes in Tesla's share price, using daily data from 2010-2026.

## what's in here

- `src/data_cleaning.py` downloads the full tesla price history and cleans it up
- `src/ode_model.py` is the ode model on its own. it checks our rk4 solver against the exact answer (momentum switched off), then runs a stable and an unstable case and works out the jacobian and eigenvalues
- `src/backtest.py` is the actual test on tesla. it fits kappa and beta on 2011-2021, predicts 2022 to oct 2026, and simulates trading with fees for 20, 60 and 120 day averages
- `src/bootstrap.py` checks whether the small forecast improvement is real or just noise
- `data/tsla_daily_2011_2026.csv` is the yahoo finance snapshot (adjusted prices) we used for the results in our presentation

## the model

x is the gap between log price and log moving average, m is the remembered trend

```
dx/dt = -kappa*x + beta*tanh(gamma*m)
dm/dt = alpha*(dx/dt - m)
```

each day we predict tomorrow's log return with `-kappa*ln(close/gma) + beta*tanh(gamma*trend)`

## running it

```
pip install -r requirements.txt
python src/ode_model.py
python src/backtest.py
python src/bootstrap.py
```

run them from the repo folder. results go into `results/`
