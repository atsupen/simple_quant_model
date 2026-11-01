# ode model for momentum vs mean reversion
# x = gap between log price and log average (x = p - v), m = remembered trend (per day)
#   dx/dt = -kappa*x + beta*tanh(gamma*m)
#   dm/dt = alpha*(dx/dt - m)
# we hold the average v fixed here so the gap x is the only thing that moves
#
# this script does three things
#   1. checks our rk4 solver against the exact answer when momentum is off (beta = 0)
#   2. runs a stable case (beta = 0.01) and an unstable case (beta = 0.02)
#   3. works out the jacobian and eigenvalues for both
# run from the repo folder with: python src/ode_model.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

os.makedirs("results", exist_ok=True)


def derivative(state, kappa, alpha, beta, gamma):
    x, m = state
    dx = -kappa * x + beta * np.tanh(gamma * m)
    dm = alpha * (dx - m)
    return np.array([dx, dm])


def rk4(state0, t_end, h, params):
    # classic 4th order runge kutta with a fixed step h (in days)
    n_steps = int(round(t_end / h))
    t = np.arange(n_steps + 1) * h
    y = np.zeros((n_steps + 1, 2))
    y[0] = state0
    for i in range(n_steps):
        k1 = derivative(y[i], **params)
        k2 = derivative(y[i] + h * k1 / 2, **params)
        k3 = derivative(y[i] + h * k2 / 2, **params)
        k4 = derivative(y[i] + h * k3, **params)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, y


def exact_solution(t, x0, m0, kappa, alpha):
    # with beta = 0 the x equation is just dx/dt = -kappa*x so x = x0*exp(-kappa*t)
    # then dm/dt = alpha*(-kappa*x - m) is linear so we solved it by hand (integrating factor)
    x = x0 * np.exp(-kappa * t)
    m = m0 * np.exp(-alpha * t) - alpha * kappa * x0 / (alpha - kappa) * (np.exp(-kappa * t) - np.exp(-alpha * t))
    return np.column_stack([x, m])


# same numbers as the slides
kappa = 0.05   # pull back strength, per day
alpha = 0.1    # how fast the trend memory reacts, per day
gamma = 100.0  # sensitivity to the trend, days


# ---------- 1. check rk4 against the exact answer ----------
print("rk4 check with momentum switched off (beta = 0)")
print("step (days)   max error in x   order")
params = dict(kappa=kappa, alpha=alpha, beta=0.0, gamma=gamma)
previous_error = None
for h in [1.0, 0.5, 0.25, 0.125]:
    t, y = rk4([0.1, 0.0], 100.0, h, params)
    exact = exact_solution(t, 0.1, 0.0, kappa, alpha)
    error = np.max(np.abs(y[:, 0] - exact[:, 0]))
    # halving the step should make the error about 2^4 = 16 times smaller
    order = np.log2(previous_error / error) if previous_error else float("nan")
    print(f"{h:<13} {error:<16.3e} {order:.2f}")
    previous_error = error
    if h == 1.0:
        t_one_day, y_one_day, exact_one_day = t, y, exact


# ---------- 2. stable vs unstable ----------
print()
cases = {"stable": 0.01, "unstable": 0.02}
runs = {}
for name, beta in cases.items():
    params = dict(kappa=kappa, alpha=alpha, beta=beta, gamma=gamma)
    t, y = rk4([0.01, 0.0], 300.0, 0.125, params)
    runs[name] = (t, y)

    # compare against scipy's solver as a second check
    ref = solve_ivp(lambda time, s: derivative(s, **params), (0, 300), [0.01, 0.0],
                    method="DOP853", t_eval=t, rtol=1e-11, atol=[1e-13, 1e-15])
    diff = np.max(np.abs(y[:, 0] - ref.y[0]))

    # linearise around (x, m) = (0, 0) using tanh(gamma*m) ~ gamma*m
    jacobian = np.array([[-kappa, beta * gamma],
                         [-alpha * kappa, alpha * (beta * gamma - 1)]])
    eigenvalues = np.linalg.eigvals(jacobian)

    # stable when beta*gamma < 1 + kappa/alpha, i.e. b < 1 + k
    big_b = beta * gamma
    big_k = kappa / alpha
    print(f"{name}: beta = {beta}, b = {big_b:g}, k = {big_k:g}, stable if b < {1 + big_k:g}")
    print(f"  trace = {np.trace(jacobian):.4f}, det = {np.linalg.det(jacobian):.4f}")
    print(f"  eigenvalues = {eigenvalues[0]:.4f} and {eigenvalues[1]:.4f}")
    print(f"  biggest |x| in the last 100 days = {np.max(np.abs(y[t >= 200, 0])):.3e}")
    print(f"  max difference from scipy dop853 = {diff:.1e}")


# ---------- plots ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(t_one_day, exact_one_day[:, 0], "k-", label="exact")
axes[0].plot(t_one_day[::10], y_one_day[::10, 0], "ko", mfc="white", label="rk4, h = 1 day")
axes[0].set_xlabel("time (days)")
axes[0].set_ylabel("gap x (momentum off)")
axes[0].legend()
for name, (t, y) in runs.items():
    axes[1].plot(t, y[:, 0], label=f"{name}, beta = {cases[name]}")
axes[1].set_xlabel("time (days)")
axes[1].set_ylabel("gap x")
axes[1].legend()
fig.tight_layout()
fig.savefig("results/ode_model.png", dpi=150)
print()
print("saved results/ode_model.png")
