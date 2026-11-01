import os
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FloatArray = npt.NDArray[np.float64]

os.makedirs("results", exist_ok=True)


class ModelParameters(TypedDict):
    mean_reversion_rate: float
    memory_rate: float    
    momentum_strength: float
    trend_sensitivity: float


def compute_derivative(
    state: FloatArray,
    mean_reversion_rate: float,
    memory_rate: float,
    momentum_strength: float,
    trend_sensitivity: float,
) -> FloatArray:
    price_gap, trend_memory = state
    price_gap_rate = (
        -mean_reversion_rate * price_gap
        + momentum_strength * np.tanh(trend_sensitivity * trend_memory)
    )
    trend_memory_rate = memory_rate * (price_gap_rate - trend_memory)
    return np.array([price_gap_rate, trend_memory_rate])


def integrate_rk4(
    initial_state: list[float],
    end_time: float,
    step_size: float,
    params: ModelParameters,
) -> tuple[FloatArray, FloatArray]:

    n_steps = int(round(end_time / step_size))
    times = np.arange(n_steps + 1) * step_size
    states = np.zeros((n_steps + 1, 2))
    states[0] = initial_state
    for i in range(n_steps):
        k1 = compute_derivative(states[i], **params)
        k2 = compute_derivative(states[i] + step_size * k1 / 2, **params)
        k3 = compute_derivative(states[i] + step_size * k2 / 2, **params)
        k4 = compute_derivative(states[i] + step_size * k3, **params)
        states[i + 1] = states[i] + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return times, states


def exact_solution_without_momentum(
    times: FloatArray,
    initial_gap: float,
    initial_trend_memory: float,
    mean_reversion_rate: float,
    memory_rate: float,
) -> FloatArray:

    gap = initial_gap * np.exp(-mean_reversion_rate * times)
    trend_memory = (
        initial_trend_memory * np.exp(-memory_rate * times)
        - memory_rate * mean_reversion_rate * initial_gap / (memory_rate - mean_reversion_rate)
        * (np.exp(-mean_reversion_rate * times) - np.exp(-memory_rate * times))
    )
    return np.column_stack([gap, trend_memory])


MEAN_REVERSION_RATE: float = 0.05
MEMORY_RATE: float = 0.1
TREND_SENSITIVITY: float = 100.0


no_momentum_params: ModelParameters = {
    "mean_reversion_rate": MEAN_REVERSION_RATE,
    "memory_rate": MEMORY_RATE,
    "momentum_strength": 0.0,
    "trend_sensitivity": TREND_SENSITIVITY,
}
previous_max_error: float | None = None
for step_size in [1.0, 0.5, 0.25, 0.125]:
    times, rk4_states = integrate_rk4([0.1, 0.0], 100.0, step_size, no_momentum_params)
    exact_states = exact_solution_without_momentum(
        times, 0.1, 0.0, MEAN_REVERSION_RATE, MEMORY_RATE
    )
    max_gap_error = float(np.max(np.abs(rk4_states[:, 0] - exact_states[:, 0])))
    # halving the step should make the error about 16 times smaller
    convergence_order = (
        float(np.log2(previous_max_error / max_gap_error))
        if previous_max_error is not None
        else float("nan")
    )
    print(f"{step_size:<13} {max_gap_error:<18.3e} {convergence_order:.2f}")
    previous_max_error = max_gap_error
    if step_size == 1.0:
        one_day_times, one_day_states, one_day_exact_states = times, rk4_states, exact_states


momentum_strength_by_case: dict[str, float] = {"stable": 0.01, "unstable": 0.02}
trajectories: dict[str, tuple[FloatArray, FloatArray]] = {}
for case_name, momentum_strength in momentum_strength_by_case.items():
    case_params: ModelParameters = {
        "mean_reversion_rate": MEAN_REVERSION_RATE,
        "memory_rate": MEMORY_RATE,
        "momentum_strength": momentum_strength,
        "trend_sensitivity": TREND_SENSITIVITY,
    }
    times, states = integrate_rk4([0.01, 0.0], 300.0, 0.125, case_params)
    trajectories[case_name] = (times, states)

    # compare against scipy's solver
    scipy_reference = solve_ivp(
        lambda _time, state: compute_derivative(state, **case_params),
        (0, 300),
        [0.01, 0.0],
        method = "DOP853",
        t_eval = times,
        rtol = 1e-11,
        atol = [1e-13, 1e-15],
    )
    max_diff_from_scipy = float(np.max(np.abs(states[:, 0] - scipy_reference.y[0])))

    linearised_jacobian = np.array([
        [-MEAN_REVERSION_RATE, momentum_strength * TREND_SENSITIVITY],
        [
            -MEMORY_RATE * MEAN_REVERSION_RATE,
            MEMORY_RATE * (momentum_strength * TREND_SENSITIVITY - 1),
        ],
    ])
    eigenvalues = np.linalg.eigvals(linearised_jacobian)

    momentum_gain = momentum_strength * TREND_SENSITIVITY
    rate_ratio = MEAN_REVERSION_RATE / MEMORY_RATE
    print(
        f"\n{case_name}: beta = {momentum_strength}, b = {momentum_gain:g}, "
        f"k = {rate_ratio:g}, stable if b < {1 + rate_ratio:g}"
    )
    print(
        f"  trace = {np.trace(linearised_jacobian):.4f}, "
        f"det = {np.linalg.det(linearised_jacobian):.4f}"
    )
    print(f"  eigenvalues = {eigenvalues[0]:.4f} and {eigenvalues[1]:.4f}")
    print(f"  biggest |gap| in the last 100 days = {np.max(np.abs(states[times >= 200, 0])):.3e}")
    print(f"  max difference from scipy dop853 = {max_diff_from_scipy:.1e}")


figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(one_day_times, one_day_exact_states[:, 0], "k-", label="exact")
axes[0].plot(
    one_day_times[::10], one_day_states[::10, 0], "ko", mfc="white", label="rk4, h = 1 day"
)
axes[0].set_xlabel("time (days)")
axes[0].set_ylabel("gap x (momentum off)")
axes[0].legend()
for case_name, (times, states) in trajectories.items():
    axes[1].plot(
        times, states[:, 0], label=f"{case_name}, beta = {momentum_strength_by_case[case_name]}"
    )
axes[1].set_xlabel("time (days)")
axes[1].set_ylabel("gap x")
axes[1].legend()

figure.tight_layout()
figure.savefig("results/ode_model.png", dpi=150)