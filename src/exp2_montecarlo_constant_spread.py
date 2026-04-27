"""
Experiment 2: Primary numerical replication of Avellaneda-Stoikov Monte Carlo results
using the table-faithful constant-spread implementation.

Spread: Delta = (2/gamma)*ln(1+gamma/k)  [constant, table-faithful]
Mid-price: S_{t+dt} = S_t +/- sigma*sqrt(dt) with equal probability
Execution: Bernoulli with probability min(lambda(delta)*dt, 1)
# Custom Monte Carlo simulation -- Context7 found no library equivalent (paper methodology)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Model parameters (paper Section 4)
# ---------------------------------------------------------------------------
S0: float = 100.0
T: float = 1.0
SIGMA: float = 2.0
DT: float = 0.005
N_STEPS: int = int(T / DT)   # 200
Q0: int = 0
X0: float = 0.0
A: float = 140.0
K: float = 1.5
N_PATHS: int = 1000
GAMMA_VALUES: list[float] = [0.01, 0.1, 0.5]
RANDOM_SEED: int = 42


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results."""

    gamma: float
    strategy: str
    spread: float
    profit_mean: float
    profit_std: float
    inv_mean: float
    inv_std: float
    # Path-level arrays for one illustrative run
    time_path: np.ndarray = field(default_factory=lambda: np.array([]))
    S_path: np.ndarray = field(default_factory=lambda: np.array([]))
    center_path: np.ndarray = field(default_factory=lambda: np.array([]))
    pa_path: np.ndarray = field(default_factory=lambda: np.array([]))
    pb_path: np.ndarray = field(default_factory=lambda: np.array([]))
    q_path: np.ndarray = field(default_factory=lambda: np.array([]))
    X_path: np.ndarray = field(default_factory=lambda: np.array([]))
    # Aggregate arrays
    terminal_profits: np.ndarray = field(default_factory=lambda: np.array([]))
    terminal_inventories: np.ndarray = field(default_factory=lambda: np.array([]))
    n_prob_capped: int = 0
    n_negative_delta: int = 0


def compute_constant_spread(gamma: float, k: float = K) -> float:
    """
    Compute the table-faithful constant spread.

    Delta = (2/gamma)*ln(1+gamma/k)

    Parameters
    ----------
    gamma : risk-aversion coefficient
    k : intensity decay parameter

    Returns
    -------
    float spread value
    """
    return (2.0 / gamma) * np.log(1.0 + gamma / k)


def lambda_intensity(delta: float | np.ndarray, a: float = A, k: float = K) -> float | np.ndarray:
    """
    Compute execution intensity for a given quote distance.

    lambda(delta) = A * exp(-k * delta)

    Parameters
    ----------
    delta : quote distance (can be negative if quoting through mid)
    a : intensity scale parameter
    k : intensity decay parameter

    Returns
    -------
    float or array of intensities
    """
    return a * np.exp(-k * delta)


def simulate_strategy(
    gamma: float,
    strategy: str,
    n_paths: int = N_PATHS,
    seed: int = RANDOM_SEED,
    store_path_idx: int = 0,
) -> SimulationResult:
    """
    Simulate the inventory or symmetric market-making strategy.

    Uses the table-faithful constant spread Delta = (2/gamma)*ln(1+gamma/k).

    Parameters
    ----------
    gamma : risk-aversion coefficient
    strategy : 'inventory' or 'symmetric'
    n_paths : number of Monte Carlo paths
    seed : random seed for reproducibility
    store_path_idx : index of path to store for illustration

    Returns
    -------
    SimulationResult with aggregate statistics and one illustrative path.
    """
    assert strategy in ("inventory", "symmetric"), f"Unknown strategy: {strategy}"
    assert gamma > 0, "gamma must be positive"
    assert n_paths > 0, "n_paths must be positive"

    rng = np.random.default_rng(seed)
    spread = compute_constant_spread(gamma)

    # Pre-generate all random numbers for reproducibility
    # Price shocks: +1 or -1 with equal probability
    price_shocks = rng.choice([-1.0, 1.0], size=(n_paths, N_STEPS))
    # Uniform draws for execution events (ask and bid independently)
    u_ask = rng.uniform(0.0, 1.0, size=(n_paths, N_STEPS))
    u_bid = rng.uniform(0.0, 1.0, size=(n_paths, N_STEPS))

    # State arrays
    S = np.full(n_paths, S0, dtype=float)
    q = np.zeros(n_paths, dtype=float)
    X = np.full(n_paths, X0, dtype=float)

    # Storage for illustrative path
    time_arr = np.zeros(N_STEPS + 1)
    S_arr = np.zeros(N_STEPS + 1)
    center_arr = np.zeros(N_STEPS + 1)
    pa_arr = np.zeros(N_STEPS + 1)
    pb_arr = np.zeros(N_STEPS + 1)
    q_arr = np.zeros(N_STEPS + 1)
    X_arr = np.zeros(N_STEPS + 1)

    # Record initial state for illustrative path
    S_arr[0] = S[store_path_idx]
    q_arr[0] = q[store_path_idx]
    X_arr[0] = X[store_path_idx]

    n_prob_capped = 0
    n_negative_delta = 0

    for n in range(N_STEPS):
        t_now = n * DT
        tau = T - t_now

        # Compute quote center
        if strategy == "inventory":
            # Reservation center: r_t = S_t - q_t * gamma * sigma^2 * (T-t)
            center = S - q * gamma * SIGMA**2 * tau
        else:
            # Symmetric: center at mid-price
            center = S.copy()

        # Compute quotes
        pa = center + spread / 2.0
        pb = center - spread / 2.0

        # Quote distances from mid-price
        delta_a = pa - S   # should be >= 0 for normal operation
        delta_b = S - pb   # should be >= 0 for normal operation

        # Track negative quote distances
        n_negative_delta += int(np.sum(delta_a < 0) + np.sum(delta_b < 0))

        # Execution intensities
        lam_a = lambda_intensity(delta_a)
        lam_b = lambda_intensity(delta_b)

        # Bernoulli execution probabilities (capped at 1)
        prob_a = np.minimum(lam_a * DT, 1.0)
        prob_b = np.minimum(lam_b * DT, 1.0)

        # Count capping events
        n_prob_capped += int(np.sum(lam_a * DT > 1.0) + np.sum(lam_b * DT > 1.0))

        # Simulate executions
        ask_exec = u_ask[:, n] < prob_a
        bid_exec = u_bid[:, n] < prob_b

        # Update state: ask execution => sell 1 share
        X += ask_exec * pa
        q -= ask_exec.astype(float)

        # Update state: bid execution => buy 1 share
        X -= bid_exec * pb
        q += bid_exec.astype(float)

        # Evolve mid-price: S <- S + epsilon * sigma * sqrt(dt)
        S += price_shocks[:, n] * SIGMA * np.sqrt(DT)

        # Store illustrative path
        idx = store_path_idx
        time_arr[n + 1] = (n + 1) * DT
        S_arr[n + 1] = S[idx]
        center_arr[n + 1] = center[idx]
        pa_arr[n + 1] = pa[idx]
        pb_arr[n + 1] = pb[idx]
        q_arr[n + 1] = q[idx]
        X_arr[n + 1] = X[idx]

    # Terminal P&L
    terminal_profits = X + q * S
    terminal_inventories = q.copy()

    return SimulationResult(
        gamma=gamma,
        strategy=strategy,
        spread=spread,
        profit_mean=float(np.mean(terminal_profits)),
        profit_std=float(np.std(terminal_profits)),
        inv_mean=float(np.mean(terminal_inventories)),
        inv_std=float(np.std(terminal_inventories)),
        time_path=time_arr,
        S_path=S_arr,
        center_path=center_arr,
        pa_path=pa_arr,
        pb_path=pb_arr,
        q_path=q_arr,
        X_path=X_arr,
        terminal_profits=terminal_profits,
        terminal_inventories=terminal_inventories,
        n_prob_capped=n_prob_capped,
        n_negative_delta=n_negative_delta,
    )


def plot_sample_path(result: SimulationResult, out_path: Path) -> None:
    """
    Plot one illustrative sample path showing mid-price, reservation center,
    bid/ask quotes, and inventory.

    Parameters
    ----------
    result : SimulationResult with path arrays
    out_path : output file path
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(result.time_path, result.S_path, label="Mid-price S_t", color="black", linewidth=1.5)
    ax1.plot(
        result.time_path, result.center_path,
        label="Quote center (reservation r_t)", color="blue", linewidth=1.2, linestyle="--"
    )
    ax1.plot(
        result.time_path, result.pa_path,
        label="Ask quote p^a_t", color="red", linewidth=0.8, alpha=0.7
    )
    ax1.plot(
        result.time_path, result.pb_path,
        label="Bid quote p^b_t", color="green", linewidth=0.8, alpha=0.7
    )
    ax1.set_ylabel("Price")
    ax1.set_title(
        f"Sample Path: {result.strategy.capitalize()} Strategy "
        f"(gamma={result.gamma}, spread={result.spread:.4f})"
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(result.time_path, result.q_path, label="Inventory q_t", color="purple", linewidth=1.2)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Inventory q_t")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[exp2] Saved sample path: {out_path}")


def plot_profit_histograms(
    inv_result: SimulationResult,
    sym_result: SimulationResult,
    out_path: Path,
) -> None:
    """
    Plot terminal profit histograms comparing inventory and symmetric strategies.

    Parameters
    ----------
    inv_result : SimulationResult for inventory strategy
    sym_result : SimulationResult for symmetric strategy
    out_path : output file path
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(
        min(inv_result.terminal_profits.min(), sym_result.terminal_profits.min()),
        max(inv_result.terminal_profits.max(), sym_result.terminal_profits.max()),
        50,
    )
    ax.hist(
        inv_result.terminal_profits, bins=bins, alpha=0.6, label="Inventory strategy",
        color="blue", density=True
    )
    ax.hist(
        sym_result.terminal_profits, bins=bins, alpha=0.6, label="Symmetric strategy",
        color="red", density=True
    )
    ax.set_xlabel("Terminal P&L")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Terminal Profit Distribution (gamma={inv_result.gamma}, "
        f"constant spread={inv_result.spread:.4f})\n"
        f"Inventory: mean={inv_result.profit_mean:.2f}, std={inv_result.profit_std:.2f}  |  "
        f"Symmetric: mean={sym_result.profit_mean:.2f}, std={sym_result.profit_std:.2f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[exp2] Saved histogram: {out_path}")


def run_experiment_2() -> list[dict]:
    """
    Run the primary Monte Carlo replication with table-faithful constant spread.

    Returns
    -------
    list of result dicts, one per gamma value.
    """
    print("=" * 70)
    print("EXPERIMENT 2: Monte Carlo Replication (Constant Spread)")
    print("=" * 70)
    print(f"Parameters: S0={S0}, T={T}, sigma={SIGMA}, dt={DT}, N={N_STEPS}")
    print(f"A={A}, k={K}, M={N_PATHS}, gamma_values={GAMMA_VALUES}")

    all_results = []
    summary_rows = []

    for gamma in GAMMA_VALUES:
        print(f"\n--- gamma = {gamma} ---")
        spread = compute_constant_spread(gamma)
        print(f"    Constant spread Delta = {spread:.4f}")

        # Run inventory strategy
        inv_res = simulate_strategy(gamma, "inventory", seed=RANDOM_SEED)
        # Run symmetric strategy (same underlying price path via same seed offset)
        sym_res = simulate_strategy(gamma, "symmetric", seed=RANDOM_SEED + 1000)

        print(f"    Inventory:  profit={inv_res.profit_mean:.2f} +/- {inv_res.profit_std:.2f}, "
              f"inv={inv_res.inv_mean:.3f} +/- {inv_res.inv_std:.2f}")
        print(f"    Symmetric:  profit={sym_res.profit_mean:.2f} +/- {sym_res.profit_std:.2f}, "
              f"inv={sym_res.inv_mean:.3f} +/- {sym_res.inv_std:.2f}")
        print(f"    Prob-capped events (inv/sym): {inv_res.n_prob_capped}/{sym_res.n_prob_capped}")
        print(f"    Negative delta events (inv/sym): "
              f"{inv_res.n_negative_delta}/{sym_res.n_negative_delta}")

        # Qualitative checks
        assert inv_res.profit_std < sym_res.profit_std, (
            f"gamma={gamma}: inventory std should be < symmetric std"
        )
        assert inv_res.inv_std < sym_res.inv_std, (
            f"gamma={gamma}: inventory inv_std should be < symmetric inv_std"
        )

        # Sample path plot for inventory strategy
        plot_sample_path(
            inv_res,
            RESULTS_DIR / f"exp2_sample_path_gamma{gamma}.png",
        )

        # Profit histogram
        plot_profit_histograms(
            inv_res, sym_res,
            RESULTS_DIR / f"exp2_profit_hist_gamma{gamma}.png",
        )

        summary_rows.append({
            "gamma": gamma,
            "spread": round(spread, 4),
            "inv_profit_mean": round(inv_res.profit_mean, 2),
            "inv_profit_std": round(inv_res.profit_std, 2),
            "inv_inv_mean": round(inv_res.inv_mean, 3),
            "inv_inv_std": round(inv_res.inv_std, 2),
            "sym_profit_mean": round(sym_res.profit_mean, 2),
            "sym_profit_std": round(sym_res.profit_std, 2),
            "sym_inv_mean": round(sym_res.inv_mean, 3),
            "sym_inv_std": round(sym_res.inv_std, 2),
        })

        all_results.append({
            "gamma": gamma,
            "inventory": inv_res,
            "symmetric": sym_res,
            "summary": summary_rows[-1],
        })

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (Constant Spread)")
    print("=" * 70)
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    # Paper reference values
    paper_ref = {
        0.1: {"inv": (62.94, 5.89, 0.10, 2.80), "sym": (67.21, 13.43, -0.018, 8.66)},
        0.01: {"inv": (66.78, 8.76, -0.02, 4.70), "sym": (67.36, 13.40, -0.31, 8.65)},
        0.5: {"inv": (33.92, 4.72, -0.02, 1.88), "sym": (66.20, 14.53, 0.25, 9.06)},
    }
    print("\nPaper reference values (gamma: inv(mean,std,q_mean,q_std), sym(...)):")
    for g, vals in paper_ref.items():
        print(f"  gamma={g}: inv={vals['inv']}, sym={vals['sym']}")

    print("\n[exp2] Completed successfully.")
    return all_results


if __name__ == "__main__":
    run_experiment_2()
