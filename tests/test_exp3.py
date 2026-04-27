"""
Tests for Experiment 3: Monte Carlo replication with time-varying spread.
Verifies derivation-faithful spread implementation and qualitative findings.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exp3_montecarlo_timevarying_spread import (
    compute_timevarying_spread,
    lambda_intensity,
    simulate_strategy_tv,
    S0, T, SIGMA, DT, N_STEPS, A, K, N_PATHS, GAMMA_VALUES, RANDOM_SEED,
)


class TestTimeVaryingSpread:
    """Tests for the time-varying spread formula."""

    def test_spread_at_t0(self) -> None:
        """Spread at t=0 (tau=T) should include time-varying term."""
        gamma = 0.1
        spread_t0 = compute_timevarying_spread(gamma, T)
        constant_part = (2.0 / gamma) * np.log(1.0 + gamma / K)
        time_varying_part = gamma * SIGMA**2 * T
        expected = constant_part + time_varying_part
        assert abs(spread_t0 - expected) < 1e-10

    def test_spread_at_tT(self) -> None:
        """Spread at t=T (tau=0) should equal constant spread."""
        gamma = 0.1
        spread_tT = compute_timevarying_spread(gamma, 0.0)
        constant_spread = (2.0 / gamma) * np.log(1.0 + gamma / K)
        assert abs(spread_tT - constant_spread) < 1e-10

    def test_spread_decreases_over_time(self) -> None:
        """Spread should decrease as tau decreases (time passes)."""
        gamma = 0.1
        taus = [1.0, 0.5, 0.1, 0.0]
        spreads = [compute_timevarying_spread(gamma, tau) for tau in taus]
        for i in range(len(spreads) - 1):
            assert spreads[i] >= spreads[i + 1], (
                f"Spread should decrease: tau={taus[i]}: {spreads[i]:.4f} >= "
                f"tau={taus[i+1]}: {spreads[i+1]:.4f}"
            )

    def test_spread_positive_for_all_tau(self) -> None:
        """Spread should be positive for all tau >= 0."""
        gamma = 0.1
        for tau in [0.0, 0.1, 0.5, 1.0]:
            assert compute_timevarying_spread(gamma, tau) > 0

    def test_spread_larger_than_constant(self) -> None:
        """Time-varying spread should be >= constant spread for tau > 0."""
        gamma = 0.1
        constant = (2.0 / gamma) * np.log(1.0 + gamma / K)
        for tau in [0.1, 0.5, 1.0]:
            tv_spread = compute_timevarying_spread(gamma, tau)
            assert tv_spread >= constant - 1e-10

    def test_spread_formula_exact(self) -> None:
        """Verify exact formula: gamma*sigma^2*tau + (2/gamma)*ln(1+gamma/k)."""
        gamma, tau = 0.1, 0.5
        expected = gamma * SIGMA**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / K)
        assert abs(compute_timevarying_spread(gamma, tau) - expected) < 1e-10


class TestSimulationResultsTV:
    """Tests for Monte Carlo simulation results with time-varying spread."""

    def test_inventory_lower_profit_std_than_symmetric(self) -> None:
        """Inventory strategy should have lower profit std than symmetric."""
        for gamma in GAMMA_VALUES:
            inv = simulate_strategy_tv(gamma, "inventory", n_paths=200, seed=RANDOM_SEED)
            sym = simulate_strategy_tv(gamma, "symmetric", n_paths=200, seed=RANDOM_SEED + 1000)
            assert inv.profit_std < sym.profit_std, (
                f"gamma={gamma}: inventory std={inv.profit_std:.2f} should be < "
                f"symmetric std={sym.profit_std:.2f}"
            )

    def test_inventory_lower_inv_std_than_symmetric(self) -> None:
        """Inventory strategy should have lower inventory std than symmetric."""
        for gamma in GAMMA_VALUES:
            inv = simulate_strategy_tv(gamma, "inventory", n_paths=200, seed=RANDOM_SEED)
            sym = simulate_strategy_tv(gamma, "symmetric", n_paths=200, seed=RANDOM_SEED + 1000)
            assert inv.inv_std < sym.inv_std, (
                f"gamma={gamma}: inventory inv_std={inv.inv_std:.2f} should be < "
                f"symmetric inv_std={sym.inv_std:.2f}"
            )

    def test_spread_boundary_conditions(self) -> None:
        """Spread at t=T should equal constant spread."""
        for gamma in GAMMA_VALUES:
            inv = simulate_strategy_tv(gamma, "inventory", n_paths=5, seed=RANDOM_SEED)
            constant_spread = (2.0 / gamma) * np.log(1.0 + gamma / K)
            assert abs(inv.spread_terminal - constant_spread) < 1e-10

    def test_spread_path_decreasing(self) -> None:
        """Spread path should be monotonically decreasing."""
        inv = simulate_strategy_tv(0.1, "inventory", n_paths=1, seed=RANDOM_SEED)
        spread_path = inv.spread_path
        # Spread should decrease over time (or stay same at last step)
        diffs = np.diff(spread_path)
        assert np.all(diffs <= 1e-10), "Spread path should be non-increasing"

    def test_initial_state_correct(self) -> None:
        """Initial state should match S0, q0=0."""
        inv = simulate_strategy_tv(0.1, "inventory", n_paths=5, seed=RANDOM_SEED)
        assert abs(inv.S_path[0] - S0) < 1e-10
        assert abs(inv.q_path[0] - 0.0) < 1e-10

    def test_path_length_correct(self) -> None:
        """Path arrays should have N_STEPS+1 elements."""
        inv = simulate_strategy_tv(0.1, "inventory", n_paths=5, seed=RANDOM_SEED)
        assert len(inv.time_path) == N_STEPS + 1
        assert len(inv.spread_path) == N_STEPS + 1

    def test_positive_profit_mean(self) -> None:
        """Mean terminal profit should be positive."""
        for gamma in GAMMA_VALUES:
            inv = simulate_strategy_tv(gamma, "inventory", n_paths=200, seed=RANDOM_SEED)
            assert inv.profit_mean > 0, f"gamma={gamma}: mean profit should be positive"


class TestComparisonWithConstantSpread:
    """Tests comparing time-varying vs constant spread implementations."""

    def test_tv_spread_wider_early(self) -> None:
        """Time-varying spread should be wider than constant spread early in horizon."""
        gamma = 0.1
        tv_spread_t0 = compute_timevarying_spread(gamma, T)
        constant_spread = (2.0 / gamma) * np.log(1.0 + gamma / K)
        assert tv_spread_t0 > constant_spread

    def test_tv_spread_equals_constant_at_T(self) -> None:
        """Time-varying spread should equal constant spread at maturity."""
        gamma = 0.1
        tv_spread_tT = compute_timevarying_spread(gamma, 0.0)
        constant_spread = (2.0 / gamma) * np.log(1.0 + gamma / K)
        assert abs(tv_spread_tT - constant_spread) < 1e-10

    def test_qualitative_finding_robust(self) -> None:
        """Qualitative risk-control finding should hold under time-varying spread."""
        # The inventory strategy should still have lower risk than symmetric
        gamma = 0.1
        inv = simulate_strategy_tv(gamma, "inventory", n_paths=200, seed=RANDOM_SEED)
        sym = simulate_strategy_tv(gamma, "symmetric", n_paths=200, seed=RANDOM_SEED + 1000)
        assert inv.profit_std < sym.profit_std
        assert inv.inv_std < sym.inv_std


class TestEdgeCasesTV:
    """Tests for edge cases in the time-varying spread simulation."""

    def test_single_path(self) -> None:
        """Simulation should work with a single path."""
        inv = simulate_strategy_tv(0.1, "inventory", n_paths=1, seed=RANDOM_SEED)
        assert len(inv.terminal_profits) == 1

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy name should raise AssertionError."""
        with pytest.raises(AssertionError):
            simulate_strategy_tv(0.1, "invalid_strategy", n_paths=5)

    def test_negative_gamma_raises(self) -> None:
        """Negative gamma should raise AssertionError."""
        with pytest.raises(AssertionError):
            simulate_strategy_tv(-0.1, "inventory", n_paths=5)
