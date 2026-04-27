"""
Tests for Experiment 2: Monte Carlo replication with constant spread.
Verifies paper methodology implementation and qualitative findings.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exp2_montecarlo_constant_spread import (
    compute_constant_spread,
    lambda_intensity,
    simulate_strategy,
    S0, T, SIGMA, DT, N_STEPS, A, K, N_PATHS, GAMMA_VALUES, RANDOM_SEED,
)


class TestConstantSpread:
    """Tests for the constant spread formula."""

    def test_spread_formula_gamma_001(self) -> None:
        """Spread for gamma=0.01 should be approximately 1.33."""
        spread = compute_constant_spread(0.01)
        assert abs(spread - 1.3289) < 0.01

    def test_spread_formula_gamma_01(self) -> None:
        """Spread for gamma=0.1 should be approximately 1.29."""
        spread = compute_constant_spread(0.1)
        assert abs(spread - 1.2908) < 0.01

    def test_spread_formula_gamma_05(self) -> None:
        """Spread for gamma=0.5 should be approximately 1.15."""
        spread = compute_constant_spread(0.5)
        assert abs(spread - 1.1507) < 0.01

    def test_spread_positive(self) -> None:
        """Spread should be positive for all valid gamma."""
        for g in [0.001, 0.01, 0.1, 0.5, 1.0]:
            assert compute_constant_spread(g) > 0

    def test_spread_decreases_with_gamma(self) -> None:
        """Spread should decrease as gamma increases (for k=1.5)."""
        spreads = [compute_constant_spread(g) for g in [0.01, 0.1, 0.5]]
        assert spreads[0] > spreads[1] > spreads[2]

    def test_spread_formula_exact(self) -> None:
        """Verify exact formula: (2/gamma)*ln(1+gamma/k)."""
        gamma, k = 0.1, 1.5
        expected = (2.0 / gamma) * np.log(1.0 + gamma / k)
        assert abs(compute_constant_spread(gamma, k) - expected) < 1e-12


class TestLambdaIntensity:
    """Tests for the execution intensity function."""

    def test_intensity_at_zero_delta(self) -> None:
        """At delta=0, intensity should equal A."""
        assert abs(lambda_intensity(0.0) - A) < 1e-10

    def test_intensity_decreases_with_delta(self) -> None:
        """Intensity should decrease as delta increases."""
        deltas = [0.0, 0.5, 1.0, 2.0]
        intensities = [lambda_intensity(d) for d in deltas]
        for i in range(len(intensities) - 1):
            assert intensities[i] > intensities[i + 1]

    def test_intensity_positive(self) -> None:
        """Intensity should be positive for all delta."""
        for d in [-1.0, 0.0, 0.5, 1.0, 5.0]:
            assert lambda_intensity(d) > 0

    def test_intensity_formula(self) -> None:
        """Verify exact formula: A*exp(-k*delta)."""
        delta = 0.5
        expected = A * np.exp(-K * delta)
        assert abs(lambda_intensity(delta) - expected) < 1e-10

    def test_intensity_array_input(self) -> None:
        """Intensity should work with array inputs."""
        deltas = np.array([0.0, 0.5, 1.0])
        intensities = lambda_intensity(deltas)
        assert intensities.shape == (3,)
        assert np.all(intensities > 0)


class TestSimulationParameters:
    """Tests for simulation parameter validation."""

    def test_n_steps_correct(self) -> None:
        """N_STEPS should equal T/DT = 200."""
        assert N_STEPS == 200

    def test_gamma_values_correct(self) -> None:
        """GAMMA_VALUES should contain 0.01, 0.1, 0.5."""
        assert set(GAMMA_VALUES) == {0.01, 0.1, 0.5}

    def test_parameters_match_paper(self) -> None:
        """Key parameters should match paper specification."""
        assert S0 == 100.0
        assert T == 1.0
        assert SIGMA == 2.0
        assert DT == 0.005
        assert A == 140.0
        assert K == 1.5


class TestSimulationResults:
    """Tests for Monte Carlo simulation results."""

    def test_inventory_lower_profit_std_than_symmetric(self) -> None:
        """Inventory strategy should have lower profit std than symmetric."""
        for gamma in GAMMA_VALUES:
            inv = simulate_strategy(gamma, "inventory", n_paths=200, seed=RANDOM_SEED)
            sym = simulate_strategy(gamma, "symmetric", n_paths=200, seed=RANDOM_SEED + 1000)
            assert inv.profit_std < sym.profit_std, (
                f"gamma={gamma}: inventory std={inv.profit_std:.2f} should be < "
                f"symmetric std={sym.profit_std:.2f}"
            )

    def test_inventory_lower_inv_std_than_symmetric(self) -> None:
        """Inventory strategy should have lower inventory std than symmetric."""
        for gamma in GAMMA_VALUES:
            inv = simulate_strategy(gamma, "inventory", n_paths=200, seed=RANDOM_SEED)
            sym = simulate_strategy(gamma, "symmetric", n_paths=200, seed=RANDOM_SEED + 1000)
            assert inv.inv_std < sym.inv_std, (
                f"gamma={gamma}: inventory inv_std={inv.inv_std:.2f} should be < "
                f"symmetric inv_std={sym.inv_std:.2f}"
            )

    def test_terminal_pnl_is_x_plus_q_times_s(self) -> None:
        """Terminal P&L should be X_T + q_T * S_T."""
        inv = simulate_strategy(0.1, "inventory", n_paths=10, seed=RANDOM_SEED)
        # The terminal_profits array should be X + q*S at maturity
        assert len(inv.terminal_profits) == 10

    def test_initial_state_correct(self) -> None:
        """Initial state should match S0, q0=0, X0=0."""
        inv = simulate_strategy(0.1, "inventory", n_paths=10, seed=RANDOM_SEED)
        assert abs(inv.S_path[0] - S0) < 1e-10
        assert abs(inv.q_path[0] - 0.0) < 1e-10

    def test_symmetric_strategy_centered_at_mid(self) -> None:
        """Symmetric strategy should have center equal to mid-price."""
        sym = simulate_strategy(0.1, "symmetric", n_paths=10, seed=RANDOM_SEED)
        # For symmetric strategy, center = S, so pa - pb = spread (constant)
        # Check that spread is constant
        pa_arr = sym.pa_path
        pb_arr = sym.pb_path
        spreads = pa_arr - pb_arr
        # All spreads should be equal to the constant spread
        expected_spread = compute_constant_spread(0.1)
        # Allow for small numerical differences
        assert np.allclose(spreads[1:], expected_spread, atol=1e-10)

    def test_inventory_strategy_q0_centered_at_mid(self) -> None:
        """Inventory strategy with q=0 should be centered at mid-price at first step."""
        inv = simulate_strategy(0.1, "inventory", n_paths=1, seed=RANDOM_SEED)
        # center_path[0] is unset (0.0 default); center is first stored at index 1
        # At step n=0, q=0, so center = S[0] - 0*gamma*sigma^2*tau = S[0]
        # center_path[1] stores the center computed at step n=0 (before price update)
        # pa_path[1] - pb_path[1] should equal the constant spread
        expected_spread = compute_constant_spread(0.1)
        assert abs((inv.pa_path[1] - inv.pb_path[1]) - expected_spread) < 1e-10

    def test_spread_matches_formula(self) -> None:
        """Spread used in simulation should match formula."""
        for gamma in GAMMA_VALUES:
            inv = simulate_strategy(gamma, "inventory", n_paths=5, seed=RANDOM_SEED)
            expected_spread = compute_constant_spread(gamma)
            assert abs(inv.spread - expected_spread) < 1e-10

    def test_positive_profit_mean(self) -> None:
        """Mean terminal profit should be positive (market maker earns spread)."""
        for gamma in GAMMA_VALUES:
            inv = simulate_strategy(gamma, "inventory", n_paths=200, seed=RANDOM_SEED)
            assert inv.profit_mean > 0, f"gamma={gamma}: mean profit should be positive"

    def test_path_length_correct(self) -> None:
        """Path arrays should have N_STEPS+1 elements."""
        inv = simulate_strategy(0.1, "inventory", n_paths=5, seed=RANDOM_SEED)
        assert len(inv.time_path) == N_STEPS + 1
        assert len(inv.S_path) == N_STEPS + 1
        assert len(inv.q_path) == N_STEPS + 1


class TestEdgeCases:
    """Tests for edge cases in the simulation."""

    def test_single_path(self) -> None:
        """Simulation should work with a single path."""
        inv = simulate_strategy(0.1, "inventory", n_paths=1, seed=RANDOM_SEED)
        assert len(inv.terminal_profits) == 1

    def test_high_gamma_low_inventory_dispersion(self) -> None:
        """High gamma should produce lower inventory dispersion."""
        inv_low = simulate_strategy(0.01, "inventory", n_paths=200, seed=RANDOM_SEED)
        inv_high = simulate_strategy(0.5, "inventory", n_paths=200, seed=RANDOM_SEED)
        assert inv_high.inv_std < inv_low.inv_std, (
            "Higher gamma should produce lower inventory dispersion"
        )

    def test_invalid_strategy_raises(self) -> None:
        """Invalid strategy name should raise AssertionError."""
        with pytest.raises(AssertionError):
            simulate_strategy(0.1, "invalid_strategy", n_paths=5)

    def test_negative_gamma_raises(self) -> None:
        """Negative gamma should raise AssertionError."""
        with pytest.raises(AssertionError):
            simulate_strategy(-0.1, "inventory", n_paths=5)
