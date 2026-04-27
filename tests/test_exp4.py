"""
Tests for Experiment 4: GBM mean-variance appendix derivation.
Verifies the alternative model's analytical formulas.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np
import sympy as sp
from sympy import symbols, exp, simplify

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exp4_gbm_meanvariance import (
    derive_gbm_moments,
    derive_gbm_value_function,
    derive_gbm_reservation_prices,
    compare_models,
    s, q, t, T, x, gamma, sigma,
)


class TestGBMMoments:
    """Tests for GBM conditional moments."""

    def test_conditional_mean_equals_s(self) -> None:
        """Under zero-drift GBM, E_t[S_T] = s."""
        moments = derive_gbm_moments()
        assert simplify(moments["E_ST"] - s) == 0

    def test_conditional_variance_formula(self) -> None:
        """Var_t(S_T) should equal s^2*(exp(sigma^2*(T-t))-1)."""
        moments = derive_gbm_moments()
        tau = T - t
        expected = s**2 * (exp(sigma**2 * tau) - 1)
        assert simplify(moments["Var_ST"] - expected) == 0

    def test_variance_zero_at_T(self) -> None:
        """Variance should be zero at t=T."""
        moments = derive_gbm_moments()
        var_T = moments["Var_ST"].subs(t, T)
        assert simplify(var_T) == 0

    def test_variance_positive_before_T(self) -> None:
        """Variance should be positive for t < T."""
        moments = derive_gbm_moments()
        var_num = float(moments["Var_ST"].subs(
            [(s, 100), (sigma, 0.2), (t, 0), (T, 1)]
        ))
        assert var_num > 0


class TestGBMValueFunction:
    """Tests for the GBM mean-variance value function."""

    def test_value_function_formula(self) -> None:
        """V should equal x + q*s - (gamma*q^2*s^2/2)*(exp(sigma^2*(T-t))-1)."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        tau = T - t
        expected = x + q * s - (gamma * q**2 * s**2 / 2) * (exp(sigma**2 * tau) - 1)
        assert simplify(vf["V"] - expected) == 0

    def test_terminal_condition(self) -> None:
        """At t=T, V should equal x + q*s."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        V_T = vf["V"].subs(t, T)
        assert simplify(V_T - (x + q * s)) == 0

    def test_value_function_linear_in_x(self) -> None:
        """V should be linear in x (coefficient 1)."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        # dV/dx = 1
        dV_dx = sp.diff(vf["V"], x)
        assert simplify(dV_dx - 1) == 0

    def test_description_contains_key_terms(self) -> None:
        """Description should mention key formula components."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        desc = vf["description"]
        assert "gamma" in desc
        assert "sigma" in desc
        assert "exp" in desc


class TestGBMReservationPrices:
    """Tests for GBM mean-variance reservation prices."""

    def test_R_a_formula(self) -> None:
        """R^a should equal s + ((1-2q)/2)*gamma*s^2*(exp(sigma^2*(T-t))-1)."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        tau = T - t
        R_a_paper = s + sp.Rational(1, 2) * (1 - 2 * q) * gamma * s**2 * (exp(sigma**2 * tau) - 1)
        assert simplify(res["R_a"] - R_a_paper) == 0

    def test_R_b_formula(self) -> None:
        """R^b should equal s + ((-1-2q)/2)*gamma*s^2*(exp(sigma^2*(T-t))-1)."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        tau = T - t
        R_b_paper = s + sp.Rational(1, 2) * (-1 - 2 * q) * gamma * s**2 * (exp(sigma**2 * tau) - 1)
        assert simplify(res["R_b"] - R_b_paper) == 0

    def test_midpoint_formula(self) -> None:
        """Midpoint should equal s - q*gamma*s^2*(exp(sigma^2*(T-t))-1)."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        tau = T - t
        midpoint_paper = s - q * gamma * s**2 * (exp(sigma**2 * tau) - 1)
        assert simplify(res["midpoint"] - midpoint_paper) == 0

    def test_spread_formula(self) -> None:
        """Spread R^a - R^b should equal gamma*s^2*(exp(sigma^2*(T-t))-1)."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        tau = T - t
        spread_paper = gamma * s**2 * (exp(sigma**2 * tau) - 1)
        assert simplify(res["spread"] - spread_paper) == 0

    def test_terminal_conditions(self) -> None:
        """At t=T, both R^a and R^b should collapse to s."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        assert simplify(res["R_a"].subs(t, T) - s) == 0
        assert simplify(res["R_b"].subs(t, T) - s) == 0

    def test_midpoint_at_q0_equals_s(self) -> None:
        """At q=0, midpoint should equal s."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        mid_q0 = res["midpoint"].subs(q, 0)
        assert simplify(mid_q0 - s) == 0

    def test_inventory_sign_convention(self) -> None:
        """For q>0, midpoint should be below s; for q<0, above s."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        params_pos = [(q, 2), (gamma, 0.01), (sigma, 0.2), (t, 0), (T, 1), (s, 100)]
        params_neg = [(q, -2), (gamma, 0.01), (sigma, 0.2), (t, 0), (T, 1), (s, 100)]
        mid_pos = float(res["midpoint"].subs(params_pos))
        mid_neg = float(res["midpoint"].subs(params_neg))
        assert mid_pos < 100.0, "q>0 should give midpoint below s"
        assert mid_neg > 100.0, "q<0 should give midpoint above s"

    def test_R_a_greater_than_R_b(self) -> None:
        """R^a should be greater than R^b (positive spread)."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        spread_num = float(res["spread"].subs(
            [(gamma, 0.01), (sigma, 0.2), (t, 0), (T, 1), (s, 100)]
        ))
        assert spread_num > 0


class TestModelComparison:
    """Tests for comparison between ABM+CARA and GBM+MV models."""

    def test_comparison_contains_both_models(self) -> None:
        """Comparison should reference both ABM+CARA and GBM+MV."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        # Import ABM results
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from exp1_analytical_derivations import derive_reservation_prices as derive_abm
        abm_res = derive_abm()
        comparison = compare_models(abm_res, res)
        assert "abm_cara_r_a" in comparison
        assert "gbm_mv_R_a" in comparison
        assert "structural_difference" in comparison

    def test_structural_difference_mentions_s_squared(self) -> None:
        """Structural difference should mention s^2 scaling for GBM+MV."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        from exp1_analytical_derivations import derive_reservation_prices as derive_abm
        abm_res = derive_abm()
        comparison = compare_models(abm_res, res)
        assert "s^2" in comparison["structural_difference"] or "s2" in comparison["structural_difference"]

    def test_gbm_adjustment_scales_with_s_squared(self) -> None:
        """GBM+MV adjustment should scale with s^2."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        # Adjustment factor = (gamma*s^2/2)*(exp(sigma^2*(T-t))-1)
        adj = res["adjustment_factor"]
        # Check that s^2 appears in the adjustment
        adj_s1 = float(adj.subs([(gamma, 0.01), (sigma, 0.2), (t, 0), (T, 1), (s, 100)]))
        adj_s2 = float(adj.subs([(gamma, 0.01), (sigma, 0.2), (t, 0), (T, 1), (s, 200)]))
        # adj_s2 / adj_s1 should be approximately (200/100)^2 = 4
        ratio = adj_s2 / adj_s1
        assert abs(ratio - 4.0) < 0.01, f"Adjustment should scale with s^2, ratio={ratio}"


class TestEdgeCasesGBM:
    """Tests for edge cases in the GBM model."""

    def test_zero_sigma_limit(self) -> None:
        """As sigma->0, variance should approach 0."""
        moments = derive_gbm_moments()
        var_sigma0 = moments["Var_ST"].subs(sigma, 0)
        assert simplify(var_sigma0) == 0

    def test_spread_zero_at_T(self) -> None:
        """Spread should be zero at t=T."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        spread_T = res["spread"].subs(t, T)
        assert simplify(spread_T) == 0

    def test_numerical_values_reasonable(self) -> None:
        """Numerical values should be in reasonable range."""
        moments = derive_gbm_moments()
        vf = derive_gbm_value_function(moments)
        res = derive_gbm_reservation_prices(vf)
        params = [(q, 1), (gamma, 0.01), (sigma, 0.2), (t, 0), (T, 1), (s, 100)]
        R_a_num = float(res["R_a"].subs(params))
        R_b_num = float(res["R_b"].subs(params))
        # Both should be near 100
        assert 90 < R_b_num < 110
        assert 90 < R_a_num < 110
        assert R_a_num > R_b_num
