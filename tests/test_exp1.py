"""
Tests for Experiment 1: Analytical derivations of the Avellaneda-Stoikov model.
Verifies that the paper's methodology is implemented correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np
import sympy as sp
from sympy import symbols, exp, log, simplify

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exp1_analytical_derivations import (
    derive_frozen_inventory_value_function,
    derive_reservation_prices,
    derive_hjb_and_ansatz,
    derive_first_order_conditions,
    derive_asymptotic_expansion,
    verify_terminal_conditions,
    s, q, t, T, gamma, sigma, k,
)


class TestFrozenInventoryValueFunction:
    """Tests for the frozen-inventory value function derivation."""

    def test_value_function_structure(self) -> None:
        """Value function should have three exponential factors."""
        vf = derive_frozen_inventory_value_function()
        assert "v" in vf
        assert "description" in vf
        assert "mgf" in vf

    def test_mgf_formula(self) -> None:
        """MGF should match exp(-gamma*q*s)*exp(gamma^2*q^2*sigma^2*tau/2)."""
        vf = derive_frozen_inventory_value_function()
        tau = T - t
        expected = exp(-gamma * q * s) * exp(gamma**2 * q**2 * sigma**2 * tau / 2)
        assert simplify(vf["mgf"] - expected) == 0

    def test_value_function_negative(self) -> None:
        """Value function should be negative (CARA utility)."""
        # Numerically: v = -exp(-gamma*x)*exp(-gamma*q*s)*exp(...)
        # All exponentials are positive, so v < 0
        vf = derive_frozen_inventory_value_function()
        # Use the module-level x symbol (same object) for substitution
        x_sym = symbols("x", real=True)
        v_num = vf["v"].subs([(gamma, 0.1), (q, 1), (s, 100), (t, 0), (T, 1),
                               (sigma, 2), (x_sym, 0)])
        assert float(v_num) < 0, "Value function should be negative"

    def test_description_contains_key_terms(self) -> None:
        """Description should mention key formula components."""
        vf = derive_frozen_inventory_value_function()
        desc = vf["description"]
        assert "gamma" in desc
        assert "sigma" in desc


class TestReservationPrices:
    """Tests for reservation bid and ask price derivations."""

    def test_r_a_formula(self) -> None:
        """r^a should equal s + ((1-2q)/2)*gamma*sigma^2*(T-t)."""
        res = derive_reservation_prices()
        tau = T - t
        r_a_paper = s + sp.Rational(1, 2) * (1 - 2 * q) * gamma * sigma**2 * tau
        assert simplify(res["r_a"] - r_a_paper) == 0

    def test_r_b_formula(self) -> None:
        """r^b should equal s + ((-1-2q)/2)*gamma*sigma^2*(T-t)."""
        res = derive_reservation_prices()
        tau = T - t
        r_b_paper = s + sp.Rational(1, 2) * (-1 - 2 * q) * gamma * sigma**2 * tau
        assert simplify(res["r_b"] - r_b_paper) == 0

    def test_midpoint_formula(self) -> None:
        """Midpoint should equal s - q*gamma*sigma^2*(T-t)."""
        res = derive_reservation_prices()
        tau = T - t
        midpoint_paper = s - q * gamma * sigma**2 * tau
        assert simplify(res["midpoint"] - midpoint_paper) == 0

    def test_spread_formula(self) -> None:
        """Spread r^a - r^b should equal gamma*sigma^2*(T-t)."""
        res = derive_reservation_prices()
        tau = T - t
        assert simplify(res["spread_ra_minus_rb"] - gamma * sigma**2 * tau) == 0

    def test_terminal_conditions(self) -> None:
        """At t=T, both reservation prices should collapse to s."""
        res = derive_reservation_prices()
        assert simplify(res["r_a"].subs(t, T) - s) == 0
        assert simplify(res["r_b"].subs(t, T) - s) == 0

    def test_r_a_greater_than_r_b(self) -> None:
        """r^a should always be greater than r^b (positive spread)."""
        res = derive_reservation_prices()
        spread = res["spread_ra_minus_rb"]
        # Spread = gamma*sigma^2*(T-t) > 0 for t < T
        # Numerically verify
        spread_num = float(spread.subs([(gamma, 0.1), (sigma, 2), (t, 0), (T, 1)]))
        assert spread_num > 0

    def test_inventory_sign_convention(self) -> None:
        """For q>0, midpoint should be below s; for q<0, above s."""
        res = derive_reservation_prices()
        # midpoint = s - q*gamma*sigma^2*(T-t)
        # For q=2, gamma=0.1, sigma=2, t=0, T=1, s=100:
        mid_q_pos = float(res["midpoint"].subs(
            [(q, 2), (gamma, 0.1), (sigma, 2), (t, 0), (T, 1), (s, 100)]
        ))
        mid_q_neg = float(res["midpoint"].subs(
            [(q, -2), (gamma, 0.1), (sigma, 2), (t, 0), (T, 1), (s, 100)]
        ))
        assert mid_q_pos < 100.0, "q>0 should give midpoint below s"
        assert mid_q_neg > 100.0, "q<0 should give midpoint above s"

    def test_r_a_r_b_numerical_values(self) -> None:
        """Verify numerical values for specific parameters."""
        res = derive_reservation_prices()
        params = [(gamma, 0.1), (sigma, 2), (t, 0), (T, 1), (s, 100), (q, 0)]
        r_a_num = float(res["r_a"].subs(params))
        r_b_num = float(res["r_b"].subs(params))
        # For q=0: r^a = s + 0.5*gamma*sigma^2*T = 100 + 0.5*0.1*4*1 = 100.2
        assert abs(r_a_num - 100.2) < 1e-10
        # For q=0: r^b = s - 0.5*gamma*sigma^2*T = 100 - 0.2 = 99.8
        assert abs(r_b_num - 99.8) < 1e-10


class TestHJBAndAnsatz:
    """Tests for HJB equation and exponential-utility ansatz."""

    def test_ansatz_form(self) -> None:
        """Ansatz should have the correct exponential form."""
        hjb = derive_hjb_and_ansatz()
        assert "exp(-gamma*x)" in hjb["ansatz"]
        assert "theta" in hjb["ansatz"]

    def test_terminal_condition(self) -> None:
        """Terminal condition should be theta(s,q,T) = q*s."""
        hjb = derive_hjb_and_ansatz()
        assert "q*s" in hjb["terminal_condition"] or "q s" in hjb["terminal_condition"]

    def test_nonlinear_term_present(self) -> None:
        """PDE should contain the nonlinear term -(sigma^2*gamma/2)*theta_s^2."""
        hjb = derive_hjb_and_ansatz()
        assert "theta_s^2" in hjb["nonlinear_term"] or "theta_s" in hjb["nonlinear_term"]

    def test_reservation_prices_from_theta(self) -> None:
        """Reservation prices should be expressed as theta differences."""
        hjb = derive_hjb_and_ansatz()
        assert "theta(s,q+1,t) - theta(s,q,t)" in hjb["r_b_from_theta"]
        assert "theta(s,q,t) - theta(s,q-1,t)" in hjb["r_a_from_theta"]


class TestFirstOrderConditions:
    """Tests for first-order optimality conditions."""

    def test_log_argument(self) -> None:
        """Log argument should be 1 + gamma/k."""
        foc = derive_first_order_conditions()
        expected = 1 + gamma / k
        assert simplify(foc["log_argument"] - expected) == 0

    def test_half_spread_log(self) -> None:
        """Half-spread log term should be (1/gamma)*ln(1+gamma/k)."""
        foc = derive_first_order_conditions()
        expected = (1 / gamma) * sp.log(1 + gamma / k)
        assert simplify(foc["half_spread_log"] - expected) == 0

    def test_half_spread_positive(self) -> None:
        """Half-spread log term should be positive for gamma>0, k>0."""
        foc = derive_first_order_conditions()
        val = float(foc["half_spread_log"].subs([(gamma, 0.1), (k, 1.5)]))
        assert val > 0

    def test_price_quote_consistency(self) -> None:
        """p^a = r^a + half_spread, p^b = r^b - half_spread."""
        foc = derive_first_order_conditions()
        r_a_var, r_b_var = symbols("r_a r_b", real=True)
        half = foc["half_spread_log"]
        assert simplify(foc["p_a_opt"] - (r_a_var + half)) == 0
        assert simplify(foc["p_b_opt"] - (r_b_var - half)) == 0

    def test_numerical_half_spread(self) -> None:
        """Verify numerical half-spread for paper parameters."""
        foc = derive_first_order_conditions()
        # (1/gamma)*ln(1+gamma/k) for gamma=0.1, k=1.5
        val = float(foc["half_spread_log"].subs([(gamma, 0.1), (k, 1.5)]))
        expected = (1 / 0.1) * np.log(1 + 0.1 / 1.5)
        assert abs(val - expected) < 1e-10


class TestAsymptoticExpansion:
    """Tests for asymptotic inventory expansion and spread formulas."""

    def test_theta1_equals_s(self) -> None:
        """theta^1 should equal s."""
        exp_res = derive_asymptotic_expansion()
        assert simplify(exp_res["theta1"] - s) == 0

    def test_theta2_formula(self) -> None:
        """theta^2 should equal sigma^2*gamma*(T-t)."""
        exp_res = derive_asymptotic_expansion()
        tau = T - t
        assert simplify(exp_res["theta2"] - sigma**2 * gamma * tau) == 0

    def test_r_approx_formula(self) -> None:
        """Approximate r should equal s - q*gamma*sigma^2*(T-t)."""
        exp_res = derive_asymptotic_expansion()
        tau = T - t
        assert simplify(exp_res["r_approx"] - (s - q * gamma * sigma**2 * tau)) == 0

    def test_constant_spreads_match_paper(self) -> None:
        """Constant spreads should approximately match paper values 1.33, 1.29, 1.15."""
        exp_res = derive_asymptotic_expansion()
        spreads = exp_res["constant_spreads_numerical"]
        expected = {0.01: 1.33, 0.1: 1.29, 0.5: 1.15}
        for g, exp_val in expected.items():
            assert abs(spreads[g] - exp_val) < 0.01, (
                f"gamma={g}: spread={spreads[g]}, expected~{exp_val}"
            )

    def test_spread_at_T_equals_constant(self) -> None:
        """Time-varying spread at t=T should equal constant spread."""
        exp_res = derive_asymptotic_expansion()
        spread_tv_T = exp_res["spread_total_derivation_faithful"].subs(t, T)
        spread_const = exp_res["spread_constant_table_faithful"]
        assert simplify(spread_tv_T - spread_const) == 0

    def test_spread_decreases_over_time(self) -> None:
        """Time-varying spread should decrease as t approaches T."""
        exp_res = derive_asymptotic_expansion()
        spread_tv = exp_res["spread_total_derivation_faithful"]
        # At t=0: spread = gamma*sigma^2*T + (2/gamma)*ln(1+gamma/k)
        # At t=T: spread = (2/gamma)*ln(1+gamma/k)
        # So spread at t=0 > spread at t=T
        params_t0 = [(gamma, 0.1), (sigma, 2), (k, 1.5), (t, 0), (T, 1)]
        params_tT = [(gamma, 0.1), (sigma, 2), (k, 1.5), (t, 1), (T, 1)]
        spread_t0 = float(spread_tv.subs(params_t0))
        spread_tT = float(spread_tv.subs(params_tT))
        assert spread_t0 > spread_tT

    def test_ambiguity_documented(self) -> None:
        """Ambiguity note should be present and mention both spread formulas."""
        exp_res = derive_asymptotic_expansion()
        note = exp_res["ambiguity_note"]
        assert "constant" in note.lower() or "table" in note.lower()
        assert "time-varying" in note.lower() or "derivation" in note.lower()


class TestTerminalConditions:
    """Tests for terminal condition verification."""

    def test_all_terminal_conditions_pass(self) -> None:
        """All terminal conditions should pass."""
        res = derive_reservation_prices()
        exp_res = derive_asymptotic_expansion()
        term = verify_terminal_conditions(res, exp_res)
        for key, val in term.items():
            if key != "spread_limit_gamma_to_0":
                assert val is True, f"Terminal condition {key} failed"

    def test_spread_limit_gamma_to_0(self) -> None:
        """Spread limit as gamma->0 should be 2/k."""
        res = derive_reservation_prices()
        exp_res = derive_asymptotic_expansion()
        term = verify_terminal_conditions(res, exp_res)
        limit_val = term["spread_limit_gamma_to_0"]
        # 2/k for k=1.5 is 4/3
        k_sym = symbols("k", positive=True)
        assert simplify(limit_val - 2 / k) == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_inventory_midpoint_equals_s(self) -> None:
        """At q=0, midpoint should equal s."""
        res = derive_reservation_prices()
        mid_q0 = res["midpoint"].subs(q, 0)
        assert simplify(mid_q0 - s) == 0

    def test_zero_gamma_limit(self) -> None:
        """As gamma->0, reservation prices should approach s."""
        res = derive_reservation_prices()
        r_a_g0 = sp.limit(res["r_a"], gamma, 0, "+")
        assert simplify(r_a_g0 - s) == 0

    def test_spread_positive_for_all_gamma(self) -> None:
        """Spread should be positive for all gamma > 0."""
        exp_res = derive_asymptotic_expansion()
        for g in [0.001, 0.01, 0.1, 0.5, 1.0]:
            spread_val = float(
                exp_res["spread_constant_table_faithful"].subs([(gamma, g), (k, 1.5)])
            )
            assert spread_val > 0, f"Spread should be positive for gamma={g}"
