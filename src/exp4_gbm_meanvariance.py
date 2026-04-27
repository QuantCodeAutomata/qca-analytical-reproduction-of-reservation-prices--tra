"""
Experiment 4: Analytical appendix reproduction for the alternative GBM and
mean-variance reservation-price model.

Derives the value function and reservation prices under:
- Geometric Brownian motion: dS_u/S_u = sigma dW_u
- Mean-variance objective: V(x,s,q,t) = E_t[(x+q*S_T) - (gamma/2)*(q*S_T - q*s)^2]

# Using SymPy for symbolic mathematics -- Context7 confirmed (/sympy/sympy)
"""
from __future__ import annotations

import sympy as sp
from sympy import symbols, exp, simplify, Rational, sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Symbol definitions
# ---------------------------------------------------------------------------
s, q, t, T, x = symbols("s q t T x", real=True)
gamma, sigma = symbols("gamma sigma", positive=True)


def derive_gbm_moments() -> dict:
    """
    Derive conditional moments of S_T under GBM with zero drift.

    Under dS_u/S_u = sigma*dW_u with S_t = s:
        E_t[S_T] = s  (zero drift)
        Var_t(S_T) = s^2 * (exp(sigma^2*(T-t)) - 1)

    Returns
    -------
    dict with conditional mean and variance expressions.
    """
    tau = T - t
    E_ST = s  # zero drift GBM: E[S_T | S_t=s] = s
    Var_ST = s**2 * (exp(sigma**2 * tau) - 1)
    E_ST2 = Var_ST + E_ST**2  # E[S_T^2] = Var + (E[S_T])^2

    return {
        "tau": tau,
        "E_ST": E_ST,
        "Var_ST": Var_ST,
        "E_ST2": E_ST2,
    }


def derive_gbm_value_function(moments: dict) -> dict:
    """
    Derive the closed-form mean-variance value function under GBM.

    Objective: V(x,s,q,t) = E_t[(x + q*S_T) - (gamma/2)*(q*S_T - q*s)^2]
                           = x + q*E[S_T] - (gamma/2)*q^2*E[(S_T - s)^2]
                           = x + q*s - (gamma/2)*q^2*Var_t(S_T)
                           = x + q*s - (gamma/2)*q^2*s^2*(exp(sigma^2*(T-t)) - 1)

    Note: E[(S_T - s)^2] = Var_t(S_T) since E[S_T] = s under zero-drift GBM.

    Parameters
    ----------
    moments : output of derive_gbm_moments

    Returns
    -------
    dict with value function expression and description.
    """
    tau = moments["tau"]
    E_ST = moments["E_ST"]
    Var_ST = moments["Var_ST"]

    # V = x + q*E[S_T] - (gamma/2)*q^2*Var_t(S_T)
    # Since E[S_T] = s and Var_t(S_T) = s^2*(exp(sigma^2*tau)-1):
    V = x + q * E_ST - (gamma / 2) * q**2 * Var_ST
    V_simplified = simplify(V)

    # Paper formula: V(x,s,q,t) = x + q*s + (gamma*q^2*s^2/2)*(exp(sigma^2*(T-t)) - 1)
    # Note: the sign: -gamma/2 * q^2 * s^2*(exp-1) = -(gamma*q^2*s^2/2)*(exp-1)
    # The paper writes it as: x + q*s + (gamma*q^2*s^2/2)*(exp(sigma^2*(T-t))-1)
    # This is NEGATIVE because (exp-1) > 0 and the term subtracts from utility.
    # Let's check: V = x + q*s - (gamma/2)*q^2*s^2*(exp(sigma^2*tau)-1)
    V_paper = x + q * s - (gamma * q**2 * s**2 / 2) * (exp(sigma**2 * tau) - 1)
    assert simplify(V_simplified - V_paper) == 0, (
        f"Value function mismatch: {V_simplified} vs {V_paper}"
    )

    # Terminal condition: at t=T, tau=0, exp(0)-1=0, so V(x,s,q,T) = x + q*s
    V_terminal = V_paper.subs(t, T)
    assert simplify(V_terminal - (x + q * s)) == 0, "Terminal condition failed"

    return {
        "V": V_paper,
        "V_simplified": V_simplified,
        "description": (
            "V(x,s,q,t) = x + q*s - (gamma*q^2*s^2/2)*(exp(sigma^2*(T-t)) - 1)"
        ),
        "terminal_value": "V(x,s,q,T) = x + q*s",
    }


def derive_gbm_reservation_prices(vf: dict) -> dict:
    """
    Derive reservation bid and ask prices under GBM mean-variance objective.

    Reservation ask R^a: V(x + R^a, s, q-1, t) = V(x, s, q, t)
    Reservation bid R^b: V(x - R^b, s, q+1, t) = V(x, s, q, t)

    Since V is linear in x:
        V(x + R^a, s, q-1, t) = V(x, s, q-1, t) + R^a
        V(x - R^b, s, q+1, t) = V(x, s, q+1, t) - R^b

    So:
        R^a = V(x, s, q, t) - V(x, s, q-1, t)
        R^b = V(x, s, q+1, t) - V(x, s, q, t)

    Parameters
    ----------
    vf : output of derive_gbm_value_function

    Returns
    -------
    dict with R_a, R_b, midpoint formulas.
    """
    tau = T - t
    adj = (gamma * s**2 / 2) * (exp(sigma**2 * tau) - 1)

    def V_q(q_val: sp.Expr) -> sp.Expr:
        """Value function for inventory q_val (x-independent part)."""
        return q_val * s - (gamma * q_val**2 * s**2 / 2) * (exp(sigma**2 * tau) - 1)

    # R^a = V(q) - V(q-1)
    R_a = simplify(V_q(q) - V_q(q - 1))
    # R^b = V(q+1) - V(q)
    R_b = simplify(V_q(q + 1) - V_q(q))

    # Paper formulas:
    # R^a = s + ((1-2q)/2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)
    # R^b = s + ((-1-2q)/2) * gamma * s^2 * (exp(sigma^2*(T-t)) - 1)
    R_a_paper = s + Rational(1, 2) * (1 - 2 * q) * gamma * s**2 * (exp(sigma**2 * tau) - 1)
    R_b_paper = s + Rational(1, 2) * (-1 - 2 * q) * gamma * s**2 * (exp(sigma**2 * tau) - 1)
    midpoint_paper = s - q * gamma * s**2 * (exp(sigma**2 * tau) - 1)

    assert simplify(R_a - R_a_paper) == 0, f"R^a mismatch: {R_a} vs {R_a_paper}"
    assert simplify(R_b - R_b_paper) == 0, f"R^b mismatch: {R_b} vs {R_b_paper}"

    midpoint = simplify((R_a + R_b) / 2)
    assert simplify(midpoint - midpoint_paper) == 0, f"Midpoint mismatch: {midpoint}"

    # Spread R^a - R^b = gamma * s^2 * (exp(sigma^2*(T-t)) - 1)
    spread = simplify(R_a - R_b)
    spread_paper = gamma * s**2 * (exp(sigma**2 * tau) - 1)
    assert simplify(spread - spread_paper) == 0, "Spread mismatch"

    # Terminal conditions: at t=T, tau=0, exp(0)-1=0
    R_a_T = R_a.subs(t, T)
    R_b_T = R_b.subs(t, T)
    assert simplify(R_a_T - s) == 0, "R^a terminal condition failed"
    assert simplify(R_b_T - s) == 0, "R^b terminal condition failed"

    # q=0 check: midpoint at q=0 should be s
    mid_q0 = midpoint.subs(q, 0)
    assert simplify(mid_q0 - s) == 0, "Midpoint at q=0 should be s"

    return {
        "R_a": R_a,
        "R_b": R_b,
        "midpoint": midpoint,
        "spread": spread,
        "R_a_paper_form": R_a_paper,
        "R_b_paper_form": R_b_paper,
        "adjustment_factor": adj,
    }


def compare_models(res_prices_abm: dict, res_prices_gbm: dict) -> dict:
    """
    Compare reservation price structures between ABM+CARA and GBM+mean-variance models.

    Parameters
    ----------
    res_prices_abm : reservation prices from ABM+CARA model (exp1)
    res_prices_gbm : reservation prices from GBM+mean-variance model (exp4)

    Returns
    -------
    dict with comparison summary.
    """
    tau = T - t
    return {
        "abm_cara_r_a": str(res_prices_abm["r_a"]),
        "gbm_mv_R_a": str(res_prices_gbm["R_a"]),
        "abm_cara_adjustment": f"gamma*sigma^2*(T-t)  [linear in tau]",
        "gbm_mv_adjustment": f"gamma*s^2*(exp(sigma^2*(T-t))-1)  [nonlinear, scales with s^2]",
        "structural_difference": (
            "ABM+CARA: adjustment proportional to sigma^2*(T-t). "
            "GBM+MV: adjustment proportional to s^2*(exp(sigma^2*(T-t))-1). "
            "For small sigma^2*(T-t), exp(sigma^2*(T-t))-1 ~ sigma^2*(T-t), "
            "so GBM+MV adjustment ~ s^2*sigma^2*(T-t) vs ABM+CARA ~ sigma^2*(T-t). "
            "The GBM+MV adjustment scales with s^2, reflecting multiplicative price dynamics."
        ),
    }


def plot_gbm_reservation_prices(
    s0: float = 100.0,
    sigma_val: float = 0.2,
    T_val: float = 1.0,
    gamma_vals: list | None = None,
    q_vals: list | None = None,
) -> None:
    """
    Plot GBM mean-variance reservation prices as functions of time.

    Parameters
    ----------
    s0 : current mid-price
    sigma_val : GBM volatility
    T_val : horizon
    gamma_vals : list of risk-aversion values
    q_vals : list of inventory levels
    """
    if gamma_vals is None:
        gamma_vals = [0.001, 0.01, 0.05]
    if q_vals is None:
        q_vals = [-2, 0, 2]

    t_arr = np.linspace(0, T_val * 0.999, 200)
    tau_arr = T_val - t_arr

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "GBM Mean-Variance Model: Reservation Prices\n"
        r"$R^a(s,q,t)=s+\frac{1-2q}{2}\gamma s^2(\exp(\sigma^2(T-t))-1)$",
        fontsize=11,
    )

    for col, g in enumerate(gamma_vals):
        ax = axes[col]
        adj = g * s0**2 * (np.exp(sigma_val**2 * tau_arr) - 1)
        for q_val in q_vals:
            R_a_arr = s0 + (1 - 2 * q_val) / 2 * adj
            ax.plot(t_arr, R_a_arr, label=f"R^a, q={q_val}")
        ax.axhline(s0, color="black", linestyle="--", linewidth=0.8, label="mid-price")
        ax.set_title(f"GBM Reservation Ask, gamma={g}")
        ax.set_xlabel("Time t")
        ax.set_ylabel("R^a(s,q,t)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "exp4_gbm_reservation_prices.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[exp4] Saved plot: {out_path}")


def run_experiment_4() -> dict:
    """
    Run all analytical derivations for Experiment 4 (GBM mean-variance appendix).

    Returns
    -------
    dict with all derivation results.
    """
    print("=" * 70)
    print("EXPERIMENT 4: GBM Mean-Variance Appendix Derivation")
    print("=" * 70)

    print("\n[1] GBM conditional moments...")
    moments = derive_gbm_moments()
    print(f"    E_t[S_T] = {moments['E_ST']}")
    print(f"    Var_t(S_T) = {moments['Var_ST']}")

    print("\n[2] Mean-variance value function...")
    vf = derive_gbm_value_function(moments)
    print(f"    {vf['description']}")
    print(f"    Terminal: {vf['terminal_value']}")

    print("\n[3] Reservation bid and ask prices...")
    res = derive_gbm_reservation_prices(vf)
    print(f"    R^a = {res['R_a']}")
    print(f"    R^b = {res['R_b']}")
    print(f"    Midpoint = {res['midpoint']}")
    print(f"    Spread R^a - R^b = {res['spread']}")

    print("\n[4] Terminal condition verification...")
    R_a_T = res["R_a"].subs(t, T)
    R_b_T = res["R_b"].subs(t, T)
    mid_T = res["midpoint"].subs(t, T)
    print(f"    R^a at t=T: {R_a_T} (should be s)")
    print(f"    R^b at t=T: {R_b_T} (should be s)")
    print(f"    Midpoint at t=T: {mid_T} (should be s)")
    assert simplify(R_a_T - s) == 0, "R^a terminal failed"
    assert simplify(R_b_T - s) == 0, "R^b terminal failed"
    print("    All terminal conditions: PASS")

    print("\n[5] Model comparison (ABM+CARA vs GBM+MV)...")
    # Import exp1 results for comparison
    from exp1_analytical_derivations import derive_reservation_prices as derive_abm
    abm_res = derive_abm()
    comparison = compare_models(abm_res, res)
    print(f"    ABM+CARA r^a: {comparison['abm_cara_r_a']}")
    print(f"    GBM+MV R^a:   {comparison['gbm_mv_R_a']}")
    print(f"    ABM+CARA adjustment: {comparison['abm_cara_adjustment']}")
    print(f"    GBM+MV adjustment:   {comparison['gbm_mv_adjustment']}")
    print(f"\n    Structural difference: {comparison['structural_difference']}")

    print("\n[6] Generating plots...")
    plot_gbm_reservation_prices()

    print("\n[exp4] All derivations completed successfully.")
    return {
        "moments": moments,
        "value_function": vf,
        "reservation_prices": res,
        "model_comparison": comparison,
    }


if __name__ == "__main__":
    run_experiment_4()
