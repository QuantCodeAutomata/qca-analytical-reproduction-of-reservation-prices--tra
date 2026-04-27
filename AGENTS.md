# AGENTS.md — Repository Knowledge Base

## Project Overview
Avellaneda-Stoikov (2008) market-making model: analytical reproduction and Monte Carlo replication.

## Repository Structure
```
src/
  exp1_analytical_derivations.py   # SymPy symbolic derivations (HJB, reservation prices, spread)
  exp2_montecarlo_constant_spread.py  # Monte Carlo with table-faithful constant spread
  exp3_montecarlo_timevarying_spread.py  # Monte Carlo with derivation-faithful time-varying spread
  exp4_gbm_meanvariance.py         # GBM mean-variance appendix derivation
  __init__.py
tests/
  test_exp1.py  test_exp2.py  test_exp3.py  test_exp4.py
results/
  RESULTS.md   *.png
run_all_experiments.py   # Master runner
conftest.py              # Adds src/ to sys.path for pytest
pytest.ini
requirements.txt
```

## Key Parameters (Paper)
- S0=100, T=1, sigma=2, dt=0.005, N=200, A=140, k=1.5, M=1000
- gamma ∈ {0.01, 0.1, 0.5}
- Constant spread: Δ = (2/γ)ln(1+γ/k)  → {0.01: 1.3289, 0.1: 1.2908, 0.5: 1.1507}
- Time-varying spread: Δ_t = γσ²(T-t) + (2/γ)ln(1+γ/k)

## Key Formulas
- Reservation price: r(s,q,t) = s - q·γ·σ²·(T-t)
- r^a = s + ((1-2q)/2)·γ·σ²·(T-t)
- r^b = s + ((-1-2q)/2)·γ·σ²·(T-t)
- Half-spread offset: (1/γ)·ln(1+γ/k)
- GBM+MV: R^a = s + ((1-2q)/2)·γ·s²·(exp(σ²(T-t))-1)

## Library Choices
- SymPy: symbolic derivations (exp1, exp4) — Context7 confirmed
- NumPy: vectorized Monte Carlo (exp2, exp3)
- Matplotlib/Seaborn: all plots

## Testing
- 101 tests, all passing
- conftest.py adds src/ to sys.path
- Run: `pytest tests/ -v`

## Internal Paper Ambiguity (exp1/exp2/exp3)
The paper's derivation yields a time-varying spread Δ_t = γσ²(T-t) + (2/γ)ln(1+γ/k),
but the published simulation tables appear to use the constant spread Δ = (2/γ)ln(1+γ/k).
Exp2 uses the table-faithful constant spread; Exp3 uses the derivation-faithful time-varying spread.

## Common Pitfalls
- SymPy `symbols("x", real=True)` must match exactly when substituting — use the module-level symbol
- `center_path[0]` in simulation is uninitialized (0.0); first valid center is at index 1
- Probability capping: `min(λ·dt, 1)` for Bernoulli execution; log capping events
- Negative quote distances arise naturally for large inventory — expected behavior
