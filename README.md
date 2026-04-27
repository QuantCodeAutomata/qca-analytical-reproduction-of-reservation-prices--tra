# Avellaneda-Stoikov Market-Making Model: Analytical Reproduction

This repository reproduces the analytical derivations and numerical experiments from the
Avellaneda-Stoikov (2008) market-making model paper.

## Experiments

| ID | Title |
|----|-------|
| exp_1 | Analytical reproduction of reservation prices, transformed HJB, and approximate quoting formulas |
| exp_2 | Primary numerical replication using table-faithful constant-spread implementation |
| exp_3 | Secondary numerical replication using derivation-faithful time-varying spread |
| exp_4 | Analytical appendix reproduction for GBM and mean-variance reservation-price model |

## Structure

```
src/
  exp1_analytical_derivations.py   # SymPy symbolic derivations (HJB, reservation prices)
  exp2_montecarlo_constant_spread.py  # Monte Carlo with constant spread
  exp3_montecarlo_timevarying_spread.py  # Monte Carlo with time-varying spread
  exp4_gbm_meanvariance.py         # GBM mean-variance appendix derivation
results/
  RESULTS.md                       # All metrics and findings
  *.png                            # Generated plots
tests/
  test_exp1.py
  test_exp2.py
  test_exp3.py
  test_exp4.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/exp1_analytical_derivations.py
python src/exp2_montecarlo_constant_spread.py
python src/exp3_montecarlo_timevarying_spread.py
python src/exp4_gbm_meanvariance.py
pytest tests/ -v
```

## Parameters

- S0 = 100, T = 1, sigma = 2, dt = 0.005, N = 200
- A = 140, k = 1.5
- gamma ∈ {0.01, 0.1, 0.5}
- M = 1000 Monte Carlo paths

## Reference

Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
*Quantitative Finance*, 8(3), 217–224.
