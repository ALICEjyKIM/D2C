# D2C Assortment and Allocation MILP

## Sets and parameters

- `I`: SKUs; `R`: retailers; `T`: periods; `I_r`: SKUs carried by retailer `r`.
- `q_bar[i,t]`: D2C demand, `U[i,t]`: SKU supply limit, `C[t]`: shared capacity.
- `mu[r,i,t]`: baseline retailer order, `a[i]`: capacity use.
- `pi_d2c[i]`, `pi_wholesale[r,i]`: unit margins.
- `K`, `beta`, `rho`, `kappa`, `gamma`: assortment and dynamic parameters.

The toy instance uses period-invariant values for `q_bar`, `U`, `C`, and `mu`.

## Variables

- `y[i,t]` is one when SKU `i` is listed in D2C.
- `q[i,t] >= 0` is D2C quantity.
- `x[r,i,t] >= 0` is retailer allocation, defined only for `i in I_r`.
- `g[r,t] in [0,1]` is retailer order retention.
- `e[r,t] in [0,1]` is retailer exposure to D2C activity.

## Objective

For active periods from `s` through `s + H - 1`, truncated at the end of `T`:

```text
maximize sum_t gamma^(t-s) * (
    sum_i pi_d2c[i] q[i,t]
    + sum_(r,i feasible) pi_wholesale[r,i] x[r,i,t]
)
```

## Constraints

- C1: `0 <= q[i,t] <= q_bar[i,t] y[i,t]`.
- C2: `sum_i y[i,t] <= K`.
- C3: `0 <= x[r,i,t] <= mu[r,i,t] g[r,t]` for feasible pairs.
- C4: `q[i,t] + sum_r x[r,i,t] <= U[i,t]`.
- C5: `sum_i a[i] (q[i,t] + sum_r x[r,i,t]) <= C[t]`.
- C6: `e[r,t] = (1 / |I_r|) sum_(i in I_r) [beta y[i,t] + (1-beta) q[i,t] / q_bar[i,t]]`.
- C7: `g[r,t+1] = rho g[r,t] + (1-rho)(1-kappa e[r,t])`.
- C8: `y` is binary and `q,x >= 0`; `g,e` lie in `[0,1]`.

At the active start period, `g[r,s]` is fixed to the supplied `State`. C7 is
added only when both `t` and `t+1` are active.

## Python mapping

| Math | Python |
| --- | --- |
| `y_it` | `y[i, t]` |
| `q_it` | `q[i, t]` |
| `x_rit` | `x[r, i, t]` |
| `g_rt` | `g[r, t]` |
| `e_rt` | `e[r, t]` |
