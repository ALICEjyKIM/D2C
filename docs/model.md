# D2C Assortment and Allocation MILP

## Sets and parameters

- `I`: SKUs; `R`: retailers; `T`: periods; `I_r`: SKUs carried by retailer `r`.
- `q_bar[i,t]`: D2C demand, `U[i,t]`: SKU supply limit, `C[t]`: shared capacity.
- `mu[r,i,t]`: baseline retailer order, `a[i]`: capacity use.
- `pi_d2c[i]`, `pi_wholesale[r,i]`: 모든 채널 변동비를 차감한 단위 순기여이익.
- `K`, `beta`, `rho`, `kappa`, `gamma`: assortment and dynamic parameters.

The toy instance uses period-invariant values for `q_bar`, `U`, `C`, and `mu`.
고정비와 재고비용은 현재 목적함수에 포함하지 않는다.

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

## Information timing

계획 시나리오는 현재 기간과 미래 기간을 구분한다. 현재 기간의 공급한도,
capacity, D2C 수요, 리테일러 기본 주문수요, 채널별 순기여이익은
`current_observation`에서 복사하며, 이를 생략하면 기준 `Instance` 값을 쓴다.
현재 주문유지율은 `current_state`에서 복사한다. 따라서 이 값들은 모든 계획
시나리오에서 같다.

반응 민감도와 상태 지속계수는 현재에도 알 수 없는 특성이므로 시나리오별로
유지한다. 현재 노출과 이 계수들이 결합되어 다음 기간 주문유지율 `g[r,s+1]`을
결정한다. 미래 수요와 순기여이익은 `s+1`부터 시나리오별로 분기한다.

평가경로는 전 기간의 실제 시장상황을 나타내므로 현재 행도 고정하지 않는다.
향후 rolling horizon에서는 평가경로의 현재 행을 계획 함수의
`current_observation`으로 전달할 수 있다. 이 프로젝트에는 아직 그 rolling
regret 정책은 연결하지 않았다.

## Minimax relative regret

시나리오별 수요, 순기여이익, 반응계수와 지속계수를 C1, C3, C6, C7 및
목적함수에 직접 적용한다. 각 시나리오의 결정론적 최적값을 `z_star[s]`로
계산한 뒤 다음 extensive-form을 푼다.

```text
minimize theta
z[s] >= z_star[s] - theta * max(epsilon, abs(z_star[s]))
theta >= 0
```

현재 기간의 `y`, `q`, `x`는 모든 시나리오에서 같고, 이후 기간의 계획은
시나리오별로 달라질 수 있다. `z_star[s]`와 `z[s]`는 기존 MILP와 동일하게
`gamma`가 적용된 계획이익이다. 일반 입력에서는 `gamma < 1`도 지원하지만,
toy 실험은 `gamma=1`이므로 두 값 모두 비할인 누적이익이며 Simulator의
`cumulative_profit` 및 `discounted_profit`과 같은 기준이다. D2C 고정비용은
0으로 유지한다.

## Python mapping

| Math | Python |
| --- | --- |
| `y_it` | `y[i, t]` |
| `q_it` | `q[i, t]` |
| `x_rit` | `x[r, i, t]` |
| `g_rt` | `g[r, t]` |
| `e_rt` | `e[r, t]` |
