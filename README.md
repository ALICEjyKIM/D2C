# D2C Assortment and Allocation Research Code

This project studies a manufacturer's dynamic D2C assortment and channel
allocation decisions while independent retailers adjust their future baseline
orders in response to D2C exposure. It also supports exact minimax relative
regret planning over a finite scenario set.

Planning scenarios freeze the observed current-period market row and order
retention, then branch only future demand and margins. Evaluation paths remain
full realized paths; one of their period rows can be passed as
`current_observation` when planning. The toy three-period experiment sets
`gamma=1`, so MILP planning profit and simulator cumulative profit use the same
undiscounted basis.

## Current scope

The current baseline includes a controlled 4-SKU, 3-retailer instance, validated
JSON loading, one Gurobi MILP formulation for both myopic (`horizon=1`) and
look-ahead (`horizon>1`) optimization, and a rolling-horizon simulator that
compares planning horizons on realized profit.

Core paths:

- `configs/toy.json`: controlled research instance.
- `src/types.py`: domain and solution data structures.
- `src/instance.py`: JSON parsing, validation, and initial state construction.
- `src/milp.py`: linear Gurobi formulation and solution extraction.
- `src/transition.py`: deterministic order-retention update (constraint C7).
- `src/simulator.py`: rolling-horizon run and per-period profit accounting.
- `src/scenarios.py`: LHS paths and the three regret verification scenarios.
- `src/regret_experiment.py`: exact three-scenario regret verification run.
- `docs/model.md`: compact formulation reference.
- `tests/`: data, solver, transition, and simulation checks.

## Setup

Python 3.11 and a working Gurobi license are required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run these commands from the `d2c/` project root.

```bash
python main.py
python -m src.regret_experiment
pytest -q
```

Solver-dependent tests are skipped when `gurobipy` is unavailable. A present but
unusable Gurobi license is also reported as a skip.

## Next steps

Connect the exact regret decision to rolling evaluation paths before adding any
scenario-screening heuristic.
