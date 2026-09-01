# D2C Assortment and Allocation Research Code

This project studies a manufacturer's dynamic D2C assortment and channel
allocation decisions while independent retailers adjust their future baseline
orders in response to D2C exposure.

## Current scope

The current baseline includes a controlled 4-SKU, 3-retailer instance, validated
JSON loading, and one Gurobi MILP formulation for both myopic (`horizon=1`) and
look-ahead (`horizon>1`) optimization. It does not yet run a rolling-horizon
simulation.

Core paths:

- `configs/toy.json`: controlled research instance.
- `src/types.py`: domain and solution data structures.
- `src/instance.py`: JSON parsing, validation, and initial state construction.
- `src/milp.py`: linear Gurobi formulation and solution extraction.
- `docs/model.md`: compact formulation reference.
- `tests/`: data and solver checks.

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
pytest -q
```

Solver-dependent tests are skipped when `gurobipy` is unavailable. A present but
unusable Gurobi license is also reported as a skip.

## Next steps

The next implementation sequence is `transition.py` then `simulator.py`, followed
by rolling-horizon myopic-versus-dynamic experiments and sensitivity analysis.
