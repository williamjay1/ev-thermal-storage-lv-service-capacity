[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23671880.svg)](https://doi.org/10.5281/zenodo.23671880)

# Data and code: thermal storage and managed electric vehicle charging in low voltage networks

This archive supports the study *Thermal Storage And Managed Electric Vehicle Charging In Low
Voltage Networks: Service Capacity And Reinforcement Decisions*. It contains the scripts, the
processed inputs, the scenario outputs and the audit files behind the reported capacity bounds,
the certified cost intervals and the staged heat investment decisions.

## Contents

| Path | Content |
|---|---|
| `ev_experiment.py` | vehicle service capacity linear programs: 288 main annual cases and 192 service sensitivity cases |
| `ev_cost_screen.py` | fixed service cost screen over the finite transformer and corridor menu |
| `ev_cost_joint.py` | joint heat production, charging and network loss optimisation with lower and upper cost bounds |
| `data/development_cache.npz` | processed metered heat and electricity profiles for the two development years |
| `data/catalog/capacity_*.npz` | hourly AC envelopes for each network and asset configuration |
| `data/catalog/loss_*.npz` | fourth order active loss interpolation for each configuration |
| `data/catalog/operations_*.csv` | frozen operating schedules |
| `results_ev/` | capacity results, witnesses, AC validation and the independent audit |
| `results_ev_cost/` | fixed service decisions, fleet selection and cost options |
| `results_ev_cost_joint/` | joint cost bounds, convexity audit, loss validation and decisions |
| `original_heat_results/` | frozen staged heat investment plans, control benchmarks, cost, horizon and topology validation tables |

## Requirements

Python 3 with numpy, pandas, scipy and numba. The optional `--ac` and `--losscheck` switches
additionally require the network generation modules and a SimBench with pandapower environment.

## Running

From this directory in PowerShell:

```powershell
$env:PROJECT09_EV_DATA_ROOT = (Resolve-Path .\data).Path
python ev_experiment.py --main --sensitivity
python ev_cost_screen.py
python ev_cost_joint.py --run
```

Result rows already present are reused by the resumable scripts. To recompute from scratch, copy
the scripts and the `data` directory into a fresh working directory without result folders.

## Scope

Vehicle sessions are deterministic synthetic service contracts, not measured travel. Annual
schedules are separately periodic and assume complete information. The fixed service cost
comparison is exploratory with respect to the capacity calculation. The cost certificates apply to
the finite equipment menu and the stated loss surrogate, and they do not establish a utility
specific investment or a deferral date.

Original data platforms retain their rights in the original data. The scripts in this archive are
provided for review and reuse with attribution.
