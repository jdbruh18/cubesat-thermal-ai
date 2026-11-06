# Lightweight ML for CubeSat Predictive Thermal Management

Independent academic simulation study: a transparent **4-node lumped-parameter 3U CubeSat** thermal model in LEO, plus lightweight predictors (Random Forest, XGBoost, small MLP) and a comparison of **thermostat vs predictive battery-heater control**.

**Data are entirely simulated** from a documented public-physics model. This repository contains **no flight telemetry** and does not claim on-orbit performance.

## Quick start

```bash
cd /path/to/cubesat-thermal-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# Full pipeline (simulate → train/evaluate → control → figures)
python code/run_all.py
```

Or step by step (any working directory; paths are resolved from the script location):

```bash
python code/simulate.py
python code/train_predict.py
python code/control_compare.py
python code/plot_figures.py
```

Expected runtime on a modest multi-core CPU: roughly **8–15 minutes** for the full pipeline (physics open-loop prediction over long horizons dominates).

## What the study asks

Can a standard lumped CubeSat thermal model plus lightweight ML forecast battery temperature accurately enough (10-minute horizon) to reduce heater energy and/or thermal-limit violations compared with a simple hysteresis thermostat, in simulation?

## Model (summary)

| Item | Choice |
|------|--------|
| Platform | 3U CubeSat, 4 nodes: solar panel (SP), chassis (CH), battery (BAT), electronics (EL) |
| Orbit | Circular LEO, altitude 500 km; β ∈ {0°, 30°, 60°, 75°} |
| Integration | RK4, Δt = 5 s |
| Truth vs predictor | Truth includes attitude-dependent solar view factors, albedo modulation, and perturbed conductances; the physics predictor uses a nominal/simplified model |
| Battery limits | Hard window 0–40 °C; thermostat T_on = 2 °C, T_off = 8 °C |
| Prediction horizon | 600 s (10 min) |
| Validation | Leave-one-regime-out (hold out one β angle) |

## Outputs

- `results/*.csv` — trajectories, prediction metrics, control metrics, latency/size
- `figures/fig1_…fig6_….png` — publication figures
- `paper/manuscript.md` / `paper/manuscript.docx` — paper-only draft
- `CubeSat_Thermal_Management_Draft.md` / `.docx` — named-author draft at repo root

## Reproducibility

- RNG seeds fixed (`SEED = 0` / regime seeds `100+i`)
- Dependency versions: see `requirements.txt`
- Re-run commands above; metrics are written to CSV and must match the manuscript tables

## Layout

```
cubesat-thermal-ai/
  code/           simulator, training, control, plotting
  results/        CSVs and cached models
  figures/        PNG figures
  paper/          manuscript
  requirements.txt
  README.md
```
