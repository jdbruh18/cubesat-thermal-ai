"""Nominal physics-model open-loop predictor for battery temperature.

Integrates the *nominal* (non-truth) 4-node model forward from the current
observed state for a prediction horizon. This is intentionally mismatched
to the truth model (no attitude VF modulation, constant albedo, nominal G).
"""

from __future__ import annotations

import os
import sys
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_CODE_DIR, ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
RESULTS = os.path.join(_ROOT, "results")
FIGDIR = os.path.join(_ROOT, "figures")

import numpy as np
from thermal_model import (
    OrbitParams,
    ThermalParams,
    rhs,
    c_to_kelvin,
    kelvin_to_c,
    N_NODES,
)


def predict_battery_temp(
    T_obs_c: np.ndarray,
    t0: float,
    horizon_s: float,
    beta_deg: float,
    heater_w: float = 0.0,
    dt: float = 5.0,
    altitude_km: float = 500.0,
) -> float:
    """Predict BAT temperature (°C) at t0+horizon using nominal physics."""
    params = ThermalParams()
    orbit = OrbitParams(altitude_km=altitude_km, beta_deg=beta_deg)
    T = c_to_kelvin(np.asarray(T_obs_c, dtype=float)).copy()
    n_steps = max(1, int(round(horizon_s / dt)))
    hdt = horizon_s / n_steps
    t = t0
    for _ in range(n_steps):
        # Euler is fine for short horizons with 5 s; use RK2 for stability
        k1 = rhs(T, t, params, orbit, heater_w, truth=False)
        k2 = rhs(T + hdt * k1, t + hdt, params, orbit, heater_w, truth=False)
        T = T + 0.5 * hdt * (k1 + k2)
        T = np.maximum(T, 50.0)
        t += hdt
    return float(kelvin_to_c(T)[2])


def predict_batch(
    df,
    idx: np.ndarray,
    horizon_s: float,
    dt: float = 5.0,
    assume_heater: str = "current",
) -> np.ndarray:
    """Predict for many indices. assume_heater: 'current' or 'zero'."""
    preds = np.zeros(len(idx))
    for i, k in enumerate(idx):
        row = df.iloc[k]
        T_obs = np.array(
            [
                row["T_SP_obs_c"],
                row["T_CH_obs_c"],
                row["T_BAT_obs_c"],
                row["T_EL_obs_c"],
            ]
        )
        h = float(row["heater_w"]) if assume_heater == "current" else 0.0
        preds[i] = predict_battery_temp(
            T_obs,
            float(row["t_s"]),
            horizon_s,
            float(row["beta_deg"]),
            heater_w=h,
            dt=dt,
        )
    return preds
