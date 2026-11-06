"""Generate multi-orbit CubeSat thermal trajectories under several beta angles."""

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
import pandas as pd

from thermal_model import (
    OrbitParams,
    ThermalParams,
    integrate_segment,
    kelvin_to_c,
    c_to_kelvin,
    estimate_solar_input_nominal,
    NODE_NAMES,
    N_NODES,
)

RNG = np.random.default_rng(42)

# Battery temperature limits (°C) — conservative operational window from literature
T_BAT_MIN_C = 0.0  # charging lower bound (CEAS 2023 / Energies 2020)
T_BAT_MAX_C = 40.0  # below upper charge limit 45 °C
T_ON_C = 2.0  # thermostat turn-on (near hard limit)
T_OFF_C = 8.0  # thermostat turn-off (hysteresis)
# Electronics soft limits (monitored, not actively heated in baseline)
T_EL_MIN_C = -10.0
T_EL_MAX_C = 50.0


def thermostat_controller_factory(T_on=T_ON_C, T_off=T_OFF_C, power=5.0):
    """Hysteresis thermostat on battery temperature (°C). State via closure."""
    state = {"on": False}

    def fn(t, T_k):
        T_bat_c = float(T_k[2] - 273.15)
        if state["on"]:
            if T_bat_c >= T_off:
                state["on"] = False
        else:
            if T_bat_c <= T_on:
                state["on"] = True
        return power if state["on"] else 0.0

    return fn


def open_loop_zero(t, T_k):
    return 0.0


def q_internal_varying(t, orbit: OrbitParams):
    """Mild duty-cycled electronics dissipation (truth only)."""
    base = np.array([0.5, 0.2, 1.5, 2.5], dtype=float)
    phase = orbit.orbit_phase(t)
    # EL higher in sun (payload ops), BAT higher when charging in sun
    el_scale = 1.0 + 0.4 * (0.0 if orbit.in_eclipse(t) else 1.0)
    bat_scale = 1.0 + 0.3 * (0.0 if orbit.in_eclipse(t) else 1.0)
    q = base.copy()
    q[3] *= el_scale
    q[2] *= bat_scale
    # Small high-frequency ripple
    q[3] += 0.15 * np.sin(2 * np.pi * phase * 8)
    return q


def simulate_regime(
    beta_deg: float,
    n_orbits: float = 8.0,
    dt: float = 5.0,
    sensor_noise_std_c: float = 0.15,
    T0_c: np.ndarray | None = None,
    heater_fn=None,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate truth dynamics; return dataframe with truth + noisy observations."""
    rng = np.random.default_rng(seed)
    orbit = OrbitParams(altitude_km=500.0, beta_deg=beta_deg)
    params = ThermalParams()
    if T0_c is None:
        # Start near room-ish cold-biased LEO
        T0_c = np.array([-2.0, 2.0, 3.0, 5.0])
    T0 = c_to_kelvin(T0_c)
    period = orbit.period_s
    n_steps = int(n_orbits * period / dt)
    if heater_fn is None:
        heater_fn = thermostat_controller_factory()

    def q_fn(t):
        return q_internal_varying(t, orbit)

    times, temps, heaters = integrate_segment(
        T0,
        0.0,
        dt,
        n_steps,
        params,
        orbit,
        heater_fn,
        truth=True,
        q_internal_fn=q_fn,
    )
    T_c = kelvin_to_c(temps)
    noise = rng.normal(0.0, sensor_noise_std_c, size=T_c.shape)
    T_obs = T_c + noise

    rows = {
        "t_s": times,
        "beta_deg": np.full_like(times, beta_deg),
        "orbit_phase": np.array([orbit.orbit_phase(tt) for tt in times]),
        "eclipse": np.array([1.0 if orbit.in_eclipse(tt) else 0.0 for tt in times]),
        "sun": np.array([orbit.sun_flag(tt) for tt in times]),
        "heater_w": heaters,
        "period_s": np.full_like(times, period),
        "eclipse_fraction": np.full_like(times, orbit.eclipse_fraction),
    }
    for i, name in enumerate(NODE_NAMES):
        rows[f"T_{name}_true_c"] = T_c[:, i]
        rows[f"T_{name}_obs_c"] = T_obs[:, i]
    rows["Q_solar_est_w"] = np.array(
        [estimate_solar_input_nominal(params, orbit, tt) for tt in times]
    )
    rows["Q_internal_el_w"] = np.array([q_fn(tt)[3] for tt in times])
    rows["Q_internal_bat_w"] = np.array([q_fn(tt)[2] for tt in times])
    return pd.DataFrame(rows)


def build_feature_matrix(
    df: pd.DataFrame,
    horizon_s: float = 180.0,
    dt: float = 5.0,
    history_steps: int = 6,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build supervised learning set for battery temperature prediction.

    Target: T_BAT_true at t + horizon (no leakage from future features).
    Features use only information available at time t (observed temps, etc.).
    """
    h_steps = int(round(horizon_s / dt))
    n = len(df)
    feats = []
    targets = []
    meta_idx = []
    for k in range(history_steps, n - h_steps):
        row = df.iloc[k]
        hist = df.iloc[k - history_steps : k + 1]
        f = {
            "T_SP_obs": row["T_SP_obs_c"],
            "T_CH_obs": row["T_CH_obs_c"],
            "T_BAT_obs": row["T_BAT_obs_c"],
            "T_EL_obs": row["T_EL_obs_c"],
            "dT_BAT_dt": (row["T_BAT_obs_c"] - hist.iloc[0]["T_BAT_obs_c"])
            / (history_steps * dt),
            "dT_EL_dt": (row["T_EL_obs_c"] - hist.iloc[0]["T_EL_obs_c"])
            / (history_steps * dt),
            "eclipse": row["eclipse"],
            "sun": row["sun"],
            "orbit_phase_sin": np.sin(2 * np.pi * row["orbit_phase"]),
            "orbit_phase_cos": np.cos(2 * np.pi * row["orbit_phase"]),
            "Q_solar_est": row["Q_solar_est_w"],
            "Q_internal_el": row["Q_internal_el_w"],
            "Q_internal_bat": row["Q_internal_bat_w"],
            "heater_w": row["heater_w"],
            "heater_recent_mean": hist["heater_w"].mean(),
            "time_in_sun_frac": hist["sun"].mean(),
            "beta_deg": row["beta_deg"],
            "eclipse_fraction": row["eclipse_fraction"],
            "T_BAT_hist_mean": hist["T_BAT_obs_c"].mean(),
            "T_BAT_hist_min": hist["T_BAT_obs_c"].min(),
        }
        target = df.iloc[k + h_steps]["T_BAT_true_c"]
        feats.append(f)
        targets.append(target)
        meta_idx.append(k)
    X = pd.DataFrame(feats)
    y = np.asarray(targets, dtype=float)
    idx = np.asarray(meta_idx, dtype=int)
    return X, y, idx


if __name__ == "__main__":
    out = RESULTS
    os.makedirs(out, exist_ok=True)
    betas = [0.0, 30.0, 60.0, 75.0]
    frames = []
    for i, b in enumerate(betas):
        print(f"Simulating beta={b} ...")
        df = simulate_regime(b, n_orbits=8.0, dt=5.0, seed=100 + i)
        df.to_csv(os.path.join(out, f"traj_beta_{int(b)}.csv"), index=False)
        frames.append(df)
        print(
            f"  n={len(df)}, T_BAT range [{df.T_BAT_true_c.min():.1f}, {df.T_BAT_true_c.max():.1f}] °C, "
            f"fe={df.eclipse_fraction.iloc[0]:.3f}"
        )
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(os.path.join(out, "traj_all.csv"), index=False)
    print("Wrote traj_all.csv")
