"""Compare thermostat vs predictive heater control across orbital regimes.

Predictive controller: if the best predictor (XGBoost trained leave-one-regime-out)
forecasts T_BAT below T_ON within the horizon with heater=0, turn heater on;
turn off if current T > T_OFF or predicted with heater on would exceed T_OFF+margin.
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

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from thermal_model import (
    OrbitParams,
    ThermalParams,
    integrate_segment,
    c_to_kelvin,
    kelvin_to_c,
    estimate_solar_input_nominal,
    NODE_NAMES,
)
from simulate import (
    thermostat_controller_factory,
    q_internal_varying,
    T_ON_C,
    T_OFF_C,
    T_BAT_MIN_C,
    T_BAT_MAX_C,
    build_feature_matrix,
)
from physics_predictor import predict_battery_temp

BETAS = [0.0, 30.0, 60.0, 75.0]
HORIZON_S = 600.0
DT = 5.0
HISTORY = 6
SEED = 0


def feature_row_from_history(hist_rows, orbit, params, beta_deg):
    """Build one feature dict from a list of recent state dicts (oldest→newest)."""
    cur = hist_rows[-1]
    dt = DT
    history_steps = len(hist_rows) - 1
    dT_BAT = (cur["T_BAT_obs"] - hist_rows[0]["T_BAT_obs"]) / (history_steps * dt)
    dT_EL = (cur["T_EL_obs"] - hist_rows[0]["T_EL_obs"]) / (history_steps * dt)
    heaters = [h["heater_w"] for h in hist_rows]
    suns = [h["sun"] for h in hist_rows]
    bats = [h["T_BAT_obs"] for h in hist_rows]
    phase = cur["orbit_phase"]
    return {
        "T_SP_obs": cur["T_SP_obs"],
        "T_CH_obs": cur["T_CH_obs"],
        "T_BAT_obs": cur["T_BAT_obs"],
        "T_EL_obs": cur["T_EL_obs"],
        "dT_BAT_dt": dT_BAT,
        "dT_EL_dt": dT_EL,
        "eclipse": cur["eclipse"],
        "sun": cur["sun"],
        "orbit_phase_sin": np.sin(2 * np.pi * phase),
        "orbit_phase_cos": np.cos(2 * np.pi * phase),
        "Q_solar_est": cur["Q_solar_est"],
        "Q_internal_el": cur["Q_internal_el"],
        "Q_internal_bat": cur["Q_internal_bat"],
        "heater_w": cur["heater_w"],
        "heater_recent_mean": float(np.mean(heaters)),
        "time_in_sun_frac": float(np.mean(suns)),
        "beta_deg": beta_deg,
        "eclipse_fraction": cur["eclipse_fraction"],
        "T_BAT_hist_mean": float(np.mean(bats)),
        "T_BAT_hist_min": float(np.min(bats)),
    }


def make_predictive_controller(model, feature_names, beta_deg, use_physics=False):
    """Stateful predictive heater using online feature history."""
    params = ThermalParams()
    orbit = OrbitParams(altitude_km=500.0, beta_deg=beta_deg)
    hist = []
    state = {"on": False}
    rng = np.random.default_rng(123 + int(beta_deg))

    def controller(t, T_k):
        T_c = kelvin_to_c(T_k)
        # Sensor noise on observations used by controller
        noise = rng.normal(0.0, 0.15, size=4)
        T_obs = T_c + noise
        q_int = q_internal_varying(t, orbit)
        snap = {
            "t": t,
            "T_SP_obs": float(T_obs[0]),
            "T_CH_obs": float(T_obs[1]),
            "T_BAT_obs": float(T_obs[2]),
            "T_EL_obs": float(T_obs[3]),
            "eclipse": 1.0 if orbit.in_eclipse(t) else 0.0,
            "sun": orbit.sun_flag(t),
            "orbit_phase": orbit.orbit_phase(t),
            "Q_solar_est": estimate_solar_input_nominal(params, orbit, t),
            "Q_internal_el": float(q_int[3]),
            "Q_internal_bat": float(q_int[2]),
            "heater_w": params.heater_max_w if state["on"] else 0.0,
            "eclipse_fraction": orbit.eclipse_fraction,
        }
        hist.append(snap)
        if len(hist) > HISTORY + 1:
            hist.pop(0)

        # Need history before predicting
        if len(hist) <= HISTORY:
            # fall back to thermostat
            if state["on"]:
                if T_obs[2] >= T_OFF_C:
                    state["on"] = False
            else:
                if T_obs[2] <= T_ON_C:
                    state["on"] = True
            return params.heater_max_w if state["on"] else 0.0

        feats = feature_row_from_history(hist, orbit, params, beta_deg)
        # Predict future BAT temp assuming heater OFF (worst case cooling)
        feats_off = dict(feats)
        feats_off["heater_w"] = 0.0
        feats_off["heater_recent_mean"] = 0.0

        if use_physics:
            T_pred = predict_battery_temp(
                np.array(
                    [
                        feats["T_SP_obs"],
                        feats["T_CH_obs"],
                        feats["T_BAT_obs"],
                        feats["T_EL_obs"],
                    ]
                ),
                t,
                HORIZON_S,
                beta_deg,
                heater_w=0.0,
                dt=DT,
            )
        else:
            X = pd.DataFrame([[feats_off[c] for c in feature_names]], columns=feature_names)
            T_pred = float(model.predict(X)[0])

        # Decision logic:
        # Predictive target band is warmer than the late-reacting thermostat:
        # pre-heat if forecast (heater off) falls below 5 °C within the horizon,
        # or if current obs already at thermostat T_ON.
        PRED_TARGET_C = 5.0
        if state["on"]:
            if T_obs[2] >= T_OFF_C:
                state["on"] = False
            elif T_pred > PRED_TARGET_C + 3.0 and T_obs[2] > PRED_TARGET_C:
                state["on"] = False
        else:
            if (T_pred <= PRED_TARGET_C or T_obs[2] <= T_ON_C) and T_obs[2] < T_OFF_C:
                state["on"] = True

        # Hard safety: never heat above max
        if T_obs[2] >= T_BAT_MAX_C - 2.0:
            state["on"] = False

        return params.heater_max_w if state["on"] else 0.0

    return controller


def evaluate_run(times, temps, heaters, period_s):
    T_c = kelvin_to_c(temps)
    T_bat = T_c[:, 2]
    dt = times[1] - times[0] if len(times) > 1 else DT
    energy_j = float(np.sum(heaters[:-1]) * dt)  # J = W*s
    energy_wh = energy_j / 3600.0
    n_orbits = (times[-1] - times[0]) / period_s
    energy_wh_per_orbit = energy_wh / max(n_orbits, 1e-9)

    viol_low = float(np.mean(T_bat < T_BAT_MIN_C))
    viol_high = float(np.mean(T_bat > T_BAT_MAX_C))
    # heater cycles: off->on transitions
    on = heaters > 0.5
    cycles = int(np.sum((~on[:-1]) & on[1:]))
    return {
        "energy_J": energy_j,
        "energy_Wh": energy_wh,
        "energy_Wh_per_orbit": energy_wh_per_orbit,
        "frac_below_Tmin": viol_low,
        "frac_above_Tmax": viol_high,
        "time_below_Tmin_s": viol_low * (times[-1] - times[0]),
        "heater_cycles": cycles,
        "T_bat_min": float(T_bat.min()),
        "T_bat_max": float(T_bat.max()),
        "T_bat_mean": float(T_bat.mean()),
        "n_orbits": n_orbits,
    }


def run_control_for_beta(beta_deg, controller_name, controller_fn, n_orbits=6.0, seed=0):
    orbit = OrbitParams(altitude_km=500.0, beta_deg=beta_deg)
    params = ThermalParams()
    T0 = c_to_kelvin(np.array([-2.0, 2.0, 3.0, 5.0]))
    n_steps = int(n_orbits * orbit.period_s / DT)

    def q_fn(t):
        return q_internal_varying(t, orbit)

    times, temps, heaters = integrate_segment(
        T0,
        0.0,
        DT,
        n_steps,
        params,
        orbit,
        controller_fn,
        truth=True,
        q_internal_fn=q_fn,
    )
    stats = evaluate_run(times, temps, heaters, orbit.period_s)
    stats.update(
        {
            "beta_deg": beta_deg,
            "controller": controller_name,
            "eclipse_fraction": orbit.eclipse_fraction,
        }
    )
    # Save trajectory snippet for plotting
    out = {
        "times": times,
        "temps_c": kelvin_to_c(temps),
        "heaters": heaters,
        "eclipse": np.array([1.0 if orbit.in_eclipse(tt) else 0.0 for tt in times]),
    }
    return stats, out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    model_dir = os.path.join(RESULTS, "models")
    # Feature names must match training
    # Load a sample to get feature order
    from simulate import simulate_regime

    sample = simulate_regime(0.0, n_orbits=1.0, dt=DT, seed=0)
    X_sample, _, _ = build_feature_matrix(sample, horizon_s=HORIZON_S, dt=DT)
    feature_names = list(X_sample.columns)

    all_stats = []
    traj_cache = {}

    for beta in BETAS:
        print(f"\n=== Control comparison beta={beta} ===")
        # 1. Thermostat
        therm = thermostat_controller_factory()
        st, tr = run_control_for_beta(beta, "Thermostat", therm, n_orbits=6.0, seed=0)
        all_stats.append(st)
        traj_cache[(beta, "Thermostat")] = tr
        print(
            f"  Thermostat: E={st['energy_Wh_per_orbit']:.4f} Wh/orbit, "
            f"frac_low={st['frac_below_Tmin']:.4f}, cycles={st['heater_cycles']}"
        )

        # 2. Predictive XGBoost (model trained excluding this beta)
        xgb_path = os.path.join(model_dir, f"xgb_holdout_beta{int(beta)}.json")
        if not os.path.exists(xgb_path):
            print(f"  Missing model {xgb_path}; skip predictive")
            continue
        model = xgb.XGBRegressor()
        model.load_model(xgb_path)
        pred_ctrl = make_predictive_controller(model, feature_names, beta, use_physics=False)
        st2, tr2 = run_control_for_beta(beta, "Predictive_XGBoost", pred_ctrl, n_orbits=6.0)
        all_stats.append(st2)
        traj_cache[(beta, "Predictive_XGBoost")] = tr2
        print(
            f"  Pred-XGB: E={st2['energy_Wh_per_orbit']:.4f} Wh/orbit, "
            f"frac_low={st2['frac_below_Tmin']:.4f}, cycles={st2['heater_cycles']}"
        )

        # 3. Predictive Physics
        phys_ctrl = make_predictive_controller(None, feature_names, beta, use_physics=True)
        st3, tr3 = run_control_for_beta(beta, "Predictive_Physics", phys_ctrl, n_orbits=6.0)
        all_stats.append(st3)
        traj_cache[(beta, "Predictive_Physics")] = tr3
        print(
            f"  Pred-Phys: E={st3['energy_Wh_per_orbit']:.4f} Wh/orbit, "
            f"frac_low={st3['frac_below_Tmin']:.4f}, cycles={st3['heater_cycles']}"
        )

    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(RESULTS, "control_metrics_folds.csv"), index=False)

    summary_rows = []
    for ctrl in stats_df["controller"].unique():
        sub = stats_df[stats_df["controller"] == ctrl]
        summary_rows.append(
            {
                "controller": ctrl,
                "energy_Wh_per_orbit_mean": sub["energy_Wh_per_orbit"].mean(),
                "energy_Wh_per_orbit_std": sub["energy_Wh_per_orbit"].std(),
                "frac_below_Tmin_mean": sub["frac_below_Tmin"].mean(),
                "frac_below_Tmin_std": sub["frac_below_Tmin"].std(),
                "frac_above_Tmax_mean": sub["frac_above_Tmax"].mean(),
                "heater_cycles_mean": sub["heater_cycles"].mean(),
                "T_bat_min_mean": sub["T_bat_min"].mean(),
                "T_bat_max_mean": sub["T_bat_max"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(RESULTS, "control_metrics_summary.csv"), index=False)
    print("\n=== Control summary ===")
    print(summary.to_string(index=False))

    # Save one comparison trajectory for figures (beta=0, coldest eclipse)
    for ctrl in ["Thermostat", "Predictive_XGBoost", "Predictive_Physics"]:
        key = (0.0, ctrl)
        if key in traj_cache:
            tr = traj_cache[key]
            pd.DataFrame(
                {
                    "t_s": tr["times"],
                    "T_SP_c": tr["temps_c"][:, 0],
                    "T_CH_c": tr["temps_c"][:, 1],
                    "T_BAT_c": tr["temps_c"][:, 2],
                    "T_EL_c": tr["temps_c"][:, 3],
                    "heater_w": tr["heaters"],
                    "eclipse": tr["eclipse"],
                }
            ).to_csv(
                os.path.join(RESULTS, f"control_traj_beta0_{ctrl}.csv"), index=False
            )

    print("Done control comparison.")


if __name__ == "__main__":
    main()
