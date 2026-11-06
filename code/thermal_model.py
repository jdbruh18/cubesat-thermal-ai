"""
4-node lumped-parameter thermal model of a 3U CubeSat in LEO.

Nodes (indices):
  0 SP  – solar panel / outer body (exposed)
  1 CH  – chassis / primary structure
  2 BAT – battery pack
  3 EL  – electronics (OBC / payload)

Truth vs nominal:
  The *truth* dynamics include attitude-dependent solar view factors,
  orbit-phase albedo modulation, and slightly perturbed conductances.
  The *nominal* physics predictor used for forecasting intentionally
  omits those effects (fixed average view factors, constant albedo,
  nominal G matrix) so that residual ML has a meaningful task.

Parameter sources (documented; see paper References):
  - Solar constant ~1367 W/m^2: Gilmore, Spacecraft Thermal Control Handbook.
  - Earth albedo ~0.30, Earth IR ~237 W/m^2: typical LEO design values
    (Gilmore; NASA Preferred Practice 2301).
  - Stefan–Boltzmann sigma = 5.670374419e-8 W/m^2/K^4.
  - Optical properties: solar absorptivity / IR emissivity for typical
    external finishes (black paint / anodized Al) from STCH App. A /
    preliminary small-sat analyses.
  - Thermal capacitances and conductances: order-of-magnitude values
    consistent with published CubeSat lumped models
    (e.g. Corpino et al. Appl. Sci. 2021; Bárczy et al. HJIC 2019;
     SmallSat battery-heater studies).
  - Battery operational window: Li-ion charge 0–45 °C, preferred ~10–35 °C
    (CEAS Space J. 2023 COTS modular battery; Energies 2020 CubeSat battery review).

This module contains NO proprietary company material.
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
from dataclasses import dataclass, field
from typing import Dict, Tuple

SIGMA = 5.670374419e-8  # Stefan–Boltzmann [W/m^2/K^4]
SOLAR_CONSTANT = 1367.0  # W/m^2 (Gilmore / WRC)
EARTH_RADIUS_KM = 6371.0
MU_EARTH = 3.986004418e14  # m^3/s^2

NODE_NAMES = ("SP", "CH", "BAT", "EL")
N_NODES = 4


@dataclass
class OrbitParams:
    """Circular LEO orbit parameters."""

    altitude_km: float = 500.0
    beta_deg: float = 0.0  # beta angle [deg]; affects eclipse

    @property
    def radius_m(self) -> float:
        return (EARTH_RADIUS_KM + self.altitude_km) * 1000.0

    @property
    def period_s(self) -> float:
        return 2.0 * np.pi * np.sqrt(self.radius_m**3 / MU_EARTH)

    @property
    def eclipse_fraction(self) -> float:
        """Geometric eclipse fraction for circular orbit (cylindrical Earth shadow).

        For beta = 0, eclipse is longest. For |beta| large enough that the
        orbit never enters shadow, fraction is 0.
        Formula from standard LEO geometry (approx.):
          rho = arcsin(R_E / (R_E+h))
          if |beta| >= rho: no eclipse
          else: f_e = (1/pi) * acos( sqrt(h*(2*R_E+h)) / ((R_E+h)*cos(beta)) )
        """
        re = EARTH_RADIUS_KM
        h = self.altitude_km
        beta = np.deg2rad(abs(self.beta_deg))
        rho = np.arcsin(re / (re + h))
        if beta >= rho:
            return 0.0
        arg = np.sqrt(h * (2 * re + h)) / ((re + h) * np.cos(beta))
        arg = np.clip(arg, -1.0, 1.0)
        return float(np.arccos(arg) / np.pi)

    def in_eclipse(self, t: float) -> bool:
        """True if spacecraft is in eclipse at time t (t=0 at eclipse exit / dawn).

        Eclipse is centered on the anti-sun point of the orbit.
        """
        period = self.period_s
        fe = self.eclipse_fraction
        if fe <= 0:
            return False
        phase = (t % period) / period  # 0..1
        # Place eclipse symmetrically around phase 0.5
        half = fe / 2.0
        return (0.5 - half) <= phase < (0.5 + half)

    def sun_flag(self, t: float) -> float:
        return 0.0 if self.in_eclipse(t) else 1.0

    def orbit_phase(self, t: float) -> float:
        return (t % self.period_s) / self.period_s


@dataclass
class ThermalParams:
    """Lumped thermal parameters for the 4-node CubeSat."""

    # Thermal capacitances [J/K] — order-of-magnitude for 3U components
    C: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                180.0,  # SP: panels + outer skin (low mass, responsive)
                450.0,  # CH: Al chassis
                250.0,  # BAT: cells + pack structure
                150.0,  # EL: OBC/boards
            ],
            dtype=float,
        )
    )

    # Conductive couplings G_ij [W/K], symmetric; diagonal unused
    # Nominal values (physics predictor uses these)
    G_nominal: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.0, 0.35, 0.05, 0.05],  # SP–CH primary; weak to BAT/EL
                [0.35, 0.0, 0.55, 0.40],  # CH–BAT, CH–EL
                [0.05, 0.55, 0.0, 0.15],  # BAT–EL weak
                [0.05, 0.40, 0.15, 0.0],
            ],
            dtype=float,
        )
    )

    # Truth-model conductance perturbation (multiplicative on off-diagonal)
    G_truth_scale: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.0, 1.25, 0.75, 0.80],
                [1.25, 1.0, 0.72, 1.20],
                [0.75, 0.72, 1.0, 1.30],
                [0.80, 1.20, 1.30, 1.0],
            ],
            dtype=float,
        )
    )

    # Radiating / absorbing areas [m^2] per node (effective)
    # 3U: ~0.1 x 0.1 x 0.3 m; exposed SP ~ 2*0.03 + ends; rough effective areas
    A_rad: np.ndarray = field(
        default_factory=lambda: np.array([0.06, 0.04, 0.008, 0.010], dtype=float)
    )
    A_solar: np.ndarray = field(
        default_factory=lambda: np.array([0.04, 0.015, 0.0, 0.0], dtype=float)
    )
    A_planet: np.ndarray = field(
        default_factory=lambda: np.array([0.03, 0.02, 0.0, 0.0], dtype=float)
    )

    # Optical properties
    alpha_s: np.ndarray = field(
        default_factory=lambda: np.array([0.75, 0.45, 0.0, 0.0], dtype=float)
    )  # solar absorptivity (SP dark cells/coverglass blend; CH anodized)
    epsilon: np.ndarray = field(
        default_factory=lambda: np.array([0.82, 0.70, 0.85, 0.85], dtype=float)
    )  # IR emissivity

    # Internal dissipation [W] (average; can be time-varying externally)
    Q_internal: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.5, 0.2, 1.5, 2.5], dtype=float
        )  # SP losses, CH, BAT self-heat, EL
    )

    # Earth IR flux [W/m^2] and albedo factor (nominal)
    q_earth_ir: float = 237.0
    albedo: float = 0.30

    # Heater max power on battery node [W]
    heater_max_w: float = 5.0

    # Deep-space temperature for radiation sink [K] (approx 3 K ≈ 0 for practical purposes)
    T_space: float = 3.0


def view_factor_solar_truth(orbit: OrbitParams, t: float) -> float:
    """Attitude-dependent solar view factor multiplier for truth model.

    Models a slowly tumbling / sun-tracking imperfect attitude: the projected
    solar area varies between ~0.55 and 1.0 of the nominal A_solar while in sun.
    """
    if orbit.in_eclipse(t):
        return 0.0
    phase = orbit.orbit_phase(t)
    # Two peaks per orbit from body rotation relative to sun vector
    return 0.55 + 0.45 * np.sin(4.0 * np.pi * phase)


def albedo_truth(orbit: OrbitParams, t: float) -> float:
    """Orbit-phase albedo modulation (land/ocean/cloud variability proxy)."""
    if orbit.in_eclipse(t):
        return 0.0
    phase = orbit.orbit_phase(t)
    # Higher near subsolar (phase away from eclipse center 0.5)
    base = 0.30
    return base * (0.55 + 0.55 * np.cos(2.0 * np.pi * (phase - 0.0)))


def external_heat(
    params: ThermalParams,
    orbit: OrbitParams,
    t: float,
    truth: bool,
) -> np.ndarray:
    """External environmental heat loads [W] for each node."""
    sun = orbit.sun_flag(t)
    if truth:
        vf = view_factor_solar_truth(orbit, t)
        alb = albedo_truth(orbit, t)
    else:
        vf = sun  # fixed average = 1 when in sun
        alb = params.albedo * sun

    q = np.zeros(N_NODES)
    # Direct solar
    q += params.alpha_s * params.A_solar * SOLAR_CONSTANT * vf
    # Albedo (only meaningful in sun; simplified Earth-view factor absorbed in A_planet)
    # Use a geometric Earth-view factor ~0.5 for LEO as a crude average.
    F_earth = 0.5
    q += params.alpha_s * params.A_planet * SOLAR_CONSTANT * alb * F_earth
    # Earth IR (always, proportional to Earth view; eclipse does not remove IR)
    q += params.epsilon * params.A_planet * params.q_earth_ir * F_earth
    return q


def rhs(
    T: np.ndarray,
    t: float,
    params: ThermalParams,
    orbit: OrbitParams,
    heater_w: float,
    truth: bool,
    q_internal_override: np.ndarray | None = None,
) -> np.ndarray:
    """dT/dt for the 4-node network [K/s]."""
    T = np.asarray(T, dtype=float)
    G = params.G_nominal * (params.G_truth_scale if truth else 1.0)
    Q_ext = external_heat(params, orbit, t, truth=truth)
    Q_int = params.Q_internal if q_internal_override is None else q_internal_override
    Q_h = np.zeros(N_NODES)
    Q_h[2] = float(np.clip(heater_w, 0.0, params.heater_max_w))  # BAT heater

    # Conduction
    Q_cond = np.zeros(N_NODES)
    for i in range(N_NODES):
        for j in range(N_NODES):
            if i == j:
                continue
            Q_cond[i] += G[i, j] * (T[j] - T[i])

    # Radiation to space (linearized not used; full T^4)
    Q_rad = -params.epsilon * params.A_rad * SIGMA * (T**4 - params.T_space**4)

    # Weak internal radiation BAT–EL (optional coupling via linearized rad)
    # Keep simple: already covered by conduction.

    Q_tot = Q_ext + Q_int + Q_h + Q_cond + Q_rad
    return Q_tot / params.C


def integrate_segment(
    T0: np.ndarray,
    t0: float,
    dt: float,
    n_steps: int,
    params: ThermalParams,
    orbit: OrbitParams,
    heater_fn,
    truth: bool = True,
    q_internal_fn=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RK4 integrate for n_steps of size dt.

    heater_fn(t, T) -> heater watts
    Returns times, temperatures (n_steps+1, 4), heater history (n_steps+1,)
    """
    times = np.zeros(n_steps + 1)
    temps = np.zeros((n_steps + 1, N_NODES))
    heaters = np.zeros(n_steps + 1)
    times[0] = t0
    temps[0] = np.asarray(T0, dtype=float)
    heaters[0] = float(heater_fn(t0, temps[0]))

    T = temps[0].copy()
    t = t0
    for k in range(n_steps):
        h = float(heater_fn(t, T))
        q_ov = None if q_internal_fn is None else q_internal_fn(t)

        def f(tt, TT, hh=h, qo=q_ov):
            return rhs(TT, tt, params, orbit, hh, truth=truth, q_internal_override=qo)

        k1 = f(t, T)
        k2 = f(t + 0.5 * dt, T + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, T + 0.5 * dt * k2)
        k4 = f(t + dt, T + dt * k3)
        T = T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Physical floor
        T = np.maximum(T, 50.0)  # > absolute zero with margin
        t = t + dt
        times[k + 1] = t
        temps[k + 1] = T
        heaters[k + 1] = float(heater_fn(t, T))
    return times, temps, heaters


def estimate_solar_input_nominal(
    params: ThermalParams, orbit: OrbitParams, t: float
) -> float:
    """Scalar estimated solar+albedo input on SP+CH for features [W]."""
    return float(np.sum(external_heat(params, orbit, t, truth=False)[:2]))


def kelvin_to_c(T_k: np.ndarray) -> np.ndarray:
    return np.asarray(T_k) - 273.15


def c_to_kelvin(T_c: np.ndarray) -> np.ndarray:
    return np.asarray(T_c) + 273.15


if __name__ == "__main__":
    o = OrbitParams(altitude_km=500, beta_deg=0)
    print(f"Period: {o.period_s/60:.2f} min, eclipse fraction: {o.eclipse_fraction:.3f}")
    for b in [0, 30, 60, 75]:
        oo = OrbitParams(beta_deg=b)
        print(f"  beta={b}: fe={oo.eclipse_fraction:.3f}")
