"""Publication-quality figures for the CubeSat thermal paper."""

from __future__ import annotations

import os
import sys
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_CODE_DIR, ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
RESULTS = os.path.join(_ROOT, "results")
FIGDIR = os.path.join(_ROOT, "figures")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

FIG = FIGDIR
os.makedirs(FIG, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
    }
)


def shade_eclipse(ax, t, eclipse, t_max=None):
    """Shade eclipse intervals."""
    if t_max is not None:
        mask = t <= t_max
        t = t[mask]
        eclipse = eclipse[mask]
    in_e = False
    start = None
    for i in range(len(t)):
        if eclipse[i] > 0.5 and not in_e:
            in_e = True
            start = t[i]
        elif eclipse[i] < 0.5 and in_e:
            ax.axvspan(start / 60.0, t[i] / 60.0, color="0.85", zorder=0)
            in_e = False
    if in_e:
        ax.axvspan(start / 60.0, t[-1] / 60.0, color="0.85", zorder=0)


def fig1_multi_orbit_temps():
    """Multi-orbit temperatures with eclipse shading (thermostat, beta=0)."""
    path = os.path.join(RESULTS, "traj_beta_0.csv")
    df = pd.read_csv(path)
    # First 3 orbits
    period = df["period_s"].iloc[0]
    tmax = 3 * period
    d = df[df["t_s"] <= tmax]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    ax = axes[0]
    shade_eclipse(ax, d["t_s"].values, d["eclipse"].values)
    for col, label, ls in [
        ("T_SP_true_c", "Solar panel", "-"),
        ("T_CH_true_c", "Chassis", "-"),
        ("T_BAT_true_c", "Battery", "-"),
        ("T_EL_true_c", "Electronics", "-"),
    ]:
        ax.plot(d["t_s"] / 60.0, d[col], ls, lw=1.2, label=label)
    ax.axhline(0.0, color="k", ls=":", lw=0.8, label=r"$T_{\mathrm{min}}=0\,^\circ$C")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("3U CubeSat node temperatures (β = 0°, thermostat heater)")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    shade_eclipse(ax2, d["t_s"].values, d["eclipse"].values)
    ax2.plot(d["t_s"] / 60.0, d["heater_w"], color="C3", lw=1.0)
    ax2.set_ylabel("Heater power (W)")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylim(-0.2, 5.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Battery heater command (eclipse shaded)")
    fig.tight_layout()
    out = os.path.join(FIG, "fig1_multi_orbit_temps.png")
    fig.savefig(out, bbox_inches="tight")
    # also to results
    fig.savefig(os.path.join(RESULTS, "fig1_multi_orbit_temps.png"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def fig2_pred_vs_actual():
    path = os.path.join(RESULTS, "preds_holdout_beta30.csv")
    df = pd.read_csv(path)
    # Plot a contiguous window of ~2 orbits worth of samples (index order ≈ time)
    n = min(len(df), 2000)
    d = df.iloc[:n]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(d))
    ax.plot(x, d["y_true"], "k-", lw=1.3, label="True $T_{BAT}$")
    ax.plot(x, d["pred_XGBoost"], lw=1.0, label="XGBoost")
    ax.plot(x, d["pred_Physics"], lw=1.0, alpha=0.85, label="Physics-only")
    ax.plot(x, d["pred_Persistence"], lw=0.8, alpha=0.7, label="Persistence")
    ax.set_xlabel("Sample index (held-out β = 30°)")
    ax.set_ylabel("Battery temperature (°C)")
    ax.set_title("10-minute-ahead battery temperature prediction (leave-one-regime-out)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig2_pred_vs_actual.png")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS, "fig2_pred_vs_actual.png"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def fig3_parity_residuals():
    path = os.path.join(RESULTS, "preds_holdout_beta30.csv")
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    ax = axes[0]
    ax.scatter(df["y_true"], df["pred_XGBoost"], s=4, alpha=0.25, c="C0", rasterized=True)
    lims = [
        min(df["y_true"].min(), df["pred_XGBoost"].min()) - 1,
        max(df["y_true"].max(), df["pred_XGBoost"].max()) + 1,
    ]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("True $T_{BAT}$ (°C)")
    ax.set_ylabel("Predicted $T_{BAT}$ (°C)")
    ax.set_title("XGBoost parity (β = 30° hold-out)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    resid = df["pred_XGBoost"] - df["y_true"]
    ax2.hist(resid, bins=40, color="C0", edgecolor="white", alpha=0.9)
    ax2.axvline(0, color="k", ls="--", lw=1)
    ax2.set_xlabel("Residual (°C)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Residuals (mean {resid.mean():.2f}, std {resid.std():.2f} °C)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig3_parity_residuals.png")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS, "fig3_parity_residuals.png"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def fig4_mae_by_model():
    path = os.path.join(RESULTS, "prediction_metrics_folds.csv")
    df = pd.read_csv(path)
    df = df[~df["model"].str.contains("ablation")]
    order = ["Persistence", "TrainMean", "Physics", "MLP", "RF", "XGBoost"]
    order = [m for m in order if m in df["model"].unique()]
    means = [df[df["model"] == m]["MAE"].mean() for m in order]
    stds = [df[df["model"] == m]["MAE"].std() for m in order]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=stds, capsize=3, color="steelblue", edgecolor="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Leave-one-regime-out battery temperature MAE (mean ± std)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig4_mae_by_model.png")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS, "fig4_mae_by_model.png"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


def fig5_control_energy_violations():
    path = os.path.join(RESULTS, "control_metrics_folds.csv")
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    ctrls = ["Thermostat", "Predictive_Physics", "Predictive_XGBoost"]
    labels = ["Thermostat", "Pred. Physics", "Pred. XGBoost"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    ax = axes[0]
    for i, (c, lab, col) in enumerate(zip(ctrls, labels, colors)):
        sub = df[df["controller"] == c]
        ax.bar(
            np.arange(len(sub)) + i * 0.25,
            sub["energy_Wh_per_orbit"],
            width=0.25,
            label=lab,
            color=col,
            edgecolor="k",
            lw=0.4,
        )
    ax.set_xticks(np.arange(len(df[df["controller"] == "Thermostat"])) + 0.25)
    betas = df[df["controller"] == "Thermostat"]["beta_deg"].astype(int).tolist()
    ax.set_xticklabels([f"β={b}°" for b in betas])
    ax.set_ylabel("Heater energy (Wh / orbit)")
    ax.set_title("Heater energy by regime")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    width = 0.25
    for i, (c, lab, col) in enumerate(zip(ctrls, labels, colors)):
        sub = df[df["controller"] == c]
        ax2.bar(
            np.arange(len(sub)) + i * width,
            100 * sub["frac_below_Tmin"],
            width=width,
            label=lab,
            color=col,
            edgecolor="k",
            lw=0.4,
        )
    ax2.set_xticks(np.arange(len(betas)) + width)
    ax2.set_xticklabels([f"β={b}°" for b in betas])
    ax2.set_ylabel("Time below 0 °C (%)")
    ax2.set_title("Lower-limit violations")
    ax2.legend(fontsize=7)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig5_control_energy_violations.png")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(
        os.path.join(RESULTS, "fig5_control_energy_violations.png"), bbox_inches="tight"
    )
    plt.close(fig)
    print("Wrote", out)


def fig6_heater_duty_compare():
    """Overlay BAT temp + heater for thermostat vs predictive at beta=0."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    styles = {
        "Thermostat": ("C0", "-"),
        "Predictive_XGBoost": ("C3", "-"),
        "Predictive_Physics": ("C2", "--"),
    }
    period = None
    for ctrl, (color, ls) in styles.items():
        path = os.path.join(RESULTS, f"control_traj_beta0_{ctrl}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        if period is None:
            # estimate period from eclipse pattern ~ 5700s
            period = 5677.0
        tmax = 2 * period
        d = d[d["t_s"] <= tmax]
        axes[0].plot(
            d["t_s"] / 60.0, d["T_BAT_c"], ls=ls, color=color, lw=1.2, label=ctrl.replace("_", " ")
        )
        axes[1].plot(
            d["t_s"] / 60.0, d["heater_w"], ls=ls, color=color, lw=1.0, label=ctrl.replace("_", " ")
        )
    # eclipse from thermostat file
    path = os.path.join(RESULTS, "control_traj_beta0_Thermostat.csv")
    if os.path.exists(path):
        d = pd.read_csv(path)
        d = d[d["t_s"] <= 2 * (period or 5677)]
        shade_eclipse(axes[0], d["t_s"].values, d["eclipse"].values)
        shade_eclipse(axes[1], d["t_s"].values, d["eclipse"].values)
    axes[0].axhline(0.0, color="k", ls=":", lw=0.8)
    axes[0].axhline(5.0, color="gray", ls=":", lw=0.8)
    axes[0].set_ylabel("$T_{BAT}$ (°C)")
    axes[0].set_title("Battery temperature under three heater policies (β = 0°)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Heater (W)")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylim(-0.2, 5.5)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig6_heater_duty_compare.png")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS, "fig6_heater_duty_compare.png"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    fig1_multi_orbit_temps()
    fig2_pred_vs_actual()
    fig3_parity_residuals()
    fig4_mae_by_model()
    fig5_control_energy_violations()
    fig6_heater_duty_compare()
    print("All figures done.")
