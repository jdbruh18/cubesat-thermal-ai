"""Leave-one-regime-out training of RF, XGBoost, MLP + baselines + physics."""

from __future__ import annotations

import os
import sys
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_CODE_DIR, ".."))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
RESULTS = os.path.join(_ROOT, "results")
FIGDIR = os.path.join(_ROOT, "figures")

import time
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from simulate import build_feature_matrix, simulate_regime, T_ON_C, T_OFF_C
from physics_predictor import predict_batch

SEED = 0
HORIZON_S = 600.0  # 10-minute horizon
DT = 5.0
BETAS = [0.0, 30.0, 60.0, 75.0]
os.makedirs(RESULTS, exist_ok=True)


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    maxae = float(np.max(np.abs(y_true - y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MaxAE": maxae, "R2": r2}


def count_mlp_params(mlp: MLPRegressor) -> int:
    n = 0
    for a, b in zip(mlp.coefs_, mlp.intercepts_):
        n += a.size + b.size
    return n


def run():
    np.random.seed(SEED)
    # Load or generate trajectories (with thermostat so heater feature is informative)
    trajs = {}
    for i, b in enumerate(BETAS):
        path = os.path.join(RESULTS, f"traj_beta_{int(b)}.csv")
        if os.path.exists(path):
            trajs[b] = pd.read_csv(path)
        else:
            print(f"Generating beta={b}")
            trajs[b] = simulate_regime(b, n_orbits=8.0, dt=DT, seed=100 + i)
            trajs[b].to_csv(path, index=False)

    feature_cols = None
    fold_rows = []
    latency_rows = []
    size_rows = []
    pred_store = {}

    for hold in BETAS:
        print(f"\n=== Hold-out beta={hold} ===")
        train_dfs = [trajs[b] for b in BETAS if b != hold]
        test_df = trajs[hold]
        X_tr_list, y_tr_list = [], []
        for d in train_dfs:
            X, y, _ = build_feature_matrix(d, horizon_s=HORIZON_S, dt=DT)
            X_tr_list.append(X)
            y_tr_list.append(y)
        X_train = pd.concat(X_tr_list, ignore_index=True)
        y_train = np.concatenate(y_tr_list)
        X_test, y_test, idx_test = build_feature_matrix(
            test_df, horizon_s=HORIZON_S, dt=DT
        )
        feature_cols = list(X_train.columns)

        # Persistence baseline: predict current observed BAT temp
        y_pers = test_df.iloc[idx_test]["T_BAT_obs_c"].to_numpy()
        # Mean baseline
        y_mean = np.full_like(y_test, y_train.mean())

        # Physics-only
        print("  Physics predictor...")
        y_phys = predict_batch(test_df, idx_test, HORIZON_S, dt=DT)

        models = {}
        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )
        t0 = time.perf_counter()
        rf.fit(X_train, y_train)
        train_t = time.perf_counter() - t0
        models["RF"] = rf

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        models["XGBoost"] = xgb_model

        # Small MLP
        mlp = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        solver="adam",
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        mlp.fit(X_train, y_train)
        models["MLP"] = mlp

        # Evaluate
        preds = {
            "Persistence": y_pers,
            "TrainMean": y_mean,
            "Physics": y_phys,
        }
        for name, model in models.items():
            preds[name] = model.predict(X_test)

        for name, yp in preds.items():
            m = metrics(y_test, yp)
            fold_rows.append(
                {
                    "holdout_beta": hold,
                    "model": name,
                    "n_test": len(y_test),
                    **m,
                }
            )
            print(f"  {name}: MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} R2={m['R2']:.3f}")

        # Latency & size on this fold (report for hold=30 as representative later)
        for name, model in models.items():
            # single-sample latency
            x1 = X_test.iloc[[0]]
            for _ in range(5):
                model.predict(x1)
            ts = []
            for _ in range(100):
                t0 = time.perf_counter()
                model.predict(x1)
                ts.append(time.perf_counter() - t0)
            med_ms = 1e3 * float(np.median(ts))
            # batch
            t0 = time.perf_counter()
            model.predict(X_test)
            batch_per = 1e3 * (time.perf_counter() - t0) / max(len(X_test), 1)
            latency_rows.append(
                {
                    "holdout_beta": hold,
                    "model": name,
                    "single_ms_median": med_ms,
                    "batch_ms_per_sample": batch_per,
                    "n_test": len(X_test),
                }
            )
            # size
            import tempfile, pickle

            with tempfile.NamedTemporaryFile(delete=False) as f:
                path = f.name
            if name == "XGBoost":
                model.save_model(path + ".json")
                sz = os.path.getsize(path + ".json")
                os.remove(path + ".json")
            else:
                with open(path, "wb") as f:
                    pickle.dump(model, f)
                sz = os.path.getsize(path)
            try:
                os.remove(path)
            except OSError:
                pass
            n_params = None
            if name == "MLP":
                n_params = count_mlp_params(model.named_steps["mlp"])
            size_rows.append(
                {
                    "holdout_beta": hold,
                    "model": name,
                    "size_bytes": sz,
                    "n_params": n_params,
                }
            )

        pred_store[hold] = {
            "y_true": y_test,
            "preds": preds,
            "idx": idx_test,
            "X_test": X_test,
        }

        # Ablation on XGBoost for this fold
        full_cols = feature_cols
        reduced = [
            "T_BAT_obs",
            "dT_BAT_dt",
            "eclipse",
            "sun",
            "Q_solar_est",
            "heater_w",
            "orbit_phase_sin",
            "orbit_phase_cos",
        ]
        for tag, cols in [("full", full_cols), ("reduced", reduced)]:
            m2 = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            )
            m2.fit(X_train[cols], y_train)
            yp = m2.predict(X_test[cols])
            mm = metrics(y_test, yp)
            fold_rows.append(
                {
                    "holdout_beta": hold,
                    "model": f"XGBoost_ablation_{tag}",
                    "n_test": len(y_test),
                    **mm,
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(RESULTS, "prediction_metrics_folds.csv"), index=False)

    # Aggregate mean ± std (exclude ablation from main summary)
    main = fold_df[~fold_df["model"].str.contains("ablation")]
    summary = (
        main.groupby("model")[["MAE", "RMSE", "MaxAE", "R2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    # Flatten columns
    summary.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns
    ]
    # Cleaner summary table
    rows = []
    for model in main["model"].unique():
        sub = main[main["model"] == model]
        rows.append(
            {
                "model": model,
                "MAE_mean": sub["MAE"].mean(),
                "MAE_std": sub["MAE"].std(),
                "RMSE_mean": sub["RMSE"].mean(),
                "RMSE_std": sub["RMSE"].std(),
                "MaxAE_mean": sub["MaxAE"].mean(),
                "R2_mean": sub["R2"].mean(),
                "R2_std": sub["R2"].std(),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("MAE_mean")
    summary_df.to_csv(os.path.join(RESULTS, "prediction_metrics_summary.csv"), index=False)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    pd.DataFrame(latency_rows).to_csv(
        os.path.join(RESULTS, "latency.csv"), index=False
    )
    pd.DataFrame(size_rows).to_csv(os.path.join(RESULTS, "model_size.csv"), index=False)

    # Save one fold's predictions for plotting (hold 30)
    hold_plot = 30.0
    ps = pred_store[hold_plot]
    plot_df = pd.DataFrame(
        {
            "y_true": ps["y_true"],
            **{f"pred_{k}": v for k, v in ps["preds"].items()},
        }
    )
    plot_df.to_csv(os.path.join(RESULTS, "preds_holdout_beta30.csv"), index=False)

    # Feature list
    with open(os.path.join(RESULTS, "feature_list.json"), "w") as f:
        json.dump({"features": feature_cols, "horizon_s": HORIZON_S, "dt": DT}, f, indent=2)

    # Retrain best model (XGBoost) on all but one for control experiments later
    # Save models trained leaving out each beta
    import pickle

    model_dir = os.path.join(RESULTS, "models")
    os.makedirs(model_dir, exist_ok=True)
    for hold in BETAS:
        train_dfs = [trajs[b] for b in BETAS if b != hold]
        X_tr_list, y_tr_list = [], []
        for d in train_dfs:
            X, y, _ = build_feature_matrix(d, horizon_s=HORIZON_S, dt=DT)
            X_tr_list.append(X)
            y_tr_list.append(y)
        X_train = pd.concat(X_tr_list, ignore_index=True)
        y_train = np.concatenate(y_tr_list)
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        model.save_model(os.path.join(model_dir, f"xgb_holdout_beta{int(hold)}.json"))
        # Also RF
        rf = RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        with open(os.path.join(model_dir, f"rf_holdout_beta{int(hold)}.pkl"), "wb") as f:
            pickle.dump(rf, f)

    print("Done prediction experiments.")


if __name__ == "__main__":
    run()
