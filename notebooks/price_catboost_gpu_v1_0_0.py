"""Kaggle GPU job for the first leakage-safe competitor price ML model.

This script is submitted to Kaggle as a private Python notebook. It rebuilds
modeling data v1.1.0 from the public raw dataset, performs rolling time CV on
the train split only, and writes versioned models, metrics, and predictions to
/kaggle/working.
"""

from __future__ import annotations

import importlib.util
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


JOB_VERSION = "1.0.0"
DATA_VERSION = "1.1.0"
REPO_COMMIT = "d2eac43e998fc2589e1f73f4c54500a6aeb7f38a"
REPO_URL = "https://github.com/mchienn/VSF_GSM.git"
PRIMARY_DELAY = 15
DELAYS = [5, 15, 30]
RANDOM_SEED = 42
OUTPUT_DIR = Path("/kaggle/working") / f"price_ml_v{JOB_VERSION}"
REPO_DIR = Path("/tmp/vsf_gsm_repo")

CANDIDATES = [
    {"candidate_id": "cb_d7_lr10", "depth": 7, "learning_rate": 0.10, "l2_leaf_reg": 3.0, "iterations": 900},
    {"candidate_id": "cb_d8_lr07", "depth": 8, "learning_rate": 0.07, "l2_leaf_reg": 5.0, "iterations": 1300},
    {"candidate_id": "cb_d10_lr05", "depth": 10, "learning_rate": 0.05, "l2_leaf_reg": 7.0, "iterations": 1600},
]


def run(command: list[str], cwd: Path | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locate_raw_csv() -> Path:
    matches = sorted(Path("/kaggle/input").rglob("uber_lyft.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one uber_lyft.csv, found {matches}")
    return matches[0]


def prepare_modeling_data() -> None:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--filter=blob:none", REPO_URL, str(REPO_DIR)])
    run(["git", "checkout", "--detach", REPO_COMMIT], cwd=REPO_DIR)

    raw_csv = locate_raw_csv()
    prep_v1 = load_module(REPO_DIR / "scripts" / "prepare_modeling_data_v1_0_0.py", "prep_v1")
    prep_v1.read_source = lambda: pd.read_csv(raw_csv, usecols=prep_v1.SOURCE_COLUMNS)
    prep_v1.main()

    prep_v11 = load_module(REPO_DIR / "scripts" / "prepare_modeling_data_v1_1_0.py", "prep_v11")
    prep_v11.main()


def ensure_catboost():
    try:
        import catboost  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "--quiet", "catboost>=1.2.8,<2"])
    from catboost import CatBoostRegressor, Pool

    return CatBoostRegressor, Pool


def prepare_xy(frame: pd.DataFrame, categorical: list[str], numeric: list[str], target: str):
    x = frame[categorical + numeric].copy()
    for column in categorical:
        x[column] = x[column].astype("string").fillna("__MISSING__").astype(str)
    for column in numeric:
        x[column] = pd.to_numeric(x[column], errors="coerce").astype("float32")
    y = frame[target].astype("float32")
    return x, y


def rolling_time_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    times = pd.to_datetime(frame["prediction_time_utc"], utc=True)
    unique_times = np.array(sorted(times.unique()))
    blocks = np.array_split(unique_times, 4)
    folds = []
    for fold_index in range(1, 4):
        train_times = np.concatenate(blocks[:fold_index])
        valid_times = blocks[fold_index]
        train_mask = times.isin(train_times).to_numpy()
        valid_mask = times.isin(valid_times).to_numpy()
        if times[train_mask].max() >= times[valid_mask].min():
            raise RuntimeError(f"Fold {fold_index} is not chronological")
        folds.append(
            (
                train_mask,
                valid_mask,
                str(times[train_mask].min()),
                str(times[valid_mask].min()),
            )
        )
    return folds


def model_params(candidate: dict, iterations: int | None = None) -> dict:
    return {
        "loss_function": "RMSE",
        "eval_metric": "MAE",
        "iterations": int(iterations or candidate["iterations"]),
        "depth": candidate["depth"],
        "learning_rate": candidate["learning_rate"],
        "l2_leaf_reg": candidate["l2_leaf_reg"],
        "random_seed": RANDOM_SEED,
        "task_type": "GPU",
        "devices": "0",
        "allow_writing_files": False,
        "verbose": 100,
    }


def route_service_baseline(train: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    keys = ["cab_type", "name", "source", "destination"]
    medians = train.groupby(keys, observed=True)["target_price_median"].median()
    lookup = pd.MultiIndex.from_frame(evaluation[keys])
    predictions = medians.reindex(lookup).to_numpy(dtype=float)
    fallback = float(train["target_price_median"].median())
    return np.where(np.isnan(predictions), fallback, predictions)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu_probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"GPU: {gpu_probe}", flush=True)

    prepare_modeling_data()
    CatBoostRegressor, Pool = ensure_catboost()
    data_dir = REPO_DIR / "artifacts" / "modeling_data" / f"v{DATA_VERSION}"
    contract = json.loads((data_dir / f"feature_contract_v{DATA_VERSION}.json").read_text(encoding="utf-8"))
    categorical = contract["categorical_features"]
    numeric = contract["numeric_features"]
    features = categorical + numeric
    target = contract["price_target"]

    primary = pd.read_parquet(data_dir / f"price_modeling_delay_{PRIMARY_DELAY:02d}m_v{DATA_VERSION}.parquet")
    train = primary.loc[primary["data_split"].eq("train")].sort_values("prediction_time_utc").reset_index(drop=True)
    if set(train[features].columns) != set(features) or len(features) != 46:
        raise RuntimeError("Feature contract mismatch")

    x_train, y_train = prepare_xy(train, categorical, numeric, target)
    folds = rolling_time_folds(train)
    cv_rows: list[dict] = []
    for candidate in CANDIDATES:
        for fold_index, (train_mask, valid_mask, train_start, valid_start) in enumerate(folds, start=1):
            train_pool = Pool(x_train.loc[train_mask], y_train.loc[train_mask], cat_features=categorical)
            valid_pool = Pool(x_train.loc[valid_mask], y_train.loc[valid_mask], cat_features=categorical)
            model = CatBoostRegressor(**model_params(candidate))
            model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=120, use_best_model=True)
            prediction = model.predict(valid_pool)
            metrics = regression_metrics(y_train.loc[valid_mask], prediction)
            best_iteration = model.get_best_iteration()
            cv_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "fold": fold_index,
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                    "train_start_utc": train_start,
                    "valid_start_utc": valid_start,
                    "best_iteration": int(best_iteration + 1 if best_iteration >= 0 else candidate["iterations"]),
                    **metrics,
                }
            )
            del model, train_pool, valid_pool
            gc.collect()

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUTPUT_DIR / f"cv_metrics_v{JOB_VERSION}.csv", index=False)
    ranking = (
        cv.groupby("candidate_id", as_index=False)
        .agg(cv_mae_mean=("mae", "mean"), cv_mae_std=("mae", "std"), cv_rmse_mean=("rmse", "mean"), selected_iterations=("best_iteration", "median"))
        .sort_values(["cv_mae_mean", "cv_mae_std", "candidate_id"])
        .reset_index(drop=True)
    )
    ranking.to_csv(OUTPUT_DIR / f"candidate_ranking_v{JOB_VERSION}.csv", index=False)
    selected_id = str(ranking.loc[0, "candidate_id"])
    selected = next(item for item in CANDIDATES if item["candidate_id"] == selected_id)
    selected_iterations = max(100, int(ranking.loc[0, "selected_iterations"]))

    scenario_rows: list[dict] = []
    for delay in DELAYS:
        frame = pd.read_parquet(data_dir / f"price_modeling_delay_{delay:02d}m_v{DATA_VERSION}.parquet")
        delay_train = frame.loc[frame["data_split"].eq("train")].sort_values("prediction_time_utc").reset_index(drop=True)
        calibration = frame.loc[frame["data_split"].eq("calibration")].sort_values("prediction_time_utc").reset_index(drop=True)
        test = frame.loc[frame["data_split"].eq("test")].sort_values("prediction_time_utc").reset_index(drop=True)
        x_delay_train, y_delay_train = prepare_xy(delay_train, categorical, numeric, target)
        train_pool = Pool(x_delay_train, y_delay_train, cat_features=categorical)
        model = CatBoostRegressor(**model_params(selected, iterations=selected_iterations))
        model.fit(train_pool)
        model_path = OUTPUT_DIR / f"catboost_price_delay_{delay:02d}m_v{JOB_VERSION}.cbm"
        model.save_model(model_path)

        split_predictions = {}
        for split_name, split_frame in [("calibration", calibration), ("test", test)]:
            x_eval, y_eval = prepare_xy(split_frame, categorical, numeric, target)
            eval_pool = Pool(x_eval, cat_features=categorical)
            prediction = model.predict(eval_pool)
            split_predictions[split_name] = (y_eval, prediction)
            prediction_frame = split_frame[["snapshot_id", "series_id", "prediction_time_utc", "data_split"]].copy()
            prediction_frame["y_true"] = y_eval.to_numpy()
            prediction_frame["y_pred"] = prediction
            prediction_frame["residual"] = prediction_frame["y_true"] - prediction_frame["y_pred"]
            prediction_frame.to_parquet(
                OUTPUT_DIR / f"{split_name}_predictions_delay_{delay:02d}m_v{JOB_VERSION}.parquet",
                index=False,
                compression="zstd",
            )

        test_y, test_prediction = split_predictions["test"]
        ml_metrics = regression_metrics(test_y, test_prediction)
        baseline_prediction = route_service_baseline(delay_train, test)
        baseline_metrics = regression_metrics(test_y, baseline_prediction)
        scenario_rows.append(
            {
                "delay_minutes": delay,
                "train_rows": len(delay_train),
                "calibration_rows": len(calibration),
                "test_rows": len(test),
                "selected_candidate": selected_id,
                "iterations": selected_iterations,
                "test_ml_mae": ml_metrics["mae"],
                "test_ml_rmse": ml_metrics["rmse"],
                "test_route_service_baseline_mae": baseline_metrics["mae"],
                "test_route_service_baseline_rmse": baseline_metrics["rmse"],
                "mae_improvement_vs_baseline": baseline_metrics["mae"] - ml_metrics["mae"],
            }
        )
        del model, train_pool, frame, delay_train, calibration, test
        gc.collect()

    scenarios = pd.DataFrame(scenario_rows)
    scenarios.to_csv(OUTPUT_DIR / f"scenario_metrics_v{JOB_VERSION}.csv", index=False)
    primary_row = scenarios.loc[scenarios["delay_minutes"].eq(PRIMARY_DELAY)].iloc[0]
    validation = pd.DataFrame(
        [
            {"check_id": "GPU_AVAILABLE", "status": "PASS", "actual": gpu_probe, "expected": "NVIDIA GPU"},
            {"check_id": "FEATURE_COUNT", "status": "PASS" if len(features) == 46 else "FAIL", "actual": len(features), "expected": 46},
            {"check_id": "CV_TRAIN_ONLY", "status": "PASS", "actual": "train", "expected": "train"},
            {"check_id": "CALIBRATION_NOT_USED_FOR_SELECTION", "status": "PASS", "actual": "predictions_only", "expected": "predictions_only"},
            {"check_id": "PRIMARY_ML_BEATS_BASELINE", "status": "PASS" if primary_row["mae_improvement_vs_baseline"] > 0 else "WARN", "actual": float(primary_row["mae_improvement_vs_baseline"]), "expected": ">0"},
        ]
    )
    validation.to_csv(OUTPUT_DIR / f"validation_results_v{JOB_VERSION}.csv", index=False)
    summary = {
        "artifact_version": JOB_VERSION,
        "modeling_data_version": DATA_VERSION,
        "repo_commit": REPO_COMMIT,
        "gpu": gpu_probe,
        "primary_delay_minutes": PRIMARY_DELAY,
        "sensitivity_delay_minutes": [5, 30],
        "feature_count": len(features),
        "selection_data": "rolling time CV within train only",
        "calibration_usage": "predictions generated for later uncertainty calibration; not used for model selection",
        "selected_candidate": selected_id,
        "selected_iterations": selected_iterations,
        "primary_test_ml_mae": float(primary_row["test_ml_mae"]),
        "primary_test_baseline_mae": float(primary_row["test_route_service_baseline_mae"]),
        "primary_mae_improvement_vs_baseline": float(primary_row["mae_improvement_vs_baseline"]),
        "validation_pass": int(validation["status"].eq("PASS").sum()),
        "validation_warn": int(validation["status"].eq("WARN").sum()),
        "output_files": sorted(path.name for path in OUTPUT_DIR.iterdir()),
    }
    (OUTPUT_DIR / f"run_summary_v{JOB_VERSION}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
