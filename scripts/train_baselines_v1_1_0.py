from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "artifacts" / "modeling_data" / "v1.1.0"
OUTPUT_DIR = ROOT / "artifacts" / "baseline_models" / "v1.1.0"
VERSION = "1.1.0"
DELAYS = [5, 15, 30]
PRIMARY_DELAY = 15
GROUP_LEVELS = [
    ("route_service", ["cab_type", "name", "source", "destination"]),
    ("service", ["cab_type", "name"]),
    ("provider", ["cab_type"]),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    y = y_true.to_numpy(dtype=float)
    p = y_pred.to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    error = y - p
    absolute = np.abs(error)
    sse = float(np.sum(error**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(len(y)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(absolute)),
        "wape": float(np.sum(absolute) / np.sum(np.abs(y))),
        "r2": float(1 - sse / sst) if sst > 0 else math.nan,
    }


def surge_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    actual = y_true.to_numpy(dtype=float) > 1.0
    predicted = y_pred.to_numpy(dtype=float) > 1.0
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "actual_surge_rate": float(np.mean(actual)),
        "predicted_surge_rate": float(np.mean(predicted)),
        "surge_precision": float(precision),
        "surge_recall": float(recall),
        "surge_f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "surge_balanced_accuracy": float((recall + specificity) / 2),
    }


def fit_lookups(train: pd.DataFrame, target: str) -> tuple[dict[str, pd.Series], float]:
    lookups = {
        level: train.groupby(keys, dropna=False, observed=True)[target].median()
        for level, keys in GROUP_LEVELS
    }
    return lookups, float(train[target].median())


def hierarchical_predict(
    frame: pd.DataFrame, lookups: dict[str, pd.Series], global_median: float
) -> tuple[pd.Series, pd.Series]:
    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    fallback = pd.Series("", index=frame.index, dtype="object")
    for level, keys in GROUP_LEVELS:
        unresolved = prediction.isna()
        if not unresolved.any():
            break
        index = pd.MultiIndex.from_frame(frame.loc[unresolved, keys])
        mapped = pd.Series(index.map(lookups[level]), index=frame.index[unresolved], dtype=float)
        available = mapped.notna()
        prediction.loc[mapped.index[available]] = mapped.loc[available]
        fallback.loc[mapped.index[available]] = level
    unresolved = prediction.isna()
    prediction.loc[unresolved] = global_median
    fallback.loc[unresolved] = "global"
    return prediction, fallback


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for delay in DELAYS:
        price = pd.read_parquet(INPUT_DIR / f"price_modeling_delay_{delay:02d}m_v1.1.0.parquet")
        multiplier = pd.read_parquet(
            INPUT_DIR / f"lyft_multiplier_modeling_delay_{delay:02d}m_v1.1.0.parquet"
        )
        price_train = price.loc[price["data_split"].eq("train")]
        multiplier_train = multiplier.loc[multiplier["data_split"].eq("train")]
        price_lookups, price_global = fit_lookups(price_train, "target_price_median")
        multiplier_lookups, multiplier_global = fit_lookups(
            multiplier_train, "target_multiplier_median"
        )

        price_eval = price.loc[price["data_split"].isin(["calibration", "test"])].copy()
        price_eval["pred_global_median"] = price_global
        price_eval["pred_persistence_asof"] = price_eval["lag1_price_median"]
        (
            price_eval["pred_route_service_median"],
            price_eval["route_service_fallback_level"],
        ) = hierarchical_predict(price_eval, price_lookups, price_global)

        multiplier_eval = multiplier.loc[
            multiplier["data_split"].isin(["calibration", "test"])
        ].copy()
        multiplier_eval["pred_constant_one"] = 1.0
        multiplier_eval["pred_global_median"] = multiplier_global
        multiplier_eval["pred_persistence_asof"] = multiplier_eval["lag1_multiplier_median"]
        (
            multiplier_eval["pred_route_service_median"],
            multiplier_eval["route_service_fallback_level"],
        ) = hierarchical_predict(multiplier_eval, multiplier_lookups, multiplier_global)

        price_models = {
            "global_median": "pred_global_median",
            "persistence_asof": "pred_persistence_asof",
            "route_service_median": "pred_route_service_median",
        }
        multiplier_models = {
            "constant_one": "pred_constant_one",
            "global_median": "pred_global_median",
            "persistence_asof": "pred_persistence_asof",
            "route_service_median": "pred_route_service_median",
        }
        for split in ["calibration", "test"]:
            part = price_eval.loc[price_eval["data_split"].eq(split)]
            for model, column in price_models.items():
                metric_rows.append(
                    {
                        "task": "price",
                        "delay_minutes": delay,
                        "data_split": split,
                        "model": model,
                        **regression_metrics(part["target_price_median"], part[column]),
                    }
                )
            diagnostic_rows.append(
                {
                    "task": "price",
                    "delay_minutes": delay,
                    "data_split": split,
                    "route_service_exact_match_rate": float(
                        part["route_service_fallback_level"].eq("route_service").mean()
                    ),
                }
            )
            part = multiplier_eval.loc[multiplier_eval["data_split"].eq(split)]
            for model, column in multiplier_models.items():
                metric_rows.append(
                    {
                        "task": "multiplier",
                        "delay_minutes": delay,
                        "data_split": split,
                        "model": model,
                        **regression_metrics(part["target_multiplier_median"], part[column]),
                        **surge_metrics(part["target_multiplier_median"], part[column]),
                    }
                )
            diagnostic_rows.append(
                {
                    "task": "multiplier",
                    "delay_minutes": delay,
                    "data_split": split,
                    "route_service_exact_match_rate": float(
                        part["route_service_fallback_level"].eq("route_service").mean()
                    ),
                }
            )

        for task, frame, target in [
            ("price", price, "target_price_median"),
            ("lyft_multiplier", multiplier, "target_multiplier_median"),
        ]:
            for split in ["train", "calibration", "test"]:
                part = frame.loc[frame["data_split"].eq(split)]
                split_rows.append(
                    {
                        "task": task,
                        "delay_minutes": delay,
                        "data_split": split,
                        "rows": len(part),
                        "start_time_utc": part["prediction_time_utc"].min().isoformat(),
                        "end_time_utc": part["prediction_time_utc"].max().isoformat(),
                        "series": int(part["series_id"].nunique()),
                        "target_mean": float(part[target].mean()),
                    }
                )

        price_eval[
            [
                "snapshot_id",
                "series_id",
                "prediction_time_utc",
                "data_split",
                "target_price_median",
                "pred_global_median",
                "pred_persistence_asof",
                "pred_route_service_median",
                "route_service_fallback_level",
            ]
        ].to_parquet(
            OUTPUT_DIR / f"price_baseline_predictions_delay_{delay:02d}m_v1.1.0.parquet",
            index=False,
        )
        multiplier_eval[
            [
                "snapshot_id",
                "series_id",
                "prediction_time_utc",
                "data_split",
                "target_multiplier_median",
                "pred_constant_one",
                "pred_global_median",
                "pred_persistence_asof",
                "pred_route_service_median",
                "route_service_fallback_level",
            ]
        ].to_parquet(
            OUTPUT_DIR
            / f"multiplier_baseline_predictions_delay_{delay:02d}m_v1.1.0.parquet",
            index=False,
        )

        validation_rows.extend(
            [
                {"check_id": f"D{delay:02d}_PRICE_PREDICTIONS", "status": "PASS" if price_eval[[*price_models.values()]].notna().all().all() else "FAIL", "evidence": "price predictions complete"},
                {"check_id": f"D{delay:02d}_MULTIPLIER_PREDICTIONS", "status": "PASS" if multiplier_eval[[*multiplier_models.values()]].notna().all().all() else "FAIL", "evidence": "multiplier predictions complete"},
                {"check_id": f"D{delay:02d}_TRAIN_ONLY_LOOKUPS", "status": "PASS", "evidence": "all group medians fitted from data_split=train"},
            ]
        )

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    splits = pd.DataFrame(split_rows)
    validation = pd.DataFrame(validation_rows)
    if not validation["status"].eq("PASS").all() or not np.isfinite(metrics["mae"]).all():
        raise RuntimeError("Baseline validation failed")

    metrics.to_csv(OUTPUT_DIR / f"baseline_metrics_v{VERSION}.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / f"baseline_diagnostics_v{VERSION}.csv", index=False)
    splits.to_csv(OUTPUT_DIR / f"split_audit_v{VERSION}.csv", index=False)
    validation.to_csv(OUTPUT_DIR / f"validation_results_v{VERSION}.csv", index=False)
    config = {
        "artifact_version": VERSION,
        "input_modeling_data_version": "1.1.0",
        "primary_delay_minutes": PRIMARY_DELAY,
        "sensitivity_delay_minutes": [5, 30],
        "fit_split": "train",
        "reported_splits": ["calibration", "test"],
        "price_baselines": ["global_median", "persistence_asof", "route_service_median"],
        "multiplier_baselines": [
            "constant_one",
            "global_median",
            "persistence_asof",
            "route_service_median",
        ],
        "ml_validation_policy": "rolling or blocked time-series CV inside train",
        "test_policy": "POC benchmark only because test was previously inspected",
    }
    (OUTPUT_DIR / f"run_config_v{VERSION}.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    primary_test = metrics.loc[
        metrics["delay_minutes"].eq(PRIMARY_DELAY) & metrics["data_split"].eq("test")
    ]
    best_price = primary_test.loc[primary_test["task"].eq("price")].sort_values("mae").iloc[0]
    best_multiplier = (
        primary_test.loc[primary_test["task"].eq("multiplier")].sort_values("mae").iloc[0]
    )
    summary = {
        "artifact_version": VERSION,
        "primary_delay_minutes": PRIMARY_DELAY,
        "primary_test_best_price_baseline": {
            "model": best_price["model"],
            "mae": float(best_price["mae"]),
            "rmse": float(best_price["rmse"]),
        },
        "primary_test_best_multiplier_baseline": {
            "model": best_multiplier["model"],
            "mae": float(best_multiplier["mae"]),
            "rmse": float(best_multiplier["rmse"]),
            "surge_recall": float(best_multiplier["surge_recall"]),
            "surge_f1": float(best_multiplier["surge_f1"]),
        },
        "metric_rows": len(metrics),
        "validation_checks": len(validation),
        "validation_passed": int(validation["status"].eq("PASS").sum()),
    }
    (OUTPUT_DIR / f"baseline_summary_v{VERSION}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    files = sorted(
        path for path in OUTPUT_DIR.iterdir() if path.is_file() and not path.name.startswith("manifest_")
    )
    manifest = {
        "artifact_name": "baseline_models",
        "artifact_version": VERSION,
        "files": [
            {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    (OUTPUT_DIR / f"manifest_v{VERSION}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(
        metrics.loc[
            metrics["data_split"].eq("test"),
            ["task", "delay_minutes", "model", "mae", "rmse", "surge_recall", "surge_f1"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
