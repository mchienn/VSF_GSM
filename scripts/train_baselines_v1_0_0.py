from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "artifacts" / "modeling_data" / "v1.0.0"
OUTPUT_DIR = ROOT / "artifacts" / "baseline_models" / "v1.0.0"
VERSION = "1.0.0"

GROUP_LEVELS = [
    ("route_service", ["cab_type", "name", "source", "destination"]),
    ("service", ["cab_type", "name"]),
    ("provider", ["cab_type"]),
]


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    y = y_true.to_numpy(dtype=float)
    p = y_pred.to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    error = y - p
    abs_error = np.abs(error)
    sse = float(np.sum(error**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(len(y)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(abs_error)),
        "wape": float(np.sum(abs_error) / np.sum(np.abs(y))),
        "r2": float(1.0 - sse / sst) if sst > 0 else math.nan,
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
        "surge_f1": float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0,
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
        key_index = pd.MultiIndex.from_frame(frame.loc[unresolved, keys])
        mapped = pd.Series(key_index.map(lookups[level]), index=frame.index[unresolved], dtype=float)
        available = mapped.notna()
        prediction.loc[mapped.index[available]] = mapped.loc[available]
        fallback.loc[mapped.index[available]] = level
    unresolved = prediction.isna()
    prediction.loc[unresolved] = global_median
    fallback.loc[unresolved] = "global"
    return prediction, fallback


def split_audit(dataset_name: str, frame: pd.DataFrame, target: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ["train", "calibration", "test"]:
        part = frame.loc[frame["data_split"] == split]
        rows.append(
            {
                "dataset": dataset_name,
                "data_split": split,
                "rows": int(len(part)),
                "start_time_utc": part["snapshot_time_utc"].min().isoformat(),
                "end_time_utc": part["snapshot_time_utc"].max().isoformat(),
                "target_mean": float(part[target].mean()),
                "target_median": float(part[target].median()),
                "series": int(part["series_id"].nunique()),
            }
        )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    price = pd.read_parquet(INPUT_DIR / "price_modeling_v1.0.0.parquet")
    multiplier = pd.read_parquet(INPUT_DIR / "lyft_multiplier_modeling_v1.0.0.parquet")

    price_train = price.loc[price["data_split"] == "train"]
    multiplier_train = multiplier.loc[multiplier["data_split"] == "train"]
    price_lookups, price_global = fit_lookups(price_train, "target_price_median")
    multiplier_lookups, multiplier_global = fit_lookups(
        multiplier_train, "target_multiplier_median"
    )

    for level, series in price_lookups.items():
        series.rename("prediction").reset_index().to_parquet(
            OUTPUT_DIR / f"price_{level}_lookup_v{VERSION}.parquet", index=False
        )
    for level, series in multiplier_lookups.items():
        series.rename("prediction").reset_index().to_parquet(
            OUTPUT_DIR / f"multiplier_{level}_lookup_v{VERSION}.parquet", index=False
        )

    price_eval = price.loc[price["data_split"].isin(["calibration", "test"])].copy()
    price_eval["pred_global_median"] = price_global
    price_eval["pred_persistence_lag1"] = price_eval["lag1_price_median"]
    (
        price_eval["pred_route_service_median"],
        price_eval["route_service_fallback_level"],
    ) = hierarchical_predict(price_eval, price_lookups, price_global)

    multiplier_eval = multiplier.loc[
        multiplier["data_split"].isin(["calibration", "test"])
    ].copy()
    multiplier_eval["pred_constant_one"] = 1.0
    multiplier_eval["pred_global_median"] = multiplier_global
    multiplier_eval["pred_persistence_lag1"] = multiplier_eval["lag1_multiplier_median"]
    (
        multiplier_eval["pred_route_service_median"],
        multiplier_eval["route_service_fallback_level"],
    ) = hierarchical_predict(multiplier_eval, multiplier_lookups, multiplier_global)

    price_prediction_columns = [
        "snapshot_id",
        "series_id",
        "snapshot_time_utc",
        "data_split",
        "target_price_median",
        "pred_global_median",
        "pred_persistence_lag1",
        "pred_route_service_median",
        "route_service_fallback_level",
    ]
    multiplier_prediction_columns = [
        "snapshot_id",
        "series_id",
        "snapshot_time_utc",
        "data_split",
        "target_multiplier_median",
        "pred_constant_one",
        "pred_global_median",
        "pred_persistence_lag1",
        "pred_route_service_median",
        "route_service_fallback_level",
    ]
    price_eval[price_prediction_columns].to_parquet(
        OUTPUT_DIR / f"price_baseline_predictions_v{VERSION}.parquet", index=False
    )
    multiplier_eval[multiplier_prediction_columns].to_parquet(
        OUTPUT_DIR / f"multiplier_baseline_predictions_v{VERSION}.parquet", index=False
    )

    metrics: list[dict[str, object]] = []
    price_models = {
        "global_median": "pred_global_median",
        "persistence_lag1": "pred_persistence_lag1",
        "route_service_median": "pred_route_service_median",
    }
    multiplier_models = {
        "constant_one": "pred_constant_one",
        "global_median": "pred_global_median",
        "persistence_lag1": "pred_persistence_lag1",
        "route_service_median": "pred_route_service_median",
    }
    for split in ["calibration", "test"]:
        part = price_eval.loc[price_eval["data_split"] == split]
        for model, column in price_models.items():
            metrics.append(
                {
                    "task": "price",
                    "data_split": split,
                    "model": model,
                    **regression_metrics(part["target_price_median"], part[column]),
                }
            )
        part = multiplier_eval.loc[multiplier_eval["data_split"] == split]
        for model, column in multiplier_models.items():
            metrics.append(
                {
                    "task": "multiplier",
                    "data_split": split,
                    "model": model,
                    **regression_metrics(part["target_multiplier_median"], part[column]),
                    **surge_metrics(part["target_multiplier_median"], part[column]),
                }
            )
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(OUTPUT_DIR / f"baseline_metrics_v{VERSION}.csv", index=False)

    audit = pd.DataFrame(
        split_audit("price", price, "target_price_median")
        + split_audit("lyft_multiplier", multiplier, "target_multiplier_median")
    )
    audit.to_csv(OUTPUT_DIR / f"split_audit_v{VERSION}.csv", index=False)

    diagnostics = pd.DataFrame(
        [
            {
                "task": "price",
                "split": split,
                "route_service_exact_match_rate": float(
                    np.mean(
                        price_eval.loc[price_eval["data_split"] == split, "route_service_fallback_level"]
                        == "route_service"
                    )
                ),
            }
            for split in ["calibration", "test"]
        ]
        + [
            {
                "task": "multiplier",
                "split": split,
                "route_service_exact_match_rate": float(
                    np.mean(
                        multiplier_eval.loc[
                            multiplier_eval["data_split"] == split,
                            "route_service_fallback_level",
                        ]
                        == "route_service"
                    )
                ),
            }
            for split in ["calibration", "test"]
        ]
    )
    diagnostics.to_csv(OUTPUT_DIR / f"baseline_diagnostics_v{VERSION}.csv", index=False)

    run_config = {
        "artifact_version": VERSION,
        "input_modeling_data_version": "1.0.0",
        "fit_split": "train",
        "reported_splits": ["calibration", "test"],
        "standalone_validation_split": False,
        "future_ml_validation_policy": "rolling or blocked time-series CV inside train",
        "calibration_policy": "reserve for uncertainty calibration; do not tune ML hyperparameters on it",
        "test_policy": "fixed chronological holdout; baseline benchmark recorded before ML",
        "price_baselines": list(price_models),
        "multiplier_baselines": list(multiplier_models),
        "group_median_fallback": [level for level, _ in GROUP_LEVELS] + ["global"],
    }
    (OUTPUT_DIR / f"run_config_v{VERSION}.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    price_test = metrics_frame.loc[
        (metrics_frame["task"] == "price") & (metrics_frame["data_split"] == "test")
    ].sort_values("mae")
    multiplier_test = metrics_frame.loc[
        (metrics_frame["task"] == "multiplier")
        & (metrics_frame["data_split"] == "test")
    ].sort_values("mae")
    best_price = price_test.iloc[0]
    best_multiplier = multiplier_test.iloc[0]
    report = f"""# Baseline Modeling Report v{VERSION}

## Kết quả

- **Price:** baseline tốt nhất trên test là `{best_price['model']}`, MAE **{best_price['mae']:.4f} USD**, RMSE **{best_price['rmse']:.4f} USD**.
- **Lyft multiplier:** baseline tốt nhất theo MAE trên test là `{best_multiplier['model']}`, MAE **{best_multiplier['mae']:.4f}**. Constant/median có thể đạt MAE tốt do target chủ yếu bằng 1 nhưng không bắt được surge; cần đọc thêm recall/F1 trong metrics.
- Các group lookup chỉ fit từ `train`; không sử dụng target của calibration/test.

## Split đang dùng

Modeling data v1.0.0 đã chia tuần tự theo UTC:

- Train: 2018-11-26 đến 2018-12-10.
- Calibration: 2018-12-13 đến 2018-12-15.
- Test: 2018-12-16 đến 2018-12-18.
- Không có validation riêng. ML sẽ dùng rolling/blocked time-series CV bên trong train. Calibration được giữ cho uncertainty calibration.

Các ngày trống 2018-12-05..08 và 2018-12-11..12 không được nội suy. Ranh giới ngày local có thể lệch một phần so với UTC; `split_audit_v1.0.0.csv` ghi timestamp UTC thực tế.

## Input và output

- Input price: `artifacts/modeling_data/v1.0.0/price_modeling_v1.0.0.parquet`
- Input multiplier: `artifacts/modeling_data/v1.0.0/lyft_multiplier_modeling_v1.0.0.parquet`
- Metrics đầy đủ: `baseline_metrics_v1.0.0.csv`
- Row-level predictions: `price_baseline_predictions_v1.0.0.parquet`, `multiplier_baseline_predictions_v1.0.0.parquet`

## Lưu ý đánh giá

Test đã từng được dùng cho diagnostic trong EDA v1.0.0, nên đây là POC benchmark chứ chưa phải estimate production hoàn toàn độc lập. Từ bước ML trở đi cần khóa test và chỉ mở lại sau khi model/feature set được chốt bằng train CV.
"""
    (OUTPUT_DIR / f"BASELINE_MODELING_REPORT_v{VERSION}.md").write_text(report, encoding="utf-8")

    checks = [
        ("B01", len(price_train) == 356054, "price train row count"),
        ("B02", len(multiplier_train) == 172583, "multiplier train row count"),
        ("B03", price_eval["pred_persistence_lag1"].notna().all(), "price lag1 complete"),
        (
            "B04",
            multiplier_eval["pred_persistence_lag1"].notna().all(),
            "multiplier lag1 complete",
        ),
        (
            "B05",
            price_eval["pred_route_service_median"].notna().all(),
            "price hierarchical prediction complete",
        ),
        (
            "B06",
            multiplier_eval["pred_route_service_median"].notna().all(),
            "multiplier hierarchical prediction complete",
        ),
        ("B07", np.isfinite(metrics_frame["mae"]).all(), "all MAE values finite"),
        (
            "B08",
            set(price_eval["data_split"]) == {"calibration", "test"},
            "price evaluation excludes train",
        ),
    ]
    validation = pd.DataFrame(
        [
            {"check_id": check_id, "status": "PASS" if passed else "FAIL", "evidence": evidence}
            for check_id, passed, evidence in checks
        ]
    )
    validation.to_csv(OUTPUT_DIR / f"validation_results_v{VERSION}.csv", index=False)
    if not all(passed for _, passed, _ in checks):
        raise RuntimeError("Baseline validation failed")

    artifact_files = sorted(
        path for path in OUTPUT_DIR.iterdir() if path.is_file() and not path.name.startswith("manifest_")
    )
    manifest = {
        "artifact_name": "baseline_models",
        "artifact_version": VERSION,
        "input_modeling_data_version": "1.0.0",
        "files": [
            {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifact_files
        ],
    }
    (OUTPUT_DIR / f"manifest_v{VERSION}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(price_test[["model", "n", "mae", "rmse", "r2"]].to_string(index=False))
    print(multiplier_test[["model", "n", "mae", "rmse", "surge_recall", "surge_f1"]].to_string(index=False))
    print(f"validation_passed={sum(passed for _, passed, _ in checks)}/{len(checks)}")
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
