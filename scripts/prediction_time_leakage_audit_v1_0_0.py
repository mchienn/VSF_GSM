from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "modeling_data" / "v1.0.0"
OUTPUT_DIR = ROOT / "artifacts" / "leakage_audit" / "v1.0.0"
VERSION = "1.0.0"
SERIES_KEYS = ["cab_type", "name", "source", "destination"]
DELAY_SCENARIOS_MINUTES = [5, 15, 30, 60]

CURRENT_WEATHER = {
    "latitude",
    "longitude",
    "temperature",
    "apparentTemperature",
    "precipIntensity",
    "precipProbability",
    "humidity",
    "windSpeed",
    "windGust",
    "visibility",
    "dewPoint",
    "pressure",
    "windBearing",
    "cloudCover",
    "uvIndex",
    "ozone",
}
TIME_FEATURES = {
    "event_hour_local",
    "event_weekday_local",
    "event_month_local",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
}
STRUCTURAL_FEATURES = {"cab_type", "name", "source", "destination"}
DELAY_DERIVED_FEATURES = {
    "observation_age_bucket",
    "observation_age_minutes",
    "lag1_price_median",
    "lag2_price_median",
    "lag3_price_median",
    "lag1_multiplier_median",
    "lag1_quote_count",
    "lag1_price_spread",
    "lag1_distance_median",
    "lag_price_delta_1_2",
    "history_price_mean_last3",
    "history_price_std_last3",
    "history_price_mean_last6",
    "history_observation_count",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mismatch_count(actual: pd.Series, expected: pd.Series, tolerance: float = 1e-8) -> int:
    left = actual.to_numpy(dtype=float)
    right = expected.to_numpy(dtype=float)
    return int(np.sum(~np.isclose(left, right, rtol=tolerance, atol=tolerance, equal_nan=True)))


def feature_availability(contract: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    categorical = set(contract["categorical_features"])
    all_features = contract["categorical_features"] + contract["numeric_features"]
    for feature in all_features:
        feature_type = "categorical" if feature in categorical else "numeric"
        if feature in STRUCTURAL_FEATURES or feature in TIME_FEATURES:
            pipeline_source = "request context or deterministic prediction timestamp"
            status = "SAFE_AS_IS"
            action = "none"
            allowed = True
        elif feature in DELAY_DERIVED_FEATURES:
            pipeline_source = "strictly previous snapshots via shift/rolling"
            status = "SAFE_AFTER_DELAY_PROTOCOL"
            action = "select observations using an as-of cutoff t-delay"
            allowed = False
        elif feature == "distance_median":
            pipeline_source = "median of quote rows inside the target 5-minute bucket"
            status = "REBUILD_FROM_REQUEST_TIME_SOURCE"
            action = "use request/route distance known at prediction time or lagged/train-only route distance"
            allowed = False
        elif feature in CURRENT_WEATHER or feature == "short_summary":
            pipeline_source = "aggregation of weather fields inside the target 5-minute bucket"
            status = "REBUILD_ASOF_PREDICTION_TIME"
            action = "join weather issued at or before prediction time; do not aggregate future rows in target bucket"
            allowed = False
        else:
            pipeline_source = "unclassified"
            status = "REVIEW_REQUIRED"
            action = "define source timestamp and availability"
            allowed = False
        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "pipeline_source": pipeline_source,
                "audit_status": status,
                "required_action": action,
                "allowed_for_strict_ml_v1_0": allowed,
            }
        )
    return pd.DataFrame(rows)


def delay_coverage(task: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ["train", "calibration", "test"]:
        part = frame.loc[frame["data_split"].eq(split)]
        for delay in DELAY_SCENARIOS_MINUTES:
            eligible = part["observation_age_minutes"].ge(delay)
            rows.append(
                {
                    "task": task,
                    "data_split": split,
                    "minimum_delay_minutes": delay,
                    "rows": int(len(part)),
                    "lag1_rows_meeting_delay": int(eligible.sum()),
                    "lag1_rows_violating_delay": int((~eligible).sum()),
                    "lag1_coverage_pct": float(100 * eligible.mean()),
                    "implementation_note": "coverage only; correct implementation requires an as-of lookup at t-delay, not row deletion",
                }
            )
    return rows


def dependency_profile(task: str, frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(SERIES_KEYS + ["snapshot_time_utc"], kind="mergesort").copy()
    ordered["previous_split"] = ordered.groupby(SERIES_KEYS, observed=True, sort=False)[
        "data_split"
    ].shift(1)
    profile = (
        ordered.loc[ordered["previous_split"].notna()]
        .groupby(["data_split", "previous_split"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    profile.insert(0, "task", task)
    profile["dependency_interpretation"] = np.where(
        profile["data_split"].eq(profile["previous_split"]),
        "within-split past observation",
        "cross-split past observation; valid only for online rolling evaluation",
    )
    return profile


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = json.loads((MODEL_DIR / "feature_contract_v1.0.0.json").read_text(encoding="utf-8"))
    snapshots = pd.read_parquet(MODEL_DIR / "price_snapshots_5min_v1.0.0.parquet")
    price = pd.read_parquet(MODEL_DIR / "price_modeling_v1.0.0.parquet")
    multiplier = pd.read_parquet(MODEL_DIR / "lyft_multiplier_modeling_v1.0.0.parquet")

    ordered = snapshots.sort_values(SERIES_KEYS + ["snapshot_time_utc"], kind="mergesort").copy()
    grouped = ordered.groupby(SERIES_KEYS, observed=True, sort=False)
    expected_previous_time = grouped["snapshot_time_utc"].shift(1)
    expected_age = (ordered["snapshot_time_utc"] - expected_previous_time).dt.total_seconds() / 60
    lag_expectations = {
        "lag1_price_median": grouped["target_price_median"].shift(1),
        "lag2_price_median": grouped["target_price_median"].shift(2),
        "lag3_price_median": grouped["target_price_median"].shift(3),
        "lag1_multiplier_median": grouped["target_multiplier_median"].shift(1),
        "lag1_quote_count": grouped["target_quote_count"].shift(1),
        "lag1_price_spread": grouped["target_price_spread"].shift(1),
        "lag1_distance_median": grouped["distance_median"].shift(1),
    }
    shifted_price = lag_expectations["lag1_price_median"]
    shifted_groups = shifted_price.groupby(
        [ordered[column] for column in SERIES_KEYS], observed=True, sort=False
    )
    rolling_expectations = {
        "history_price_mean_last3": shifted_groups.transform(
            lambda values: values.rolling(3, min_periods=1).mean()
        ),
        "history_price_std_last3": shifted_groups.transform(
            lambda values: values.rolling(3, min_periods=2).std(ddof=0)
        ),
        "history_price_mean_last6": shifted_groups.transform(
            lambda values: values.rolling(6, min_periods=1).mean()
        ),
    }

    lag_mismatches = {
        column: mismatch_count(ordered[column], expected)
        for column, expected in lag_expectations.items()
    }
    rolling_mismatches = {
        column: mismatch_count(ordered[column], expected)
        for column, expected in rolling_expectations.items()
    }
    age_mismatches = mismatch_count(ordered["observation_age_minutes"], expected_age)
    previous_time_mismatches = int(
        (~(ordered["previous_observation_time_utc"].eq(expected_previous_time) | (
            ordered["previous_observation_time_utc"].isna() & expected_previous_time.isna()
        ))).sum()
    )
    delta_mismatches = mismatch_count(
        ordered["lag_price_delta_1_2"],
        lag_expectations["lag1_price_median"] - lag_expectations["lag2_price_median"],
    )
    count_mismatches = int(
        (ordered["history_observation_count"].astype(int) != grouped.cumcount()).sum()
    )

    split_bounds = price.groupby("data_split", observed=True)["snapshot_time_utc"].agg(["min", "max"])
    chronological = bool(
        split_bounds.loc["train", "max"] < split_bounds.loc["calibration", "min"]
        and split_bounds.loc["calibration", "max"] < split_bounds.loc["test", "min"]
    )
    split_rank = {"train": 0, "calibration": 1, "test": 2}
    ordered["previous_split"] = grouped["data_split"].shift(1)
    dependency_rows = ordered.loc[ordered["previous_split"].notna(), ["data_split", "previous_split"]]
    future_dependency_count = int(
        sum(split_rank[previous] > split_rank[current] for current, previous in dependency_rows.itertuples(index=False))
    )

    all_features = contract["categorical_features"] + contract["numeric_features"]
    forbidden_overlap = sorted(set(all_features) & set(contract["forbidden_contemporaneous_features"]))
    target_overlap = sorted(set(all_features) & {contract["price_target"], contract["multiplier_target"]})
    availability = feature_availability(contract)
    blocked_features = availability.loc[
        ~availability["allowed_for_strict_ml_v1_0"], "feature"
    ].tolist()

    delay_frame = pd.DataFrame(
        delay_coverage("price", price) + delay_coverage("lyft_multiplier", multiplier)
    )
    dependency_frame = pd.concat(
        [dependency_profile("price", price), dependency_profile("lyft_multiplier", multiplier)],
        ignore_index=True,
    )

    test_previously_inspected = bool(
        (ROOT / "artifacts" / "eda" / "v1.0.0" / "baseline_diagnostics_v1.0.0.csv").exists()
        and (ROOT / "artifacts" / "baseline_models" / "v1.0.0" / "baseline_metrics_v1.0.0.csv").exists()
    )
    checks = [
        ("L01", "PASS" if len(all_features) == 44 else "FAIL", len(all_features), 44, "feature contract size"),
        ("L02", "PASS" if chronological else "FAIL", chronological, True, "strict chronological split order"),
        ("L03", "PASS" if not forbidden_overlap else "FAIL", forbidden_overlap, [], "forbidden current diagnostics excluded"),
        ("L04", "PASS" if not target_overlap else "FAIL", target_overlap, [], "targets excluded from predictors"),
        ("L05", "PASS" if previous_time_mismatches == 0 else "FAIL", previous_time_mismatches, 0, "previous timestamp is causal group shift"),
        ("L06", "PASS" if sum(lag_mismatches.values()) == 0 else "FAIL", lag_mismatches, {}, "lag values match prior targets/diagnostics"),
        ("L07", "PASS" if sum(rolling_mismatches.values()) == 0 else "FAIL", rolling_mismatches, {}, "rolling histories use shifted price"),
        ("L08", "PASS" if age_mismatches == 0 else "FAIL", age_mismatches, 0, "observation age matches prior timestamp"),
        ("L09", "PASS" if delta_mismatches == 0 and count_mismatches == 0 else "FAIL", {"delta": delta_mismatches, "history_count": count_mismatches}, {"delta": 0, "history_count": 0}, "derived history checks"),
        ("L10", "PASS" if future_dependency_count == 0 else "FAIL", future_dependency_count, 0, "no dependency on a later split"),
        ("L11", "BLOCKER" if blocked_features else "PASS", blocked_features, [], "prediction-time feature provenance"),
        ("L12", "BLOCKER", "no minimum competitor observation delay is selected", "explicit delay plus as-of feature builder", "delay protocol"),
        ("L13", "WARNING" if test_previously_inspected else "PASS", test_previously_inspected, False, "test blindness"),
        ("L14", "WARNING", "no standalone validation split", "rolling/blocked CV inside train", "ML model-selection protocol"),
    ]
    check_frame = pd.DataFrame(
        [
            {
                "check_id": check_id,
                "status": status,
                "actual": json.dumps(actual, default=str),
                "expected": json.dumps(expected, default=str),
                "evidence": evidence,
            }
            for check_id, status, actual, expected, evidence in checks
        ]
    )

    overall_status = "BLOCKED_PENDING_PREDICTION_TIME_FIXES" if (check_frame["status"] == "BLOCKER").any() else "READY_FOR_ML"
    summary = {
        "artifact_version": VERSION,
        "input_modeling_data_version": "1.0.0",
        "overall_status": overall_status,
        "feature_contract_columns": len(all_features),
        "causal_history_checks": "PASS" if sum(lag_mismatches.values()) + sum(rolling_mismatches.values()) + age_mismatches + previous_time_mismatches + delta_mismatches + count_mismatches == 0 else "FAIL",
        "chronological_split_check": "PASS" if chronological else "FAIL",
        "future_split_dependencies": future_dependency_count,
        "prediction_time_blocked_feature_count": len(blocked_features),
        "prediction_time_blocked_features": blocked_features,
        "minimum_delay_selected": None,
        "evaluation_test_blindness": "COMPROMISED" if test_previously_inspected else "PRESERVED",
        "approved_evaluation_mode_after_fix": "online rolling one-step with observations selected at or before t-delay",
        "required_actions_before_price_ml": [
            "Select a minimum competitor observation delay scenario",
            "Rebuild lag/rolling features with an as-of cutoff at t-delay",
            "Rebuild distance from request-time information or use a lagged/train-only route proxy",
            "Join weather using only observations issued at or before prediction time",
            "Use rolling/blocked time-series CV inside train and keep calibration for uncertainty calibration",
            "Treat the existing test as a POC benchmark because it has already been inspected",
        ],
    }

    availability.to_csv(OUTPUT_DIR / f"feature_availability_matrix_v{VERSION}.csv", index=False)
    delay_frame.to_csv(OUTPUT_DIR / f"delay_scenario_coverage_v{VERSION}.csv", index=False)
    dependency_frame.to_csv(OUTPUT_DIR / f"split_dependency_profile_v{VERSION}.csv", index=False)
    check_frame.to_csv(OUTPUT_DIR / f"check_results_v{VERSION}.csv", index=False)
    (OUTPUT_DIR / f"audit_summary_v{VERSION}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    output_files = sorted(
        path for path in OUTPUT_DIR.iterdir() if path.is_file() and not path.name.startswith("manifest_")
    )
    manifest = {
        "artifact_name": "prediction_time_leakage_audit",
        "artifact_version": VERSION,
        "files": [
            {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
    }
    (OUTPUT_DIR / f"manifest_v{VERSION}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "overall_status": overall_status,
        "checks": check_frame["status"].value_counts().to_dict(),
        "blocked_feature_count": len(blocked_features),
        "causal_history_mismatches": sum(lag_mismatches.values()) + sum(rolling_mismatches.values()) + age_mismatches + previous_time_mismatches + delta_mismatches + count_mismatches,
        "future_split_dependencies": future_dependency_count,
        "output_dir": str(OUTPUT_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
