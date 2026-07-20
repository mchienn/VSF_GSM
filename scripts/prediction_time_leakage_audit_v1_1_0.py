from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_V10_DIR = ROOT / "artifacts" / "modeling_data" / "v1.0.0"
MODEL_DIR = ROOT / "artifacts" / "modeling_data" / "v1.1.0"
OUTPUT_DIR = ROOT / "artifacts" / "leakage_audit" / "v1.1.0"
VERSION = "1.1.0"
DELAYS = [5, 15, 30]
SERIES_KEYS = ["cab_type", "name", "source", "destination"]
WEATHER_NUMERIC = [
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
]


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


def expected_history(snapshots: pd.DataFrame) -> pd.DataFrame:
    frame = snapshots.sort_values(SERIES_KEYS + ["snapshot_time_utc"], kind="mergesort").copy()
    grouped = frame.groupby(SERIES_KEYS, observed=True, sort=False)
    frame["expected_lag1_price_median"] = frame["target_price_median"]
    frame["expected_lag2_price_median"] = grouped["target_price_median"].shift(1)
    frame["expected_lag3_price_median"] = grouped["target_price_median"].shift(2)
    frame["expected_lag1_multiplier_median"] = frame["target_multiplier_median"]
    frame["expected_lag1_quote_count"] = frame["target_quote_count"]
    frame["expected_lag1_price_spread"] = frame["target_price_spread"]
    frame["expected_lag1_distance_median"] = frame["distance_median"]
    frame["expected_lag_price_delta_1_2"] = (
        frame["expected_lag1_price_median"] - frame["expected_lag2_price_median"]
    )
    prices = frame["target_price_median"]
    price_groups = prices.groupby(
        [frame[column] for column in SERIES_KEYS], observed=True, sort=False
    )
    frame["expected_history_price_mean_last3"] = price_groups.transform(
        lambda values: values.rolling(3, min_periods=1).mean()
    )
    frame["expected_history_price_std_last3"] = price_groups.transform(
        lambda values: values.rolling(3, min_periods=2).std(ddof=0)
    )
    frame["expected_history_price_mean_last6"] = price_groups.transform(
        lambda values: values.rolling(6, min_periods=1).mean()
    )
    frame["expected_history_observation_count"] = grouped.cumcount() + 1
    return frame.rename(columns={"snapshot_time_utc": "history_observation_time_utc"})


def feature_availability(contract: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    categorical = set(contract["categorical_features"])
    history_features = {
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
    weather_features = set(WEATHER_NUMERIC) | {
        "short_summary",
        "weather_age_minutes",
        "weather_available_asof",
    }
    for feature in contract["categorical_features"] + contract["numeric_features"]:
        if feature in history_features:
            source = "latest series observation at or before prediction_time-delay"
            status = "SAFE_ASOF_DELAYED"
        elif feature in weather_features:
            source = "latest weather observation at or before prediction time; stale values masked"
            status = "SAFE_ASOF_WEATHER"
        elif feature == "route_geodesic_km":
            source = "frozen source/destination geocodes"
            status = "SAFE_STATIC_ROUTE"
        else:
            source = "request context or deterministic prediction timestamp"
            status = "SAFE_AS_IS"
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical" if feature in categorical else "numeric",
                "prediction_time_source": source,
                "audit_status": status,
                "allowed_for_price_ml": True,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = json.loads((MODEL_DIR / "feature_contract_v1.1.0.json").read_text(encoding="utf-8"))
    snapshots = pd.read_parquet(MODEL_V10_DIR / "price_snapshots_5min_v1.0.0.parquet")
    history = expected_history(snapshots)
    expected_columns = [
        "series_id",
        "history_observation_time_utc",
        *[column for column in history.columns if column.startswith("expected_")],
        "data_split",
    ]
    history_lookup = history[expected_columns].rename(columns={"data_split": "history_data_split"})

    all_features = contract["categorical_features"] + contract["numeric_features"]
    forbidden_overlap = sorted(set(all_features) & set(contract["forbidden_contemporaneous_features"]))
    target_overlap = sorted(set(all_features) & {contract["price_target"], contract["multiplier_target"]})
    check_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    split_rank = {"train": 0, "calibration": 1, "test": 2}

    def add_check(check_id: str, status: str, actual: object, expected: object, evidence: str) -> None:
        check_rows.append(
            {
                "check_id": check_id,
                "status": status,
                "actual": json.dumps(actual, default=str),
                "expected": json.dumps(expected, default=str),
                "evidence": evidence,
            }
        )

    add_check("A01", "PASS" if len(all_features) == 46 else "FAIL", len(all_features), 46, "feature contract size")
    add_check("A02", "PASS" if not forbidden_overlap else "FAIL", forbidden_overlap, [], "forbidden contemporaneous fields excluded")
    add_check("A03", "PASS" if not target_overlap else "FAIL", target_overlap, [], "targets excluded from predictors")

    history_feature_pairs = {
        "lag1_price_median": "expected_lag1_price_median",
        "lag2_price_median": "expected_lag2_price_median",
        "lag3_price_median": "expected_lag3_price_median",
        "lag1_multiplier_median": "expected_lag1_multiplier_median",
        "lag1_quote_count": "expected_lag1_quote_count",
        "lag1_price_spread": "expected_lag1_price_spread",
        "lag1_distance_median": "expected_lag1_distance_median",
        "lag_price_delta_1_2": "expected_lag_price_delta_1_2",
        "history_price_mean_last3": "expected_history_price_mean_last3",
        "history_price_std_last3": "expected_history_price_std_last3",
        "history_price_mean_last6": "expected_history_price_mean_last6",
        "history_observation_count": "expected_history_observation_count",
    }

    for delay in DELAYS:
        price = pd.read_parquet(MODEL_DIR / f"price_modeling_delay_{delay:02d}m_v1.1.0.parquet")
        multiplier = pd.read_parquet(
            MODEL_DIR / f"lyft_multiplier_modeling_delay_{delay:02d}m_v1.1.0.parquet"
        )
        joined = price.merge(
            history_lookup,
            on=["series_id", "history_observation_time_utc"],
            how="left",
            validate="many_to_one",
        )
        cutoff_expected = joined["prediction_time_utc"] - pd.to_timedelta(delay, unit="m")
        cutoff_mismatches = int((joined["history_cutoff_utc"] != cutoff_expected).sum())
        future_history = int(
            (joined["history_observation_time_utc"] > joined["history_cutoff_utc"]).sum()
        )
        age_expected = (
            joined["prediction_time_utc"] - joined["history_observation_time_utc"]
        ).dt.total_seconds().div(60)
        age_mismatches = mismatch_count(joined["observation_age_minutes"], age_expected)
        history_mismatches = {
            feature: mismatch_count(joined[feature], joined[expected])
            for feature, expected in history_feature_pairs.items()
        }
        weather_future = int(
            (joined["weather_observation_time_utc"] > joined["prediction_time_utc"]).sum()
        )
        stale_with_values = int(
            joined.loc[~joined["weather_available_asof"], WEATHER_NUMERIC].notna().any(axis=1).sum()
        )
        available_too_old = int(
            joined.loc[joined["weather_available_asof"], "weather_age_minutes"]
            .gt(contract["maximum_weather_age_minutes"])
            .sum()
        )
        future_split_dependencies = int(
            sum(
                split_rank[previous] > split_rank[current]
                for current, previous in joined[["data_split", "history_data_split"]].itertuples(index=False)
            )
        )
        bounds = price.groupby("data_split", observed=True)["prediction_time_utc"].agg(["min", "max"])
        chronological = bool(
            bounds.loc["train", "max"] < bounds.loc["calibration", "min"]
            and bounds.loc["calibration", "max"] < bounds.loc["test", "min"]
        )
        route_invalid = int((~np.isfinite(joined["route_geodesic_km"]) | joined["route_geodesic_km"].le(0)).sum())

        add_check(f"D{delay:02d}_CUTOFF", "PASS" if cutoff_mismatches == 0 else "FAIL", cutoff_mismatches, 0, "cutoff equals prediction time minus configured delay")
        add_check(f"D{delay:02d}_HISTORY_TIME", "PASS" if future_history == 0 else "FAIL", future_history, 0, "history observation is at or before cutoff")
        add_check(f"D{delay:02d}_AGE", "PASS" if age_mismatches == 0 and joined["observation_age_minutes"].ge(delay).all() else "FAIL", {"mismatches": age_mismatches, "minimum": float(joined["observation_age_minutes"].min())}, {"mismatches": 0, "minimum": f">={delay}"}, "observation age consistency")
        add_check(f"D{delay:02d}_HISTORY_VALUES", "PASS" if sum(history_mismatches.values()) == 0 else "FAIL", history_mismatches, {}, "history features match selected past snapshot")
        add_check(f"D{delay:02d}_WEATHER_TIME", "PASS" if weather_future == 0 else "FAIL", weather_future, 0, "weather observation is not from the future")
        add_check(f"D{delay:02d}_WEATHER_STALE", "PASS" if stale_with_values == 0 and available_too_old == 0 else "FAIL", {"stale_with_values": stale_with_values, "available_too_old": available_too_old}, {"stale_with_values": 0, "available_too_old": 0}, "stale weather values masked")
        add_check(f"D{delay:02d}_ROUTE", "PASS" if route_invalid == 0 else "FAIL", route_invalid, 0, "static geodesic route distance complete")
        add_check(f"D{delay:02d}_SPLIT", "PASS" if chronological and future_split_dependencies == 0 else "FAIL", {"chronological": chronological, "future_dependencies": future_split_dependencies}, {"chronological": True, "future_dependencies": 0}, "chronological split and causal cross-boundary history")
        add_check(f"D{delay:02d}_LYFT_ONLY", "PASS" if multiplier["cab_type"].eq("Lyft").all() else "FAIL", int(multiplier["cab_type"].ne("Lyft").sum()), 0, "multiplier table restricted to Lyft")
        scenario_rows.append(
            {
                "delay_minutes": delay,
                "price_rows": len(price),
                "multiplier_rows": len(multiplier),
                "minimum_observation_age_minutes": float(price["observation_age_minutes"].min()),
                "median_observation_age_minutes": float(price["observation_age_minutes"].median()),
                "weather_available_pct": float(100 * price["weather_available_asof"].mean()),
                "history_value_mismatches": int(sum(history_mismatches.values())),
                "future_history_rows": future_history,
                "future_weather_rows": weather_future,
                "future_split_dependencies": future_split_dependencies,
            }
        )

    test_previously_inspected = True
    add_check("A04", "WARNING", test_previously_inspected, False, "test was inspected in EDA/baseline v1.0.0; use as POC benchmark")
    add_check("A05", "WARNING", "no standalone validation split", "rolling/blocked CV inside train", "model selection protocol")

    checks = pd.DataFrame(check_rows)
    scenarios = pd.DataFrame(scenario_rows)
    availability = feature_availability(contract)
    blockers = int(checks["status"].eq("BLOCKER").sum())
    failures = int(checks["status"].eq("FAIL").sum())
    overall = "READY_FOR_PRICE_ML_POC_WITH_WARNINGS" if blockers == 0 and failures == 0 else "NOT_READY_FOR_ML"
    summary = {
        "artifact_version": VERSION,
        "input_modeling_data_version": "1.1.0",
        "overall_status": overall,
        "primary_delay_minutes": contract["primary_delay_minutes"],
        "sensitivity_delay_minutes": contract["sensitivity_delay_minutes"],
        "feature_count": len(all_features),
        "check_status_counts": checks["status"].value_counts().to_dict(),
        "blocker_count": blockers,
        "failure_count": failures,
        "warning_count": int(checks["status"].eq("WARNING").sum()),
        "strict_prediction_time_features_allowed": bool(availability["allowed_for_price_ml"].all()),
        "remaining_warnings": [
            "Existing test is not fully blind because earlier EDA/baseline inspected it",
            "Use rolling or blocked time-series CV inside train for ML selection",
        ],
    }
    checks.to_csv(OUTPUT_DIR / f"check_results_v{VERSION}.csv", index=False)
    scenarios.to_csv(OUTPUT_DIR / f"scenario_audit_profile_v{VERSION}.csv", index=False)
    availability.to_csv(OUTPUT_DIR / f"feature_availability_matrix_v{VERSION}.csv", index=False)
    (OUTPUT_DIR / f"audit_summary_v{VERSION}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    files = sorted(
        path for path in OUTPUT_DIR.iterdir() if path.is_file() and not path.name.startswith("manifest_")
    )
    manifest = {
        "artifact_name": "prediction_time_leakage_audit",
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
    if failures or blockers:
        raise RuntimeError("Prediction-time leakage audit did not pass")


if __name__ == "__main__":
    main()
