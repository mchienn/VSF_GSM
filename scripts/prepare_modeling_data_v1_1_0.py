from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "artifacts" / "modeling_data" / "v1.0.0"
OUTPUT_DIR = ROOT / "artifacts" / "modeling_data" / "v1.1.0"
VERSION = "1.1.0"
PRIMARY_DELAY_MINUTES = 15
DELAY_SCENARIOS_MINUTES = [5, 15, 30]
MAX_WEATHER_AGE_MINUTES = 180
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

# Geocodes were frozen from the relation-study v1.0.0 Nominatim results.
LOCATION_COORDINATES = {
    "Back Bay": (42.3507067, -71.0797297),
    "Beacon Hill": (42.3587085, -71.0678290),
    "Boston University": (42.3504215, -71.1032247),
    "Fenway": (42.3488317, -71.0972113),
    "Financial District": (42.3524096, -71.0563898),
    "Haymarket Square": (42.3629502, -71.0578447),
    "North End": (42.3650974, -71.0544954),
    "North Station": (42.3662986, -71.0621622),
    "Northeastern University": (42.3351065, -71.0892575),
    "South Station": (42.3507662, -71.0554618),
    "Theatre District": (42.3513832, -71.0641331),
    "West End": (42.3639186, -71.0638993),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def haversine_km(source: pd.Series, destination: pd.Series) -> np.ndarray:
    source_lat = source.map(lambda value: LOCATION_COORDINATES[value][0]).to_numpy(dtype=float)
    source_lon = source.map(lambda value: LOCATION_COORDINATES[value][1]).to_numpy(dtype=float)
    destination_lat = destination.map(lambda value: LOCATION_COORDINATES[value][0]).to_numpy(dtype=float)
    destination_lon = destination.map(lambda value: LOCATION_COORDINATES[value][1]).to_numpy(dtype=float)
    lat1 = np.radians(source_lat)
    lon1 = np.radians(source_lon)
    lat2 = np.radians(destination_lat)
    lon2 = np.radians(destination_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def build_weather_asof(canonical: pd.DataFrame, prediction_times: pd.Series) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        column: (column, "median") for column in WEATHER_NUMERIC
    }
    aggregations["short_summary"] = ("short_summary", "first")
    weather = (
        canonical.groupby("event_time_utc", observed=True)
        .agg(**aggregations)
        .reset_index()
        .rename(columns={"event_time_utc": "weather_observation_time_utc"})
        .sort_values("weather_observation_time_utc")
    )
    query = pd.DataFrame({"prediction_time_utc": pd.Series(prediction_times.unique()).sort_values()})
    joined = pd.merge_asof(
        query,
        weather,
        left_on="prediction_time_utc",
        right_on="weather_observation_time_utc",
        direction="backward",
        allow_exact_matches=True,
    )
    joined["weather_age_minutes"] = (
        joined["prediction_time_utc"] - joined["weather_observation_time_utc"]
    ).dt.total_seconds().div(60).astype("float32")
    joined["weather_available_asof"] = joined["weather_age_minutes"].le(
        MAX_WEATHER_AGE_MINUTES
    )
    stale = ~joined["weather_available_asof"]
    joined.loc[stale, WEATHER_NUMERIC] = np.nan
    joined.loc[stale, "short_summary"] = "no_asof_weather"
    return joined


def history_table(snapshots: pd.DataFrame) -> pd.DataFrame:
    history = snapshots.sort_values(SERIES_KEYS + ["snapshot_time_utc"], kind="mergesort").copy()
    grouped = history.groupby(SERIES_KEYS, observed=True, sort=False)
    history["lag1_price_median_asof"] = history["target_price_median"]
    history["lag2_price_median_asof"] = grouped["target_price_median"].shift(1)
    history["lag3_price_median_asof"] = grouped["target_price_median"].shift(2)
    history["lag1_multiplier_median_asof"] = history["target_multiplier_median"]
    history["lag1_quote_count_asof"] = history["target_quote_count"]
    history["lag1_price_spread_asof"] = history["target_price_spread"]
    history["lag1_distance_median_asof"] = history["distance_median"]
    history["lag_price_delta_1_2_asof"] = (
        history["lag1_price_median_asof"] - history["lag2_price_median_asof"]
    )
    current_price = history["target_price_median"]
    price_groups = current_price.groupby(
        [history[column] for column in SERIES_KEYS], observed=True, sort=False
    )
    history["history_price_mean_last3_asof"] = price_groups.transform(
        lambda values: values.rolling(3, min_periods=1).mean()
    )
    history["history_price_std_last3_asof"] = price_groups.transform(
        lambda values: values.rolling(3, min_periods=2).std(ddof=0)
    )
    history["history_price_mean_last6_asof"] = price_groups.transform(
        lambda values: values.rolling(6, min_periods=1).mean()
    )
    history["history_observation_count_asof"] = (grouped.cumcount() + 1).clip(upper=32767).astype("int16")
    return history.rename(columns={"snapshot_time_utc": "history_observation_time_utc"})


def attach_history(base: pd.DataFrame, history: pd.DataFrame, delay_minutes: int) -> pd.DataFrame:
    left = base.copy()
    left["delay_minutes"] = np.int16(delay_minutes)
    left["history_cutoff_utc"] = left["prediction_time_utc"] - pd.to_timedelta(delay_minutes, unit="m")
    history_columns = SERIES_KEYS + [
        "history_observation_time_utc",
        "lag1_price_median_asof",
        "lag2_price_median_asof",
        "lag3_price_median_asof",
        "lag1_multiplier_median_asof",
        "lag1_quote_count_asof",
        "lag1_price_spread_asof",
        "lag1_distance_median_asof",
        "lag_price_delta_1_2_asof",
        "history_price_mean_last3_asof",
        "history_price_std_last3_asof",
        "history_price_mean_last6_asof",
        "history_observation_count_asof",
    ]
    left = left.sort_values(["history_cutoff_utc", *SERIES_KEYS], kind="mergesort")
    right = history[history_columns].sort_values(
        ["history_observation_time_utc", *SERIES_KEYS], kind="mergesort"
    )
    joined = pd.merge_asof(
        left,
        right,
        left_on="history_cutoff_utc",
        right_on="history_observation_time_utc",
        by=SERIES_KEYS,
        direction="backward",
        allow_exact_matches=True,
    )
    rename = {
        column: column.removesuffix("_asof")
        for column in joined.columns
        if column.endswith("_asof") and column != "weather_available_asof"
    }
    joined = joined.rename(columns=rename)
    joined["observation_age_minutes"] = (
        joined["prediction_time_utc"] - joined["history_observation_time_utc"]
    ).dt.total_seconds().div(60).astype("float32")
    joined["observation_age_bucket"] = pd.cut(
        joined["observation_age_minutes"],
        bins=[-np.inf, 15, 30, 60, 120, np.inf],
        labels=["00-15", "15-30", "30-60", "60-120", "120+"],
        right=True,
    ).astype("string").fillna("no_history")
    return joined.sort_values(["prediction_time_utc", "snapshot_id"], kind="mergesort").reset_index(drop=True)


def feature_contract() -> dict[str, object]:
    metadata = [
        "snapshot_id",
        "series_id",
        "prediction_time_utc",
        "event_time_local",
        "data_split",
        "delay_minutes",
        "history_cutoff_utc",
        "history_observation_time_utc",
        "weather_observation_time_utc",
    ]
    categorical = [
        "cab_type",
        "name",
        "source",
        "destination",
        "short_summary",
        "observation_age_bucket",
    ]
    numeric = [
        "route_geodesic_km",
        *WEATHER_NUMERIC,
        "weather_age_minutes",
        "weather_available_asof",
        "event_hour_local",
        "event_weekday_local",
        "event_month_local",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
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
    ]
    return {
        "pipeline_version": VERSION,
        "feature_policy_version": VERSION,
        "split_policy_version": "1.0.0",
        "prediction_protocol_version": VERSION,
        "prediction_target_window": "5-minute quote median for bucket beginning at prediction_time_utc",
        "primary_delay_minutes": PRIMARY_DELAY_MINUTES,
        "sensitivity_delay_minutes": [5, 30],
        "history_availability_rule": "history_observation_time_utc <= prediction_time_utc - delay_minutes",
        "weather_availability_rule": "weather_observation_time_utc <= prediction_time_utc",
        "maximum_weather_age_minutes": MAX_WEATHER_AGE_MINUTES,
        "distance_policy": "static geodesic distance from frozen source/destination coordinates",
        "metadata_columns": metadata,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "price_target": "target_price_median",
        "multiplier_target": "target_multiplier_median",
        "forbidden_contemporaneous_features": [
            "distance_median",
            "target_quote_count",
            "target_price_min",
            "target_price_max",
            "target_price_spread",
            "target_multiplier_max",
        ],
        "split_definition": {
            "train": "2018-11-26 through 2018-12-10 UTC",
            "calibration": "2018-12-13 through 2018-12-15 UTC",
            "test": "2018-12-16 through 2018-12-18 UTC",
            "validation": "rolling or blocked time-series CV inside train",
            "missing_gaps_preserved": ["2018-12-05..2018-12-08", "2018-12-11..2018-12-12"],
        },
    }


def split_profile(frame: pd.DataFrame, task: str, target: str, delay: int) -> pd.DataFrame:
    profile = (
        frame.groupby("data_split", observed=True)
        .agg(
            rows=(target, "size"),
            series=("series_id", "nunique"),
            start_time=("prediction_time_utc", "min"),
            end_time=("prediction_time_utc", "max"),
            target_mean=(target, "mean"),
            target_median=(target, "median"),
            median_observation_age_minutes=("observation_age_minutes", "median"),
        )
        .reset_index()
    )
    profile.insert(0, "delay_minutes", delay)
    profile.insert(0, "task", task)
    return profile


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_parquet(INPUT_DIR / "canonical_quotes_v1.0.0.parquet")
    snapshots = pd.read_parquet(INPUT_DIR / "price_snapshots_5min_v1.0.0.parquet")
    contract = feature_contract()

    base_columns = [
        "snapshot_id",
        "series_id",
        "snapshot_time_utc",
        "event_time_local",
        "data_split",
        *SERIES_KEYS,
        "event_hour_local",
        "event_weekday_local",
        "event_month_local",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "target_price_median",
        "target_multiplier_median",
    ]
    base = snapshots[base_columns].rename(columns={"snapshot_time_utc": "prediction_time_utc"}).copy()
    base["route_geodesic_km"] = haversine_km(base["source"], base["destination"])
    weather = build_weather_asof(canonical, base["prediction_time_utc"])
    base = base.merge(weather, on="prediction_time_utc", how="left", validate="many_to_one")
    history = history_table(snapshots)

    model_columns = contract["metadata_columns"] + contract["categorical_features"] + contract["numeric_features"]
    split_profiles: list[pd.DataFrame] = []
    validation_rows: list[dict[str, object]] = []
    scenario_summary: dict[str, object] = {}

    for delay in DELAY_SCENARIOS_MINUTES:
        scenario = attach_history(base, history, delay)
        scenario = scenario.loc[scenario["history_observation_time_utc"].notna()].copy()
        price = scenario[model_columns + [contract["price_target"]]].copy()
        multiplier = scenario.loc[scenario["cab_type"].eq("Lyft"), model_columns + [contract["multiplier_target"]]].copy()
        price_path = OUTPUT_DIR / f"price_modeling_delay_{delay:02d}m_v{VERSION}.parquet"
        multiplier_path = OUTPUT_DIR / f"lyft_multiplier_modeling_delay_{delay:02d}m_v{VERSION}.parquet"
        price.to_parquet(price_path, index=False, compression="zstd")
        multiplier.to_parquet(multiplier_path, index=False, compression="zstd")

        split_profiles.extend(
            [
                split_profile(price, "price", contract["price_target"], delay),
                split_profile(multiplier, "lyft_multiplier", contract["multiplier_target"], delay),
            ]
        )
        history_safe = bool(
            (scenario["history_observation_time_utc"] <= scenario["history_cutoff_utc"]).all()
        )
        weather_safe = bool(
            (scenario["weather_observation_time_utc"] <= scenario["prediction_time_utc"]).all()
        )
        stale_weather_masked = bool(
            scenario.loc[
                ~scenario["weather_available_asof"], WEATHER_NUMERIC
            ].isna().all().all()
        )
        age_safe = bool(scenario["observation_age_minutes"].ge(delay).all())
        validation_rows.extend(
            [
                {"check_id": f"D{delay:02d}_HISTORY_ASOF", "status": "PASS" if history_safe else "FAIL", "actual": history_safe, "expected": True},
                {"check_id": f"D{delay:02d}_WEATHER_ASOF", "status": "PASS" if weather_safe else "FAIL", "actual": weather_safe, "expected": True},
                {"check_id": f"D{delay:02d}_WEATHER_STALE_MASK", "status": "PASS" if stale_weather_masked else "FAIL", "actual": stale_weather_masked, "expected": True},
                {"check_id": f"D{delay:02d}_MINIMUM_AGE", "status": "PASS" if age_safe else "FAIL", "actual": float(scenario["observation_age_minutes"].min()), "expected": f">={delay}"},
                {"check_id": f"D{delay:02d}_ROUTE_DISTANCE", "status": "PASS" if scenario["route_geodesic_km"].notna().all() else "FAIL", "actual": int(scenario["route_geodesic_km"].isna().sum()), "expected": 0},
                {"check_id": f"D{delay:02d}_PRICE_ROWS", "status": "PASS" if len(price) > 500_000 else "FAIL", "actual": len(price), "expected": ">500000"},
                {"check_id": f"D{delay:02d}_MULTIPLIER_ROWS", "status": "PASS" if len(multiplier) > 250_000 else "FAIL", "actual": len(multiplier), "expected": ">250000"},
            ]
        )
        scenario_summary[str(delay)] = {
            "price_rows": len(price),
            "multiplier_rows": len(multiplier),
            "price_history_coverage_pct": float(100 * len(price) / len(base)),
            "minimum_observation_age_minutes": float(scenario["observation_age_minutes"].min()),
            "median_observation_age_minutes": float(scenario["observation_age_minutes"].median()),
            "weather_available_pct": float(100 * scenario["weather_available_asof"].mean()),
            "maximum_available_weather_age_minutes": float(
                scenario.loc[scenario["weather_available_asof"], "weather_age_minutes"].max()
            ),
        }

    profile_frame = pd.concat(split_profiles, ignore_index=True)
    profile_frame.to_csv(OUTPUT_DIR / f"split_profile_v{VERSION}.csv", index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUTPUT_DIR / f"validation_results_v{VERSION}.csv", index=False)
    if not validation["status"].eq("PASS").all():
        raise RuntimeError(validation.loc[~validation["status"].eq("PASS")].to_dict("records"))

    (OUTPUT_DIR / f"feature_contract_v{VERSION}.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    summary = {
        "pipeline_version": VERSION,
        "input_modeling_data_version": "1.0.0",
        "primary_delay_minutes": PRIMARY_DELAY_MINUTES,
        "sensitivity_delay_minutes": [5, 30],
        "feature_count": len(contract["categorical_features"] + contract["numeric_features"]),
        "categorical_feature_count": len(contract["categorical_features"]),
        "numeric_feature_count": len(contract["numeric_features"]),
        "scenario_summary": scenario_summary,
        "validation_checks": len(validation),
        "validation_passed": int(validation["status"].eq("PASS").sum()),
    }
    (OUTPUT_DIR / f"preprocessing_summary_v{VERSION}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    artifact_files = sorted(
        path for path in OUTPUT_DIR.iterdir() if path.is_file() and not path.name.startswith("manifest_")
    )
    manifest = {
        "artifact_name": "modeling_data",
        "artifact_version": VERSION,
        "files": [
            {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifact_files
        ],
    }
    (OUTPUT_DIR / f"manifest_v{VERSION}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
