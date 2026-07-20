from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


PIPELINE_VERSION = "1.0.0"
AUDIT_VERSION = "1.0.0"
POLICY_VERSION = "1.0.0"
SPLIT_POLICY_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rideshare_kaggle.csv.zip"
AUDIT_DIR = ROOT / "artifacts" / "data_readiness" / "v1.0.0"
OUT = ROOT / "artifacts" / "modeling_data" / "v1.0.0"

SERIES_KEYS = ["cab_type", "name", "source", "destination"]
CURRENT_WEATHER_NUMERIC = [
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
CURRENT_WEATHER_CATEGORICAL = ["short_summary"]
SOURCE_COLUMNS = [
    "id",
    "timestamp",
    "datetime",
    "timezone",
    "source",
    "destination",
    "cab_type",
    "product_id",
    "name",
    "price",
    "distance",
    "surge_multiplier",
] + CURRENT_WEATHER_NUMERIC + CURRENT_WEATHER_CATEGORICAL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source() -> pd.DataFrame:
    with ZipFile(SOURCE) as archive:
        csv_name = archive.namelist()[0]
        with archive.open(csv_name) as stream:
            return pd.read_csv(stream, usecols=SOURCE_COLUMNS)


def local_time_features(event_time_utc: pd.Series) -> pd.DataFrame:
    local = event_time_utc.dt.tz_convert("America/New_York")
    hour = local.dt.hour.astype("int8")
    weekday = local.dt.weekday.astype("int8")
    return pd.DataFrame(
        {
            "event_time_local": local,
            "event_date_local": local.dt.strftime("%Y-%m-%d"),
            "event_hour_local": hour,
            "event_weekday_local": weekday,
            "event_month_local": local.dt.month.astype("int8"),
            "is_weekend": weekday.isin([5, 6]),
            "hour_sin": np.sin(2 * np.pi * hour / 24).astype("float32"),
            "hour_cos": np.cos(2 * np.pi * hour / 24).astype("float32"),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7).astype("float32"),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7).astype("float32"),
        },
        index=event_time_utc.index,
    )


def assign_split(event_date_local: pd.Series) -> pd.Series:
    dates = pd.to_datetime(event_date_local)
    split = pd.Series(pd.NA, index=event_date_local.index, dtype="string")
    split.loc[dates <= pd.Timestamp("2018-12-10")] = "train"
    split.loc[dates.between(pd.Timestamp("2018-12-13"), pd.Timestamp("2018-12-15"))] = "calibration"
    split.loc[dates >= pd.Timestamp("2018-12-16")] = "test"
    if split.isna().any():
        unknown = sorted(event_date_local.loc[split.isna()].unique().tolist())
        raise ValueError(f"Dates outside split policy: {unknown}")
    return split


def canonicalize(raw: pd.DataFrame) -> pd.DataFrame:
    event_time_utc = pd.to_datetime(raw["timestamp"], unit="s", utc=True, errors="raise")
    raw_datetime = pd.to_datetime(raw["datetime"], errors="coerce")
    mismatch_minutes = (
        event_time_utc.dt.tz_localize(None) - raw_datetime
    ).dt.total_seconds().abs() / 60

    canonical = raw.rename(columns={"datetime": "raw_datetime_qa"}).copy()
    canonical.insert(2, "event_time_utc", event_time_utc)
    canonical.insert(3, "event_date_utc", event_time_utc.dt.strftime("%Y-%m-%d"))
    local_features = local_time_features(event_time_utc)
    insert_at = 4
    for column in local_features.columns:
        canonical.insert(insert_at, column, local_features[column])
        insert_at += 1
    canonical["timestamp_mismatch_minutes_qa"] = mismatch_minutes.astype("float32")
    canonical["timestamp_mismatch_gt_5min_qa"] = mismatch_minutes.gt(5)
    canonical["price_target_eligible"] = canonical["price"].notna() & canonical["name"].ne("Taxi")
    canonical["multiplier_target_eligible"] = canonical["cab_type"].eq("Lyft")
    canonical["data_split"] = assign_split(canonical["event_date_utc"])
    canonical = canonical.sort_values(["event_time_utc", "id"], kind="mergesort").reset_index(drop=True)
    return canonical


def create_snapshots(canonical: pd.DataFrame) -> pd.DataFrame:
    priced = canonical.loc[canonical["price_target_eligible"]].copy()
    priced["snapshot_time_utc"] = priced["event_time_utc"].dt.floor("5min")

    aggregations: dict[str, tuple[str, str]] = {
        "target_quote_count": ("id", "size"),
        "target_price_median": ("price", "median"),
        "target_price_min": ("price", "min"),
        "target_price_max": ("price", "max"),
        "target_multiplier_median": ("surge_multiplier", "median"),
        "target_multiplier_max": ("surge_multiplier", "max"),
        "distance_median": ("distance", "median"),
        "distance_min": ("distance", "min"),
        "distance_max": ("distance", "max"),
        "short_summary": ("short_summary", "first"),
    }
    for column in CURRENT_WEATHER_NUMERIC:
        aggregations[column] = (column, "median")

    snapshots = (
        priced.groupby(SERIES_KEYS + ["snapshot_time_utc"], observed=True, sort=True)
        .agg(**aggregations)
        .reset_index()
        .sort_values(SERIES_KEYS + ["snapshot_time_utc"], kind="mergesort")
        .reset_index(drop=True)
    )
    snapshots["target_price_spread"] = snapshots["target_price_max"] - snapshots["target_price_min"]
    snapshots["distance_spread"] = snapshots["distance_max"] - snapshots["distance_min"]
    snapshots["series_id"] = snapshots[SERIES_KEYS].astype(str).agg("|".join, axis=1)
    snapshots["snapshot_id"] = (
        snapshots["series_id"] + "|" + snapshots["snapshot_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    time_features = local_time_features(snapshots["snapshot_time_utc"])
    for column in time_features.columns:
        snapshots[column] = time_features[column]
    snapshots["snapshot_date_utc"] = snapshots["snapshot_time_utc"].dt.strftime("%Y-%m-%d")
    snapshots["data_split"] = assign_split(snapshots["snapshot_date_utc"])

    grouped = snapshots.groupby(SERIES_KEYS, observed=True, sort=False)
    snapshots["previous_observation_time_utc"] = grouped["snapshot_time_utc"].shift(1)
    snapshots["observation_age_minutes"] = (
        snapshots["snapshot_time_utc"] - snapshots["previous_observation_time_utc"]
    ).dt.total_seconds().div(60).astype("float32")
    snapshots["lag1_price_median"] = grouped["target_price_median"].shift(1)
    snapshots["lag2_price_median"] = grouped["target_price_median"].shift(2)
    snapshots["lag3_price_median"] = grouped["target_price_median"].shift(3)
    snapshots["lag1_multiplier_median"] = grouped["target_multiplier_median"].shift(1)
    snapshots["lag1_quote_count"] = grouped["target_quote_count"].shift(1)
    snapshots["lag1_price_spread"] = grouped["target_price_spread"].shift(1)
    snapshots["lag1_distance_median"] = grouped["distance_median"].shift(1)
    snapshots["lag_price_delta_1_2"] = snapshots["lag1_price_median"] - snapshots["lag2_price_median"]

    shifted_price = snapshots["lag1_price_median"]
    shifted_groups = shifted_price.groupby(
        [snapshots[column] for column in SERIES_KEYS], observed=True, sort=False
    )
    snapshots["history_price_mean_last3"] = shifted_groups.transform(
        lambda values: values.rolling(3, min_periods=1).mean()
    )
    snapshots["history_price_std_last3"] = shifted_groups.transform(
        lambda values: values.rolling(3, min_periods=2).std(ddof=0)
    )
    snapshots["history_price_mean_last6"] = shifted_groups.transform(
        lambda values: values.rolling(6, min_periods=1).mean()
    )
    snapshots["history_observation_count"] = grouped.cumcount().clip(upper=32767).astype("int16")
    snapshots["history_available"] = snapshots["lag1_price_median"].notna()
    snapshots["observation_age_bucket"] = pd.cut(
        snapshots["observation_age_minutes"],
        bins=[-np.inf, 15, 30, 60, 120, np.inf],
        labels=["00-15", "15-30", "30-60", "60-120", "120+"],
        right=True,
    ).astype("string").fillna("no_history")
    return snapshots


def feature_contract() -> dict[str, object]:
    metadata = [
        "snapshot_id",
        "series_id",
        "snapshot_time_utc",
        "event_time_local",
        "data_split",
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
        "distance_median",
        *CURRENT_WEATHER_NUMERIC,
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
        "pipeline_version": PIPELINE_VERSION,
        "feature_policy_version": POLICY_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "metadata_columns": metadata,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "price_target": "target_price_median",
        "multiplier_target": "target_multiplier_median",
        "forbidden_contemporaneous_features": [
            "target_quote_count",
            "target_price_min",
            "target_price_max",
            "target_price_spread",
            "target_multiplier_max",
        ],
        "split_definition": {
            "train": "2018-11-26 through 2018-12-10 (missing dates remain absent)",
            "calibration": "2018-12-13 through 2018-12-15",
            "test": "2018-12-16 through 2018-12-18",
            "missing_gaps_preserved": ["2018-12-05..2018-12-08", "2018-12-11..2018-12-12"],
        },
    }


def create_modeling_tables(snapshots: pd.DataFrame, contract: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_columns = (
        contract["metadata_columns"]
        + contract["categorical_features"]
        + contract["numeric_features"]
    )
    price_columns = base_columns + [contract["price_target"]]
    multiplier_columns = base_columns + [contract["multiplier_target"]]

    eligible = snapshots.loc[snapshots["history_available"]].copy()
    price_modeling = eligible[price_columns].sort_values(
        ["snapshot_time_utc", "snapshot_id"], kind="mergesort"
    ).reset_index(drop=True)
    multiplier_modeling = eligible.loc[eligible["cab_type"].eq("Lyft"), multiplier_columns].sort_values(
        ["snapshot_time_utc", "snapshot_id"], kind="mergesort"
    ).reset_index(drop=True)
    return price_modeling, multiplier_modeling


def validation_results(
    canonical: pd.DataFrame,
    snapshots: pd.DataFrame,
    price_modeling: pd.DataFrame,
    multiplier_modeling: pd.DataFrame,
    contract: dict[str, object],
) -> pd.DataFrame:
    expected_split_order = {"train": 0, "calibration": 1, "test": 2}
    split_bounds = price_modeling.groupby("data_split")["snapshot_time_utc"].agg(["min", "max"])
    split_sizes = price_modeling["data_split"].value_counts()
    calibration_series = price_modeling.loc[
        price_modeling["data_split"].eq("calibration"), "series_id"
    ].nunique()
    all_series = price_modeling["series_id"].nunique()
    chronological = all(
        split_bounds.loc[left, "max"] < split_bounds.loc[right, "min"]
        for left, right in [("train", "calibration"), ("calibration", "test")]
    )
    model_features = set(contract["categorical_features"] + contract["numeric_features"])
    forbidden = set(contract["forbidden_contemporaneous_features"])
    validations = [
        ("row_count_source", len(canonical) == 693_071, len(canonical), 693_071),
        ("unique_raw_id", canonical["id"].is_unique, canonical["id"].nunique(), len(canonical)),
        ("canonical_timestamp_complete", canonical["event_time_utc"].notna().all(), int(canonical["event_time_utc"].isna().sum()), 0),
        ("taxi_excluded_from_price_target", not snapshots["name"].eq("Taxi").any(), int(snapshots["name"].eq("Taxi").sum()), 0),
        ("price_model_has_history", price_modeling["lag1_price_median"].notna().all(), int(price_modeling["lag1_price_median"].isna().sum()), 0),
        ("multiplier_model_lyft_only", multiplier_modeling["cab_type"].eq("Lyft").all(), int(multiplier_modeling["cab_type"].ne("Lyft").sum()), 0),
        ("no_forbidden_current_target_features", model_features.isdisjoint(forbidden), sorted(model_features & forbidden), []),
        ("all_rows_have_split", price_modeling["data_split"].notna().all(), int(price_modeling["data_split"].isna().sum()), 0),
        ("chronological_split_order", chronological, expected_split_order, expected_split_order),
        ("calibration_minimum_rows", split_sizes.get("calibration", 0) >= 50_000, int(split_sizes.get("calibration", 0)), ">=50000"),
        ("calibration_series_coverage", calibration_series / all_series >= 0.95, round(calibration_series / all_series, 6), ">=0.95"),
        ("snapshot_id_unique", snapshots["snapshot_id"].is_unique, snapshots["snapshot_id"].nunique(), len(snapshots)),
        ("nonnegative_observation_age", snapshots.loc[snapshots["history_available"], "observation_age_minutes"].gt(0).all(), float(snapshots["observation_age_minutes"].min()), ">0"),
        ("source_sha_matches_audit", sha256(SOURCE) == json.loads((AUDIT_DIR / "audit_summary_v1.0.0.json").read_text(encoding="utf-8"))["source_sha256"], sha256(SOURCE), "audit source hash"),
    ]
    return pd.DataFrame(
        [(name, bool(passed), json.dumps(actual, default=str), json.dumps(expected, default=str)) for name, passed, actual, expected in validations],
        columns=["check", "passed", "actual", "expected"],
    )


def split_profile(table: pd.DataFrame, target: str, dataset: str) -> pd.DataFrame:
    profile = table.groupby("data_split", observed=True).agg(
        rows=(target, "size"),
        target_mean=(target, "mean"),
        target_median=(target, "median"),
        start_time=("snapshot_time_utc", "min"),
        end_time=("snapshot_time_utc", "max"),
        series=("series_id", "nunique"),
    ).reset_index()
    profile.insert(0, "dataset", dataset)
    return profile


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_source()
    canonical = canonicalize(raw)
    snapshots = create_snapshots(canonical)
    contract = feature_contract()
    price_modeling, multiplier_modeling = create_modeling_tables(snapshots, contract)
    checks = validation_results(canonical, snapshots, price_modeling, multiplier_modeling, contract)
    if not checks["passed"].all():
        failures = checks.loc[~checks["passed"]].to_dict("records")
        raise AssertionError(f"Data contract validation failed: {failures}")

    canonical_path = OUT / "canonical_quotes_v1.0.0.parquet"
    snapshots_path = OUT / "price_snapshots_5min_v1.0.0.parquet"
    price_path = OUT / "price_modeling_v1.0.0.parquet"
    multiplier_path = OUT / "lyft_multiplier_modeling_v1.0.0.parquet"
    canonical.to_parquet(canonical_path, index=False, compression="zstd")
    snapshots.to_parquet(snapshots_path, index=False, compression="zstd")
    price_modeling.to_parquet(price_path, index=False, compression="zstd")
    multiplier_modeling.to_parquet(multiplier_path, index=False, compression="zstd")

    (OUT / "feature_contract_v1.0.0.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checks.to_csv(OUT / "validation_results_v1.0.0.csv", index=False)
    split_profiles = pd.concat(
        [
            split_profile(price_modeling, "target_price_median", "price"),
            split_profile(multiplier_modeling, "target_multiplier_median", "lyft_multiplier"),
        ],
        ignore_index=True,
    )
    split_profiles.to_csv(OUT / "split_profile_v1.0.0.csv", index=False)

    treatment_log = pd.DataFrame(
        [
            ("DR-001", "RESOLVED", "Canonical UTC timestamp parsed from epoch seconds; local features recomputed."),
            ("DR-002", "RESOLVED_WITH_QA_FLAG", "Raw datetime retained only for QA; canonical timestamp drives all logic."),
            ("DR-003", "RESOLVED_FOR_SUPERVISED_TARGET", "Uber Taxi retained in canonical layer and excluded from price snapshots/modeling."),
            ("DR-004", "EXTERNAL_LIMITATION", "Uber multiplier modeling is not produced; only Lyft multiplier table is produced."),
            ("DR-005", "MITIGATED", "Lyft-only table and split profile preserve tail distribution for later weighted metrics/modeling."),
            ("DR-006", "MITIGATED", "Event-time lag features, observation age, age bucket and history count created without label fill."),
            ("DR-007", "MITIGATED", "Missing dates preserved; chronological train/calibration/test use separated observed blocks."),
            ("DR-008", "RESOLVED", "Raw IDs retained; deterministic 5-minute snapshot aggregates created."),
            ("DR-009", "RESOLVED_FOR_POC", "Only current/as-of weather whitelist retained; daily/future weather omitted."),
            ("DR-010", "EXTERNAL_LIMITATION", "Boston/17-day scope remains and must constrain claims."),
            ("DR-011", "EXTERNAL_LIMITATION", "No Vietnam validation claim; local data collection remains required."),
        ],
        columns=["issue_id", "status", "implementation"],
    )
    treatment_log.to_csv(OUT / "treatment_log_v1.0.0.csv", index=False)

    split_counts = price_modeling["data_split"].value_counts().to_dict()
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "audit_version": AUDIT_VERSION,
        "feature_policy_version": POLICY_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "source_sha256": sha256(SOURCE),
        "canonical_rows": len(canonical),
        "snapshot_rows": len(snapshots),
        "price_modeling_rows": len(price_modeling),
        "lyft_multiplier_modeling_rows": len(multiplier_modeling),
        "price_split_rows": {key: int(value) for key, value in split_counts.items()},
        "validation_checks": len(checks),
        "validation_passed": int(checks["passed"].sum()),
        "readiness_for_eda": "READY",
        "production_vietnam_readiness": "NOT_VALIDATED",
    }
    (OUT / "preprocessing_summary_v1.0.0.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# Pre-EDA Data Preparation Report v{PIPELINE_VERSION}

## Kết luận

Các xử lý bắt buộc từ Data Readiness Audit v{AUDIT_VERSION} đã được áp dụng. Dữ liệu hiện **sẵn sàng cho EDA và POC modeling**, nhưng vẫn **chưa xác nhận được khả năng áp dụng production tại Việt Nam**.

## Đầu ra chính

- Canonical quotes: {len(canonical):,} dòng; giữ raw ID, canonical time, QA flags và feature whitelist.
- 5-minute snapshots: {len(snapshots):,} dòng; Uber Taxi đã loại khỏi supervised price target.
- Price modeling table: {len(price_modeling):,} dòng có ít nhất một quan sát lịch sử.
- Lyft multiplier modeling table: {len(multiplier_modeling):,} dòng; không tạo Uber multiplier target giả.
- Validation: {int(checks['passed'].sum())}/{len(checks)} checks passed.

## Split policy v{SPLIT_POLICY_VERSION}

- Train: 2018-11-26 đến 2018-12-10 — {split_counts.get('train', 0):,} price rows; các ngày thiếu vẫn được giữ trống.
- Calibration: 2018-12-13 đến 2018-12-15 — {split_counts.get('calibration', 0):,} price rows.
- Test: 2018-12-16 đến 2018-12-18 — {split_counts.get('test', 0):,} price rows.
- Các ngày trống không được nội suy và tạo khoảng cách tự nhiên giữa các split.

## Leakage controls

- Price/multiplier hiện tại chỉ là target, không nằm trong feature list.
- Lịch sử giá chỉ dùng các snapshot trước đó qua lag/rolling features.
- Current-bucket quote count/spread/min/max chỉ có trong snapshot layer, bị cấm trong modeling feature contract.
- Daily/future weather và `visibility.1` không có trong canonical/modeling layer.

## Giới hạn còn lại

Boston-only, 17 ngày, không có ground-truth Uber surge và không có dữ liệu Việt Nam. Đây là giới hạn nguồn dữ liệu, không thể sửa bằng preprocessing.
"""
    (OUT / "REPORT_v1.0.0.md").write_text(report, encoding="utf-8")

    output_files = sorted(
        path for path in OUT.iterdir() if path.is_file() and path.name != "manifest_v1.0.0.json"
    )
    script_path = Path(__file__).resolve()
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "audit_version": AUDIT_VERSION,
        "feature_policy_version": POLICY_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "source": {"path": str(SOURCE), "sha256": sha256(SOURCE)},
        "script": {"path": str(script_path), "sha256": sha256(script_path)},
        "outputs": [
            {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
    }
    (OUT / "manifest_v1.0.0.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
