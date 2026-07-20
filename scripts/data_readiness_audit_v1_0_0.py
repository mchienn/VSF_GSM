from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


AUDIT_VERSION = "1.0.0"
POLICY_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rideshare_kaggle.csv.zip"
OUT = ROOT / "artifacts" / "data_readiness" / "v1.0.0"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def q(series: pd.Series, probs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)) -> dict[str, float]:
    values = series.dropna().quantile(list(probs))
    return {f"p{int(p*100):02d}": round(float(values.loc[p]), 4) for p in probs}


def load_header_and_column_profile() -> tuple[list[str], pd.DataFrame, int]:
    missing = None
    dtypes = None
    rows = 0
    with ZipFile(SOURCE) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            for chunk in pd.read_csv(f, chunksize=100_000):
                if dtypes is None:
                    dtypes = chunk.dtypes.astype(str)
                m = chunk.isna().sum()
                missing = m if missing is None else missing.add(m, fill_value=0)
                rows += len(chunk)
    assert missing is not None and dtypes is not None
    profile = pd.DataFrame(
        {
            "column": missing.index,
            "dtype": [dtypes.get(c, "unknown") for c in missing.index],
            "missing_count": missing.astype(int).values,
        }
    )
    profile["missing_pct"] = (100 * profile["missing_count"] / rows).round(6)
    return profile["column"].tolist(), profile, rows


def load_core() -> pd.DataFrame:
    cols = [
        "id", "timestamp", "datetime", "hour", "day", "month", "timezone",
        "source", "destination", "cab_type", "product_id", "name", "price",
        "distance", "surge_multiplier",
    ]
    with ZipFile(SOURCE) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, usecols=cols)


def feature_policy(columns: list[str]) -> pd.DataFrame:
    identifier = {"id", "product_id"}
    canonical_time = {"timestamp"}
    redundant_time = {"datetime", "hour", "day", "month", "timezone"}
    categorical = {"source", "destination", "cab_type", "name"}
    targets = {"price", "surge_multiplier"}
    route_numeric = {"distance"}
    geo = {"latitude", "longitude"}
    current_weather = {
        "temperature", "apparentTemperature", "short_summary", "precipIntensity",
        "precipProbability", "humidity", "windSpeed", "windGust", "visibility",
        "dewPoint", "pressure", "windBearing", "cloudCover", "uvIndex", "ozone",
    }
    daily_or_future = {
        "long_summary", "windGustTime", "temperatureHigh", "temperatureHighTime",
        "temperatureLow", "temperatureLowTime", "apparentTemperatureHigh",
        "apparentTemperatureHighTime", "apparentTemperatureLow",
        "apparentTemperatureLowTime", "sunriseTime", "sunsetTime", "moonPhase",
        "precipIntensityMax", "uvIndexTime", "temperatureMin", "temperatureMinTime",
        "temperatureMax", "temperatureMaxTime", "apparentTemperatureMin",
        "apparentTemperatureMinTime", "apparentTemperatureMax",
        "apparentTemperatureMaxTime", "icon",
    }
    rows = []
    for c in columns:
        if c in identifier:
            role, decision, reason = "identifier", "EXCLUDE_MODEL", "Traceability only; high-cardinality identifier."
        elif c in canonical_time:
            role, decision, reason = "event_time", "KEEP_CANONICAL", "Parse as epoch seconds and use for chronological joins/splits."
        elif c in redundant_time:
            role, decision, reason = "derived_time", "RECOMPUTE", "Derive consistently from canonical timestamp/timezone."
        elif c in categorical:
            role, decision, reason = "context", "KEEP", "Core provider/service/route context."
        elif c in targets:
            role, decision, reason = "target", "TARGET_ONLY", "Never use as contemporaneous feature for the same prediction target."
        elif c in route_numeric:
            role, decision, reason = "route", "KEEP", "Core quote/route feature."
        elif c in geo:
            role, decision, reason = "weather_geo", "KEEP_WITH_CAUTION", "Verify whether coordinates represent weather station or trip endpoint."
        elif c == "visibility.1":
            role, decision, reason = "duplicate", "DROP", "Duplicate/ambiguous copy of visibility."
        elif c in current_weather:
            role, decision, reason = "weather_current", "KEEP_AS_OF", "Use only if value was available at quote time."
        elif c in daily_or_future:
            role, decision, reason = "weather_daily", "EXCLUDE_PENDING_PROVENANCE", "Potential forecast/future-day leakage; provenance unavailable."
        else:
            role, decision, reason = "unclassified", "REVIEW", "Manual provenance review required."
        rows.append((c, role, decision, reason, POLICY_VERSION))
    return pd.DataFrame(rows, columns=["column", "role", "decision", "reason", "policy_version"])


def main() -> None:
    columns, col_profile, total_rows = load_header_and_column_profile()
    core = load_core()
    assert len(core) == total_rows == 693_071
    assert core["id"].nunique() == total_rows

    ts = pd.to_datetime(core["timestamp"], unit="s", utc=True, errors="coerce")
    dt = pd.to_datetime(core["datetime"], errors="coerce")
    mismatch_min = (ts.dt.tz_localize(None) - dt).dt.total_seconds().abs() / 60
    core["ts"] = ts
    core["bucket5"] = ts.dt.floor("5min")

    service_profile = (
        core.groupby(["cab_type", "name"], dropna=False)
        .agg(
            rows=("id", "size"),
            price_non_null=("price", "count"),
            price_missing=("price", lambda x: int(x.isna().sum())),
            price_mean=("price", "mean"),
            price_median=("price", "median"),
            multiplier_nunique=("surge_multiplier", "nunique"),
            multiplier_min=("surge_multiplier", "min"),
            multiplier_max=("surge_multiplier", "max"),
            surge_rows=("surge_multiplier", lambda x: int((x > 1).sum())),
        )
        .reset_index()
    )
    service_profile["price_missing_pct"] = (100 * service_profile.price_missing / service_profile.rows).round(4)
    service_profile["surge_pct"] = (100 * service_profile.surge_rows / service_profile.rows).round(4)

    daily = core.assign(date=ts.dt.date).groupby("date").agg(
        rows=("id", "size"),
        priced_rows=("price", "count"),
        unique_timestamps=("timestamp", "nunique"),
        five_min_buckets=("bucket5", "nunique"),
        routes=("source", lambda _: 0),
    ).reset_index()
    # Route counts require both source and destination.
    route_counts = core.assign(date=ts.dt.date).groupby("date").apply(
        lambda x: x[["source", "destination"]].drop_duplicates().shape[0],
        include_groups=False,
    )
    daily["routes"] = daily["date"].map(route_counts)
    daily["date"] = daily["date"].astype(str)

    valid = core.dropna(subset=["price"]).copy().sort_values("ts")
    keys = ["cab_type", "name", "source", "destination"]
    valid["gap_minutes"] = valid.groupby(keys, observed=True)["ts"].diff().dt.total_seconds() / 60

    gap_rows = []
    overall_gaps = valid["gap_minutes"].dropna()
    overall = {"cab_type": "ALL", "name": "ALL", "observations": len(valid), "series": valid.groupby(keys).ngroups}
    overall.update(q(overall_gaps))
    overall["gap_gt_60_pct"] = round(100 * (overall_gaps > 60).mean(), 4)
    overall["gap_le_15_pct"] = round(100 * (overall_gaps <= 15).mean(), 4)
    gap_rows.append(overall)
    for (cab, name), x in valid.groupby(["cab_type", "name"], observed=True):
        gaps = x["gap_minutes"].dropna()
        row = {"cab_type": cab, "name": name, "observations": len(x), "series": x.groupby(["source", "destination"]).ngroups}
        row.update(q(gaps))
        row["gap_gt_60_pct"] = round(100 * (gaps > 60).mean(), 4)
        row["gap_le_15_pct"] = round(100 * (gaps <= 15).mean(), 4)
        gap_rows.append(row)
    gap_profile = pd.DataFrame(gap_rows)

    natural_key = ["timestamp", "source", "destination", "cab_type", "name"]
    dup_mask = core.duplicated(natural_key, keep=False)
    dup = core.loc[dup_mask]
    dup_groups = dup.groupby(natural_key, dropna=False)
    duplicate_summary = {
        "duplicate_natural_key_rows": int(dup_mask.sum()),
        "duplicate_natural_key_groups": int(dup_groups.ngroups),
        "groups_with_price_conflict": int((dup_groups.price.nunique(dropna=False) > 1).sum()),
        "groups_with_distance_conflict": int((dup_groups.distance.nunique(dropna=False) > 1).sum()),
        "interpretation": "Quote collisions, not exact duplicates; IDs are unique and distance/price can differ.",
    }

    snap = valid.groupby(keys + ["bucket5"], observed=True).agg(
        quote_count=("id", "size"),
        price_median=("price", "median"),
        price_min=("price", "min"),
        price_max=("price", "max"),
        distance_median=("distance", "median"),
        multiplier_median=("surge_multiplier", "median"),
    ).reset_index()
    snap["price_spread"] = snap.price_max - snap.price_min
    snapshot_profile = pd.DataFrame(
        [
            {
                "priced_rows": len(valid),
                "snapshot_groups": len(snap),
                "five_min_buckets": valid.bucket5.nunique(),
                "priced_series": valid.groupby(keys).ngroups,
                "panel_fill_pct": round(100 * len(snap) / (valid.bucket5.nunique() * valid.groupby(keys).ngroups), 4),
                "quote_count_p50": float(snap.quote_count.quantile(0.5)),
                "quote_count_p90": float(snap.quote_count.quantile(0.9)),
                "price_spread_gt_0_pct": round(100 * (snap.price_spread > 0).mean(), 4),
            }
        ]
    )

    observed_dates = pd.DatetimeIndex(sorted(pd.to_datetime(core.ts.dt.date.unique())))
    full_dates = pd.date_range(observed_dates.min(), observed_dates.max(), freq="D")
    missing_dates = [d.strftime("%Y-%m-%d") for d in full_dates.difference(observed_dates)]

    issues = pd.DataFrame(
        [
            ("DR-001", "Timestamp unit", "timestamp is 10-digit epoch seconds", "HIGH", "FIXABLE", "Parse with unit='s'; canonicalize timezone and recompute hour/day/month.", "None after validation."),
            ("DR-002", "Timestamp representation mismatch", f"{int((mismatch_min > 5).sum())} rows >5 min; max {mismatch_min.max():.2f} min", "LOW", "FIXABLE", "Use timestamp as canonical; allow 6-minute validation tolerance; keep datetime for QA only.", "Second-level query jitter remains."),
            ("DR-003", "Missing price", "55,095 rows; all Uber Taxi", "HIGH", "PARTIAL", "Exclude Uber Taxi from supervised price training; retain for availability analysis.", "Original Taxi prices cannot be recovered."),
            ("DR-004", "Uber multiplier has no variation", "385,663 Uber rows; multiplier always 1", "HIGH", "NOT_FIXABLE", "Do not train explicit Uber multiplier; predict price or construct separately validated pseudo-multiplier.", "No ground-truth Uber surge label."),
            ("DR-005", "Lyft surge class imbalance", "Most Lyft rows have multiplier=1; surge tail is sparse", "MEDIUM", "MITIGATABLE", "Report tail metrics; use class/sample weighting or ordinal model if modeling discrete levels.", "Very rare 2.5/3.0 levels remain uncertain."),
            ("DR-006", "Irregular sampling", f"Median gap {overall_gaps.median():.2f} min; p90 {overall_gaps.quantile(.9):.2f} min", "HIGH", "MITIGATABLE", "Use event-time lags, observation_age and age-bucket evaluation; never treat missing intervals as labels.", "Cannot create unobserved true quotes."),
            ("DR-007", "Missing calendar dates", f"{len(missing_dates)} dates absent: {', '.join(missing_dates)}", "MEDIUM", "NOT_FIXABLE", "Preserve gaps; split chronologically by contiguous blocks; do not interpolate across missing days.", "Only 17 observed dates."),
            ("DR-008", "Natural-key quote collisions", f"{duplicate_summary['duplicate_natural_key_groups']} timestamp/route/service groups; many price/distance conflicts", "MEDIUM", "FIXABLE", "Keep raw IDs; for history build 5-minute snapshot aggregates with count/spread/distance statistics.", "Aggregation loses quote-level detail; retain raw layer."),
            ("DR-009", "Potential weather leakage", "Daily high/low/max and event-time provenance are unclear", "HIGH", "FIXABLE", "Apply feature whitelist; exclude daily/future summaries pending provenance; only as-of joins.", "Historical POC may use fewer weather features."),
            ("DR-010", "Short and narrow domain", "Boston only, 72 routes, 17 observed dates", "HIGH", "NOT_FIXABLE", "Limit claims to methodology/POC; use temporal slices and abstention for sparse contexts.", "No production generalization proof."),
            ("DR-011", "No Vietnam market data", "Source market is Boston", "HIGH", "NOT_FIXABLE", "Do not claim Vietnam calibration; require future local collection before deployment.", "Domain shift cannot be estimated from this dataset."),
        ],
        columns=["issue_id", "issue", "evidence", "severity", "fixability", "treatment", "residual_limitation"],
    )

    policy = feature_policy(columns)

    col_profile.to_csv(OUT / "column_profile_v1.0.0.csv", index=False)
    service_profile.to_csv(OUT / "service_profile_v1.0.0.csv", index=False)
    daily.to_csv(OUT / "daily_coverage_v1.0.0.csv", index=False)
    gap_profile.to_csv(OUT / "gap_profile_v1.0.0.csv", index=False)
    snapshot_profile.to_csv(OUT / "snapshot_profile_v1.0.0.csv", index=False)
    issues.to_csv(OUT / "issue_register_v1.0.0.csv", index=False)
    policy.to_csv(OUT / "feature_policy_v1.0.0.csv", index=False)

    summary = {
        "audit_version": AUDIT_VERSION,
        "policy_version": POLICY_VERSION,
        "source_file": SOURCE.name,
        "source_sha256": sha256(SOURCE),
        "rows": total_rows,
        "columns": len(columns),
        "unique_ids": int(core.id.nunique()),
        "timestamp_start": ts.min().isoformat(),
        "timestamp_end": ts.max().isoformat(),
        "observed_dates": int(len(observed_dates)),
        "missing_dates": missing_dates,
        "routes": int(core.groupby(["source", "destination"]).ngroups),
        "services_total": int(core.groupby(["cab_type", "name"]).ngroups),
        "priced_series": int(valid.groupby(keys).ngroups),
        "price_missing": int(core.price.isna().sum()),
        "price_missing_pct": round(100 * core.price.isna().mean(), 4),
        "timestamp_mismatch_gt_5min": int((mismatch_min > 5).sum()),
        "timestamp_mismatch_max_min": round(float(mismatch_min.max()), 4),
        "gap_minutes": q(overall_gaps),
        "duplicate_summary": duplicate_summary,
        "snapshot_summary": snapshot_profile.iloc[0].to_dict(),
        "readiness_verdict": "READY_FOR_POC_AFTER_MANDATORY_TRANSFORMS; NOT_READY_FOR_VIETNAM_PRODUCTION_VALIDATION",
    }
    (OUT / "audit_summary_v1.0.0.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# Data Readiness Audit Report v{AUDIT_VERSION}

## Kết luận

Dataset full **đủ để làm EDA và POC forecasting sau các biến đổi bắt buộc**, nhưng **không đủ để xác nhận production tại Việt Nam**.

## Phạm vi và version

- Input: `{SOURCE.name}`
- Source SHA-256: `{summary['source_sha256']}`
- Audit version: `{AUDIT_VERSION}`
- Feature policy version: `{POLICY_VERSION}`
- Source file không bị chỉnh sửa.

## Số liệu chính

- {total_rows:,} dòng, {len(columns)} cột, {core.id.nunique():,} ID duy nhất.
- Thời gian: {ts.min().strftime('%Y-%m-%d')} đến {ts.max().strftime('%Y-%m-%d')}; {len(observed_dates)} ngày có dữ liệu.
- 72 route pairs; 13 provider-service combinations; 864 priced route-service series.
- 55,095 price bị thiếu ({100 * core.price.isna().mean():.2f}%), toàn bộ là Uber Taxi.
- Median observation gap: {overall_gaps.median():.2f} phút; p90: {overall_gaps.quantile(.9):.2f} phút; p95: {overall_gaps.quantile(.95):.2f} phút.
- Uber multiplier luôn bằng 1; Lyft có 7 mức từ 1 đến 3.
- {duplicate_summary['duplicate_natural_key_groups']:,} natural-key collision groups; đây không phải exact duplicates vì ID duy nhất và nhiều nhóm khác distance/price.

## Xử lý bắt buộc trước EDA/modeling

1. Parse `timestamp` bằng epoch seconds và dùng làm canonical event time.
2. Loại Uber Taxi khỏi supervised price target; không impute 55,095 giá không tồn tại.
3. Không train explicit Uber multiplier; chỉ model price hoặc pseudo-multiplier được định nghĩa riêng.
4. Giữ raw quote layer; tạo history snapshot 5 phút bằng median/min/max/count/spread và distance statistics.
5. Dùng event-time lag, `observation_age` và đánh giá theo age bucket; không forward-fill label.
6. Áp dụng feature policy để loại daily/future weather fields có nguy cơ leakage.
7. Chia train/calibration/test hoàn toàn theo thời gian và tôn trọng các ngày bị thiếu.

## Vấn đề không thể sửa từ dataset

- Không có dữ liệu thị trường Việt Nam.
- Chỉ có 17 ngày tại Boston và 72 route pairs.
- Không có ground-truth Uber surge multiplier.
- Không thể khôi phục các quote không được quan sát trong sampling gaps.

## Data Readiness Audit bao phủ đến đâu?

Audit v1.0.0 bao phủ schema, missingness, timestamp, service/target validity, collision/aggregation, sampling gap, calendar coverage, leakage policy và chronological split readiness. Nó **xử lý hoặc định nghĩa mitigation cho mọi lỗi có thể xử lý bằng data pipeline**, đồng thời ghi rõ các giới hạn không thể sửa trong `issue_register_v1.0.0.csv`.
"""
    (OUT / "REPORT_v1.0.0.md").write_text(report, encoding="utf-8")

    output_files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest_v1.0.0.json")
    manifest = {
        "audit_version": AUDIT_VERSION,
        "policy_version": POLICY_VERSION,
        "source": {"path": str(SOURCE), "sha256": summary["source_sha256"], "rows": total_rows, "columns": len(columns)},
        "outputs": [{"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in output_files],
    }
    (OUT / "manifest_v1.0.0.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
