from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr


STUDY_VERSION = "1.0.0"
EDA_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "artifacts" / "eda" / f"v{EDA_VERSION}"
MODEL_DIR = ROOT / "artifacts" / "modeling_data" / "v1.0.0"
OUT = ROOT / "artifacts" / "relation_study" / f"v{STUDY_VERSION}"
GEOCODE_PATH = OUT / "location_geocodes_v1.0.0.csv"

PRICE_PATH = MODEL_DIR / "price_modeling_v1.0.0.parquet"
MULTIPLIER_PATH = MODEL_DIR / "lyft_multiplier_modeling_v1.0.0.parquet"
FEATURE_CONTRACT_PATH = MODEL_DIR / "feature_contract_v1.0.0.json"
CENTER_NAME = "Boston City Hall"
CENTER_QUERY = "Boston City Hall, Boston, MA"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
LOCATIONS = [
    "Back Bay",
    "Beacon Hill",
    "Boston University",
    "Fenway",
    "Financial District",
    "Haymarket Square",
    "North End",
    "North Station",
    "Northeastern University",
    "South Station",
    "Theatre District",
    "West End",
]
QUERY_OVERRIDES = {
    "Fenway": "Fenway-Kenmore, Boston, MA",
    "Theatre District": "Theater District, Boston, Massachusetts",
}
BOSTON_VIEWBOX = "-71.1912,42.2279,-70.9239,42.3969"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_geocodes() -> pd.DataFrame:
    if GEOCODE_PATH.exists():
        cached = pd.read_csv(GEOCODE_PATH)
        inside_boston = cached["latitude"].between(42.22, 42.40) & cached["longitude"].between(-71.20, -70.92)
        if len(cached) == 13 and inside_boston.all():
            return cached

    rows = []
    queries = [(CENTER_NAME, CENTER_QUERY, "center_proxy")] + [
        (location, QUERY_OVERRIDES.get(location, f"{location}, Boston, Massachusetts, USA"), "dataset_location")
        for location in LOCATIONS
    ]
    headers = {"User-Agent": "VSF-GSM-research-relation-study/1.0"}
    for index, (location, query, role) in enumerate(queries):
        response = requests.get(
            NOMINATIM_URL,
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": "us",
                "addressdetails": 1,
                "viewbox": BOSTON_VIEWBOX,
                "bounded": 1,
            },
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            raise ValueError(f"No geocode result for {query}")
        result = results[0]
        rows.append(
            {
                "location": location,
                "role": role,
                "query": query,
                "latitude": float(result["lat"]),
                "longitude": float(result["lon"]),
                "osm_type": result.get("osm_type"),
                "osm_id": result.get("osm_id"),
                "result_type": result.get("type"),
                "display_name": result.get("display_name"),
                "provider": "OpenStreetMap Nominatim",
                "provider_url": "https://nominatim.org/release-docs/latest/api/Search/",
                "attribution": "OpenStreetMap contributors, ODbL 1.0",
            }
        )
        if index < len(queries) - 1:
            time.sleep(1.1)
    frame = pd.DataFrame(rows)
    frame.to_csv(GEOCODE_PATH, index=False)
    return frame


def haversine_km(lat1: float | pd.Series, lon1: float | pd.Series, lat2: float, lon2: float) -> pd.Series:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def destination_profiles(price_train: pd.DataFrame, multiplier_train: pd.DataFrame, geocodes: pd.DataFrame) -> pd.DataFrame:
    price = price_train.copy()
    price["distance_band_half_mile"] = np.floor(price["distance_median"] * 2) / 2
    baseline = price.groupby(
        ["cab_type", "name", "distance_band_half_mile"], observed=True
    )["target_price_median"].transform("median")
    price["adjusted_price"] = price["target_price_median"] - baseline
    price_profile = price.groupby("destination", observed=True).agg(
        price_rows=("target_price_median", "size"),
        raw_median_price=("target_price_median", "median"),
        mean_adjusted_price=("adjusted_price", "mean"),
        median_adjusted_price=("adjusted_price", "median"),
        median_trip_distance_miles=("distance_median", "median"),
    )

    multiplier_profile = multiplier_train.assign(
        surge=multiplier_train["target_multiplier_median"].gt(1)
    ).groupby("destination", observed=True).agg(
        multiplier_rows=("target_multiplier_median", "size"),
        mean_multiplier=("target_multiplier_median", "mean"),
        surge_rate_pct=("surge", lambda values: 100 * values.mean()),
    )

    coordinates = geocodes.loc[geocodes["role"].eq("dataset_location")].set_index("location")
    center = geocodes.loc[geocodes["role"].eq("center_proxy")].iloc[0]
    profile = price_profile.join(multiplier_profile, how="outer").join(
        coordinates[["latitude", "longitude", "osm_type", "osm_id", "display_name"]], how="left"
    ).reset_index()
    profile["distance_to_city_hall_km"] = haversine_km(
        profile["latitude"], profile["longitude"], center["latitude"], center["longitude"]
    )
    return profile.sort_values("distance_to_city_hall_km").reset_index(drop=True)


def relation_metrics(profile: pd.DataFrame) -> pd.DataFrame:
    relations = [
        ("centrality_vs_raw_price", "distance_to_city_hall_km", "raw_median_price"),
        ("centrality_vs_adjusted_price", "distance_to_city_hall_km", "mean_adjusted_price"),
        ("centrality_vs_lyft_surge", "distance_to_city_hall_km", "surge_rate_pct"),
    ]
    rows = []
    for name, feature, target in relations:
        valid = profile[[feature, target]].dropna()
        result = spearmanr(valid[feature], valid[target])
        rows.append(
            {
                "relation": name,
                "unit_of_analysis": "12 destination representatives",
                "feature": feature,
                "target": target,
                "n": len(valid),
                "spearman_rho": float(result.statistic),
                "p_value_exploratory": float(result.pvalue),
                "interpretation": (
                    "Negative rho means destinations closer to City Hall tend to have higher target values."
                    if result.statistic < 0
                    else "Positive rho means destinations farther from City Hall tend to have higher target values."
                ),
            }
        )
    return pd.DataFrame(rows)


def evidence_and_coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    findings = json.loads((EDA_DIR / "key_findings_v1.0.0.json").read_text(encoding="utf-8"))
    weather = pd.read_csv(EDA_DIR / "weather_associations_v1.0.0.csv")
    price_time = pd.read_csv(EDA_DIR / "price_time_profile_v1.0.0.csv")
    multiplier_time = pd.read_csv(EDA_DIR / "multiplier_time_profile_v1.0.0.csv")
    price_routes = pd.read_csv(EDA_DIR / "price_route_profile_v1.0.0.csv")
    multiplier_routes = pd.read_csv(EDA_DIR / "multiplier_route_profile_v1.0.0.csv")
    price_freshness = pd.read_csv(EDA_DIR / "price_freshness_profile_v1.0.0.csv")

    hourly_price = price_time.groupby(["cab_type", "event_hour_local"], observed=True).apply(
        lambda frame: np.average(frame["mean_adjusted_price"], weights=frame["rows"]), include_groups=False
    )
    hourly_surge = multiplier_time.groupby("event_hour_local", observed=True).apply(
        lambda frame: np.average(frame["surge_rate_pct"], weights=frame["rows"]), include_groups=False
    )
    destination_price = price_routes.groupby("destination", observed=True).apply(
        lambda frame: np.average(frame["mean_adjusted_price"], weights=frame["rows"]), include_groups=False
    )
    destination_surge = multiplier_routes.groupby("destination", observed=True).apply(
        lambda frame: np.average(frame["surge_rate_pct"], weights=frame["rows"]), include_groups=False
    )
    test_freshness = price_freshness.loc[price_freshness["data_split"].eq("test")]

    evidence = pd.DataFrame(
        [
            ("service_distance", "price", "STRONG", "Service median price spans $7-$30; route-service median baseline test MAE is $1.301.", "price_service_profile; baseline_diagnostics"),
            ("delayed_price_history", "price", "STRONG_BUT_NOT_STANDALONE", f"Lag1 rho={findings['price_lag1_spearman_train']:.3f}; persistence MAE $1.956 is worse than structural baseline.", "key_findings; baseline_diagnostics"),
            ("time", "price", "SMALL", f"Adjusted hourly mean range={hourly_price.max()-hourly_price.min():.3f} USD across provider-hour profiles.", "price_time_profile"),
            ("time", "multiplier", "SMALL", f"Hourly Lyft surge-rate range={hourly_surge.max()-hourly_surge.min():.3f} percentage points.", "multiplier_time_profile"),
            ("location_category", "price", "SMALL_TO_MODERATE", f"Destination adjusted-price range={destination_price.max()-destination_price.min():.3f} USD.", "price_route_profile"),
            ("location_category", "multiplier", "SMALL", f"Destination surge-rate range={destination_surge.max()-destination_surge.min():.3f} percentage points.", "multiplier_route_profile"),
            ("weather", "price", "NEGLIGIBLE_MARGINAL", f"Max adjusted |rho|={weather['price_adjusted_spearman'].abs().max():.4f}.", "weather_associations"),
            ("weather", "multiplier", "NEGLIGIBLE_MARGINAL", f"Max surge-indicator |rho|={weather['surge_indicator_spearman'].abs().max():.4f}.", "weather_associations"),
            ("delayed_multiplier_history", "multiplier", "WEAK", f"Lag1 rho={findings['multiplier_lag1_spearman_train']:.3f}; constant-1 beats lag1 persistence on test MAE.", "key_findings; baseline_diagnostics"),
            ("observation_age", "price", "SMALL_DIRECT_BUT_OPERATIONAL", f"Test persistence MAE range={test_freshness['persistence_mae'].max()-test_freshness['persistence_mae'].min():.3f} USD; age remains needed for reliability gating.", "price_freshness_profile"),
        ],
        columns=["feature_group", "target", "evidence_strength", "result", "evidence_source"],
    )

    coverage = pd.DataFrame(
        [
            ("Base fare structure", "provider, service, route, direction, distance", "price", "COVERED", "Mandatory"),
            ("Recent competitor state", "lag1/2/3 price, rolling mean/std, price delta", "price", "COVERED", "Mandatory"),
            ("Observation reliability", "observation age, age bucket, history count", "price; multiplier", "COVERED", "Mandatory for uncertainty/gating"),
            ("Periodic demand", "hour, weekday, cyclic encodings, hour-by-weekday", "price; multiplier", "COVERED", "Conditional interaction"),
            ("Location", "source, destination, route, City Hall distance proxy", "price; multiplier", "COVERED_WITH_PROXY", "Categorical mandatory; centrality ablation"),
            ("Weather", "14 current-weather numerics, short summary", "price; multiplier", "COVERED", "Low-priority grouped ablation"),
            ("Multiplier tail", "median multiplier target, surge>1 auxiliary target", "multiplier", "COVERED", "Tail metrics mandatory"),
            ("Unavailable market state", "traffic, events, driver supply, demand, ETA", "price; multiplier", "NOT_AVAILABLE", "Dataset limitation; future collection"),
            ("Vietnam transfer", "local routes, competitors, weather, events", "price; multiplier", "NOT_AVAILABLE", "Required before production validation"),
        ],
        columns=["mechanism", "feature_groups", "targets", "coverage", "decision"],
    )
    return evidence, coverage


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geocodes = fetch_geocodes()
    price = pd.read_parquet(PRICE_PATH)
    multiplier = pd.read_parquet(MULTIPLIER_PATH)
    price_train = price.loc[price["data_split"].eq("train")].copy()
    multiplier_train = multiplier.loc[multiplier["data_split"].eq("train")].copy()

    destination_profile = destination_profiles(price_train, multiplier_train, geocodes)
    centrality = relation_metrics(destination_profile)
    evidence, coverage = evidence_and_coverage()
    feature_contract = json.loads(FEATURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    categorical_feature_count = len(feature_contract["categorical_features"])
    numeric_feature_count = len(feature_contract["numeric_features"])
    candidate_feature_count = categorical_feature_count + numeric_feature_count

    destination_profile.to_csv(OUT / "destination_centrality_profile_v1.0.0.csv", index=False)
    centrality.to_csv(OUT / "centrality_relations_v1.0.0.csv", index=False)
    evidence.to_csv(OUT / "relation_evidence_summary_v1.0.0.csv", index=False)
    coverage.to_csv(OUT / "feature_coverage_matrix_v1.0.0.csv", index=False)

    adjusted = centrality.loc[centrality["relation"].eq("centrality_vs_adjusted_price")].iloc[0]
    surge = centrality.loc[centrality["relation"].eq("centrality_vs_lyft_surge")].iloc[0]
    raw = centrality.loc[centrality["relation"].eq("centrality_vs_raw_price")].iloc[0]
    centrality_decision = (
        "KEEP_CONDITIONAL_ABLATION" if abs(adjusted["spearman_rho"]) >= 0.30 else "LOW_PRIORITY_PROXY"
    )
    summary = {
        "relation_study_version": STUDY_VERSION,
        "eda_version": EDA_VERSION,
        "center_proxy": CENTER_NAME,
        "center_proxy_basis": "Boston.gov describes City Hall as being in the heart of downtown Boston.",
        "destinations": len(destination_profile),
        "centrality_vs_raw_price_spearman": float(raw["spearman_rho"]),
        "centrality_vs_adjusted_price_spearman": float(adjusted["spearman_rho"]),
        "centrality_vs_adjusted_price_p_exploratory": float(adjusted["p_value_exploratory"]),
        "centrality_vs_lyft_surge_spearman": float(surge["spearman_rho"]),
        "centrality_feature_decision": centrality_decision,
        "candidate_feature_columns": candidate_feature_count,
        "categorical_feature_columns": categorical_feature_count,
        "numeric_feature_columns": numeric_feature_count,
        "recommended_feature_groups": int((coverage["coverage"] != "NOT_AVAILABLE").sum()),
        "missing_feature_groups": int((coverage["coverage"] == "NOT_AVAILABLE").sum()),
        "completion": "PART_I_RELATION_STUDY_COMPLETE_FOR_AVAILABLE_DATA",
    }
    (OUT / "relation_study_summary_v1.0.0.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# Key Feature–Target Relation Study v{STUDY_VERSION}

## Kết luận

Phần i đã hoàn thành cho các feature hiện có. **Service, route và distance** là nhóm liên hệ mạnh nhất với price; **recent price history** có signal cao nhưng không nên dùng riêng; **time và location category** có signal nhỏ; **weather có marginal association gần 0**. Lyft multiplier rất mất cân bằng và delayed multiplier history yếu.

## Quan hệ với price

| Nhóm feature | Kết quả ngắn | Quyết định |
|---|---|---|
| Service, route, distance | Service median $7–$30; route-service baseline test MAE $1.301 | Giữ bắt buộc |
| Delayed price history | Lag1 rho 0.949, nhưng persistence MAE $1.956 | Giữ và kết hợp structural features |
| Time | Adjusted hourly effect nhỏ | Giữ hour/weekday + cyclic interaction |
| Location category | Destination adjusted-price có khác biệt | Giữ source/destination/route |
| Weather | Adjusted abs(rho) tối đa <0.01 | Ưu tiên thấp; grouped ablation |
| Observation age | Direct effect nhỏ nhưng mô tả độ cũ của tín hiệu | Giữ cho uncertainty/gating |

## Quan hệ với Lyft multiplier

| Nhóm feature | Kết quả ngắn | Quyết định |
|---|---|---|
| Target distribution | 92.63% train snapshots bằng 1.0 | Regression + surge>1 auxiliary task |
| Delayed multiplier | Lag1 rho 0.036; constant-1 tốt hơn persistence | Không dựa vào lag1 đơn lẻ |
| Time/location | Surge-rate thay đổi nhỏ theo hour/destination | Giữ để ablation |
| Weather | Surge-indicator abs(rho) tối đa <0.01 | Ưu tiên thấp |

## Bổ sung “gần trung tâm có đắt hơn không?”

- Proxy trung tâm: **Boston City Hall**, vì Boston.gov mô tả City Hall nằm ở “heart of downtown Boston”.
- 12 source/destination được geocode thành representative points bằng OpenStreetMap Nominatim; đây không phải trip endpoint chính xác.
- Distance-to-center vs raw median price: Spearman rho **{raw['spearman_rho']:.3f}**.
- Distance-to-center vs price đã điều chỉnh service/distance: rho **{adjusted['spearman_rho']:.3f}**, exploratory p={adjusted['p_value_exploratory']:.3f}.
- Distance-to-center vs Lyft surge rate: rho **{surge['spearman_rho']:.3f}**.

Kết luận: `distance_to_city_hall_km` được xếp **{centrality_decision}**. Kết quả này chỉ kiểm tra một proxy trên 12 địa điểm, không chứng minh quan hệ nhân quả.

## Feature recommendations đã đủ chưa?

Không đánh giá chỉ bằng số lượng cột. Modeling contract hiện có **{candidate_feature_count} candidate features** ({categorical_feature_count} categorical, {numeric_feature_count} numeric), bao phủ **7 mechanism groups khả dụng**: fare structure, recent competitor state, staleness/reliability, periodic demand, location, weather và multiplier tail. Đây là mức vừa phải cho tabular modeling; chưa phải danh sách feature cuối. Hai nhóm còn thiếu do dataset không có là **traffic/events/supply-demand** và **Vietnam market context**.

Chứng minh feature set bằng hai tầng:

1. **Relation evidence:** đã hoàn thành trong report này và các bảng EDA.
2. **Out-of-time ablation ở phần ii:** so sánh model theo từng block feature; chỉ giữ block nếu cải thiện test/calibration metrics và không làm uncertainty calibration xấu đi.

Vì vậy recommendations hiện **đủ để bắt đầu modeling**, nhưng feature cuối cùng chỉ được chốt sau ablation.

## Nguồn location

- [Boston City Hall – official Boston.gov](https://www.boston.gov/departments/mayors-office/contact-boston-city-hall)
- [Nominatim Search API](https://nominatim.org/release-docs/latest/api/Search/)
- Geodata attribution: OpenStreetMap contributors, ODbL 1.0.
"""
    (OUT / "RELATION_STUDY_REPORT_v1.0.0.md").write_text(report, encoding="utf-8")

    validations = pd.DataFrame(
        [
            ("all_12_locations_geocoded", len(geocodes.loc[geocodes["role"].eq("dataset_location")]) == 12, len(geocodes.loc[geocodes["role"].eq("dataset_location")]), 12),
            ("center_proxy_geocoded", len(geocodes.loc[geocodes["role"].eq("center_proxy")]) == 1, len(geocodes.loc[geocodes["role"].eq("center_proxy")]), 1),
            ("destination_profile_complete", len(destination_profile) == 12 and destination_profile["distance_to_city_hall_km"].notna().all(), len(destination_profile), 12),
            ("centrality_relations_complete", len(centrality) == 3 and centrality["spearman_rho"].notna().all(), len(centrality), 3),
            ("required_relation_groups_covered", {"weather", "time", "location_category", "service_distance", "delayed_price_history"}.issubset(set(evidence["feature_group"])), sorted(evidence["feature_group"].unique()), "required groups"),
            ("feature_coverage_has_decisions", coverage["decision"].notna().all(), int(coverage["decision"].isna().sum()), 0),
        ],
        columns=["check", "passed", "actual", "expected"],
    )
    validations.to_csv(OUT / "validation_results_v1.0.0.csv", index=False)
    if not validations["passed"].all():
        raise AssertionError(validations.loc[~validations["passed"]].to_dict("records"))

    outputs = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "manifest_v1.0.0.json")
    script = Path(__file__).resolve()
    manifest = {
        "relation_study_version": STUDY_VERSION,
        "eda_version": EDA_VERSION,
        "inputs": [
            {"name": PRICE_PATH.name, "sha256": sha256(PRICE_PATH)},
            {"name": MULTIPLIER_PATH.name, "sha256": sha256(MULTIPLIER_PATH)},
            {"name": "key_findings_v1.0.0.json", "sha256": sha256(EDA_DIR / "key_findings_v1.0.0.json")},
        ],
        "script": {"path": str(script), "sha256": sha256(script)},
        "outputs": [
            {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    (OUT / "manifest_v1.0.0.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
