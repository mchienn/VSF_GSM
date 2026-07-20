from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EDA_VERSION = "1.0.0"
MODELING_DATA_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "modeling_data" / f"v{MODELING_DATA_VERSION}"
OUT = ROOT / "artifacts" / "eda" / f"v{EDA_VERSION}"
FIG = OUT / "figures"

PRICE_PATH = MODEL_DIR / "price_modeling_v1.0.0.parquet"
MULTIPLIER_PATH = MODEL_DIR / "lyft_multiplier_modeling_v1.0.0.parquet"
CANONICAL_PATH = MODEL_DIR / "canonical_quotes_v1.0.0.parquet"

WEATHER = [
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
SPLIT_ORDER = ["train", "calibration", "test"]
AGE_ORDER = ["00-15", "15-30", "30-60", "60-120", "120+"]
COLORS = {"Uber": "#2F6B9A", "Lyft": "#B14E7A", "price": "#2F6B9A", "multiplier": "#B14E7A"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", metadata={"Software": "VSF_GSM EDA v1.0.0"})
    plt.close(fig)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x.loc[valid].nunique() < 2 or y.loc[valid].nunique() < 2:
        return float("nan")
    return float(spearmanr(x.loc[valid], y.loc[valid]).statistic)


def add_eda_residuals(price_train: pd.DataFrame) -> pd.DataFrame:
    frame = price_train.copy()
    frame["distance_band_half_mile"] = np.floor(frame["distance_median"] * 2) / 2

    series_distance_baseline = frame.groupby(
        ["series_id", "distance_band_half_mile"], observed=True
    )["target_price_median"].transform("median")
    frame["price_residual_within_route_service"] = frame["target_price_median"] - series_distance_baseline

    service_distance_baseline = frame.groupby(
        ["cab_type", "name", "distance_band_half_mile"], observed=True
    )["target_price_median"].transform("median")
    frame["price_residual_service_distance"] = frame["target_price_median"] - service_distance_baseline
    frame["price_per_mile_eda"] = frame["target_price_median"] / frame["distance_median"].clip(lower=0.25)
    frame["price_changed_from_lag1"] = frame["target_price_median"].ne(frame["lag1_price_median"])
    frame["absolute_price_change"] = (frame["target_price_median"] - frame["lag1_price_median"]).abs()
    return frame


def target_and_service_profiles(price: pd.DataFrame, multiplier: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_profile = (
        price.groupby(["data_split", "cab_type", "name"], observed=True)
        .agg(
            rows=("target_price_median", "size"),
            mean_price=("target_price_median", "mean"),
            median_price=("target_price_median", "median"),
            p10_price=("target_price_median", lambda x: x.quantile(0.10)),
            p90_price=("target_price_median", lambda x: x.quantile(0.90)),
            median_distance=("distance_median", "median"),
            median_price_per_mile=("price_per_mile_eda", "median"),
        )
        .reset_index()
    )
    multiplier_profile = (
        multiplier.groupby(["data_split", "target_multiplier_median"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    multiplier_profile["share_pct"] = (
        100
        * multiplier_profile["rows"]
        / multiplier_profile.groupby("data_split", observed=True)["rows"].transform("sum")
    )
    return price_profile, multiplier_profile


def time_profiles(price_train: pd.DataFrame, multiplier_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_time = (
        price_train.groupby(["cab_type", "event_weekday_local", "event_hour_local"], observed=True)
        .agg(
            rows=("target_price_median", "size"),
            median_price=("target_price_median", "median"),
            median_adjusted_price=("price_residual_within_route_service", "median"),
            mean_adjusted_price=("price_residual_within_route_service", "mean"),
        )
        .reset_index()
    )
    multiplier_time = multiplier_train.assign(
        surge=multiplier_train["target_multiplier_median"].gt(1)
    ).groupby(["event_weekday_local", "event_hour_local"], observed=True).agg(
        rows=("target_multiplier_median", "size"),
        mean_multiplier=("target_multiplier_median", "mean"),
        surge_rate_pct=("surge", lambda x: 100 * x.mean()),
    ).reset_index()
    return price_time, multiplier_time


def location_profiles(price_train: pd.DataFrame, multiplier_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_location = (
        price_train.groupby(["source", "destination", "cab_type"], observed=True)
        .agg(
            rows=("target_price_median", "size"),
            median_price=("target_price_median", "median"),
            median_distance=("distance_median", "median"),
            median_adjusted_price=("price_residual_service_distance", "median"),
            mean_adjusted_price=("price_residual_service_distance", "mean"),
        )
        .reset_index()
    )
    multiplier_location = multiplier_train.assign(
        surge=multiplier_train["target_multiplier_median"].gt(1)
    ).groupby(["source", "destination"], observed=True).agg(
        rows=("target_multiplier_median", "size"),
        mean_multiplier=("target_multiplier_median", "mean"),
        surge_rate_pct=("surge", lambda x: 100 * x.mean()),
    ).reset_index()
    return price_location, multiplier_location


def weather_associations(price_train: pd.DataFrame, multiplier_train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    surge = multiplier_train["target_multiplier_median"].gt(1).astype(int)
    for feature in WEATHER:
        price_valid = price_train[[feature, "target_price_median", "price_residual_within_route_service"]].dropna()
        multiplier_valid = multiplier_train[[feature, "target_multiplier_median"]].dropna()

        price_bins = pd.qcut(price_valid[feature], q=5, duplicates="drop")
        price_by_bin = price_valid.groupby(price_bins, observed=True)["price_residual_within_route_service"].median()
        price_q_delta = float(price_by_bin.iloc[-1] - price_by_bin.iloc[0]) if len(price_by_bin) >= 2 else float("nan")

        mult_bins = pd.qcut(multiplier_valid[feature], q=5, duplicates="drop")
        surge_aligned = surge.loc[multiplier_valid.index]
        surge_by_bin = surge_aligned.groupby(mult_bins, observed=True).mean() * 100
        surge_q_delta = float(surge_by_bin.iloc[-1] - surge_by_bin.iloc[0]) if len(surge_by_bin) >= 2 else float("nan")

        rows.append(
            {
                "feature": feature,
                "price_rows": len(price_valid),
                "price_raw_spearman": safe_spearman(price_valid[feature], price_valid["target_price_median"]),
                "price_adjusted_spearman": safe_spearman(price_valid[feature], price_valid["price_residual_within_route_service"]),
                "price_adjusted_q5_minus_q1": price_q_delta,
                "multiplier_rows": len(multiplier_valid),
                "multiplier_spearman": safe_spearman(multiplier_valid[feature], multiplier_valid["target_multiplier_median"]),
                "surge_indicator_spearman": safe_spearman(multiplier_valid[feature], surge_aligned),
                "surge_rate_q5_minus_q1_pp": surge_q_delta,
            }
        )
    return pd.DataFrame(rows).sort_values("price_adjusted_spearman", key=lambda s: s.abs(), ascending=False)


def weather_summary_profiles(price_train: pd.DataFrame, multiplier_train: pd.DataFrame) -> pd.DataFrame:
    price_summary = price_train.groupby("short_summary", observed=True).agg(
        price_rows=("target_price_median", "size"),
        median_price=("target_price_median", "median"),
        median_adjusted_price=("price_residual_within_route_service", "median"),
    )
    mult_summary = multiplier_train.assign(surge=multiplier_train["target_multiplier_median"].gt(1)).groupby(
        "short_summary", observed=True
    ).agg(
        multiplier_rows=("target_multiplier_median", "size"),
        mean_multiplier=("target_multiplier_median", "mean"),
        surge_rate_pct=("surge", lambda x: 100 * x.mean()),
    )
    return price_summary.join(mult_summary, how="outer").reset_index()


def freshness_profiles(price: pd.DataFrame, multiplier: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_freshness = (
        price.groupby(["data_split", "observation_age_bucket"], observed=True)
        .agg(
            rows=("target_price_median", "size"),
            median_age_minutes=("observation_age_minutes", "median"),
            persistence_mae=("absolute_price_change", "mean"),
            persistence_median_ae=("absolute_price_change", "median"),
            price_change_rate_pct=("price_changed_from_lag1", lambda x: 100 * x.mean()),
        )
        .reset_index()
    )
    mult = multiplier.assign(
        multiplier_changed=multiplier["target_multiplier_median"].ne(multiplier["lag1_multiplier_median"]),
        absolute_multiplier_change=(multiplier["target_multiplier_median"] - multiplier["lag1_multiplier_median"]).abs(),
        surge=multiplier["target_multiplier_median"].gt(1),
    )
    multiplier_freshness = (
        mult.groupby(["data_split", "observation_age_bucket"], observed=True)
        .agg(
            rows=("target_multiplier_median", "size"),
            median_age_minutes=("observation_age_minutes", "median"),
            persistence_mae=("absolute_multiplier_change", "mean"),
            multiplier_change_rate_pct=("multiplier_changed", lambda x: 100 * x.mean()),
            surge_rate_pct=("surge", lambda x: 100 * x.mean()),
        )
        .reset_index()
    )
    return price_freshness, multiplier_freshness


def split_drift(price: pd.DataFrame, multiplier: pd.DataFrame) -> pd.DataFrame:
    price_drift = price.groupby("data_split", observed=True).agg(
        rows=("target_price_median", "size"),
        target_mean=("target_price_median", "mean"),
        target_median=("target_price_median", "median"),
        target_p90=("target_price_median", lambda x: x.quantile(0.9)),
        median_lag1=("lag1_price_median", "median"),
        median_age_minutes=("observation_age_minutes", "median"),
    ).reset_index()
    price_drift.insert(0, "target", "price")
    multiplier_drift = multiplier.assign(
        surge=multiplier["target_multiplier_median"].gt(1)
    ).groupby("data_split", observed=True).agg(
        rows=("target_multiplier_median", "size"),
        target_mean=("target_multiplier_median", "mean"),
        target_median=("target_multiplier_median", "median"),
        target_p90=("target_multiplier_median", lambda x: x.quantile(0.9)),
        median_lag1=("lag1_multiplier_median", "median"),
        median_age_minutes=("observation_age_minutes", "median"),
        surge_rate_pct=("surge", lambda x: 100 * x.mean()),
    ).reset_index()
    multiplier_drift.insert(0, "target", "lyft_multiplier")
    return pd.concat([price_drift, multiplier_drift], ignore_index=True, sort=False)


def baseline_diagnostics(price: pd.DataFrame, multiplier: pd.DataFrame) -> pd.DataFrame:
    price_train = price.loc[price["data_split"].eq("train")]
    price_test = price.loc[price["data_split"].eq("test")].copy()
    global_price = price_train["target_price_median"].median()
    series_price = price_train.groupby("series_id", observed=True)["target_price_median"].median()
    price_test["series_train_median"] = price_test["series_id"].map(series_price).fillna(global_price)

    multiplier_test = multiplier.loc[multiplier["data_split"].eq("test")].copy()
    actual_surge = multiplier_test["target_multiplier_median"].gt(1)
    lag_surge = multiplier_test["lag1_multiplier_median"].gt(1)

    def mae(actual: pd.Series, predicted: pd.Series | float) -> float:
        return float(np.mean(np.abs(actual - predicted)))

    def rmse(actual: pd.Series, predicted: pd.Series | float) -> float:
        return float(np.sqrt(np.mean(np.square(actual - predicted))))

    tp = int((actual_surge & lag_surge).sum())
    fp = int((~actual_surge & lag_surge).sum())
    fn = int((actual_surge & ~lag_surge).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return pd.DataFrame(
        [
            ("price", "global_train_median", "MAE", mae(price_test["target_price_median"], global_price)),
            ("price", "global_train_median", "RMSE", rmse(price_test["target_price_median"], global_price)),
            ("price", "route_service_train_median", "MAE", mae(price_test["target_price_median"], price_test["series_train_median"])),
            ("price", "route_service_train_median", "RMSE", rmse(price_test["target_price_median"], price_test["series_train_median"])),
            ("price", "lag1_persistence", "MAE", mae(price_test["target_price_median"], price_test["lag1_price_median"])),
            ("price", "lag1_persistence", "RMSE", rmse(price_test["target_price_median"], price_test["lag1_price_median"])),
            ("lyft_multiplier", "constant_1", "MAE", mae(multiplier_test["target_multiplier_median"], 1.0)),
            ("lyft_multiplier", "lag1_persistence", "MAE", mae(multiplier_test["target_multiplier_median"], multiplier_test["lag1_multiplier_median"])),
            ("lyft_surge_binary", "always_no_surge", "accuracy", float((~actual_surge).mean())),
            ("lyft_surge_binary", "always_no_surge", "balanced_accuracy", 0.5),
            ("lyft_surge_binary", "lag1_surge", "accuracy", float((actual_surge == lag_surge).mean())),
            ("lyft_surge_binary", "lag1_surge", "precision", precision),
            ("lyft_surge_binary", "lag1_surge", "recall", recall),
            ("lyft_surge_binary", "lag1_surge", "f1", f1),
        ],
        columns=["target", "baseline", "metric", "value"],
    )


def numeric_feature_quality(price_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [column for column in price_train.columns if pd.api.types.is_numeric_dtype(price_train[column])]
    rows = []
    for column in numeric:
        series = price_train[column]
        stats_series = series.astype(float) if pd.api.types.is_bool_dtype(series) else series
        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "missing_pct": 100 * series.isna().mean(),
                "nunique": series.nunique(dropna=True),
                "mean": stats_series.mean(),
                "std": stats_series.std(),
                "p01": stats_series.quantile(0.01),
                "p50": stats_series.quantile(0.50),
                "p99": stats_series.quantile(0.99),
            }
        )
    quality = pd.DataFrame(rows)

    corr_cols = [
        column for column in WEATHER + ["distance_median", "observation_age_minutes", "lag1_price_median", "history_price_mean_last3"]
        if column in price_train.columns and price_train[column].nunique() > 1
    ]
    corr = price_train[corr_cols].corr(method="spearman")
    pairs = []
    for i, left in enumerate(corr_cols):
        for right in corr_cols[i + 1 :]:
            value = corr.loc[left, right]
            if abs(value) >= 0.80:
                pairs.append((left, right, value))
    correlated = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "spearman"])
    if not correlated.empty:
        correlated = correlated.sort_values("spearman", key=lambda s: s.abs(), ascending=False)
    return quality, correlated


def recommendations(weather: pd.DataFrame, correlated: pd.DataFrame) -> pd.DataFrame:
    top_weather_price = weather.reindex(weather["price_adjusted_spearman"].abs().sort_values(ascending=False).index).head(5)["feature"].tolist()
    top_weather_multiplier = weather.reindex(weather["surge_indicator_spearman"].abs().sort_values(ascending=False).index).head(5)["feature"].tolist()
    rows = [
        ("Delayed price history", "lag1/2/3 price, rolling mean/std, price delta", "KEEP_HIGH", "Central signal for delayed-observation forecasting; all values are shifted."),
        ("Freshness", "observation_age_minutes, log1p(age), age bucket interactions", "KEEP_HIGH_ENGINEER", "Persistence error changes with staleness; use nonlinear age and age-by-lag interactions."),
        ("Route and service", "cab_type, name, source, destination, direction-specific route", "KEEP_HIGH", "Large structural fare differences; use native categorical handling or leakage-safe encoding."),
        ("Distance", "distance_median plus service interaction", "KEEP_HIGH_ENGINEER", "Strong pricing basis; test nonlinear distance and distance-by-service interactions."),
        ("Time", "hour/weekday plus sin-cos and hour-by-weekday", "KEEP_ENGINEER", "Ride pricing is periodic; preserve cyclic representation and rush-hour interactions."),
        ("Weather price candidates", ", ".join(top_weather_price), "LOW_PRIORITY_ABLATION", "Adjusted associations are near zero in this dataset; retain only for a grouped out-of-time ablation."),
        ("Weather multiplier candidates", ", ".join(top_weather_multiplier), "LOW_PRIORITY_ABLATION", "Surge associations are near zero here despite domain literature; retain only for grouped out-of-time ablation."),
        ("Correlated weather", f"{len(correlated)} pairs with |rho| >= 0.80", "GROUP_OR_REGULARIZE", "Do not interpret individual correlated weather importances naively; cluster or use grouped permutation."),
        ("Lyft multiplier target", "5-minute median regression plus surge>1 auxiliary label", "MODEL_BOTH", "Snapshot median can create intermediate values; use regression for this target and binary surge diagnostics. Build a separate mode/raw-level target only if ordinal classification is required."),
        ("Price per mile", "price/distance", "EDA_ONLY", "Useful diagnostic but unstable for short trips and derived from current target, so never use as predictor."),
        ("Current bucket diagnostics", "quote_count, price min/max/spread", "EXCLUDE_MODEL", "They contain contemporaneous target information and are retained only in snapshot QA."),
        ("Daily/future weather and IDs", "daily extrema, event times, id/product_id", "EXCLUDE", "Leakage/provenance or high-cardinality trace-only fields."),
    ]
    return pd.DataFrame(rows, columns=["feature_group", "features", "decision", "rationale"])


def research_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Temporal evaluation", "Chronological train/calibration/test; EDA decisions use train only.", "Random splits can leak future structure and make lag forecasting look overly optimistic.", "https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html"),
            ("Lag engineering", "Use shifted lags and rolling history; never current target.", "Official forecasting example shifts target before rolling aggregation.", "https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html"),
            ("Cyclic time", "Retain hour/weekday and sin-cos; test hour-by-weekday interaction.", "Periodic encoding avoids artificial discontinuity at midnight/week boundary.", "https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html"),
            ("Categorical handling", "Keep route/service as categorical; do not pre-one-hot for CatBoost candidate.", "CatBoost natively transforms categorical features and advises against manual one-hot preprocessing.", "https://catboost.ai/docs/en/features/categorical-features"),
            ("Correlated features", "Report weather correlation clusters and use grouped/held-out importance later.", "Individual permutation importance is misleading when predictors are strongly correlated.", "https://scikit-learn.org/stable/modules/permutation_importance.html"),
            ("Surge feature scope", "Use recent spatiotemporal, environment and weather features for Lyft multiplier.", "Published ride-sourcing work models surge with recent urban, traffic and weather signals.", "https://www.sciencedirect.com/science/article/pii/S0968090X19301627"),
            ("Uncertainty path", "Reserve chronological calibration; compare split conformal baseline with time-series-aware EnbPI.", "EnbPI targets dynamic time series without requiring exchangeability and produces sequential intervals.", "https://proceedings.mlr.press/v139/xu21h.html"),
        ],
        columns=["topic", "decision", "reason", "source"],
    )


def plot_price_services(price_train: pd.DataFrame, profile: pd.DataFrame) -> None:
    train_profile = profile.loc[profile["data_split"].eq("train")].sort_values("median_price")
    labels = (train_profile["cab_type"] + " · " + train_profile["name"]).tolist()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [COLORS.get(provider, "#666666") for provider in train_profile["cab_type"]]
    ax.barh(labels, train_profile["median_price"], color=colors)
    ax.set_xlabel("Median 5-minute fare (USD)")
    ax.set_title("Train: median fare by provider and service")
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "01_price_by_service_v1.0.0.png")


def plot_multiplier_distribution(profile: pd.DataFrame) -> None:
    train = profile.loc[profile["data_split"].eq("train")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(train["target_multiplier_median"].astype(str), train["share_pct"], color=COLORS["Lyft"])
    ax.set_xlabel("Lyft multiplier")
    ax.set_ylabel("Share of train snapshots (%)")
    ax.set_title("Train: Lyft multiplier is strongly concentrated at 1.0")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "02_lyft_multiplier_distribution_v1.0.0.png")


def plot_hourly(price_time: pd.DataFrame, multiplier_time: pd.DataFrame) -> None:
    price_hour = price_time.groupby(["cab_type", "event_hour_local"], observed=True).apply(
        lambda x: np.average(x["mean_adjusted_price"], weights=x["rows"]), include_groups=False
    ).rename("adjusted_price").reset_index()
    multiplier_hour = multiplier_time.groupby("event_hour_local", observed=True).apply(
        lambda x: np.average(x["surge_rate_pct"], weights=x["rows"]), include_groups=False
    ).rename("surge_rate_pct").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for provider, frame in price_hour.groupby("cab_type", observed=True):
        axes[0].plot(frame["event_hour_local"], frame["adjusted_price"], marker="o", label=provider, color=COLORS[provider])
    axes[0].axhline(0, color="#777777", linewidth=1)
    axes[0].set(title="Time effect after route/service/distance adjustment", xlabel="Local hour", ylabel="Mean adjusted fare (USD)")
    axes[0].legend(frameon=False)
    axes[1].plot(multiplier_hour["event_hour_local"], multiplier_hour["surge_rate_pct"], marker="o", color=COLORS["Lyft"])
    axes[1].set(title="Lyft surge rate by local hour", xlabel="Local hour", ylabel="Multiplier > 1 (%)")
    for ax in axes:
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.2)
    save_figure(fig, "03_hourly_effects_v1.0.0.png")


def plot_distance(price_train: pd.DataFrame) -> None:
    frame = price_train.copy()
    frame["distance_decile"] = pd.qcut(frame["distance_median"], 10, duplicates="drop")
    profile = frame.groupby(["cab_type", "distance_decile"], observed=True).agg(
        distance=("distance_median", "median"), price=("target_price_median", "median")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for provider, group in profile.groupby("cab_type", observed=True):
        ax.plot(group["distance"], group["price"], marker="o", label=provider, color=COLORS[provider])
    ax.set(title="Train: fare increases nonlinearly with trip distance", xlabel="Median distance in bin (miles)", ylabel="Median fare (USD)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "04_distance_price_curve_v1.0.0.png")


def plot_weather(weather: pd.DataFrame) -> None:
    ordered = weather.sort_values("price_adjusted_spearman")
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axes[0].barh(y, ordered["price_adjusted_spearman"], color=COLORS["price"])
    axes[0].set_yticks(y, ordered["feature"])
    axes[0].set(title="Adjusted price association", xlabel="Spearman rho")
    axes[1].barh(y, ordered["surge_indicator_spearman"], color=COLORS["multiplier"])
    axes[1].set(title="Lyft surge association", xlabel="Spearman rho")
    for ax in axes:
        ax.axvline(0, color="#777777", linewidth=1)
        ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "05_weather_associations_v1.0.0.png")


def plot_freshness(price_freshness: pd.DataFrame, multiplier_freshness: pd.DataFrame) -> None:
    price_test = price_freshness.loc[price_freshness["data_split"].eq("test")].set_index("observation_age_bucket").reindex(AGE_ORDER).reset_index()
    mult_test = multiplier_freshness.loc[multiplier_freshness["data_split"].eq("test")].set_index("observation_age_bucket").reindex(AGE_ORDER).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(price_test["observation_age_bucket"], price_test["persistence_mae"], color=COLORS["price"])
    axes[0].set(title="Test: stale price history raises persistence error", xlabel="Observation age (minutes)", ylabel="Persistence MAE (USD)")
    axes[1].bar(mult_test["observation_age_bucket"], mult_test["multiplier_change_rate_pct"], color=COLORS["multiplier"])
    axes[1].set(title="Test: Lyft multiplier change rate by staleness", xlabel="Observation age (minutes)", ylabel="Changed from lag1 (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "06_observation_freshness_v1.0.0.png")


def plot_location(price_location: pd.DataFrame, multiplier_location: pd.DataFrame) -> None:
    destinations = price_location.groupby("destination", observed=True).apply(
        lambda x: np.average(x["mean_adjusted_price"], weights=x["rows"]), include_groups=False
    ).rename("adjusted_price").sort_values()
    surge_dest = multiplier_location.groupby("destination", observed=True).apply(
        lambda x: np.average(x["surge_rate_pct"], weights=x["rows"]), include_groups=False
    ).rename("surge_rate_pct").reindex(destinations.index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    y = np.arange(len(destinations))
    axes[0].barh(y, destinations.values, color=COLORS["price"])
    axes[0].set_yticks(y, destinations.index)
    axes[0].axvline(0, color="#777777", linewidth=1)
    axes[0].set(title="Destination effect after service/distance adjustment", xlabel="Mean adjusted fare (USD)")
    axes[1].barh(y, surge_dest.values, color=COLORS["multiplier"])
    axes[1].set(title="Lyft surge rate by destination", xlabel="Multiplier > 1 (%)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "07_location_effects_v1.0.0.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    price = pd.read_parquet(PRICE_PATH)
    multiplier = pd.read_parquet(MULTIPLIER_PATH)
    canonical = pd.read_parquet(CANONICAL_PATH, columns=["cab_type", "name", "price", "price_target_eligible"])
    price = add_eda_residuals(price)
    price_train = price.loc[price["data_split"].eq("train")].copy()
    multiplier_train = multiplier.loc[multiplier["data_split"].eq("train")].copy()

    price_profile, multiplier_profile = target_and_service_profiles(price, multiplier)
    price_time, multiplier_time = time_profiles(price_train, multiplier_train)
    price_location, multiplier_location = location_profiles(price_train, multiplier_train)
    weather = weather_associations(price_train, multiplier_train)
    weather_summary = weather_summary_profiles(price_train, multiplier_train)
    price_freshness, multiplier_freshness = freshness_profiles(price, multiplier)
    drift = split_drift(price, multiplier)
    baselines = baseline_diagnostics(price, multiplier)
    quality, correlated = numeric_feature_quality(price_train)
    recommendation = recommendations(weather, correlated)
    research = research_decisions()

    availability = canonical.groupby(["cab_type", "name"], observed=True).agg(
        raw_rows=("name", "size"),
        priced_rows=("price", "count"),
        eligible_price_rows=("price_target_eligible", "sum"),
    ).reset_index()
    availability["missing_price_pct"] = 100 * (1 - availability["priced_rows"] / availability["raw_rows"])

    save_csv(availability, "availability_profile_v1.0.0.csv")
    save_csv(price_profile, "price_service_profile_v1.0.0.csv")
    save_csv(multiplier_profile, "lyft_multiplier_distribution_v1.0.0.csv")
    save_csv(price_time, "price_time_profile_v1.0.0.csv")
    save_csv(multiplier_time, "multiplier_time_profile_v1.0.0.csv")
    save_csv(price_location, "price_route_profile_v1.0.0.csv")
    save_csv(multiplier_location, "multiplier_route_profile_v1.0.0.csv")
    save_csv(weather, "weather_associations_v1.0.0.csv")
    save_csv(weather_summary, "weather_summary_profile_v1.0.0.csv")
    save_csv(price_freshness, "price_freshness_profile_v1.0.0.csv")
    save_csv(multiplier_freshness, "multiplier_freshness_profile_v1.0.0.csv")
    save_csv(drift, "split_drift_v1.0.0.csv")
    save_csv(baselines, "baseline_diagnostics_v1.0.0.csv")
    save_csv(quality, "numeric_feature_quality_v1.0.0.csv")
    save_csv(correlated, "correlated_feature_pairs_v1.0.0.csv")
    save_csv(recommendation, "feature_recommendations_v1.0.0.csv")
    save_csv(research, "research_decisions_v1.0.0.csv")

    plot_price_services(price_train, price_profile)
    plot_multiplier_distribution(multiplier_profile)
    plot_hourly(price_time, multiplier_time)
    plot_distance(price_train)
    plot_weather(weather)
    plot_freshness(price_freshness, multiplier_freshness)
    plot_location(price_location, multiplier_location)

    train_multiplier_share = multiplier_profile.loc[
        multiplier_profile["data_split"].eq("train") & multiplier_profile["target_multiplier_median"].eq(1), "share_pct"
    ].iloc[0]
    price_lag_rho = safe_spearman(price_train["lag1_price_median"], price_train["target_price_median"])
    multiplier_lag_rho = safe_spearman(multiplier_train["lag1_multiplier_median"], multiplier_train["target_multiplier_median"])
    price_persistence_test_mae = baselines.loc[(baselines["target"] == "price") & (baselines["baseline"] == "lag1_persistence") & (baselines["metric"] == "MAE"), "value"].iloc[0]
    price_route_test_mae = baselines.loc[(baselines["target"] == "price") & (baselines["baseline"] == "route_service_train_median") & (baselines["metric"] == "MAE"), "value"].iloc[0]
    multiplier_constant_test_mae = baselines.loc[(baselines["target"] == "lyft_multiplier") & (baselines["baseline"] == "constant_1") & (baselines["metric"] == "MAE"), "value"].iloc[0]
    multiplier_persistence_test_mae = baselines.loc[(baselines["target"] == "lyft_multiplier") & (baselines["baseline"] == "lag1_persistence") & (baselines["metric"] == "MAE"), "value"].iloc[0]
    top_weather_price = weather.iloc[weather["price_adjusted_spearman"].abs().argmax()]
    top_weather_surge = weather.iloc[weather["surge_indicator_spearman"].abs().argmax()]
    test_fresh = price_freshness.loc[
        price_freshness["data_split"].eq("test") & price_freshness["observation_age_bucket"].eq("00-15"), "persistence_mae"
    ].iloc[0]
    test_stale = price_freshness.loc[
        price_freshness["data_split"].eq("test") & price_freshness["observation_age_bucket"].eq("120+"), "persistence_mae"
    ].iloc[0]

    key_findings = {
        "eda_version": EDA_VERSION,
        "decision_scope": "Train split only for feature conclusions; calibration/test used only for drift/freshness diagnostics.",
        "price_train_rows": len(price_train),
        "multiplier_train_rows": len(multiplier_train),
        "uber_price_rows_all_splits": int(price["cab_type"].eq("Uber").sum()),
        "lyft_price_rows_all_splits": int(price["cab_type"].eq("Lyft").sum()),
        "uber_taxi_raw_rows_excluded_from_price_target": int(((canonical["cab_type"] == "Uber") & (canonical["name"] == "Taxi")).sum()),
        "price_lag1_spearman_train": price_lag_rho,
        "multiplier_lag1_spearman_train": multiplier_lag_rho,
        "price_persistence_test_mae": float(price_persistence_test_mae),
        "price_route_service_median_test_mae": float(price_route_test_mae),
        "multiplier_constant_1_test_mae": float(multiplier_constant_test_mae),
        "multiplier_persistence_test_mae": float(multiplier_persistence_test_mae),
        "lyft_multiplier_1_share_train_pct": float(train_multiplier_share),
        "strongest_adjusted_price_weather_feature": {
            "feature": top_weather_price["feature"],
            "spearman": float(top_weather_price["price_adjusted_spearman"]),
        },
        "strongest_surge_weather_feature": {
            "feature": top_weather_surge["feature"],
            "spearman": float(top_weather_surge["surge_indicator_spearman"]),
        },
        "test_persistence_mae_00_15": float(test_fresh),
        "test_persistence_mae_120_plus": float(test_stale),
        "correlated_numeric_pairs_abs_rho_ge_0_80": len(correlated),
    }
    (OUT / "key_findings_v1.0.0.json").write_text(json.dumps(key_findings, ensure_ascii=False, indent=2), encoding="utf-8")

    service_top = price_profile.loc[price_profile["data_split"].eq("train")].sort_values("median_price", ascending=False).iloc[0]
    service_bottom = price_profile.loc[price_profile["data_split"].eq("train")].sort_values("median_price").iloc[0]
    report = f"""# Competitor Fare EDA Report v{EDA_VERSION}

## Kết luận ngắn

EDA được thực hiện cho cả **price (Uber + Lyft)** và **Lyft multiplier**. Kết luận chọn feature chỉ dùng train split để tránh nhìn trước calibration/test. Tín hiệu lịch sử giá, service/route, distance và observation age là nhóm feature chính; time và weather được giữ để kiểm định bằng ablation, nhưng quan hệ EDA không được diễn giải là quan hệ nhân quả.

## Scope

- Price: {len(price):,} snapshots, gồm {int(price['cab_type'].eq('Uber').sum()):,} Uber và {int(price['cab_type'].eq('Lyft').sum()):,} Lyft.
- Multiplier: {len(multiplier):,} Lyft snapshots; Uber không có multiplier variation trong nguồn.
- Train-only EDA: {len(price_train):,} price rows và {len(multiplier_train):,} multiplier rows.
- Uber Taxi: {key_findings['uber_taxi_raw_rows_excluded_from_price_target']:,} raw rows được giữ ở canonical layer nhưng không có price target.

## Phát hiện chính

1. **Service/distance là structural drivers.** Median service fare chạy từ ${service_bottom['median_price']:.2f} ({service_bottom['cab_type']} {service_bottom['name']}) đến ${service_top['median_price']:.2f} ({service_top['cab_type']} {service_top['name']}); quan hệ distance–price rõ nhưng phi tuyến.
2. **Delayed history có signal mạnh cho price nhưng không nên dùng kiểu copy-last-value.** Spearman lag1–current là {price_lag_rho:.3f}, nhưng price persistence test MAE ${price_persistence_test_mae:.3f}, kém hơn route/service train-median baseline (${price_route_test_mae:.3f}). Model nên kết hợp history với structural features. Với Lyft multiplier, lag1 correlation chỉ {multiplier_lag_rho:.3f}.
3. **Multiplier bị mất cân bằng mạnh.** {train_multiplier_share:.2f}% train snapshots có multiplier=1.0. Trên test, constant-1 MAE ({multiplier_constant_test_mae:.4f}) còn thấp hơn lag1 persistence ({multiplier_persistence_test_mae:.4f}); model phải được đánh giá bằng surge recall/F1 và tail error, không chỉ overall MAE.
4. **Time/location có signal sau adjustment.** Báo cáo dùng residual đã điều chỉnh service/distance; `source` và `destination` được dùng như location category, không tuyên bố tọa độ là trung tâm thành phố.
5. **Weather có signal nhỏ và tương quan chéo cao.** Mạnh nhất cho adjusted price là `{top_weather_price['feature']}` (rho={top_weather_price['price_adjusted_spearman']:.3f}); cho Lyft surge là `{top_weather_surge['feature']}` (rho={top_weather_surge['surge_indicator_spearman']:.3f}). Có {len(correlated)} cặp numeric features với |rho|>=0.80, nên cần grouped importance/ablation thay vì đọc từng hệ số riêng lẻ.

## Feature decision cho modeling

- **Giữ ưu tiên cao:** lag/rolling price history, observation age, distance, provider/service, source/destination.
- **Feature engineering:** log-age và age×lag, nonlinear distance×service, hour×weekday, cyclic time, route direction, price-change dynamics.
- **Multiplier:** target hiện tại là median 5 phút nên có thể xuất hiện mức trung gian; dùng regression + auxiliary `surge > 1`. Chỉ dùng ordinal classification nếu tạo target mode/raw-level riêng.
- **Ưu tiên thấp/ablation:** current weather và short summary có association gần 0 trong bộ này; chỉ giữ trong model cuối nếu grouped out-of-time ablation chứng minh có lift.
- **Không dùng làm predictor:** price-per-mile từ current target, current-bucket price min/max/spread/count, IDs, daily/future weather.

Chi tiết từng quyết định nằm trong `feature_recommendations_v1.0.0.csv`.

## Research basis

- Lag/rolling phải shift và đánh giá theo thời gian: [scikit-learn lagged forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
- Hour/weekday nên có cyclic encoding: [scikit-learn time feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html).
- Route/service categorical có thể dùng native categorical model: [CatBoost categorical features](https://catboost.ai/docs/en/features/categorical-features).
- Correlated weather làm individual importance dễ sai lệch: [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
- Surge prediction có cơ sở dùng spatiotemporal và weather history: [Transportation Research Part C study](https://www.sciencedirect.com/science/article/pii/S0968090X19301627).
- Uncertainty phase nên so sánh chronological split conformal với EnbPI: [Xu & Xie, ICML 2021](https://proceedings.mlr.press/v139/xu21h.html).

## Hạn chế

Boston-only, 17 ngày, weather provenance hạn chế, không có traffic/events/supply-demand và không có Vietnam data. EDA cho thấy association, không chứng minh causality hay production generalization.
"""
    (OUT / "EDA_REPORT_v1.0.0.md").write_text(report, encoding="utf-8")

    validations = pd.DataFrame(
        [
            ("price_contains_uber", price["cab_type"].eq("Uber").any(), int(price["cab_type"].eq("Uber").sum()), ">0"),
            ("price_contains_lyft", price["cab_type"].eq("Lyft").any(), int(price["cab_type"].eq("Lyft").sum()), ">0"),
            ("multiplier_lyft_only", multiplier["cab_type"].eq("Lyft").all(), int(multiplier["cab_type"].ne("Lyft").sum()), 0),
            ("feature_decisions_train_only", price_train["data_split"].eq("train").all() and multiplier_train["data_split"].eq("train").all(), "train", "train"),
            ("price_per_mile_not_recommended_as_predictor", recommendation.loc[recommendation["feature_group"].eq("Price per mile"), "decision"].eq("EDA_ONLY").all(), "EDA_ONLY", "EDA_ONLY"),
            ("weather_profile_complete", set(weather["feature"]) == set(WEATHER), len(weather), len(WEATHER)),
            ("all_figures_created", len(list(FIG.glob("*.png"))) == 7, len(list(FIG.glob("*.png"))), 7),
        ],
        columns=["check", "passed", "actual", "expected"],
    )
    validations.to_csv(OUT / "validation_results_v1.0.0.csv", index=False)
    if not validations["passed"].all():
        raise AssertionError(validations.loc[~validations["passed"]].to_dict("records"))

    outputs = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "manifest_v1.0.0.json")
    script = Path(__file__).resolve()
    manifest = {
        "eda_version": EDA_VERSION,
        "modeling_data_version": MODELING_DATA_VERSION,
        "inputs": [
            {"name": path.name, "sha256": sha256(path)}
            for path in [PRICE_PATH, MULTIPLIER_PATH, CANONICAL_PATH]
        ],
        "script": {"path": str(script), "sha256": sha256(script)},
        "outputs": [
            {"path": str(path.relative_to(OUT)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    (OUT / "manifest_v1.0.0.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(key_findings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
