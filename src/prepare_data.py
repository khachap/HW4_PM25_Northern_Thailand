"""
Clean, aggregate, join, and construct the modelling dataset.

Inputs:
    data/raw/openmeteo_air_*.json
    data/raw/openmeteo_weather_*.json

Outputs:
    data/processed/daily_pm25_weather.csv
    data/processed/missing_values.csv
    data/processed/join_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

AIR_COLS = ["pm2_5", "pm10", "carbon_monoxide", "dust"]
WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
]


def read_hourly(path: Path, columns: list[str]) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    hourly = data["hourly"]
    df = pd.DataFrame({c: hourly.get(c) for c in ["time", *columns]})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def aggregate_air(df: pd.DataFrame) -> pd.DataFrame:
    # Thailand local timestamps are already supplied by the API.
    df["date"] = df["time"].dt.date
    agg = {
        "pm2_5": "mean",
        "pm10": "mean",
        "carbon_monoxide": "mean",
        "dust": "mean",
    }
    return df.groupby("date", as_index=False).agg(agg)


def aggregate_weather(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = df["time"].dt.date
    agg = {
        "temperature_2m": "mean",
        "relative_humidity_2m": "mean",
        "wind_speed_10m": "mean",
        "wind_direction_10m": "mean",
        "precipitation": "sum",
        "surface_pressure": "mean",
    }
    return df.groupby("date", as_index=False).agg(agg)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["location", "date"]).copy()
    df["pm25_today"] = df["pm2_5"]
    df["pm25_lag1"] = df.groupby("location")["pm2_5"].shift(1)
    df["pm25_lag7"] = df.groupby("location")["pm2_5"].shift(7)
    df["pm25_7d_mean"] = (
        df.groupby("location")["pm2_5"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    )
    # Target is tomorrow's daily mean PM2.5.
    df["pm25_tomorrow"] = df.groupby("location")["pm2_5"].shift(-1)
    df["tomorrow_exceeds_37_5"] = (df["pm25_tomorrow"] > 37.5).astype("Int64")
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    location_frames = []
    audit = {"locations": {}, "join_method": "inner join on date within each location"}

    for location in ["chiang_mai", "chiang_rai"]:
        air_path = RAW_DIR / f"openmeteo_air_{location}.json"
        weather_path = RAW_DIR / f"openmeteo_weather_{location}.json"

        air_hourly = read_hourly(air_path, AIR_COLS)
        weather_hourly = read_hourly(weather_path, WEATHER_COLS)

        air_daily = aggregate_air(air_hourly)
        weather_daily = aggregate_weather(weather_hourly)

        merged = air_daily.merge(weather_daily, on="date", how="inner")
        merged["location"] = location

        audit["locations"][location] = {
            "air_hourly_rows": int(len(air_hourly)),
            "weather_hourly_rows": int(len(weather_hourly)),
            "air_daily_rows": int(len(air_daily)),
            "weather_daily_rows": int(len(weather_daily)),
            "joined_daily_rows": int(len(merged)),
            "air_dates_not_in_weather": int(
                len(set(air_daily["date"]) - set(weather_daily["date"]))
            ),
            "weather_dates_not_in_air": int(
                len(set(weather_daily["date"]) - set(air_daily["date"]))
            ),
        }
        location_frames.append(merged)

    df = pd.concat(location_frames, ignore_index=True)
    df = add_features(df)

    # Quality rule: impossible negative concentrations are treated as missing.
    for col in ["pm2_5", "pm10", "carbon_monoxide", "dust"]:
        df.loc[df[col] < 0, col] = pd.NA

    # Keep rows with a complete target and required prediction-time features.
    required = [
        "pm2_5", "pm10", "dust",
        "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "precipitation", "surface_pressure",
        "pm25_lag1", "pm25_lag7", "pm25_7d_mean",
        "pm25_tomorrow",
    ]
    df_model = df.dropna(subset=required).copy()

    missing = (
        df.isna().mean()
        .mul(100)
        .round(3)
        .rename("missing_percent")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing.to_csv(OUT_DIR / "missing_values.csv", index=False)

    audit["combined_rows_before_model_dropna"] = int(len(df))
    audit["model_rows_after_dropna"] = int(len(df_model))
    audit["target_definition"] = "next calendar day's daily mean PM2.5"
    audit["standard_ug_m3"] = 37.5
    audit["timezone"] = "Asia/Bangkok"

    df.to_csv(OUT_DIR / "daily_pm25_weather_all.csv", index=False)
    df_model.to_csv(OUT_DIR / "daily_pm25_weather.csv", index=False)

    (OUT_DIR / "join_audit.json").write_text(
        json.dumps(audit, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Saved {len(df_model)} modelling rows to data/processed/daily_pm25_weather.csv")


if __name__ == "__main__":
    main()
