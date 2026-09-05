"""
Fetch raw data for HW4 PM2.5 in Northern Thailand.

Sources
-------
1) Open-Meteo Air Quality API
2) Open-Meteo Historical Weather API
3) Air4Thai current station readings (for C6 only)

Run from repository root:
    python src/fetch_data.py

Raw API responses are saved unchanged under data/raw/.
Metadata is saved separately so the raw response remains exactly what the API returned.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Asia/Bangkok"

LOCATIONS = {
    "chiang_mai": {"latitude": 18.7883, "longitude": 98.9853},
    "chiang_rai": {"latitude": 19.9105, "longitude": 99.8406},
}

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR4THAI_URL = "https://air4thai.pcd.go.th/services/getNewAQI_JSON.php"

AIR_VARIABLES = ["pm2_5", "pm10", "carbon_monoxide", "dust"]
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
]


def get_json(url: str, params: dict | None = None, *, verify: bool = True) -> dict:
    """GET JSON with a retry loop and a clear failure message."""
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=120,
                verify=verify,
                headers={"User-Agent": "DS-270702-HW4/1.0"},
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Request failed: {url}") from last_error


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_openmeteo_air(location_name: str, coords: dict) -> dict:
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": ",".join(AIR_VARIABLES),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
    }
    data = get_json(AIR_URL, params)
    save_json(data, RAW_DIR / f"openmeteo_air_{location_name}.json")
    return {
        "source": "Open-Meteo Air Quality API",
        "endpoint": AIR_URL,
        "location": location_name,
        "parameters": params,
        "rows": len(data.get("hourly", {}).get("time", [])),
    }


def fetch_openmeteo_weather(location_name: str, coords: dict) -> dict:
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": ",".join(WEATHER_VARIABLES),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
    }
    data = get_json(WEATHER_URL, params)
    save_json(data, RAW_DIR / f"openmeteo_weather_{location_name}.json")
    return {
        "source": "Open-Meteo Historical Weather API",
        "endpoint": WEATHER_URL,
        "location": location_name,
        "parameters": params,
        "rows": len(data.get("hourly", {}).get("time", [])),
    }


def fetch_air4thai() -> dict:
    """
    Air4Thai is current-only, so this file is intentionally timestamped.
    The raw response is preserved for C6; it is not joined to the historical data.
    """
    data = get_json(AIR4THAI_URL, verify=False)
    path = RAW_DIR / "air4thai_current.json"
    save_json(data, path)
    return {
        "source": "Air4Thai / Pollution Control Department",
        "endpoint": AIR4THAI_URL,
        "location": "all stations",
        "parameters": {},
        "rows": len(data.get("stations", data.get("data", [])))
        if isinstance(data, dict)
        else None,
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
        "locations": LOCATIONS,
        "requests": [],
    }

    for location_name, coords in LOCATIONS.items():
        print(f"Fetching air quality: {location_name}")
        metadata["requests"].append(fetch_openmeteo_air(location_name, coords))
        print(f"Fetching weather: {location_name}")
        metadata["requests"].append(fetch_openmeteo_weather(location_name, coords))

    print("Fetching Air4Thai current station readings")
    metadata["requests"].append(fetch_air4thai())

    save_json(metadata, RAW_DIR / "fetch_metadata.json")
    print("Done. Raw responses are in data/raw/.")


if __name__ == "__main__":
    main()
