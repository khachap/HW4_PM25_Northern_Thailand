# HW4 — PM2.5 in Northern Thailand

## Research question

**Can today's PM2.5 level and same-day weather conditions predict whether tomorrow's daily mean PM2.5 will exceed Thailand's 37.5 µg/m³ standard in Chiang Mai and Chiang Rai?**

The project uses two historical Open-Meteo endpoints and an Air4Thai current reading for the ground-truth comparison.

## Sources

1. Open-Meteo Air Quality API — hourly PM2.5, PM10, CO and dust.
2. Open-Meteo Historical Weather API — hourly temperature, humidity, wind, precipitation and surface pressure.
3. Air4Thai — current station readings only; used for checkpoint C6, not joined to historical data.

The Open-Meteo air-quality source is model output from CAMS rather than a local instrument measurement. Air4Thai is an instrument-based station source.

## Locations

- Chiang Mai: 18.7883, 98.9853
- Chiang Rai: 19.9105, 99.8406

Date range: 2023-01-01 to 2025-12-31  
Timezone: Asia/Bangkok

## Target

`tomorrow_exceeds_37_5`

- 1 = tomorrow's daily mean PM2.5 > 37.5 µg/m³
- 0 = otherwise

Prediction moment: after today's complete daily observation is available.

## Run order

From the repository root:

```bash
python -m pip install -r requirements.txt
python src/fetch_data.py
python src/prepare_data.py
python src/check_dataset.py data/processed/daily_pm25_weather.csv --target pm25_tomorrow --time date --fetch-script src/fetch_data.py
python src/analyse.py
python src/model.py
```

Outputs:

- `data/raw/` — unchanged API responses plus fetch metadata
- `data/processed/` — cleaned daily modelling data
- `outputs/figures/` — PNG figures
- `outputs/results/` — descriptive tables and model metrics

## Reproducibility note

Open-Meteo historical data should be stable for a fixed request, while Air4Thai is a current-reading endpoint and therefore changes when re-run. The Air4Thai file is timestamped and used only for C6.

## Important methodology choices

- Daily aggregation is performed using Asia/Bangkok timestamps.
- Historical rows are joined by local calendar date within each location.
- The test set is the later 20% of calendar dates.
- `TimeSeriesSplit` is used on the training data.
- The majority-class baseline is reported before the model.
- Recall for the dangerous class (>37.5) is reported because missed dangerous days matter for a warning system.
- Features use today's information or earlier; tomorrow's weather is not used.

## AI disclosure

AI was used as a coding assistant during development. The student must review, understand, test, and modify every line before submission and should complete the AI disclosure table in the report.

## Data attribution

See the Open-Meteo documentation and CAMS acknowledgement. All external-source claims in the final report should be cited to the original source.
