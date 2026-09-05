# PM2.5 in Northern Thailand: From Raw Data to a Recommendation

**DS-270702 Data Science Programming — Homework 4**

> This draft is a reproducible report structure. Numerical results must be populated from `outputs/results/` after the API pipeline is run. Do not invent or manually type model results.

## 1. Problem Background

Northern Thailand experiences recurrent PM2.5 pollution episodes, especially during the dry-season period. The Thai 24-hour ambient PM2.5 standard used in this assignment is 37.5 µg/m³.

This project focuses on Chiang Mai and Chiang Rai and asks whether information available by the end of today can support a useful warning about tomorrow.

[Add 1–2 pages with properly cited background sources.]

## 2. Approach

### Research question

Can today's PM2.5 and same-day weather conditions predict whether tomorrow's daily mean PM2.5 will exceed 37.5 µg/m³?

### Expected finding before analysis

We expected recent PM2.5 levels to be strong predictors because pollution episodes can persist from one day to the next. We also expected low wind and limited precipitation to accompany higher PM2.5 concentrations. These are hypotheses, not conclusions.

## 3. Method

### 3.1 Data sources

**Source 1 — Open-Meteo Air Quality API**

Hourly PM2.5, PM10, carbon monoxide and dust.

**Source 2 — Open-Meteo Historical Weather API**

Hourly temperature, relative humidity, wind speed, wind direction, precipitation and surface pressure.

**Source 3 — Air4Thai**

Current measured station readings used only for checkpoint C6.

### 3.2 Locations and period

- Chiang Mai: 18.7883, 98.9853
- Chiang Rai: 19.9105, 99.8406
- 2023-01-01 through 2025-12-31
- Asia/Bangkok timezone

### 3.3 Cleaning and aggregation

Hourly data were converted into daily observations using the local Thailand calendar date. Daily PM2.5 was calculated as the arithmetic mean of the hourly PM2.5 values. Weather variables were aggregated using means for instantaneous variables and a daily sum for precipitation.

Negative pollutant values were treated as missing. Rows without a complete target and required prediction-time features were excluded from the final modelling table.

### 3.4 Join

Air-quality and weather daily tables were joined using `date` within each location. Exact row counts before and after the join are reported in `data/processed/join_audit.json`.

### 3.5 Target and features

Target:

`tomorrow_exceeds_37_5`

where 1 means tomorrow's daily mean PM2.5 is greater than 37.5 µg/m³.

Features include today's PM2.5, one-day and seven-day PM2.5 lags, seven-day historical mean, PM10, dust, temperature, humidity, wind speed, precipitation, surface pressure, day of week, month and location.

All prediction features are available by the prediction moment. Tomorrow's weather and tomorrow's PM2.5 are not used as predictors.

### 3.6 Split, baseline and model

The final 20% of calendar dates are held out as the test set. This respects the temporal ordering of the problem.

The baseline predicts the majority class from the training set.

The model is logistic regression with preprocessing for numeric and categorical features. `TimeSeriesSplit` with five folds is used on the training portion.

Metrics include accuracy, balanced accuracy, positive-class precision, positive-class recall, F1 and ROC-AUC. Recall is especially important because a false negative means failing to warn about a dangerous day.

## 4. Results

### 4.1 Exploratory analysis

**Figure 1 — Daily PM2.5**

Insert `outputs/figures/fig01_daily_pm25.png`.

Interpretation: [Write what the plot actually shows.]

**Figure 2 — Days above the standard**

Insert `outputs/figures/fig02_days_above_standard.png`.

Interpretation: [Compare locations and years using the generated table.]

**Figure 3 — Monthly pattern**

Insert `outputs/figures/fig03_monthly_pattern.png`.

Interpretation: [Identify the observed seasonal pattern and whether it differs by location.]

**Figure 4 — Weather on high-PM2.5 days**

Insert `outputs/figures/fig04_weather_high_pm25.png`.

Interpretation: [Describe the observed differences between the top 10% PM2.5 days and other days.]

**Figure 5 — Weekly pattern**

Insert `outputs/figures/fig05_weekly_pattern.png`.

Interpretation: [State whether a meaningful weekly pattern is present.]

### 4.2 C1 — Time

Evidence: `timezone` is set to `Asia/Bangkok` in both API requests. The raw response metadata and hourly timestamps should be inspected to verify that each day starts at local midnight. The report should include one example date showing the first and last hourly timestamps.

### 4.3 C2 — Join

Use `data/processed/join_audit.json` to report hourly and daily row counts for each location and explain any unmatched dates.

### 4.4 C3 — Missing values

Use `data/processed/missing_values.csv`. Report the missing percentage for every column. If any variable has exactly 0.00% missingness across the entire period, discuss whether this reflects a model-generated/reanalysis source rather than a sensor network.

### 4.5 C4 — Comparing places

The two requested coordinates must be demonstrated to return different time series before making a spatial claim. Report the correlation and at least one date where the daily values differ.

### 4.6 C5 — Model versus baseline

Insert `outputs/results/metrics.csv`.

Discuss the baseline and model on the same test set. If the model does not beat the baseline, report that honestly.

### 4.7 C6 — Ground truth

Use `data/raw/air4thai_current.json` and a current Open-Meteo reading for a nearby station/location. Report the timestamp, values and difference. Explain that Air4Thai is an instrument measurement while Open-Meteo is model output, so neither should automatically be treated as universally "right" without considering spatial representativeness and measurement context.

### 4.8 Data quality problem

Document one concrete issue found during the pipeline, the fix, and evidence that the fix was applied.

### 4.9 Limitation

A major limitation is that the Open-Meteo air-quality values are model output rather than direct local station measurements. Another limitation is that Air4Thai provides current readings only for the tested endpoint, so it cannot provide a full historical validation series through this workflow.

## 5. Conclusion and Recommendation

### Findings

Replace this section with the three strongest findings that were not obvious before analysis.

1. [Finding supported by Figure/Table]
2. [Finding supported by Figure/Table]
3. [Finding supported by model/checkpoint evidence]

### Recommendation

**Audience:** [Name the specific audience, e.g. Chiang Mai provincial working group.]

**Action:** [Concrete action derived from the strongest evidence.]

**Timing:** [Specify when the action should happen.]

**Model safety:** The recommendation should account for false negatives. If recall for dangerous days is low, the model should not be used as the sole warning mechanism.

### What the analysis does not support

The analysis does not establish causal effects of weather variables or prove that a specific emission source caused an observed PM2.5 episode. It also does not justify broad claims beyond the locations and period studied.

### Limitations and next steps

[Discuss data resolution, modelled air-quality values, current-only Air4Thai comparison, and what additional measured data or fire/hotspot information would improve the analysis.]

## Appendix A — AI disclosure

| Where you used it | What you asked for | What you changed or rejected |
|---|---|---|
| Coding | Help structure a reproducible Python pipeline for API acquisition, preparation, EDA and modelling | Student reviewed and tested code; document concrete changes made |
| Report planning | Help interpret assignment requirements and structure the report | Student verified requirements against the Lab Sheet |

## Appendix B — Submission checklist

- [ ] Repository is public or instructor added as collaborator
- [ ] README explains run order
- [ ] requirements.txt lists packages actually used
- [ ] Fresh-clone rerun completed
- [ ] data/raw contains exactly API responses
- [ ] all figures saved separately
- [ ] model metrics saved in outputs/results
- [ ] report PDF follows required structure
- [ ] C1–C6 answered
- [ ] baseline and model scores shown together
- [ ] recommendation names a specific audience
- [ ] AI disclosure completed
- [ ] student can explain every submitted line of code
