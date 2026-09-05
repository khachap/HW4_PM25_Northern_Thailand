
"""
Exploratory Data Analysis for Northern Thailand PM2.5 project.

EDA requirements:
1. Describe dataset size, date range, locations, and missing values.
2. Examine PM2.5 trends over time.
3. Compare PM2.5 conditions between Chiang Mai and Chiang Rai.
4. Examine yearly exceedance patterns.
5. Examine weather conditions associated with high PM2.5.
6. Examine weekly PM2.5 patterns.

Important:
- EDA uses observed daily PM2.5 (pm25_today).
- The ML target tomorrow_exceeds_37_5 is NOT used to define
  observed PM2.5 exceedance in the location comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed" / "daily_pm25_weather.csv"
FIGURES = ROOT / "outputs" / "figures"

PM25_STANDARD = 37.5


def load_data() -> pd.DataFrame:
    """Load the processed dataset."""
    print("Loading dataset...")

    df = pd.read_csv(
        DATA,
        parse_dates=["date"],
    )

    df = df.sort_values(
        ["date", "location"]
    ).reset_index(drop=True)

    return df


def basic_summary(df: pd.DataFrame) -> None:
    """Print basic dataset information."""

    print("=" * 70)
    print("BASIC DATA SUMMARY")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    print("\nLocations:")
    print(df["location"].value_counts())

    print("\nPM2.5 summary (µg/m³):")
    print(
        df["pm25_today"]
        .describe()
        .round(2)
    )

    print("\nMissing values (%):")

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .round(3)
    )

    print(missing)



def figure_1_pm25_time_series(
    df: pd.DataFrame,
) -> None:
    """
    Figure 1:
    Monthly mean PM2.5 by year.

    Purpose:
    Examine when the PM2.5 pollution season starts and ends
    and whether the pattern changes between years.

    Thailand PM2.5 standard:
        37.5 µg/m³
    """

    data = df.copy()

    # Extract year and month.
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month

    # Calculate monthly mean PM2.5 for each year.
    monthly = (
        data
        .groupby(["year", "month"])["pm25_today"]
        .mean()
        .reset_index()
    )

    # Convert month number to a real date for plotting.
    monthly["month_date"] = pd.to_datetime(
        monthly["year"].astype(str)
        + "-"
        + monthly["month"].astype(str)
        + "-01"
    )

    # Print the monthly values so the result is traceable.
    print("\nMonthly mean PM2.5 by year:")
    print(
        monthly[
            ["year", "month", "pm25_today"]
        ]
        .round(2)
        .to_string(index=False)
    )

    # Create figure.
    plt.figure(figsize=(14, 7))

    # Plot each year separately.
    for year, group in monthly.groupby("year"):

        plt.plot(
            group["month_date"],
            group["pm25_today"],
            marker="o",
            linewidth=2,
            label=str(year),
        )

    # Thailand's 24-hour PM2.5 standard.
    plt.axhline(
        PM25_STANDARD,
        linestyle="--",
        linewidth=1.5,
        label="37.5 µg/m³ standard",
    )

    # Title and labels.
    plt.title(
        "Monthly Mean PM2.5 by Year in Northern Thailand"
    )

    plt.xlabel(
        "Month"
    )

    plt.ylabel(
        "Mean PM2.5 (µg/m³)"
    )

    # Show month labels.
    month_dates = pd.date_range(
        start="2023-01-01",
        end="2023-12-01",
        freq="MS",
    )

    plt.xticks(
        month_dates,
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )

    plt.legend(
        title="Year"
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output = (
        FIGURES
        / "fig01_pm25_time_series.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


def figure_2_exceedance_by_year(
    df: pd.DataFrame,
) -> None:
    """
    Figure 2:
    Number of observed days where today's PM2.5 exceeds 37.5 µg/m³,
    grouped by year and location.
    """

    data = df.copy()

    data["year"] = data["date"].dt.year

    data["exceeds_standard"] = (
        data["pm25_today"] > PM25_STANDARD
    )

    yearly = (
        data[data["exceeds_standard"]]
        .groupby(
            ["year", "location"]
        )
        .size()
        .reset_index(
            name="exceeds_standard"
        )
    )

    print("\nDays exceeding standard by year:")
    print(yearly)

    pivot = yearly.pivot(
        index="year",
        columns="location",
        values="exceeds_standard",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Observed PM2.5 Exceedance Days by Year"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "Number of days > 37.5 µg/m³"
    )

    ax.legend(
        title="Location"
    )

    plt.tight_layout()

    output = (
        FIGURES
        / "fig02_exceedance_by_year.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


def figure_3_location_comparison(
    df: pd.DataFrame,
) -> None:
    """
    Figure 3:
    Compare observed PM2.5 characteristics between locations.

    IMPORTANT:
    Exceedance is defined using today's observed PM2.5:

        pm25_today > 37.5

    It does NOT use the ML target
    tomorrow_exceeds_37_5.
    """

    data = df.copy()

    # Define observed PM2.5 exceedance.
    data["exceeds_standard"] = (
        data["pm25_today"] > PM25_STANDARD
    )

    summary = (
        data
        .groupby("location")
        .agg(
            mean_pm25=(
                "pm25_today",
                "mean",
            ),
            median_pm25=(
                "pm25_today",
                "median",
            ),
            max_pm25=(
                "pm25_today",
                "max",
            ),
            exceedance_days=(
                "exceeds_standard",
                "sum",
            ),
            total_days=(
                "exceeds_standard",
                "count",
            ),
        )
    )

    summary["exceedance_rate_percent"] = (
        summary["exceedance_days"]
        / summary["total_days"]
        * 100
    )

    summary = summary.round(2)

    print("\nLocation comparison:")
    print(summary)

    # Create comparison figure.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    # Mean and median PM2.5.
    summary[
        ["mean_pm25", "median_pm25"]
    ].plot(
        kind="bar",
        ax=axes[0],
    )

    axes[0].set_title(
        "PM2.5 by Location"
    )

    axes[0].set_xlabel(
        "Location"
    )

    axes[0].set_ylabel(
        "PM2.5 (µg/m³)"
    )

    axes[0].tick_params(
        axis="x",
        rotation=0,
    )

    # Exceedance rate.
    summary[
        "exceedance_rate_percent"
    ].plot(
        kind="bar",
        ax=axes[1],
    )

    axes[1].set_title(
        "Observed PM2.5 Exceedance Rate"
    )

    axes[1].set_xlabel(
        "Location"
    )

    axes[1].set_ylabel(
        "Days > 37.5 µg/m³ (%)"
    )

    axes[1].tick_params(
        axis="x",
        rotation=0,
    )

    plt.tight_layout()

    output = (
        FIGURES
        / "fig03_location_comparison.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


def figure_4_weather_high_pm25(
    df: pd.DataFrame,
) -> None:
    """
    Figure 4:
    Compare weather conditions on days before high-PM2.5 events.

    High-PM2.5 event:
        tomorrow_exceeds_37_5 == 1

    This is appropriate here because the question is about
    weather conditions associated with the prediction target.
    """

    data = df.copy()

    high = data[
        data["tomorrow_exceeds_37_5"] == 1
    ]

    normal = data[
        data["tomorrow_exceeds_37_5"] == 0
    ]

    variables = {
        "relative_humidity_2m":
            "Relative humidity (%)",

        "wind_speed_10m":
            "Wind speed (m/s)",

        "precipitation":
            "Precipitation (mm)",

        "temperature_2m":
            "Temperature (°C)",
    }

    rows = []

    for column, label in variables.items():

        rows.append(
            {
                "variable": label,
                "category": "High PM2.5 tomorrow",
                "median": high[column].median(),
            }
        )

        rows.append(
            {
                "variable": label,
                "category": "Not high PM2.5 tomorrow",
                "median": normal[column].median(),
            }
        )

    weather_summary = pd.DataFrame(rows)

    print("\nWeather conditions:")
    print(
        weather_summary.to_string(
            index=False
        )
    )

    comparison_rows = []

    for column, label in variables.items():

        normal_median = (
            normal[column].median()
        )

        high_median = (
            high[column].median()
        )

        if normal_median != 0:
            change = (
                (high_median - normal_median)
                / normal_median
                * 100
            )
        else:
            change = float("nan")

        comparison_rows.append(
            {
                "variable": label,
                "normal_median":
                    normal_median,
                "high_median":
                    high_median,
                "change_percent":
                    change,
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    print("\nWeather median comparison:")
    print(
        comparison.round(2).to_string(
            index=False
        )
    )

    # Plot median comparison.
    plot_data = comparison.set_index(
        "variable"
    )[
        [
            "normal_median",
            "high_median",
        ]
    ]

    ax = plot_data.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_title(
        "Weather Conditions Before High-PM2.5 Days"
    )

    ax.set_xlabel(
        "Weather variable"
    )

    ax.set_ylabel(
        "Median value"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.legend(
        [
            "Not high PM2.5 tomorrow",
            "High PM2.5 tomorrow",
        ]
    )

    plt.tight_layout()

    output = (
        FIGURES
        / "fig04_weather_high_pm25.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


def figure_5_weekly_pattern(
    df: pd.DataFrame,
) -> None:
    """
    Figure 5:
    Average observed PM2.5 by day of week.
    """

    data = df.copy()

    data["day_name"] = (
        data["date"]
        .dt.day_name()
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekly = (
        data
        .groupby("day_name")[
            "pm25_today"
        ]
        .mean()
        .reindex(day_order)
    )

    print("\nAverage PM2.5 by day of week:")
    print(
        weekly.round(2)
    )

    ax = weekly.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Average Daily PM2.5 by Day of Week"
    )

    ax.set_xlabel(
        "Day of week"
    )

    ax.set_ylabel(
        "Average PM2.5 (µg/m³)"
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    plt.tight_layout()

    output = (
        FIGURES
        / "fig05_weekly_pattern.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output}")


def main() -> None:

    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    basic_summary(df)

    print("\n" + "=" * 70)
    print("GENERATING EDA FIGURES")
    print("=" * 70)

    figure_1_pm25_time_series(df)

    figure_2_exceedance_by_year(df)

    figure_3_location_comparison(df)

    figure_4_weather_high_pm25(df)

    figure_5_weekly_pattern(df)

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    print(
        f"Figures saved in: {FIGURES}"
    )


if __name__ == "__main__":
    main()

