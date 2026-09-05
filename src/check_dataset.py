"""
A lightweight feasibility checker matching the assignment's requested checks.

Example:
    python src/check_dataset.py data/processed/daily_pm25_weather.csv \
        --target pm25_tomorrow --time date --fetch-script src/fetch_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--target", required=True)
    parser.add_argument("--time", required=True)
    parser.add_argument("--fetch-script", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    issues = []

    if len(df) < 500:
        issues.append(f"Only {len(df)} rows; need at least 500.")
    if args.target not in df.columns:
        issues.append(f"Target column not found: {args.target}")
    if args.time not in df.columns:
        issues.append(f"Time column not found: {args.time}")
    if not Path(args.fetch_script).exists():
        issues.append(f"Fetch script not found: {args.fetch_script}")

    if args.time in df.columns:
        parsed = pd.to_datetime(df[args.time], errors="coerce")
        if parsed.isna().any():
            issues.append("Time column contains unparseable values.")
        if not parsed.is_monotonic_increasing:
            issues.append("Data is not sorted by time; modelling code must sort before splitting.")

    missing_target = df[args.target].isna().mean() if args.target in df else 1
    if missing_target > 0:
        issues.append(f"Target has {missing_target:.2%} missing values.")

    verdict = "FEASIBLE" if not issues else "FEASIBLE WITH ISSUES"
    print(f"VERDICT: {verdict}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"- {issue}")


if __name__ == "__main__":
    main()
