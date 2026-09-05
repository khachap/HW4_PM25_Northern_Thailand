
"""
Time-series classification model for predicting whether tomorrow's
daily mean PM2.5 will exceed Thailand's 37.5 µg/m³ standard.

Target:
    tomorrow_exceeds_37_5

Prediction moment:
    end of today's daily observation.

Baseline:
    Majority class in training data.

Split:
    Later 20% of ordered calendar dates as test data.

CV:
    TimeSeriesSplit on the training portion.

Important:
    Accuracy alone is not sufficient because the target is imbalanced.
    Positive-class recall, precision, F1, and balanced accuracy are also
    reported.

The final chronological test period contains no positive cases in this
dataset. This is reported as a limitation rather than changing the split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "data"
    / "processed"
    / "daily_pm25_weather.csv"
)

RESULTS = (
    ROOT
    / "outputs"
    / "results"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "location",
    "pm25_today",
    "pm25_lag1",
    "pm25_lag7",
    "pm25_7d_mean",
    "pm10",
    "dust",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
    "day_of_week",
    "month",
]


# ============================================================
# TARGET
# ============================================================

TARGET = "tomorrow_exceeds_37_5"


# ============================================================
# METRICS
# ============================================================

def metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Calculate classification metrics.

    Positive class:
        1 = tomorrow exceeds 37.5 µg/m³
    """

    out = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "precision_positive": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall_positive": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1_positive": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    # ROC-AUC is only meaningful when both classes are present.
    if (
        y_prob is not None
        and len(np.unique(y_true)) == 2
    ):
        out["roc_auc"] = roc_auc_score(
            y_true,
            y_prob,
        )
    else:
        out["roc_auc"] = np.nan

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    df = pd.read_csv(
        DATA,
        parse_dates=["date"],
    )

    print(
        f"Loaded dataset: {df.shape}"
    )

    # ========================================================
    # 2. SORT BY TIME BEFORE SPLITTING
    # ========================================================
    #
    # The assignment requires the later portion of a
    # time-ordered dataset to be used as the test set.
    #
    # Therefore:
    #
    #       LOAD
    #         ↓
    #       SORT
    #         ↓
    #       SPLIT
    #
    # Never split before sorting.
    # ========================================================

    df = (
        df
        .sort_values(
            ["date", "location"]
        )
        .reset_index(drop=True)
    )

    print(
        "\nData after sorting:"
    )

    print(
        df[
            ["date", "location"]
        ].head(10)
    )

    print(
        "\nTime sorted:",
        df["date"].is_monotonic_increasing,
    )

    # ========================================================
    # 3. CHECK TARGET
    # ========================================================

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' "
            "was not found."
        )

    # ========================================================
    # 4. LEAKAGE CHECK
    # ========================================================
    #
    # pm25_tomorrow is future information.
    # It must NOT be used as a predictor.
    # ========================================================

    if "pm25_tomorrow" in FEATURES:
        raise ValueError(
            "LEAKAGE ERROR: "
            "'pm25_tomorrow' must not be "
            "included in FEATURES."
        )

    # ========================================================
    # 5. CHRONOLOGICAL TRAIN / TEST SPLIT
    # ========================================================
    #
    # The assignment says:
    #
    # "If your data has time order, the test set must
    #  be the later portion."
    #
    # We therefore use the final 20% of unique dates.
    # ========================================================

    unique_dates = np.sort(
        df["date"]
        .dropna()
        .unique()
    )

    split_index = int(
        len(unique_dates) * 0.80
    )

    split_date = unique_dates[
        split_index
    ]

    train = df[
        df["date"] < split_date
    ].copy()

    test = df[
        df["date"] >= split_date
    ].copy()

    # ========================================================
    # 6. PRINT SPLIT INFORMATION
    # ========================================================

    print(
        "\nChronological split:"
    )

    print(
        "Train:",
        train["date"].min(),
        "to",
        train["date"].max(),
        "| rows =",
        len(train),
    )

    print(
        "Test :",
        test["date"].min(),
        "to",
        test["date"].max(),
        "| rows =",
        len(test),
    )

    # ========================================================
    # 7. CHECK CLASS DISTRIBUTION
    # ========================================================

    train_positive = int(
        train[TARGET].sum()
    )

    test_positive = int(
        test[TARGET].sum()
    )

    train_negative = (
        len(train)
        - train_positive
    )

    test_negative = (
        len(test)
        - test_positive
    )

    print(
        "\nClass distribution:"
    )

    print(
        f"Train class 0: "
        f"{train_negative}"
    )

    print(
        f"Train class 1: "
        f"{train_positive}"
    )

    print(
        f"Test class 0:  "
        f"{test_negative}"
    )

    print(
        f"Test class 1:  "
        f"{test_positive}"
    )

    # --------------------------------------------------------
    # Important limitation
    # --------------------------------------------------------

    test_has_positive = (
        test_positive > 0
    )

    if not test_has_positive:
        print(
            "\nWARNING:"
        )
        print(
            "The chronological test period "
            "contains no positive cases."
        )
        print(
            "Positive-class test metrics are "
            "therefore not informative."
        )

    # ========================================================
    # 8. X / y
    # ========================================================

    X_train = train[
        FEATURES
    ]

    y_train = train[
        TARGET
    ].astype(int)

    X_test = test[
        FEATURES
    ]

    y_test = test[
        TARGET
    ].astype(int)

    # ========================================================
    # 9. FEATURE TYPES
    # ========================================================

    numeric_features = [
        c
        for c in FEATURES
        if c != "location"
    ]

    categorical_features = [
        "location"
    ]

    # ========================================================
    # 10. PREPROCESSING
    # ========================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scale",
                            StandardScaler()
                        ),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    # ========================================================
    # 11. LOGISTIC REGRESSION
    # ========================================================

    model = Pipeline(
        [
            (
                "preprocess",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # ========================================================
    # 12. BASELINE
    # ========================================================
    #
    # Assignment requirement:
    #
    # Classification → predict the majority class.
    #
    # The baseline is calculated using the training set,
    # then evaluated on the SAME test set as the model.
    # ========================================================

    majority_class = int(
        y_train.mode().iloc[0]
    )

    baseline_pred = np.full(
        len(y_test),
        majority_class,
    )

    baseline_metrics = metrics(
        y_test,
        baseline_pred,
    )

    # ========================================================
    # 13. TIMESERIES SPLIT
    # ========================================================
    #
    # Assignment requirement:
    #
    # Ordered data → TimeSeriesSplit
    #
    # No random KFold.
    # No random train_test_split.
    # ========================================================

    tscv = TimeSeriesSplit(
        n_splits=5
    )

    # Save information about every fold.
    fold_information = []

    for fold_number, (
        train_index,
        validation_index,
    ) in enumerate(
        tscv.split(X_train),
        start=1,
    ):

        y_fold_train = (
            y_train.iloc[train_index]
        )

        y_fold_test = (
            y_train.iloc[validation_index]
        )

        fold_information.append(
            {
                "fold": fold_number,
                "train_rows": len(
                    train_index
                ),
                "validation_rows": len(
                    validation_index
                ),
                "train_positive": int(
                    y_fold_train.sum()
                ),
                "validation_positive": int(
                    y_fold_test.sum()
                ),
            }
        )

    print(
        "\nTimeSeriesSplit folds:"
    )

    for fold in fold_information:

        print(
            f"Fold {fold['fold']}: "
            f"train={fold['train_rows']}, "
            f"validation={fold['validation_rows']}, "
            f"train positives={fold['train_positive']}, "
            f"validation positives={fold['validation_positive']}"
        )

    # ========================================================
    # 14. CROSS-VALIDATION
    # ========================================================

    cv = cross_validate(
        model,
        X_train,
        y_train,
        cv=tscv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy":
                "balanced_accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=-1,
        error_score="raise",
    )

    cv_mean = {
        k.replace(
            "test_",
            ""
        ): float(
            np.mean(v)
        )
        for k, v in cv.items()
        if k.startswith("test_")
    }

    # ========================================================
    # 15. FIT FINAL MODEL
    # ========================================================

    model.fit(
        X_train,
        y_train,
    )

    # ========================================================
    # 16. PREDICT TEST SET
    # ========================================================

    pred = model.predict(
        X_test
    )

    prob = model.predict_proba(
        X_test
    )[:, 1]

    # ========================================================
    # 17. TEST METRICS
    # ========================================================

    model_metrics = metrics(
        y_test,
        pred,
        prob,
    )

    # ========================================================
    # 18. CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_test,
        pred,
        labels=[0, 1],
        target_names=[
            "<=37.5",
            ">37.5",
        ],
        zero_division=0,
        output_dict=True,
    )

    # ========================================================
    # 19. RESULTS
    # ========================================================

    results = {

        "target":
            TARGET,

        "target_definition":
            "Whether tomorrow's daily mean "
            "PM2.5 exceeds 37.5 µg/m³",

        "prediction_moment":
            "end of today's daily observation",

        "split_strategy":
            "Chronological split: final 20% "
            "of ordered calendar dates as test",

        "cv_strategy":
            "TimeSeriesSplit with 5 splits",

        "train_rows":
            len(train),

        "test_rows":
            len(test),

        "train_start":
            str(
                train["date"].min()
            ),

        "train_end":
            str(
                train["date"].max()
            ),

        "test_start":
            str(
                test["date"].min()
            ),

        "test_end":
            str(
                test["date"].max()
            ),

        "split_date":
            str(
                pd.Timestamp(
                    split_date
                ).date()
            ),

        # ----------------------------------------------------
        # Class distribution
        # ----------------------------------------------------

        "train_class_0":
            train_negative,

        "train_class_1":
            train_positive,

        "test_class_0":
            test_negative,

        "test_class_1":
            test_positive,

        "test_has_positive_class":
            test_has_positive,

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        "baseline_majority_class":
            majority_class,

        "baseline":
            baseline_metrics,

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "model":
            model_metrics,

        # ----------------------------------------------------
        # CV
        # ----------------------------------------------------

        "cv_mean":
            cv_mean,

        "cv_folds":
            fold_information,

        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        "classification_report":
            report,

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        "features":
            FEATURES,

        # ----------------------------------------------------
        # Leakage
        # ----------------------------------------------------

        "leakage_check":
            {
                "pm25_tomorrow_used_as_feature":
                    False
            },

        # ----------------------------------------------------
        # Sorting
        # ----------------------------------------------------

        "sorting":
            {
                "sorted_before_split":
                    True,

                "sort_columns":
                    [
                        "date",
                        "location",
                    ],
            },

        # ----------------------------------------------------
        # Limitation
        # ----------------------------------------------------

        "evaluation_limitation":
            (
                "The final chronological test "
                "period contains no positive "
                "high-PM2.5 cases. Therefore, "
                "positive-class test precision, "
                "recall, and F1 are not "
                "informative for this test period. "
                "The split was retained because "
                "the assignment requires the "
                "later portion of time-ordered "
                "data to be used as the test set."
                if not test_has_positive
                else None
            ),
    }

    # ========================================================
    # 20. SAVE JSON
    # ========================================================

    (
        RESULTS / "metrics.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 21. SAVE CSV
    # ========================================================

    pd.DataFrame(
        [
            {
                "method":
                    "majority_baseline",
                **baseline_metrics,
            },
            {
                "method":
                    "logistic_regression",
                **model_metrics,
            },
        ]
    ).to_csv(
        RESULTS / "metrics.csv",
        index=False,
    )

    # ========================================================
    # 22. PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "BASELINE"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            baseline_metrics,
            indent=2,
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LOGISTIC REGRESSION"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            model_metrics,
            indent=2,
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TIME SERIES CV"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            cv_mean,
            indent=2,
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "EVALUATION LIMITATION"
    )

    print(
        "=" * 60
    )

    if not test_has_positive:

        print(
            "The final chronological test "
            "period contains 0 positive cases."
        )

        print(
            "Do NOT interpret the high test "
            "accuracy as evidence that the model "
            "can detect high-PM2.5 days."
        )

    else:

        print(
            "The test set contains positive cases."
        )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS / "metrics.json"
    )

    print(
        RESULTS / "metrics.csv"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
