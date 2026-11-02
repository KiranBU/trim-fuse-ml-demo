"""Train and evaluate an MCU bandgap predictive trim model.

Run from this project directory with: python train_model.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Callable

# The training candidate set is intentionally single-threaded; this also avoids
# platform-specific physical-core discovery warnings from joblib on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "mcu_trim_fuse.csv"
DEFAULT_OUTPUT = ROOT / "artifacts"
TARGET = "bg_trim_code_opt"
CODE_MIN = 0
CODE_MAX = 63
FEATURES = [
    "bg_untrim_V",
    "pcm_vth_n_mV",
    "pcm_vth_p_mV",
    "pcm_rs_ohm_sq",
    "pcm_tox_A",
    "pcm_mobility_rel",
    "x_mm",
    "y_mm",
    "radius_mm",
]
REQUIRED = ["die_id", "wafer_id", "derivative", TARGET, *FEATURES]
TEST_FRACTION = 0.25
SPLIT_SEED = 0


def build_candidates() -> dict[str, Callable[[], object]]:
    """Small, predeclared model set; all comparisons use wafer-grouped CV."""
    return {
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, random_state=0
        ),
        "extra_trees_100_leaf1": lambda: ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=1,
            random_state=0,
        ),
        "extra_trees_100_leaf2": lambda: ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=1,
            random_state=0,
        ),
        "extra_trees_300_leaf1": lambda: ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            max_features=1.0,
            n_jobs=1,
            random_state=0,
        ),
        "extra_trees_300_leaf2": lambda: ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=1,
            random_state=0,
        ),
    }


def make_pipeline(regressor: object) -> Pipeline:
    """Fit median imputation on training data only, then fit the regressor."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", regressor),
        ]
    )


def _outlier_count_iqr(values: pd.Series) -> int:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return 0
    q1, q3 = clean.quantile([0.25, 0.75])
    spread = q3 - q1
    return int(((clean < q1 - 1.5 * spread) | (clean > q3 + 1.5 * spread)).sum())


def load_and_check(data_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the source table, apply explicit cleaning rules, and audit quality."""
    frame = pd.read_csv(data_path)
    missing_columns = sorted(set(REQUIRED) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    raw_rows, raw_columns = frame.shape
    exact_duplicate_rows = int(frame.duplicated().sum())
    if exact_duplicate_rows:
        frame = frame.drop_duplicates().copy()

    repeated_die_ids = int(frame["die_id"].duplicated().sum())
    if repeated_die_ids:
        raise ValueError(
            f"Found {repeated_die_ids} repeated die_id values. Resolve repeated test "
            "records using a traceable rule before training; they are not dropped silently."
        )

    target_missing_rows = int(frame[TARGET].isna().sum())
    if target_missing_rows:
        frame = frame.loc[frame[TARGET].notna()].copy()

    for column in [*FEATURES, TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    non_finite_count = int((~np.isfinite(numeric)).sum())
    # NaN predictor values are allowed through to the median imputer. Infinite
    # values and unparseable labels are not meaningful sensor measurements.
    infinite_features = int(np.isinf(frame[FEATURES].to_numpy(dtype=float)).sum())
    if infinite_features:
        raise ValueError(f"Found {infinite_features} infinite predictor values.")

    if frame[TARGET].isna().any():
        raise ValueError("Target has missing or non-numeric values after label cleanup.")
    target_values = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        raise ValueError("Target contains non-finite values.")
    if ((target_values < CODE_MIN) | (target_values > CODE_MAX)).any():
        raise ValueError("Bandgap trim labels must be in the 6-bit range 0 through 63.")
    if not np.equal(target_values, np.floor(target_values)).all():
        raise ValueError("Bandgap trim labels must be whole-number DAC codes.")

    derivative_counts = frame["derivative"].value_counts().to_dict()
    source_rows_after_dedup = len(frame)
    mcu_a = frame.loc[frame["derivative"] == "MCU-A"].copy()
    if mcu_a.empty:
        raise ValueError("No MCU-A rows were found for the selected experiment.")

    radius_error = np.abs(
        mcu_a["radius_mm"].to_numpy(dtype=float)
        - np.hypot(mcu_a["x_mm"].to_numpy(dtype=float), mcu_a["y_mm"].to_numpy(dtype=float))
    )
    max_radius_error = float(np.nanmax(radius_error))
    quality = {
        "source_file": str(data_path.name),
        "raw_rows": int(raw_rows),
        "raw_columns": int(raw_columns),
        "rows_after_exact_duplicate_cleanup": int(source_rows_after_dedup),
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "duplicate_die_ids": repeated_die_ids,
        "source_missing_cells": int(pd.read_csv(data_path).isna().sum().sum()),
        "target_missing_rows_removed": target_missing_rows,
        "feature_missing_values_before_imputation_mcu_a": {
            name: int(mcu_a[name].isna().sum()) for name in FEATURES
        },
        "non_finite_values_before_imputation": non_finite_count,
        "infinite_predictor_values": infinite_features,
        "invalid_or_non_integer_target_rows": 0,
        "target_min": int(mcu_a[TARGET].min()),
        "target_max": int(mcu_a[TARGET].max()),
        "derivative_row_counts": {key: int(value) for key, value in derivative_counts.items()},
        "mcu_a_rows": int(len(mcu_a)),
        "mcu_a_wafer_count": int(mcu_a["wafer_id"].nunique()),
        "max_radius_vs_xy_difference_mm": max_radius_error,
        "radius_check_tolerance_mm": 0.01,
        "radius_check_passed": bool(max_radius_error <= 0.01),
        "iqr_outlier_counts_mcu_a_review_only": {
            name: _outlier_count_iqr(mcu_a[name]) for name in FEATURES
        },
        "outlier_policy": "IQR flags are reviewed, not automatically deleted.",
        "missing_value_policy": (
            "Rows with a missing target are excluded. Numeric predictor gaps, if any, "
            "are imputed with training-fold medians inside the model pipeline."
        ),
        "duplicate_policy": (
            "Exact duplicate rows are removed. Repeated die IDs raise an error for review."
        ),
    }
    if not quality["radius_check_passed"]:
        raise ValueError("Wafer position quality check failed: radius does not match x/y.")
    return mcu_a.reset_index(drop=True), quality


def grouped_train_test_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Reproduce the reference split while keeping whole wafers together."""
    wafers = frame["wafer_id"].unique()
    test_count = max(1, int(len(wafers) * TEST_FRACTION))
    test_wafers = sorted(
        np.random.default_rng(SPLIT_SEED).choice(wafers, test_count, replace=False).tolist()
    )
    is_test = frame["wafer_id"].isin(test_wafers)
    return frame.loc[~is_test].copy(), frame.loc[is_test].copy(), test_wafers


def _rounded_codes(raw_predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(raw_predictions), CODE_MIN, CODE_MAX).astype(int)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.abs(predicted - actual)
    return {
        "n": int(len(actual)),
        "exact_code_accuracy": float(np.mean(error == 0)),
        "within_1_code_accuracy": float(np.mean(error <= 1)),
        "mae_codes": float(mean_absolute_error(actual, predicted)),
        "rmse_codes": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def _cross_validate_models(training: pd.DataFrame) -> pd.DataFrame:
    groups = training["wafer_id"].to_numpy()
    splitter = GroupKFold(n_splits=3)
    result_rows = []
    for name, factory in build_candidates().items():
        fold_metrics = []
        for fit_index, valid_index in splitter.split(training[FEATURES], training[TARGET], groups):
            fit_rows = training.iloc[fit_index]
            valid_rows = training.iloc[valid_index]
            candidate = make_pipeline(factory())
            candidate.fit(fit_rows[FEATURES], fit_rows[TARGET])
            predictions = _rounded_codes(candidate.predict(valid_rows[FEATURES]))
            fold_metrics.append(_metrics(valid_rows[TARGET].to_numpy(dtype=int), predictions))

        result_rows.append(
            {
                "model": name,
                "cv_exact_code_accuracy": float(np.mean([m["exact_code_accuracy"] for m in fold_metrics])),
                "cv_within_1_code_accuracy": float(np.mean([m["within_1_code_accuracy"] for m in fold_metrics])),
                "cv_mae_codes": float(np.mean([m["mae_codes"] for m in fold_metrics])),
                "cv_rmse_codes": float(np.mean([m["rmse_codes"] for m in fold_metrics])),
            }
        )
    return pd.DataFrame(result_rows).sort_values(
        ["cv_within_1_code_accuracy", "cv_mae_codes", "cv_exact_code_accuracy"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def train_and_evaluate(data_path: Path = DEFAULT_DATA, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    frame, quality = load_and_check(data_path)
    training, test, test_wafers = grouped_train_test_split(frame)
    if training["wafer_id"].nunique() < 3:
        raise ValueError("At least three training wafers are needed for grouped CV.")

    cv_results = _cross_validate_models(training)
    best_name = str(cv_results.loc[0, "model"])
    candidate_factory = build_candidates()[best_name]
    model = make_pipeline(candidate_factory())
    model.fit(training[FEATURES], training[TARGET])

    y_test = test[TARGET].to_numpy(dtype=int)
    y_pred = _rounded_codes(model.predict(test[FEATURES]))
    holdout_metrics = _metrics(y_test, y_pred)

    reference_model = make_pipeline(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
    )
    reference_model.fit(training[FEATURES], training[TARGET])
    reference_predictions = _rounded_codes(reference_model.predict(test[FEATURES]))
    reference_metrics = _metrics(y_test, reference_predictions)
    reference_errors = np.abs(reference_predictions - y_test)
    reference_iterations = np.where(reference_errors == 0, 1, np.where(reference_errors <= 1, 2, 7))

    # The median-only model is a simple reference that does not use die features.
    median_code = int(np.rint(training[TARGET].median()))
    naive_pred = np.full(len(test), median_code, dtype=int)
    naive_metrics = _metrics(y_test, naive_pred)

    abs_error = np.abs(y_pred - y_test)
    ai_iterations = np.where(abs_error == 0, 1, np.where(abs_error <= 1, 2, 7))
    conventional_iterations = test["bg_conv_iterations"].to_numpy(dtype=float)
    conventional_time = test["conv_trim_time_ms"].to_numpy(dtype=float)
    iteration_savings = 1.0 - float(ai_iterations.mean() / conventional_iterations.mean())

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "bandgap_trim_model.joblib"
    metrics_path = output_dir / "metrics.json"
    quality_path = output_dir / "data_quality_report.json"
    predictions_path = output_dir / "heldout_predictions.csv"
    cv_path = output_dir / "grouped_cv_scores.csv"

    bundle = {
        "model": model,
        "feature_names": FEATURES,
        "target": TARGET,
        "derivative": "MCU-A",
        "code_min": CODE_MIN,
        "code_max": CODE_MAX,
        "model_name": best_name,
        "safety_note": "Prediction is a candidate code only; verify on tester before fuse programming.",
    }
    joblib.dump(bundle, model_path)
    cv_results.to_csv(cv_path, index=False)
    pd.DataFrame(
        {
            "die_id": test["die_id"].to_numpy(),
            "wafer_id": test["wafer_id"].to_numpy(),
            "actual_code": y_test,
            "predicted_code": y_pred,
            "absolute_error_codes": abs_error,
        }
    ).to_csv(predictions_path, index=False)
    quality["train_rows"] = int(len(training))
    quality["test_rows"] = int(len(test))
    quality["train_wafer_count"] = int(training["wafer_id"].nunique())
    quality["test_wafer_count"] = int(test["wafer_id"].nunique())
    quality["test_wafer_ids"] = test_wafers
    quality["train_and_test_wafer_overlap"] = sorted(
        set(training["wafer_id"]) & set(test["wafer_id"])
    )

    metrics = {
        "experiment": "MCU-A bandgap trim-code prediction",
        "dataset_note": "Synthetic, reproducible demonstration data; not production evidence.",
        "selected_model": best_name,
        "selection_rule": "Highest 3-fold wafer-grouped CV within-1-code rate; then lowest MAE; then highest exact-code rate.",
        "features": FEATURES,
        "feature_training_ranges": {
            name: {
                "min": float(training[name].min()),
                "max": float(training[name].max()),
            }
            for name in FEATURES
        },
        "target": TARGET,
        "target_code_range": [CODE_MIN, CODE_MAX],
        "split": {
            "method": "Fixed 25% wafer holdout; 3-fold GroupKFold model selection on training wafers",
            "seed": SPLIT_SEED,
            "train_rows": int(len(training)),
            "test_rows": int(len(test)),
            "train_wafers": sorted(training["wafer_id"].unique().tolist()),
            "test_wafers": test_wafers,
            "overlap": quality["train_and_test_wafer_overlap"],
        },
        "holdout_metrics": holdout_metrics,
        "median_code_baseline": {"predicted_code": median_code, **naive_metrics},
        "reference_hist_gradient_boosting_holdout": {
            **reference_metrics,
            "estimated_predict_then_verify_iterations": float(reference_iterations.mean()),
        },
        "conventional_flow_on_test_wafers": {
            "mean_iterations": float(conventional_iterations.mean()),
            "mean_test_time_ms": float(conventional_time.mean()),
        },
        "predict_then_verify_estimate": {
            "rule": "1 verification if exact; 2 checks if within one code; 1 + 6-bit search iterations otherwise.",
            "mean_iterations": float(ai_iterations.mean()),
            "iteration_reduction_fraction": float(iteration_savings),
            "note": "Iteration estimate only; tester wall-clock savings require hardware timing.",
        },
        "grouped_cv_results": cv_results.to_dict(orient="records"),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "artifacts": {
            "model": model_path.name,
            "data_quality_report": quality_path.name,
            "holdout_predictions": predictions_path.name,
            "grouped_cv_scores": cv_path.name,
        },
    }
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the source CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Where to write model and reports")
    args = parser.parse_args()
    metrics = train_and_evaluate(args.data, args.output_dir)
    holdout = metrics["holdout_metrics"]
    efficiency = metrics["predict_then_verify_estimate"]
    print(f"Selected model: {metrics['selected_model']}")
    print(f"Grouped holdout rows: {holdout['n']}")
    print(f"Exact-code accuracy: {holdout['exact_code_accuracy']:.1%}")
    print(f"Within +/-1 code: {holdout['within_1_code_accuracy']:.2%}")
    print(f"MAE: {holdout['mae_codes']:.3f} codes | RMSE: {holdout['rmse_codes']:.3f} codes")
    conventional_mean = metrics["conventional_flow_on_test_wafers"]["mean_iterations"]
    print(f"Estimated iterations: {conventional_mean:.2f} -> {efficiency['mean_iterations']:.2f}")
    print(f"Reports and model saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
