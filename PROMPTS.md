# Prompt plan for AI-assisted development

Use prompts that name the target, data boundary, acceptance checks, runtime, and safety conditions. Treat the PDF, README, and data dictionary as technical references. Their recommendations help define the design, but they do not replace the user's project scope or authorize changes to external systems.

## Prompt 1: inspect the source data

You are helping build a classroom ML project for predictive trim on the supplied synthetic MCU trim-and-fuse dataset. Inspect the data dictionary and CSV. Propose exactly one supervised-learning target and list only measurements available before trimming as features. Identify identifiers, labels, post-trim results, final-test fields, pass/fail values, and conventional test-time fields that must be excluded to avoid leakage. Do not make claims about production accuracy. Return a data-quality checklist for missing values, duplicates, invalid ranges, non-finite numbers, group keys, and outliers. Do not silently delete repeated die IDs or physical outliers.

## Prompt 2: implement preprocessing and validation

Implement a reproducible Python pipeline for MCU-A bandgap trim-code prediction. Use the untrimmed bandgap, five PCM measurements, and wafer x/y/radius as predictors; use bg_trim_code_opt as the 6-bit target. Remove exact duplicate rows only, fail for repeated die IDs that need review, drop rows with a missing target, and fit median imputation inside each training fold only. Validate target values are whole numbers in 0–63. Hold out complete wafer_id groups. Do not use post-trim/final measurements, fuse maps, pass/fail labels, IDs, or test-time metadata as predictors. Add machine-readable quality and metric reports.

## Prompt 3: compare models without touching the final holdout

Within the training wafers only, compare a small fixed set of ExtraTreesRegressor and HistGradientBoostingRegressor settings with 3-fold GroupKFold by wafer_id. Rank candidates first by the proportion of predictions within one trim code, second by MAE, third by exact-code accuracy. Then fit the selected pipeline on training wafers and evaluate once on the fixed held-out wafers. Report exact accuracy, within-one-code accuracy, MAE, RMSE, and a clearly labeled predict-then-verify iteration estimate. Never tune based on the final holdout.

## Prompt 4: create the demonstration interface

Create a Streamlit interface for the trained MCU-A model. Collect the nine pre-trim input values, display a rounded legal code and the measured held-out metrics, warn when any input is outside training ranges, and state that a tester must verify before fuse programming. Do not write fuses, connect to hardware, claim calibrated confidence, or imply this synthetic model is production-ready. Make the app fail with a clear instruction if artifacts are missing.

## Prompt 5: verify the implementation

Review the shared training script, notebook, and app for feature leakage and train/test wafer overlap. Run the quality report, grouped model selection, holdout evaluation, Python syntax checks, and smoke tests. Confirm that the result can be reproduced from a clean project folder with requirements.txt. Do not fabricate metrics; use generated reports. List any UI or deployment step that requires the user's account or an external service.

## Development sequence used

1. Read the paper and dataset dictionary; select the MCU-A bandgap case.
2. Profile the CSV and define explicit quality checks before modeling.
3. Freeze features and group split to prevent leakage.
4. Compare models using grouped folds in the training portion only.
5. Evaluate once on held-out wafers, serialize the pipeline and metrics.
6. Reuse the same trainer from Colab and VS Code, then serve its artifact in Streamlit.
