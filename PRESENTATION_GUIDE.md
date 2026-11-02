# 45-minute presentation guide

## Suggested running order

| Time | Topic | What to show |
|---|---|---|
| 0–3 min | Project question | Why iterative trim testing costs time; the narrow question: predict an MCU-A bandgap code from pre-trim information. |
| 3–7 min | Reference and scope | The Trim & Fuse paper's predictive-trim and predict-then-verify approach. Explain that the supplied dataset is synthetic. |
| 7–12 min | Dataset | 8,420 rows, 42 columns, four chip families in the archive; this experiment uses MCU-A only: 6,315 dies across 15 wafers. Show data dictionary roles. |
| 12–18 min | Cleaning and data quality | Show missingness, duplicate checks, label range, finite-value check, radius check, IQR review-only policy, and why no data was imputed or removed in this supplied file. |
| 18–23 min | Features and model selection | Explain leakage exclusions, ordinal target, Extra Trees, 3-fold GroupKFold on training wafers, and why the final wafers stay untouched. |
| 23–28 min | Training results | Show grouped CV comparison and final holdout metrics; compare the selected model with the supplied HGB baseline. State synthetic-data limitation. |
| 28–34 min | Code and architecture | Walk train_model.py functions, artifact files, app.py inference, and data flow from CSV to candidate code. |
| 34–39 min | Live UI demonstration | Open Streamlit, submit sample values, inspect predicted code and OOD warning, explain verification/fallback. Do not claim it programs a fuse. |
| 39–40 min | Reproducibility and deployment | Show Colab notebook, VS Code commands, repository structure, and planned Streamlit Community Cloud deployment. |
| 40–45 min | Questions | Use the technical Q&A below. Keep a screenshot and saved metrics ready if Wi-Fi or app launch fails. |

## Opening in plain language

"Traditional trim search repeatedly measures a device, changes a code, and measures again. This project tests whether pre-trim electrical and wafer-position measurements can predict a useful starting code. The model is only a candidate generator. The tester verifies the result and falls back to search when it misses."

## Data-quality explanation

Describe the attached data as synthetic and reproducible, not as factory evidence. The source table has 8,420 records and 42 columns. The MCU-A experiment uses 6,315 records on 15 wafers. It has no missing cells, exact duplicate rows, or duplicate die IDs. The target is a legal integer from 0 through 63. Wafer radius agrees with x/y within about 0.005 mm. IQR flags are inspected but not automatically deleted, because rare process measurements may be valid silicon variation. The median imputer sits inside the model pipeline; on this source CSV it does not change any values.

Explain duplicate handling carefully: exact duplicate rows can be removed, but repeated die IDs cause a review error. Without a timestamp or tester sequence, choosing which retest to keep would be arbitrary.

## Feature and leakage explanation

The nine features are bandgap before trim; NMOS and PMOS threshold voltage; poly sheet resistance; oxide thickness; relative mobility; wafer x; wafer y; and wafer radius. The target is bg_trim_code_opt. Post-trim and final measurements, the target code itself, fuse_map_hex, pass/fail results, die ID, and conventional test-iteration/time fields are excluded from model inputs. Several are labels or outcomes that would reveal information unavailable at prediction time.

## Results to state accurately

On 1,263 MCU-A dies from three wafers excluded from training and model selection, Extra Trees achieved 78.86% exact-code accuracy, 99.68% within one code, MAE 0.222 code, and RMSE 0.525 code. The included HGB baseline on the same split achieved 75.77% exact and 97.94% within one code. On those test wafers, conventional search averaged 6.60 iterations. The stated 1.23 AI iterations is calculated using the reference predict-then-verify rule; it is not a physical tester timing measurement. The data is synthetic, so these figures demonstrate the workflow only.

## Live demonstration sequence

1. Start the app with python -m streamlit run app.py.
2. Point out the sidebar's held-out wafer metrics and selected model.
3. Use the pre-filled sample or enter nine measurements from one row in the CSV.
4. Select Predict candidate code and read the legal 0–63 output.
5. Change a feature to a value outside the training range and show the warning.
6. Explain the real operational next step: apply the code in a controlled test flow, measure the trimmed output, and use a local or full conventional search if verification fails. The app itself does not program fuses.

## System architecture and code walkthrough

CSV input → schema and quality checks in load_and_check → MCU-A selection → wafer-grouped holdout → GroupKFold model comparison on training wafers → fit final Extra Trees pipeline → evaluate once on unseen wafers → write model, quality report, CV scores, metrics, and predictions → Streamlit collects nine values → pipeline imputes if needed → regression output is rounded and clipped to 0–63 → tester/human verification.

Walkthrough anchors:

- train_model.py: FEATURES and TARGET define the contract; load_and_check records quality; grouped_train_test_split holds out whole wafers; _cross_validate_models ranks candidates; train_and_evaluate writes all results.
- artifacts/metrics.json: result, split IDs, software versions, and iteration estimate.
- app.py: form collection, model load, candidate prediction, training-range warning, and fuse-safety message.
- notebooks/trim_fuse_training_colab.ipynb: installs dependencies and invokes the shared trainer so Colab and VS Code use the same implementation.
- tests/test_pipeline.py: data-quality, no-wafer-overlap, and legal-inference checks.

## Technical questions and prepared answers

### Why choose this dataset?

The archive's Trim & Fuse section contains a data dictionary, a reproducible synthetic dataset, baseline code, and explicit wafer grouping information. MCU-A bandgap trim is narrow enough to explain and is a documented use case in the reference. This is useful for demonstrating the workflow, but a production study needs real traceable tester data.

### Why only MCU-A?

The included reference baseline trains and tests on held-out MCU-A wafers. Restricting the experiment to the same derivative makes a fair comparison and avoids mixing package/specification differences before the basic method is understood. Transfer to MCU-B or MCU-C is future work.

### Why predict a numeric code instead of classify each code?

Codes are ordered and nearby codes are operationally closer than distant codes. Regression preserves that distance. After predicting, the output is rounded and clipped to the physical 6-bit range. Exact and within-one-code rates both show how the rounded code performs.

### Why Extra Trees?

It handles nonlinear relationships and feature interactions in small tabular data, requires no scaling, and trains quickly. Grouped CV selected the 300-tree, one-sample-leaf version by the preregistered within-one-code metric, with MAE as the next criterion. It improved held-out results over the supplied HGB baseline on the same wafer split.

### Why not a neural network?

The dataset is modest and tabular. Tree ensembles are a strong, simpler fit, require little scaling or architecture tuning, and are easier to reproduce in this project. A neural network could be compared later under the same wafer-grouped validation.

### Why group by wafer?

Dies from one wafer share process conditions and spatial patterns. A random row split could put near-related dies in both training and test, making the result too optimistic. Grouping tests generalization to wafers the model did not see.

### How did you handle missing values?

The supplied file has zero missing cells. The code drops target-missing rows if they appear and keeps median imputation for predictors inside the scikit-learn pipeline, so medians are learned from each training fold only. The test data never determines its own imputation values.

### What happened to duplicates and outliers?

The source has no exact duplicate rows and no repeated die IDs. Exact row duplicates can be removed; repeated IDs raise a review error because there is no timestamp to identify the canonical retest. IQR outliers are counted for review but retained unless an engineering bound proves them invalid.

### How do you prevent target leakage?

Only nine pre-trim measurements enter the model. The target, post-trim/final readings, fuse image, pass/fail labels, die ID, and test iteration/time fields are excluded. The code records the fixed feature list and tests it against known leakage fields.

### What does 99.68% accuracy mean?

It means 99.68% of the 1,263 held-out die predictions were within one code of the known exhaustive-search label. Exact-code accuracy was 78.86%. Both are measured on synthetic held-out wafers and do not establish production yield or escape rates.

### Is the 81% reduction a measured test-time saving?

No. It is an estimate from the reference predict-then-verify rule: one check for an exact code, two for a one-code miss, and a 7-iteration fallback otherwise. Physical tester timing must be measured in hardware. The source conventional flow reports 6.60 iterations and 28.68 ms per die on the holdout.

### Does the app program an OTP fuse?

No. It returns a candidate code only. OTP is irreversible, so a real system must verify with a measurement, apply range and schema checks, follow release controls, and use conventional search on a failed check. Security fuses are not in the AI path.

### What would be needed for production?

Real traceable PCM, trim, packaging, final-test and lot data; forward validation on new lots; calibrated tester timing; process drift monitoring; model version control; safety/rule checks; hardware-in-the-loop verification; and engineering qualification. The synthetic result is not sufficient.

### What is the deployment architecture?

GitHub stores the app, dependencies, synthetic CSV, and fitted model artifact. Streamlit Community Cloud builds the Python app from app.py and requirements.txt. The app runs inference on user-entered values. No backend device or fuse interface is connected.

## If the live demo fails

Open artifacts/metrics.json and artifacts/heldout_predictions.csv, show the recorded result, then walk the audience through app.py. Explain that live service availability is separate from model validation. Do not claim deployment succeeded unless the public app URL opens.
