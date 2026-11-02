# TrimFuse AI: MCU bandgap trim demo

This project predicts a candidate 6-bit MCU bandgap trim code from measurements available before trimming. It includes a data-quality audit, wafer-grouped model selection, a held-out wafer evaluation, a Streamlit interface, and a Google Colab notebook.

## Scope and source

The selected table is mcu_trim_fuse.csv from the supplied Test Engineering Solutions archive. It contains 8,420 synthetic MCU records and 42 columns across three derivatives. The experiment follows the MCU-A bandgap case from the supplied Trim & Fuse reference and its included baseline implementation. It filters to MCU-A (6,315 dies, 15 wafers) so the result can be compared fairly with that reference baseline.

All rows are synthetic. They are suitable for demonstrating a preprocessing and modeling workflow, not for claiming production performance. The reference recommends predicting a trim code, measuring the result on the tester, and falling back to search when verification fails. This app follows that flow and does not program fuses.

## Prediction task

- Input: untrimmed bandgap reading, five process-control-monitor values, and wafer x/y/radius.
- Target: bg_trim_code_opt, a whole-number 6-bit trim code from 0 to 63.
- Unit of generalization: a wafer. The same wafer is never split between training and test.
- Explicitly excluded: post-trim and final measurements, trim-code labels, fuse-map image, pass/fail outcomes, die ID, and conventional iteration/time fields. These are targets, outcomes, identifiers, or post-decision values and would leak information.

## Preprocessing and data-quality checks

The data-quality report is generated in artifacts/data_quality_report.json. Checks include required columns, shape, missing cells, exact duplicates, repeated die IDs, finite numeric values, legal integer target range, derivative counts, wafer counts, IQR outlier flags, and consistency of radius with the x/y coordinates.

On the supplied source data: 8,420 rows by 42 columns; zero blank cells; zero exact duplicate rows; zero duplicate die IDs; no non-finite values in numeric columns; all trim labels are integer codes in 0–63. The MCU-A subset contains 6,315 rows across 15 wafers. The radius check differs from sqrt(x²+y²) by at most 0.005 mm.

Cleaning rules are explicit:

1. Exact duplicate records are removed if present.
2. Repeated die IDs are reported as an error for review; records are not silently discarded because the source has no timestamp to select a canonical retest.
3. Rows with a missing target are excluded. Malformed, fractional, or out-of-range target labels fail fast. Predictor gaps, if present, are imputed with a median learned from the training fold only through a scikit-learn Pipeline.
4. Non-finite inputs and out-of-range or fractional trim labels fail fast.
5. IQR outliers are counted for review and kept unless an engineering rule proves a value is invalid. A real process extreme must not be erased merely because it is rare.
6. Wafer radius is checked against wafer x/y within 0.01 mm.

No predictor imputation or duplicate removal was needed for this supplied CSV because its quality checks passed. The median imputer remains in the pipeline so any future imputation is learned only from training data.

## Model selection and validation

The task is ordinal numeric prediction, so regressors preserve the distance between codes. Predictions are rounded and clipped to the legal 0–63 range. The final choice is ExtraTreesRegressor with 300 trees, one sample minimum per leaf, all features considered at each split, and a fixed random seed. Extra Trees averages randomized tree predictions, handles nonlinear tabular relationships, needs no scaling, and trains quickly for this small dataset.

The training portion is compared using 3-fold GroupKFold over wafer_id. The selection rule is highest within-one-code rate, then lowest mean absolute error, then highest exact-code rate. The model-selection scores are saved in artifacts/grouped_cv_scores.csv. The fixed 25% holdout uses three entirely unseen MCU-A wafers. The supplied baseline split is reproduced with seed 0 so its result is directly comparable.

The leading grouped-CV scores were 98.36% within one code for Extra Trees with 300 trees and leaf size 1, 97.98% for the 100-tree version, and 93.76% for the reference HGB model. The fixed holdout was evaluated only after selection.

| Metric on held-out wafers | Selected Extra Trees | Included HGB baseline |
|---|---:|---:|
| Exact trim-code accuracy | 78.86% | 75.77% |
| Within ±1 code | 99.68% | 97.94% |
| MAE | 0.222 code | 0.280 code |
| RMSE | 0.525 code | 0.686 code |
| Estimated search iterations | 1.23 | 1.35 |

The holdout contains 1,263 dies from three unseen wafers. The reference conventional flow averages 6.60 iterations and 28.68 ms per die on those wafers. The estimated predict-then-verify flow averages 1.23 iterations, about 81.4% fewer iterations. This is an iteration estimate, not measured tester wall-clock savings. The held-out error distribution and full precision metrics are in artifacts/metrics.json and artifacts/heldout_predictions.csv.

## Run in Google Colab

1. Download and extract this project ZIP locally.
2. Open notebooks/trim_fuse_training_colab.ipynb in Google Colab using File → Upload notebook.
3. Run the first cell and upload this project ZIP when prompted.
4. Run the remaining cells. The notebook installs requirements, runs the exact same train_model.py used in VS Code, displays the generated metrics, and runs smoke tests.

The notebook keeps training and reporting code in the shared Python script, which avoids a separate notebook-only model implementation.

## Run in VS Code on Windows

Open this folder in VS Code, then run these PowerShell commands from its integrated terminal:

    py -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python train_model.py
    python -m unittest discover -s tests -v
    python -m streamlit run app.py

The Streamlit app opens locally. Enter one die's pre-trim measurements and press Predict candidate code. It displays a rounded 0–63 suggestion, the held-out metrics, and an out-of-training-range warning. It never writes a fuse.

## Key files and architecture

- data/mcu_trim_fuse.csv: selected source table.
- train_model.py: input validation, cleaning, grouped split, model selection, final fit, evaluation, and artifact writing.
- artifacts/bandgap_trim_model.joblib: fitted model pipeline for the demo.
- artifacts/metrics.json: model, split, software versions, and evaluation metrics.
- artifacts/data_quality_report.json: data checks and cleaning audit.
- artifacts/grouped_cv_scores.csv: grouped model-comparison table.
- artifacts/heldout_predictions.csv: actual vs predicted codes for the untouched test wafers.
- app.py: interactive inference page.
- notebooks/trim_fuse_training_colab.ipynb: Colab wrapper around the shared training script.
- tests/test_pipeline.py: data-quality, group-leakage, and inference smoke tests.
- PRESENTATION_GUIDE.md: 45-minute talk track, live demo sequence, and technical Q&A.
- PPT_SUMMARY.md: concise facts and slide outline for another slide-generation tool.
- PROMPTS.md and SOURCE_NOTES.md: prompt templates and source attribution.

System flow: source CSV → validation and MCU-A selection → training-only median imputer → wafer-grouped model selection → final model fit → untouched wafer evaluation → serialized model and reports → Streamlit input → rounded candidate code → human/tester verification.

Architecture mapping: the paper's shared platform describes data, feature, IP-model/derivative, optimization, tester inference, and monitoring layers. This classroom implementation realizes a smaller version: the CSV is the data layer; fixed schema checks and nine pre-trim features form the feature layer; one MCU-A bandgap Extra Trees pipeline is the IP model; grouped CV chooses the model; Streamlit serves candidate inference; JSON/CSV results support review. A human or tester still verifies the measurement and falls back to search. There is no hardware connection, automatic fuse writer, or production monitoring loop in this demo.

## GitHub and deployment

Create a new GitHub repository named trim-fuse-ml-demo. From the project folder, after creating the empty repository on GitHub, run:

    git init
    git add .
    git commit -m "Build MCU trim prediction demo"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/trim-fuse-ml-demo.git
    git push -u origin main

The GitHub project link will be https://github.com/YOUR_USERNAME/trim-fuse-ml-demo after replacing YOUR_USERNAME.

For a public demo, sign in to Streamlit Community Cloud with GitHub, choose Create app, then select that repository, branch main, and entrypoint app.py. Keep requirements.txt at the repository root. Community Cloud assigns or lets you choose a streamlit.app subdomain; the actual deployed URL is produced after deployment.

Official walkthroughs: [GitHub: add locally hosted code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github) and [Streamlit: deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Do not commit account credentials, access tokens, customer data, or real device measurements. The included CSV is synthetic. A public deployment runs in a hosted service, so use synthetic inputs in the public demo and do not enter confidential production measurements.

## Testing and acceptance checks

The training command itself checks required columns, labels, bounds, duplicates, finite inputs, radius consistency, and split isolation before fitting. The smoke tests verify data shape and quality, no wafer overlap, no target/post-trim fields in the feature list, and legal model output. The reported result is accepted only if the test rows come from wafers absent from all model-selection folds and training rows.

## Limitations and safe use

- Data and reported scores are synthetic and cannot establish real silicon yield, DPPM, process robustness, or tester-time savings.
- The model is trained for MCU-A only; it does not claim transfer to MCU-B or MCU-C.
- The demo estimates a candidate code only. A tester must verify the trim and use conventional search on a failed check.
- One-time programmable fuses require separate rule-based validation, controlled release, and hardware-in-the-loop checks. Security fuses are outside this model.
- Before production, retrain on traceable real PCM and test data, validate on future lots and wafers, measure tester timing, monitor drift, document model versions, and obtain engineering qualification.
