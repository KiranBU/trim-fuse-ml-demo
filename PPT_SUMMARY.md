# Slide-generation brief: TrimFuse AI

Create a clear 12–14 slide technical project presentation for a 45-minute university presentation titled “AI-Optimized Trim & Fuse for Derivative Chips: MCU Bandgap Predictive Trim.” Allocate roughly 40 minutes to presentation and live implementation walkthrough and 5 minutes to technical questions. Use concise diagrams, one validation chart/table, and readable text. Distinguish documented facts from synthetic demonstration results. Do not invent citations, data, or production claims.

## Project summary

The project uses the Trim & Fuse section of the supplied Test Engineering Solutions archive and the supplied AI-Optimized Trim & Fuse reference. The archive contains a synthetic reproducible dataset with 8,420 rows and 42 columns for three MCU derivatives. The focused experiment uses 6,315 MCU-A dies across 15 wafers and predicts the six-bit bandgap trim code from nine pre-trim measurements: untrimmed bandgap, five PCM measurements, and wafer x/y/radius. The output is a candidate code, not an automatic fuse write.

## Problem and motivation

Conventional trim performs repeated measure-adjust-measure searches. The paper describes predictive trim as a candidate code followed by a tester verification and fallback search if needed. A wafer is a natural validation group because dies on one wafer share process variation; random row splits could leak wafer information.

## Data preparation and quality

The source CSV contains 8,420 rows and 42 columns. Checks find zero missing cells, zero exact duplicate rows, zero repeated die IDs, legal whole-number target values in 0–63, finite numeric values, 15 MCU-A wafers, and wafer-radius consistency with x/y within about 0.005 mm. Exact duplicates would be removed; conflicting repeated die IDs are flagged for review; missing targets are excluded; predictor gaps are median-imputed within training folds. IQR outliers are flagged but not automatically removed because rare process values may be valid. No actual imputation or duplicate cleanup was needed for the provided file.

Exclude the target, post-trim/final readings, fuse map, pass/fail outcomes, identifiers, and conventional iteration/time values from model features. Emphasize this leakage prevention.

## Training and model selection

Hold out 25% of MCU-A wafers (3 whole wafers, 1,263 dies) with a fixed seed. Use 3-fold GroupKFold on the remaining 12 wafers to compare five Extra Trees / HistGradientBoosting settings. Rank by within-one-code rate, then MAE, then exact-code accuracy. The selected ExtraTreesRegressor uses 300 trees and min_samples_leaf=1. It handles nonlinear tabular relationships, requires no scaling, and performed better than the supplied HGB baseline in the grouped CV and fixed holdout. Round predictions and clip to legal 0–63.

## Verified results to display

On the held-out wafers, selected Extra Trees achieved:

- Exact-code accuracy: 78.86%
- Within ±1 code: 99.68%
- MAE: 0.222 code
- RMSE: 0.525 code
- Predict-then-verify iteration estimate: 1.23 average

The included HistGradientBoosting baseline on the same split achieved 75.77% exact accuracy, 97.94% within one code, MAE 0.280, RMSE 0.686, and estimated 1.35 iterations. Conventional search averaged 6.60 iterations and 28.68 ms per die on these test wafers. The 1.23 value implies about 81.4% fewer iterations under the reference rule. State clearly that this is an iteration estimate, not a measured tester-time reduction. The dataset and scores are synthetic and cannot establish real production yield, DPPM, tester savings, or safety.

## System flow and code walkthrough

Show this flow: CSV → schema/quality checks → MCU-A filter → training-only median imputer → wafer-grouped CV → final Extra Trees fit → untouched wafer evaluation → model and JSON/CSV reports → Streamlit input form → legal rounded code → tester verification/fallback.

Key files: train_model.py contains preprocessing, fixed grouped split, model comparison, metrics, and artifact writing; artifacts/metrics.json contains scores and split details; app.py collects nine measurements and displays the prediction and out-of-range warning; notebooks/trim_fuse_training_colab.ipynb invokes the same trainer in Colab; tests/test_pipeline.py checks data quality, group isolation, and legal inference output.

## UI demonstration

Demonstrate the Streamlit page: enter nine pre-trim features, click Predict candidate code, observe a 0–63 code and held-out metrics, then move one feature outside training ranges to trigger the warning. Explain that the interface never programs fuses. Its only output is a candidate for a controlled tester flow.

## End-to-end workflow

Explain prompt design by specifying the target, valid pre-trim features, leakage exclusions, missing/duplicate rules, group split, metrics, safety boundary, and artifact requirements. Then show Colab notebook training, VS Code reuse of train_model.py, smoke tests, GitHub push, and Streamlit Community Cloud deployment from app.py with requirements.txt. The repository and deployed URL require the presenter's GitHub and Streamlit account setup; do not invent or display placeholder links as live deployments.

## Suggested slide order

1. Title and presenter
2. Test-engineering problem and motivation
3. Reference approach: predict, verify, fallback
4. Dataset choice and synthetic-data note
5. Target and feature definition
6. Preprocessing and data-quality checks
7. Leakage exclusions and wafer-aware split
8. Candidate models and selection rule
9. Holdout results vs reference HGB baseline
10. Architecture and data flow
11. Key code components
12. Streamlit UI live walk-through
13. Colab → VS Code → GitHub → Streamlit deployment
14. Limitations, safeguards, and future work / Q&A

## Likely technical questions

Prepare to explain why MCU-A only, why regression for an ordered code, why Extra Trees, why wafer-grouped splitting, how missingness/duplicates/outliers were handled, what 99.68% within-one-code means, how leakage was avoided, why the synthetic data is limited, why the app does not program fuses, and what real-hardware work is required before production. Use the detailed answers in PRESENTATION_GUIDE.md.
