"""Smoke tests for data quality, grouped validation, and inference range."""

import unittest

import joblib
import pandas as pd

from train_model import DEFAULT_DATA, DEFAULT_OUTPUT, FEATURES, grouped_train_test_split, load_and_check


class TrimFusePipelineTests(unittest.TestCase):
    def test_source_quality_checks(self):
        frame, report = load_and_check(DEFAULT_DATA)
        self.assertEqual(report["raw_rows"], 8420)
        self.assertEqual(report["raw_columns"], 42)
        self.assertEqual(report["exact_duplicate_rows_removed"], 0)
        self.assertEqual(report["duplicate_die_ids"], 0)
        self.assertEqual(report["source_missing_cells"], 0)
        self.assertEqual(len(frame), 6315)

    def test_wafer_split_has_no_group_overlap(self):
        frame, _ = load_and_check(DEFAULT_DATA)
        training, testing, _ = grouped_train_test_split(frame)
        self.assertFalse(set(training["wafer_id"]) & set(testing["wafer_id"]))
        leakage_fields = {
            "bg_trim_code_opt", "bg_trimmed_sort_V", "bg_final_V", "fuse_map_hex",
            "final_pass", "sort_pass", "bg_conv_iterations", "conv_trim_time_ms", "die_id",
        }
        self.assertTrue(set(FEATURES).isdisjoint(leakage_fields))

    def test_saved_model_returns_legal_code(self):
        model_path = DEFAULT_OUTPUT / "bandgap_trim_model.joblib"
        self.assertTrue(model_path.exists(), "Run python train_model.py before smoke tests.")
        bundle = joblib.load(model_path)
        sample = pd.read_csv(DEFAULT_DATA).query("derivative == 'MCU-A'").iloc[[0]]
        prediction = bundle["model"].predict(sample[FEATURES])[0]
        rounded = round(float(prediction))
        self.assertGreaterEqual(rounded, bundle["code_min"])
        self.assertLessEqual(rounded, bundle["code_max"])


if __name__ == "__main__":
    unittest.main()
