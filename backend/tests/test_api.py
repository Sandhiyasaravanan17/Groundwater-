import unittest
import math
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import main as api
from backend.app.main import app
from backend.app.wqi.analysis import WQI_OUTPUT_FIELDS


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_summary_reports_calculated_wqi_and_real_records(self):
        response = self.client.get("/api/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["record_count"], 1138)
        self.assertEqual(data["wqi"]["status"], "calculated")
        self.assertEqual(data["wqi_summary"]["record_count"], 1138)
        self.assertEqual(data["models"]["status"], "available")

    def test_dataset_is_paginated_and_supports_filters(self):
        response = self.client.get("/api/dataset?page=1&page_size=10&year=2018&season=Pre-Monsoon")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertLessEqual(len(data["records"]), 10)
        self.assertTrue(all(record["year"] == 2018 for record in data["records"]))
        self.assertTrue(all(record["season"] == "Pre-Monsoon" for record in data["records"]))

    def test_spatial_endpoint_caps_points_and_reports_coordinate_review(self):
        response = self.client.get("/api/spatial?year=2018&season=Pre-Monsoon")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertLessEqual(len(data["points"]), 500)
        self.assertIn("years", data["metadata"]["available_filters"])
        bounds = data["metadata"]["coordinate_bounds"]
        self.assertLess(bounds["latitude_min"], 9)
        self.assertLess(bounds["longitude_min"], 78)
        self.assertTrue(all(point["observation_status"] == "observed" for point in data["points"]))
        self.assertTrue(all(point["wqi"] is not None and point["wqi_class"] for point in data["points"]))
        self.assertNotIn("raw_field_01", data["points"][0])
        self.assertNotIn("parameter_text_group_1_raw", data["points"][0])

    def test_temporal_endpoint_is_observed_only(self):
        response = self.client.get("/api/temporal")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["observed_only"])
        self.assertEqual(data["time_coverage"]["years"], list(range(2018, 2026)))
        self.assertGreater(data["annual_parameter_summary"][0]["parameters"]["wqi"]["count"], 0)
        self.assertTrue(data["annual_wqi_class_distribution"])

    def test_report_endpoint_includes_provenance_and_limits(self):
        response = self.client.get("/api/reports")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["summary"]["record_count"], 1138)
        self.assertIn("field_dictionary", data["preprocessing_audit"])
        self.assertTrue(data["temporal"]["observed_only"])
        self.assertEqual(set(data["model_metrics"]["metrics"]), {"Random Forest", "XGBoost"})
        self.assertTrue(data["global_shap"]["feature_importance"])
        self.assertTrue(any("coordinates" in item for item in data["limitations"]))

    def test_wqi_endpoint_returns_scores_and_contributions(self):
        response = self.client.get("/api/wqi?page=1&page_size=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_records"], 1138)
        self.assertEqual(len(data["records"]), 10)
        record = data["records"][0]
        self.assertIsInstance(record["wqi"], float)
        self.assertEqual(len(record["wqi_contributions"]), 6)
        self.assertIn("quality_class_source", record)

    def test_first_wqi_request_prepares_observations_before_calculation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            processed_file = output_directory / "groundwater_observations.csv"
            wqi_file = output_directory / "water_quality_index.csv"
            summary_file = output_directory / "wqi_summary.json"
            observation = {
                "source_dataset": "sample.csv",
                "source_record_number": "1",
            }

            def prepare_observations():
                processed_file.write_text("source_dataset,source_record_number\nsample.csv,1\n")
                return [observation]

            def calculate_wqi(observations_path, target_directory):
                self.assertTrue(observations_path.exists())
                with wqi_file.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=WQI_OUTPUT_FIELDS)
                    writer.writeheader()
                    writer.writerow({
                        "source_dataset": "sample.csv",
                        "source_record_number": "1",
                        "wqi": "10",
                        "wqi_contributions": "{}",
                    })
                summary_file.write_text(json.dumps({"record_count": 1}))

            api.load_records.cache_clear()
            api.load_wqi_records.cache_clear()
            try:
                with (
                    patch.object(api, "PROCESSED_FILE", processed_file),
                    patch.object(api, "WQI_FILE", wqi_file),
                    patch.object(api, "WQI_SUMMARY_FILE", summary_file),
                    patch.object(api, "PROCESSED_DIR", output_directory),
                    patch.object(api, "load_records", side_effect=prepare_observations),
                    patch.object(api, "write_wqi_outputs", side_effect=calculate_wqi),
                ):
                    result = api.load_wqi_records()
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["wqi"], 10.0)
            finally:
                api.load_records.cache_clear()
                api.load_wqi_records.cache_clear()

    def test_models_evaluate_actual_wqi_and_predict_from_valid_features(self):
        metrics = self.client.get("/api/model-metrics")
        self.assertEqual(metrics.status_code, 200)
        result = metrics.json()
        self.assertEqual(result["target_column"], "wqi")
        self.assertEqual(result["test_records"], 242)
        self.assertEqual(set(result["metrics"]), {"Random Forest", "XGBoost"})
        self.assertIn(result["best_model"], result["metrics"])
        self.assertGreater(result["metrics"][result["best_model"]]["r2"], 0)
        self.assertEqual(
            result["best_model"],
            min(result["metrics"], key=lambda name: (
                result["metrics"][name]["rmse"],
                result["metrics"][name]["mae"],
            )),
        )
        self.assertEqual(
            result["model_selection"]["selected_model"],
            result["best_model"],
        )
        self.assertEqual(
            result["feature_columns"],
            [
                "nitrate_mg_l", "fluoride_mg_l", "chloride_mg_l", "tds_mg_l",
                "ph", "iron_mg_l", "alkalinity_mg_l",
            ],
        )
        self.assertLessEqual(
            result["training_feature_ranges"]["nitrate_mg_l"]["minimum"],
            result["training_feature_ranges"]["nitrate_mg_l"]["maximum"],
        )

        observation = self.client.get("/api/dataset?page=1&page_size=1").json()["records"][0]
        prediction = self.client.post("/api/prediction", json={
            "year": observation["year"],
            "season": observation["season"],
            "well_type": observation["well_type"],
            "block": observation["block"],
        })
        self.assertEqual(prediction.status_code, 200)
        self.assertEqual(prediction.json()["prediction_type"], "predicted")
        self.assertIn(prediction.json()["predicted_class"], {
            "Excellent", "Good", "Poor", "Very Poor", "Unsuitable for Drinking"
        })
        self.assertLess(prediction.json()["model_performance"]["r2"], 0)
        local = prediction.json()["local_explanation"]
        self.assertEqual(local["model"], prediction.json()["model"])
        self.assertTrue(math.isclose(
            local["baseline_wqi"] + sum(item["shap_value"] for item in local["contributions"]),
            prediction.json()["predicted_wqi"],
            rel_tol=1e-4,
            abs_tol=1e-2,
        ))

    def test_chemistry_prediction_returns_formula_guidelines_and_domain_warnings(self):
        fields = [
            "nitrate_mg_l", "fluoride_mg_l", "chloride_mg_l", "tds_mg_l",
            "ph", "iron_mg_l", "alkalinity_mg_l",
        ]
        ranges = self.client.get("/api/model-metrics").json()["training_feature_ranges"]
        inputs = {
            field: (ranges[field]["minimum"] + ranges[field]["maximum"]) / 2
            for field in fields
        }
        response = self.client.post("/api/prediction", json=inputs)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["model"], result["model_selection"]["selected_model"])
        self.assertEqual(len(result["parameter_assessments"]), 7)
        self.assertEqual(len(result["out_of_domain_features"]), 0)
        self.assertIsInstance(result["calculated_wqi"], float)
        self.assertIsInstance(result["calculated_class"], str)
        self.assertEqual(
            {item["field"] for item in result["parameter_assessments"]},
            set(fields),
        )
        self.assertEqual(result["parameter_assessments"][-1]["field"], "alkalinity_mg_l")
        self.assertTrue(
            result["parameter_assessments"][-1]["requires_basis_review"]
        )
        by_field = {
            item["field"]: item for item in result["parameter_assessments"]
        }
        self.assertIn("no relaxation", by_field["nitrate_mg_l"]["reference"])
        self.assertIn("no relaxation", by_field["ph"]["reference"])

        extrapolated = dict(inputs, tds_mg_l=10_000)
        extrapolated_result = self.client.post("/api/prediction", json=extrapolated)
        self.assertEqual(extrapolated_result.status_code, 200)
        self.assertIn(
            "tds_mg_l",
            {
                item["field"]
                for item in extrapolated_result.json()["out_of_domain_features"]
            },
        )

        invalid = dict(inputs, ph=14.1)
        self.assertEqual(
            self.client.post("/api/prediction", json=invalid).status_code,
            422,
        )
        missing = dict(inputs)
        del missing["iron_mg_l"]
        self.assertEqual(
            self.client.post("/api/prediction", json=missing).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/api/prediction",
                json={**inputs, "model": "Random Forest"},
            ).status_code,
            422,
        )

    def test_prediction_options_mark_sample_profiles_as_illustrative(self):
        response = self.client.get("/api/prediction/options")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["samples_are_illustrative"])
        self.assertEqual(len(data["sample_profiles"]), 6)
        self.assertTrue(
            all(set(profile["values"]) == set(api.CHEMISTRY_FEATURES)
                for profile in data["sample_profiles"])
        )
        expected_classes = {
            "good-quality": "Good",
            "moderate-quality": "Good",
            "poor-quality": "Poor",
        }
        for profile in data["sample_profiles"]:
            result = self.client.post("/api/prediction", json=profile["values"])
            self.assertEqual(result.status_code, 200, profile["label"])
            self.assertEqual(
                result.json()["model"],
                self.client.get("/api/model-metrics").json()["best_model"],
            )
            if profile["id"] in expected_classes:
                self.assertEqual(
                    result.json()["calculated_class"],
                    expected_classes[profile["id"]],
                )

    def test_global_shap_returns_actual_feature_rankings(self):
        response = self.client.get("/api/shap/global?model=XGBoost")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["target"], "wqi")
        self.assertEqual(data["validation_records"], 242)
        self.assertTrue(data["feature_importance"])
        self.assertEqual(data["feature_importance"][0]["feature"], "block")


if __name__ == "__main__":
    unittest.main()
