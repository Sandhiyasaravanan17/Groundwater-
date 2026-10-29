import unittest
from pathlib import Path
import tempfile
import csv

from backend.app.wqi.calculator import WQIMethodology, WQIMethodologyError
from backend.app.preprocessing.pipeline import run_pipeline
from backend.app.wqi.analysis import load_methodology, score_observations


class WQIMethodologyTests(unittest.TestCase):
    def test_missing_methodology_is_rejected(self):
        with self.assertRaises(WQIMethodologyError):
            WQIMethodology.from_mapping({})

    def test_missing_references_are_rejected(self):
        with self.assertRaisesRegex(WQIMethodologyError, "method_reference"):
            WQIMethodology.from_mapping({
                "standard_source": "source required",
                "classification_source": "source required",
            })

    def test_incomplete_parameter_limits_are_rejected(self):
        with self.assertRaisesRegex(WQIMethodologyError, "Parameter configuration mismatch"):
            WQIMethodology.from_mapping({
                "method_reference": "source required",
                "standard_source": "source required",
                "classification_source": "source required",
                "parameters": {},
                "classification_bands": [],
            })

    def test_configured_wqi_class_boundaries(self):
        methodology, _ = load_methodology()
        self.assertEqual(methodology.classify(25.0), "Excellent")
        self.assertEqual(methodology.classify(25.0001), "Good")
        self.assertEqual(methodology.classify(50.0), "Good")
        self.assertEqual(methodology.classify(50.0001), "Poor")
        self.assertEqual(methodology.classify(75.0), "Poor")
        self.assertEqual(methodology.classify(75.0001), "Very Poor")
        self.assertEqual(methodology.classify(100.0), "Very Poor")
        self.assertEqual(methodology.classify(100.0001), "Unsuitable for Drinking")

    def test_ph_rating_uses_both_sides_of_ideal(self):
        methodology, _ = load_methodology()
        measurements = {
            name: parameter.ideal_value
            for name, parameter in methodology.parameter_limits.items()
        }
        measurements["ph"] = 6.75
        below_ideal = methodology.calculate(measurements)["contributions"]["ph"]["quality_rating"]
        measurements["ph"] = 7.75
        above_ideal = methodology.calculate(measurements)["contributions"]["ph"]["quality_rating"]
        self.assertAlmostEqual(below_ideal, 50.0)
        self.assertAlmostEqual(above_ideal, 50.0)

    def test_real_records_receive_finite_auditable_scores(self):
        repository_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            run_pipeline(repository_root / "dataset", output_directory)
            with (output_directory / "groundwater_observations.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                records = list(csv.DictReader(stream))
        methodology, _ = load_methodology()
        scores = score_observations(records, methodology)
        self.assertEqual(len(scores), 1138)
        self.assertTrue(all(score["wqi"] >= 0 for score in scores))
        self.assertTrue(all(len(score["wqi_contributions"]) == 6 for score in scores))
        for score in scores:
            contributions = score["wqi_contributions"].values()
            self.assertAlmostEqual(
                sum(item["normalized_weight"] for item in contributions),
                1.0,
            )
            self.assertAlmostEqual(
                sum(item["wqi_contribution"] for item in contributions),
                score["wqi"],
            )
            self.assertEqual(methodology.classify(score["wqi"]), score["wqi_class"])
        self.assertIn("Unsuitable for Drinking", {score["wqi_class"] for score in scores})


if __name__ == "__main__":
    unittest.main()