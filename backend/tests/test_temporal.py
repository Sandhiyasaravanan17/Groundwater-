import csv
import tempfile
import unittest
from pathlib import Path

from backend.app.preprocessing.pipeline import run_pipeline
from backend.app.temporal.analysis import analyze_temporal_records


class TemporalAnalysisTests(unittest.TestCase):
    def test_actual_records_produce_observed_year_and_season_summaries(self):
        repository_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            run_pipeline(repository_root / "dataset", output_directory)
            with (output_directory / "groundwater_observations.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                records = list(csv.DictReader(stream))

        summary = analyze_temporal_records(records)
        self.assertTrue(summary["observed_only"])
        self.assertEqual(summary["time_coverage"]["years"], list(range(2018, 2026)))
        self.assertEqual(len(summary["time_coverage"]["seasons"]), 4)
        self.assertEqual(len(summary["annual_parameter_summary"]), 8)
        self.assertEqual(len(summary["year_season_parameter_summary"]), 32)
        self.assertEqual(summary["records_with_invalid_year"], 0)
        self.assertNotIn("wqi", summary)


if __name__ == "__main__":
    unittest.main()