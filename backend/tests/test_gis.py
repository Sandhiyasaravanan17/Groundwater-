import csv
import tempfile
import unittest
from pathlib import Path

from backend.app.gis.sampling import sample_map_points
from backend.app.preprocessing.pipeline import run_pipeline


class MapSamplingTests(unittest.TestCase):
    def test_full_dataset_is_deterministically_limited_and_stratified(self):
        repository_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            run_pipeline(repository_root / "dataset", output_directory)
            with (output_directory / "groundwater_observations.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                records = list(csv.DictReader(stream))

        result = sample_map_points(records, limit=500)
        repeat = sample_map_points(records, limit=500)
        points = result["points"]
        metadata = result["metadata"]

        self.assertEqual(len(points), 500)
        self.assertTrue(metadata["is_limited"])
        self.assertEqual(points, repeat["points"])
        self.assertEqual(
            len({(row["latitude"], row["longitude"]) for row in points}),
            len(points),
        )
        self.assertEqual(
            set(metadata["class_counts"]),
            set(metadata["display_class_counts"]),
        )
        self.assertTrue(all(row["observation_status"] == "observed" for row in points))
        self.assertIn("no study-area boundary", metadata["coordinate_note"])


if __name__ == "__main__":
    unittest.main()