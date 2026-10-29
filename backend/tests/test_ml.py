import unittest

from backend.app.ml.trainer import TrainingDataError, require_target_column


class TrainingTargetTests(unittest.TestCase):
    def test_source_categories_do_not_satisfy_wqi_target(self):
        with self.assertRaisesRegex(TrainingDataError, "numeric 'wqi' target"):
            require_target_column(("quality_class_source", "risk_class_source"))

    def test_numeric_wqi_target_is_required_by_name(self):
        require_target_column(("quality_class_source", "wqi"))


if __name__ == "__main__":
    unittest.main()