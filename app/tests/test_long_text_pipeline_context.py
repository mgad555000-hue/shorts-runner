import sys
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from long_text_review_runtime_final import _validate_selected_ids  # noqa: E402


class _Context:
    topic_ids = {1502, 1501}


class PipelineContextCompatibilityTests(unittest.TestCase):
    def test_runtime_accepts_the_set_produced_by_pipeline_context(self):
        self.assertEqual([1501, 1502], _validate_selected_ids(_Context()))


if __name__ == "__main__":
    unittest.main()
