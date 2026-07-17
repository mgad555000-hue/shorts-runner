import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from engine import EngineError  # noqa: E402
from long_text_reviewer import _issue  # noqa: E402
from long_text_semantic_reviewer import (  # noqa: E402
    action_long_text_review_build_prompts,
    action_long_text_review_parse_verdicts,
)
from tests.test_long_text_semantic_global import (  # noqa: E402
    FakeContext,
    TEST_INSTRUCTIONS,
    global_verdict,
    local_verdict,
    make_cards,
)


class LongTextPolicyGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp.name)
        self.ctx = FakeContext(self.output_dir)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, cards):
        self.ctx.results["cards"] = cards
        return action_long_text_review_build_prompts(
            {
                "id": "prompts",
                "input": "{cards}",
                "instructions": TEST_INSTRUCTIONS,
            },
            self.ctx,
        )

    def parse(self, cards, responses):
        self.ctx.results["cards"] = cards
        self.ctx.results["responses"] = [
            json.dumps(item, ensure_ascii=False) for item in responses
        ]
        return action_long_text_review_parse_verdicts(
            {
                "id": "report",
                "input": "{responses}",
                "cards": "{cards}",
                "fail_incomplete": False,
            },
            self.ctx,
        )

    def test_stale_policy_fingerprint_cannot_be_reused(self):
        cards = make_cards()
        self.build(cards)
        responses = [
            local_verdict(1501, cards["topics"]["1501"]),
            local_verdict(1502, cards["topics"]["1502"]),
            global_verdict(cards),
        ]
        stale = deepcopy(responses[0])
        stale["policy_sha256"] = "0" * 64
        report = self.parse(cards, [stale, responses[1], responses[2]])
        self.assertFalse(report["review_completed"])
        self.assertFalse(report["all_clear"])
        self.assertIn(1501, report["regenerate_topics"])
        messages = " | ".join(
            item["message"] for item in report["response_errors"]
        )
        self.assertIn("policy_sha256", messages)

    def test_global_payload_overflow_fails_before_any_send(self):
        cards = make_cards()
        cards["topics"]["1501"]["fields"]["youtube_description_1"] = (
            "محتوى " * 60_000
        )
        with self.assertRaises(EngineError) as captured:
            self.build(cards)
        self.assertEqual(
            "LONG_TEXT_REVIEW_CROSS_TOPIC_PAYLOAD_TOO_LARGE",
            captured.exception.code,
        )

    def test_readable_font_and_size_differences_are_warnings(self):
        self.assertEqual(
            "warning",
            _issue("DOCX_RUN_FONT_WRONG", "اختبار")["severity"],
        )
        self.assertEqual(
            "warning",
            _issue("DOCX_RUN_SIZE_WRONG", "اختبار")["severity"],
        )


if __name__ == "__main__":
    unittest.main()
