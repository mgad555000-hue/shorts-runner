import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from long_text_reviewer import (
    LONG_TEXT_REVIEW_SCHEMA_VERSION,
    SEMANTIC_FIELD_KEYS,
    _request_id,
)  # noqa: E402
from long_text_semantic_reviewer import (  # noqa: E402
    CROSS_TOPIC_REVIEW_KIND,
    INDIVIDUAL_CROSS_CHECKS,
    _cross_topic_identity,
    _evidence_is_quote,
    _strict_json_object,
    _policy_sha256,
    _topic_review_request_id,
    action_long_text_review_build_prompts,
    action_long_text_review_parse_verdicts,
)


TEST_INSTRUCTIONS = "راجع الدقة والمعنى والتكرار بدقة."


class FakeContext:
    def __init__(self, output_dir):
        self.output_dir = str(output_dir)
        self.results = {}

    def output_path(self, filename):
        return os.path.join(self.output_dir, filename)

    def resolve(self, value):
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            return self.results[value[1:-1]]
        return value


def make_topic(topic_id):
    label = "الأول" if topic_id == 1501 else "الثاني"
    return {
        "title": f"عنوان الموضوع {label} عن صحة الكلى",
        "fields": {
            key: f"محتوى {key} الخاص بالموضوع {label} واضح ومحدد"
            for key in SEMANTIC_FIELD_KEYS
        },
    }


def make_cards():
    return {
        "schema_version": LONG_TEXT_REVIEW_SCHEMA_VERSION,
        "issues": [],
        "blocked": {},
        "topics": {
            "1501": make_topic(1501),
            "1502": make_topic(1502),
        },
        "topic_filter": None,
    }


def quote(source):
    return " ".join(str(source).replace("\n", " ").split()[:3])


def local_verdict(topic_id, topic):
    policy_sha256 = _policy_sha256(TEST_INSTRUCTIONS)
    return {
        "schema_version": LONG_TEXT_REVIEW_SCHEMA_VERSION,
        "policy_sha256": policy_sha256,
        "request_id": _topic_review_request_id(
            str(topic_id),
            topic,
            policy_sha256,
        ),
        "topic_id": topic_id,
        "field_reviews": [
            {
                "field": key,
                "status": "سليم",
                "evidence": quote(topic["fields"][key]),
                "reason": "",
                "error_codes": [],
            }
            for key in SEMANTIC_FIELD_KEYS
        ],
        "cross_checks": [
            {
                "check": key,
                "status": "سليم",
                "evidence": quote(topic["title"]),
                "reason": "",
                "error_codes": [],
            }
            for key in INDIVIDUAL_CROSS_CHECKS
        ],
    }


def global_verdict(cards, status="سليم", findings=None):
    topic_ids = ["1501", "1502"]
    policy_sha256 = _policy_sha256(TEST_INSTRUCTIONS)
    request_id, fingerprint = _cross_topic_identity(
        cards["topics"],
        topic_ids,
        policy_sha256,
    )
    return {
        "schema_version": LONG_TEXT_REVIEW_SCHEMA_VERSION,
        "policy_sha256": policy_sha256,
        "review_kind": CROSS_TOPIC_REVIEW_KIND,
        "request_id": request_id,
        "batch_fingerprint": fingerprint,
        "topic_ids": [1501, 1502],
        "status": status,
        "findings": [] if findings is None else findings,
    }


def duplicate_finding(cards):
    field_name = "youtube_title_1"
    return {
        "topic_id_a": 1501,
        "topic_id_b": 1502,
        "field_a": field_name,
        "field_b": field_name,
        "relation": "DUPLICATE",
        "evidence_a": quote(cards["topics"]["1501"]["fields"][field_name]),
        "evidence_b": quote(cards["topics"]["1502"]["fields"][field_name]),
        "reason": "العنوانان يكرران نفس الصياغة الخاصة بموضوع واحد",
        "error_codes": ["DUPLICATION"],
    }


class GlobalSemanticReviewTests(unittest.TestCase):
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

    def valid_responses(self, cards):
        return [
            local_verdict(1501, cards["topics"]["1501"]),
            local_verdict(1502, cards["topics"]["1502"]),
            global_verdict(cards),
        ]

    def test_evidence_uses_safe_arabic_normalization_and_boundary_conjunction(self):
        source = "مراقبة مستويات الفلترة مهمة لصحة الكلى"
        self.assertTrue(
            _evidence_is_quote("ومراقبه مستويات الفلترة", [source])
        )
        self.assertTrue(
            _evidence_is_quote("مُراقبة، مستويات الفلترة", [source])
        )
        self.assertFalse(
            _evidence_is_quote("عبارة مختلفة ليست في المصدر", [source])
        )

    def test_builder_adds_one_global_prompt_with_all_topic_data_and_fingerprints(self):
        cards = make_cards()
        prompts = self.build(cards)
        self.assertEqual(3, len(prompts))
        self.assertIn("NO_INTERNAL_PLATFORM_SWAP", prompts[0])
        self.assertIn("<BEGIN_UNTRUSTED_CROSS_TOPIC_DATA>", prompts[-1])
        self.assertIn(cards["topics"]["1501"]["title"], prompts[-1])
        self.assertIn(cards["topics"]["1502"]["title"], prompts[-1])
        identity = self.ctx.results["prompts_cross_topic_identity"]
        self.assertEqual([1501, 1502], identity["topic_ids"])
        self.assertEqual(24, len(identity["request_id"]))
        self.assertEqual(24, len(identity["batch_fingerprint"]))
        audit = json.loads(
            (self.output_dir / "long_text_review_requests.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(CROSS_TOPIC_REVIEW_KIND, audit[-1]["review_kind"])

    def test_clean_global_verdict_completes_the_review(self):
        cards = make_cards()
        self.build(cards)
        report = self.parse(cards, self.valid_responses(cards))
        self.assertTrue(report["review_completed"])
        self.assertTrue(report["all_clear"])
        self.assertTrue(report["cross_topic_completed"])
        self.assertEqual([], report["regenerate_topics"])

    def test_one_bare_json_fence_is_accepted_but_prose_and_nested_fences_fail(self):
        payload = {"schema_version": 2, "value": "سليم"}
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(
            payload,
            _strict_json_object(f"```json\n{encoded}\n```"),
        )
        self.assertEqual(
            payload,
            _strict_json_object(f"```\n{encoded}\n```"),
        )
        self.assertIsNone(
            _strict_json_object(f"شرح قبل الرد\n```json\n{encoded}\n```")
        )
        self.assertIsNone(
            _strict_json_object(
                f"```json\n{encoded}\n```\n```json\n{encoded}\n```"
            )
        )
        self.assertIsNone(
            _strict_json_object(f"```json\n{encoded}")
        )

    def test_valid_duplicate_or_swap_regenerates_both_topics(self):
        cards = make_cards()
        self.build(cards)
        responses = self.valid_responses(cards)[:-1]
        responses.append(
            global_verdict(
                cards,
                status="خطأ",
                findings=[duplicate_finding(cards)],
            )
        )
        report = self.parse(cards, responses)
        self.assertTrue(report["review_completed"])
        self.assertFalse(report["all_clear"])
        self.assertEqual([1501, 1502], report["regenerate_topics"])
        self.assertEqual(2, report["semantic_error_count"])
        self.assertEqual(
            {1501, 1502},
            {item["topic_id"] for item in report["semantic_findings"]},
        )

    def test_missing_duplicate_alien_and_inconclusive_global_fail_closed(self):
        cards = make_cards()
        self.build(cards)
        locals_only = self.valid_responses(cards)[:-1]
        clean_global = global_verdict(cards)

        alien_global = deepcopy(clean_global)
        alien_global["topic_ids"] = [1501, 9999]
        inconclusive = deepcopy(clean_global)
        inconclusive["status"] = "غير حاسم"
        cases = {
            "missing": locals_only,
            "duplicate": locals_only + [clean_global, deepcopy(clean_global)],
            "alien": locals_only + [alien_global],
            "inconclusive": locals_only + [inconclusive],
        }
        for name, responses in cases.items():
            with self.subTest(name=name):
                report = self.parse(cards, responses)
                self.assertFalse(report["review_completed"])
                self.assertFalse(report["all_clear"])
                self.assertEqual([1501, 1502], report["regenerate_topics"])
                self.assertGreater(report["response_error_count"], 0)

    def test_single_topic_keeps_backward_compatible_prompt_and_parser(self):
        cards = make_cards()
        cards["topics"].pop("1502")
        prompts = self.build(cards)
        self.assertEqual(1, len(prompts))
        report = self.parse(
            cards,
            [local_verdict(1501, cards["topics"]["1501"])],
        )
        self.assertTrue(report["review_completed"])
        self.assertTrue(report["all_clear"])
        self.assertFalse(report["cross_topic_required"])

    def test_global_evidence_must_be_independent_and_from_named_fields(self):
        cards = make_cards()
        self.build(cards)
        finding = duplicate_finding(cards)
        finding["evidence_b"] = (
            "محتوى youtube_title_1 الخاص بالموضوع الأول"
        )
        responses = self.valid_responses(cards)[:-1]
        responses.append(global_verdict(cards, status="خطأ", findings=[finding]))
        report = self.parse(cards, responses)
        self.assertFalse(report["review_completed"])
        self.assertEqual([1501, 1502], report["regenerate_topics"])
        messages = " | ".join(item["message"] for item in report["response_errors"])
        self.assertIn("evidence_b", messages)


if __name__ == "__main__":
    unittest.main()
