import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from engine import EngineError  # noqa: E402
import long_text_reviewer as base  # noqa: E402
from long_text_contract import (  # noqa: E402
    BANNED_PHRASES,
    CANONICAL_LABELS,
    CONSUMER_EXACT,
    CONTRACT,
    CONTRACT_FINGERPRINT,
    CONTRACT_HASH,
    CONTEXT_FIELDS,
    EXPECTED_TOKEN_KEYS,
    FIELD_ALIASES,
    FIELD_KEYS,
    FORMATTING,
    HEADER_ALIASES,
    LongTextContractError,
    REQUIRED_ARABIC_HASHTAGS,
    SEMANTIC_CROSS_CHECKS,
    SEMANTIC_ERROR_CODES,
    SEMANTIC_FIELD_KEYS,
    TOKEN_KINDS,
    TOKEN_SPECS,
    contract_fingerprint,
    review_contract_document,
    validate_contract_payload,
)
from long_text_review_runtime_final import (  # noqa: E402
    action_long_text_review_build_cards_final,
)
from tests.test_long_text_reviewer import (  # noqa: E402
    FakeContext,
    TITLE,
    token_lines,
    valid_fields,
    write_scripts_docx,
    write_thumbnail_docx,
)


class LongTextContractTests(unittest.TestCase):
    def test_fingerprint_is_stable_and_self_consistent(self):
        self.assertEqual(64, len(CONTRACT_HASH))
        self.assertEqual(f"sha256:{CONTRACT_HASH}", CONTRACT_FINGERPRINT)
        self.assertEqual(CONTRACT_FINGERPRINT, contract_fingerprint(CONTRACT))
        self.assertEqual(1, CONTRACT["contract_version"])
        self.assertEqual(2, CONTRACT["semantic"]["response_schema_version"])

    def test_derived_constants_match_current_reviewer(self):
        self.assertEqual(base.TOKEN_SPECS, TOKEN_SPECS)
        self.assertEqual(base.EXPECTED_TOKEN_KEYS, EXPECTED_TOKEN_KEYS)
        self.assertEqual(base.CANONICAL_LABELS, CANONICAL_LABELS)
        self.assertEqual(base.TOKEN_KINDS, TOKEN_KINDS)
        self.assertEqual(base.FIELD_KEYS, FIELD_KEYS)
        self.assertEqual(base.HEADER_ALIASES, HEADER_ALIASES)
        self.assertEqual(base.FIELD_ALIASES, FIELD_ALIASES)
        self.assertEqual(base.CONTEXT_FIELDS, CONTEXT_FIELDS)
        self.assertEqual(base.SEMANTIC_FIELD_KEYS, SEMANTIC_FIELD_KEYS)
        self.assertEqual(base.SEMANTIC_CROSS_CHECKS, SEMANTIC_CROSS_CHECKS)
        self.assertEqual(base.SEMANTIC_ERROR_CODES, set(SEMANTIC_ERROR_CODES))
        self.assertEqual(
            base.REQUIRED_ARABIC_HASHTAGS,
            set(REQUIRED_ARABIC_HASHTAGS),
        )
        self.assertEqual(base.BANNED_PHRASES, BANNED_PHRASES)

    def test_contract_records_only_labels_used_exactly_by_consumer(self):
        self.assertEqual(
            ["youtube_thumbnail_1", "youtube_thumbnail_2"],
            CONSUMER_EXACT["consumer_exact_label_keys"],
        )
        self.assertTrue(
            FORMATTING["source"]["paragraph_bidi"]["presence_required"]
        )
        self.assertTrue(
            FORMATTING["source"]["paragraph_bidi"][
                "allow_word_native_zero_with_right_alignment_and_run_rtl"
            ]
        )
        self.assertEqual(
            "NO_INTERNAL_PLATFORM_SWAP",
            SEMANTIC_CROSS_CHECKS[-1],
        )

    def test_bare_contract_is_preferred_and_envelope_remains_compatible(self):
        self.assertIs(CONTRACT, validate_contract_payload(CONTRACT))
        envelope = review_contract_document()
        self.assertEqual(CONTRACT, validate_contract_payload(envelope))

    def test_tampered_contract_and_fingerprint_are_rejected(self):
        changed = copy.deepcopy(CONTRACT)
        changed["counts"]["keywords_each"] = 17
        with self.assertRaises(LongTextContractError):
            validate_contract_payload(changed)
        envelope = review_contract_document()
        envelope["fingerprint"] = "sha256:" + ("0" * 64)
        with self.assertRaises(LongTextContractError):
            validate_contract_payload(envelope)


class LongTextFinalRuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        self.write_topics([{"id": 1501, "title": TITLE}])
        self.write_contract()
        write_scripts_docx(self.input_dir / "scripts_output.docx")
        write_thumbnail_docx(self.input_dir / "thumbnail_texts.docx")

    def tearDown(self):
        self.temp.cleanup()

    def write_contract(self, payload=None):
        (self.input_dir / "review_contract.json").write_text(
            json.dumps(
                CONTRACT if payload is None else payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def write_topics(self, items):
        (self.input_dir / "topics.json").write_text(
            json.dumps(
                {"total_count": len(items), "titles": items},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def step(self, strict=False, **overrides):
        value = {"id": "cards", "strict": strict}
        value.update(overrides)
        return value

    def run_review(self, topic_ids=None, strict=False, **overrides):
        ctx = FakeContext(
            self.input_dir,
            self.output_dir,
            [1501] if topic_ids is None else topic_ids,
        )
        cards = action_long_text_review_build_cards_final(
            self.step(strict=strict, **overrides),
            ctx,
        )
        return ctx, cards

    def report(self):
        return json.loads(
            (
                self.output_dir / "long_text_structure_report.json"
            ).read_text(encoding="utf-8")
        )

    def test_valid_bare_contract_passes_and_fingerprint_is_reported(self):
        _ctx, cards = self.run_review(strict=True)
        report = self.report()
        self.assertTrue(report["structure_passed"])
        self.assertTrue(report["ingestion_usable"])
        self.assertTrue(report["quality_clean"])
        self.assertEqual(CONTRACT_FINGERPRINT, report["contract_fingerprint"])
        self.assertEqual(CONTRACT_FINGERPRINT, cards["contract_fingerprint"])

    def test_missing_contract_fails_closed_before_review(self):
        (self.input_dir / "review_contract.json").unlink()
        ctx = FakeContext(self.input_dir, self.output_dir, [1501])
        with self.assertRaises(EngineError) as captured:
            action_long_text_review_build_cards_final(
                self.step(strict=False),
                ctx,
            )
        self.assertEqual("REVIEW_CONTRACT_MISSING", captured.exception.code)
        report = self.report()
        self.assertFalse(report["ingestion_usable"])
        self.assertFalse(report["quality_clean"])
        self.assertEqual(
            "REVIEW_CONTRACT_MISSING",
            report["issues"][0]["code"],
        )

    def test_tampered_contract_fails_even_with_non_strict_review(self):
        changed = copy.deepcopy(CONTRACT)
        changed["required_hashtags"][0] = "#مختلف"
        self.write_contract(changed)
        ctx = FakeContext(self.input_dir, self.output_dir, [1501])
        with self.assertRaises(EngineError) as captured:
            action_long_text_review_build_cards_final(
                self.step(strict=False),
                ctx,
            )
        self.assertEqual("REVIEW_CONTRACT_MISMATCH", captured.exception.code)

    def test_contract_can_only_be_skipped_explicitly(self):
        (self.input_dir / "review_contract.json").unlink()
        _ctx, cards = self.run_review(strict=True, require_contract=False)
        self.assertTrue(cards["structure_report"]["structure_passed"])

    def test_duplicate_and_invalid_selected_ids_are_rejected(self):
        for topic_ids, code in (
            ([1501, 1501], "TOPIC_ID_SELECTION_DUPLICATE"),
            ([True], "TOPIC_ID_SELECTION_INVALID"),
            (["1501"], "TOPIC_ID_SELECTION_INVALID"),
            ([], "TOPIC_IDS_REQUIRED"),
        ):
            with self.subTest(topic_ids=topic_ids):
                ctx = FakeContext(self.input_dir, self.output_dir, topic_ids)
                with self.assertRaises(EngineError) as captured:
                    action_long_text_review_build_cards_final(
                        self.step(strict=False),
                        ctx,
                    )
                self.assertEqual(code, captured.exception.code)

    def test_missing_selected_id_is_rejected_before_base_reviewer(self):
        ctx = FakeContext(self.input_dir, self.output_dir, [1501, 1502])
        with self.assertRaises(EngineError) as captured:
            action_long_text_review_build_cards_final(
                self.step(strict=False),
                ctx,
            )
        self.assertEqual("SELECTED_TOPICS_MISSING", captured.exception.code)

    def test_ambiguous_topics_root_is_rejected(self):
        (self.input_dir / "topics.json").write_text(
            json.dumps(
                {
                    "titles": [{"id": 1501, "title": TITLE}],
                    "topics": [{"id": 1501, "title": TITLE}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        ctx = FakeContext(self.input_dir, self.output_dir, [1501])
        with self.assertRaises(EngineError) as captured:
            action_long_text_review_build_cards_final(
                self.step(strict=False),
                ctx,
            )
        self.assertEqual("TOPICS_AMBIGUOUS_ROOT", captured.exception.code)

    def test_duplicate_json_key_is_rejected(self):
        (self.input_dir / "topics.json").write_text(
            '{"titles":[{"id":1501,"title":"أ"}],'
            '"titles":[{"id":1501,"title":"ب"}]}',
            encoding="utf-8",
        )
        ctx = FakeContext(self.input_dir, self.output_dir, [1501])
        with self.assertRaises(EngineError) as captured:
            action_long_text_review_build_cards_final(
                self.step(strict=False),
                ctx,
            )
        self.assertEqual("TOPICS_INVALID_JSON", captured.exception.code)

    def test_snapshot_keeps_original_archive_order(self):
        self.write_topics(
            [
                {"id": 1502, "title": "عنوان مختلف للموضوع الثاني"},
                {"id": 1501, "title": TITLE},
                {"id": 1503, "title": "موضوع غير مختار"},
            ]
        )
        self.run_review(topic_ids=[1501, 1502], strict=False)
        snapshot = json.loads(
            (
                self.output_dir
                / "_long_text_review_snapshot"
                / "topics.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            [1502, 1501],
            [item["id"] for item in snapshot["titles"]],
        )
        self.assertEqual(2, snapshot["total_count"])

    def test_facebook_thumbnail_is_checked_before_final_report(self):
        fields = valid_fields()
        fields["facebook_thumbnail"] = "قصيرة جدا"
        write_scripts_docx(
            self.input_dir / "scripts_output.docx",
            token_lines(fields),
        )
        _ctx, cards = self.run_review(strict=False)
        codes = {item["code"] for item in cards["issues"]}
        self.assertIn("SHORT_PHRASE_WORD_COUNT", codes)
        self.assertIn("SHORT_PHRASE_LINE_COUNT", codes)
        report = self.report()
        self.assertFalse(report["ingestion_usable"])
        self.assertFalse(report["quality_clean"])

    def test_strict_is_applied_after_facebook_check_and_final_report(self):
        fields = valid_fields()
        fields["facebook_thumbnail"] = "قصيرة جدا"
        write_scripts_docx(
            self.input_dir / "scripts_output.docx",
            token_lines(fields),
        )
        ctx = FakeContext(self.input_dir, self.output_dir, [1501])
        with self.assertRaises(EngineError) as captured:
            action_long_text_review_build_cards_final(
                self.step(strict=True),
                ctx,
            )
        self.assertEqual("LONG_TEXT_STRUCTURE_FAILED", captured.exception.code)
        self.assertFalse(self.report()["structure_passed"])


if __name__ == "__main__":
    unittest.main()
