"""Versioned, fail-closed contract for long-text generator outputs.

The JSON file beside this module is the single serializable contract.  This
module only validates it, derives convenient constants, and verifies that a
review input carries the exact same contract plus its SHA-256 fingerprint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


CONTRACT_PATH = Path(__file__).with_name("long_text_contract.v1.json")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_id",
    "contract_version",
    "token_specs",
    "aliases",
    "contexts",
    "consumer_exact",
    "semantic",
    "counts",
    "required_hashtags",
    "banned_phrases",
    "formatting",
}
_CONSUMER_KEYS = {
    "source_file",
    "thumbnail_file",
    "topics_file",
    "review_contract_file",
    "topic_root_preferred",
    "topic_root_alternative",
    "topic_id_key",
    "topic_title_key",
    "script_heading_template",
    "script_heading_style",
    "body_style",
    "thumbnail_extract_start_key",
    "thumbnail_extract_end_key",
    "thumbnail_extract_start_label",
    "thumbnail_extract_end_label",
    "consumer_exact_label_keys",
    "token_order_is_exact",
    "topic_order_is_exact",
    "thumbnail_matches_youtube_thumbnail_1_exactly",
}
_SEMANTIC_KEYS = {
    "field_keys",
    "cross_checks",
    "error_codes",
    "statuses",
    "evidence_min_words",
    "evidence_max_words",
    "response_schema_version",
}
_COUNT_KEYS = {
    "description_hashtags",
    "youtube_shared_required_hashtags",
    "youtube_unique_hashtags_each",
    "keywords_each",
    "keywords_delimiter",
    "youtube_thumbnail_min_words",
    "youtube_thumbnail_max_words",
    "facebook_thumbnail_min_words",
    "facebook_thumbnail_max_words",
    "tiktok_screen_min_words",
    "tiktok_screen_max_words",
    "short_phrase_min_lines",
    "short_phrase_max_lines",
    "short_phrase_max_words_per_line",
    "cross_topic_exact_duplicate_min_normalized_chars",
}
_FORMAT_KEYS = {
    "font_name",
    "font_size_pt",
    "line_spacing_pt",
    "paragraph_alignment",
    "paragraph_bidi",
    "run_rtl",
    "run_bidi_language_prefix",
    "run_font_slots",
}


class LongTextContractError(ValueError):
    """Raised when the versioned contract or a supplied copy is not exact."""


def _strict_json_load(path):
    def reject_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise LongTextContractError(f"مفتاح JSON مكرر: {key}")
            value[key] = item
        return value

    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicate_keys)


def _exact_keys(value, expected, location):
    if not isinstance(value, Mapping):
        raise LongTextContractError(f"{location} لازم يكون JSON object")
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise LongTextContractError(
            f"{location} مفاتيحه غير مطابقة؛ ناقص={missing} زائد={extra}"
        )


def _nonempty_string(value, location):
    if not isinstance(value, str) or not value.strip():
        raise LongTextContractError(f"{location} لازم يكون نصا غير فارغ")


def _positive_int(value, location):
    if type(value) is not int or value < 1:
        raise LongTextContractError(f"{location} لازم يكون integer موجب")


def _unique_strings(value, location, *, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise LongTextContractError(f"{location} لازم يكون قائمة نصوص غير فارغة")
    for index, item in enumerate(value):
        _nonempty_string(item, f"{location}[{index}]")
    if len(value) != len(set(value)):
        raise LongTextContractError(f"{location} يحتوي قيما مكررة")


def _validate_contract_shape(contract):
    _exact_keys(contract, _TOP_LEVEL_KEYS, "contract")
    if contract["schema_version"] != 1:
        raise LongTextContractError("schema_version غير مدعوم")
    if contract["contract_version"] != 1:
        raise LongTextContractError("contract_version غير مدعوم")
    if contract["contract_id"] != "mg_runner.long_text_output":
        raise LongTextContractError("contract_id غير معروف")

    token_specs = contract["token_specs"]
    if not isinstance(token_specs, list) or not token_specs:
        raise LongTextContractError("token_specs لازم يكون قائمة غير فارغة")
    token_keys = []
    field_keys = []
    header_keys = []
    labels = {}
    for index, item in enumerate(token_specs):
        location = f"token_specs[{index}]"
        _exact_keys(item, {"key", "kind", "label"}, location)
        _nonempty_string(item["key"], f"{location}.key")
        _nonempty_string(item["label"], f"{location}.label")
        if item["kind"] not in {"header", "field"}:
            raise LongTextContractError(f"{location}.kind غير صالح")
        token_keys.append(item["key"])
        labels[item["key"]] = item["label"]
        if item["kind"] == "field":
            field_keys.append(item["key"])
        else:
            header_keys.append(item["key"])
    if len(token_keys) != len(set(token_keys)):
        raise LongTextContractError("token_specs يحتوي مفاتيح مكررة")

    aliases = contract["aliases"]
    _exact_keys(aliases, {"headers", "fields"}, "aliases")
    _exact_keys(aliases["headers"], set(header_keys), "aliases.headers")
    _exact_keys(aliases["fields"], set(field_keys), "aliases.fields")
    for group_name in ("headers", "fields"):
        for token_key, values in aliases[group_name].items():
            _unique_strings(values, f"aliases.{group_name}.{token_key}")

    contexts = contract["contexts"]
    if not isinstance(contexts, Mapping) or not contexts:
        raise LongTextContractError("contexts لازم يكون JSON object غير فارغ")
    assigned_fields = []
    for context_key, values in contexts.items():
        if context_key not in header_keys:
            raise LongTextContractError(f"context غير معروف: {context_key}")
        _unique_strings(values, f"contexts.{context_key}")
        unknown = sorted(set(values) - set(field_keys))
        if unknown:
            raise LongTextContractError(
                f"contexts.{context_key} يحتوي بنودا غير معروفة: {unknown}"
            )
        assigned_fields.extend(values)
    if assigned_fields != field_keys:
        raise LongTextContractError(
            "contexts لا توزع field token keys مرة واحدة وبنفس ترتيب العقد"
        )

    consumer = contract["consumer_exact"]
    _exact_keys(consumer, _CONSUMER_KEYS, "consumer_exact")
    for key in _CONSUMER_KEYS - {
        "consumer_exact_label_keys",
        "token_order_is_exact",
        "topic_order_is_exact",
        "thumbnail_matches_youtube_thumbnail_1_exactly",
    }:
        _nonempty_string(consumer[key], f"consumer_exact.{key}")
    for key in (
        "token_order_is_exact",
        "topic_order_is_exact",
        "thumbnail_matches_youtube_thumbnail_1_exactly",
    ):
        if consumer[key] is not True:
            raise LongTextContractError(f"consumer_exact.{key} لازم يكون true")
    _unique_strings(
        consumer["consumer_exact_label_keys"],
        "consumer_exact.consumer_exact_label_keys",
    )
    if set(consumer["consumer_exact_label_keys"]) - set(field_keys):
        raise LongTextContractError("consumer_exact_label_keys يحتوي توكن غير معروف")
    if consumer["consumer_exact_label_keys"] != [
        consumer["thumbnail_extract_start_key"],
        consumer["thumbnail_extract_end_key"],
    ]:
        raise LongTextContractError(
            "consumer_exact_label_keys لازم يطابق حدود مستهلك الصور المصغرة"
        )
    if consumer["thumbnail_extract_start_key"] not in field_keys:
        raise LongTextContractError("thumbnail_extract_start_key غير معروف")
    if consumer["thumbnail_extract_end_key"] not in field_keys:
        raise LongTextContractError("thumbnail_extract_end_key غير معروف")
    if (
        consumer["thumbnail_extract_start_label"]
        != labels[consumer["thumbnail_extract_start_key"]]
    ):
        raise LongTextContractError("thumbnail_extract_start_label لا يطابق التوكن")
    if (
        consumer["thumbnail_extract_end_label"]
        != labels[consumer["thumbnail_extract_end_key"]]
    ):
        raise LongTextContractError("thumbnail_extract_end_label لا يطابق التوكن")

    semantic = contract["semantic"]
    _exact_keys(semantic, _SEMANTIC_KEYS, "semantic")
    _unique_strings(semantic["field_keys"], "semantic.field_keys")
    if semantic["field_keys"] != field_keys:
        raise LongTextContractError(
            "semantic.field_keys لا يطابق field token order"
        )
    _unique_strings(semantic["cross_checks"], "semantic.cross_checks")
    _unique_strings(semantic["error_codes"], "semantic.error_codes")
    _unique_strings(semantic["statuses"], "semantic.statuses")
    for key in (
        "evidence_min_words",
        "evidence_max_words",
        "response_schema_version",
    ):
        _positive_int(semantic[key], f"semantic.{key}")
    if semantic["evidence_min_words"] > semantic["evidence_max_words"]:
        raise LongTextContractError("مدى كلمات الدليل معكوس")

    counts = contract["counts"]
    _exact_keys(counts, _COUNT_KEYS, "counts")
    for key, value in counts.items():
        if key == "keywords_delimiter":
            _nonempty_string(value, "counts.keywords_delimiter")
        else:
            _positive_int(value, f"counts.{key}")
    for minimum, maximum in (
        ("youtube_thumbnail_min_words", "youtube_thumbnail_max_words"),
        ("facebook_thumbnail_min_words", "facebook_thumbnail_max_words"),
        ("tiktok_screen_min_words", "tiktok_screen_max_words"),
        ("short_phrase_min_lines", "short_phrase_max_lines"),
    ):
        if counts[minimum] > counts[maximum]:
            raise LongTextContractError(f"مدى {minimum}/{maximum} معكوس")

    _unique_strings(contract["required_hashtags"], "required_hashtags")
    if any(not value.startswith("#") for value in contract["required_hashtags"]):
        raise LongTextContractError("required_hashtags يحتوي قيمة بلا #")
    _unique_strings(contract["banned_phrases"], "banned_phrases")

    formatting = contract["formatting"]
    _exact_keys(formatting, {"source", "thumbnail"}, "formatting")
    for role in ("source", "thumbnail"):
        value = formatting[role]
        _exact_keys(value, _FORMAT_KEYS, f"formatting.{role}")
        _nonempty_string(value["font_name"], f"formatting.{role}.font_name")
        _positive_int(value["font_size_pt"], f"formatting.{role}.font_size_pt")
        _positive_int(
            value["line_spacing_pt"],
            f"formatting.{role}.line_spacing_pt",
        )
        if value["paragraph_alignment"] != "RIGHT":
            raise LongTextContractError(
                f"formatting.{role}.paragraph_alignment لازم يكون RIGHT"
            )
        _exact_keys(
            value["paragraph_bidi"],
            {
                "presence_required",
                "allow_word_native_zero_with_right_alignment_and_run_rtl",
            },
            f"formatting.{role}.paragraph_bidi",
        )
        if (
            value["paragraph_bidi"]["presence_required"] is not True
            or value["paragraph_bidi"][
                "allow_word_native_zero_with_right_alignment_and_run_rtl"
            ]
            is not True
            or value["run_rtl"] is not True
        ):
            raise LongTextContractError(f"formatting.{role} لازم يفرض RTL الآمن")
        _nonempty_string(
            value["run_bidi_language_prefix"],
            f"formatting.{role}.run_bidi_language_prefix",
        )
        _unique_strings(
            value["run_font_slots"],
            f"formatting.{role}.run_font_slots",
        )
    return contract


def canonical_contract_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def contract_fingerprint(value):
    canonical = canonical_contract_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


try:
    CONTRACT = _validate_contract_shape(_strict_json_load(CONTRACT_PATH))
except (
    OSError,
    UnicodeError,
    json.JSONDecodeError,
    LongTextContractError,
) as exc:
    raise RuntimeError(f"تعذر تحميل عقد تكست لونج: {exc}") from exc

CONTRACT_HASH = contract_fingerprint(CONTRACT).split(":", 1)[1]
CONTRACT_FINGERPRINT = f"sha256:{CONTRACT_HASH}"
CONTRACT_VERSION = CONTRACT["contract_version"]

TOKEN_SPECS = tuple(
    (item["key"], item["kind"], item["label"])
    for item in CONTRACT["token_specs"]
)
EXPECTED_TOKEN_KEYS = tuple(item[0] for item in TOKEN_SPECS)
CANONICAL_LABELS = {key: label for key, _kind, label in TOKEN_SPECS}
TOKEN_KINDS = {key: kind for key, kind, _label in TOKEN_SPECS}
FIELD_KEYS = tuple(
    key for key, kind, _label in TOKEN_SPECS if kind == "field"
)
HEADER_ALIASES = {
    key: tuple(values)
    for key, values in CONTRACT["aliases"]["headers"].items()
}
FIELD_ALIASES = {
    key: tuple(values)
    for key, values in CONTRACT["aliases"]["fields"].items()
}
CONTEXT_FIELDS = {
    key: tuple(values) for key, values in CONTRACT["contexts"].items()
}
CONSUMER_EXACT = copy.deepcopy(CONTRACT["consumer_exact"])
SEMANTIC_FIELD_KEYS = tuple(CONTRACT["semantic"]["field_keys"])
SEMANTIC_CROSS_CHECKS = tuple(CONTRACT["semantic"]["cross_checks"])
SEMANTIC_ERROR_CODES = frozenset(CONTRACT["semantic"]["error_codes"])
COUNTS = copy.deepcopy(CONTRACT["counts"])
REQUIRED_ARABIC_HASHTAGS = frozenset(CONTRACT["required_hashtags"])
BANNED_PHRASES = tuple(CONTRACT["banned_phrases"])
FORMATTING = copy.deepcopy(CONTRACT["formatting"])


def review_contract_document():
    """Return the exact envelope expected in reviewer input."""
    return {
        "fingerprint": CONTRACT_FINGERPRINT,
        "contract": copy.deepcopy(CONTRACT),
    }


def validate_contract_payload(payload, fingerprint=None):
    """Validate a bare exact contract or the compatible envelope form.

    The preferred review input is the bare JSON contract.  A two-key
    ``fingerprint``/``contract`` envelope remains accepted for compatibility.
    """
    if fingerprint is not None:
        contract = payload
    elif (
        isinstance(payload, Mapping)
        and set(payload) == {"fingerprint", "contract"}
    ):
        fingerprint = payload["fingerprint"]
        contract = payload["contract"]
    else:
        contract = payload
        fingerprint = contract_fingerprint(contract) if isinstance(
            contract, Mapping
        ) else ""
    _nonempty_string(fingerprint, "review_contract.fingerprint")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint):
        raise LongTextContractError("بصمة العقد ليست SHA-256 صالحة")
    if not isinstance(contract, Mapping):
        raise LongTextContractError("نسخة العقد لازم تكون JSON object")
    calculated = contract_fingerprint(contract)
    if calculated != fingerprint:
        raise LongTextContractError("محتوى العقد لا يطابق بصمته")
    if fingerprint != CONTRACT_FINGERPRINT:
        raise LongTextContractError("بصمة العقد لا تطابق الإصدار المثبت")
    if contract != CONTRACT:
        raise LongTextContractError(
            "نسخة review_contract.json لا تطابق العقد المثبت حرفيا"
        )
    return contract
def load_review_contract(path):
    """Read a strict JSON envelope and verify its contract and fingerprint."""
    payload = _strict_json_load(path)
    validate_contract_payload(payload)
    return payload


__all__ = [
    "BANNED_PHRASES",
    "CANONICAL_LABELS",
    "CONSUMER_EXACT",
    "CONTEXT_FIELDS",
    "CONTRACT",
    "CONTRACT_FINGERPRINT",
    "CONTRACT_HASH",
    "CONTRACT_PATH",
    "CONTRACT_VERSION",
    "COUNTS",
    "EXPECTED_TOKEN_KEYS",
    "FIELD_ALIASES",
    "FIELD_KEYS",
    "FORMATTING",
    "HEADER_ALIASES",
    "LongTextContractError",
    "REQUIRED_ARABIC_HASHTAGS",
    "SEMANTIC_CROSS_CHECKS",
    "SEMANTIC_ERROR_CODES",
    "SEMANTIC_FIELD_KEYS",
    "TOKEN_KINDS",
    "TOKEN_SPECS",
    "canonical_contract_json",
    "contract_fingerprint",
    "load_review_contract",
    "review_contract_document",
    "validate_contract_payload",
]
