"""Single production runtime for deterministic long-text output review.

This module intentionally does not import either legacy runtime wrapper.  It
owns the preconditions, immutable topic snapshot, contract gate, one final
report rebuild, and strict-mode decision.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from engine import EngineError, log
from long_text_contract import (
    CONSUMER_EXACT,
    CONTRACT_FINGERPRINT,
    CONTRACT_VERSION,
    COUNTS,
    FORMATTING,
    LongTextContractError,
    validate_contract_payload,
)
from long_text_reviewer import (
    LONG_TEXT_REVIEW_SCHEMA_VERSION,
    _add_unique,
    _issue,
    _strict_json_load,
    _validate_thumbnail_value,
    _write_structure_reports,
    action_long_text_review_build_cards,
)


class _RuntimeInputError(ValueError):
    def __init__(self, code, message, *, file=None, expected=None, actual=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.file = file
        self.expected = expected
        self.actual = actual


class _TopicsSnapshotContext:
    def __init__(self, original, topics_file, snapshot_path, topic_ids):
        self._original = original
        self._topics_file = topics_file
        self._snapshot_path = snapshot_path
        self.input_dir = original.input_dir
        self.output_dir = original.output_dir
        self.topic_ids = list(topic_ids)
        self.results = original.results

    def input_path(self, filename):
        if filename == self._topics_file:
            return self._snapshot_path
        return self._original.input_path(filename)

    def output_path(self, filename):
        return self._original.output_path(filename)

    def resolve(self, value):
        return self._original.resolve(value)


def _bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _report_names(step):
    return (
        step.get("save_json", "long_text_structure_report.json"),
        step.get("save_text", "long_text_structure_report.txt"),
        step.get(
            "save_regenerate",
            "long_text_topics_to_regenerate.json",
        ),
    )


def _write_precondition_report(step, ctx, issue, expected_ids=()):
    json_name, text_name, regenerate_name = _report_names(step)
    report = {
        "schema_version": LONG_TEXT_REVIEW_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": CONTRACT_FINGERPRINT,
        "structure_passed": False,
        "topics_expected": len(expected_ids),
        "topics_read": 0,
        "expected_ids": list(expected_ids),
        "source_ids": [],
        "thumbnail_ids": [],
        "error_count": 1,
        "warning_count": 0,
        "ingestion_usable": False,
        "quality_clean": False,
        "issues": [issue],
    }
    _write_structure_reports(
        ctx,
        report,
        json_name=json_name,
        text_name=text_name,
        regenerate_name=regenerate_name,
    )


def _fail(step, ctx, error, expected_ids=()):
    issue = _issue(
        error.code,
        error.message,
        file=error.file,
        expected=error.expected,
        actual=error.actual,
    )
    _write_precondition_report(step, ctx, issue, expected_ids)
    raise EngineError(error.message, code=error.code)


def _validate_selected_ids(ctx):
    raw_ids = getattr(ctx, "topic_ids", None)
    if not raw_ids:
        raise _RuntimeInputError(
            "TOPIC_IDS_REQUIRED",
            "لازم تختار نفس معرفات الموضوعات المستخدمة في تشغيل إنشاء تكست لونج",
            expected="قائمة JSON integer موجبة وغير فارغة",
            actual=raw_ids,
        )
    if isinstance(raw_ids, (set, frozenset)):
        # PipelineContext parses the environment into a set. The authoritative
        # order is recovered later from the immutable topics snapshot.
        iterable_ids = sorted(raw_ids)
    elif isinstance(raw_ids, (list, tuple)):
        iterable_ids = raw_ids
    else:
        raise _RuntimeInputError(
            "TOPIC_IDS_INVALID",
            "اختيار TOPIC_IDS لازم يكون قائمة أو مجموعة معرفات",
            expected="list أو tuple أو set",
            actual=type(raw_ids).__name__,
        )
    selected = []
    seen = set()
    for index, value in enumerate(iterable_ids):
        if type(value) is not int or value < 1:
            raise _RuntimeInputError(
                "TOPIC_ID_SELECTION_INVALID",
                f"معرف الموضوع المختار رقم {index + 1} مش integer موجب",
                expected="integer موجب",
                actual=value,
            )
        if value in seen:
            raise _RuntimeInputError(
                "TOPIC_ID_SELECTION_DUPLICATE",
                f"معرف الموضوع {value} مكرر في الاختيار",
                expected="معرفات فريدة",
                actual=value,
            )
        seen.add(value)
        selected.append(value)
    return selected


def _validate_review_contract(step, ctx):
    if not _bool(step.get("require_contract", True), True):
        return
    contract_file = step.get(
        "contract_file",
        CONSUMER_EXACT["review_contract_file"],
    )
    expected_filename = CONSUMER_EXACT["review_contract_file"]
    if contract_file != expected_filename:
        raise _RuntimeInputError(
            "REVIEW_CONTRACT_FILENAME_MISMATCH",
            "اسم ملف العقد لا يطابق اسم المستهلك المثبت",
            file=contract_file,
            expected=expected_filename,
            actual=contract_file,
        )
    contract_path = ctx.input_path(contract_file)
    if not os.path.isfile(contract_path):
        raise _RuntimeInputError(
            "REVIEW_CONTRACT_MISSING",
            "ملف review_contract.json مطلوب قبل بدء المراجعة",
            file=contract_file,
            expected=CONTRACT_FINGERPRINT,
        )
    try:
        payload = _strict_json_load(contract_path)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise _RuntimeInputError(
            "REVIEW_CONTRACT_INVALID",
            f"تعذر قراءة review_contract.json كـ JSON صارم: {exc}",
            file=contract_file,
        ) from exc
    try:
        validate_contract_payload(payload)
    except LongTextContractError as exc:
        raise _RuntimeInputError(
            "REVIEW_CONTRACT_MISMATCH",
            f"العقد أو بصمته لا يطابق الإصدار المثبت: {exc}",
            file=contract_file,
            expected=CONTRACT_FINGERPRINT,
            actual=payload.get("fingerprint")
            if isinstance(payload, dict)
            else type(payload).__name__,
        ) from exc


def _validate_step_contract(step):
    expected_values = {
        "source_file": CONSUMER_EXACT["source_file"],
        "thumbnail_file": CONSUMER_EXACT["thumbnail_file"],
        "topics_file": CONSUMER_EXACT["topics_file"],
        "font_name": FORMATTING["source"]["font_name"],
        "font_size": FORMATTING["source"]["font_size_pt"],
        "line_spacing": FORMATTING["source"]["line_spacing_pt"],
        "thumbnail_font_name": FORMATTING["thumbnail"]["font_name"],
        "thumbnail_font_size": FORMATTING["thumbnail"]["font_size_pt"],
        "thumbnail_line_spacing": FORMATTING["thumbnail"]["line_spacing_pt"],
    }
    for key, expected in expected_values.items():
        if key not in step:
            continue
        actual = step[key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                matches = float(actual) == float(expected)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise _RuntimeInputError(
                "REVIEW_RUNTIME_CONTRACT_MISMATCH",
                f"إعداد التشغيل {key} لا يطابق العقد المثبت",
                expected=expected,
                actual=actual,
            )


def _snapshot_selected_topics(step, ctx, selected_ids):
    topics_file = step.get("topics_file", CONSUMER_EXACT["topics_file"])
    topics_path = ctx.input_path(topics_file)
    if not os.path.isfile(topics_path):
        raise _RuntimeInputError(
            "TOPICS_FILE_MISSING",
            "ملف topics.json غير موجود",
            file=topics_file,
        )
    try:
        data = _strict_json_load(topics_path)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise _RuntimeInputError(
            "TOPICS_INVALID_JSON",
            f"تعذر قراءة topics.json كـ JSON صارم: {exc}",
            file=topics_file,
        ) from exc

    root_key = None
    if isinstance(data, dict):
        has_preferred = CONSUMER_EXACT["topic_root_preferred"] in data
        has_alternative = CONSUMER_EXACT["topic_root_alternative"] in data
        if has_preferred and has_alternative:
            raise _RuntimeInputError(
                "TOPICS_AMBIGUOUS_ROOT",
                "topics.json يحتوي titles وtopics معا",
                file=topics_file,
            )
        if not has_preferred and not has_alternative:
            raise _RuntimeInputError(
                "TOPICS_ROOT_NOT_LIST",
                "topics.json لا يحتوي قائمة titles أو topics",
                file=topics_file,
            )
        root_key = (
            CONSUMER_EXACT["topic_root_preferred"]
            if has_preferred
            else CONSUMER_EXACT["topic_root_alternative"]
        )
        items = data[root_key]
        declared_total = data.get("total_count")
        if declared_total is not None and (
            type(declared_total) is not int
            or not isinstance(items, list)
            or declared_total != len(items)
        ):
            raise _RuntimeInputError(
                "TOPICS_TOTAL_COUNT_MISMATCH",
                "total_count لا يساوي عدد عناصر قائمة العناوين",
                file=topics_file,
                expected=len(items) if isinstance(items, list) else "list",
                actual=declared_total,
            )
    else:
        items = data
    if not isinstance(items, list):
        raise _RuntimeInputError(
            "TOPICS_ROOT_NOT_LIST",
            "جذر topics.json المختار لازم يكون قائمة",
            file=topics_file,
            actual=type(items).__name__,
        )

    id_key = CONSUMER_EXACT["topic_id_key"]
    title_key = CONSUMER_EXACT["topic_title_key"]
    archive_ids = set()
    selected_set = set(selected_ids)
    selected_items = []
    ordered_selected_ids = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise _RuntimeInputError(
                "TOPIC_ITEM_NOT_OBJECT",
                f"عنصر topics.json رقم {index + 1} مش JSON object",
                file=topics_file,
                actual=type(item).__name__,
            )
        topic_id = item.get(id_key)
        if type(topic_id) is not int or topic_id < 1:
            raise _RuntimeInputError(
                "TOPIC_ID_INVALID",
                f"معرف العنصر رقم {index + 1} مش integer موجب",
                file=topics_file,
                actual=topic_id,
            )
        if topic_id in archive_ids:
            raise _RuntimeInputError(
                "TOPIC_ID_DUPLICATE",
                f"معرف الموضوع {topic_id} مكرر في topics.json",
                file=topics_file,
                actual=topic_id,
            )
        archive_ids.add(topic_id)
        title = item.get(title_key)
        if not isinstance(title, str) or not title.strip():
            raise _RuntimeInputError(
                "TOPIC_TITLE_EMPTY",
                f"عنوان الموضوع {topic_id} فارغ أو مش نص",
                file=topics_file,
                actual=title,
            )
        if topic_id in selected_set:
            selected_items.append(item)
            ordered_selected_ids.append(topic_id)

    missing = [topic_id for topic_id in selected_ids if topic_id not in archive_ids]
    if missing:
        raise _RuntimeInputError(
            "SELECTED_TOPICS_MISSING",
            "معرفات مختارة غير موجودة في topics.json",
            file=topics_file,
            expected=selected_ids,
            actual={"missing": missing},
        )

    if root_key is None:
        snapshot = selected_items
    else:
        snapshot = {
            key: value
            for key, value in data.items()
            if key
            not in {
                CONSUMER_EXACT["topic_root_preferred"],
                CONSUMER_EXACT["topic_root_alternative"],
                "total_count",
            }
        }
        snapshot[root_key] = selected_items
        snapshot["total_count"] = len(selected_items)

    snapshot_dir = os.path.join(
        ctx.output_dir,
        "_long_text_review_snapshot",
    )
    try:
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, "topics.json")
        with open(snapshot_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise _RuntimeInputError(
            "TOPICS_SNAPSHOT_WRITE_FAILED",
            f"تعذر حفظ لقطة topics.json: {exc}",
            file=topics_file,
        ) from exc
    return topics_file, snapshot_path, ordered_selected_ids, snapshot_dir


def _canonical_base_step(step, snapshot_dir):
    source_format = FORMATTING["source"]
    thumbnail_format = FORMATTING["thumbnail"]
    base_step = dict(step)
    base_step.update(
        {
            "strict": False,
            "require_topic_ids": True,
            "source_file": CONSUMER_EXACT["source_file"],
            "thumbnail_file": CONSUMER_EXACT["thumbnail_file"],
            "topics_file": CONSUMER_EXACT["topics_file"],
            "font_name": source_format["font_name"],
            "font_size": source_format["font_size_pt"],
            "line_spacing": source_format["line_spacing_pt"],
            "thumbnail_font_name": thumbnail_format["font_name"],
            "thumbnail_font_size": thumbnail_format["font_size_pt"],
            "thumbnail_line_spacing": thumbnail_format["line_spacing_pt"],
            "save_json": os.path.join(
                "_long_text_review_snapshot",
                "_base_structure_report.json",
            ),
            "save_text": os.path.join(
                "_long_text_review_snapshot",
                "_base_structure_report.txt",
            ),
            "save_regenerate": os.path.join(
                "_long_text_review_snapshot",
                "_base_topics_to_regenerate.json",
            ),
        }
    )
    return base_step


def _rebuild_final_report(step, ctx, cards):
    issues = cards["issues"]
    error_count = sum(item.get("severity") == "error" for item in issues)
    warning_count = sum(item.get("severity") == "warning" for item in issues)
    report = cards["structure_report"]
    report["contract_version"] = CONTRACT_VERSION
    report["contract_fingerprint"] = CONTRACT_FINGERPRINT
    report["structure_passed"] = error_count == 0
    report["error_count"] = error_count
    report["warning_count"] = warning_count
    report["ingestion_usable"] = error_count == 0
    report["quality_clean"] = error_count == 0 and warning_count == 0
    report["issues"] = issues

    blocked = defaultdict(list)
    global_errors = [
        item
        for item in issues
        if item.get("severity") == "error" and item.get("topic_id") is None
    ]
    for item in issues:
        if item.get("severity") == "error" and item.get("topic_id") is not None:
            blocked[str(item["topic_id"])].append(
                f"{item['code']}: {item['message']}"
            )
    if global_errors:
        for topic_id in cards["expected_ids"]:
            blocked[str(topic_id)].extend(
                f"{item['code']}: {item['message']}" for item in global_errors
            )
    cards["blocked"] = {
        key: list(dict.fromkeys(values)) for key, values in blocked.items()
    }
    cards["contract_version"] = CONTRACT_VERSION
    cards["contract_fingerprint"] = CONTRACT_FINGERPRINT

    json_name, text_name, regenerate_name = _report_names(step)
    _write_structure_reports(
        ctx,
        report,
        json_name=json_name,
        text_name=text_name,
        regenerate_name=regenerate_name,
    )


def action_long_text_review_build_cards_final(step, ctx):
    """Validate preconditions, review the immutable selection, then enforce strict."""
    try:
        selected_ids = _validate_selected_ids(ctx)
        _validate_review_contract(step, ctx)
        _validate_step_contract(step)
        (
            topics_file,
            snapshot_path,
            ordered_selected_ids,
            snapshot_dir,
        ) = _snapshot_selected_topics(step, ctx, selected_ids)
    except _RuntimeInputError as exc:
        _fail(step, ctx, exc)

    review_ctx = _TopicsSnapshotContext(
        ctx,
        topics_file,
        snapshot_path,
        ordered_selected_ids,
    )
    base_step = _canonical_base_step(step, snapshot_dir)
    cards = action_long_text_review_build_cards(base_step, review_ctx)

    issues = cards["issues"]
    seen = {
        json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        for item in issues
    }
    for topic_id, topic in cards["topics"].items():
        _validate_thumbnail_value(
            topic.get("fields", {}).get("facebook_thumbnail", ""),
            key="facebook_thumbnail",
            topic_id=topic_id,
            filename=CONSUMER_EXACT["source_file"],
            min_words=COUNTS["facebook_thumbnail_min_words"],
            max_words=COUNTS["facebook_thumbnail_max_words"],
            issues=issues,
            seen=seen,
        )

    _rebuild_final_report(step, ctx, cards)
    error_count = cards["structure_report"]["error_count"]
    log(
        "  long_text_review_final: "
        f"{len(cards['topics'])} موضوع | أخطاء={error_count} | "
        f"عقد={CONTRACT_FINGERPRINT}"
    )
    if _bool(step.get("strict", True), True) and error_count:
        json_name, text_name, _regenerate_name = _report_names(step)
        raise EngineError(
            f"فشل الفحص البنيوي لمخرجات تكست لونج ({error_count} خطأ). "
            f"راجع {text_name} و{json_name}",
            code="LONG_TEXT_STRUCTURE_FAILED",
        )
    return cards


__all__ = ["action_long_text_review_build_cards_final"]
