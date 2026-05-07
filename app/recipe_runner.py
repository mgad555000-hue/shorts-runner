"""
JSON Pipeline Runner
====================
يفسّر وصفات JSON وينفذها خطوة بخطوة.
كل خطوة بتستدعي دالة من engine.py.
"""

import os
import sys
import json
import re
import subprocess
import time
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import generate, tts, transcribe, transcribe_with_timestamps, batch_send, batch_retrieve, log, EngineError, BatchInfo, detect_provider


# ========== PipelineContext ==========

class PipelineContext:
    """سياق التنفيذ - يخزن المتغيرات والنتائج"""

    def __init__(self):
        self.input_dir = os.environ.get("INPUT_DIR", "/mnt/input")
        # RECIPE_OUTPUT_DIR = المجلد الدائم (data/channels/.../output)
        # OUTPUT_DIR = المجلد المؤقت (shorts/out/{run_id})
        # الأولوية لـ RECIPE_OUTPUT_DIR عشان نكتب مباشرة في المكان النهائي
        self.output_dir = os.environ.get("RECIPE_OUTPUT_DIR", "") or os.environ.get("OUTPUT_DIR", "/mnt/output")
        self._work_dir = os.environ.get("OUTPUT_DIR", "/mnt/output")  # المجلد المؤقت (للـ config فقط)
        # التأكد من وجود مجلد المخرجات
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
        self.tts_provider = os.environ.get("TTS_PROVIDER", "vertex")
        self.tts_voice = os.environ.get("TTS_VOICE_ID", "Achird")
        self.execution_mode = os.environ.get("EXECUTION_MODE", "instant")
        self.channel_name = os.environ.get("CHANNEL_NAME", "")
        self.run_id = os.environ.get("RUN_ID", "")
        self.recipe_name = os.environ.get("RECIPE_NAME", "")
        self.topic_ids = self._parse_topic_ids()
        self.results = {}
        # تتبع استهلاك التوكنز لكل خطوة
        self.usage_records = []  # [{step_id, call_type, provider, model, input, output, thinking, total}]

    def _parse_topic_ids(self):
        """قراءة TOPIC_IDS من البيئة وتحويلها لـ set"""
        raw = os.environ.get("TOPIC_IDS", "").strip()
        if not raw:
            return None
        ids = set()
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
        if ids:
            log(f"  TOPIC_IDS: {sorted(ids)}")
        return ids if ids else None

    def resolve(self, value):
        """حل المتغيرات: {step_id} -> نتيجة الخطوة"""
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        # لو القيمة كلها reference واحد -> ارجع الكائن الفعلي
        if stripped.startswith("{") and stripped.endswith("}") and stripped.count("{") == 1:
            ref = stripped[1:-1]
            if ref in self.results:
                return self.results[ref]

        # لو فيها references ضمن نص -> string interpolation
        result = value
        for step_id, step_result in self.results.items():
            result = result.replace(f"{{{step_id}}}", str(step_result))
        return result

    def resolve_list(self, value):
        """حل قائمة - كل عنصر يتحل"""
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        resolved = self.resolve(value)
        if isinstance(resolved, list):
            return resolved
        return [resolved]

    def record_usage(self, step_id: str, call_type: str, provider: str, model: str, token_usage: dict, send_run_id: str = None):
        """تسجيل استهلاك توكنز من API call"""
        if not token_usage or token_usage.get("total", 0) == 0:
            return
        record = {
            "step_id": step_id,
            "call_type": call_type,
            "provider": provider,
            "model": model,
            "input_tokens": token_usage.get("input", 0),
            "output_tokens": token_usage.get("output", 0),
            "thinking_tokens": token_usage.get("thinking", 0),
            "total_tokens": token_usage.get("total", 0),
        }
        if send_run_id:
            record["send_run_id"] = send_run_id
        self.usage_records.append(record)

    def save_usage_summary(self):
        """حفظ ملخص الاستهلاك في ملف JSON في مجلد الإخراج"""
        if not self.usage_records:
            return
        summary = {
            "run_id": self.run_id,
            "recipe_name": self.recipe_name,
            "records": self.usage_records,
            "totals": {
                "input_tokens": sum(r["input_tokens"] for r in self.usage_records),
                "output_tokens": sum(r["output_tokens"] for r in self.usage_records),
                "thinking_tokens": sum(r["thinking_tokens"] for r in self.usage_records),
                "total_tokens": sum(r["total_tokens"] for r in self.usage_records),
                "api_calls": len(self.usage_records),
            }
        }
        path = os.path.join(self.output_dir, "usage_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log(f"  [usage] حفظ ملخص الاستهلاك: {summary['totals']['total_tokens']} tokens في {len(self.usage_records)} calls")

    def input_path(self, filename):
        """مسار ملف في مجلد الإدخال"""
        return os.path.join(self.input_dir, filename)

    def output_path(self, filename):
        """مسار ملف في مجلد الإخراج"""
        return os.path.join(self.output_dir, filename)


# ========== Action Functions ==========

def action_read_input(step, ctx):
    """قراءة ملف نصي من INPUT_DIR — مع فلترة المواضيع لو TOPIC_IDS محدد"""
    filename = step["file"]
    filepath = ctx.input_path(filename)
    log(f"  قراءة ملف: {filepath}")

    if not os.path.exists(filepath):
        raise EngineError(f"الملف غير موجود: {filepath}", code="FILE_NOT_FOUND")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    log(f"  تم قراءة {len(content)} حرف")

    # فلترة المواضيع بـ TOPIC_IDS (لو الملف JSON فيه عناصر بـ id)
    if ctx.topic_ids and filename.endswith(".json"):
        content = _filter_topics_by_ids(content, ctx.topic_ids)

    return content


def action_read_json(step, ctx):
    """قراءة ملف JSON من INPUT_DIR — مع فلترة المواضيع لو TOPIC_IDS محدد"""
    filename = step["file"]
    filepath = ctx.input_path(filename)
    log(f"  قراءة JSON: {filepath}")

    if not os.path.exists(filepath):
        raise EngineError(f"الملف غير موجود: {filepath}", code="FILE_NOT_FOUND")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    log(f"  تم قراءة JSON ({type(data).__name__})")

    # فلترة المواضيع بـ TOPIC_IDS
    if ctx.topic_ids:
        data = _filter_topics_data(data, ctx.topic_ids)

    return data


def action_generate(step, ctx):
    """استدعاء engine.generate()"""
    prompt = ctx.resolve(step["input"])
    system_prompt = ctx.resolve(step.get("system_prompt", "")) if step.get("system_prompt") else ""
    temperature = step.get("temperature", 0.7)
    max_tokens = step.get("max_tokens", None)
    thinking_budget = step.get("thinking_budget", None)  # None = الموديل يقرر، 0 = بدون تفكير
    thinking_level = step.get("thinking_level", None)  # "low", "medium", "high" — أولوية أعلى من thinking_budget
    step_model = step.get("model", None)  # موديل خاص بالخطوة — لو None يستخدم ctx.model
    effective_model = step_model or ctx.model
    prompt_str = str(prompt)

    if step_model:
        log(f"  [model override] {step_model}")

    # Labels لتتبع التكلفة الفعلية في BigQuery Billing
    direct_labels = {}
    if ctx.run_id:
        direct_labels["run_id"] = ctx.run_id
    if ctx.recipe_name:
        direct_labels["recipe"] = ctx.recipe_name
    if ctx.channel_name:
        direct_labels["channel"] = ctx.channel_name
    if step.get("id"):
        direct_labels["step"] = step["id"]

    # === لو المدخل فيه أكتر من ماركر → توليد كل واحد لوحده بالتوازي ===
    per_marker_result = _generate_per_marker(prompt_str, ctx, system_prompt, temperature, max_tokens, thinking_budget, thinking_level, effective_model, labels=direct_labels)
    if per_marker_result is not None:
        text = per_marker_result
    else:
        # مدخل عادي (ماركر واحد أو بدون) → توليد عادي
        result = generate(
            prompt=prompt_str,
            model=effective_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
            labels=direct_labels,
        )
        if not result.success:
            raise EngineError(f"فشل التوليد: {result.error}", code="GENERATE_FAILED")
        text = result.data
        # تسجيل استهلاك التوكنز
        ctx.record_usage(step["id"], "direct", result.provider, result.model, result.token_usage)

    # حفظ لو محدد
    if step.get("save_as"):
        save_path = ctx.output_path(step["save_as"])
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        log(f"  تم حفظ النص في: {save_path}")

    return text


def _generate_per_marker(prompt_str, ctx, system_prompt, temperature, max_tokens, thinking_budget=None, thinking_level=None, effective_model=None, labels=None):
    """
    لو المدخل فيه أكتر من ماركر SCRIPT/INTRO → يقسّمه ويولّد كل واحد لوحده بالتوازي.
    بيرجع None لو المدخل مش multi-marker (يعني استخدم generate العادي).
    """
    model_to_use = effective_model or ctx.model
    MARKER_PAT = r'<<<((?:SCRIPT|INTRO)_\d+)>>>'
    MAX_WORKERS = 5

    # --- استخراج ماركرز المدخل من قسم المحتوى فقط (بعد آخر ---) ---
    # التعليمات (instructions.txt) ممكن تحتوي ماركرز مثال — نتجاهلها
    separator = "\n---\n"
    if separator in prompt_str:
        content_section = prompt_str[prompt_str.rfind(separator) + len(separator):]
    else:
        content_section = prompt_str

    seen = set()
    input_markers = []
    for m in re.findall(MARKER_PAT, content_section):
        if m not in seen:
            seen.add(m)
            input_markers.append(m)

    # فلترة الماركرز حسب TOPIC_IDS (لو محدد)
    if ctx.topic_ids and input_markers:
        filtered_markers = []
        for m in input_markers:
            id_match = re.search(r'_(\d+)', m)
            if id_match and int(id_match.group(1)) in ctx.topic_ids:
                filtered_markers.append(m)
        if filtered_markers:
            log(f"  [filter] فلترة الماركرز: {len(filtered_markers)} من {len(input_markers)} (TOPIC_IDS)")
            input_markers = filtered_markers

    if len(input_markers) <= 1:
        return None  # مش multi-marker — الـ caller يستخدم generate العادي

    # --- استخراج التعليمات (كل شيء قبل أول ماركر في المحتوى) ---
    content_start = prompt_str.rfind(separator) + len(separator) if separator in prompt_str else 0
    first_match_in_content = re.search(MARKER_PAT, content_section)
    if not first_match_in_content:
        return None
    instructions_part = prompt_str[:content_start + first_match_in_content.start()]

    # --- استخراج كل مقطع (من قسم المحتوى فقط) ---
    content_with_markers = prompt_str[content_start:]
    sections = {}
    for marker in input_markers:
        escaped = re.escape(f'<<<{marker}>>>')
        pattern = rf'({escaped}.*?)(?=<<<(?:SCRIPT|INTRO)_\d+>>>|\Z)'
        match = re.search(pattern, content_with_markers, re.DOTALL)
        if match:
            sections[marker] = match.group(1).strip()

    log(f"  [*] {len(input_markers)} ماركر — توليد كل واحد منفصلاً ({MAX_WORKERS} بالتوازي)...")

    # --- توليد كل واحد بالتوازي ---
    def _gen_one(marker):
        section = sections.get(marker)
        if not section:
            return marker, None, {}
        single_prompt = instructions_part + section
        # نحط marker كـ label إضافي لكل طلب (يفيد للتتبع داخل التشغيلة الواحدة)
        per_call_labels = dict(labels) if labels else {}
        per_call_labels["marker"] = marker.lower()
        result = generate(
            prompt=single_prompt,
            model=model_to_use,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
            labels=per_call_labels,
        )
        return marker, result.data if result.success else None, result.token_usage if result.success else {}

    results = {}
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_gen_one, m): m for m in input_markers}
        done_count = 0
        for future in as_completed(futures):
            marker, data, usage = future.result()
            done_count += 1
            if data:
                results[marker] = data
                # تسجيل التوكنز من كل marker
                ctx.record_usage(f"generate_{marker}", "direct", detect_provider(model_to_use) if hasattr(ctx, 'model') else "gemini", model_to_use, usage)
                log(f"  ✓ {marker} ({done_count}/{len(input_markers)})")
            else:
                failed.append(marker)
                log(f"  [!] فشل {marker} ({done_count}/{len(input_markers)})")

    # --- إعادة محاولة الفاشلين (مرة واحدة) ---
    if failed:
        log(f"  → إعادة محاولة {len(failed)} ماركر فاشل...")
        for marker in failed:
            _, data, usage = _gen_one(marker)
            if data:
                results[marker] = data
                ctx.record_usage(f"generate_{marker}_retry", "direct", detect_provider(model_to_use) if hasattr(ctx, 'model') else "gemini", model_to_use, usage)
                log(f"  ✓ {marker} (إعادة)")
            else:
                log(f"  [!!] فشل نهائي: {marker}")

    # --- تجميع بالترتيب الأصلي ---
    combined = []
    for marker in input_markers:
        if marker in results:
            combined.append(results[marker])

    log(f"  تم توليد {len(results)}/{len(input_markers)} ماركر بنجاح")

    if not combined:
        raise EngineError(f"فشل توليد كل الماركرز", code="GENERATE_ALL_FAILED")

    return "\n".join(combined)


def action_tts(step, ctx):
    """استدعاء engine.tts() + حفظ WAV + تحويل MP3"""
    text = str(ctx.resolve(step["input"]))

    # قطع النص لو محدد max_chars
    max_chars = step.get("max_chars")
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        log(f"  تم قطع النص إلى {max_chars} حرف")

    result = tts(text)

    if not result.success:
        raise EngineError(f"فشل TTS: {result.error}", code="TTS_FAILED")

    audio_data = result.data

    # حفظ WAV
    save_as = step.get("save_as", "audio")
    wav_path = ctx.output_path(f"{save_as}.wav")
    with open(wav_path, "wb") as f:
        f.write(audio_data)
    log(f"  تم حفظ WAV: {wav_path} ({len(audio_data)} bytes)")

    # تحويل لـ MP3
    mp3_path = ctx.output_path(f"{save_as}.mp3")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", "192k", mp3_path],
            capture_output=True, timeout=120
        )
        if os.path.exists(mp3_path):
            log(f"  تم تحويل MP3: {mp3_path}")
    except Exception as e:
        log(f"  [!] فشل تحويل MP3: {e}")

    return wav_path


def _tts_and_save(text, filename_base, ctx, max_chars=None, subfolder=None, max_retries=3):
    """دالة مساعدة: TTS لنص واحد + حفظ WAV — ترجع (True, wav_path) لو نجح أو (False, None)"""
    import time as _time
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]

    if subfolder:
        out_dir = os.path.join(ctx.output_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        wav_path = os.path.join(out_dir, f"{filename_base}.wav")
    else:
        wav_path = ctx.output_path(f"{filename_base}.wav")

    for attempt in range(1, max_retries + 1):
        result = tts(text)
        if not result.success:
            if attempt < max_retries:
                log(f"  [!] {filename_base}: محاولة {attempt}/{max_retries} فشلت — {result.error} — إعادة بعد 3s")
                _time.sleep(3)
                continue
            log(f"  [!] {filename_base}: فشل TTS بعد {max_retries} محاولات — {result.error}")
            return False, None

        audio_data = result.data
        with open(wav_path, "wb") as f:
            f.write(audio_data)

        if attempt > 1:
            log(f"  {filename_base}: OK ({len(audio_data)} bytes) [نجح في المحاولة {attempt}]")
        else:
            log(f"  {filename_base}: OK ({len(audio_data)} bytes)")
        return True, wav_path

    return False, None


def action_tts_multi(step, ctx):
    """
    تحويل نص بتنسيق MG Ranner لملفات صوتية WAV.
    - شورتس (بدون PART): كل سكريبت → WAV واحد في audio/
    - لونج (مع PART): كل موضوع → مجلد SCRIPT_N/ — كل جزء → WAV

    الضمانات الإلزامية:
    1. كل ملف: TTS → Whisper إلزامي → لو تطابق < min_match → إعادة TTS
    2. WAV الفاشل يُحذف فوراً قبل إعادة المحاولة (لا يبقى ملف سيئ على الديسك)
    3. loop تلقائي على الفاشلين حتى max_passes جولات
    4. كل ملف في المخرج = verified بـ Whisper بدون استثناء
    5. لو فضل ناقص بعد max_passes → الوصفة تنهي بـ FAILED
    """
    import time as _time
    text = str(ctx.resolve(step["input"]))
    prefix = step.get("marker_prefix", "SCRIPT")
    max_chars = step.get("max_chars")
    min_match = step.get("min_match", 0.7)
    language = step.get("language", "ar")
    tts_retries = step.get("tts_retries", 3)    # محاولات TTS داخل كل جولة
    max_passes = step.get("max_passes", 5)       # أقصى عدد جولات retry

    parts = re.split(rf'<<<{prefix}_(\d+)>>>', text)
    if len(parts) < 3:
        raise EngineError(
            f"لم يتم العثور على ماركرز <<<{prefix}_N>>> في النص",
            code="NO_MARKERS_FOUND"
        )

    # بناء قائمة كل الملفات المطلوبة
    all_items = []
    for i in range(1, len(parts), 2):
        script_num = parts[i]
        script_text = parts[i + 1] if i + 1 < len(parts) else ""
        script_text = script_text.replace(f"<<<END_{prefix}>>>", "").strip()
        if not script_text:
            continue
        part_parts = re.split(r'<<<PART_(\d+)>>>', script_text)
        if len(part_parts) >= 3:
            topic_folder = f"{prefix}_{script_num}"
            for j in range(1, len(part_parts), 2):
                part_num = part_parts[j]
                part_text = part_parts[j + 1] if j + 1 < len(part_parts) else ""
                part_text = part_text.replace("<<<END_PART>>>", "").strip()
                if part_text:
                    all_items.append({"filename": f"{prefix}_{script_num}_{part_num}", "text": part_text, "subfolder": topic_folder})
        else:
            all_items.append({"filename": f"{prefix}_{script_num}", "text": script_text, "subfolder": "audio"})

    total = len(all_items)
    log(f"  إجمالي الملفات المطلوبة: {total}")

    def get_wav_path(item):
        subfolder = item["subfolder"]
        if subfolder:
            out_dir = os.path.join(ctx.output_dir, subfolder)
            os.makedirs(out_dir, exist_ok=True)
            return os.path.join(out_dir, f"{item['filename']}.wav")
        return ctx.output_path(f"{item['filename']}.wav")

    def process_one(item, pass_num):
        """TTS + Whisper لملف واحد. يحذف WAV لو فشل التحقق. ترجع (True, match_pct) أو (False, reason)"""
        filename = item["filename"]
        text_content = item["text"]
        wav_path = get_wav_path(item)

        log(f"  {filename}: TTS ({len(text_content)} حرف) [جولة {pass_num}]...")

        for attempt in range(1, tts_retries + 1):
            # حذف أي WAV قديم/سيئ قبل البدء
            if os.path.exists(wav_path):
                os.remove(wav_path)

            ok, _ = _tts_and_save(text_content, filename, ctx, max_chars, subfolder=item["subfolder"])
            if not ok:
                if attempt < tts_retries:
                    log(f"  [!] {filename}: TTS فشل ({attempt}/{tts_retries}) — إعادة بعد 5s")
                    _time.sleep(5)
                    continue
                return False, "tts_failed"

            # Whisper تحقق إلزامي
            try:
                w_result = transcribe(wav_path, language=language)
                if not w_result.success:
                    log(f"  [!] {filename}: Whisper فشل — {w_result.error} — إعادة TTS")
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                    if attempt < tts_retries:
                        _time.sleep(3)
                        continue
                    return False, "whisper_failed"

                similarity = _calculate_text_similarity(text_content, w_result.data)
                match_pct = round(similarity * 100, 1)

                if similarity >= min_match:
                    log(f"  ✅ {filename}: Whisper {match_pct}%")
                    return True, match_pct
                else:
                    log(f"  ⚠️ {filename}: Whisper {match_pct}% < {int(min_match*100)}% — حذف WAV وإعادة TTS ({attempt}/{tts_retries})")
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                    if attempt < tts_retries:
                        _time.sleep(3)
                        continue
                    return False, f"low_match_{match_pct}"

            except Exception as e:
                log(f"  [!] {filename}: خطأ Whisper — {str(e)[:100]}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                if attempt < tts_retries:
                    _time.sleep(3)
                    continue
                return False, f"whisper_error"

        return False, "max_tts_retries"

    # === loop تلقائي: حتى max_passes جولات ===
    success_count = 0
    pending = list(all_items)  # الملفات المتبقية

    for pass_num in range(1, max_passes + 1):
        if not pending:
            break

        log(f"  === جولة {pass_num}/{max_passes}: {len(pending)} ملف ===")
        still_pending = []

        for item in pending:
            ok, reason = process_one(item, pass_num)
            if ok:
                success_count += 1
            else:
                still_pending.append({**item, "last_reason": str(reason)})

        pending = still_pending

        if pending and pass_num < max_passes:
            wait = min(10 * pass_num, 60)  # انتظار متصاعد: 10s, 20s, 30s...
            log(f"  ↻ {len(pending)} ملف فاشل — انتظار {wait}s قبل الجولة التالية...")
            _time.sleep(wait)

    log(f"  === TTS Multi: {success_count} نجح | {len(pending)} فشل نهائي من {total} ===")

    if pending:
        failed_path = ctx.output_path("failed_tts.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(pending, f, ensure_ascii=False, indent=2)
        log(f"  ✗ failed_tts.json: {len(pending)} ملف فشل بعد {max_passes} جولات")
        raise EngineError(
            f"الوصفة لم تكتمل — {len(pending)} ملف فاشل بعد {max_passes} جولات. راجع failed_tts.json",
            code="TTS_INCOMPLETE"
        )

    if success_count == 0:
        raise EngineError("فشل TTS لكل السكريبتات", code="TTS_MULTI_ALL_FAILED")

    return f"تم تحويل {success_count} ملف صوتي — جميع الملفات تم التحقق منها بـ Whisper ✅"


def action_transcribe(step, ctx):
    """استدعاء engine.transcribe()"""
    audio_file = str(ctx.resolve(step["input"]))
    language = step.get("language", None)

    result = transcribe(audio_file, language=language)

    if not result.success:
        raise EngineError(f"فشل Transcribe: {result.error}", code="TRANSCRIBE_FAILED")

    text = result.data

    # حفظ لو محدد
    if step.get("save_as"):
        save_path = ctx.output_path(step["save_as"])
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        log(f"  تم حفظ النص في: {save_path}")

    return text


def action_batch_send(step, ctx):
    """استدعاء engine.batch_send()"""
    prompts = ctx.resolve_list(step["prompts"])
    system_prompt = ctx.resolve(step.get("system_prompt", "")) if step.get("system_prompt") else ""
    temperature = step.get("temperature", 0.7)
    max_tokens = step.get("max_tokens", 8192)
    thinking_budget = step.get("thinking_budget", None)
    thinking_level = step.get("thinking_level", None)
    step_model = step.get("model", None)
    effective_model = step_model or ctx.model

    if step_model:
        log(f"  [model override] {step_model}")

    save_path = None
    if step.get("save_as"):
        save_path = ctx.output_path(step["save_as"])

    # Labels للتتبع في Google Cloud — run_id الكامل يظهر في display_name
    batch_labels = {}
    if ctx.run_id:
        batch_labels["run_id"] = ctx.run_id
        log(f"  [TRACKING] Run ID → Google Cloud: {ctx.run_id}")
    if ctx.recipe_name:
        batch_labels["recipe"] = ctx.recipe_name
    if ctx.channel_name:
        batch_labels["channel"] = ctx.channel_name
    batch_labels["step"] = step["id"]

    # method: vertex = labels تظهر في Google Cloud Billing، sdk = بدون labels
    batch_method = step.get("method", "vertex")
    log(f"  [BATCH] method={batch_method} (vertex = labels في Billing)")

    result = batch_send(
        prompts=prompts,
        model=effective_model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        save_path=save_path,
        method=batch_method,
        thinking_budget=thinking_budget,
        thinking_level=thinking_level,
        labels=batch_labels if batch_labels else None,
    )

    if not result.success:
        raise EngineError(f"فشل Batch Send: {result.error}", code="BATCH_SEND_FAILED")

    # لو مفيش save_as، نحفظ في المسار الافتراضي
    if not save_path:
        save_path = ctx.output_path("batch_job_info.json")
        result.data.save(save_path)

    log(f"  تم حفظ معلومات الدفعة في: {save_path}")
    return save_path


def action_batch_retrieve(step, ctx):
    """استدعاء engine.batch_retrieve() مع polling"""
    batch_info_path = str(ctx.resolve(step["input"]))
    poll_interval = step.get("poll_interval", 30)
    max_wait = step.get("max_wait", 3600)

    log(f"  انتظار نتائج الدفعة (كل {poll_interval}s، حد أقصى {max_wait}s)")

    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise EngineError(
                f"تجاوز الوقت المسموح ({max_wait}s) في انتظار نتائج الدفعة",
                code="BATCH_RETRIEVE_TIMEOUT"
            )

        try:
            result = batch_retrieve(batch_info_path=batch_info_path)
            if result.success:
                log(f"  تم استقبال {len(result.data)} نتيجة")
                # استخراج send_run_id الأصلي من batch_info (اللي اتبعت لجوجل)
                _send_run_id = None
                try:
                    from engine import BatchInfo
                    _bi = BatchInfo.load(batch_info_path)
                    _send_run_id = (_bi.extra or {}).get("labels", {}).get("run_id", None)
                except Exception:
                    pass
                # تسجيل استهلاك التوكنز من الباتش
                ctx.record_usage(step["id"], "batch", result.provider, result.model, result.token_usage, send_run_id=_send_run_id)
                return result.data
        except EngineError as e:
            if e.code == "BATCH_JOB_NOT_READY":
                log(f"  المهمة لم تكتمل بعد ({int(elapsed)}s). انتظار {poll_interval}s...")
                time.sleep(poll_interval)
                continue
            raise

    return []


def action_save_file(step, ctx):
    """حفظ محتوى في ملف"""
    content = ctx.resolve(step["input"])
    filename = step["save_as"]
    filepath = ctx.output_path(filename)

    # لو dict أو list -> JSON
    if isinstance(content, (dict, list)):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(str(content))

    log(f"  تم حفظ الملف: {filepath}")
    return filepath


def action_format_text(step, ctx):
    """معالجة النص: مسافات بعد علامات الترقيم + تلوين كلمات"""
    text = str(ctx.resolve(step["input"]))

    # الكلمات اللي هنلونها بـ <r>
    color_words = step.get("color_words", ["بس", "ده", "دي", "دى", "دول"])
    spaces = step.get("spaces_after_punctuation", 5)
    space_str = " " * spaces

    # --- 1) إضافة مسافات بعد الفاصلة والنقطة ---
    # بس مش جوا الـ tags ولا الـ markers <<<...>>>
    def add_spaces(t):
        result = []
        i = 0
        while i < len(t):
            # تخطي markers <<<...>>>
            if t[i:i+3] == "<<<":
                end = t.find(">>>", i)
                if end != -1:
                    result.append(t[i:end+3])
                    i = end + 3
                    continue
            # تخطي tags <r> و </r>
            if t[i] == "<" and (t[i:i+3] == "<r>" or t[i:i+4] == "</r>"):
                tag_end = t.find(">", i)
                if tag_end != -1:
                    result.append(t[i:tag_end+1])
                    i = tag_end + 1
                    continue
            # فاصلة أو نقطة
            if t[i] in ("،", ",", "."):
                # تجاهل لو الفاصلة/النقطة بين رقمين (مثل: 100,000 أو 1.5)
                prev_char = t[i-1] if i > 0 else ""
                next_char = t[i+1] if i+1 < len(t) else ""
                if prev_char.isdigit() and next_char.isdigit():
                    result.append(t[i])
                    i += 1
                    continue
                result.append(t[i])
                # شيل المسافات الموجودة بعدها
                j = i + 1
                while j < len(t) and t[j] == " ":
                    j += 1
                # لو بعدها سطر جديد أو نهاية النص، متضفش مسافات
                if j < len(t) and t[j] not in ("\n", "\r"):
                    result.append(space_str)
                i = j
                continue
            result.append(t[i])
            i += 1
        return "".join(result)

    text = add_spaces(text)

    # --- 2) تلوين الكلمات ---
    for word in color_words:
        # لو الكلمة مش ملفوفة أصلاً بـ <r>
        # pattern: الكلمة لوحدها (مش جوا <r>)
        pattern = r'(?<!<r>)(?<!\w)(' + re.escape(word) + r')(?!\w)(?!</r>)'
        text = re.sub(pattern, r'<r>\1</r>', text)

    log(f"  تم تنسيق النص ({len(text)} حرف)")

    # حفظ لو محدد
    if step.get("save_as"):
        save_path = ctx.output_path(step["save_as"])
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        log(f"  تم حفظ النص في: {save_path}")

    return text


def _set_paragraph_rtl(paragraph):
    """ضبط RTL + Right Alignment على فقرة Word مع احترام ترتيب OOXML schema"""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    paragraph.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    pPr = paragraph._element.get_or_add_pPr()

    # إزالة bidi و textDirection القديمة
    for tag in ('w:bidi', 'w:textDirection'):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)

    # العناصر التي تأتي بعد bidi في OOXML schema — bidi يجب أن يكون قبلها
    elements_after_bidi = (
        'w:adjustRightInd', 'w:snapToGrid', 'w:spacing', 'w:ind',
        'w:contextualSpacing', 'w:mirrorIndents', 'w:suppressOverlap',
        'w:jc', 'w:textDirection', 'w:textAlignment', 'w:textboxTightWrap',
        'w:outlineLvl', 'w:divId', 'w:cnfStyle', 'w:rPr', 'w:sectPr',
    )

    bidi = OxmlElement('w:bidi')
    bidi.set(qn('w:val'), '1')

    # إيجاد أول عنصر يأتي بعد bidi في الـ schema، وإدراج bidi قبله
    insert_before = None
    for el in pPr:
        tag_local = el.tag.split('}', 1)[-1]
        if f'w:{tag_local}' in elements_after_bidi:
            insert_before = el
            break

    if insert_before is not None:
        insert_before.addprevious(bidi)
    else:
        pPr.append(bidi)

    # ضمان jc=right (في حال لم يضعه python-docx)
    jc = pPr.find(qn('w:jc'))
    if jc is None:
        jc = OxmlElement('w:jc')
        pPr.append(jc)
    jc.set(qn('w:val'), 'right')


def _set_document_rtl(doc):
    """ضبط اتجاه RTL على مستوى المستند (Section بس).
    ⚠️ مش بنلمس Normal style — لازم يفضل default علشان الفقرات
    اللي فيها bidi/jc تبان كـ override → Word يظلّل زرار Align Right."""
    for section in doc.sections:
        sectPr = section._sectPr
        if sectPr.find(qn('w:bidi')) is None:
            bidi = OxmlElement('w:bidi')
            sectPr.append(bidi)

    # تنظيف Normal style لو فيه bidi/jc من تشغيل قديم
    try:
        normal_style = doc.styles["Normal"]
        pPr = normal_style.element.find(qn('w:pPr'))
        if pPr is not None:
            for tag in ('w:bidi', 'w:jc'):
                for el in pPr.findall(qn(tag)):
                    pPr.remove(el)
    except Exception:
        pass


def _set_run_rtl(run):
    """ضبط اتجاه RTL على Run في Word"""
    rPr = run._element.get_or_add_rPr()
    rtl = OxmlElement('w:rtl')
    rtl.set(qn('w:val'), '1')
    rPr.append(rtl)


def _post_process_docx_force_rtl(filepath):
    """
    ضمانة نهائية تطابق نمط Word-native بالضبط:
    - paragraph: bidi=0 + jc=right (مش bidi=1 — ده اللي كان بيخلي COM يقول Left)
    - paragraph mark rPr: <w:lang w:bidi="ar-SA"/>
    - run rPr: <w:rtl/> + <w:lang w:bidi="ar-SA"/>
    - Normal style: <w:lang w:bidi="ar-EG"/>

    اكتشفت ده بعد ما خليت Word ينشئ ملف RTL وعملت compare للـ XML.
    """
    import zipfile
    from lxml import etree as ET

    W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    W = f'{{{W_NS}}}'

    # ترتيب OOXML schema الكامل لـ CT_PPr / CT_PPrBase children
    PPR_ORDER = [
        'pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr',
        'widowControl', 'numPr', 'suppressLineNumbers', 'pBdr', 'shd',
        'tabs', 'suppressAutoHyphens', 'kinsoku', 'wordWrap', 'overflowPunct',
        'topLinePunct', 'autoSpaceDE', 'autoSpaceDN', 'bidi',
        'adjustRightInd', 'snapToGrid', 'spacing', 'ind', 'contextualSpacing',
        'mirrorIndents', 'suppressOverlap', 'jc', 'textDirection',
        'textAlignment', 'textboxTightWrap', 'outlineLvl', 'divId', 'cnfStyle',
        'rPr', 'sectPr', 'pPrChange',
    ]
    PPR_ORDER_INDEX = {tag: idx for idx, tag in enumerate(PPR_ORDER)}

    # ترتيب OOXML schema لـ CT_SectPr children (للأقسام)
    SECTPR_ORDER = [
        'headerReference', 'footerReference', 'footnotePr', 'endnotePr',
        'type', 'pgSz', 'pgMar', 'paperSrc', 'pgBorders', 'lnNumType',
        'pgNumType', 'cols', 'formProt', 'vAlign', 'noEndnote', 'titlePg',
        'textDirection', 'bidi', 'rtlGutter', 'docGrid', 'printerSettings',
        'sectPrChange',
    ]
    SECTPR_ORDER_INDEX = {tag: idx for idx, tag in enumerate(SECTPR_ORDER)}

    def _local(tag):
        return tag.split('}', 1)[-1] if '}' in tag else tag

    LANG_CODE = 'ar-SA'

    def _ensure_lang_in_rPr(rPr, with_rtl=False):
        """ضمان <w:lang w:bidi='ar-SA'/> + اختياري <w:rtl/> في rPr."""
        # إزالة lang/rtl قديمة
        for tag in ('lang', 'rtl'):
            for el in list(rPr.findall(f'{W}{tag}')):
                rPr.remove(el)
        if with_rtl:
            ET.SubElement(rPr, f'{W}rtl')
        lang = ET.SubElement(rPr, f'{W}lang')
        lang.set(f'{W}bidi', LANG_CODE)

    def _rebuild_pPr(pPr):
        """نمط Word-native: bidi=0 + jc=right + paragraph mark rPr فيه lang (بدون rtl)."""
        children = list(pPr)
        for child in children:
            pPr.remove(child)
        children = [c for c in children if _local(c.tag) not in ('bidi', 'jc')]

        # bidi=0 (LTR paragraph — زي اللي Word بيكتب لـ Arabic right-aligned)
        bidi = ET.Element(f'{W}bidi')
        bidi.set(f'{W}val', '0')
        children.append(bidi)

        # jc=right
        jc = ET.Element(f'{W}jc')
        jc.set(f'{W}val', 'right')
        children.append(jc)

        # paragraph mark rPr — lang بس (مش rtl)
        existing_rPr = None
        for c in children:
            if _local(c.tag) == 'rPr':
                existing_rPr = c
                break
        if existing_rPr is None:
            existing_rPr = ET.Element(f'{W}rPr')
            children.append(existing_rPr)
        _ensure_lang_in_rPr(existing_rPr, with_rtl=False)

        children.sort(key=lambda c: PPR_ORDER_INDEX.get(_local(c.tag), 999))
        for c in children:
            pPr.append(c)

    def _rebuild_sectPr(sectPr):
        """نمط Word-native: مفيش <w:bidi/> على section."""
        for el in list(sectPr.findall(f'{W}bidi')):
            sectPr.remove(el)

    with zipfile.ZipFile(filepath, 'r') as zin:
        contents = {name: zin.read(name) for name in zin.namelist()}

    # ⚠️ نلمس بس document.xml + headers/footers — مش styles.xml
    # علشان الـ Normal style يفضل default، فالـ direct formatting يبان كـ override
    target_files = [
        n for n in contents
        if n.startswith('word/') and n.endswith('.xml') and (
            'document' in n or 'header' in n or 'footer' in n
        )
    ]

    # رفع compatibilityMode من 14 (Word 2010) لـ 15 (Word 2013+)
    # علشان "Compatibility Mode" ميتفعّلش في Word
    if 'word/settings.xml' in contents:
        try:
            settings_root = ET.fromstring(contents['word/settings.xml'])
            for cs in settings_root.iter(f'{W}compatSetting'):
                if cs.get(f'{W}name') == 'compatibilityMode':
                    cs.set(f'{W}val', '15')
            contents['word/settings.xml'] = ET.tostring(
                settings_root, xml_declaration=True, encoding='UTF-8', standalone=True
            )
        except Exception:
            pass

    for tf in target_files:
        try:
            root = ET.fromstring(contents[tf])
        except Exception:
            continue

        for pPr in root.iter(f'{W}pPr'):
            _rebuild_pPr(pPr)

        for sectPr in root.iter(f'{W}sectPr'):
            _rebuild_sectPr(sectPr)

        # ضمان <w:rtl/> + <w:lang w:bidi='ar-SA'/> على كل run في المستند
        for r in root.iter(f'{W}r'):
            if r.getparent() is not None and _local(r.getparent().tag) == 'pPr':
                continue
            rPr = r.find(f'{W}rPr')
            if rPr is None:
                rPr = ET.Element(f'{W}rPr')
                r.insert(0, rPr)
            _ensure_lang_in_rPr(rPr, with_rtl=True)

        contents[tf] = ET.tostring(
            root, xml_declaration=True, encoding='UTF-8', standalone=True
        )

    # كتابة الـ docx من جديد
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name, data in contents.items():
            zout.writestr(name, data)


def _set_run_arabic_font(run, font_name, font_size):
    """ضبط الخط للنص العربي على مستوى Run (Complex Script font)"""
    if not font_name and not font_size:
        return
    rPr = run._element.get_or_add_rPr()
    if font_name:
        rFonts = rPr.find(qn('w:rFonts'))
        if rFonts is None:
            rFonts = OxmlElement('w:rFonts')
            rPr.insert(0, rFonts)
        rFonts.set(qn('w:ascii'), font_name)
        rFonts.set(qn('w:hAnsi'), font_name)
        rFonts.set(qn('w:cs'), font_name)
        rFonts.set(qn('w:eastAsia'), font_name)
    if font_size:
        size_half_points = str(int(font_size * 2))
        sz = rPr.find(qn('w:sz'))
        if sz is None:
            sz = OxmlElement('w:sz')
            rPr.append(sz)
        sz.set(qn('w:val'), size_half_points)
        szCs = rPr.find(qn('w:szCs'))
        if szCs is None:
            szCs = OxmlElement('w:szCs')
            rPr.append(szCs)
        szCs.set(qn('w:val'), size_half_points)


def action_save_docx(step, ctx):
    """حفظ النص المنسق كملف Word مع تلوين <r> بالأحمر"""
    text = str(ctx.resolve(step["input"]))
    filename = step.get("save_as", "output.docx")
    if not filename.endswith(".docx"):
        filename += ".docx"
    filepath = ctx.output_path(filename)

    # حماية: رفض حفظ docx فاضي
    if not text or not text.strip():
        msg = f"رفض حفظ {filename}: الـ input فاضي تماماً — احتمال فشل في خطوة سابقة"
        log(f"  [XX] save_docx: {msg}")
        raise RuntimeError(msg)

    doc = Document()

    font_name = step.get("font_name", "Arial")
    font_size = step.get("font_size", 14)

    # ضبط RTL على مستوى المستند كله (Section + Default Style)
    _set_document_rtl(doc)

    # ضبط RTL للمستند + خط Normal style
    style = doc.styles["Normal"]
    style.font.size = Pt(font_size)
    style.font.name = font_name

    # تقسيم النص بالماركر (SCRIPT أو INTRO أو أي prefix تاني)
    prefix = step.get("marker_prefix", "SCRIPT")
    sections = re.split(rf'<<<{prefix}_\d+>>>', text)
    titles = re.findall(rf'<<<{prefix}_(\d+)>>>', text)

    def _add_text_to_doc(doc, content, line_spacing):
        """إضافة نص لملف Word مع معالجة <r> tags"""
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        _set_paragraph_rtl(para)
        if isinstance(line_spacing, (int, float)) and line_spacing <= 5:
            para.paragraph_format.line_spacing = float(line_spacing)
        else:
            para.paragraph_format.line_spacing = Pt(line_spacing)

        parts = re.split(r'(<r>.*?</r>)', content)
        for part in parts:
            m = re.match(r'<r>(.*?)</r>', part)
            if m:
                run = para.add_run(m.group(1))
                run.font.color.rgb = RGBColor(255, 0, 0)
                _set_run_rtl(run)
                _set_run_arabic_font(run, font_name, font_size)
            else:
                if part:
                    run = para.add_run(part)
                    _set_run_rtl(run)
                    _set_run_arabic_font(run, font_name, font_size)

    line_spacing = step.get("line_spacing", 28)

    for idx, section in enumerate(sections):
        section = section.replace(f"<<<END_{prefix}>>>", "").strip()
        if not section:
            continue

        # عنوان السكريبت
        if idx > 0 and idx - 1 < len(titles):
            heading = doc.add_heading(f"Script {titles[idx-1]}", level=2)
            heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            _set_paragraph_rtl(heading)

        # معالجة PART markers لو موجودة
        part_sections = re.split(r'<<<PART_(\d+)>>>', section)
        if len(part_sections) > 1:
            # فيه PART markers — نعالجها
            # أول قطعة قبل أي PART (لو فيها محتوى)
            pre_part = part_sections[0].replace("<<<END_PART>>>", "").strip()
            if pre_part:
                _add_text_to_doc(doc, pre_part, line_spacing)

            for pi in range(1, len(part_sections), 2):
                part_num = part_sections[pi]
                part_content = part_sections[pi + 1].replace("<<<END_PART>>>", "").strip() if pi + 1 < len(part_sections) else ""

                # عنوان الجزء
                part_heading = doc.add_heading(f"Part {part_num}", level=3)
                part_heading.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                _set_paragraph_rtl(part_heading)

                if part_content:
                    _add_text_to_doc(doc, part_content, line_spacing)
        else:
            # مفيش PART markers — نص عادي
            _add_text_to_doc(doc, section, line_spacing)

    doc.save(filepath)

    # ضمانة نهائية: post-process لإجبار RTL + Right Alignment على كل XML
    try:
        _post_process_docx_force_rtl(filepath)
        log(f"  تم تطبيق post-process لـ RTL/Right Alignment")
    except Exception as e:
        log(f"  [!] post-process فشل (الملف محفوظ بدون post-process): {e}")

    log(f"  تم حفظ Word: {filepath}")
    return filepath


def action_template(step, ctx):
    """string formatting مع متغيرات"""
    text = step["text"]
    result = ctx.resolve(text)

    # حفظ لو محدد
    if step.get("save_as"):
        save_path = ctx.output_path(step["save_as"])
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(str(result))
        log(f"  تم حفظ النص في: {save_path}")

    return result


def _read_single_docx(filepath, reconstruct_markers, prefix):
    """قراءة ملف Word واحد — مع إعادة بناء الماركرز لو مطلوب"""
    doc = Document(filepath)

    if not reconstruct_markers:
        text = "\n".join([p.text for p in doc.paragraphs])
        return text

    # إعادة بناء الماركرز من headings "Script N" و "Part N"
    # الخطوة 1: تجميع كل paragraphs في بلوكات حسب الـ headings
    blocks = []  # [(type, num, lines), ...] — type: "script" أو "part"
    current_type = None
    current_num = None
    current_lines = []

    for p in doc.paragraphs:
        is_heading = p.style.name.startswith("Heading")
        script_match = re.match(r'Script\s+(\d+)', p.text) if is_heading else None
        part_match = re.match(r'Part\s+(\d+)', p.text) if is_heading else None

        if script_match:
            if current_type:
                blocks.append((current_type, current_num, current_lines))
            current_type = "script"
            current_num = script_match.group(1)
            current_lines = []
        elif part_match:
            if current_type:
                blocks.append((current_type, current_num, current_lines))
            current_type = "part"
            current_num = part_match.group(1)
            current_lines = []
        else:
            current_lines.append(p.text)

    if current_type:
        blocks.append((current_type, current_num, current_lines))

    # الخطوة 2: بناء الماركرز — تجميع الأجزاء تحت السكريبتات
    sections = []
    i = 0
    while i < len(blocks):
        btype, bnum, blines = blocks[i]
        if btype == "script":
            body_parts = []
            body_text = "\n".join(blines).strip()
            if body_text:
                body_parts.append(body_text)

            # نشوف لو اللي بعده parts تابعة لنفس السكريبت
            j = i + 1
            while j < len(blocks) and blocks[j][0] == "part":
                pt, pn, pl = blocks[j]
                part_body = "\n".join(pl).strip()
                body_parts.append(f"<<<PART_{pn}>>>\n{part_body}\n<<<END_PART>>>")
                j += 1

            full_body = "\n".join(body_parts)
            sections.append(f"<<<{prefix}_{bnum}>>>\n{full_body}\n<<<END_{prefix}>>>")
            i = j
        else:
            i += 1

    result = "\n\n".join(sections)
    log(f"  تم إعادة بناء {len(sections)} ماركر من headings")
    return result


def action_read_docx(step, ctx):
    """قراءة ملف/ملفات Word من مجلد الإدخال"""
    filename = step.get("file")
    reconstruct_markers = step.get("reconstruct_markers", False)
    prefix = step.get("marker_prefix", "SCRIPT")

    if filename:
        # ملف واحد محدد
        filepath = ctx.input_path(filename)
        if not os.path.exists(filepath):
            raise EngineError(f"ملف Word غير موجود: {filepath}", code="FILE_NOT_FOUND")
        text = _read_single_docx(filepath, reconstruct_markers, prefix)
        log(f"  تم قراءة Word: {filepath} ({len(text)} حرف)")
        return text
    else:
        # كل ملفات Word في مجلد الإدخال
        docx_files = sorted([f for f in os.listdir(ctx.input_dir) if f.endswith('.docx')])
        if not docx_files:
            raise EngineError(f"لا توجد ملفات Word في: {ctx.input_dir}", code="FILE_NOT_FOUND")

        all_text = []
        for fname in docx_files:
            filepath = ctx.input_path(fname)
            text = _read_single_docx(filepath, reconstruct_markers, prefix)
            all_text.append(text)
            log(f"  تم قراءة: {fname} ({len(text)} حرف)")

        combined = "\n\n".join(all_text)
        log(f"  إجمالي: {len(docx_files)} ملف Word ({len(combined)} حرف)")
        return combined


def action_read_excel(step, ctx):
    """قراءة ملف Excel من مجلد الإدخال وإرجاع القائمة كنص"""
    import openpyxl

    filename = step["file"]
    filepath = ctx.input_path(filename)
    log(f"  قراءة Excel: {filepath}")

    if not os.path.exists(filepath):
        raise EngineError(f"ملف Excel غير موجود: {filepath}", code="FILE_NOT_FOUND")

    wb = openpyxl.load_workbook(filepath, read_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if len(rows) < 2:
        raise EngineError("ملف Excel فارغ أو بدون بيانات", code="EMPTY_FILE")

    # تحويل لنص منسق: (رقم) عنوان — مع تجاهل صفوف فاضية
    lines = []
    for row in rows[1:]:
        num = row[0]
        if num is None:
            continue
        title = row[1] if len(row) > 1 and row[1] else ""
        lines.append(f"({num}) {title}")

    result = "\n".join(lines)
    log(f"  تم قراءة {len(rows) - 1} فيديو من Excel")
    return result


def action_copy_videos(step, ctx):
    """تحليل مخرجات AI واستخراج أرقام الفيديوهات ونسخها لمجلدات السكريبتات"""
    text = str(ctx.resolve(step["input"]))
    prefix = step.get("marker_prefix", "SCRIPT")

    # مسار الفيديوهات — افتراضي: /app/data/channels/{channel}/videos
    videos_dir = step.get("videos_dir")
    if not videos_dir:
        videos_dir = f"/app/data/channels/{ctx.channel_name}/videos"

    if not os.path.exists(videos_dir):
        raise EngineError(f"مجلد الفيديوهات غير موجود: {videos_dir}", code="DIR_NOT_FOUND")

    # تقسيم النص بالماركرز <<<SCRIPT_N>>>
    parts = re.split(rf'<<<{prefix}_(\d+)>>>', text)
    # parts[0] = نص قبل أول ماركر (فاضي عادةً)
    # parts[1] = رقم السكريبت, parts[2] = نص السكريبت, ...

    if len(parts) < 3:
        log(f"  [!] لم يتم العثور على ماركرز <<<{prefix}_N>>> في النص")
        raise EngineError(
            f"مخرجات AI لا تحتوي على ماركرز <<<{prefix}_N>>>",
            code="NO_MARKERS_FOUND"
        )

    copied_count = 0
    script_count = 0

    for i in range(1, len(parts), 2):
        script_num = parts[i]
        script_text = parts[i + 1] if i + 1 < len(parts) else ""
        script_text = script_text.replace(f"<<<END_{prefix}>>>", "").strip()

        if not script_text:
            continue

        script_count += 1

        # استخراج أرقام الفيديوهات: (رقم) (مطابق/تقريبي) — يقبل مسافات داخل وخارج الأقواس
        video_matches = re.findall(r'\(\s*(\d+)\s*\)\s*\(\s*(مطابق|تقريبي)\s*\)', script_text)

        if not video_matches:
            # fallback: استخراج أسماء الفيديوهات والبحث عنها في videos_list.xlsx
            excel_file = step.get("excel_file")
            if not excel_file:
                excel_file = os.path.join(ctx.input_dir, "videos_list.xlsx")
            if os.path.exists(excel_file):
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(excel_file, read_only=True)
                    ws = wb.active
                    title_to_num = {}
                    for row in ws.iter_rows(values_only=True):
                        if row[0] and row[1] and str(row[0]).isdigit():
                            title_to_num[str(row[1]).strip()] = str(row[0])
                    wb.close()
                    # استخراج أسماء الفيديوهات: (N) عنوان الفيديو (مطابق/تقريبي)
                    name_matches = re.findall(r'\(?\d+\)\s+(.+?)\s*\(\s*(مطابق|تقريبي)\s*\)', script_text)
                    for title, match_type in name_matches:
                        title = title.strip()
                        vid_num = title_to_num.get(title)
                        if vid_num:
                            video_matches.append((vid_num, match_type))
                        else:
                            log(f"  [!] {prefix}_{script_num}: مش لاقي رقم لـ '{title[:40]}'")
                except Exception as e:
                    log(f"  [!] {prefix}_{script_num}: فشل قراءة xlsx: {e}")
            if not video_matches:
                log(f"  [!] {prefix}_{script_num}: لم يتم العثور على أرقام فيديوهات")
                continue

        # إنشاء مجلد للسكريبت — حذف الفيديوهات القديمة أولاً لتجنب التراكم
        script_folder = os.path.join(ctx.output_dir, f"{prefix}_{script_num}")
        if os.path.exists(script_folder):
            for old_file in os.listdir(script_folder):
                if old_file.endswith(".mp4"):
                    os.remove(os.path.join(script_folder, old_file))
        os.makedirs(script_folder, exist_ok=True)

        for vid_num, match_type in video_matches:
            src = os.path.join(videos_dir, f"{vid_num}.mp4")
            dst = os.path.join(script_folder, f"{vid_num}.mp4")

            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_count += 1
                log(f"  {prefix}_{script_num}: نسخ فيديو {vid_num}.mp4 ({match_type})")
            else:
                log(f"  [!] {prefix}_{script_num}: فيديو {vid_num}.mp4 غير موجود في {videos_dir}")

    log(f"  تم نسخ {copied_count} فيديو لـ {script_count} سكريبت")
    return f"تم نسخ {copied_count} فيديو لـ {script_count} سكريبت"


def _extract_screen_phrase(text, section_name, extract_label, extract_end):
    """استخراج نص بين extract_label و extract_end داخل قسم محدد
    - extract_label: بداية النص المطلوب (مثلاً "جملة الصورة المصغرة للفيديو الأول:")
    - extract_end: نهاية النص المطلوب (مثلاً "جملة الصورة المصغرة للفيديو الثاني:")
    """
    section_start = text.find(section_name)
    if section_start == -1:
        return None

    next_section = text.find("القسم الثاني", section_start)
    if next_section == -1:
        section_text = text[section_start:]
    else:
        section_text = text[section_start:next_section]

    marker_pos = section_text.find(extract_label)
    if marker_pos == -1:
        return None

    after_marker = section_text[marker_pos + len(extract_label):]

    # قص النص عند extract_end
    end_pos = after_marker.find(extract_end)
    if end_pos != -1:
        raw = after_marker[:end_pos]
    else:
        raw = after_marker

    # تنظيف: شيل markdown وسطور فاضية
    cleaned = re.sub(r'\*+', '', raw).strip()
    return cleaned if cleaned else None


def action_extract_screen_text(step, ctx):
    """استخراج جمل الشاشة من مخرج وصفة إنشاء تكست قصير وحفظها في ملف Word
    يدعم وضعين:
    1. input من خطوة سابقة (نص بماركرز MG Ranner)
    2. قراءة من ملف docx (الوضع القديم)
    """
    from docx import Document

    raw_input = ctx.resolve(step.get("input", "")) if step.get("input") else ""
    source_file = step.get("file")
    section_name = step.get("section", "القسم الأول")
    save_as = step.get("save_as", "screen_texts.docx")
    extract_label = step.get("extract_label", "جملة الشاشة:")
    extract_end = step.get("extract_end", "الكلمات المفتاحية:")

    results = []

    if raw_input:
        # === وضع جديد: نص مباشر بماركرز MG Ranner ===
        log(f"  استخراج [{extract_label}] → [{extract_end}] من نص مباشر ({len(raw_input)} حرف)")
        marker_pattern = re.compile(r'<<<SCRIPT_(\d+)>>>(.*?)<<<END_SCRIPT>>>', re.DOTALL)
        for m in marker_pattern.finditer(raw_input):
            script_num = m.group(1)
            script_text = m.group(2)
            screen_text = _extract_screen_phrase(script_text, section_name, extract_label, extract_end)
            if screen_text:
                results.append((script_num, screen_text))
            else:
                log(f"  [!] Script {script_num}: لم يتم العثور على [{extract_label}]")
    else:
        # === وضع قديم: قراءة من ملف docx ===
        if source_file:
            filepath = ctx.input_path(source_file)
        else:
            docx_files = sorted([f for f in os.listdir(ctx.input_dir) if f.endswith('.docx')])
            if not docx_files:
                raise EngineError("لا توجد ملفات Word في مجلد الإدخال", code="FILE_NOT_FOUND")
            filepath = ctx.input_path(docx_files[0])

        if not os.path.exists(filepath):
            raise EngineError(f"ملف غير موجود: {filepath}", code="FILE_NOT_FOUND")

        log(f"  قراءة: {filepath}")
        doc = Document(filepath)

        current_num = None
        current_text = ""

        for p in doc.paragraphs:
            is_heading = p.style.name.startswith("Heading")
            heading_match = re.match(r'Script\s+(\d+)', p.text) if is_heading else None

            if heading_match:
                if current_num is not None and current_text:
                    screen_text = _extract_screen_phrase(current_text, section_name, extract_label, extract_end)
                    if screen_text:
                        results.append((current_num, screen_text))
                    else:
                        log(f"  [!] Script {current_num}: لم يتم العثور على [{extract_label}]")
                current_num = heading_match.group(1)
                current_text = ""
            elif current_num is not None:
                current_text += p.text + "\n"

        if current_num is not None and current_text:
            screen_text = _extract_screen_phrase(current_text, section_name, extract_label, extract_end)
            if screen_text:
                results.append((current_num, screen_text))
            else:
                log(f"  [!] Script {current_num}: لم يتم العثور على جملة الشاشة")

    if not results:
        raise EngineError("لم يتم العثور على أي جمل شاشة", code="NO_SCREEN_TEXT_FOUND")

    # حفظ في ملف Word جديد بتنسيق MG Ranner
    out_doc = Document()
    for script_num, screen_text in results:
        out_doc.add_heading(f"Script {script_num}", level=2)
        out_doc.add_paragraph(screen_text)

    out_path = ctx.output_path(save_as)
    out_doc.save(out_path)

    log(f"  تم استخراج {len(results)} جملة شاشة → {save_as}")
    return f"تم استخراج {len(results)} جملة شاشة"


# ========== Montage Short ==========

FONT_PATH = "/usr/share/fonts/truetype/noto/NotoKufiArabic-Bold.ttf"
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
SECTION_HEIGHT = VIDEO_HEIGHT // 3  # 640
FPS = 30
TITLE_MARGIN = 60  # هامش من الجوانب


def _smart_wrap(text, font, draw, max_width):
    """تقسيم النص العربي لأسطر بناءً على عرض الشاشة"""
    words = text.split()
    lines = []
    current = ''
    for word in words:
        test = (current + ' ' + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font, direction='rtl')
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def _create_title_card(text, color="white", font_path=FONT_PATH):
    """رسم صورة عنوان: خلفية سوداء + نص عربي كبير متعدد الأسطر في المنتصف"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new('RGB', (VIDEO_WIDTH, SECTION_HEIGHT), 'black')
    draw = ImageDraw.Draw(img)

    fill_color = color if color in ('white', 'yellow') else 'white'
    max_w = VIDEO_WIDTH - TITLE_MARGIN * 2

    # تحجيم تلقائي: أكبر خط ممكن يملا الثلث العلوي (2-4 أسطر)
    for font_size in range(90, 30, -2):
        font = ImageFont.truetype(font_path, font_size)
        lines = _smart_wrap(text, font, draw, max_w)
        line_spacing = int(font_size * 0.4)
        total_h = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font, direction='rtl')
            total_h += bbox[3] - bbox[1]
        total_h += line_spacing * (len(lines) - 1)
        if total_h <= SECTION_HEIGHT * 0.75 and len(lines) <= 4:
            break

    # رسم الأسطر في المنتصف
    start_y = (SECTION_HEIGHT - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, direction='rtl')
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (VIDEO_WIDTH - tw) // 2
        draw.text((x, start_y), line, fill=fill_color, font=font, direction='rtl')
        start_y += th + line_spacing

    return img


def _calc_segment_durations_whisper(audio_path, segments, total_dur, language="ar"):
    """
    حساب مدد البنود باستخدام Whisper word timestamps.
    بدل ما نعمل TTS لكل بند، بنعمل Whisper call واحد على الصوت الأصلي
    ونوزع الكلمات على البنود بالتناسب.

    الخوارزمية:
    1. Whisper بيرجع كلمات مرتبة زمنياً (نفس ترتيب البنود)
    2. بنحسب عدد كلمات كل بند
    3. بنوزع كلمات Whisper على البنود بنسبة عدد كلماتها
    4. بنقيس المدة من أول كلمة لآخر كلمة في كل بند

    Returns: list of durations (float) لكل بند
    """
    import re

    n_segments = len(segments)
    if n_segments == 0:
        return []

    try:
        result = transcribe_with_timestamps(audio_path, language=language)
        if not result.success or not result.data:
            raise Exception("Whisper timestamps فشل")

        words = result.data  # [{"word": "...", "start": 0.0, "end": 0.5}, ...]

        if not words:
            raise Exception("Whisper ما رجعش كلمات")

        # --- عدد كلمات كل بند ---
        def count_words(text):
            cleaned = re.sub(r'[^\w\s]', '', text).strip()
            return max(len(cleaned.split()), 1)  # حد أدنى 1

        seg_word_counts = [count_words(seg.get('narration', '')) for seg in segments]
        total_seg_words = sum(seg_word_counts)

        # --- توزيع كلمات Whisper على البنود بالتناسب ---
        n_words = len(words)
        seg_durations = []
        cumulative = 0

        for i, wcount in enumerate(seg_word_counts):
            # حدود الكلمات لهذا البند
            start_idx = int(cumulative / total_seg_words * n_words)
            cumulative += wcount
            end_idx = int(cumulative / total_seg_words * n_words) - 1

            # ضمان حدود صالحة
            start_idx = min(start_idx, n_words - 1)
            end_idx = max(end_idx, start_idx)
            end_idx = min(end_idx, n_words - 1)

            # المدة = من بداية أول كلمة لنهاية آخر كلمة
            duration = words[end_idx]['end'] - words[start_idx]['start']
            seg_durations.append(max(duration, 0.1))  # حد أدنى 0.1 ثانية

        # --- تعديل المدد بنسبة عشان المجموع = total_dur ---
        dur_total = sum(seg_durations)
        if dur_total > 0:
            seg_durations = [d * (total_dur / dur_total) for d in seg_durations]
        else:
            seg_durations = [total_dur / n_segments] * n_segments

        log(f"  Whisper timestamps: {n_segments} بنود | {n_words} كلمة | مدد: {[f'{d:.1f}s' for d in seg_durations]}")
        return seg_durations

    except Exception as e:
        log(f"  [!] Whisper timestamps فشل ({e}) — fallback للتوزيع المتساوي")
        return [total_dur / n_segments] * n_segments


def _get_audio_duration(wav_path):
    """قياس مدة ملف صوت بالثواني عن طريق ffprobe"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", wav_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _prepare_broll_video(src_path, target_duration, temp_dir):
    """تجهيز فيديو توضيحي: resize + trim أو loop ليناسب المدة المطلوبة"""
    out_path = os.path.join(temp_dir, f"broll_{os.path.basename(src_path)}")

    # قياس مدة الفيديو الأصلي
    src_duration = _get_audio_duration(src_path)
    if src_duration <= 0:
        src_duration = target_duration

    if src_duration >= target_duration:
        # الفيديو أطول أو مساوي → trim
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-t", str(target_duration),
            "-vf", f"scale={VIDEO_WIDTH}:{SECTION_HEIGHT}:force_original_aspect_ratio=decrease,pad={VIDEO_WIDTH}:{SECTION_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black",
            "-an", "-r", str(FPS),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            out_path
        ]
    else:
        # الفيديو أقصر → loop
        loops_needed = int(target_duration / src_duration) + 1
        cmd = [
            "ffmpeg", "-y", "-stream_loop", str(loops_needed), "-i", src_path,
            "-t", str(target_duration),
            "-vf", f"scale={VIDEO_WIDTH}:{SECTION_HEIGHT}:force_original_aspect_ratio=decrease,pad={VIDEO_WIDTH}:{SECTION_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black",
            "-an", "-r", str(FPS),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            out_path
        ]

    subprocess.run(cmd, capture_output=True, timeout=120)
    return out_path if os.path.exists(out_path) else None


def _prepare_creator_section(creator_path, total_duration, temp_dir):
    """تجهيز قسم صانع المحتوى — صورة ثابتة أو مجلد صور (slideshow) أو فيديو"""
    out_path = os.path.join(temp_dir, "creator_section.mp4")

    # --- مجلد صور → slideshow ---
    if os.path.isdir(creator_path):
        img_exts = ('.jpg', '.jpeg', '.png', '.webp')
        images = sorted([
            os.path.join(creator_path, f) for f in os.listdir(creator_path)
            if f.lower().endswith(img_exts)
        ])
        if not images:
            log(f"  [!] مجلد creator فارغ — مفيش صور")
            return None

        n = len(images)
        dur_each = total_duration / n
        log(f"  creator slideshow: {n} صور × {dur_each:.1f}s")

        slide_parts = []
        for i, img in enumerate(images):
            part_path = os.path.join(temp_dir, f"creator_slide_{i}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-loop", "1", "-i", img,
                "-t", str(dur_each),
                "-vf", f"scale={VIDEO_WIDTH}:{SECTION_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{SECTION_HEIGHT}",
                "-r", str(FPS), "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                part_path
            ], capture_output=True, timeout=60)
            if os.path.exists(part_path):
                slide_parts.append(part_path)

        if not slide_parts:
            return None

        if len(slide_parts) == 1:
            return slide_parts[0]

        concat_list = os.path.join(temp_dir, "creator_slides.txt")
        with open(concat_list, "w") as f:
            for sp in slide_parts:
                f.write(f"file '{sp}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            out_path
        ], capture_output=True, timeout=120)
        return out_path if os.path.exists(out_path) else None

    # --- صورة واحدة ---
    if creator_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", creator_path,
            "-t", str(total_duration),
            "-vf", f"scale={VIDEO_WIDTH}:{SECTION_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{SECTION_HEIGHT}",
            "-r", str(FPS), "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            out_path
        ]
    else:
        # فيديو → resize + trim/loop
        src_dur = _get_audio_duration(creator_path)
        if src_dur >= total_duration:
            cmd = [
                "ffmpeg", "-y", "-i", creator_path,
                "-t", str(total_duration),
                "-vf", f"scale={VIDEO_WIDTH}:{SECTION_HEIGHT}:force_original_aspect_ratio=decrease,pad={VIDEO_WIDTH}:{SECTION_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black",
                "-an", "-r", str(FPS),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                out_path
            ]
        else:
            loops = int(total_duration / src_dur) + 1
            cmd = [
                "ffmpeg", "-y", "-stream_loop", str(loops), "-i", creator_path,
                "-t", str(total_duration),
                "-vf", f"scale={VIDEO_WIDTH}:{SECTION_HEIGHT}:force_original_aspect_ratio=decrease,pad={VIDEO_WIDTH}:{SECTION_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black",
                "-an", "-r", str(FPS),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                out_path
            ]

    subprocess.run(cmd, capture_output=True, timeout=180)
    return out_path if os.path.exists(out_path) else None


def _compose_final_video(title_video, broll_concat, creator_video, audio_path, output_path):
    """تركيب الفيديو النهائي: stack عمودي 3 أقسام + صوت"""
    cmd = [
        "ffmpeg", "-y",
        "-i", title_video,
        "-i", broll_concat,
        "-i", creator_video,
        "-i", audio_path,
        "-filter_complex",
        "[0:v][1:v][2:v]vstack=inputs=3[v]",
        "-map", "[v]", "-map", "3:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    return os.path.exists(output_path)


def _find_script_audio_in_dir(script_dir):
    """البحث عن ملف صوت أو تسجيل داخل مجلد سكريبت"""
    if not os.path.exists(script_dir):
        return None, None
    # أولاً: recording.mp4 (فيديو شخصي)
    rec = os.path.join(script_dir, "recording.mp4")
    if os.path.exists(rec):
        return "recording", rec
    # ثانياً: ملف صوت (audio.mp3/wav/m4a/ogg)
    for ext in ['mp3', 'wav', 'm4a', 'ogg', 'aac']:
        aud = os.path.join(script_dir, f"audio.{ext}")
        if os.path.exists(aud):
            return "audio", aud
    return None, None


def _build_folder_map(input_dir, output_dir, script_nums, prefix="SCRIPT"):
    """
    بناء خريطة: رقم سكريبت → مسار المجلد الفعلي.
    الأولوية:
    1. مجلد باسم SCRIPT_N (التسمية الرسمية)
    2. مجلد باسم N (رقم فقط)
    3. أي مجلدات تانية بالترتيب الأبجدي
    """
    folder_map = {}
    SYSTEM_FOLDERS = {'audio', 'creator', '__pycache__'}

    for snum in script_nums:
        # بحث بالاسم الرسمي أو بالرقم
        for search_dir in [output_dir, input_dir]:
            for name in [f"{prefix}_{snum}", snum]:
                candidate = os.path.join(search_dir, name)
                if os.path.isdir(candidate):
                    folder_map[snum] = candidate
                    break
            if snum in folder_map:
                break

    # لو فيه سكريبتات مش لاقيالها مجلد → نوزع المجلدات المتبقية بالترتيب
    unmapped = [s for s in script_nums if s not in folder_map]
    if unmapped:
        # نجمع المجلدات اللي مش متربطة
        used_dirs = set(folder_map.values())
        free_folders = []
        for search_dir in [input_dir, output_dir]:
            if not os.path.exists(search_dir):
                continue
            for name in sorted(os.listdir(search_dir)):
                full = os.path.join(search_dir, name)
                if os.path.isdir(full) and name.lower() not in SYSTEM_FOLDERS and full not in used_dirs:
                    free_folders.append(full)

        # إزالة التكرار (نفس الاسم في output و input)
        seen_names = set()
        unique_free = []
        for f in free_folders:
            bname = os.path.basename(f)
            if bname not in seen_names:
                seen_names.add(bname)
                unique_free.append(f)

        for i, snum in enumerate(unmapped):
            if i < len(unique_free):
                folder_map[snum] = unique_free[i]

    return folder_map


def action_montage_short(step, ctx):
    """
    مونتاج فيديو قصير — يركّب العنوان + BROLL + صانع المحتوى.

    Parameters:
    - input: نص قائمة الفيديوهات (بماركرز SCRIPT)
    - screen_texts: نص جمل الشاشة (بماركرز SCRIPT)
    - audio_mode: "auto" (افتراضي) أو "tts" أو "recording" أو "audio"
    - text_color: "white" (افتراضي) أو "yellow"

    الصوت لكل سكريبت (أوتوماتيك):
    - SCRIPT_N/recording.mp4 → صوت + فيديو شخصي
    - SCRIPT_N/audio.mp3     → صوت + creator.jpg من الـ root
    - audio/SCRIPT_N/seg_*.wav → TTS جاهز + creator.jpg
    """
    import tempfile

    video_list_text = str(ctx.resolve(step["input"]))
    screen_text_raw = str(ctx.resolve(step["screen_texts"]))
    audio_mode = step.get("audio_mode", "auto")
    text_color = step.get("text_color", "white")
    prefix = step.get("marker_prefix", "SCRIPT")

    # --- تحليل جمل الشاشة ---
    screen_map = {}
    st_parts = re.split(rf'<<<{prefix}_(\d+)>>>', screen_text_raw)
    for i in range(1, len(st_parts), 2):
        snum = st_parts[i]
        stxt = st_parts[i + 1].replace(f"<<<END_{prefix}>>>", "").strip() if i + 1 < len(st_parts) else ""
        screen_map[snum] = stxt
    log(f"  جمل الشاشة: {len(screen_map)} سكريبت")

    # --- تحليل قائمة الفيديوهات ---
    vl_parts = re.split(rf'<<<{prefix}_(\d+)>>>', video_list_text)
    if len(vl_parts) < 3:
        raise EngineError("لم يتم العثور على ماركرز في قائمة الفيديوهات", code="NO_MARKERS_FOUND")

    # --- البحث عن صانع المحتوى: مجلد صور (slideshow) أو صورة واحدة ---
    creator_img_path = None
    for search_dir in [ctx.output_dir, ctx.input_dir]:
        # أولاً: مجلد creator/ (slideshow)
        creator_dir = os.path.join(search_dir, "creator")
        if os.path.isdir(creator_dir):
            img_exts = ('.jpg', '.jpeg', '.png', '.webp')
            has_images = any(f.lower().endswith(img_exts) for f in os.listdir(creator_dir))
            if has_images:
                creator_img_path = creator_dir
                break
        # ثانياً: صورة واحدة creator.{ext}
        for ext in ['jpg', 'jpeg', 'png']:
            candidate = os.path.join(search_dir, f"creator.{ext}")
            if os.path.exists(candidate):
                creator_img_path = candidate
                break
        if creator_img_path:
            break

    search_dirs = [ctx.output_dir, ctx.input_dir]

    # --- بناء خريطة المجلدات: رقم سكريبت → مسار فعلي ---
    script_nums = []
    for j in range(1, len(vl_parts), 2):
        script_nums.append(vl_parts[j])
    folder_map = _build_folder_map(ctx.input_dir, ctx.output_dir, script_nums, prefix)
    for sn, fp in folder_map.items():
        log(f"  📁 {prefix}_{sn} → {os.path.basename(fp)}/")

    # --- تجميع مهام الفيديوهات ---
    video_tasks = []
    for i in range(1, len(vl_parts), 2):
        script_num = vl_parts[i]
        script_text = vl_parts[i + 1].replace(f"<<<END_{prefix}>>>", "").strip() if i + 1 < len(vl_parts) else ""
        if not script_text:
            continue
        if ctx.topic_ids and int(script_num) not in ctx.topic_ids:
            continue
        video_tasks.append((script_num, script_text))

    # --- حساب عدد الـ workers بناءً على CPU ---
    # كل worker بيشغل ffmpeg بالتتابع (مش بالتوازي)
    # يعني N workers = N عمليات ffmpeg متزامنة كحد أقصى
    # القاعدة: نصف الأنوية الفيزيائية (مش logical threads)
    cpu_cores = os.cpu_count() or 4
    physical_cores = max(1, cpu_cores // 2)  # تقريب عدد الأنوية الفيزيائية
    max_workers_count = min(physical_cores, len(video_tasks), 4) if video_tasks else 1

    log(f"  📋 {len(video_tasks)} فيديو للمونتاج | {max_workers_count} workers | CPU: {cpu_cores} threads ({physical_cores} cores)")

    def _do_one_video(task):
        """معالجة فيديو واحد — thread-safe"""
        script_num, script_text = task
        script_dir = folder_map.get(script_num)
        script_folder = f"{prefix}_{script_num}"

        log(f"  === {prefix}_{script_num}: بدء المونتاج ===")
        if not script_dir:
            log(f"  [!] {prefix}_{script_num}: مجلد غير موجود — تخطي")
            return (script_num, False)

        segments = _parse_video_list_segments(script_text)
        if not segments:
            log(f"  [!] {prefix}_{script_num}: لا توجد بنود — تخطي")
            return (script_num, False)

        # --- 1. جملة الشاشة ---
        screen_text = screen_map.get(script_num, "")
        if not screen_text:
            log(f"  [!] {prefix}_{script_num}: جملة شاشة غير موجودة")
            screen_text = segments[0]['description'] if segments else "---"

        # --- 2. اكتشاف مصدر الصوت ---
        detected_mode, detected_path = _find_script_audio_in_dir(script_dir)
        effective_mode = audio_mode if audio_mode != "auto" else (detected_mode or "tts")

        seg_durations = []
        audio_files = []
        script_creator_path = creator_img_path
        full_audio = None

        if effective_mode == "recording":
            rec_path = detected_path
            if not rec_path:
                for sd in search_dirs:
                    c = os.path.join(sd, "recording.mp4")
                    if os.path.exists(c):
                        rec_path = c
                        break
            if not rec_path:
                log(f"  [!] {prefix}_{script_num}: recording.mp4 غير موجود — تخطي")
                return (script_num, False)

            log(f"  {prefix}_{script_num}: وضع التسجيل الشخصي")
            total_dur = _get_audio_duration(rec_path)
            if total_dur <= 0:
                log(f"  [!] {prefix}_{script_num}: فشل قياس مدة التسجيل — تخطي")
                return (script_num, False)

            temp_audio = os.path.join(ctx.output_dir, f"_temp_audio_{script_num}.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", rec_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", temp_audio],
                capture_output=True, timeout=120
            )

            seg_durations = _calc_segment_durations_whisper(temp_audio, segments, total_dur)
            full_audio = temp_audio
            script_creator_path = rec_path

        elif effective_mode == "audio":
            aud_path = detected_path
            if not aud_path:
                log(f"  [!] {prefix}_{script_num}: ملف صوت غير موجود — تخطي")
                return (script_num, False)

            if not script_creator_path:
                log(f"  [!] {prefix}_{script_num}: creator.jpg غير موجود — تخطي")
                return (script_num, False)

            log(f"  {prefix}_{script_num}: وضع الصوت الخارجي ({os.path.basename(aud_path)})")
            total_dur = _get_audio_duration(aud_path)
            if total_dur <= 0:
                log(f"  [!] {prefix}_{script_num}: فشل قياس مدة الصوت — تخطي")
                return (script_num, False)

            if not aud_path.endswith('.wav'):
                temp_audio = os.path.join(ctx.output_dir, f"_temp_audio_{script_num}.wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", aud_path, "-acodec", "pcm_s16le",
                     "-ar", "16000", "-ac", "1", temp_audio],
                    capture_output=True, timeout=120
                )
                full_audio = temp_audio
            else:
                full_audio = aud_path

            whisper_audio = full_audio if full_audio else aud_path
            seg_durations = _calc_segment_durations_whisper(whisper_audio, segments, total_dur)

        elif effective_mode == "tts":
            if not script_creator_path:
                log(f"  [!] {prefix}_{script_num}: creator.jpg غير موجود — تخطي")
                return (script_num, False)

            # --- محاولة 1: ملفات صوت مقسمة (seg_01.wav, seg_02.wav) ---
            audio_dir = None
            for sd in search_dirs:
                candidate_dir = os.path.join(sd, "audio", script_folder)
                if os.path.exists(candidate_dir):
                    audio_dir = candidate_dir
                    break

            if audio_dir:
                log(f"  {prefix}_{script_num}: وضع TTS (ملفات مقسمة)")
                for seg in segments:
                    wav = os.path.join(audio_dir, f"seg_{seg['num']:02d}.wav")
                    if os.path.exists(wav):
                        dur = _get_audio_duration(wav)
                        seg_durations.append(dur)
                        audio_files.append(wav)
                        log(f"    بند {seg['num']}: {dur:.1f}s | فيديو {seg['video_num']}")
                    else:
                        log(f"  [!] بند {seg['num']}: ملف صوت غير موجود: {wav}")
                        seg_durations.append(3.0)
                        audio_files.append(None)
            else:
                # --- محاولة 2: ملف صوت واحد من tts_multi (SCRIPT_N.wav/mp3) ---
                single_audio = None
                for sd in search_dirs:
                    for ext in ['wav', 'mp3', 'm4a']:
                        candidate = os.path.join(sd, f"{prefix}_{script_num}.{ext}")
                        if os.path.exists(candidate):
                            single_audio = candidate
                            break
                    if single_audio:
                        break

                if not single_audio:
                    log(f"  [!] {prefix}_{script_num}: لا يوجد مصدر صوت TTS — تخطي")
                    return (script_num, False)

                log(f"  {prefix}_{script_num}: وضع TTS (ملف واحد + Whisper)")
                total_dur = _get_audio_duration(single_audio)
                if total_dur <= 0:
                    log(f"  [!] {prefix}_{script_num}: فشل قياس مدة الصوت — تخطي")
                    return (script_num, False)

                # تحويل لـ WAV لو مش WAV (Whisper محتاج WAV)
                if not single_audio.endswith('.wav'):
                    temp_audio = os.path.join(ctx.output_dir, f"_temp_audio_{script_num}.wav")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", single_audio, "-acodec", "pcm_s16le",
                         "-ar", "16000", "-ac", "1", temp_audio],
                        capture_output=True, timeout=120
                    )
                    full_audio = temp_audio
                else:
                    full_audio = single_audio

                seg_durations = _calc_segment_durations_whisper(full_audio, segments, total_dur)

        else:
            log(f"  [!] {prefix}_{script_num}: لا يوجد مصدر صوت — تخطي")
            return (script_num, False)

        total_duration = sum(seg_durations)
        if total_duration <= 0:
            log(f"  [!] {prefix}_{script_num}: مجموع المدد = 0 — تخطي")
            return (script_num, False)
        log(f"  {prefix}_{script_num}: {len(segments)} بنود | إجمالي {total_duration:.1f}s")

        # ملفات مؤقتة لتنظيفها لاحقاً (حتى لو حصل خطأ)
        _temp_files_to_clean = [
            os.path.join(ctx.output_dir, f"_temp_audio_{script_num}.wav"),
            os.path.join(ctx.output_dir, f"_temp_audio_{script_num}.mp3"),
        ]

        # --- 3. إنشاء الفيديو ---
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                title_img = _create_title_card(screen_text, text_color)
                title_png = os.path.join(temp_dir, "title.png")
                title_img.save(title_png)

                title_vid = os.path.join(temp_dir, "title.mp4")
                subprocess.run([
                    "ffmpeg", "-y", "-loop", "1", "-i", title_png,
                    "-t", str(total_duration),
                    "-vf", f"fps={FPS}", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    title_vid
                ], capture_output=True, timeout=60)

                # دالة مساعدة: إنشاء إطار أسود كـ fallback
                def _make_black_frame(dur, idx_num):
                    bp = os.path.join(temp_dir, f"black_{idx_num}.mp4")
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi",
                        "-i", f"color=black:s={VIDEO_WIDTH}x{SECTION_HEIGHT}:d={dur}:r={FPS}",
                        "-c:v", "libx264", "-preset", "fast", bp
                    ], capture_output=True, timeout=30)
                    return bp if os.path.exists(bp) else None

                broll_parts = []
                for idx, seg in enumerate(segments):
                    vid_num = seg['video_num']
                    seg_dur = seg_durations[idx] if idx < len(seg_durations) else 3.0

                    vid_path = None
                    candidate = os.path.join(script_dir, f"{vid_num}.mp4")
                    if os.path.exists(candidate):
                        vid_path = candidate

                    if vid_path:
                        prepared = _prepare_broll_video(vid_path, seg_dur, temp_dir)
                        if prepared:
                            broll_parts.append(prepared)
                        else:
                            log(f"  [!] فشل تجهيز BROLL {vid_num} — إنشاء إطار أسود بديل")
                            fallback = _make_black_frame(seg_dur, idx)
                            if fallback:
                                broll_parts.append(fallback)
                    else:
                        log(f"  [!] فيديو {vid_num}.mp4 غير موجود — إنشاء إطار أسود")
                        fallback = _make_black_frame(seg_dur, idx)
                        if fallback:
                            broll_parts.append(fallback)

                if not broll_parts:
                    log(f"  [!] {prefix}_{script_num}: لا توجد أي فيديوهات b-roll — تخطي")
                    return (script_num, False)

                broll_concat = os.path.join(temp_dir, "broll_all.mp4")
                if len(broll_parts) == 1:
                    # ملف واحد — مفيش حاجة لـ concat
                    broll_concat = broll_parts[0]
                else:
                    concat_list = os.path.join(temp_dir, "broll_list.txt")
                    with open(concat_list, "w") as f:
                        for bp in broll_parts:
                            f.write(f"file '{bp}'\n")
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_list,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        broll_concat
                    ], capture_output=True, timeout=120)

                creator_vid = _prepare_creator_section(script_creator_path, total_duration, temp_dir)
                if not creator_vid:
                    log(f"  [!] فشل تجهيز قسم صانع المحتوى")
                    return (script_num, False)

                if effective_mode == "tts" and audio_files:
                    full_audio = os.path.join(temp_dir, "full_audio.wav")
                    valid_audio = [a for a in audio_files if a and os.path.exists(a)]
                    if valid_audio:
                        audio_list = os.path.join(temp_dir, "audio_list.txt")
                        with open(audio_list, "w") as f:
                            for af in valid_audio:
                                f.write(f"file '{af}'\n")
                        subprocess.run([
                            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", audio_list, "-c", "copy", full_audio
                        ], capture_output=True, timeout=60)
                    else:
                        full_audio = None

                output_path = os.path.join(ctx.output_dir, f"SHORT_{script_num}.mp4")

                if full_audio and os.path.exists(full_audio):
                    success = _compose_final_video(title_vid, broll_concat, creator_vid, full_audio, output_path)
                else:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", title_vid, "-i", broll_concat, "-i", creator_vid,
                        "-filter_complex", "[0:v][1:v][2:v]vstack=inputs=3[v]",
                        "-map", "[v]",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        output_path
                    ], capture_output=True, timeout=600)
                    success = os.path.exists(output_path)

                if success:
                    file_size = os.path.getsize(output_path)
                    log(f"  ✅ SHORT_{script_num}.mp4 — {file_size / 1024 / 1024:.1f} MB")
                else:
                    log(f"  ❌ فشل إنشاء SHORT_{script_num}.mp4")

                return (script_num, success)

        except Exception as e:
            log(f"  ❌ {prefix}_{script_num}: خطأ — {str(e)[:200]}")
            return (script_num, False)
        finally:
            # تنظيف ملفات مؤقتة — يشتغل دايماً حتى لو حصل خطأ
            for tf in _temp_files_to_clean:
                try:
                    if os.path.exists(tf):
                        os.remove(tf)
                except OSError:
                    pass

    # --- تشغيل متوازي ---
    success_count = 0
    fail_count = 0

    if len(video_tasks) <= 1:
        # فيديو واحد أو أقل — تشغيل عادي بدون threading
        for task in video_tasks:
            sn, ok = _do_one_video(task)
            if ok:
                success_count += 1
            else:
                fail_count += 1
    else:
        with ThreadPoolExecutor(max_workers=max_workers_count) as executor:
            futures = {executor.submit(_do_one_video, task): task for task in video_tasks}
            for future in as_completed(futures):
                try:
                    sn, ok = future.result()
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    log(f"  ❌ خطأ في worker: {str(e)[:200]}")

    log(f"  === المونتاج: {success_count} نجح | {fail_count} فشل ===")

    if success_count == 0:
        raise EngineError("فشل مونتاج كل الفيديوهات", code="MONTAGE_ALL_FAILED")

    return f"تم مونتاج {success_count} فيديو قصير"


# ========== TTS Segments Helpers ==========

def _parse_video_list_segments(text):
    """تحليل قائمة الفيديوهات المرقمة واستخراج البنود"""
    segments = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # مطابقة: "N. وصف (رقم_فيديو) (مطابق/تقريبي)" أو "N- وصف..."
        match = re.match(r'(\d+)[.\-]\s*(.+?)\((\d+)\)\s*\((مطابق|تقريبي)\)', line)
        if match:
            seg_num = int(match.group(1))
            description = match.group(2).strip()
            video_num = match.group(3)
            match_type = match.group(4)
            # السطر/الأسطر التالية = نص البند (اللي بيتقال)
            narration_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                # توقف لو وصلنا لبند جديد أو سطر فاضي بعد نص
                if not next_line:
                    if narration_lines:
                        break
                    i += 1
                    continue
                if re.match(r'\d+[.\-]\s*', next_line) and re.search(r'\(\d+\)\s*\((مطابق|تقريبي)\)', next_line):
                    break
                # شيل النقطة من أول السطر لو موجودة
                clean = next_line.lstrip('.').strip()
                if clean:
                    narration_lines.append(clean)
                i += 1
            narration = " ".join(narration_lines).strip()
            segments.append({
                'num': seg_num,
                'description': description,
                'video_num': video_num,
                'match_type': match_type,
                'narration': narration,
            })
        else:
            i += 1
    return segments


def _normalize_arabic(text):
    """تنظيف نص عربي للمقارنة — حذف التشكيل وتوحيد المسافات"""
    # حذف التشكيل
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # توحيد الألف
    text = re.sub(r'[إأآا]', 'ا', text)
    # توحيد المسافات
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _calculate_text_similarity(text1, text2):
    """حساب نسبة التشابه بين نصين عربيين"""
    from difflib import SequenceMatcher
    t1 = _normalize_arabic(text1)
    t2 = _normalize_arabic(text2)
    return SequenceMatcher(None, t1, t2).ratio()


def action_tts_segments(step, ctx):
    """
    TTS لكل بند من قائمة الفيديوهات + تحقق إجباري بويسبر.
    المدخل: نص قائمة الفيديوهات (بماركرز SCRIPT)
    المخرج: ملفات WAV لكل بند + full.wav + verification.json لكل سكريبت
    """
    text = str(ctx.resolve(step["input"]))
    prefix = step.get("marker_prefix", "SCRIPT")
    max_chars = step.get("max_chars")
    min_match = step.get("min_match", 0.7)
    language = step.get("language", "ar")

    # تقسيم النص بالماركرز <<<PREFIX_N>>>
    parts = re.split(rf'<<<{prefix}_(\d+)>>>', text)

    if len(parts) < 3:
        raise EngineError(
            f"لم يتم العثور على ماركرز <<<{prefix}_N>>> في النص",
            code="NO_MARKERS_FOUND"
        )

    total_success = 0
    total_fail = 0
    total_warnings = 0

    for i in range(1, len(parts), 2):
        script_num = parts[i]
        script_text = parts[i + 1] if i + 1 < len(parts) else ""
        script_text = script_text.replace(f"<<<END_{prefix}>>>", "").strip()

        if not script_text:
            continue

        # فلترة بـ topic_ids لو موجودة
        if ctx.topic_ids and int(script_num) not in ctx.topic_ids:
            continue

        # تحليل البنود
        segments = _parse_video_list_segments(script_text)

        if not segments:
            log(f"  [!] {prefix}_{script_num}: لم يتم العثور على بنود مرقمة")
            continue

        # إنشاء مجلد صوت السكريبت
        script_folder = os.path.join(ctx.output_dir, "audio", f"{prefix}_{script_num}")
        os.makedirs(script_folder, exist_ok=True)

        log(f"  {prefix}_{script_num}: {len(segments)} بنود")

        script_report = []
        wav_files = []

        for seg in segments:
            seg_label = f"seg_{seg['num']:02d}"
            narration = seg['narration']

            if not narration:
                log(f"  [!] {prefix}_{script_num}/{seg_label}: نص فارغ — تخطي")
                continue

            if max_chars and len(narration) > max_chars:
                narration = narration[:max_chars]

            # --- 1. TTS ---
            log(f"  {prefix}_{script_num}/{seg_label}: TTS ({len(narration)} حرف)...")
            tts_result = tts(narration)
            if not tts_result.success:
                log(f"  [!] {seg_label}: فشل TTS — {tts_result.error}")
                total_fail += 1
                script_report.append({
                    'segment': seg['num'],
                    'video_num': seg['video_num'],
                    'status': 'tts_failed',
                    'error': tts_result.error,
                })
                continue

            wav_path = os.path.join(script_folder, f"{seg_label}.wav")
            with open(wav_path, "wb") as f:
                f.write(tts_result.data)
            wav_files.append(wav_path)

            # --- 2. Whisper verification (إجباري) ---
            log(f"  {prefix}_{script_num}/{seg_label}: تحقق ويسبر...")
            try:
                whisper_result = transcribe(wav_path, language=language)
            except Exception as e:
                log(f"  [!] {seg_label}: خطأ ويسبر — {str(e)[:200]}")
                total_success += 1
                script_report.append({
                    'segment': seg['num'],
                    'video_num': seg['video_num'],
                    'status': 'whisper_error',
                    'error': str(e)[:200],
                    'file': f"{seg_label}.wav",
                })
                continue

            if whisper_result.success:
                whisper_text = whisper_result.data
                similarity = _calculate_text_similarity(narration, whisper_text)
                match_pct = round(similarity * 100, 1)

                if similarity >= min_match:
                    status = "pass"
                    log(f"  ✅ {seg_label}: تطابق {match_pct}% | فيديو {seg['video_num']}")
                else:
                    status = "warning"
                    total_warnings += 1
                    log(f"  ⚠️ {seg_label}: تطابق {match_pct}% | فيديو {seg['video_num']}")
                    log(f"      الأصلي: {narration[:100]}")
                    log(f"      ويسبر:  {whisper_text[:100]}")

                total_success += 1
                script_report.append({
                    'segment': seg['num'],
                    'video_num': seg['video_num'],
                    'status': status,
                    'match_pct': match_pct,
                    'original_text': narration,
                    'whisper_text': whisper_text,
                    'file': f"{seg_label}.wav",
                })
            else:
                log(f"  [!] {seg_label}: فشل ويسبر — {whisper_result.error}")
                total_success += 1
                script_report.append({
                    'segment': seg['num'],
                    'video_num': seg['video_num'],
                    'status': 'whisper_failed',
                    'error': whisper_result.error,
                    'file': f"{seg_label}.wav",
                })

        # --- 3. دمج كل البنود في full.wav ---
        if wav_files:
            full_wav = os.path.join(script_folder, "full.wav")
            list_file = os.path.join(script_folder, "_concat.txt")
            with open(list_file, "w", encoding="utf-8") as f:
                for wf in wav_files:
                    f.write(f"file '{wf}'\n")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", list_file, "-c", "copy", full_wav],
                    capture_output=True, timeout=120
                )
                if os.path.exists(full_wav):
                    log(f"  {prefix}_{script_num}/full.wav: دمج {len(wav_files)} ملف")
                os.remove(list_file)
            except Exception as e:
                log(f"  [!] فشل دمج full.wav: {e}")

        # --- 4. حفظ تقرير التحقق ---
        report_path = os.path.join(script_folder, "verification.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(script_report, f, ensure_ascii=False, indent=2)
        log(f"  {prefix}_{script_num}: تقرير التحقق → verification.json")

    # ملخص
    log(f"  === TTS Segments: {total_success} نجح | {total_fail} فشل | {total_warnings} تحذيرات ===")

    if total_success == 0:
        raise EngineError("فشل TTS لكل البنود", code="TTS_SEGMENTS_ALL_FAILED")

    result_msg = f"تم تحويل {total_success} بند"
    if total_warnings > 0:
        result_msg += f" | {total_warnings} تحذيرات"
    return result_msg


# ========== Topic Filtering ==========

def _extract_items(data):
    """استخراج قائمة العناصر من أي فورمات JSON للمواضيع"""
    # فورمات 1: [{"id":1, "title":"..."}, ...]
    if isinstance(data, list):
        return data
    # فورمات 2: {"titles": [{"id":1, "title":"..."}, ...], "total_count": N}
    if isinstance(data, dict) and "titles" in data:
        return data["titles"]
    return None


def _filter_topics_data(data, topic_ids):
    """فلترة كائن JSON بـ topic_ids — يرجع القائمة المفلترة"""
    items = _extract_items(data)
    if items is None or not isinstance(items, list):
        return data
    if not items or not isinstance(items[0], dict) or "id" not in items[0]:
        return data

    filtered = [item for item in items if item.get("id") in topic_ids]
    log(f"  فلترة المواضيع: {len(filtered)} من {len(items)} (المطلوب: {sorted(topic_ids)})")

    if not filtered:
        raise EngineError(
            f"لم يتم العثور على أي مواضيع بالأرقام: {sorted(topic_ids)}",
            code="TOPICS_NOT_FOUND"
        )
    return filtered


def _filter_topics_by_ids(content, topic_ids):
    """فلترة نص JSON بـ topic_ids — يرجع نص مفلتر"""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content

    filtered = _filter_topics_data(data, topic_ids)
    return json.dumps(filtered, ensure_ascii=False, indent=2)


def _filter_combined_by_topic_ids(combined, topic_ids):
    """فلترة نص مجمّع (بماركرز SCRIPT/INTRO) حسب topic_ids.

    بيشيل أي بلوك مش موجود في topic_ids.
    مثال: لو topic_ids={204, 211} — بيرجع بس البلوكات اللي فيها SCRIPT_204 و SCRIPT_211.
    """
    # كشف كل البلوكات بالماركرز
    pattern = r'(<<<(?:SCRIPT|INTRO)_(\d+)>>>.*?<<<END_(?:SCRIPT|INTRO)>>>)'
    blocks = re.findall(pattern, combined, re.DOTALL)

    if not blocks:
        return combined  # مفيش ماركرز — رجّع النص كما هو

    filtered = []
    for block_text, block_id in blocks:
        if int(block_id) in topic_ids:
            filtered.append(block_text)

    original_count = len(blocks)
    filtered_count = len(filtered)
    log(f"  [filter] فلترة النتائج: {filtered_count} من {original_count} (TOPIC_IDS: {sorted(topic_ids)})")

    return "\n\n".join(filtered)


def action_remove_tashkeel(step, ctx):
    """إزالة التشكيل من النص باستخدام regex"""
    text = str(ctx.resolve(step["input"]))
    tashkeel_pattern = re.compile(r'[\u064B-\u065F]')
    cleaned = tashkeel_pattern.sub('', text)
    removed_count = len(text) - len(cleaned)
    log(f"  تم إزالة {removed_count} علامة تشكيل")
    return cleaned


def action_clean_text(step, ctx):
    """تنظيف نصوص التوليد من أخطاء الموديل الشائعة.
    يصلح: باج اَل، تشكيل آخر حرف، Tatweel، أقواس حول حروف،
    كلمات ممنوعة، إملاء غلط، حروف أجنبية.
    """
    text = str(ctx.resolve(step["input"]))
    original_len = len(text)
    fixes = {}

    # 1. إزالة Tatweel (U+0640) من داخل الكلمات
    tatweel_count = text.count('\u0640')
    if tatweel_count:
        text = text.replace('\u0640', '')
        fixes['tatweel'] = tatweel_count

    # 2. إصلاح باج اَل — ألف + فتحة + لام التعريف → ال عادية
    al_bug = len(re.findall(r'\u0627\u064E\u0644', text))
    if al_bug:
        text = re.sub(r'\u0627\u064E\u0644', '\u0627\u0644', text)
        fixes['al_bug'] = al_bug

    # 3. إزالة تشكيل الحرف الأخير من كل كلمة
    TASHKEEL = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670'
    last_letter_fixes = 0
    def _fix_last_letter(m):
        nonlocal last_letter_fixes
        word = m.group(0)
        # أزل كل التشكيل من آخر حرف (الحرف الأخير + أي تشكيل بعده)
        i = len(word) - 1
        while i >= 0 and word[i] in TASHKEEL:
            i -= 1
        if i < len(word) - 1 and i >= 0:
            last_letter_fixes += 1
            return word[:i+1]
        return word
    # كلمة عربية = حروف عربية + تشكيل
    text = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u064B-\u065F\u0670]+', _fix_last_letter, text)
    if last_letter_fixes:
        fixes['last_letter_tashkeel'] = last_letter_fixes

    # 4. إصلاح أقواس حول حروف مفردة داخل كلمات: يُعْتَبَ(ر) → يُعْتَبَر
    paren_fixes = len(re.findall(r'\([\u0600-\u06FF]\)', text))
    if paren_fixes:
        text = re.sub(r'\(([\u0600-\u06FF])\)', r'\1', text)
        fixes['paren_single_char'] = paren_fixes

    # 5. حذف أقواس عادية متبقية
    remaining_parens = text.count('(') + text.count(')')
    if remaining_parens:
        text = text.replace('(', '').replace(')', '')
        fixes['remaining_parens'] = remaining_parens

    # 6. كلمات ممنوعة
    forbidden_replacements = {
        'موت': 'تلف',
        'الموت': 'التلف',
        'يموت': 'يتلف',
        'تموت': 'تتلف',
        'يقتل': 'يتلف',
        'القتل': 'الإتلاف',
    }
    forbidden_count = 0
    for bad, good in forbidden_replacements.items():
        count = len(re.findall(r'\b' + re.escape(bad) + r'\b', text))
        if count:
            text = re.sub(r'\b' + re.escape(bad) + r'\b', good, text)
            forbidden_count += count
    if forbidden_count:
        fixes['forbidden_words'] = forbidden_count

    # 7. إصلاح إملاء
    spelling_fixes = 0
    # الكلي/الكلى → الكلا (بدون تشكيل بين الحروف)
    for wrong in ['الكلي', 'الكلى', 'الْكِلَي', 'الْكِلَى']:
        count = text.count(wrong)
        if count:
            text = text.replace(wrong, 'الْكِلَا' if 'ْ' in wrong else 'الكلا')
            spelling_fixes += count
    if spelling_fixes:
        fixes['spelling'] = spelling_fixes

    # 8. إزالة حروف أجنبية (عبرية/فارسية ياء)
    foreign_count = len(re.findall(r'[\u0590-\u05FF\u06CC]', text))
    if foreign_count:
        text = re.sub(r'[\u0590-\u05FF]', '', text)  # عبري
        text = text.replace('\u06CC', '\u064A')  # ياء فارسية → ياء عربية
        fixes['foreign_chars'] = foreign_count

    # 9. إزالة ZWNJ وحروف غير مرئية
    invisible_count = len(re.findall(r'[\u200C\u200D\u200E\u200F\uFEFF]', text))
    if invisible_count:
        text = re.sub(r'[\u200C\u200D\u200E\u200F\uFEFF]', '', text)
        fixes['invisible_chars'] = invisible_count

    # 10. استبدال سكون بديل U+06E1 بسكون عادي U+0652
    alt_sukun = text.count('\u06E1')
    if alt_sukun:
        text = text.replace('\u06E1', '\u0652')
        fixes['alt_sukun'] = alt_sukun

    total_fixes = sum(fixes.values())
    if fixes:
        details = ' | '.join(f"{k}:{v}" for k, v in fixes.items())
        log(f"  clean_text: {total_fixes} إصلاح ({details})")
    else:
        log(f"  clean_text: النص نظيف — لا يحتاج إصلاح")

    return text


def action_split_script(step, ctx):
    """تقسيم السكريبت إلى مقدمات ونصوص.
    الطريقة الأساسية: <<<PART_1>>> كفاصل (deterministic — مش بيعتمد على الـ AI).
    الطريقة الاحتياطية: كلمة "النصوص" كفاصل نصي.
    """
    text = str(ctx.resolve(step["input"]))
    part = step.get("part", "intros")  # "intros" or "texts"
    separator = step.get("separator", "النصوص")

    pattern = r'(<<<SCRIPT_(\d+)>>>)(.*?)(<<<END_SCRIPT>>>)'

    result_parts = []
    skipped = 0
    for match in re.finditer(pattern, text, re.DOTALL):
        marker_start = match.group(1)
        script_num = match.group(2)
        content = match.group(3)
        marker_end = match.group(4)

        section = None

        # الطريقة الأساسية: تقسيم بـ <<<PART_1>>> (الأدق)
        part1_match = re.search(r'<<<PART_1>>>', content)
        if part1_match:
            if part == "intros":
                section = content[:part1_match.start()].strip()
                # إزالة كلمة الفاصل من آخر المقدمة (لو موجودة)
                if section.endswith(separator):
                    section = section[:-len(separator)].strip()
            else:
                # النصوص = من <<<PART_1>>> لحد الآخر (بما فيها الماركرز)
                section = content[part1_match.start():].strip()
        else:
            # الطريقة الاحتياطية: كلمة الفاصل النصي
            if separator in content:
                sep_idx = content.index(separator)
                if part == "intros":
                    section = content[:sep_idx].strip()
                else:
                    section = content[sep_idx + len(separator):].strip()

        # الطريقة الثالثة: لو مفيش أي فاصل — أول فقرة = مقدمة، الباقي = نصوص
        if section is None:
            # نقسم على أول سطر فاضي (فقرة فاضية)
            paragraphs = re.split(r'\n\s*\n', content.strip(), maxsplit=1)
            if len(paragraphs) >= 2:
                if part == "intros":
                    section = paragraphs[0].strip()
                else:
                    section = paragraphs[1].strip()
                log(f"  [~] SCRIPT_{script_num}: fallback فقرات (مفيش PART_1 ولا separator)")
            else:
                # فقرة واحدة بس — للنصوص ناخدها كلها، للمقدمات ناخد أول جملة
                if part == "texts":
                    section = content.strip()
                    log(f"  [~] SCRIPT_{script_num}: فقرة واحدة — أخذت كنص كامل")
                else:
                    # أول جملة (لحد أول نقطة أو فاصلة منقوطة أو سطر جديد)
                    first_sentence = re.split(r'[.\u060C\u061B\n]', content.strip(), maxsplit=1)[0].strip()
                    if first_sentence:
                        section = first_sentence
                        log(f"  [~] SCRIPT_{script_num}: فقرة واحدة — أول جملة كمقدمة")
                    else:
                        log(f"  [!] SCRIPT_{script_num}: فقرة واحدة فاضية — تخطي المقدمة")
                        skipped += 1
                        continue

        # لو المقدمة فاضية — تخطي مع تحذير
        if part == "intros" and not section:
            log(f"  [!] SCRIPT_{script_num}: مقدمة فاضية — تخطي")
            skipped += 1
            continue

        result_parts.append(f"{marker_start}\n{section}\n{marker_end}")

    log(f"  split_script: استخرجت {len(result_parts)} بلوك ({part})" +
        (f" | تحذير: {skipped} تم تخطيهم" if skipped else ""))
    return "\n\n".join(result_parts)


def action_topics_to_markers(step, ctx):
    """تحويل topics.json إلى نص بتنسيق MG Ranner (<<<SCRIPT_N>>>...<<<END_SCRIPT>>>)
    المدخل: قيمة JSON (string أو list أو dict بفيه titles)
    المخرج: نص بماركرز جاهز لـ tts_multi أو أي أكشن تاني"""
    import json as _json
    raw = ctx.resolve(step["input"])
    prefix = step.get("marker_prefix", "SCRIPT")

    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except Exception:
            pass

    topics = []
    if isinstance(raw, dict) and "titles" in raw:
        topics = raw["titles"]
    elif isinstance(raw, list):
        topics = raw

    topics.sort(key=lambda x: int(x.get("id", 0)))

    parts = []
    for t in topics:
        tid = t.get("id", 0)
        title = t.get("title", "").strip()
        parts.append(f"<<<{prefix}_{tid}>>>\n{title}\n<<<END_{prefix}>>>")

    result = "\n\n".join(parts)
    log(f"  topics_to_markers: {len(topics)} موضوع → تنسيق MG Ranner")
    return result


def action_scripts_to_topics_json(step, ctx):
    """تحويل نص MG Ranner (<<<SCRIPT_N>>>...<<<END_SCRIPT>>>) إلى topics.json
    الناتج: [{id: N, title: "نص السكريبت"}, ...]
    يُستخدم لجعل مخرجات توليد_سكريبتات صالحة مباشرةً كمدخلات لوصفات الباتش."""
    text = str(ctx.resolve(step["input"]))
    prefix = step.get("marker_prefix", "SCRIPT")
    save_as = step.get("save_as", "topics.json")

    pattern = rf'<<<{prefix}_(\d+)>>>(.*?)<<<END_{prefix}>>>'
    topics = []
    for match in re.finditer(pattern, text, re.DOTALL):
        topic_id = int(match.group(1))
        content = match.group(2).strip()
        topics.append({"id": topic_id, "title": content})

    topics.sort(key=lambda x: x["id"])
    output = {"total_count": len(topics), "titles": topics}

    save_path = ctx.output_path(save_as)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"  scripts_to_topics_json: {len(topics)} موضوع → {save_path}")
    return save_path


# ========== Thumbnail Generation ==========

THUMBNAILS_CONFIG_PATH = "/app/data/thumbnails/thumbnails_config.json"
THUMBNAILS_TEMPLATES_DIR = "/app/data/thumbnails/templates"

def _generate_thumbnail_html(template_config, bg_path, texts):
    """Generate HTML for a single thumbnail."""
    import html as _html
    text_areas = template_config.get("text_areas", [])

    # Build CSS and divs for each text area
    text_css = ""
    text_divs = ""
    for i, area in enumerate(text_areas):
        cx, cy = area.get("center", [640, 360])
        w, h = area.get("size", [600, 150])
        angle = area.get("angle", 0)
        color = area.get("color", "#000000")
        left = cx - w / 2
        top = cy - h / 2
        text = _html.escape(texts[i]) if i < len(texts) else ""

        text_css += f"""
  .text-{i+1} {{
    left: {left}px; top: {top}px;
    width: {w}px; height: {h}px;
    transform: rotate({angle}deg);
    color: {color};
  }}
"""
        text_divs += f'    <div class="text-area text-{i+1}">{text}</div>\n'

    return f"""<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head><meta charset="UTF-8">
<style>
  @font-face {{
    font-family: 'ArialBold';
    src: url('file:///usr/share/fonts/truetype/arialbd.ttf');
    font-weight: bold;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ width: 1280px; height: 720px; overflow: hidden; }}
  .container {{
    position: relative; width: 1280px; height: 720px;
    background-image: url('file://{bg_path}');
    background-size: 1280px 720px; background-repeat: no-repeat;
  }}
  .text-area {{
    position: absolute; display: flex; align-items: center;
    justify-content: center; text-align: center;
    font-family: 'ArialBold', 'Arial', sans-serif; font-weight: bold;
    direction: rtl; line-height: 1.0; overflow: hidden;
    padding: 5px 20px; white-space: nowrap;
  }}
{text_css}
</style></head>
<body>
  <div class="container">
{text_divs}  </div>
  <script>
    function autoFit(el) {{
      let size = 120;
      el.style.fontSize = size + 'px';
      while ((el.scrollWidth > el.clientWidth || el.scrollHeight > el.clientHeight) && size > 20) {{
        size -= 2;
        el.style.fontSize = size + 'px';
      }}
    }}
    document.querySelectorAll('.text-area').forEach(autoFit);
  </script>
</body></html>"""


def action_draw_thumbnail(step, ctx):
    """رسم صور مصغرة (thumbnails) باستخدام Playwright — round-robin على 12 تمبليت"""
    from playwright.sync_api import sync_playwright
    from PIL import Image

    text_content = str(ctx.resolve(step["input"]))
    save_prefix = step.get("save_prefix", "thumbnail")

    # Load config
    config_path = step.get("config_path", THUMBNAILS_CONFIG_PATH)
    templates_dir = step.get("templates_dir", THUMBNAILS_TEMPLATES_DIR)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    template_ids = sorted(config.keys(), key=int)
    total_templates = len(template_ids)

    if total_templates == 0:
        log("  draw_thumbnail: لا توجد تمبليتات في ملف الإعدادات")
        return []

    # Parse texts into groups of 3 lines each
    # entries = list of (topic_id, [lines]) tuples
    entries = []

    # Method 1: MG Ranner markers <<<SCRIPT_N>>> or <<<THUMB_N>>>
    marker_pattern = re.compile(r'<<<(?:SCRIPT|THUMB)_(\d+)>>>(.*?)<<<END_(?:SCRIPT|THUMB)>>>', re.DOTALL)
    matches = marker_pattern.findall(text_content)

    if matches:
        for topic_id, m in matches:
            lines = [l.strip() for l in m.strip().split('\n') if l.strip()]
            if lines:
                entries.append((topic_id, lines))
    else:
        # Method 2: "Script NNNN" headers (from thumbnail_texts.docx)
        all_lines = [l.strip() for l in text_content.split('\n') if l.strip()]
        script_header = re.compile(r'^Script\s+(\d+)', re.IGNORECASE)
        current = []
        current_id = None
        for line in all_lines:
            m = script_header.match(line)
            if m:
                if current:
                    entries.append((current_id, current))
                current_id = m.group(1)
                current = []
            else:
                current.append(line)
        if current:
            entries.append((current_id, current))

    # Fallback: group every 3 non-empty lines (no topic ID)
    if not entries:
        all_lines = [l.strip() for l in text_content.split('\n') if l.strip()]
        for i in range(0, len(all_lines), 3):
            group = all_lines[i:i+3]
            if group:
                entries.append((str(i // 3 + 1), group))

    if not entries:
        log("  draw_thumbnail: لا توجد نصوص للمعالجة")
        return []

    log(f"  draw_thumbnail: {len(entries)} صورة مصغرة × {total_templates} تمبليت (round-robin)")

    output_paths = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for i, (topic_id, entry_texts) in enumerate(entries):
                tid = template_ids[i % total_templates]
                bg_path = os.path.join(templates_dir, f"{tid}.png")

                # Generate HTML
                html = _generate_thumbnail_html(config[tid], bg_path, entry_texts)
                temp_html = os.path.join(ctx.output_dir, f"_temp_{i}.html")
                raw_path = os.path.join(ctx.output_dir, f"_raw_{i}.png")
                page = None
                try:
                    with open(temp_html, "w", encoding="utf-8") as f:
                        f.write(html)

                    # Render
                    page = browser.new_page(
                        viewport={"width": 1280, "height": 720},
                        device_scale_factor=2
                    )
                    page.goto(f"file://{temp_html}")
                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(300)

                    # Screenshot at 2x then resize to 1280x720
                    page.screenshot(path=raw_path, type="png")
                    page.close()
                    page = None

                    # Resize to final 1280x720
                    img = None
                    try:
                        img = Image.open(raw_path)
                        if img.size != (1280, 720):
                            img = img.resize((1280, 720), Image.LANCZOS)
                        output_name = f"{save_prefix}_{topic_id}.png"
                        output_path = ctx.output_path(output_name)
                        img.save(output_path, format="PNG")
                    finally:
                        if img is not None:
                            img.close()

                    output_paths.append(output_path)
                    log(f"  [{i+1}/{len(entries)}] template {tid} → {output_name}")
                finally:
                    if page is not None:
                        page.close()
                    # Cleanup temp files safely
                    for tmp in (temp_html, raw_path):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
        finally:
            browser.close()

    log(f"  draw_thumbnail: تم إنشاء {len(output_paths)} صورة مصغرة")
    return output_paths


def action_filter_by_topics(step, ctx):
    """فلترة النص حسب TOPIC_IDS — يحتفظ فقط بالموضوعات المطلوبة.
    لو مفيش TOPIC_IDS محدد، يرجع النص كامل بدون تغيير.
    """
    text = str(ctx.resolve(step["input"]))

    if not ctx.topic_ids:
        log("  filter_by_topics: مفيش TOPIC_IDS — تمرير النص كامل")
        return text

    pattern = r'(<<<SCRIPT_(\d+)>>>.*?<<<END_SCRIPT>>>)'
    matches = list(re.finditer(pattern, text, re.DOTALL))

    if not matches:
        log("  filter_by_topics: مفيش ماركرز SCRIPT — تمرير النص كامل")
        return text

    filtered = []
    for match in matches:
        script_num = int(match.group(2))
        if script_num in ctx.topic_ids:
            filtered.append(match.group(1))

    log(f"  filter_by_topics: {len(filtered)} من {len(matches)} (TOPIC_IDS: {sorted(ctx.topic_ids)})")
    return "\n\n".join(filtered)


TATWEEL_CHAR = '\u0640'

MEDICAL_WORDS_FIX = {
    # نمط الكلمة بدون تشكيل → الكلمة بالتشكيل الصحيح
    'الكلا': 'الْكِلَا',
    'النفرونات': 'النِّفْرُونَات',
    'الكيسات': 'الْكِيسَات',
    'التكيسات': 'التَّكَيُّسَات',
    'الكرياتينين': 'الْكِرْيَاتِينِين',
    'الكلوية': 'الْكُلَوِيَّة',
}

ALLOWED_2_LETTERS_FIX = {
    'في', 'من', 'أو', 'لا', 'ما', 'هو', 'هي', 'أن', 'إن', 'قد', 'لم', 'لن',
    'بل', 'كل', 'بس', 'يا', 'عن', 'مع', 'بك', 'لك', 'له', 'لي', 'بي', 'ذا',
    'دا', 'ده', 'دي', 'إذ', 'أي', 'أى', 'أم', 'إى', 'به', 'بها', 'لها', 'لهم',
    'فى', 'كم', 'كي', 'هل', 'لو',
}

END_OF_WORD_LETTERS_FIX = set('اىةيوءأؤإ')
INNER_DIACRITICS_FIX = '\u064E\u064F\u0650'  # فتحة، ضمة، كسرة


def _fix_remove_tatweel(text):
    """يحذف كل أحرف التطويل ـ من النص بدون التأثير على الماركرز."""
    return text.replace(TATWEEL_CHAR, '')


def _looks_like_split_word_fix(before, after, tashkeel_chars):
    """يحدد إذا كانت 'before after' كلمة واحدة مقطوعة بمسافة.

    معايير الكشف (أيٌّ منها يكفي):
    1. الجزء الثاني 1 حرف من حروف نهاية الكلمة (ء/ة/ى)
    2. الجزء الثاني 2 حرف ينتهي بحرف نهاية كلمة (ء/ة/ى)
    3. الجزء الأول ينتهي بحرف داخلي + حركة، والجزء الثاني بدون تشكيل تقريباً
       — مثل: 'تَنْظِ يف' → 'تنظيف'
    """
    clean_after = re.sub(f'[{tashkeel_chars}]', '', after).strip()
    clean_before = re.sub(f'[{tashkeel_chars}]', '', before).strip()

    if clean_after in ALLOWED_2_LETTERS_FIX:
        return False
    if len(clean_before) > 12:
        return False
    if len(clean_after) == 1:
        return clean_after in 'ءةى'
    if len(clean_after) != 2:
        return False
    if clean_after[-1] in 'ءةى':
        return True

    # قاعدة 3: حرف داخلي + حركة + جزء ثاني بتشكيل منخفض
    last_arabic_match = re.search(r'([\u0621-\u064A])([\u064B-\u0652\u0670]*)$', before)
    if not last_arabic_match:
        return False
    last_letter = last_arabic_match.group(1)
    trailing_diacritics = last_arabic_match.group(2)
    if last_letter in END_OF_WORD_LETTERS_FIX:
        return False
    has_inner_diacritic = any(d in trailing_diacritics for d in INNER_DIACRITICS_FIX)
    after_clean = after.strip()
    after_diacritics = sum(1 for c in after_clean if c in tashkeel_chars)
    after_letters = sum(1 for c in after_clean if '\u0621' <= c <= '\u064A')
    low_tashkeel_ratio = (after_diacritics / after_letters) < 0.5 if after_letters else True
    return has_inner_diacritic and low_tashkeel_ratio


def _fix_mid_word_spaces(text, tashkeel_chars):
    """يصلح المسافات داخل الكلمات العربية (مثل 'الْأَطِبَّا ء' → 'الْأَطِبَّاء').

    يرجع: (النص المصلح، عدد الإصلاحات)
    """
    fixes = 0

    def replace_match(m):
        nonlocal fixes
        before = m.group(1)
        after = m.group(2)
        if _looks_like_split_word_fix(before, after, tashkeel_chars):
            fixes += 1
            return before + after
        return m.group(0)

    pattern = r'([\u0621-\u064A\u0670' + re.escape(tashkeel_chars) + r']{4,})\s([\u0621-\u064A\u0670' + re.escape(tashkeel_chars) + r']{1,2})(?=\s|[.،؟!:؛]|$)'
    fixed = re.sub(pattern, replace_match, text)
    return fixed, fixes


def _fix_medical_words(text, tashkeel_chars):
    """يصحح الكلمات الطبية للتشكيل المحدد.

    لو الكلمة موجودة بدون تشكيل (أو بتشكيل خاطئ) ومش في صيغة المعجم → استبدال بالصحيحة.
    يرجع: (النص المصلح، عدد الإصلاحات)
    """
    fixes = 0
    for raw, expected in MEDICAL_WORDS_FIX.items():
        # نبحث عن كل ظهور للكلمة بدون تشكيل
        # نمط: كلمة عربية يكون نواتها = raw بعد إزالة التشكيل
        # علشان ما نلمسش كلمات أكبر (مثلاً: الكلوية لو فيها كلوية)

        def check_and_fix(m):
            nonlocal fixes
            word = m.group(0)
            # شيل التشكيل وقارن
            clean_word = re.sub(f'[{tashkeel_chars}]', '', word).strip()
            if clean_word == raw and word != expected:
                fixes += 1
                return expected
            return word

        # نلتقط الكلمة العربية كاملة (مع تشكيلها)
        word_pattern = r'[\u0621-\u064A\u0670' + re.escape(tashkeel_chars) + r']+'
        # فلتر مبدئي: الكلمة لازم تحتوي على raw (بدون تشكيل)
        text = re.sub(word_pattern, check_and_fix, text)

    return text, fixes


def action_auto_fix_text(step, ctx):
    """إصلاح ميكانيكي للنصوص العربية المشكّلة قبل الـ validation.

    يصلح أوتوماتيكياً (بدون AI):
    - حذف كل أحرف Tatweel ـ (U+0640)
    - دمج المسافات داخل الكلمات (مثل 'الْأَطِبَّا ء' → 'الْأَطِبَّاء')
    - تصحيح الكلمات الطبية بالتشكيل المحدد
    """
    text = str(ctx.resolve(step["input"]))
    tashkeel_chars = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670'

    # 1) Tatweel
    before_tatweel = text.count(TATWEEL_CHAR)
    text = _fix_remove_tatweel(text)

    # 2) مسافات وسط الكلمة
    text, mid_space_fixes = _fix_mid_word_spaces(text, tashkeel_chars)

    # 3) كلمات طبية
    text, medical_fixes = _fix_medical_words(text, tashkeel_chars)

    total = before_tatweel + mid_space_fixes + medical_fixes
    if total > 0:
        log(f"  auto_fix_text: {before_tatweel} Tatweel | {mid_space_fixes} مسافة وسط كلمة | {medical_fixes} كلمة طبية | إجمالي: {total} إصلاح")
    else:
        log(f"  auto_fix_text: ✓ النص نظيف — مفيش إصلاحات")

    return text


def action_strip_last_letter_diacritic(step, ctx):
    """يزيل الحركة فوق الحرف الأخير من كل كلمة عربية في النص.

    الحركة = فتحة/كسرة/ضمة/سكون/تنوين فتح/تنوين كسر/تنوين ضم.
    الشدة (U+0651) تبقى — لأنها جزء صرفي من الحرف.
    الحرف نفسه يبقى موجوداً.

    ملاحظة: في يونيكود، الشدة قد تأتي قبل أو بعد الحركة. الكود يلتقط
    كل الحركات بعد آخر حرف عربي ويزيلها (مع الحفاظ على الشدة).
    """
    text = str(ctx.resolve(step["input"]))
    SHADDA = 'ّ'
    HARAKAT = 'ًٌٍَُِْ'  # كل أنواع الحركات (بدون شدة)

    fixes = 0

    def _strip(m):
        nonlocal fixes
        word = m.group(0)
        # نلاقي آخر حرف عربي (غير حركة، غير شدة، غير ألف خنجرية)
        last_letter_idx = -1
        for i in range(len(word) - 1, -1, -1):
            ch = word[i]
            if 'ء' <= ch <= 'ي':
                last_letter_idx = i
                break
        if last_letter_idx == -1:
            return word

        prefix = word[:last_letter_idx + 1]
        suffix = word[last_letter_idx + 1:]
        # نسيب الشدة، نشيل الحركات
        new_suffix = ''.join(ch for ch in suffix if ch == SHADDA or ch == 'ٰ')
        if suffix != new_suffix:
            fixes += 1
        return prefix + new_suffix

    pattern = r'[ء-يً-ْٰ]+'
    text = re.sub(pattern, _strip, text)

    if fixes > 0:
        log(f"  strip_last_letter_diacritic: ✓ {fixes} كلمة تم تنظيف حركة آخر حرف")
    else:
        log(f"  strip_last_letter_diacritic: ✓ مفيش حركات على آخر حرف")

    return text


def action_restore_truncated_words(step, ctx):
    """يصطاد الكلمات المبتورة (محذوف منها حرف من النموذج) ويصلحها.

    المنطق:
    - يقارن المخرج (المُشكَّل من النموذج) بالمدخل الأصلي (بدون تشكيل)
    - يطابق كلمة بكلمة داخل كل PART
    - لو كلمة المخرج (بعد إزالة التشكيل) أقصر من كلمة المدخل
      → يستبدلها بكلمة المدخل (بدون تشكيل، مع الحفاظ على علامات الترقيم)
    """
    output_text = str(ctx.resolve(step["input"]))
    input_text = str(ctx.resolve(step["original"]))

    TASHKEEL = 'ًٌٍَُِّْٰ'
    PUNCT = '.،؟!:;؛'

    def strip_t(s):
        return re.sub(f'[{TASHKEEL}]', '', s)

    fixes = 0

    script_pat = r'(<<<SCRIPT_(\d+)>>>)(.*?)(<<<END_SCRIPT>>>)'
    input_scripts = {}
    for m in re.finditer(script_pat, input_text, re.DOTALL):
        input_scripts[m.group(2)] = m.group(3)

    def fix_part(input_part_text, output_part_text):
        nonlocal fixes
        input_words = input_part_text.split()
        output_words = output_part_text.split()

        if len(input_words) != len(output_words):
            return output_part_text

        fixed = []
        for inp_w, out_w in zip(input_words, output_words):
            inp_clean = strip_t(inp_w).strip(PUNCT)
            out_clean = strip_t(out_w).strip(PUNCT)

            if 0 < len(out_clean) < len(inp_clean) and inp_clean.startswith(out_clean):
                trailing_punct = ''
                for ch in reversed(out_w):
                    if ch in PUNCT:
                        trailing_punct = ch + trailing_punct
                    else:
                        break
                fixes += 1
                fixed.append(inp_w.rstrip(PUNCT) + trailing_punct)
            else:
                fixed.append(out_w)

        return ' '.join(fixed)

    def fix_script(m):
        script_id = m.group(2)
        output_script_content = m.group(3)
        if script_id not in input_scripts:
            return m.group(0)

        input_script_content = input_scripts[script_id]
        part_pat = r'(<<<PART_(\d+)>>>)(.*?)(<<<END_PART>>>)'
        input_parts = {}
        for pm in re.finditer(part_pat, input_script_content, re.DOTALL):
            input_parts[pm.group(2)] = pm.group(3).strip()

        def fix_part_match(pm):
            part_id = pm.group(2)
            output_part = pm.group(3).strip()
            if part_id not in input_parts:
                return pm.group(0)
            fixed_text = fix_part(input_parts[part_id], output_part)
            return f"{pm.group(1)}\n{fixed_text}\n{pm.group(4)}"

        new_content = re.sub(part_pat, fix_part_match, output_script_content, flags=re.DOTALL)
        return f"{m.group(1)}{new_content}{m.group(4)}"

    output_text = re.sub(script_pat, fix_script, output_text, flags=re.DOTALL)

    if fixes > 0:
        log(f"  restore_truncated_words: ✓ تم إصلاح {fixes} كلمة مبتورة")
    else:
        log(f"  restore_truncated_words: ✓ مفيش كلمات مبتورة")

    return output_text


def _check_brackets_pattern(part_text):
    """فحص وجود أقواس مربعة حول حروف عربية — مثل: التَّكَيُّسَا[ت]"""
    bracket_matches = re.findall(r'\[[^\[\]]{1,3}\]', part_text)
    arabic_brackets = [m for m in bracket_matches if re.search(r'[\u0621-\u064A]', m)]
    return arabic_brackets


def _check_tatweel_in_text(part_text):
    """فحص وجود حرف التطويل ـ في النص."""
    return part_text.count(TATWEEL_CHAR)


def _check_mid_word_spaces_count(part_text, tashkeel_chars):
    """عدّاد المسافات داخل الكلمات (يستخدم نفس logic _looks_like_split_word_fix)."""
    pattern = r'([\u0621-\u064A\u0670' + re.escape(tashkeel_chars) + r']{4,})\s([\u0621-\u064A\u0670' + re.escape(tashkeel_chars) + r']{1,2})(?=\s|[.،؟!:؛]|$)'
    count = 0
    for m in re.finditer(pattern, part_text):
        before = m.group(1)
        after = m.group(2)
        if _looks_like_split_word_fix(before, after, tashkeel_chars):
            count += 1
    return count


def _check_single_letters(part_text, tashkeel_chars):
    """فحص الحروف المنفردة — كل كلمة حرف عربي واحد بس (بعد شيل التشكيل)."""
    words = part_text.split()
    if len(words) < 5:
        return 0, 0
    single_count = 0
    for w in words:
        clean = re.sub(f'[{tashkeel_chars}]', '', w).strip()
        clean = re.sub(r'[.،؟!:\-]', '', clean)
        arabic_only = re.sub(r'[^\u0621-\u064A]', '', clean)
        if len(arabic_only) == 1:
            single_count += 1
    pct = (single_count / len(words)) * 100 if words else 0
    return single_count, pct


def _check_truncated_words(part_text, tashkeel_chars):
    """فحص الكلمات المبتورة — كلمات من حرفين عربيين أو أقل (بعد شيل التشكيل).
    نسبة عالية = مشكلة (طبيعي 5-10%، مشكلة > 20%).
    """
    words = part_text.split()
    if len(words) < 5:
        return 0, 0
    short_count = 0
    COMMON_SHORT = {'في', 'من', 'أو', 'لا', 'ما', 'هو', 'هي', 'أن', 'إن', 'قد', 'لم', 'لن', 'بل', 'كل', 'بس', 'يا', 'عن', 'مع', 'بك', 'لك', 'له', 'لي', 'بي', 'ذا', 'دا', 'ده', 'دي'}
    for w in words:
        clean = re.sub(f'[{tashkeel_chars}]', '', w).strip()
        clean = re.sub(r'[.،؟!:\-\[\]]', '', clean)
        arabic_only = re.sub(r'[^\u0621-\u064A]', '', clean)
        if 0 < len(arabic_only) <= 2 and clean not in COMMON_SHORT:
            short_count += 1
    pct = (short_count / len(words)) * 100 if words else 0
    return short_count, pct


def _check_split_hamza(part_text):
    """فحص الهمزة المنفصلة — مثل: إ لَى بدل إِلَى، أ نَّ بدل أَنَّ."""
    split_matches = re.findall(r'(?:^|\s)([إأ])\s+(\S+)', part_text)
    return split_matches


def _check_repeated_words(part_text, tashkeel_chars):
    """فحص تكرار كلمات متتالية — مثل: الكلى الكلى الكلى (AI stuttering)."""
    words = part_text.split()
    if len(words) < 5:
        return 0
    repeated = 0
    for i in range(1, len(words)):
        w_prev = re.sub(f'[{tashkeel_chars}]', '', words[i-1]).strip().rstrip('.،؟!')
        w_curr = re.sub(f'[{tashkeel_chars}]', '', words[i]).strip().rstrip('.،؟!')
        if w_prev and w_curr and w_prev == w_curr and len(w_prev) > 2:
            repeated += 1
    return repeated


def _check_latin_chars(part_text):
    """فحص وجود حروف لاتينية في النص العربي."""
    latin_matches = re.findall(r'[a-zA-Z]{2,}', part_text)
    # استثناء: كلمات علمية شائعة مسموح بيها
    ALLOWED_LATIN = {'pH', 'mg', 'ml', 'kg', 'cm', 'mm', 'DNA', 'RNA', 'BMI', 'GFR', 'HDL', 'LDL', 'II', 'III', 'IV'}
    filtered = [m for m in latin_matches if m not in ALLOWED_LATIN]
    return filtered


def _check_markdown_artifacts(part_text):
    """فحص بقايا تنسيق Markdown — **bold**, ##heading, -list, *italic*."""
    artifacts = []
    if re.search(r'\*\*[^*]+\*\*', part_text):
        artifacts.append('**bold**')
    if re.search(r'^#{1,3}\s', part_text, re.MULTILINE):
        artifacts.append('## heading')
    if re.search(r'^\s*[-*]\s+\S', part_text, re.MULTILINE):
        artifacts.append('- list')
    if re.search(r'(?<!\*)\*(?!\*)[^*\s][^*]*[^*\s]\*(?!\*)', part_text):
        artifacts.append('*italic*')
    if re.search(r'`[^`]+`', part_text):
        artifacts.append('`code`')
    return artifacts


def _check_missing_tashkeel(part_text, tashkeel_chars):
    """فحص غياب التشكيل — لو النص المفروض مشكّل بس مفيهوش تشكيل."""
    arabic_chars = re.findall(r'[\u0621-\u064A]', part_text)
    tashkeel_found = re.findall(f'[{tashkeel_chars}]', part_text)
    if len(arabic_chars) < 20:
        return 0.0
    ratio = len(tashkeel_found) / len(arabic_chars) * 100
    return ratio


def _check_repeated_phrases(part_text, tashkeel_chars):
    """فحص تكرار جمل أو عبارات كاملة (3+ كلمات متتالية مكررة)."""
    words = part_text.split()
    if len(words) < 15:
        return 0
    # فحص trigrams (3 كلمات)
    clean_words = [re.sub(f'[{tashkeel_chars}]', '', w).strip().rstrip('.،؟!') for w in words]
    trigrams = {}
    repeated = 0
    for i in range(len(clean_words) - 2):
        tri = ' '.join(clean_words[i:i+3])
        if len(tri) > 6:  # تجاهل trigrams قصيرة جداً
            trigrams[tri] = trigrams.get(tri, 0) + 1
    for tri, count in trigrams.items():
        if count >= 3:  # نفس العبارة 3 مرات أو أكثر
            repeated += count - 1
    return repeated


def _check_digit_numbers(part_text):
    """فحص وجود أرقام (digits) في النص — المفروض تكون مكتوبة بالحروف."""
    # استثناء: أرقام في سياق طبي/علمي مقبولة (pH 6.5, 500mg)
    # بس أرقام كبيرة لوحدها غير مقبولة
    digit_matches = re.findall(r'(?<!\S)\d{3,}(?!\S)', part_text)  # 3+ أرقام متتالية
    return digit_matches


def _check_unclosed_parens(part_text):
    """فحص أقواس غير مغلقة."""
    open_paren = part_text.count('(') - part_text.count(')')
    open_bracket = part_text.count('[') - part_text.count(']')
    return abs(open_paren) + abs(open_bracket)


def action_validate_texts(step, ctx):
    """مراجعة النصوص والتحقق من سلامتها.
    يفحص: عدد الأجزاء، عدد الكلمات، أقواس مربعة، حروف منفردة، كلمات مبتورة، همزة منفصلة، علامة ترقيم.
    يصلح: دمج أسطر، إضافة نقطة في الآخر.
    يرجع: النص المصلح + قائمة الموضوعات الفاشلة في ctx.results[step_id + '_failed']
    """
    text = str(ctx.resolve(step["input"]))
    min_words = step.get("min_words", 70)
    max_words = step.get("max_words", 130)
    expected_parts = step.get("expected_parts", 4)
    # حدود الفحوصات (قابلة للتعديل من الوصفة)
    bracket_threshold = step.get("bracket_threshold", 3)          # أكتر من 3 أقواس = فاشل
    single_letter_pct = step.get("single_letter_pct", 15)         # أكتر من 15% حروف منفردة = فاشل
    truncated_pct = step.get("truncated_pct", 20)                 # أكتر من 20% كلمات مبتورة = فاشل
    split_hamza_threshold = step.get("split_hamza_threshold", 3)   # أكتر من 3 همزات منفصلة = فاشل
    repeated_words_threshold = step.get("repeated_words_threshold", 3)  # أكتر من 3 تكرارات = فاشل
    latin_threshold = step.get("latin_threshold", 2)               # أكتر من 2 كلمات لاتينية = فاشل
    min_tashkeel_pct = step.get("min_tashkeel_pct", 15)            # أقل من 15% تشكيل = فاشل
    repeated_phrases_threshold = step.get("repeated_phrases_threshold", 3)  # تكرار عبارات = فاشل

    TASHKEEL = '\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670'
    pattern = r'(<<<SCRIPT_(\d+)>>>)(.*?)(<<<END_SCRIPT>>>)'

    failed_scripts = []  # قائمة أرقام الموضوعات الفاشلة
    fixed_blocks = []
    total_fixes = 0
    total_issues = 0

    for match in re.finditer(pattern, text, re.DOTALL):
        marker_start = match.group(1)
        script_num = match.group(2)
        content = match.group(3)
        marker_end = match.group(4)

        # استخراج الأجزاء
        part_pattern = r'<<<PART_(\d+)>>>(.*?)<<<END_PART>>>'
        parts = list(re.finditer(part_pattern, content, re.DOTALL))

        script_failed = False
        script_issues = []

        # فحص 1: عدد الأجزاء
        if len(parts) != expected_parts:
            script_issues.append(f"عدد أجزاء = {len(parts)} (المطلوب {expected_parts})")
            script_failed = True

        fixed_content_parts = []
        for part_match in parts:
            part_num = part_match.group(1)
            part_text = part_match.group(2).strip()

            # إصلاح 1: دمج الأسطر في قطعة واحدة
            lines = [l.strip() for l in part_text.split('\n') if l.strip()]
            if len(lines) > 1:
                part_text = ' '.join(lines)
                total_fixes += 1

            # إصلاح 2: إضافة نقطة في الآخر لو مفيش علامة ترقيم
            stripped = part_text.rstrip()
            if stripped:
                last_char = stripped[-1]
                # شيل التشكيل من الآخر عشان نشوف الحرف الفعلي
                while last_char in TASHKEEL and len(stripped) > 1:
                    stripped = stripped[:-1]
                    last_char = stripped[-1]
                if last_char not in '.،؟!':
                    part_text = part_text.rstrip() + '.'
                    total_fixes += 1

            # فحص 2: عدد الكلمات
            words = part_text.split()
            word_count = len(words)
            if word_count < min_words or word_count > max_words:
                script_issues.append(f"P{part_num}: {word_count} كلمة (النطاق {min_words}-{max_words})")
                script_failed = True

            # فحص 3: أقواس مربعة حول حروف — [ت] [ة] [ن]
            brackets = _check_brackets_pattern(part_text)
            if len(brackets) > bracket_threshold:
                script_issues.append(f"P{part_num}: {len(brackets)} قوس مربع — أقواس حول حروف")
                script_failed = True

            # فحص 4: حروف منفردة كارثية — كل كلمة = حرف واحد
            single_count, single_pct_val = _check_single_letters(part_text, TASHKEEL)
            if single_pct_val > single_letter_pct:
                script_issues.append(f"P{part_num}: {single_count} حرف منفرد ({single_pct_val:.0f}%) — نص مفكّك")
                script_failed = True

            # فحص 5: كلمات مبتورة — نسبة عالية من كلمات حرفين أو أقل
            trunc_count, trunc_pct_val = _check_truncated_words(part_text, TASHKEEL)
            if trunc_pct_val > truncated_pct:
                script_issues.append(f"P{part_num}: {trunc_count} كلمة مبتورة ({trunc_pct_val:.0f}%) — قطع في النص")
                script_failed = True

            # فحص 6: همزة منفصلة — إ لَى بدل إِلَى
            split_hamzas = _check_split_hamza(part_text)
            if len(split_hamzas) > split_hamza_threshold:
                script_issues.append(f"P{part_num}: {len(split_hamzas)} همزة منفصلة — مثل 'إ لَى'")
                script_failed = True

            # فحص 7: تكرار كلمات متتالية (AI stuttering)
            repeated_w = _check_repeated_words(part_text, TASHKEEL)
            if repeated_w > repeated_words_threshold:
                script_issues.append(f"P{part_num}: {repeated_w} كلمة مكررة متتالية — تكرار AI")
                script_failed = True

            # فحص 8: حروف لاتينية في نص عربي
            latin = _check_latin_chars(part_text)
            if len(latin) > latin_threshold:
                script_issues.append(f"P{part_num}: {len(latin)} كلمة لاتينية — '{', '.join(latin[:3])}'")
                script_failed = True

            # فحص 9: بقايا Markdown
            md_artifacts = _check_markdown_artifacts(part_text)
            if md_artifacts:
                script_issues.append(f"P{part_num}: بقايا Markdown — {', '.join(md_artifacts)}")
                script_failed = True

            # فحص 10: غياب التشكيل
            tashkeel_ratio = _check_missing_tashkeel(part_text, TASHKEEL)
            if 0 < tashkeel_ratio < min_tashkeel_pct:
                script_issues.append(f"P{part_num}: تشكيل {tashkeel_ratio:.0f}% فقط — نص بدون تشكيل كافي")
                script_failed = True

            # فحص 11: تكرار عبارات كاملة (3+ كلمات)
            repeated_ph = _check_repeated_phrases(part_text, TASHKEEL)
            if repeated_ph > repeated_phrases_threshold:
                script_issues.append(f"P{part_num}: {repeated_ph} عبارة مكررة — تكرار محتوى")
                script_failed = True

            # فحص 12: أرقام digits في النص
            digits = _check_digit_numbers(part_text)
            if digits:
                script_issues.append(f"P{part_num}: أرقام رقمية — '{', '.join(digits[:3])}'")
                script_failed = True

            # فحص 13: أقواس غير مغلقة
            unclosed = _check_unclosed_parens(part_text)
            if unclosed > 0:
                script_issues.append(f"P{part_num}: {unclosed} قوس غير مغلق")
                script_failed = True

            # فحص 14: جزء فارغ أو شبه فارغ
            if word_count < 5:
                script_issues.append(f"P{part_num}: جزء فارغ ({word_count} كلمات)")
                script_failed = True

            # فحص 15: حرف التطويل ـ
            tatweel_count = _check_tatweel_in_text(part_text)
            if tatweel_count > 0:
                script_issues.append(f"P{part_num}: {tatweel_count} حرف Tatweel ـ")
                script_failed = True

            # فحص 16: مسافات وسط الكلمات
            mid_spaces = _check_mid_word_spaces_count(part_text, TASHKEEL)
            if mid_spaces > 0:
                script_issues.append(f"P{part_num}: {mid_spaces} مسافة وسط كلمة")
                script_failed = True

            fixed_content_parts.append(f"<<<PART_{part_num}>>>\n{part_text}\n<<<END_PART>>>")

        # تجميع المحتوى المصلح
        fixed_content = '\n\n'.join(fixed_content_parts)
        fixed_blocks.append(f"{marker_start}\n\n{fixed_content}\n\n{marker_end}")

        if script_failed:
            failed_scripts.append(script_num)
            total_issues += len(script_issues)
            for issue in script_issues:
                log(f"  [!] SCRIPT_{script_num}: {issue}")

    # Edge case: لو ما لقاش match (الموديل قطع قبل END_SCRIPT)
    # نرجع النص الأصلي + نعلّم الـ SCRIPTs المفتوحة كفاشلة عشان regenerate_failed يشتغل
    if not fixed_blocks:
        open_markers = re.findall(r'<<<SCRIPT_(\d+)>>>', text)
        if open_markers:
            for sn in open_markers:
                if sn not in failed_scripts:
                    failed_scripts.append(sn)
            log(f"  [!] validate_texts: نص ناقص END_SCRIPT — markers مفتوحة={open_markers} → إضافتها للفاشل")
            result_text = text
        else:
            log(f"  [!] validate_texts: النص فاضي تماماً — مفيش SCRIPT markers")
            result_text = ""
    else:
        result_text = '\n\n'.join(fixed_blocks)

    # حفظ قائمة الفاشل في ctx
    failed_key = step["id"] + "_failed"
    ctx.results[failed_key] = failed_scripts

    log(f"  validate_texts: {len(fixed_blocks)} موضوع | {total_fixes} إصلاح أوتوماتيك | {len(failed_scripts)} موضوع فاشل ({total_issues} مشكلة)")

    if failed_scripts:
        log(f"  الموضوعات الفاشلة: {', '.join(failed_scripts)}")

    return result_text


def action_regenerate_failed(step, ctx):
    """إعادة توليد الموضوعات الفاشلة فقط مع loop داخلي.
    ياخد النص الأصلي + قائمة الفاشل من validate_texts ويعيد توليدهم.
    بعد كل generate يفحص عدد parts، لو != expected يعيد المحاولة (max_attempts).
    لو فشل بعد كل المحاولات يعلّم الموضوع كـ permanently_failed.
    """
    text = str(ctx.resolve(step["input"]))
    failed_ref = step.get("failed_ref")
    instructions_ref = step.get("instructions")

    if not failed_ref:
        log("  regenerate_failed: مفيش failed_ref محدد — تخطي")
        ctx.results[step["id"] + "_permanently_failed"] = []
        return text

    failed_scripts = ctx.resolve(failed_ref)
    if not failed_scripts or not isinstance(failed_scripts, list):
        log("  regenerate_failed: مفيش موضوعات فاشلة — تخطي")
        ctx.results[step["id"] + "_permanently_failed"] = []
        return text

    instructions = str(ctx.resolve(instructions_ref)) if instructions_ref else ""

    temperature = step.get("temperature", 0.3)
    max_tokens = step.get("max_tokens", 16000)
    thinking_budget = step.get("thinking_budget", None)
    thinking_level = step.get("thinking_level", None)
    step_model = step.get("model", None)
    effective_model = step_model or ctx.model
    expected_parts = step.get("expected_parts", 4)
    max_attempts = step.get("max_attempts", 3)

    if step_model:
        log(f"  [model override] {step_model}")

    log(f"  regenerate_failed: إعادة توليد {len(failed_scripts)} موضوع (max_attempts={max_attempts}, expected_parts={expected_parts}): {', '.join(failed_scripts)}")

    pattern = r'(<<<SCRIPT_(\d+)>>>)(.*?)(<<<END_SCRIPT>>>)'
    part_pattern = r'<<<PART_(\d+)>>>(.*?)<<<END_PART>>>'

    original_blocks = {}
    for match in re.finditer(pattern, text, re.DOTALL):
        script_num = match.group(2)
        if script_num in failed_scripts:
            original_blocks[script_num] = match.group(0)

    regenerated = {}
    permanently_failed = []

    for script_num in failed_scripts:
        fallback_block = "<<<SCRIPT_" + script_num + ">>>\n<<<END_SCRIPT>>>"
        original_block = original_blocks.get(script_num, fallback_block)
        prompt = (
            f"{instructions}\n\n---\n\n"
            "أعد كتابة النصوص التالية مع الالتزام الكامل بجميع القواعد أعلاه. "
            f"تأكد من أن كل موضوع يحتوي على {expected_parts} أجزاء بالضبط (PART_1 إلى PART_{expected_parts})، "
            "وأن كل جزء قطعة نصية واحدة متصلة، وأن عدد الكلمات في النطاق المطلوب.\n\n"
            f"الموضوع المطلوب إعادة كتابته:\n\n{original_block}"
        )

        success = False
        for attempt in range(1, max_attempts + 1):
            result = generate(
                prompt=prompt,
                model=effective_model,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
            )

            ctx.record_usage(
                f"{step['id']}_SCRIPT_{script_num}_try{attempt}",
                "retry_direct",
                result.provider,
                result.model,
                result.token_usage,
            )

            if not result.success:
                log(f"  [X] SCRIPT_{script_num} (محاولة {attempt}/{max_attempts}): فشل API: {result.error}")
                continue

            new_match = re.search(r'<<<SCRIPT_\d+>>>(.*?)<<<END_SCRIPT>>>', result.data, re.DOTALL)
            new_content = new_match.group(1) if new_match else result.data
            new_parts = list(re.finditer(part_pattern, new_content, re.DOTALL))

            if len(new_parts) == expected_parts:
                regenerated[script_num] = result.data
                success = True
                log(f"  [✓] SCRIPT_{script_num} (محاولة {attempt}/{max_attempts}): نجح ({len(new_parts)} parts)")
                break
            else:
                log(f"  [!] SCRIPT_{script_num} (محاولة {attempt}/{max_attempts}): طلع {len(new_parts)} parts بدل {expected_parts} — إعادة المحاولة")

        if not success:
            permanently_failed.append(script_num)
            log(f"  [XX] SCRIPT_{script_num}: فشل نهائياً بعد {max_attempts} محاولات")

    def replace_block(match):
        script_num = match.group(2)
        if script_num in regenerated:
            new_text = regenerated[script_num]
            new_match = re.search(r'<<<SCRIPT_\d+>>>(.*?)<<<END_SCRIPT>>>', new_text, re.DOTALL)
            if new_match:
                return f"<<<SCRIPT_{script_num}>>>{new_match.group(1)}<<<END_SCRIPT>>>"
            return f"<<<SCRIPT_{script_num}>>>\n{new_text}\n<<<END_SCRIPT>>>"
        return match.group(0)

    result_text = re.sub(pattern, replace_block, text, flags=re.DOTALL)

    ctx.results[step["id"] + "_permanently_failed"] = permanently_failed

    log(f"  regenerate_failed: {len(regenerated)}/{len(failed_scripts)} نجح | {len(permanently_failed)} فشل نهائياً")
    if permanently_failed:
        log(f"  الفاشل النهائي: {', '.join(permanently_failed)}")

    return result_text


def action_assert_no_failures(step, ctx):
    """يتحقق من قائمة الفاشل. لو فيها مواضيع → يرفع exception ويوقف الـ pipeline.
    يقبل failed_ref واحد أو list من المراجع.
    كمان يرفض النص الفاضي حتى لو مفيش failed_scripts (حماية إضافية).
    """
    text = str(ctx.resolve(step["input"]))

    # حماية: لو النص فاضي تماماً → الـ pipeline ضاع منه المحتوى
    if not text or not text.strip():
        msg = "توقف الـ pipeline: الـ input فاضي تماماً — احتمال أن validate_texts أو خطوة سابقة فقدت المحتوى"
        log(f"  [XX] assert_no_failures: {msg}")
        raise RuntimeError(msg)

    failed_ref = step.get("failed_ref")
    failed_refs = step.get("failed_refs")

    refs_to_check = []
    if failed_refs and isinstance(failed_refs, list):
        refs_to_check = failed_refs
    elif failed_ref:
        refs_to_check = [failed_ref]

    if not refs_to_check:
        log("  assert_no_failures: مفيش failed_ref/failed_refs محدد — تخطي")
        return text

    all_failures = []
    for ref in refs_to_check:
        scripts = ctx.resolve(ref)
        if scripts and isinstance(scripts, list):
            all_failures.extend(scripts)

    all_failures = list(dict.fromkeys(all_failures))

    if all_failures:
        msg = f"توقف الـ pipeline: فيه {len(all_failures)} موضوع فاشل لم يتم إصلاحه: {', '.join(all_failures)}"
        log(f"  [XX] assert_no_failures: {msg}")
        raise RuntimeError(msg)

    log(f"  assert_no_failures: ✓ كل المواضيع نجحت")
    return text


# ========== Memory Bank Auto-Update ==========

SECTION_KEYWORDS = {
    "الفشل_الكلوي": ["فشل كلوي", "الفشل الكلوي", "قصور كلوي", "القصور الكلوي", "كلوي نهائي", "مرحلة نهائية"],
    "التهاب_الكبيبات": ["التهاب الكبيبات", "كبيبات الكلى", "نفروني", "نيفريت"],
    "تكيسات_الكلى": ["تكيس", "تكيسات", "كيسات الكلى"],
    "حصوات_الكلى": ["حصوة", "حصوات", "الحصى", "ترسبات الكلى"],
    "البروتين_في_البول": ["بروتين في البول", "البروتين في البول", "زلال البول", "البومين", "ألبومين", "بيلة بروتينية"],
    "ارتفاع_الكرياتينين": ["كرياتينين", "كراتينين"],
    "اليوريا_والنيتروجين": ["يوريا", "نيتروجين"],
    "السكري_والكلى": ["سكري", "السكر", "ديابيت", "ديابتيك"],
    "ضغط_الدم_والكلى": ["ضغط الدم", "ارتفاع الضغط", "ضغطي"],
    "غسيل_الكلى": ["غسيل", "ديال", "dialysis"],
    "زراعة_الكلى": ["زراعة الكلى", "زرع الكلى", "متبرع", "transplant"],
    "أدوية_الكلى": ["دواء", "أدوية", "عقار"],
    "التغذية_للكلى": ["غذاء", "تغذية", "حمية", "نظام غذائي", "ريجيم"],
    "ماء_وسوائل_الكلى": ["جفاف", "ترطيب", "السوائل"],
    "الفحوصات_الكلوية": ["فحص", "تحليل", "اختبار", "تشخيص", "GFR", "سونار", "أشعة"],
    "أعراض_الكلى": ["أعراض", "علامات", "تورم", "رغوة"],
}


def _normalize_arabic_text(s):
    return s.replace("ى", "ي").replace("ة", "ه").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")


def _detect_section(title):
    """تحديد القسم تلقائياً من العنوان — keyword matching بسيط مع normalization عربي"""
    if not title:
        return "_غير_مصنف"
    title_norm = _normalize_arabic_text(title)
    for section, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if _normalize_arabic_text(kw) in title_norm:
                return section
    return "_غير_مصنف"


def _extract_block_summary(body, max_chars=400):
    """استخراج ملخص قصير من نص بلوك (نص أو مقدمة)"""
    if not body:
        return ""
    clean = re.sub(r'<<<(?:PART|END_PART)[^>]*>>>', '', body)
    clean = re.sub(r'<<<END_(?:SCRIPT|INTRO)>>>', '', clean)
    clean = re.sub(r'\.---\.---\.---[^\n]*', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    if len(clean) > max_chars:
        return clean[:max_chars].rsplit(' ', 1)[0] + "..."
    return clean


def action_update_memory_bank(step, ctx):
    """تحديث memory_bank.json بإضافة معلومات الفيديوهات المُنتجة حالياً"""
    text = str(ctx.resolve(step["input"]))
    bank_file = step.get("file", "memory_bank.json")
    max_entries = int(step.get("max_entries", 100))

    pattern = r'<<<(SCRIPT|INTRO)_(\d+)>>>(.*?)<<<END_\1>>>'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        log(f"  update_memory_bank: مفيش ماركرز — تخطي")
        return text

    titles = {}
    topics_ref = step.get("topics")
    if topics_ref:
        try:
            topics_raw = ctx.resolve(topics_ref)
            tdata = json.loads(topics_raw) if isinstance(topics_raw, str) else topics_raw
            items = _extract_items(tdata) or []
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    titles[int(item["id"])] = str(item.get("title", ""))
        except Exception as e:
            log(f"  update_memory_bank: تعذر قراءة topics ({e})")

    bank_path = ctx.input_path(bank_file)
    if os.path.exists(bank_path):
        try:
            with open(bank_path, "r", encoding="utf-8") as f:
                bank = json.load(f)
        except Exception:
            bank = {}
    else:
        bank = {}

    if not isinstance(bank, dict):
        bank = {}
    bank.setdefault("_description", "بنك ذاكرة أوتوماتيكي - يُحدَّث ذاتياً بعد كل تشغيل ناجح. لا تعدّل يدوياً.")
    bank.setdefault("_max_entries", max_entries)
    bank.setdefault("history", [])
    if not isinstance(bank["history"], list):
        bank["history"] = []

    today = datetime.now().strftime("%Y-%m-%d")
    new_entries = []
    for marker_type, vid_id, body in matches:
        try:
            vid = int(vid_id)
        except ValueError:
            continue
        title = titles.get(vid, "")
        section = _detect_section(title)
        summary = _extract_block_summary(body, max_chars=400)
        new_entries.append({
            "video_id": vid,
            "title": title,
            "section": section,
            "summary": summary,
            "date": today,
        })

    bank["history"].extend(new_entries)

    cap = int(bank.get("_max_entries", max_entries) or max_entries)
    if len(bank["history"]) > cap:
        bank["history"] = bank["history"][-cap:]

    try:
        with open(bank_path, "w", encoding="utf-8") as f:
            json.dump(bank, f, ensure_ascii=False, indent=2)
        log(f"  update_memory_bank: أضيف {len(new_entries)} entry — إجمالي: {len(bank['history'])}")
    except Exception as e:
        log(f"  update_memory_bank: فشل الحفظ ({e})")

    return text


# ========== ACTIONS Registry ==========

ACTIONS = {
    "read_input": action_read_input,
    "read_json": action_read_json,
    "generate": action_generate,
    "tts": action_tts,
    "transcribe": action_transcribe,
    "batch_send": action_batch_send,
    "batch_retrieve": action_batch_retrieve,
    "save_file": action_save_file,
    "template": action_template,
    "format_text": action_format_text,
    "save_docx": action_save_docx,
    "read_docx": action_read_docx,
    "read_excel": action_read_excel,
    "copy_videos": action_copy_videos,
    "tts_multi": action_tts_multi,
    "extract_screen_text": action_extract_screen_text,
    "tts_segments": action_tts_segments,
    "montage_short": action_montage_short,
    "remove_tashkeel": action_remove_tashkeel,
    "clean_text": action_clean_text,
    "split_script": action_split_script,
    "scripts_to_topics_json": action_scripts_to_topics_json,
    "topics_to_markers": action_topics_to_markers,
    "draw_thumbnail": action_draw_thumbnail,
    "validate_texts": action_validate_texts,
    "regenerate_failed": action_regenerate_failed,
    "assert_no_failures": action_assert_no_failures,
    "auto_fix_text": action_auto_fix_text,
    "strip_last_letter_diacritic": action_strip_last_letter_diacritic,
    "restore_truncated_words": action_restore_truncated_words,
    "filter_by_topics": action_filter_by_topics,
    "update_memory_bank": action_update_memory_bank,
}

# ========== REQUIRED_PARAMS ==========

REQUIRED_PARAMS = {
    "read_input": ["file"],
    "read_json": ["file"],
    "generate": ["input"],
    "tts": ["input"],
    "transcribe": ["input"],
    "batch_send": ["prompts"],
    "batch_retrieve": ["input"],
    "save_file": ["input", "save_as"],
    "template": ["text"],
    "format_text": ["input"],
    "clean_text": ["input"],
    "save_docx": ["input", "save_as"],
    "read_docx": [],
    "read_excel": ["file"],
    "copy_videos": ["input"],
    "tts_multi": ["input"],
    "extract_screen_text": [],
    "tts_segments": ["input"],
    "montage_short": ["input", "screen_texts"],
    "remove_tashkeel": ["input"],
    "split_script": ["input", "part"],
    "scripts_to_topics_json": ["input"],
    "topics_to_markers": ["input"],
    "draw_thumbnail": ["input"],
    "validate_texts": ["input"],
    "regenerate_failed": ["input"],
    "assert_no_failures": ["input"],
    "auto_fix_text": ["input"],
    "strip_last_letter_diacritic": ["input"],
    "restore_truncated_words": ["input", "original"],
    "filter_by_topics": ["input"],
    "update_memory_bank": ["input"],
}


# ========== Validation ==========

def validate_pipeline(config):
    """التحقق من صحة الـ pipeline config"""
    errors = []

    # التأكد من وجود steps
    if "steps" not in config:
        errors.append("مفيش 'steps' في الـ config")
        return errors

    steps = config["steps"]
    if not isinstance(steps, list) or len(steps) == 0:
        errors.append("'steps' لازم تكون قائمة غير فارغة")
        return errors

    seen_ids = set()
    for i, step in enumerate(steps):
        step_label = f"الخطوة {i + 1}"

        # التأكد من وجود id
        if "id" not in step:
            errors.append(f"{step_label}: مفيش 'id'")
            continue

        step_id = step["id"]
        step_label = f"الخطوة '{step_id}'"

        # التأكد من عدم تكرار الـ id
        if step_id in seen_ids:
            errors.append(f"{step_label}: الـ id مكرر")
        seen_ids.add(step_id)

        # التأكد من وجود action
        if "action" not in step:
            errors.append(f"{step_label}: مفيش 'action'")
            continue

        action = step["action"]

        # التأكد من أن الـ action موجود
        if action not in ACTIONS:
            errors.append(f"{step_label}: action غير معروف '{action}'. المتاح: {', '.join(ACTIONS.keys())}")
            continue

        # التأكد من وجود الـ required params
        required = REQUIRED_PARAMS.get(action, [])
        for param in required:
            if param not in step:
                errors.append(f"{step_label}: الـ param '{param}' مطلوب لـ action '{action}'")

        # التأكد من أن الـ references تشاور على خطوات سابقة
        for key, value in step.items():
            if key in ("id", "action"):
                continue
            if isinstance(value, str):
                # البحث عن {references}
                import re
                refs = re.findall(r'\{(\w+)\}', value)
                for ref in refs:
                    # تجاهل references اللي بتنتهي بـ _failed (بتتحفظ في runtime من validate_texts)
                    if ref not in seen_ids and not ref.endswith("_failed"):
                        errors.append(f"{step_label}: الـ reference '{{{ref}}}' بيشاور على خطوة مش موجودة قبلها")

    return errors


# ========== Step Execution (extracted from run_pipeline) ==========

def _run_steps(steps, ctx, start=0, end=None):
    """تنفيذ خطوات من الـ pipeline (من start إلى end)"""
    if end is None:
        end = len(steps)

    for i in range(start, end):
        step = steps[i]
        step_id = step["id"]
        action_name = step["action"]
        step_label = step.get("label", f"{action_name} ({step_id})")

        log(f"--- الخطوة {i + 1}/{len(steps)}: {step_label} ---")

        try:
            action_func = ACTIONS[action_name]
            result = action_func(step, ctx)
            ctx.results[step_id] = result

            # عرض ملخص النتيجة
            if isinstance(result, str):
                if len(result) > 100:
                    log(f"  النتيجة: {result[:100]}...")
                else:
                    log(f"  النتيجة: {result}")
            elif isinstance(result, list):
                log(f"  النتيجة: قائمة ({len(result)} عنصر)")
            elif isinstance(result, dict):
                log(f"  النتيجة: dict ({len(result)} مفتاح)")

        except EngineError as e:
            log(f"[X] فشل في الخطوة '{step_id}': {e.message}")
            sys.exit(1)
        except Exception as e:
            log(f"[X] خطأ غير متوقع في الخطوة '{step_id}': {str(e)}")
            sys.exit(1)


# ========== Batch Mode Helpers ==========

def _find_generate_step_index(steps):
    """إيجاد index أول خطوة generate في الـ pipeline"""
    for i, step in enumerate(steps):
        if step["action"] == "generate":
            return i
    return None


def _find_all_generate_step_indices(steps):
    """إيجاد indices كل خطوات generate في الـ pipeline"""
    return [i for i, step in enumerate(steps) if step["action"] == "generate"]


def _extract_topics_from_context(ctx):
    """استخراج قائمة المواضيع من ctx.results — يرجع list of dicts أو None"""
    for step_id, result in ctx.results.items():
        # تحويل النتيجة لـ data
        if isinstance(result, str):
            try:
                data = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                continue
        elif isinstance(result, (list, dict)):
            data = result
        else:
            continue

        # استخراج العناصر
        items = _extract_items(data)
        if items and isinstance(items, list) and len(items) > 0:
            if isinstance(items[0], dict) and "id" in items[0] and "title" in items[0]:
                log(f"  تم العثور على {len(items)} موضوع في الخطوة '{step_id}'")
                return items

    return None


def _detect_marker_prefix(config):
    """قراءة marker_prefix من خطوة save_docx (default: SCRIPT)"""
    for step in config["steps"]:
        if step["action"] == "save_docx":
            return step.get("marker_prefix", "SCRIPT")
    return "SCRIPT"


def _find_topics_step_id(steps):
    """إيجاد الخطوة اللي بتحمّل topics.json"""
    for step in steps:
        if step["action"] in ("read_input", "read_json"):
            filename = step.get("file", "")
            if "topics" in filename.lower():
                return step["id"]
    return None


def _build_batch_prompts(config, ctx, topics, marker_prefix):
    """بناء برومبت لكل موضوع — يرجع قائمة prompts"""
    steps = config["steps"]

    # إيجاد خطوة template
    template_step = None
    for step in steps:
        if step["action"] == "template":
            template_step = step
            break

    if not template_step:
        log("[!] لا توجد خطوة template — تعذر بناء البرومبتات")
        return None

    # إيجاد step_id اللي بيحمّل المواضيع
    topics_step_id = _find_topics_step_id(steps)
    if not topics_step_id:
        log("[!] لا توجد خطوة تحميل مواضيع — تعذر بناء البرومبتات")
        return None

    template_text = template_step["text"]

    prompts = []
    for topic in topics:
        topic_id = topic.get("id", 0)

        # صياغة موضوع واحد كـ JSON
        single_topic_json = json.dumps([topic], ensure_ascii=False, indent=2)

        # حل المتغيرات يدوياً — استبدال {step_id} بالنتائج
        prompt_text = template_text
        for step_id, result in ctx.results.items():
            if step_id == topics_step_id:
                prompt_text = prompt_text.replace(f"{{{step_id}}}", single_topic_json)
            else:
                prompt_text = prompt_text.replace(f"{{{step_id}}}", str(result))

        # إضافة تنبيه الماركر
        marker_instruction = (
            f"\n\nمهم جداً: ابدأ الناتج بـ <<<{marker_prefix}_{topic_id}>>> "
            f"وانهيه بـ <<<END_{marker_prefix}>>>"
        )
        prompts.append(prompt_text + marker_instruction)

    log(f"  تم بناء {len(prompts)} برومبت (موضوع لكل طلب)")
    return prompts


def _retry_truncated_batch_results(batch_results, topics, marker_prefix, config, ctx, metadata):
    """كشف السكريبتات المقطوعة (MAX_TOKENS) وإعادة توليدها عبر API عادي.

    السبب: موديلات thinking (مثل gemini-3.1-pro-preview) ممكن تستهلك
    معظم الـ token budget في التفكير، ويتبقى مساحة قليلة للكتابة.
    الباتش API بيرجع finishReason=MAX_TOKENS في الحالة دي.

    الحل: نعيد توليد السكريبتات المقطوعة واحد واحد عبر API العادي
    (اللي بيتعامل مع الـ thinking budget أحسن).
    """
    # الخطوة 1: كشف المقطوعات — نقرأ الـ predictions.jsonl مباشرة من GCS
    truncated_indices = []
    try:
        batch_info_path = metadata.get("batch_info_path", "")
        if batch_info_path:
            with open(batch_info_path, 'r') as f:
                batch_info_data = json.load(f)

            # اكتشاف الموفر
            provider = batch_info_data.get("provider", "")
            if provider == "gemini":
                from engine import _download_from_gcs

                # تحميل النتائج الخام
                job_name = batch_info_data.get("job_name", "")
                extra = batch_info_data.get("extra", {})

                from google import genai
                from engine import _setup_gcs_credentials
                project_id, location, bucket_name = _setup_gcs_credentials()
                saved_location = extra.get("location", location)
                client = genai.Client(vertexai=True, project=project_id, location=saved_location)

                batch_job = client.batches.get(name=job_name)
                if hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'gcs_uri'):
                    jsonl_content = _download_from_gcs(batch_job.dest.gcs_uri)

                    for line_idx, line in enumerate(jsonl_content.strip().split('\n')):
                        if line:
                            try:
                                data = json.loads(line)
                                finish = data['response']['candidates'][0].get('finishReason', 'UNKNOWN')
                                if finish == 'MAX_TOKENS':
                                    truncated_indices.append(line_idx)
                            except (KeyError, IndexError, json.JSONDecodeError):
                                pass
    except Exception as e:
        log(f"  [!] فشل كشف المقطوعات: {str(e)[:200]}")

    if not truncated_indices:
        return batch_results

    log(f"  [!] تم كشف {len(truncated_indices)} سكريبت مقطوع (MAX_TOKENS) — إعادة توليد...")

    # الخطوة 2: تحديد الـ topic ID لكل نتيجة مقطوعة من الـ SCRIPT marker
    truncated_topic_ids = []
    for batch_idx in truncated_indices:
        if batch_idx >= len(batch_results):
            continue
        text = batch_results[batch_idx]
        marker_match = re.search(rf'<<<{marker_prefix}_(\d+)>>>', text)
        if marker_match:
            topic_id = int(marker_match.group(1))
            truncated_topic_ids.append(topic_id)
        else:
            log(f"  [!] نتيجة مقطوعة في index {batch_idx} بدون marker — تخطي")

    if not truncated_topic_ids:
        return batch_results

    # الخطوة 3: بناء prompts لكل المواضيع (بنفس ترتيب topics)
    prompts = _build_batch_prompts(config, ctx, topics, marker_prefix)
    if not prompts:
        log(f"  [!] فشل بناء البرومبتات للـ retry")
        return batch_results

    # بناء خريطة topic_id → prompt_index
    topic_id_to_prompt_idx = {}
    for idx, topic in enumerate(topics):
        topic_id_to_prompt_idx[topic.get("id", 0)] = idx

    # الخطوة 4: إعادة توليد عبر API العادي
    gen_step_idx = metadata.get("generate_step_index", 0)
    steps = config["steps"]
    gen_step = steps[gen_step_idx]
    system_prompt = ctx.resolve(gen_step.get("system_prompt", "")) if gen_step.get("system_prompt") else ""
    temperature = gen_step.get("temperature", 0.7)
    max_tokens = gen_step.get("max_tokens", 8192)

    retry_count = 0
    max_retries = 3

    # Thinking models (مثل gemini-3.1-pro-preview) بتستهلك التوكنز في التفكير.
    # الريتراي لازم يستخدم max_tokens أعلى عشان يوفّر مساحة كافية للمخرج.
    retry_max_tokens_levels = [
        min(max_tokens * 2, 65536),   # محاولة 1: ضعف
        min(max_tokens * 2, 65536),   # محاولة 2: ضعف (random seed مختلف)
        min(max_tokens * 4, 131072),  # محاولة 3: 4 أضعاف
    ]

    for topic_id in truncated_topic_ids:
        prompt_idx = topic_id_to_prompt_idx.get(topic_id)
        if prompt_idx is None:
            log(f"  [!] SCRIPT_{topic_id}: لا يوجد prompt مطابق — تخطي")
            continue

        # إيجاد النتيجة المقطوعة في batch_results
        batch_idx = None
        for bi in truncated_indices:
            if bi < len(batch_results):
                m = re.search(rf'<<<{marker_prefix}_{topic_id}>>>', batch_results[bi])
                if m:
                    batch_idx = bi
                    break

        if batch_idx is None:
            log(f"  [!] SCRIPT_{topic_id}: لم يتم العثور على النتيجة المقطوعة — تخطي")
            continue

        success = False
        for attempt in range(max_retries):
            retry_tokens = retry_max_tokens_levels[attempt]
            try:
                log(f"  → إعادة توليد SCRIPT_{topic_id} (محاولة {attempt + 1}/{max_retries}, max_tokens={retry_tokens})...")
                result = generate(
                    prompt=prompts[prompt_idx],
                    model=ctx.model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=retry_tokens
                )

                if result.success and result.data:
                    new_text = result.data.strip()
                    old_len = len(batch_results[batch_idx])
                    new_len = len(new_text)

                    # تسجيل التوكنز — retry_direct (تكلفة كاملة مش باتش)
                    ctx.record_usage(
                        f"retry_truncated_SCRIPT_{topic_id}",
                        "retry_direct", result.provider, result.model, result.token_usage
                    )

                    end_marker = f"<<<END_{marker_prefix}>>>"
                    has_end = end_marker in new_text

                    if has_end:
                        batch_results[batch_idx] = new_text
                        log(f"  ✓ SCRIPT_{topic_id}: {old_len} → {new_len} حرف | END=True")
                        retry_count += 1
                        success = True
                        break
                    else:
                        log(f"  [!] SCRIPT_{topic_id}: بدون END marker ({new_len} حرف, max_tokens={retry_tokens})")
                else:
                    log(f"  [!] SCRIPT_{topic_id}: فشل التوليد — {getattr(result, 'error', '?')}")
            except Exception as e:
                log(f"  [!] SCRIPT_{topic_id}: خطأ — {str(e)[:200]}")

        if not success:
            log(f"  [!] SCRIPT_{topic_id}: فشلت كل المحاولات — استخدام النتيجة المقطوعة")

    log(f"  تم إصلاح {retry_count}/{len(truncated_topic_ids)} سكريبت مقطوع")
    return batch_results


def _reassemble_batch_results(results, topics, marker_prefix):
    """تجميع نتائج الباتش في نص واحد بالماركرز الصحيحة

    مهم: نتائج الباتش ممكن ترجع بترتيب مختلف عن الإرسال.
    عشان كده بنربط كل نتيجة بالموضوع الصح عن طريق الماركر اللي الـ AI حطه،
    مش عن طريق ترتيب النتائج (index).
    """
    # الخطوة 1: تصنيف النتائج حسب الماركر الموجود فيها
    marker_pattern = re.compile(rf'<<<{marker_prefix}_(\d+)>>>')
    result_by_topic_id = {}  # topic_id → result text
    unmatched_results = []   # نتائج بدون ماركر

    for text in results:
        text = text.strip()
        match = marker_pattern.search(text)
        if match:
            actual_id = int(match.group(1))
            result_by_topic_id[actual_id] = text
        else:
            unmatched_results.append(text)

    if unmatched_results:
        log(f"  [!] {len(unmatched_results)} نتيجة بدون ماركر (من {len(results)})")

    matched_by_marker = len(result_by_topic_id)
    matched_by_position = 0

    # الخطوة 2: تجميع النتائج بالترتيب الصحيح
    combined_parts = []
    unmatched_idx = 0  # مؤشر للنتائج بدون ماركر

    for topic in topics:
        topic_id = topic.get("id", 0)
        expected_marker = f"<<<{marker_prefix}_{topic_id}>>>"
        end_marker = f"<<<END_{marker_prefix}>>>"

        if topic_id in result_by_topic_id:
            # لقينا النتيجة بالماركر الصح
            text = result_by_topic_id[topic_id]
            # تأكد إن END marker موجود — لو الـ AI نسيه نضيفه
            if end_marker not in text:
                text = text.rstrip() + f"\n{end_marker}"
                log(f"  [~] موضوع {topic_id}: تم إضافة {end_marker} (الـ AI نسيه)")
            combined_parts.append(text)
        elif unmatched_idx < len(unmatched_results):
            # مفيش ماركر — نستخدم أول نتيجة متاحة ونلفها بالماركر الصح
            text = unmatched_results[unmatched_idx]
            unmatched_idx += 1
            combined_parts.append(f"{expected_marker}\n{text}\n{end_marker}")
            matched_by_position += 1
            log(f"  [!] موضوع {topic_id}: ربط بدون ماركر (ترتيب موضعي)")
        else:
            log(f"  [!] موضوع {topic_id}: لا توجد نتيجة — تخطي")

    combined = "\n\n".join(combined_parts)
    log(f"  تم تجميع {len(combined_parts)} نتيجة ({len(combined)} حرف)")
    if matched_by_marker > 0:
        log(f"  ربط بالماركر: {matched_by_marker} | ربط موضعي: {matched_by_position}")
    return combined


def _save_batch_metadata(ctx, batch_info_path, topics, gen_step_id, gen_idx, marker_prefix,
                         batch_mode="topics", prompts_count=0):
    """حفظ metadata الباتش في output_dir — للـ receive_only

    يحفظ كل المعلومات اللازمة لتعريف الباتش بشكل فريد:
    - recipe_name + channel_name + run_id → تعريف فريد
    - batch_mode → طريقة إعادة التجميع (topics/markers/single)
    - prompts_count → للتحقق عند الاستقبال
    """
    metadata = {
        "batch_info_path": batch_info_path,
        "topics": topics,
        "generate_step_id": gen_step_id,
        "generate_step_index": gen_idx,
        "marker_prefix": marker_prefix,
        "batch_mode": batch_mode,
        "prompts_count": prompts_count,
        "recipe_name": ctx.recipe_name,
        "channel_name": ctx.channel_name,
        "run_id": ctx.run_id,
        "pre_results": {},
        "saved_at": datetime.now().isoformat(),
    }

    # حفظ نتائج الخطوات اللي قبل generate
    for step_id, result in ctx.results.items():
        if isinstance(result, (str, int, float, bool, list, dict)):
            metadata["pre_results"][step_id] = result

    metadata_path = ctx.output_path("batch_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log(f"  تم حفظ batch metadata في: {metadata_path}")
    log(f"  [metadata] recipe={ctx.recipe_name} | channel={ctx.channel_name} | mode={batch_mode} | prompts={prompts_count}")


def _load_batch_metadata(ctx):
    """تحميل metadata الباتش — البحث في output_dir فقط (= مجلد الوصفة الدائم).

    بعد توحيد المجلدات: output_dir = RECIPE_OUTPUT_DIR مباشرة.
    ممنوع البحث في مجلدات تانية — ده كان سبب باج خطير.
    """
    metadata_path = ctx.output_path("batch_metadata.json")

    if not os.path.exists(metadata_path):
        raise EngineError(
            f"ملف batch_metadata.json غير موجود. شغّل 'إرسال فقط' الأول.\n"
            f"  output_dir: {ctx.output_dir}",
            code="BATCH_METADATA_NOT_FOUND"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # === تحقق من هوية الوصفة (لمنع استخدام metadata من وصفة تانية) ===
    saved_recipe = metadata.get("recipe_name", "")
    saved_channel = metadata.get("channel_name", "")

    if saved_recipe and ctx.recipe_name and saved_recipe != ctx.recipe_name:
        raise EngineError(
            f"batch_metadata من وصفة مختلفة!\n"
            f"  المحفوظ: '{saved_recipe}' | الحالي: '{ctx.recipe_name}'\n"
            f"  هذا يعني إن metadata قديمة من تشغيلة سابقة لوصفة تانية.\n"
            f"  الحل: شغّل 'إرسال فقط' لهذه الوصفة الأول.",
            code="BATCH_METADATA_RECIPE_MISMATCH"
        )

    if saved_channel and ctx.channel_name and saved_channel != ctx.channel_name:
        log(f"  [!] تحذير: metadata من قناة مختلفة (محفوظ: {saved_channel}, حالي: {ctx.channel_name})")

    log(f"  [metadata] recipe={saved_recipe} | channel={saved_channel} | mode={metadata.get('batch_mode', 'topics')} | prompts={metadata.get('prompts_count', '?')}")

    return metadata


# ========== Batch Mode Functions ==========

def _run_mode_send_only(config, ctx, steps):
    """وضع إرسال فقط: خطوات قبل generate → بناء باتش → إرسال → حفظ metadata → STOP

    يدعم طريقتين لبناء البرومبتات:
    1. topics_mode: لو فيه topics.json (بيانات JSON فيها id + title)
    2. marker_split_mode: لو المدخل فيه ماركرز SCRIPT_N/INTRO_N (من docx مثلاً)

    ممنوع fallback لوضع فوري — لو الاتنين فشلوا = خطأ صريح.
    """
    gen_idx = _find_generate_step_index(steps)
    if gen_idx is None:
        raise EngineError(
            "وضع send_only يتطلب خطوة generate في الوصفة — مفيش خطوة generate.",
            code="SEND_ONLY_NO_GENERATE"
        )

    # تنفيذ الخطوات قبل generate
    _run_steps(steps, ctx, start=0, end=gen_idx)

    gen_step = steps[gen_idx]
    marker_prefix = _detect_marker_prefix(config)
    prompts = None
    topics = None
    batch_mode = None  # "topics" أو "markers"

    # === الطريقة 1: topics_mode (JSON مع id + title) ===
    topics = _extract_topics_from_context(ctx)
    if topics and len(topics) > 1:
        prompts = _build_batch_prompts(config, ctx, topics, marker_prefix)
        if prompts:
            batch_mode = "topics"
            log(f"  [send_only] topics_mode: {len(prompts)} برومبت من {len(topics)} موضوع")

    # === الطريقة 2: marker_split_mode (ماركرز SCRIPT/INTRO في النص) ===
    if prompts is None:
        prompt_str = str(ctx.resolve(gen_step["input"]))
        MARKER_PAT = r'<<<((?:SCRIPT|INTRO)_\d+)>>>'
        separator = "\n---\n"

        if separator in prompt_str:
            content_section = prompt_str[prompt_str.rfind(separator) + len(separator):]
        else:
            content_section = prompt_str

        seen = set()
        input_markers = []
        for m in re.findall(MARKER_PAT, content_section):
            if m not in seen:
                seen.add(m)
                input_markers.append(m)

        # فلترة حسب TOPIC_IDS لو محدد
        if ctx.topic_ids and input_markers:
            filtered_markers = []
            for m in input_markers:
                id_match = re.search(r'_(\d+)', m)
                if id_match and int(id_match.group(1)) in ctx.topic_ids:
                    filtered_markers.append(m)
            if filtered_markers:
                log(f"  [send_only] فلترة الماركرز: {len(filtered_markers)} من {len(input_markers)} (TOPIC_IDS)")
                input_markers = filtered_markers

        if len(input_markers) > 1:
            # تقسيم حسب الماركرز — كل ماركر في طلب لوحده
            content_start = prompt_str.rfind(separator) + len(separator) if separator in prompt_str else 0
            first_match = re.search(MARKER_PAT, content_section)
            instructions_part = prompt_str[:content_start + first_match.start()] if first_match else prompt_str
            content_with_markers = prompt_str[content_start:]

            prompts = []
            for marker in input_markers:
                escaped = re.escape(f'<<<{marker}>>>')
                pat = rf'({escaped}.*?)(?=<<<(?:SCRIPT|INTRO)_\d+>>>|\Z)'
                match = re.search(pat, content_with_markers, re.DOTALL)
                if match:
                    prompts.append(instructions_part + match.group(1).strip())

            if prompts:
                batch_mode = "markers"
                # حفظ ترتيب الماركرز للاستعادة لاحقاً
                topics = [{"id": int(re.search(r'_(\d+)', m).group(1)), "marker": m} for m in input_markers]
                log(f"  [send_only] marker_split_mode: {len(prompts)} برومبت من {len(input_markers)} ماركر")
        elif len(input_markers) == 1:
            # ماركر واحد — برومبت واحد
            prompts = [prompt_str]
            batch_mode = "markers"
            topics = [{"id": int(re.search(r'_(\d+)', input_markers[0]).group(1)), "marker": input_markers[0]}]
            log(f"  [send_only] ماركر واحد — برومبت واحد")
        else:
            # مفيش ماركرز — برومبت واحد
            prompts = [prompt_str]
            batch_mode = "single"
            topics = []
            log(f"  [send_only] مفيش topics أو markers — برومبت واحد")

    if not prompts:
        raise EngineError(
            "فشل بناء البرومبتات لـ send_only — مفيش topics.json ولا ماركرز في المدخل.",
            code="SEND_ONLY_NO_PROMPTS"
        )

    # إرسال الباتش
    system_prompt = ctx.resolve(gen_step.get("system_prompt", "")) if gen_step.get("system_prompt") else ""
    temperature = gen_step.get("temperature", 0.7)
    max_tokens = gen_step.get("max_tokens", 8192)
    thinking_budget = gen_step.get("thinking_budget", None)
    thinking_level = gen_step.get("thinking_level", None)

    save_path = ctx.output_path("batch_job_info.json")

    effective_model = gen_step.get("model") or ctx.model
    if gen_step.get("model"):
        log(f"  [model override] {gen_step['model']}")

    # Labels للتتبع في Google Cloud Billing
    _labels = {"step": gen_step["id"]}
    if ctx.run_id:
        _labels["run_id"] = ctx.run_id
    if ctx.recipe_name:
        _labels["recipe"] = ctx.recipe_name
    if ctx.channel_name:
        _labels["channel"] = ctx.channel_name

    log(f"--- إرسال باتش: {len(prompts)} طلب (mode: {batch_mode}) ---")
    result = batch_send(
        prompts=prompts,
        model=effective_model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        save_path=save_path,
        thinking_budget=thinking_budget,
        thinking_level=thinking_level,
        labels=_labels,
    )

    if not result.success:
        raise EngineError(f"فشل إرسال الباتش: {result.error}", code="BATCH_SEND_FAILED")

    # حفظ metadata مع معلومات التعريف الكاملة
    _save_batch_metadata(ctx, save_path, topics, gen_step["id"], gen_idx, marker_prefix,
                         batch_mode=batch_mode, prompts_count=len(prompts))

    log(f"========== تم إرسال الباتش ({len(prompts)} طلب, mode={batch_mode}) — في انتظار النتائج ==========")


def _run_mode_receive_only(config, ctx, steps):
    """وضع استقبال فقط: تحميل metadata → استقبال نتائج → باقي الخطوات

    يدعم 3 أوضاع تجميع حسب batch_mode المحفوظ في metadata:
    - topics: تجميع حسب topics (id + title) مع ماركرز
    - markers: تجميع حسب ماركرز SCRIPT/INTRO بالـ ID
    - single: برومبت واحد — النتيجة مباشرة
    """
    # تحميل metadata (مع التحقق من هوية الوصفة)
    metadata = _load_batch_metadata(ctx)

    gen_idx = metadata["generate_step_index"]
    topics = metadata["topics"]
    marker_prefix = metadata["marker_prefix"]
    gen_step_id = metadata["generate_step_id"]
    batch_info_path = metadata["batch_info_path"]
    batch_mode = metadata.get("batch_mode", "topics")
    expected_prompts = metadata.get("prompts_count", 0)

    log(f"  [receive] batch_mode={batch_mode} | expected_prompts={expected_prompts}")

    # استعادة نتائج الخطوات اللي قبل generate
    for step_id, result in metadata.get("pre_results", {}).items():
        ctx.results[step_id] = result

    # استقبال نتائج الباتش مع polling
    log(f"--- استقبال نتائج الباتش ---")
    poll_interval = 30
    max_wait = 7200  # ساعتين
    start_time = time.time()

    batch_results = None
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise EngineError(
                f"تجاوز الوقت ({max_wait}s) في انتظار نتائج الباتش",
                code="BATCH_RETRIEVE_TIMEOUT"
            )

        try:
            result = batch_retrieve(batch_info_path=batch_info_path)
            if result.success:
                batch_results = result.data
                log(f"  تم استقبال {len(batch_results)} نتيجة")
                # تحقق من عدد النتائج مقابل المتوقع
                if expected_prompts and len(batch_results) != expected_prompts:
                    log(f"  [!] تحذير: عدد النتائج ({len(batch_results)}) ≠ المتوقع ({expected_prompts})")
                break
        except EngineError as e:
            if e.code == "BATCH_JOB_NOT_READY":
                log(f"  المهمة لم تكتمل ({int(elapsed)}s). انتظار {poll_interval}s...")
                time.sleep(poll_interval)
                continue
            raise

    # تسجيل استهلاك التوكنز
    if result.token_usage:
        ctx.record_usage(
            step_id=gen_step_id,
            call_type="batch",
            provider=result.provider or "unknown",
            model=result.model or "unknown",
            token_usage=result.token_usage
        )

    # === تجميع النتائج حسب batch_mode ===
    if batch_mode == "topics":
        # كشف وإعادة توليد السكريبتات المقطوعة (MAX_TOKENS)
        batch_results = _retry_truncated_batch_results(
            batch_results, topics, marker_prefix, config, ctx, metadata
        )
        # تجميع حسب topics
        combined = _reassemble_batch_results(batch_results, topics, marker_prefix)

    elif batch_mode == "markers":
        # تجميع حسب ماركرز — ترتيب بالـ ID
        MARKER_PAT_RE = re.compile(r'<<<((?:SCRIPT|INTRO)_(\d+))>>>')
        id_to_result = {}
        unmatched = []
        for r in batch_results:
            m = MARKER_PAT_RE.search(r)
            if m:
                id_to_result[int(m.group(2))] = r
            else:
                unmatched.append(r)

        # كشف المقطوعات (بدون END marker)
        truncated_ids = [tid for tid, text in id_to_result.items() if '<<<END_' not in text]
        if truncated_ids:
            log(f"  [!] {len(truncated_ids)} نتيجة بدون END marker — إعادة توليد...")
            gen_step = steps[gen_idx]
            # بناء prompts من topics (markers)
            prompts_by_id = {}
            for t in topics:
                tid = t.get("id", 0)
                prompts_by_id[tid] = True  # placeholder

            # البرومبتات الأصلية غير متاحة في receive_only — نسجل تحذير بالمقطوعات
            for tid in truncated_ids:
                text = id_to_result.get(tid, "")
                marker_match = re.search(r'<<<((?:SCRIPT|INTRO)_\d+)>>>', text)
                marker_id = marker_match.group(1) if marker_match else f"index_{tid}"
                log(f"  [!] {marker_id} مقطوع (بدون END marker) — البرومبت الأصلي غير متاح للـ retry")

        sorted_results = [id_to_result[k] for k in sorted(id_to_result.keys())]
        sorted_results.extend(unmatched)
        combined = "\n\n".join(sorted_results)
        log(f"  تم تجميع {len(sorted_results)} نتيجة ماركرز بالـ ID ({len(combined)} حرف)")

    elif batch_mode == "single":
        # برومبت واحد — النتيجة مباشرة
        combined = batch_results[0] if batch_results else ""
        log(f"  النتيجة: {combined[:100]}..." if len(combined) > 100 else f"  النتيجة: {combined}")

    else:
        raise EngineError(
            f"batch_mode غير معروف في الـ metadata: '{batch_mode}'. القيم المتاحة: topics, markers, single",
            code="UNKNOWN_BATCH_MODE"
        )

    # فلترة النتائج حسب TOPIC_IDS (لو محدد أرقام مختلفة عن الإرسال الأصلي)
    if ctx.topic_ids:
        combined = _filter_combined_by_topic_ids(combined, ctx.topic_ids)

    ctx.results[gen_step_id] = combined

    log(f"  تم تجميع النتائج ({len(combined)} حرف)")

    # تنفيذ الخطوات بعد generate — مع باتش لأي generate إضافية
    remaining_gen_indices = [i for i in _find_all_generate_step_indices(steps) if i > gen_idx]

    if not remaining_gen_indices:
        # مفيش generate تانية — تشغيل عادي
        _run_steps(steps, ctx, start=gen_idx + 1)
    else:
        # فيه generate إضافية — نشغلهم باتش
        prev_end = gen_idx + 1
        for next_gen_idx in remaining_gen_indices:
            # تنفيذ الخطوات بين الـ generate السابقة والجاية
            if prev_end < next_gen_idx:
                _run_steps(steps, ctx, start=prev_end, end=next_gen_idx)

            # تشغيل الـ generate كباتش
            next_gen_step = steps[next_gen_idx]
            step_label = next_gen_step.get("label", next_gen_step["id"])
            log(f"--- الخطوة {next_gen_idx + 1}/{len(steps)}: {step_label} [BATCH] ---")
            _batch_single_generate(next_gen_step, next_gen_idx, config, ctx, steps, is_primary=False)

            prev_end = next_gen_idx + 1

        # تنفيذ الخطوات بعد آخر generate
        if prev_end < len(steps):
            _run_steps(steps, ctx, start=prev_end)


def _retry_truncated_marker_split(batch_results, prompts, gen_step, ctx, batch_info_path):
    """كشف وإعادة توليد النتائج المقطوعة في marker_split_mode.

    يكشف المقطوعات بطريقتين:
    1. من GCS predictions (finishReason=MAX_TOKENS) — الأدق
    2. fallback: بفحص وجود END marker في النتيجة

    ثم يعيد التوليد عبر API العادي بـ max_tokens أعلى.
    """
    truncated_indices = []

    # الطريقة 1: كشف من GCS predictions
    try:
        if batch_info_path and os.path.exists(batch_info_path):
            with open(batch_info_path, 'r') as f:
                batch_info_data = json.load(f)

            provider = batch_info_data.get("provider", "")
            if provider == "gemini":
                from engine import _download_from_gcs, _setup_gcs_credentials
                from google import genai

                job_name = batch_info_data.get("job_name", "")
                extra = batch_info_data.get("extra", {})
                project_id, location, bucket_name = _setup_gcs_credentials()
                saved_location = extra.get("location", location)
                client = genai.Client(vertexai=True, project=project_id, location=saved_location)

                batch_job = client.batches.get(name=job_name)
                if hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'gcs_uri'):
                    jsonl_content = _download_from_gcs(batch_job.dest.gcs_uri)
                    for line_idx, line in enumerate(jsonl_content.strip().split('\n')):
                        if line:
                            try:
                                data = json.loads(line)
                                finish = data['response']['candidates'][0].get('finishReason', 'UNKNOWN')
                                if finish == 'MAX_TOKENS':
                                    truncated_indices.append(line_idx)
                            except (KeyError, IndexError, json.JSONDecodeError):
                                pass
    except Exception as e:
        log(f"  [!] فشل كشف المقطوعات من GCS: {str(e)[:200]}")

    # الطريقة 2 (fallback): كشف بفحص END marker
    if not truncated_indices:
        for idx, text in enumerate(batch_results):
            if text and '<<<END_' not in text:
                truncated_indices.append(idx)
        if truncated_indices:
            log(f"  [!] كشف {len(truncated_indices)} نتيجة بدون END marker (fallback)")

    if not truncated_indices:
        return batch_results

    log(f"  [!] {len(truncated_indices)} نتيجة مقطوعة (MAX_TOKENS) — إعادة توليد...")

    # إعادة التوليد عبر API العادي
    system_prompt = gen_step.get("system_prompt", "")
    if system_prompt:
        system_prompt = ctx.resolve(system_prompt)
    temperature = gen_step.get("temperature", 0.7)
    max_tokens = gen_step.get("max_tokens", 8192)
    effective_model = gen_step.get("model") or ctx.model

    retry_max_tokens_levels = [
        min(max_tokens * 2, 131072),
        min(max_tokens * 2, 131072),
        min(max_tokens * 4, 131072),
    ]

    retry_count = 0
    for trunc_idx in truncated_indices:
        if trunc_idx >= len(batch_results) or trunc_idx >= len(prompts):
            continue

        # استخراج الماركر للتعريف
        marker_match = re.search(r'<<<((?:SCRIPT|INTRO)_\d+)>>>', batch_results[trunc_idx] if batch_results[trunc_idx] else "")
        marker_id = marker_match.group(1) if marker_match else f"index_{trunc_idx}"

        success = False
        for attempt in range(3):
            retry_tokens = retry_max_tokens_levels[attempt]
            try:
                log(f"  → إعادة توليد {marker_id} (محاولة {attempt + 1}/3, max_tokens={retry_tokens})...")
                result = generate(
                    prompt=prompts[trunc_idx],
                    model=effective_model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=retry_tokens
                )

                if result.success and result.data:
                    new_text = result.data.strip()
                    has_end = '<<<END_' in new_text

                    # تسجيل التوكنز — retry_direct (تكلفة كاملة مش باتش)
                    ctx.record_usage(
                        f"retry_marker_{marker_id}",
                        "retry_direct", result.provider, result.model, result.token_usage
                    )

                    if has_end:
                        old_len = len(batch_results[trunc_idx]) if batch_results[trunc_idx] else 0
                        batch_results[trunc_idx] = new_text
                        log(f"  ✓ {marker_id}: {old_len} → {len(new_text)} حرف | END=True")
                        retry_count += 1
                        success = True
                        break
                    else:
                        log(f"  [!] {marker_id}: بدون END marker ({len(new_text)} حرف, max_tokens={retry_tokens})")
                else:
                    log(f"  [!] {marker_id}: فشل التوليد — {getattr(result, 'error', '?')}")
            except Exception as e:
                log(f"  [!] {marker_id}: خطأ — {str(e)[:200]}")

        if not success:
            log(f"  [!] {marker_id}: فشلت كل المحاولات — استخدام النتيجة المقطوعة")

    log(f"  تم إصلاح {retry_count}/{len(truncated_indices)} نتيجة مقطوعة")
    return batch_results


def _batch_single_generate(gen_step, gen_idx, config, ctx, steps, is_primary=False):
    """إرسال خطوة generate واحدة كباتش — مع polling واستقبال النتائج"""
    system_prompt = ctx.resolve(gen_step.get("system_prompt", "")) if gen_step.get("system_prompt") else ""
    temperature = gen_step.get("temperature", 0.7)
    max_tokens = gen_step.get("max_tokens", 8192)
    thinking_budget = gen_step.get("thinking_budget", None)
    thinking_level = gen_step.get("thinking_level", None)
    effective_model = gen_step.get("model") or ctx.model

    if gen_step.get("model"):
        log(f"  [model override] {gen_step['model']}")

    # بناء البرومبتات
    topics = None
    marker_prefix = None

    if is_primary:
        # الخطوة الأساسية — محاولة تقسيم حسب المواضيع
        topics = _extract_topics_from_context(ctx)
        if topics and len(topics) > 1:
            marker_prefix = _detect_marker_prefix(config)
            prompts = _build_batch_prompts(config, ctx, topics, marker_prefix)
            if prompts:
                log(f"  [batch] تقسيم حسب المواضيع: {len(prompts)} طلب")
            else:
                topics = None  # fallback لبرومبت واحد
        else:
            topics = None  # موضوع واحد أو أقل — برومبت واحد

    marker_split_mode = False  # هل تم التقسيم حسب الماركرز؟

    if topics is None:
        # نحل المدخل من ctx
        prompt_str = str(ctx.resolve(gen_step["input"]))

        # محاولة تقسيم حسب الماركرز (SCRIPT/INTRO) — لتشكيل وغيره
        MARKER_PAT = r'<<<((?:SCRIPT|INTRO)_\d+)>>>'
        separator = "\n---\n"
        if separator in prompt_str:
            content_section = prompt_str[prompt_str.rfind(separator) + len(separator):]
        else:
            content_section = prompt_str

        seen = set()
        input_markers = []
        for m in re.findall(MARKER_PAT, content_section):
            if m not in seen:
                seen.add(m)
                input_markers.append(m)

        # فلترة الماركرز حسب TOPIC_IDS (لو محدد)
        if ctx.topic_ids and input_markers:
            filtered_markers = []
            for m in input_markers:
                id_match = re.search(r'_(\d+)', m)
                if id_match and int(id_match.group(1)) in ctx.topic_ids:
                    filtered_markers.append(m)
            if filtered_markers:
                log(f"  [batch] فلترة الماركرز: {len(filtered_markers)} من {len(input_markers)} (TOPIC_IDS)")
                input_markers = filtered_markers

        if len(input_markers) > 1:
            # تقسيم حسب الماركرز — كل ماركر في طلب لوحده
            content_start = prompt_str.rfind(separator) + len(separator) if separator in prompt_str else 0
            first_match = re.search(MARKER_PAT, content_section)
            instructions_part = prompt_str[:content_start + first_match.start()] if first_match else prompt_str
            content_with_markers = prompt_str[content_start:]

            prompts = []
            for marker in input_markers:
                escaped = re.escape(f'<<<{marker}>>>')
                pat = rf'({escaped}.*?)(?=<<<(?:SCRIPT|INTRO)_\d+>>>|\Z)'
                match = re.search(pat, content_with_markers, re.DOTALL)
                if match:
                    prompts.append(instructions_part + match.group(1).strip())

            marker_split_mode = True
            log(f"  [batch] تقسيم حسب الماركرز: {len(prompts)} طلب")
        else:
            prompts = [prompt_str]
            log(f"  [batch] برومبت واحد")

    save_path = ctx.output_path(f"batch_{gen_step['id']}.json")

    # Labels للتتبع في Google Cloud Billing
    _ba_labels = {"step": gen_step["id"]}
    if ctx.run_id:
        _ba_labels["run_id"] = ctx.run_id
    if ctx.recipe_name:
        _ba_labels["recipe"] = ctx.recipe_name
    if ctx.channel_name:
        _ba_labels["channel"] = ctx.channel_name

    log(f"--- إرسال باتش: {len(prompts)} طلب (model: {effective_model}) ---")
    send_result = batch_send(
        prompts=prompts,
        model=effective_model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        save_path=save_path,
        thinking_budget=thinking_budget,
        thinking_level=thinking_level,
        labels=_ba_labels,
    )

    if not send_result.success:
        raise EngineError(f"فشل إرسال الباتش: {send_result.error}", code="BATCH_SEND_FAILED")

    # انتظار واستقبال النتائج
    log(f"--- انتظار نتائج الباتش ({gen_step['id']}) ---")
    poll_interval = 30
    max_wait = 7200  # ساعتين — التشكيل بياخد وقت أطول
    start_time = time.time()

    batch_results = None
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise EngineError(
                f"تجاوز الوقت ({max_wait}s) في انتظار نتائج الباتش",
                code="BATCH_RETRIEVE_TIMEOUT"
            )

        try:
            result = batch_retrieve(batch_info_path=save_path)
            if result.success:
                batch_results = result.data
                log(f"  تم استقبال {len(batch_results)} نتيجة")
                break
        except EngineError as e:
            if e.code == "BATCH_JOB_NOT_READY":
                log(f"  المهمة لم تكتمل ({int(elapsed)}s). انتظار {poll_interval}s...")
                time.sleep(poll_interval)
                continue
            raise

    # تسجيل استهلاك التوكنز من الباتش
    if result.token_usage:
        ctx.record_usage(
            step_id=gen_step["id"],
            call_type="batch",
            provider=result.provider or "unknown",
            model=result.model or effective_model,
            token_usage=result.token_usage
        )

    # لو كان تقسيم حسب المواضيع → تجميع + retry المقطوعة
    if topics and len(topics) > 1 and marker_prefix:
        metadata = {
            "generate_step_id": gen_step["id"],
            "generate_step_index": gen_idx,
            "marker_prefix": marker_prefix,
            "topics": topics,
            "batch_info_path": save_path,
            "pre_results": {k: v for k, v in ctx.results.items() if isinstance(v, str)},
        }

        batch_results = _retry_truncated_batch_results(
            batch_results, topics, marker_prefix, config, ctx, metadata
        )

        combined = _reassemble_batch_results(batch_results, topics, marker_prefix)
        ctx.results[gen_step["id"]] = combined
        log(f"  تم تجميع النتائج ({len(combined)} حرف)")
    elif marker_split_mode:
        # كشف وإعادة توليد النتائج المقطوعة (MAX_TOKENS) في marker_split_mode
        batch_results = _retry_truncated_marker_split(
            batch_results, prompts, gen_step, ctx, save_path
        )

        # تقسيم حسب الماركرز — تجميع النتائج بالـ ID (الباتش مش بيضمن الترتيب)
        MARKER_PAT_RE = re.compile(r'<<<((?:SCRIPT|INTRO)_(\d+))>>>')
        id_to_result = {}
        unmatched = []
        for r in batch_results:
            m = MARKER_PAT_RE.search(r)
            if m:
                id_to_result[int(m.group(2))] = r
            else:
                unmatched.append(r)
        # ترتيب حسب الـ ID الفعلي
        sorted_results = [id_to_result[k] for k in sorted(id_to_result.keys())]
        sorted_results.extend(unmatched)
        combined = "\n\n".join(sorted_results)
        ctx.results[gen_step["id"]] = combined
        log(f"  تم تجميع {len(sorted_results)} نتيجة ماركرز بالـ ID ({len(combined)} حرف)")
    else:
        # برومبت واحد — النتيجة مباشرة
        text = batch_results[0] if batch_results else ""
        ctx.results[gen_step["id"]] = text
        log(f"  النتيجة: {text[:100]}..." if len(text) > 100 else f"  النتيجة: {text}")


def _run_mode_batch_auto(config, ctx, steps):
    """وضع باتش أوتوماتيك: كل خطوات generate تشتغل باتش — في run واحد"""
    gen_indices = _find_all_generate_step_indices(steps)

    if not gen_indices:
        log("[!] لا توجد خطوة generate — تشغيل فوري")
        _run_steps(steps, ctx)
        return

    log(f"  [batch_auto] {len(gen_indices)} خطوة generate هتشتغل باتش")

    prev_end = 0
    for step_num, gen_idx in enumerate(gen_indices):
        # تنفيذ الخطوات قبل هذه الـ generate (فوري)
        if prev_end < gen_idx:
            _run_steps(steps, ctx, start=prev_end, end=gen_idx)

        # تنفيذ خطوة generate كـ batch
        gen_step = steps[gen_idx]
        step_label = gen_step.get("label", gen_step["id"])
        log(f"--- الخطوة {gen_idx + 1}/{len(steps)}: {step_label} [BATCH] ---")

        is_primary = (step_num == 0)  # أول generate = الأساسية (تقسيم حسب المواضيع)
        _batch_single_generate(gen_step, gen_idx, config, ctx, steps, is_primary=is_primary)

        prev_end = gen_idx + 1

    # تنفيذ الخطوات بعد آخر generate
    if prev_end < len(steps):
        _run_steps(steps, ctx, start=prev_end)


# ========== Pipeline Runner ==========

def run_pipeline(config):
    """تنفيذ الـ pipeline — بيفرّع حسب وضع التشغيل"""
    pipeline_name = config.get("name", "Unnamed Pipeline")
    log(f"========== بدء Pipeline: {pipeline_name} ==========")

    # التحقق
    errors = validate_pipeline(config)
    if errors:
        log(f"[X] أخطاء في الـ Pipeline:")
        for err in errors:
            log(f"  - {err}")
        sys.exit(1)

    ctx = PipelineContext()
    steps = config["steps"]

    # قراءة إعدادات التشغيل من _runtime (محقونة في recipe_config.json — أعلى أولوية)
    runtime = config.get("_runtime", {})
    if runtime:
        if runtime.get("execution_mode"):
            ctx.execution_mode = runtime["execution_mode"]
            log(f"  [runtime] execution_mode = {ctx.execution_mode} (من recipe_config.json)")
        if runtime.get("model_name"):
            ctx.model = runtime["model_name"]
        if runtime.get("tts_provider"):
            ctx.tts_provider = runtime["tts_provider"]
        if runtime.get("topic_ids"):
            os.environ["TOPIC_IDS"] = runtime["topic_ids"]
            ctx.topic_ids = ctx._parse_topic_ids()  # إعادة قراءة بعد تحديث البيئة

    mode = ctx.execution_mode

    log(f"عدد الخطوات: {len(steps)} | الموديل: {ctx.model} | TTS: {ctx.tts_provider} | الوضع: {mode}")

    if mode == "instant":
        _run_steps(steps, ctx)
    elif mode == "send_only":
        _run_mode_send_only(config, ctx, steps)
    elif mode == "receive_only":
        _run_mode_receive_only(config, ctx, steps)
    elif mode == "batch_auto":
        _run_mode_batch_auto(config, ctx, steps)
    else:
        raise EngineError(
            f"وضع تشغيل غير معروف: '{mode}'. الأوضاع المتاحة: instant, send_only, receive_only, batch_auto",
            code="UNKNOWN_EXECUTION_MODE"
        )

    # حفظ ملخص استهلاك التوكنز
    ctx.save_usage_summary()

    log(f"========== Pipeline اكتمل بنجاح: {pipeline_name} ==========")


# ========== Main ==========

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recipe_runner.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    run_pipeline(config)
