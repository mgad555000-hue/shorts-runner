"""
محرك الذكاء الصناعي - Shorts Runner Engine
=============================================
مكتبة موحدة للتعامل مع كل موديلات الذكاء الصناعي.
الوصفات بتستدعي الدوال دي وبس - مش بتتعامل مع الموديلات مباشرة.

الدوال الثمانية:
1. generate()         - توليد نص من برومبت
2. batch_send()       - إرسال دفعة برومبتات
3. batch_retrieve()   - استقبال نتائج الدفعة
4. tts()              - تحويل نص لصوت (موحدة - بتقرأ المزود من البيئة)
5. tts_elevenlabs()   - تحويل نص لصوت (ElevenLabs)
6. tts_minimax()      - تحويل نص لصوت (MiniMax)
7. tts_vertex()       - تحويل نص لصوت (Vertex AI)
8. transcribe()       - تحويل صوت لنص (Whisper)
"""

import os
import sys
import time
import json
import traceback
import threading
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

# ========== نظام النتائج الموحد ==========

@dataclass
class EngineResult:
    """نتيجة موحدة لكل دوال المكتبة"""
    success: bool
    data: Any = None           # النتيجة (نص، بايتات صوت، dict، قائمة)
    error: str = ""            # رسالة الخطأ بالعربي
    error_code: str = ""       # كود الخطأ (للبرمجة)
    model: str = ""            # الموديل المستخدم
    provider: str = ""         # المزود (gemini, openai, claude, glm, elevenlabs, vertex, whisper)
    attempts: int = 0          # عدد المحاولات
    duration_ms: int = 0       # مدة التنفيذ
    token_usage: Dict = field(default_factory=dict)  # {"input": 0, "output": 0, "thinking": 0, "total": 0}


@dataclass
class BatchInfo:
    """معلومات مهمة Batch موحدة"""
    provider: str              # gemini, claude
    model: str
    job_id: str                # المعرف الموحد
    job_name: str = ""         # الاسم الكامل (لو مختلف)
    item_order: List[Any] = field(default_factory=list)  # ترتيب العناصر
    items_count: int = 0
    created_at: str = ""
    status: str = "submitted"
    extra: Dict = field(default_factory=dict)  # بيانات إضافية حسب المزود

    def save(self, path: str):
        """حفظ في ملف JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BatchInfo':
        """قراءة من ملف JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


# اسم ملف Batch الثابت (موحد لكل الوصفات)
BATCH_INFO_FILENAME = "batch_job_info.json"


# ========== نظام السجلات ==========

def log(msg: str):
    """تسجيل رسالة مع الوقت"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    text = f"[{timestamp}] [ENGINE] {msg}"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_text = text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")
        print(safe_text, flush=True)


# ========== فحوصات ما قبل الإرسال ==========

def _check_api_key(key_name: str) -> str:
    """التأكد من وجود مفتاح API وإرجاعه"""
    key = os.getenv(key_name, "")
    if not key:
        raise EngineError(
            f"مفتاح {key_name} غير موجود. تأكد من ملف .env",
            code="MISSING_API_KEY"
        )
    return key


def _check_prompt(prompt: str) -> str:
    """التأكد من أن البرومبت مش فاضي"""
    if not prompt or not prompt.strip():
        raise EngineError(
            "البرومبت فاضي. لازم تكتب تعليمات للذكاء الصناعي",
            code="EMPTY_PROMPT"
        )
    return prompt.strip()


def _check_model(model: str) -> str:
    """التأكد من أن اسم الموديل مش فاضي"""
    if not model or not model.strip():
        raise EngineError(
            "اسم الموديل غير محدد. اختر موديل من القائمة",
            code="EMPTY_MODEL"
        )
    return model.strip()


def _check_text_for_tts(text: str) -> str:
    """التأكد من أن النص مش فاضي لتحويله لصوت"""
    if not text or not text.strip():
        raise EngineError(
            "النص فاضي. لازم يكون فيه نص لتحويله لصوت",
            code="EMPTY_TTS_TEXT"
        )
    return text.strip()


def _check_audio_file(file_path: str) -> str:
    """التأكد من وجود ملف الصوت"""
    if not file_path:
        raise EngineError(
            "مسار ملف الصوت غير محدد",
            code="EMPTY_AUDIO_PATH"
        )
    if not os.path.exists(file_path):
        raise EngineError(
            f"ملف الصوت غير موجود: {file_path}",
            code="AUDIO_FILE_NOT_FOUND"
        )
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise EngineError(
            f"ملف الصوت فاضي (0 bytes): {file_path}",
            code="AUDIO_FILE_EMPTY"
        )
    return file_path


def _check_batch_prompts(prompts: list) -> list:
    """التأكد من قائمة البرومبتات"""
    if not prompts:
        raise EngineError(
            "قائمة البرومبتات فاضية",
            code="EMPTY_BATCH_PROMPTS"
        )
    if not isinstance(prompts, list):
        raise EngineError(
            "البرومبتات لازم تكون قائمة",
            code="INVALID_BATCH_PROMPTS"
        )
    # تأكد كل عنصر مش فاضي
    empty_indices = [i for i, p in enumerate(prompts) if not p or not str(p).strip()]
    if empty_indices:
        raise EngineError(
            f"فيه {len(empty_indices)} برومبت فاضي في القائمة (الأرقام: {empty_indices[:5]})",
            code="EMPTY_ITEMS_IN_BATCH"
        )
    return prompts


def _check_batch_info(info) -> 'BatchInfo':
    """التأكد من صحة معلومات الـ Batch"""
    if info is None:
        raise EngineError(
            f"ملف {BATCH_INFO_FILENAME} غير موجود. شغّل وصفة الإرسال الأول",
            code="BATCH_INFO_NOT_FOUND"
        )
    if isinstance(info, str):
        # مسار ملف
        if not os.path.exists(info):
            raise EngineError(
                f"ملف {info} غير موجود",
                code="BATCH_INFO_FILE_NOT_FOUND"
            )
        info = BatchInfo.load(info)
    if not info.job_id and not info.job_name:
        raise EngineError(
            "معلومات الـ Batch ناقصة - مفيش رقم مهمة",
            code="BATCH_INFO_NO_JOB_ID"
        )
    return info


def _check_response_text(text: str, model: str) -> str:
    """التأكد من أن الرد فيه محتوى فعلي"""
    if text is None:
        raise EngineError(
            f"الموديل {model} رجّع رد فاضي (None)",
            code="EMPTY_RESPONSE"
        )
    text = str(text).strip()
    if not text:
        raise EngineError(
            f"الموديل {model} رجّع رد فاضي",
            code="EMPTY_RESPONSE"
        )
    # فحص لو الرد هو رسالة خطأ مش محتوى
    error_indicators = ["i cannot", "i can't", "error:", "exception:", "traceback"]
    lower_text = text[:200].lower()
    for indicator in error_indicators:
        if lower_text.startswith(indicator):
            log(f"[!] الرد يبدو كرسالة خطأ: {text[:100]}...")
            # مش نرفضه - ممكن يكون محتوى عادي، بس نسجل تحذير
            break
    return text


def _check_audio_data(data: bytes, source: str) -> bytes:
    """التأكد من أن بيانات الصوت مش فاضية"""
    if data is None or len(data) == 0:
        # Exception عادية (مش EngineError) عشان _retry_call يعيد المحاولة
        raise Exception(f"ملف الصوت فاضي من {source} — سيُعاد المحاولة")
    if len(data) < 100:
        # Exception عادية عشان _retry_call يعيد المحاولة (Vertex بترجع ملف صغير لما تكون مشغولة)
        raise Exception(f"ملف الصوت صغير جداً ({len(data)} bytes) من {source} — سيُعاد المحاولة")
    return data


# ========== نظام الأخطاء ==========

class EngineError(Exception):
    """خطأ من المكتبة مع كود خطأ"""
    def __init__(self, message: str, code: str = "UNKNOWN"):
        self.message = message
        self.code = code
        super().__init__(message)


# ========== نظام إعادة المحاولة ==========

# أخطاء تستحق إعادة المحاولة (مؤقتة)
RETRYABLE_ERRORS = [
    "rate_limit", "rate limit", "429", "quota",
    "timeout", "timed out", "deadline",
    "connection", "connect", "network",
    "503", "502", "500", "server error", "internal error",
    "overloaded", "capacity", "resource_exhausted",
    "disconnected", "reset by peer",
]

# أخطاء لا تستحق إعادة المحاولة (دائمة)
NON_RETRYABLE_ERRORS = [
    "401", "403", "invalid_api_key", "authentication",
    "not_found", "404", "model not found",
    "invalid_request", "400",
    "billing", "payment",
]


def _is_retryable(error: Exception) -> bool:
    """هل الخطأ مؤقت ويستحق إعادة المحاولة؟"""
    error_str = str(error).lower()
    # لو خطأ دائم، لا تعيد المحاولة
    for pattern in NON_RETRYABLE_ERRORS:
        if pattern in error_str:
            return False
    # لو خطأ مؤقت، أعد المحاولة
    for pattern in RETRYABLE_ERRORS:
        if pattern in error_str:
            return True
    # افتراضياً: أعد المحاولة
    return True


def _retry_call(func, max_retries: int = 3, base_delay: float = 2.0, description: str = ""):
    """تنفيذ دالة مع إعادة المحاولة عند الأخطاء المؤقتة"""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = func()
            if attempt > 1:
                log(f"[OK] نجح بعد {attempt} محاولات ({description})")
            return result
        except EngineError:
            raise  # أخطاء الفحوصات لا تعاد
        except Exception as e:
            last_error = e
            if not _is_retryable(e):
                log(f"[X] خطأ دائم ({description}): {str(e)[:200]}")
                raise EngineError(
                    f"خطأ من {description}: {str(e)[:500]}",
                    code="PERMANENT_ERROR"
                )
            if attempt < max_retries:
                wait = base_delay * (2 ** (attempt - 1))  # 2s, 4s, 8s
                log(f">> محاولة {attempt}/{max_retries} فشلت ({description}). إعادة بعد {wait:.0f}s: {str(e)[:100]}")
                time.sleep(wait)
            else:
                log(f"[X] فشل بعد {max_retries} محاولات ({description}): {str(e)[:200]}")

    raise EngineError(
        f"فشل بعد {max_retries} محاولات ({description}): {str(last_error)[:500]}",
        code="MAX_RETRIES_EXCEEDED"
    )


# ========== تحديد المزود من اسم الموديل ==========

def detect_provider(model: str) -> str:
    """تحديد المزود تلقائياً من اسم الموديل"""
    model_lower = model.lower()
    # Vertex AI: استخدم بادئة "vertex:" مثل vertex:gemini-2.5-pro
    if model_lower.startswith("vertex:"):
        return "vertex"
    elif model_lower.startswith("gemini"):
        return "gemini"
    elif model_lower.startswith("gpt-") or model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
        return "openai"
    elif model_lower.startswith("claude"):
        return "claude"
    elif model_lower.startswith("glm"):
        return "glm"
    else:
        raise EngineError(
            f"موديل غير معروف: {model}. الموديلات المدعومة: gemini, gpt, claude, glm, vertex:gemini-*",
            code="UNKNOWN_MODEL"
        )


def _normalize_thinking_level(value: str = None) -> Optional[str]:
    value = (value or "").strip().lower()
    return value if value in {"none", "low", "medium", "high"} else None


def _is_gemini_3_model(model: str = "") -> bool:
    return "gemini-3" in (model or "").lower()


def _gemini_3_supports_minimal_thinking(model: str = "") -> bool:
    model_lower = (model or "").lower()
    return "flash" in model_lower or "lite" in model_lower


def _gemini_3_thinking_level(model: str, thinking_level: str = None) -> Optional[str]:
    level = _normalize_thinking_level(thinking_level)
    if not level:
        return None
    if level == "none":
        return "minimal" if _gemini_3_supports_minimal_thinking(model) else None
    return level


def _apply_ui_thinking_override(thinking_budget: int = None, thinking_level: str = None):
    """UI/env thinking level is the global source of truth for all recipes."""
    env_level = _normalize_thinking_level(os.getenv("THINKING_LEVEL", ""))
    if env_level:
        return None, env_level
    return thinking_budget, _normalize_thinking_level(thinking_level)


def _batch_split_enabled() -> bool:
    return os.getenv("BATCH_AUTO_SPLIT", "true").lower() not in ("0", "false", "no", "off")


def _batch_split_limits():
    max_requests = int(os.getenv("BATCH_CHUNK_MAX_REQUESTS", "250") or "250")
    max_mb = float(os.getenv("BATCH_CHUNK_MAX_MB", "20") or "20")
    return max(1, max_requests), max(1, int(max_mb * 1024 * 1024))


def _split_prompts_for_batch(prompts: list):
    """Split very large batches so one provider-side failure does not kill the whole run."""
    if not _batch_split_enabled():
        return [(list(range(len(prompts))), prompts)]

    max_requests, max_bytes = _batch_split_limits()
    chunks = []
    current_indexes = []
    current_prompts = []
    current_bytes = 0

    for idx, prompt in enumerate(prompts):
        prompt_size = len(str(prompt).encode("utf-8")) + 4096
        should_flush = (
            current_prompts
            and (
                len(current_prompts) >= max_requests
                or current_bytes + prompt_size > max_bytes
            )
        )
        if should_flush:
            chunks.append((current_indexes, current_prompts))
            current_indexes = []
            current_prompts = []
            current_bytes = 0

        current_indexes.append(idx)
        current_prompts.append(prompt)
        current_bytes += prompt_size

    if current_prompts:
        chunks.append((current_indexes, current_prompts))
    return chunks


def _safe_gcp_label_value(value: str) -> str:
    import hashlib
    import re

    original = str(value or "").lower()
    clean = re.sub(r"[^a-z0-9_-]", "_", original).strip("_-")[:63].strip("_-")
    if clean:
        return clean
    return f"v-{hashlib.sha1(str(value).encode('utf-8')).hexdigest()[:12]}"


def _safe_gcp_label_key(value: str) -> str:
    import re

    clean = re.sub(r"[^a-z0-9_-]", "_", str(value or "").lower()).strip("_-")[:63].strip("_-")
    if not clean or not re.match(r"^[a-z]", clean):
        clean = f"k_{clean}" if clean else "k_label"
    return clean[:63].strip("_-") or "k_label"


def _sanitize_gcp_labels(labels: dict = None) -> dict:
    if not labels:
        return {}
    job_labels = {}
    for key, value in labels.items():
        clean_key = _safe_gcp_label_key(key)
        clean_val = _safe_gcp_label_value(value)
        if clean_key and clean_val:
            job_labels[clean_key] = clean_val
    return job_labels


def _safe_vertex_display_name(run_id: str = "", chunk_suffix: str = "", timestamp: str = "") -> str:
    base_id = (run_id or "batch")[:8]
    suffix = chunk_suffix.replace("_", "-") if chunk_suffix else ""
    name = f"mgr-{base_id}{suffix}-{timestamp}"
    return name[:120]


def _vertex_batch_location_for_model(model: str, configured_location: str = "") -> str:
    model_lower = (model or "").lower()
    location = configured_location or "us-central1"
    if "gemini-3" in model_lower:
        return "global"
    return location


def _strict_google_cost_tracking_enabled() -> bool:
    """Strict mode: a Google run must be attributable by run_id in Billing Export."""
    return os.getenv("STRICT_GOOGLE_COST_TRACKING", "true").lower() not in ("0", "false", "no", "off")


def _block_google_direct_for_cost_tracking() -> bool:
    """Optional safety switch for users who want to forbid Google direct calls entirely."""
    return os.getenv("BLOCK_GOOGLE_DIRECT_FOR_COST_TRACKING", "false").lower() in ("1", "true", "yes", "on")


def _gemini_cache_enabled() -> bool:
    """Explicit Gemini cache is opt-in because it can increase real cost for small/verbose runs."""
    return os.getenv("ENABLE_GEMINI_CACHE", "false").lower() in ("1", "true", "yes", "on")


def _has_run_label(labels: dict = None) -> bool:
    return bool((labels and labels.get("run_id")) or os.getenv("RUN_ID"))


def _enforce_no_google_direct(provider: str, labels: dict = None):
    if _block_google_direct_for_cost_tracking() and _has_run_label(labels) and provider in ("gemini", "vertex"):
        raise EngineError(
            "COST_TRACKING_REQUIRES_BATCH: Google labels لا تظهر في فواتير التشغيل الفوري. "
            "هذا الرن مضبوط على منع Google direct صراحة. استخدم batch/vertex فقط أو أوقف BLOCK_GOOGLE_DIRECT_FOR_COST_TRACKING.",
            code="COST_TRACKING_REQUIRES_BATCH",
        )


# ========== تحديد مفتاح API من المزود ==========

def _get_api_key_for_provider(provider: str) -> str:
    """جلب مفتاح API الصحيح للمزود"""
    key_map = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "CLAUDE_API_KEY",
        "glm": "GLM_API_KEY",
    }
    key_name = key_map.get(provider)
    if not key_name:
        raise EngineError(f"مزود غير معروف: {provider}", code="UNKNOWN_PROVIDER")
    return _check_api_key(key_name)


# ========== الدوال الستة الرئيسية (هيكل فارغ - يتملأ في المراحل التالية) ==========

def _clean_label_value(s: str) -> str:
    """تنظيف قيمة label حسب شروط Google Cloud:
    أحرف صغيرة + أرقام + شرطة + شرطة سفلية فقط، حد أقصى 63 حرف.
    """
    if not s:
        return ""
    import re as _re
    return _re.sub(r'[^a-z0-9_-]', '_', str(s).lower())[:63]


def _build_clean_labels(labels: dict) -> dict:
    """بناء dict من labels نظيفة تطابق شروط Google Cloud.
    يتجاهل القيم اللي بقت كلها underscores بعد التنظيف (مثلاً اسم عربي بحت)
    لتجنب labels غير مفيدة في BigQuery."""
    if not labels:
        return {}
    out = {}
    for k, v in labels.items():
        ck = _clean_label_value(k)
        cv = _clean_label_value(v)
        if not ck or not cv:
            continue
        # رفض القيم اللي كلها underscore (محصلة تنظيف نص غير ASCII فقط)
        if all(c == '_' for c in cv):
            continue
        out[ck] = cv
    return out


def generate(prompt: str, model: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = None, thinking_budget: int = None, thinking_level: str = None, labels: dict = None, cache_content: str = None) -> EngineResult:
    """
    الدالة 1: توليد نص من برومبت
    تدعم: Gemini, OpenAI, Claude, GLM, Vertex AI

    ملاحظة: لاستخدام Vertex AI، استخدم "vertex:gemini-2.5-pro" مثلاً

    labels: dict اختياري — يُرسل لـ Google Cloud Billing عبر Gemini API
            (مثل {"run_id": "abc123"}) لتتبع التكلفة الفعلية لكل تشغيلة في BigQuery.

    cache_content: نص اختياري للـ caching (Gemini فقط) — لو مُحدّد، يتم
            إنشاء/استخدام explicit cache لتقليل تكلفة input tokens.
            الـ cache يُعاد استخدامه تلقائياً لو نفس المحتوى تم تخزينه قبل كده.
    """
    start_time = time.time()

    # فحوصات ما قبل الإرسال
    prompt = _check_prompt(prompt)
    model = _check_model(model)
    provider = detect_provider(model)
    thinking_budget, thinking_level = _apply_ui_thinking_override(thinking_budget, thinking_level)
    _enforce_no_google_direct(provider, labels)
    if cache_content and not _gemini_cache_enabled():
        log("  [cache] OFF: ENABLE_GEMINI_CACHE is not enabled")
        cache_content = None

    # Vertex AI مش بيحتاج API key - بيستخدم Google Cloud credentials
    api_key = None if provider == "vertex" else _get_api_key_for_provider(provider)

    log(f"-> generate | model: {model} | provider: {provider} | prompt: {len(prompt)} chars" + (" | cache=ON" if cache_content else ""))

    try:
        token_usage = {"input": 0, "output": 0, "thinking": 0, "total": 0}

        if provider == "gemini":
            result_tuple = _retry_call(
                lambda: _generate_gemini(prompt, model, api_key, system_prompt, temperature, max_tokens, thinking_budget, thinking_level, labels=labels, cache_content=cache_content),
                max_retries=3, base_delay=3.0, description=f"Gemini {model}"
            )
        elif provider == "vertex":
            # استخرج اسم الموديل الفعلي (بعد "vertex:")
            actual_model = model.split(":", 1)[1] if ":" in model else model
            result_tuple = _retry_call(
                lambda: _generate_vertex(prompt, actual_model, system_prompt, temperature, max_tokens, thinking_budget, thinking_level, labels=labels, cache_content=cache_content),
                max_retries=3, base_delay=3.0, description=f"Vertex AI {actual_model}"
            )
        elif provider == "openai":
            result_tuple = _retry_call(
                lambda: _generate_openai(prompt, model, api_key, system_prompt, temperature, max_tokens),
                max_retries=3, base_delay=3.0, description=f"OpenAI {model}"
            )
        elif provider == "claude":
            result_tuple = _retry_call(
                lambda: _generate_claude(prompt, model, api_key, system_prompt, temperature, max_tokens, thinking_budget, thinking_level),
                max_retries=3, base_delay=3.0, description=f"Claude {model}"
            )
        elif provider == "glm":
            result_tuple = _retry_call(
                lambda: _generate_glm(prompt, model, api_key, system_prompt, temperature, max_tokens),
                max_retries=3, base_delay=3.0, description=f"GLM {model}"
            )
        else:
            raise EngineError(f"مزود غير مدعوم: {provider}", code="UNSUPPORTED_PROVIDER")

        # فك الـ tuple (text, token_usage)
        if isinstance(result_tuple, tuple):
            text, token_usage = result_tuple
        else:
            text = result_tuple

        # فحص النتيجة
        text = _check_response_text(text, model)

        duration = int((time.time() - start_time) * 1000)
        log(f"<- generate OK | {len(text)} chars | {duration}ms")

        return EngineResult(
            success=True, data=text, model=model, provider=provider,
            duration_ms=duration, token_usage=token_usage
        )

    except EngineError:
        raise
    except Exception as e:
        duration = int((time.time() - start_time) * 1000)
        raise EngineError(
            f"خطأ غير متوقع من {provider}/{model}: {str(e)[:500]}",
            code="UNEXPECTED_ERROR"
        )


# ========== دوال الربط الفعلية (منسوخة من الكود المجرب) ==========

# ========== Gemini Explicit Cache Management ==========
import hashlib as _hashlib
_GEMINI_CACHE_MAP = {}  # {(model, content_hash): cache_name} — process-level reuse
_GEMINI_CACHE_LOCK = threading.Lock()


def _get_or_create_gemini_cache(client, model: str, content: str, ttl_seconds: int = 3600):
    """يجيب cache موجود أو ينشئ جديد لنفس المحتوى.
    يرجع cache.name (string) أو None لو فشل الإنشاء أو المحتوى أصغر من الحد الأدنى.

    حدود Gemini Cache (تم تحديدها مسبقاً لتجنب API calls فاشلة):
    - Gemini 2.5/3.x Pro: ≥ 4,096 توكن
    - Gemini 2.5/3.x Flash: ≥ 1,024 توكن
    """
    from google.genai import types

    # فحص استباقي للحد الأدنى — يوفر API call فاشل
    # حدود Google الفعلية: Pro=2048، Flash=1024
    is_flash = 'flash' in model.lower()
    min_tokens_required = 1024 if is_flash else 2048
    # تقدير التوكنز عبر count_tokens API call (دقيق + مجاني)
    try:
        token_count = client.models.count_tokens(model=model, contents=content).total_tokens
    except Exception:
        # fallback: تقدير من حجم النص (Arabic ~2.5 chars/token)
        token_count = len(content) // 2

    if token_count < min_tokens_required:
        log(f"  [cache] skip: {token_count} توكن < الحد الأدنى {min_tokens_required} لـ {model}")
        return None

    h = _hashlib.md5(f"{model}:{content}".encode('utf-8')).hexdigest()
    map_key = (model, h)

    with _GEMINI_CACHE_LOCK:
        # حاول استخدم cache موجود
        if map_key in _GEMINI_CACHE_MAP:
            cache_name = _GEMINI_CACHE_MAP[map_key]
            try:
                client.caches.get(name=cache_name)
                log(f"  [cache] reusing {cache_name[-12:]} (hit)")
                return cache_name
            except Exception:
                del _GEMINI_CACHE_MAP[map_key]

        # أنشئ cache جديد
        try:
            cache = client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    contents=[content],
                    ttl=f'{ttl_seconds}s',
                )
            )
            _GEMINI_CACHE_MAP[map_key] = cache.name
            log(f"  [cache] created {cache.name[-12:]} ({token_count} توكن، ttl={ttl_seconds}s)")
            return cache.name
        except Exception as e:
            log(f"  [cache] FAILED to create: {str(e)[:200]}")
            return None


def _make_vertex_client():
    """Create a Google Gen AI client that talks to Vertex AI."""
    import json
    import tempfile
    from google import genai

    project_id = (
        os.getenv("GOOGLE_QUOTA_PROJECT")
        or os.getenv("VERTEX_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or ""
    )
    if not project_id:
        raise ValueError("GOOGLE_QUOTA_PROJECT/VERTEX_PROJECT_ID غير محدد في .env")

    location = os.getenv("VERTEX_GENERATION_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "global"

    creds = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
        "refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN", ""),
        "quota_project_id": project_id,
        "type": "authorized_user",
        "universe_domain": "googleapis.com",
    }
    if creds["client_id"] and creds["client_secret"] and creds["refresh_token"]:
        creds_file = os.path.join(tempfile.gettempdir(), "gcp_creds_generate.json")
        with open(creds_file, "w") as f:
            json.dump(creds, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file
    elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise ValueError("بيانات اعتماد Vertex AI ناقصة. تأكد من GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN")

    return genai.Client(vertexai=True, project=project_id, location=location), project_id, location


def ensure_gemini_cache(model: str, cache_content: str, ttl_seconds: int = 3600):
    """Pre-create Gemini explicit cache before parallel direct calls."""
    if not _gemini_cache_enabled():
        log("  [cache] OFF: ENABLE_GEMINI_CACHE is not enabled")
        return None
    if not cache_content or len(cache_content) <= 4000:
        return None
    provider = detect_provider(model)
    _enforce_no_google_direct(provider, {"run_id": os.getenv("RUN_ID")} if os.getenv("RUN_ID") else None)
    if provider == "gemini":
        from google import genai

        api_key = _get_api_key_for_provider("gemini")
        client = genai.Client(api_key=api_key)
        return _get_or_create_gemini_cache(client, model, cache_content, ttl_seconds)
    if provider == "vertex":
        client, _, _ = _make_vertex_client()
        actual_model = model.split(":", 1)[1] if ":" in model else model
        return _get_or_create_gemini_cache(client, actual_model, cache_content, ttl_seconds)
    return None


def _generate_gemini(prompt: str, model: str, api_key: str, system_prompt: str, temperature: float, max_tokens: int, thinking_budget: int = None, thinking_level: str = None, labels: dict = None, cache_content: str = None) -> str:
    """ربط Gemini عبر google.genai SDK الجديدة"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    config_params = {"temperature": temperature, "top_p": 0.95}
    if max_tokens:
        config_params["max_output_tokens"] = max_tokens

    # ===== Explicit Caching (Gemini فقط) =====
    cached_name = None
    if cache_content and len(cache_content) > 4000:  # min ~4K chars (~1K tokens)
        cached_name = _get_or_create_gemini_cache(client, model, cache_content)
        if cached_name:
            config_params['cached_content'] = cached_name
            # شيل cache_content من بداية الـ prompt لو موجود (لتجنب التكرار)
            if prompt.startswith(cache_content):
                prompt = prompt[len(cache_content):].lstrip()
                log(f"  [cache] stripped cached prefix; remaining prompt: {len(prompt)} chars")

    # Gemini Developer API rejects billing labels at request time.
    # Use the Vertex AI path ("vertex:gemini-*") when BigQuery label attribution is required.
    clean_labels = {}

    # تحكم في التفكير — thinking_budget=0 له الأولوية القصوى (إلغاء التفكير)
    is_gemini_3 = _is_gemini_3_model(model)
    if thinking_budget == 0:
        # إلغاء التفكير تماماً — Gemini 3.x يستخدم "none"، الباقي thinking_budget=0
        if is_gemini_3:
            gemini_level = _gemini_3_thinking_level(model, "none")
            if gemini_level:
                config_params["thinking_config"] = types.ThinkingConfig(thinking_level=gemini_level)
                log(f"  [thinking] Gemini 3 uses closest no-thinking level: {gemini_level}")
            else:
                log("  [thinking] Gemini 3 model does not support no-thinking; not forcing LOW")
        else:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            log(f"  [thinking] OFF (thinking_budget=0)")
    elif thinking_level and is_gemini_3:
        gemini_level = _gemini_3_thinking_level(model, thinking_level)
        if gemini_level:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_level=gemini_level)
        else:
            log("  [thinking] Gemini 3 model does not support no-thinking; not forcing LOW")
    elif thinking_budget is not None:
        config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    elif thinking_level and not is_gemini_3:
        # fallback: حوّل thinking_level لـ thinking_budget للموديلات القديمة
        level_to_budget = {"none": 0, "low": 1024, "medium": 8192, "high": 24576}
        budget = level_to_budget.get(thinking_level.lower(), 1024)
        config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
    config = types.GenerateContentConfig(**config_params)
    # ملاحظة: لما تستخدم cached_content، system_instruction بيكون متخزّن جوّا الـ cache
    # فلازم نتخطّاه هنا لتجنب double-instruction
    if system_prompt and not cached_name:
        config.system_instruction = system_prompt

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        # احتياطي: لو موديل/endpoint قديم رفض labels، لا نفشل التشغيل بالكامل.
        if clean_labels and "label" in str(e).lower():
            log(f"  [labels] Direct labels rejected; retrying without labels: {str(e)[:160]}")
            config_params.pop("labels", None)
            config = types.GenerateContentConfig(**config_params)
            if system_prompt and not cached_name:
                config.system_instruction = system_prompt
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        else:
            raise

    if not response or not response.text:
        raise ValueError("Gemini returned empty response")

    # تسجيل واستخراج استهلاك التوكنز
    _token_usage = {"input": 0, "output": 0, "thinking": 0, "cached": 0, "total": 0}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        um = response.usage_metadata
        _token_usage["input"] = getattr(um, 'prompt_token_count', 0) or 0
        _token_usage["output"] = getattr(um, 'candidates_token_count', 0) or 0
        _token_usage["thinking"] = getattr(um, 'thoughts_token_count', 0) or 0
        _token_usage["cached"] = getattr(um, 'cached_content_token_count', 0) or 0
        _token_usage["total"] = getattr(um, 'total_token_count', 0) or 0
        cached_str = f" cached={_token_usage['cached']}" if _token_usage['cached'] else ""
        log(f"  [tokens] input={_token_usage['input']} output={_token_usage['output']} thinking={_token_usage['thinking']}{cached_str} total={_token_usage['total']}")

    # كشف القطع — لو finish_reason=MAX_TOKENS يبقى المخرج ناقص أكيد
    if response.candidates and response.candidates[0].finish_reason:
        fr = str(response.candidates[0].finish_reason)
        if fr == "MAX_TOKENS":
            raise ValueError(f"Gemini truncated (MAX_TOKENS) — {len(response.text)} chars returned")

    return response.text, _token_usage


def _generate_openai(prompt: str, model: str, api_key: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
    """ربط OpenAI عبر openai SDK"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # بعض الموديلات (gpt-5-nano, o-series) مش بتقبل temperature غير 1
    call_params = {
        "model": model,
        "messages": messages,
    }
    # max_tokens بس لو محدد
    if max_tokens:
        is_new_model = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4")
        if is_new_model:
            call_params["max_completion_tokens"] = max_tokens
        else:
            call_params["max_tokens"] = max_tokens
    # نحط temperature بس لو الموديل بيدعمه
    is_no_temp = model.startswith("gpt-5-nano") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4")
    if not is_no_temp:
        call_params["temperature"] = temperature

    response = client.chat.completions.create(**call_params)

    if not response.choices or not response.choices[0].message.content:
        raise ValueError("OpenAI returned empty response")

    # استخراج التوكنز
    _token_usage = {"input": 0, "output": 0, "thinking": 0, "cached": 0, "total": 0}
    if hasattr(response, 'usage') and response.usage:
        _token_usage["input"] = getattr(response.usage, 'prompt_tokens', 0) or 0
        _token_usage["output"] = getattr(response.usage, 'completion_tokens', 0) or 0
        _token_usage["total"] = _token_usage["input"] + _token_usage["output"]
        log(f"  [tokens] input={_token_usage['input']} output={_token_usage['output']} total={_token_usage['total']}")

    return response.choices[0].message.content, _token_usage


def _generate_claude(prompt: str, model: str, api_key: str, system_prompt: str, temperature: float, max_tokens: int, thinking_budget: int = None, thinking_level: str = None) -> str:
    """ربط Claude عبر anthropic SDK"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    final_max_tokens = max_tokens or 16384

    kwargs = {
        "model": model,
        "max_tokens": final_max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    # Extended thinking — Sonnet 4.6
    # نستخدم enabled mode بـ budget محدد (لا adaptive) لأن adaptive بيستهلك كل max_tokens المتاح
    # القاعدة الذهبية: max_tokens = budget + مساحة output كافية (لا تتجاوز 16K بدون داعي)
    if thinking_level and thinking_level.lower() != "none":
        level_map = {"low": 2048, "medium": 5000, "high": 10000}
        budget = level_map.get(thinking_level.lower(), 5000)
        # نضمن مساحة output كافية بعد thinking (4K توكن على الأقل)
        if final_max_tokens < budget + 4096:
            kwargs["max_tokens"] = budget + 4096
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        kwargs["temperature"] = 1.0
        log(f"  [thinking] Claude enabled budget={budget} (level={thinking_level}) max_tokens={kwargs['max_tokens']}")
    elif thinking_budget is not None and thinking_budget > 0:
        if thinking_budget >= final_max_tokens:
            kwargs["max_tokens"] = thinking_budget + 4096
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        kwargs["temperature"] = 1.0
        log(f"  [thinking] Claude enabled budget={thinking_budget} max_tokens={kwargs['max_tokens']}")

    # Streaming إجباري لما thinking مفعل + max_tokens كبير (SDK يرفض غير streaming لو متوقع > 10 دقائق)
    use_stream = "thinking" in kwargs and kwargs["max_tokens"] >= 8192

    if use_stream:
        log(f"  [stream] Claude streaming mode (thinking + max_tokens={kwargs['max_tokens']})")
        with client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
    else:
        message = client.messages.create(**kwargs)

    text_out = ""
    block_types = []
    for block in (message.content or []):
        bt = getattr(block, "type", None)
        block_types.append(bt)
        if bt == "text":
            text_out += getattr(block, "text", "") or ""
    stop_reason = getattr(message, "stop_reason", None)
    if not text_out:
        log(f"  [debug] empty text — stop_reason={stop_reason} blocks={block_types}")
        raise ValueError(f"Claude returned empty response (stop_reason={stop_reason}, blocks={block_types})")

    _token_usage = {"input": 0, "output": 0, "thinking": 0, "cached": 0, "total": 0}
    if hasattr(message, 'usage') and message.usage:
        _token_usage["input"] = getattr(message.usage, 'input_tokens', 0) or 0
        _token_usage["output"] = getattr(message.usage, 'output_tokens', 0) or 0
        _token_usage["thinking"] = getattr(message.usage, 'cache_creation_input_tokens', 0) or 0
        _token_usage["total"] = _token_usage["input"] + _token_usage["output"]
        log(f"  [tokens] input={_token_usage['input']} output={_token_usage['output']} total={_token_usage['total']}")

    return text_out, _token_usage


def _generate_glm(prompt: str, model: str, api_key: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
    """ربط GLM عبر REST API"""
    import httpx

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = httpx.post(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={k: v for k, v in {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }.items() if v is not None},
        timeout=180.0,
    )
    response.raise_for_status()
    data = response.json()

    if not data.get("choices") or not data["choices"][0].get("message", {}).get("content"):
        raise ValueError("GLM returned empty response")

    # استخراج التوكنز
    _token_usage = {"input": 0, "output": 0, "thinking": 0, "cached": 0, "total": 0}
    usage = data.get("usage", {})
    if usage:
        _token_usage["input"] = usage.get("prompt_tokens", 0)
        _token_usage["output"] = usage.get("completion_tokens", 0)
        _token_usage["total"] = usage.get("total_tokens", 0) or (_token_usage["input"] + _token_usage["output"])
        log(f"  [tokens] input={_token_usage['input']} output={_token_usage['output']} total={_token_usage['total']}")

    return data["choices"][0]["message"]["content"], _token_usage


def _generate_vertex(prompt: str, model: str, system_prompt: str, temperature: float, max_tokens: int, thinking_budget: int = None, thinking_level: str = None, labels: dict = None, cache_content: str = None) -> str:
    """ربط Vertex AI لتوليد النصوص (من الوثيقة - طريقة 6)"""
    import json
    import tempfile
    from google import genai
    from google.genai import types

    # جلب بيانات الاعتماد من المتغيرات البيئية
    project_id = os.getenv("GOOGLE_QUOTA_PROJECT", "")
    if not project_id:
        raise ValueError("GOOGLE_QUOTA_PROJECT غير محدد في .env")

    location = os.getenv("VERTEX_GENERATION_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "global"

    # إعداد بيانات الاعتماد
    creds = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
        "refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN", ""),
        "quota_project_id": project_id,
        "type": "authorized_user",
        "universe_domain": "googleapis.com"
    }

    # التحقق من وجود البيانات الأساسية
    if not creds["client_id"] or not creds["client_secret"] or not creds["refresh_token"]:
        raise ValueError("بيانات اعتماد Vertex AI ناقصة. تأكد من GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN")

    # حفظ بيانات الاعتماد في ملف مؤقت
    creds_file = os.path.join(tempfile.gettempdir(), 'gcp_creds_generate.json')
    with open(creds_file, 'w') as f:
        json.dump(creds, f)

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file

    # الاتصال بـ Vertex AI
    client = genai.Client(vertexai=True, project=project_id, location=location)

    # تكوين الطلب
    config_params = {"temperature": temperature}
    if max_tokens:
        config_params["max_output_tokens"] = max_tokens

    cached_name = None
    if cache_content and len(cache_content) > 4000:
        cached_name = _get_or_create_gemini_cache(client, model, cache_content)
        if cached_name:
            config_params["cached_content"] = cached_name
            if prompt.startswith(cache_content):
                prompt = prompt[len(cache_content):].lstrip()
                log(f"  [cache] stripped cached prefix; remaining prompt: {len(prompt)} chars")

    clean_labels = _build_clean_labels(labels)
    if clean_labels:
        config_params["labels"] = clean_labels
        log(f"  [labels] Vertex Direct Labels: {clean_labels}")

    is_gemini_3 = _is_gemini_3_model(model)
    if thinking_budget == 0 and is_gemini_3:
        gemini_level = _gemini_3_thinking_level(model, "none")
        if gemini_level:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_level=gemini_level)
            log(f"  [thinking] Gemini 3 uses closest no-thinking level: {gemini_level}")
        else:
            log("  [thinking] Gemini 3 model does not support no-thinking; not forcing LOW")
    elif thinking_budget == 0:
        config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        log("  [thinking] OFF (thinking_budget=0)")
    elif thinking_level and is_gemini_3:
        gemini_level = _gemini_3_thinking_level(model, thinking_level)
        if gemini_level:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_level=gemini_level)
        else:
            log("  [thinking] Gemini 3 model does not support no-thinking; not forcing LOW")
    elif thinking_budget is not None:
        config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    elif thinking_level and not is_gemini_3:
        level_to_budget = {"none": 0, "low": 1024, "medium": 8192, "high": 24576}
        config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=level_to_budget.get(thinking_level.lower(), 1024))

    config = types.GenerateContentConfig(**config_params)
    if system_prompt and not cached_name:
        config.system_instruction = system_prompt

    # توليد النص
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )

    if not response or not response.text:
        raise ValueError("Vertex AI returned empty response")

    # استخراج التوكنز
    _token_usage = {"input": 0, "output": 0, "thinking": 0, "cached": 0, "total": 0}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        um = response.usage_metadata
        _token_usage["input"] = getattr(um, 'prompt_token_count', 0) or 0
        _token_usage["output"] = getattr(um, 'candidates_token_count', 0) or 0
        _token_usage["thinking"] = getattr(um, 'thoughts_token_count', 0) or 0
        _token_usage["cached"] = getattr(um, 'cached_content_token_count', 0) or 0
        _token_usage["total"] = getattr(um, 'total_token_count', 0) or 0
        cached_str = f" cached={_token_usage['cached']}" if _token_usage['cached'] else ""
        log(f"  [tokens] input={_token_usage['input']} output={_token_usage['output']} thinking={_token_usage['thinking']}{cached_str} total={_token_usage['total']}")

    return response.text, _token_usage


# ========== دوال مساعدة لـ Google Cloud Storage ==========

def _setup_gcs_credentials(project_id: str = None) -> tuple:
    """إعداد بيانات اعتماد GCS وإرجاع (project_id, location, bucket_name)"""
    import json
    import tempfile

    project_id = project_id or os.getenv("GOOGLE_QUOTA_PROJECT", "")
    if not project_id:
        raise EngineError(
            "GOOGLE_QUOTA_PROJECT غير محدد في .env - مطلوب لـ Batch operations",
            code="MISSING_GCS_PROJECT"
        )

    location = os.getenv("GOOGLE_LOCATION", "us-central1")
    bucket_name = os.getenv("GOOGLE_GCS_BUCKET", "")

    if not bucket_name:
        raise EngineError(
            "GOOGLE_GCS_BUCKET غير محدد في .env - مطلوب لـ Batch operations",
            code="MISSING_GCS_BUCKET"
        )

    # إعداد بيانات الاعتماد
    creds = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
        "refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN", ""),
        "quota_project_id": project_id,
        "type": "authorized_user",
        "universe_domain": "googleapis.com"
    }

    if not creds["client_id"] or not creds["client_secret"] or not creds["refresh_token"]:
        raise EngineError(
            "بيانات اعتماد Google Cloud ناقصة. تأكد من: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN",
            code="MISSING_GCS_CREDENTIALS"
        )

    # حفظ بيانات الاعتماد في ملف مؤقت
    creds_file = os.path.join(tempfile.gettempdir(), f'gcp_creds_{os.getpid()}.json')
    with open(creds_file, 'w') as f:
        json.dump(creds, f)

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file
    os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
    os.environ['GCLOUD_PROJECT'] = project_id

    return project_id, location, bucket_name


def _upload_to_gcs(bucket_name: str, blob_name: str, content: str) -> str:
    """رفع محتوى نصي إلى GCS وإرجاع gs:// URI — يدعم ملفات كبيرة حتى 500MB+"""
    import io
    from google.cloud import storage

    project_id = os.getenv("GOOGLE_QUOTA_PROJECT", "")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    content_bytes = content.encode('utf-8')
    content_size_mb = len(content_bytes) / (1024 * 1024)

    # timeout ديناميكي: 300 ثانية كحد أدنى + 60 ثانية لكل 10MB
    upload_timeout = max(300, int(content_size_mb / 10 * 60) + 300)

    if content_size_mb > 5:
        # Resumable Upload بـ chunks 5MB — أفضل للملفات الكبيرة (> 5MB)
        # يتعامل مع انقطاعات الاتصال تلقائياً
        blob = bucket.blob(blob_name, chunk_size=5 * 1024 * 1024)
        log(f"  [GCS] رفع ملف كبير: {content_size_mb:.1f} MB (resumable, timeout={upload_timeout}s)")
        blob.upload_from_file(
            io.BytesIO(content_bytes),
            content_type='application/json',
            size=len(content_bytes),
            timeout=upload_timeout,
        )
    else:
        # رفع بسيط للملفات الصغيرة (< 5MB)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type='application/json', timeout=upload_timeout)

    log(f"  [GCS] ✅ تم الرفع: {content_size_mb:.1f} MB → gs://{bucket_name}/{blob_name}")
    return f"gs://{bucket_name}/{blob_name}"


def _download_from_gcs(gcs_uri: str) -> str:
    """تنزيل محتوى من GCS — يدعم ملف واحد أو مجلد (prefix)"""
    from google.cloud import storage

    # استخراج bucket و blob من URI
    path = gcs_uri.replace("gs://", "")
    bucket_name = path.split("/")[0]
    blob_name = "/".join(path.split("/")[1:])

    project_id = os.getenv("GOOGLE_QUOTA_PROJECT", "")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    # محاولة تنزيل كملف واحد أولاً
    blob = bucket.blob(blob_name)
    if blob.exists():
        return blob.download_as_text()

    # لو مش ملف → يبقى مجلد (prefix) — ننزّل كل الملفات جواه
    prefix = blob_name.rstrip("/") + "/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise EngineError(
            f"لا يوجد ملفات في GCS: {gcs_uri}",
            code="GCS_NOT_FOUND"
        )

    log(f"  GCS prefix: {len(blobs)} ملف في {prefix}")
    all_content = []
    for b in sorted(blobs, key=lambda x: x.name):
        if b.size and b.size > 0:
            content = b.download_as_text()
            all_content.append(content)
            log(f"    ✓ {b.name.split('/')[-1]} ({b.size} bytes)")

    return "\n".join(all_content)


def _batch_send_gemini(prompts: list, model: str, api_key: str, system_prompt: str, temperature: float, max_tokens: int, thinking_budget: int = None, thinking_level: str = None, labels: dict = None) -> BatchInfo:
    """إرسال دفعة عبر Gemini Batch API (يحتاج GCS) — يدعم حتى 1000+ موضوع"""
    import json
    from google import genai

    # إعداد GCS
    project_id, location, bucket_name = _setup_gcs_credentials()

    # بناء الطلبات بصيغة JSONL
    jsonl_lines = []
    for i, prompt in enumerate(prompts):
        request = {
            "contents": [{"parts": [{"text": prompt}], "role": "user"}],
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.95,
                "maxOutputTokens": max_tokens,
            }
        }
        # تحكم في التفكير — thinking_level لـ 3.x، thinking_budget لـ 2.x
        is_gemini_3 = _is_gemini_3_model(model)
        if thinking_level and is_gemini_3:
            gemini_level = _gemini_3_thinking_level(model, thinking_level)
            if gemini_level:
                request["generationConfig"]["thinkingConfig"] = {"thinkingLevel": gemini_level.upper()}
        elif thinking_budget is not None:
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        elif thinking_level and not is_gemini_3:
            level_to_budget = {"none": 0, "low": 1024, "medium": 8192, "high": 24576}
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": level_to_budget.get(thinking_level.lower(), 1024)}
        if system_prompt:
            request["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        # كل سطر في JSONL لازم يكون {"request": {...}}
        jsonl_lines.append(json.dumps({"request": request}, ensure_ascii=False))

    jsonl_content = "\n".join(jsonl_lines)
    payload_mb = len(jsonl_content.encode('utf-8')) / (1024 * 1024)
    log(f"  [payload] {len(prompts)} طلب | حجم JSONL: {payload_mb:.1f} MB")

    # رفع JSONL إلى GCS — مع run_id في المسار للتتبع
    # الرفع يتم مرة واحدة فقط قبل حلقة الـ retry لإنشاء الـ Job
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _rid = (labels or {}).get("run_id", "")
    _chunk = (labels or {}).get("chunk", "")
    _chunk_suffix = f"_part{_chunk}" if _chunk else ""
    if _rid:
        input_blob_name = f"batch_input/{_rid}/gemini_{timestamp}{_chunk_suffix}.jsonl"
    else:
        input_blob_name = f"batch_input/gemini_{timestamp}{_chunk_suffix}.jsonl"
    input_uri = _upload_to_gcs(bucket_name, input_blob_name, jsonl_content)

    log(f"  رفع الطلبات إلى: {input_uri}")

    # إنشاء الـ Batch Job — نجرب الـ location الأصلي، ولو NOT_FOUND نجرب global
    # display_name يحتوي على run_id الكامل + recipe للتتبع في Google Cloud Console
    run_id = (labels or {}).get("run_id", "")
    recipe = (labels or {}).get("recipe", "")
    channel = (labels or {}).get("channel", "")
    if run_id:
        job_name = f"mgr-{run_id}{_chunk_suffix}-{recipe[:20]}-{timestamp}" if recipe else f"mgr-{run_id}{_chunk_suffix}-{timestamp}"
    else:
        job_name = f"mgr-batch{_chunk_suffix}-{timestamp}"
    if run_id:
        log(f"  [RUN_ID] ✅ Run ID الكامل: {run_id}")
        log(f"  [RUN_ID] ✅ display_name في Google Cloud: {job_name}")
    locations_to_try = [location, "global"] if location != "global" else ["global"]
    last_err = None

    # تنظيف Labels حسب شروط Google Cloud (أحرف صغيرة، أرقام، شرطات فقط، max 63 chars)
    job_labels = _sanitize_gcp_labels(labels)
    if job_labels:
        log(f"  [labels] Gemini Batch Labels: {job_labels}")

    for loc in locations_to_try:
        try:
            client = genai.Client(vertexai=True, project=project_id, location=loc)
            batch_config = {'display_name': job_name}
            batch_job = client.batches.create(model=model, src=input_uri, config=batch_config)

            job_full_name = batch_job.name if hasattr(batch_job, 'name') else ""
            job_id = job_full_name.split('/')[-1] if job_full_name else job_name
            if loc != location:
                log(f"  ✓ نجح مع location={loc}")

            return BatchInfo(
                provider="gemini", model=model, job_id=job_id, job_name=job_full_name,
                item_order=list(range(len(prompts))), items_count=len(prompts),
                created_at=datetime.now().isoformat(), status="submitted",
                extra={"display_name": job_name, "input_uri": input_uri, "method": "sdk", "location": loc,
                       "labels": labels or {}}
            )
        except Exception as e:
            last_err = e
            if "NOT_FOUND" in str(e) or "does not exist" in str(e):
                log(f"  [!] الموديل غير متاح في location={loc}")
                continue
            raise

    raise last_err


def _batch_send_claude(prompts: list, model: str, api_key: str, system_prompt: str, temperature: float, max_tokens: int) -> BatchInfo:
    """إرسال دفعة عبر Claude Batch API"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # بناء الطلبات
    requests = []
    for i, prompt in enumerate(prompts):
        request = {
            "custom_id": str(i),
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        }
        if system_prompt:
            request["params"]["system"] = system_prompt

        requests.append(request)

    # إرسال الدفعة
    batch = client.messages.batches.create(requests=requests)

    # استخراج معلومات المهمة
    batch_info = BatchInfo(
        provider="claude",
        model=model,
        job_id=batch.id,
        job_name=batch.id,
        item_order=list(range(len(prompts))),
        items_count=len(prompts),
        created_at=datetime.now().isoformat(),
        status=batch.processing_status if hasattr(batch, 'processing_status') else "submitted",
        extra={}
    )

    return batch_info


def _batch_send_gemini_rest(prompts: list, model: str, api_key: str, system_prompt: str, temperature: float, max_tokens: int, thinking_budget: int = None, thinking_level: str = None) -> BatchInfo:
    """إرسال دفعة عبر Gemini REST API (من الوثيقة - طريقة 4)"""
    import httpx

    # بناء الطلبات
    requests = []
    for i, prompt in enumerate(prompts):
        request = {
            "model": f"models/{model}",
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.95,
                "maxOutputTokens": max_tokens,
            }
        }
        is_gemini_3 = _is_gemini_3_model(model)
        if thinking_level and is_gemini_3:
            gemini_level = _gemini_3_thinking_level(model, thinking_level)
            if gemini_level:
                request["generationConfig"]["thinkingConfig"] = {"thinkingLevel": gemini_level.upper()}
        elif thinking_budget is not None:
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        elif thinking_level and not is_gemini_3:
            level_to_budget = {"none": 0, "low": 1024, "medium": 8192, "high": 24576}
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": level_to_budget.get(thinking_level.lower(), 1024)}
        if system_prompt:
            request["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        requests.append(request)

    # إرسال الدفعة
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchGenerateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    payload = {
        "batch": {
            "display_name": f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "input_config": {
                "requests": {
                    "requests": requests
                }
            }
        }
    }

    # timeout ديناميكي حسب حجم الـ payload (300s كحد أدنى + 60s لكل 100 طلب)
    rest_timeout = max(300, len(requests) * 0.6 + 300)
    response = httpx.post(url, headers=headers, json=payload, timeout=rest_timeout)
    response.raise_for_status()
    job_name = response.json().get('name', '')

    batch_info = BatchInfo(
        provider="gemini",
        model=model,
        job_id=job_name.split('/')[-1] if job_name else "",
        job_name=job_name,
        item_order=list(range(len(prompts))),
        items_count=len(prompts),
        created_at=datetime.now().isoformat(),
        status="submitted",
        extra={"method": "rest"}
    )

    return batch_info


def _create_vertex_batch_job_rest(project_id: str, location: str, model: str, input_uri: str, output_uri: str, display_name: str, labels: dict = None) -> dict:
    """Create Vertex batch job through REST so Google returns the real API error."""
    import requests
    import google.auth
    from google.auth.transport.requests import Request

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())

    endpoint_prefix = "" if location == "global" else f"{location}-"
    url = f"https://{endpoint_prefix}aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/batchPredictionJobs"
    payload = {
        "displayName": display_name,
        "model": f"publishers/google/models/{model}",
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [input_uri]},
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": output_uri},
        },
    }
    if labels:
        payload["labels"] = labels

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-user-project": project_id,
        },
        json=payload,
        timeout=180,
    )
    if response.status_code >= 400:
        raise EngineError(
            f"Vertex Batch create failed HTTP {response.status_code}: {response.text[:1200]}",
            code="BATCH_CREATE_FAILED",
        )
    return response.json()


def _batch_send_vertex(prompts: list, model: str, system_prompt: str, temperature: float, max_tokens: int, thinking_budget: int = None, thinking_level: str = None, labels: dict = None) -> BatchInfo:
    """إرسال دفعة عبر Vertex AI Batch Prediction (من الوثيقة - طريقة 7)"""
    import json

    # إعداد GCS
    project_id, location, bucket_name = _setup_gcs_credentials()
    configured_location = location
    location = _vertex_batch_location_for_model(model, configured_location)
    if location != configured_location:
        log(f"  [vertex-location] {model}: using {location} instead of {configured_location}")

    # بناء JSONL
    jsonl_lines = []
    for i, prompt in enumerate(prompts):
        request = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        is_gemini_3 = _is_gemini_3_model(model)
        if thinking_level and is_gemini_3:
            gemini_level = _gemini_3_thinking_level(model, thinking_level)
            if gemini_level:
                request["generationConfig"]["thinkingConfig"] = {"thinkingLevel": gemini_level.upper()}
        elif thinking_budget is not None:
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        elif thinking_level and not is_gemini_3:
            level_to_budget = {"none": 0, "low": 1024, "medium": 8192, "high": 24576}
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": level_to_budget.get(thinking_level.lower(), 1024)}
        if system_prompt:
            request["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        jsonl_lines.append(json.dumps({"request": request}, ensure_ascii=False))

    jsonl_content = "\n".join(jsonl_lines)

    # رفع إلى GCS — مع run_id في المسار للتتبع
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _pre_rid = (labels or {}).get("run_id", "")
    _chunk = (labels or {}).get("chunk", "")
    _chunk_suffix = f"_part{_chunk}" if _chunk else ""
    if _pre_rid:
        input_blob_name = f"batch_input/{_pre_rid}/vertex_{timestamp}{_chunk_suffix}.jsonl"
        output_uri = f"gs://{bucket_name}/batch_output/{_pre_rid}/vertex_{timestamp}{_chunk_suffix}/"
    else:
        input_blob_name = f"batch_input/vertex_{timestamp}{_chunk_suffix}.jsonl"
        output_uri = f"gs://{bucket_name}/batch_output/vertex_{timestamp}{_chunk_suffix}/"
    input_uri = _upload_to_gcs(bucket_name, input_blob_name, jsonl_content)

    log(f"  رفع الطلبات إلى: {input_uri}")

    # إنشاء مهمة Batch مع labels للتتبع في Google Cloud Billing
    _run_id = (labels or {}).get("run_id", "")
    _recipe = (labels or {}).get("recipe", "")
    job_display_name = _safe_vertex_display_name(_run_id, _chunk_suffix, timestamp)
    job_labels = _sanitize_gcp_labels(labels)
    if job_labels:
        log(f"  [labels] Vertex Batch Labels: {job_labels}")
    if _run_id:
        log(f"  [RUN_ID] ✅ Run ID الكامل: {_run_id}")
        log(f"  [RUN_ID] ✅ display_name في Google Cloud: {job_display_name}")
        log(f"  [RUN_ID] ✅ Labels: {job_labels}")

    batch_job = _create_vertex_batch_job_rest(
        project_id=project_id,
        location=location,
        model=model,
        input_uri=input_uri,
        output_uri=output_uri,
        display_name=job_display_name,
        labels=job_labels if job_labels else None,
    )

    batch_info = BatchInfo(
        provider="vertex",
        model=model,
        job_id=batch_job.get("name", ""),
        job_name=batch_job.get("displayName", job_display_name),
        item_order=list(range(len(prompts))),
        items_count=len(prompts),
        created_at=datetime.now().isoformat(),
        status="submitted",
        extra={
            "gcs_output": output_uri,
            "input_uri": input_uri,
            "location": location,
            "method": "vertex_rest",
            "display_name": job_display_name,
            "labels": labels or {},
            "job_labels": job_labels,
        }
    )

    return batch_info


def batch_send(prompts: list, model: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 8192, save_path: str = None, method: str = "vertex", thinking_budget: int = None, thinking_level: str = None, labels: dict = None) -> EngineResult:
    """
    الدالة 2: إرسال دفعة برومبتات
    تدعم: Gemini Batch (SDK/REST), Claude Batch, Vertex AI Batch

    method: "vertex" (افتراضي لأنه يدعم Billing labels), "sdk", "rest"
    """
    start_time = time.time()

    # فحوصات
    prompts = _check_batch_prompts(prompts)
    model = _check_model(model)
    provider = detect_provider(model)
    thinking_budget, thinking_level = _apply_ui_thinking_override(thinking_budget, thinking_level)

    if _strict_google_cost_tracking_enabled() and _has_run_label(labels) and provider in ("gemini", "vertex") and method != "vertex":
        log(f"  [cost-tracking] forcing batch method=vertex instead of {method} for run_id={(labels or {}).get('run_id') or os.getenv('RUN_ID')}")
        method = "vertex"
    if _strict_google_cost_tracking_enabled() and provider in ("gemini", "vertex") and not _has_run_label(labels):
        raise EngineError(
            "COST_TRACKING_MISSING_RUN_ID: Google batch لازم يتبعت ومعاه run_id label عشان نطلع تكلفة مؤكدة لكل رن.",
            code="COST_TRACKING_MISSING_RUN_ID",
        )
    if _strict_google_cost_tracking_enabled() and provider in ("gemini", "vertex") and os.getenv("RUN_ID") and not (labels and labels.get("run_id")):
        labels = {**(labels or {}), "run_id": os.getenv("RUN_ID")}

    # Vertex لا يحتاج API key (سواء بالـ method أو بالـ provider)
    api_key = None if method == "vertex" or provider == "vertex" else _get_api_key_for_provider(provider)

    log(f"→ إرسال دفعة | الموديل: {model} | المزود: {provider} | الطريقة: {method} | العدد: {len(prompts)}")

    def _send_single_batch(single_prompts, single_labels):
        if provider == "gemini" and method == "sdk":
            return _retry_call(
                lambda: _batch_send_gemini(single_prompts, model, api_key, system_prompt, temperature, max_tokens, thinking_budget, thinking_level, labels=single_labels),
                max_retries=3, base_delay=3.0, description=f"Gemini Batch SDK {model}"
            )
        if provider == "gemini" and method == "rest":
            return _retry_call(
                lambda: _batch_send_gemini_rest(single_prompts, model, api_key, system_prompt, temperature, max_tokens, thinking_budget, thinking_level),
                max_retries=3, base_delay=3.0, description=f"Gemini Batch REST {model}"
            )
        if (provider == "gemini" or provider == "vertex") and method == "vertex":
            # Vertex AI Batch Prediction
            actual_model = model.split(":", 1)[1] if ":" in model else model
            return _retry_call(
                lambda: _batch_send_vertex(single_prompts, actual_model, system_prompt, temperature, max_tokens, thinking_budget, thinking_level, labels=single_labels),
                max_retries=3, base_delay=3.0, description=f"Vertex AI Batch {actual_model}"
            )
        if provider == "claude":
            return _retry_call(
                lambda: _batch_send_claude(single_prompts, model, api_key, system_prompt, temperature, max_tokens),
                max_retries=3, base_delay=3.0, description=f"Claude Batch {model}"
            )
        raise EngineError(
            f"Batch غير مدعوم للمزود: {provider} بالطريقة: {method}",
            code="BATCH_NOT_SUPPORTED"
        )

    try:
        chunks = _split_prompts_for_batch(prompts)
        if len(chunks) > 1:
            max_requests, max_bytes = _batch_split_limits()
            log(f"  [auto-split] تقسيم الباتش إلى {len(chunks)} jobs | حد الطلبات/job={max_requests} | حد الحجم/job={max_bytes // (1024 * 1024)}MB")
            chunk_infos = []
            for chunk_idx, (global_indexes, chunk_prompts) in enumerate(chunks, start=1):
                chunk_labels = dict(labels or {})
                chunk_labels["chunk"] = str(chunk_idx)
                log(f"  [auto-split] إرسال جزء {chunk_idx}/{len(chunks)}: {len(chunk_prompts)} طلب")
                chunk_info = _send_single_batch(chunk_prompts, chunk_labels if chunk_labels else None)
                chunk_info.item_order = list(global_indexes)
                chunk_infos.append(chunk_info)

            first = chunk_infos[0]
            batch_info = BatchInfo(
                provider=first.provider,
                model=first.model,
                job_id=f"multi-{len(chunk_infos)}-{first.job_id[:32]}",
                job_name=f"multi-batch-{len(chunk_infos)}-jobs",
                item_order=list(range(len(prompts))),
                items_count=len(prompts),
                created_at=datetime.now().isoformat(),
                status="submitted",
                extra={
                    "method": "multi",
                    "labels": labels or {},
                    "split": {
                        "chunks": len(chunk_infos),
                        "max_requests": max_requests,
                        "max_mb": max_bytes / (1024 * 1024),
                    },
                    "chunks": [asdict(c) for c in chunk_infos],
                },
            )
        else:
            batch_info = _send_single_batch(prompts, labels)

        # حفظ معلومات الـ Batch
        if save_path:
            batch_info.save(save_path)
            log(f"✓ تم حفظ معلومات الدفعة في: {save_path}")
        else:
            # حفظ في المجلد الحالي
            default_path = BATCH_INFO_FILENAME
            batch_info.save(default_path)
            log(f"✓ تم حفظ معلومات الدفعة في: {default_path}")

        duration = int((time.time() - start_time) * 1000)
        log(f"<- batch_send OK | job_id: {batch_info.job_id[:20]}... | {duration}ms")

        return EngineResult(
            success=True,
            data=batch_info,
            model=model,
            provider=provider,
            duration_ms=duration
        )

    except EngineError:
        raise
    except Exception as e:
        duration = int((time.time() - start_time) * 1000)
        raise EngineError(
            f"خطأ غير متوقع من {provider} Batch: {str(e)[:500]}",
            code="UNEXPECTED_ERROR"
        )


def _batch_retrieve_gemini(batch_info: BatchInfo, api_key: str) -> list:
    """استقبال نتائج دفعة من Gemini"""
    import json
    from google import genai

    # إعداد GCS — استخدام الـ location المحفوظ من الإرسال
    project_id, location, bucket_name = _setup_gcs_credentials()
    saved_location = batch_info.extra.get("location", location) if batch_info.extra else location
    client = genai.Client(vertexai=True, project=project_id, location=saved_location)

    # استرجاع معلومات المهمة
    job_name = batch_info.job_name or batch_info.job_id
    batch_job = client.batches.get(name=job_name)

    # فحص الحالة
    state = batch_job.state if hasattr(batch_job, 'state') else "UNKNOWN"
    state_str = str(state)
    log(f"  حالة المهمة: {state_str}")

    if "FAILED" in state_str:
        raise EngineError(
            f"المهمة فشلت على Gemini: {job_name}",
            code="BATCH_JOB_FAILED"
        )

    if "SUCCEEDED" not in state_str:
        raise EngineError(
            f"المهمة لم تكتمل بعد. الحالة: {state_str}",
            code="BATCH_JOB_NOT_READY"
        )

    # استخراج النتائج مع finishReason
    results = []       # النصوص
    finish_reasons = [] # أسباب الإنهاء
    token_totals = {"input": 0, "output": 0, "thinking": 0, "total": 0}

    # طريقة 1: من GCS output (الطريقة الأساسية)
    if hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'gcs_uri'):
        gcs_uri = batch_job.dest.gcs_uri
        log(f"  قراءة النتائج من: {gcs_uri}")

        # تنزيل JSONL من GCS
        jsonl_content = _download_from_gcs(gcs_uri)

        # تحليل JSONL
        for line in jsonl_content.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    candidate = data['response']['candidates'][0]
                    text = candidate['content']['parts'][0]['text']
                    finish = candidate.get('finishReason', 'UNKNOWN')
                    results.append(text)
                    finish_reasons.append(finish)
                    # استخراج التوكنز
                    usage = data.get('response', {}).get('usageMetadata', {})
                    if usage:
                        token_totals["input"] += usage.get('promptTokenCount', 0)
                        token_totals["output"] += usage.get('candidatesTokenCount', 0)
                        token_totals["thinking"] += usage.get('thoughtsTokenCount', 0)
                        token_totals["total"] += usage.get('totalTokenCount', 0)
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    log(f"[!] فشل استخراج نتيجة من GCS: {str(e)}")
                    results.append("")
                    finish_reasons.append("ERROR")

    # طريقة 2: من inlined_responses (fallback)
    elif hasattr(batch_job, 'dest') and hasattr(batch_job.dest, 'inlined_responses'):
        log(f"  قراءة النتائج من inlined_responses")
        for response in batch_job.dest.inlined_responses:
            try:
                text = response.response.candidates[0].content.parts[0].text
                finish = getattr(response.response.candidates[0], 'finish_reason', 'UNKNOWN')
                results.append(text)
                finish_reasons.append(str(finish))
                # استخراج التوكنز من inlined_responses
                um = getattr(response.response, 'usage_metadata', None)
                if um:
                    token_totals["input"] += getattr(um, 'prompt_token_count', 0) or 0
                    token_totals["output"] += getattr(um, 'candidates_token_count', 0) or 0
                    token_totals["thinking"] += getattr(um, 'thoughts_token_count', 0) or 0
                    token_totals["total"] += getattr(um, 'total_token_count', 0) or 0
            except (AttributeError, IndexError) as e:
                log(f"[!] فشل استخراج نتيجة: {str(e)}")
                results.append("")
                finish_reasons.append("ERROR")
    else:
        log(f"[!] تحذير: لم يتم العثور على نتائج في batch_job.dest")

    # تسجيل إحصائيات finishReason
    truncated = [i for i, r in enumerate(finish_reasons) if r == 'MAX_TOKENS']
    if truncated:
        log(f"  [!] {len(truncated)} نتيجة مقطوعة (MAX_TOKENS) من {len(results)}")

    # تسجيل إحصائيات التوكنز الإجمالية
    if token_totals["total"] > 0:
        log(f"  [tokens] إجمالي الباتش: input={token_totals['input']} output={token_totals['output']} thinking={token_totals['thinking']} total={token_totals['total']}")

    # حفظ finish_reasons في batch_info للاستخدام لاحقاً
    if not hasattr(batch_info, '_finish_reasons'):
        batch_info._finish_reasons = finish_reasons
    batch_info._token_totals = token_totals

    return results


def _batch_retrieve_claude(batch_info: BatchInfo, api_key: str) -> list:
    """استقبال نتائج دفعة من Claude"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # استرجاع معلومات المهمة
    batch_id = batch_info.job_id
    batch_status = client.messages.batches.retrieve(batch_id)

    # فحص الحالة
    status = batch_status.processing_status
    log(f"  حالة المهمة: {status}")

    if status == "ended":
        # المهمة انتهت - نتحقق من النتائج
        pass
    elif status in ["in_progress", "pending"]:
        raise EngineError(
            f"المهمة لم تكتمل بعد. الحالة: {status}",
            code="BATCH_JOB_NOT_READY"
        )
    else:
        raise EngineError(
            f"حالة غير معروفة: {status}",
            code="BATCH_JOB_UNKNOWN_STATUS"
        )

    # استخراج النتائج مع التوكنز
    results = []
    token_totals = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    for result in client.messages.batches.results(batch_id):
        try:
            if hasattr(result, 'result') and hasattr(result.result, 'message'):
                msg = result.result.message
                text = msg.content[0].text
                results.append(text)
                # استخراج التوكنز من كل نتيجة
                if hasattr(msg, 'usage') and msg.usage:
                    inp = getattr(msg.usage, 'input_tokens', 0) or 0
                    out = getattr(msg.usage, 'output_tokens', 0) or 0
                    token_totals["input"] += inp
                    token_totals["output"] += out
                    token_totals["total"] += inp + out
            else:
                results.append("")
        except (AttributeError, IndexError) as e:
            log(f"[!] فشل استخراج نتيجة: {str(e)}")
            results.append("")

    if token_totals["total"] > 0:
        log(f"  [tokens] إجمالي Claude Batch: input={token_totals['input']} output={token_totals['output']} total={token_totals['total']}")
    batch_info._token_totals = token_totals

    return results


def _batch_retrieve_gemini_rest(batch_info: BatchInfo, api_key: str) -> list:
    """استقبال نتائج دفعة من Gemini REST API (من الوثيقة - طريقة 4)"""
    import httpx

    # استرجاع معلومات المهمة
    job_name = batch_info.job_name or batch_info.job_id
    batch_url = f"https://generativelanguage.googleapis.com/v1beta/{job_name}"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    response = httpx.get(batch_url, headers=headers, timeout=300.0)
    response.raise_for_status()
    data = response.json()

    # فحص الحالة
    state = data.get('state', 'UNKNOWN')
    log(f"  حالة المهمة: {state}")

    if state == "FAILED":
        raise EngineError(
            f"المهمة فشلت على Gemini REST: {job_name}",
            code="BATCH_JOB_FAILED"
        )

    if state != "SUCCEEDED":
        raise EngineError(
            f"المهمة لم تكتمل بعد. الحالة: {state}",
            code="BATCH_JOB_NOT_READY"
        )

    # استخراج النتائج مع التوكنز
    results = []
    token_totals = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    responses = data.get('responses', [])
    for resp in responses:
        try:
            text = resp['candidates'][0]['content']['parts'][0]['text']
            results.append(text)
            # استخراج التوكنز من كل response
            usage = resp.get('usageMetadata', {})
            if usage:
                token_totals["input"] += usage.get('promptTokenCount', 0)
                token_totals["output"] += usage.get('candidatesTokenCount', 0)
                token_totals["thinking"] += usage.get('thoughtsTokenCount', 0)
                token_totals["total"] += usage.get('totalTokenCount', 0)
        except (KeyError, IndexError) as e:
            log(f"[!] فشل استخراج نتيجة: {str(e)}")
            results.append("")

    if token_totals["total"] > 0:
        log(f"  [tokens] إجمالي Gemini REST Batch: input={token_totals['input']} output={token_totals['output']} thinking={token_totals['thinking']} total={token_totals['total']}")
    batch_info._token_totals = token_totals

    return results


def _extract_vertex_batch_response(record: dict):
    response = record.get("prediction") or record.get("response") or record
    if not isinstance(response, dict):
        raise KeyError("response")

    if response.get("error"):
        raise ValueError(f"response error: {response.get('error')}")
    if record.get("status") and str(record.get("status")).strip():
        raise ValueError(f"record status: {record.get('status')}")

    text_parts = []
    candidates = response.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        parts = content.get("parts") or []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                text_parts.append(str(part["text"]))

    text = "\n".join(text_parts).strip()
    if not text:
        raise KeyError("response.candidates[].content.parts[].text")
    return text, response.get("usageMetadata", {}) or {}


def _batch_retrieve_vertex(batch_info: BatchInfo) -> list:
    """استقبال نتائج دفعة من Vertex AI Batch Prediction (من الوثيقة - طريقة 7)"""
    import json
    from google.cloud import aiplatform
    from google.cloud import storage

    # إعداد GCS
    project_id, location, bucket_name = _setup_gcs_credentials()

    # تهيئة Vertex AI
    location = batch_info.extra.get("location", location) if batch_info.extra else location
    aiplatform.init(project=project_id, location=location)

    # جلب المهمة
    batch_job = aiplatform.BatchPredictionJob(batch_info.job_id)

    # فحص الحالة
    state = batch_job.state
    log(f"  حالة المهمة: {state}")

    if state == aiplatform.gapic.JobState.JOB_STATE_FAILED:
        raise EngineError(
            f"المهمة فشلت على Vertex AI: {batch_info.job_id}",
            code="BATCH_JOB_FAILED"
        )

    if state != aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED:
        raise EngineError(
            f"المهمة لم تكتمل بعد. الحالة: {state}",
            code="BATCH_JOB_NOT_READY"
        )

    # قراءة النتائج من GCS
    output_uri = batch_info.extra.get('gcs_output', '')
    if not output_uri:
        raise EngineError(
            "مسار النتائج غير موجود في batch_info",
            code="BATCH_OUTPUT_URI_MISSING"
        )

    # جلب ملفات النتائج من GCS
    storage_client = storage.Client()
    bucket_name = output_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(output_uri.replace("gs://", "").split("/")[1:])

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    # قراءة النتائج مع التوكنز
    results = []
    parse_errors = []
    total_lines = 0
    token_totals = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    for blob in blobs:
        if blob.name.endswith('.jsonl'):
            content = blob.download_as_text()
            for line in content.strip().split('\n'):
                if line:
                    total_lines += 1
                    data = json.loads(line)
                    try:
                        text, usage = _extract_vertex_batch_response(data)
                        results.append(text)
                        # استخراج التوكنز
                        if usage:
                            token_totals["input"] += usage.get('promptTokenCount', 0)
                            token_totals["output"] += usage.get('candidatesTokenCount', 0)
                            token_totals["thinking"] += usage.get('thoughtsTokenCount', 0)
                            token_totals["total"] += usage.get('totalTokenCount', 0)
                    except (KeyError, IndexError, TypeError, ValueError) as e:
                        sample_keys = ",".join(data.keys()) if isinstance(data, dict) else type(data).__name__
                        parse_errors.append(f"{str(e)} | keys={sample_keys}")

    if parse_errors:
        sample = " ; ".join(parse_errors[:3])
        raise EngineError(
            f"فشل استخراج {len(parse_errors)} نتيجة من Vertex Batch من أصل {total_lines}. مثال: {sample}",
            code="BATCH_RESULT_PARSE_FAILED",
        )

    if not results:
        raise EngineError(
            "Vertex Batch اكتمل لكن لم يرجع أي نتائج قابلة للقراءة.",
            code="BATCH_RESULT_EMPTY",
        )

    if token_totals["total"] > 0:
        log(f"  [tokens] إجمالي Vertex Batch: input={token_totals['input']} output={token_totals['output']} thinking={token_totals['thinking']} total={token_totals['total']}")
    batch_info._token_totals = token_totals

    return results


def _batch_retrieve_multi(info: BatchInfo, start_time: float) -> EngineResult:
    chunks = info.extra.get("chunks", []) if info.extra else []
    if not chunks:
        raise EngineError("ملف batch_info متعدد الأجزاء لا يحتوي على chunks", code="BATCH_MULTI_EMPTY")

    log(f"  [multi] استقبال باتش مقسم: {len(chunks)} jobs")
    ordered_results = {}
    token_usage = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    not_ready = []

    for idx, chunk_data in enumerate(chunks, start=1):
        chunk_info = BatchInfo(**chunk_data)
        try:
            chunk_result = batch_retrieve(batch_info=chunk_info)
        except EngineError as e:
            if e.code == "BATCH_JOB_NOT_READY":
                not_ready.append(idx)
                continue
            raise

        chunk_order = chunk_info.item_order or list(range(len(chunk_result.data or [])))
        for global_idx, text in zip(chunk_order, chunk_result.data or []):
            ordered_results[int(global_idx)] = text
        for key in token_usage:
            token_usage[key] += (chunk_result.token_usage or {}).get(key, 0) or 0
        log(f"  [multi] تم استقبال جزء {idx}/{len(chunks)}: {len(chunk_result.data or [])} نتيجة")

    if not_ready:
        raise EngineError(
            f"لم تكتمل كل أجزاء الباتش بعد: {not_ready}",
            code="BATCH_JOB_NOT_READY",
        )

    results = [ordered_results.get(i, "") for i in range(info.items_count)]
    if _strict_google_cost_tracking_enabled() and info.provider in ("vertex", "gemini") and (token_usage.get("total") or 0) <= 0:
        raise EngineError(
            "Google Batch اكتمل لكن لم يرجع usageMetadata للتكلفة. تم إيقاف الرن حتى لا يظهر بدون رقم تكلفة.",
            code="BATCH_USAGE_MISSING",
        )
    duration = int((time.time() - start_time) * 1000)
    log(f"<- batch_retrieve multi OK | {len(results)} نتيجة | {duration}ms")
    return EngineResult(
        success=True,
        data=results,
        model=info.model,
        provider=info.provider,
        duration_ms=duration,
        token_usage=token_usage,
    )


def batch_retrieve(batch_info_path: str = None, batch_info: BatchInfo = None) -> EngineResult:
    """
    الدالة 3: استقبال نتائج الدفعة
    تدعم: Gemini (SDK/REST), Claude, Vertex AI Batch
    """
    start_time = time.time()

    # فحوصات
    info = batch_info or (BatchInfo.load(batch_info_path) if batch_info_path else None)
    info = _check_batch_info(info)

    if info.extra and info.extra.get("chunks"):
        return _batch_retrieve_multi(info, start_time)

    provider = info.provider
    method = info.extra.get('method', 'sdk')  # الطريقة المستخدمة في الإرسال

    # Vertex لا يحتاج API key
    api_key = None if provider == "vertex" else _get_api_key_for_provider(provider)

    log(f"→ استقبال دفعة | المزود: {provider} | الطريقة: {method} | المهمة: {info.job_id[:20]}...")

    try:
        if provider == "gemini" and method == "sdk":
            results = _retry_call(
                lambda: _batch_retrieve_gemini(info, api_key),
                max_retries=3, base_delay=3.0, description=f"Gemini Batch SDK Retrieve"
            )
        elif provider == "gemini" and method == "rest":
            results = _retry_call(
                lambda: _batch_retrieve_gemini_rest(info, api_key),
                max_retries=3, base_delay=3.0, description=f"Gemini Batch REST Retrieve"
            )
        elif provider == "vertex":
            results = _retry_call(
                lambda: _batch_retrieve_vertex(info),
                max_retries=3, base_delay=3.0, description=f"Vertex AI Batch Retrieve"
            )
        elif provider == "claude":
            results = _retry_call(
                lambda: _batch_retrieve_claude(info, api_key),
                max_retries=3, base_delay=3.0, description=f"Claude Batch Retrieve"
            )
        else:
            raise EngineError(
                f"Batch غير مدعوم للمزود: {provider} بالطريقة: {method}",
                code="BATCH_NOT_SUPPORTED"
            )

        # التحقق من عدد النتائج
        expected_count = info.items_count
        actual_count = len(results)
        if actual_count != expected_count:
            log(f"[!] تحذير: العدد المتوقع {expected_count} لكن استلمنا {actual_count}")

        # جمع التوكنز من batch_info
        token_usage = getattr(info, '_token_totals', {"input": 0, "output": 0, "thinking": 0, "total": 0})
        if _strict_google_cost_tracking_enabled() and provider in ("vertex", "gemini") and (token_usage.get("total") or 0) <= 0:
            raise EngineError(
                "Google Batch اكتمل لكن لم يرجع usageMetadata للتكلفة. تم إيقاف الرن حتى لا يظهر بدون رقم تكلفة.",
                code="BATCH_USAGE_MISSING",
            )

        duration = int((time.time() - start_time) * 1000)
        log(f"<- batch_retrieve OK | {len(results)} نتيجة | {duration}ms")

        return EngineResult(
            success=True,
            data=results,
            model=info.model,
            provider=provider,
            duration_ms=duration,
            token_usage=token_usage
        )

    except EngineError:
        raise
    except Exception as e:
        duration = int((time.time() - start_time) * 1000)
        raise EngineError(
            f"خطأ غير متوقع من {provider} Batch Retrieve: {str(e)[:500]}",
            code="UNEXPECTED_ERROR"
        )


def tts(text: str) -> EngineResult:
    """
    الدالة الموحدة: تحويل نص لصوت
    بتقرأ المزود والصوت من متغيرات البيئة:
    - TTS_PROVIDER: elevenlabs / minimax / vertex (الافتراضي: vertex)
    - TTS_VOICE_ID: معرف الصوت (الافتراضي: Achird)
    الوصفة بتنادي tts(text) وبس - مش محتاجة تعرف أي تفاصيل.
    """
    provider = os.getenv("TTS_PROVIDER", "vertex").lower().strip()
    voice_id = os.getenv("TTS_VOICE_ID", "Achird").strip()

    log(f"→ TTS موحد | المزود: {provider} | الصوت: {voice_id}")

    if provider == "elevenlabs":
        return tts_elevenlabs(text, voice_id=voice_id)
    elif provider == "minimax":
        return tts_minimax(text, voice_id=voice_id)
    else:
        return tts_vertex(text, voice=voice_id)


def tts_elevenlabs(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> EngineResult:
    """
    الدالة 4: تحويل نص لصوت عبر ElevenLabs
    الصوت الافتراضي: Rachel (21m00Tcm4TlvDq8ikWAM)
    """
    start_time = time.time()
    text = _check_text_for_tts(text)
    api_key = _check_api_key("ELEVENLABS_API_KEY")

    log(f"→ TTS ElevenLabs | طول النص: {len(text)} | الصوت: {voice_id[:10]}...")

    try:
        audio_data = _retry_call(
            lambda: _tts_elevenlabs_impl(text, voice_id, api_key),
            max_retries=3, base_delay=3.0, description="ElevenLabs TTS"
        )

        # فحص الصوت
        audio_data = _check_audio_data(audio_data, "ElevenLabs")

        duration = int((time.time() - start_time) * 1000)
        log(f"<- TTS ElevenLabs OK | {len(audio_data)} bytes | {duration}ms")

        return EngineResult(
            success=True,
            data=audio_data,
            provider="elevenlabs",
            duration_ms=duration
        )

    except EngineError:
        raise
    except Exception as e:
        duration = int((time.time() - start_time) * 1000)
        raise EngineError(
            f"خطأ غير متوقع من ElevenLabs TTS: {str(e)[:500]}",
            code="UNEXPECTED_ERROR"
        )


def _tts_elevenlabs_impl(text: str, voice_id: str, api_key: str) -> bytes:
    """تنفيذ TTS عبر ElevenLabs API"""
    import httpx

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "model_id": "eleven_v3",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = httpx.post(url, headers=headers, json=data, timeout=60.0)
    response.raise_for_status()

    return response.content


def tts_minimax(text: str, voice_id: str = "moss_audio_c4f52a13-a60c-11f0-9b9d-12927f33a4b7", model: str = "speech-2.8-hd") -> EngineResult:
    """
    الدالة 5: تحويل نص لصوت عبر MiniMax
    الصوت الافتراضي: moss_audio_c4f52a13-a60c-11f0-9b9d-12927f33a4b7
    الموديل الافتراضي: speech-2.8-hd
    """
    start_time = time.time()
    text = _check_text_for_tts(text)
    api_key = _check_api_key("MINIMAX_API_KEY")

    log(f"→ TTS MiniMax | طول النص: {len(text)} | الصوت: {voice_id[:15]}... | الموديل: {model}")

    try:
        audio_data = _retry_call(
            lambda: _tts_minimax_impl(text, voice_id, model, api_key),
            max_retries=3, base_delay=3.0, description="MiniMax TTS"
        )

        # فحص الصوت
        audio_data = _check_audio_data(audio_data, "MiniMax")

        duration = int((time.time() - start_time) * 1000)
        log(f"<- TTS MiniMax OK | {len(audio_data)} bytes | {duration}ms")

        return EngineResult(
            success=True,
            data=audio_data,
            provider="minimax",
            model=model,
            duration_ms=duration
        )

    except EngineError:
        raise
    except Exception as e:
        duration = int((time.time() - start_time) * 1000)
        raise EngineError(
            f"خطأ غير متوقع من MiniMax TTS: {str(e)[:500]}",
            code="UNEXPECTED_ERROR"
        )


def _tts_minimax_impl(text: str, voice_id: str, model: str, api_key: str) -> bytes:
    """تنفيذ TTS عبر MiniMax API"""
    import httpx

    url = "https://api.minimax.io/v1/t2a_v2"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1
        }
    }

    response = httpx.post(url, headers=headers, json=data, timeout=60.0)
    response.raise_for_status()

    result = response.json()

    # فحص حالة الرد
    base_resp = result.get("base_resp", {})
    if base_resp.get("status_code", 0) != 0:
        raise ValueError(f"MiniMax API error: {base_resp.get('status_msg', 'unknown')}")

    # استخراج الصوت من hex
    hex_audio = result.get("data", {}).get("audio", "")
    if not hex_audio:
        raise ValueError("MiniMax returned empty audio data")

    audio_bytes = bytes.fromhex(hex_audio)
    return audio_bytes


def tts_vertex(text: str, voice: str = "Achird", project_id: str = None, location: str = "europe-west1") -> EngineResult:
    """
    الدالة 5: تحويل نص لصوت عبر Vertex AI
    الصوت الافتراضي: Achird (عربي)
    """
    start_time = time.time()
    text = _check_text_for_tts(text)

    # المشروع والموقع من البيئة أو المعاملات
    project_id = project_id or os.getenv("GOOGLE_QUOTA_PROJECT", "")
    if not project_id:
        raise EngineError(
            "GOOGLE_QUOTA_PROJECT غير محدد في .env",
            code="MISSING_PROJECT_ID"
        )

    log(f"→ TTS Vertex AI | طول النص: {len(text)} | الصوت: {voice} | المشروع: {project_id}")

    _usage_holder = {}
    try:
        audio_data = _retry_call(
            lambda: _check_audio_data(_tts_vertex_impl(text, voice, project_id, location, _usage_holder), "Vertex AI"),
            max_retries=3, base_delay=3.0, description="Vertex AI TTS"
        )

        duration = int((time.time() - start_time) * 1000)
        log(f"<- TTS Vertex AI OK | {len(audio_data)} bytes | {duration}ms")

        return EngineResult(
            success=True,
            data=audio_data,
            provider="vertex",
            model=_usage_holder.get("model", ""),
            duration_ms=duration,
            token_usage=_usage_holder.get("token_usage", {})
        )

    except EngineError:
        raise
    except Exception as e:
        duration = int((time.time() - start_time) * 1000)
        raise EngineError(
            f"خطأ غير متوقع من Vertex AI TTS: {str(e)[:500]}",
            code="UNEXPECTED_ERROR"
        )


def _tts_vertex_impl(text: str, voice: str, project_id: str, location: str, usage_holder: dict = None) -> bytes:
    """تنفيذ TTS عبر Vertex AI"""
    import json
    import tempfile
    from google import genai
    from google.genai import types

    # إعداد بيانات الاعتماد
    creds = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
        "refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN", ""),
        "quota_project_id": project_id,
        "type": "authorized_user",
        "universe_domain": "googleapis.com"
    }

    creds_file = os.path.join(tempfile.gettempdir(), 'gcp_creds_tts.json')
    with open(creds_file, 'w') as f:
        json.dump(creds, f)

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file

    # الاتصال بـ Vertex AI
    client = genai.Client(vertexai=True, project=project_id, location=location)

    # موديل الـ TTS يتحدد من الواجهة (متغير البيئة) — الافتراضي gemini-2.5-pro-tts
    tts_model = os.getenv("TTS_MODEL", "gemini-2.5-pro-tts").strip() or "gemini-2.5-pro-tts"
    log(f"  Vertex TTS model: {tts_model}")

    # تحويل النص لصوت
    response = client.models.generate_content(
        model=tts_model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            )
        )
    )

    # التقاط استهلاك التوكنز (input نصي + output صوتي) لحساب التكلفة في usage.html
    if usage_holder is not None:
        usage_holder["model"] = tts_model
        um = getattr(response, "usage_metadata", None)
        if um:
            _in = getattr(um, "prompt_token_count", 0) or 0
            _out = getattr(um, "candidates_token_count", 0) or 0
            _tot = getattr(um, "total_token_count", 0) or (_in + _out)
            usage_holder["token_usage"] = {
                "input": _in, "output": _out, "thinking": 0, "cached": 0, "total": _tot
            }
            log(f"  [tts tokens] input={_in} output(audio)={_out} total={_tot}")

    # استخراج الصوت
    if not response.candidates or not response.candidates[0].content.parts:
        raise ValueError("Vertex AI returned empty audio response")

    inline_data = response.candidates[0].content.parts[0].inline_data
    raw_data = inline_data.data
    mime_type = inline_data.mime_type if hasattr(inline_data, 'mime_type') else "unknown"

    # فحص البيانات: هل هي base64 (نص) أو binary (بايتات خام)?
    import base64
    import io
    import wave as wav_module

    pcm_data = raw_data
    data_type = "binary"

    # لو البيانات نص (base64) نفكها الأول
    if isinstance(raw_data, (str, bytes)):
        try:
            # لو bytes، نشوف لو شكلها base64 text
            test_bytes = raw_data if isinstance(raw_data, bytes) else raw_data.encode()
            # base64 بيكون حروف ASCII فقط (A-Z, a-z, 0-9, +, /, =)
            if all(b < 128 for b in test_bytes[:100]):
                decoded = base64.b64decode(test_bytes)
                # لو الناتج أصغر يبقى كان فعلاً base64
                if len(decoded) < len(test_bytes):
                    pcm_data = decoded
                    data_type = "base64-decoded"
        except Exception:
            pass  # مش base64، نستخدم البيانات الخام

    header_hex = pcm_data[:16].hex() if pcm_data else "empty"
    log(f"  Vertex TTS mime_type: {mime_type} | raw: {len(raw_data)} bytes | {data_type}: {len(pcm_data)} bytes | header: {header_hex}")

    # استخراج sample rate من mime_type
    sample_rate = 24000
    if 'rate=' in mime_type:
        try:
            rate_str = mime_type.split('rate=')[1].split(';')[0].strip()
            sample_rate = int(rate_str)
        except (ValueError, IndexError):
            pass

    # كتابة WAV من البيانات PCM
    wav_buffer = io.BytesIO()
    with wav_module.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    wav_data = wav_buffer.getvalue()
    log(f"  WAV: {len(wav_data)} bytes | rate: {sample_rate}Hz | {data_type}")

    return wav_data


def transcribe(audio_file: str, model: str = "whisper-1", language: str = None) -> EngineResult:
    """
    الدالة 6: تحويل صوت لنص عبر Whisper (OpenAI)
    language: كود اللغة (مثل 'ar' للعربي، 'en' للإنجليزي) - اختياري
    """
    start_time = time.time()
    audio_file = _check_audio_file(audio_file)
    api_key = _check_api_key("OPENAI_API_KEY")

    log(f"→ Whisper | الملف: {os.path.basename(audio_file)} | الحجم: {os.path.getsize(audio_file)} bytes" + (f" | اللغة: {language}" if language else ""))

    try:
        text = _retry_call(
            lambda: _transcribe_impl(audio_file, model, api_key, language=language),
            max_retries=3, base_delay=3.0, description="Whisper"
        )

        # فحص النتيجة
        if not text or not text.strip():
            raise EngineError(
                "Whisper رجع نص فاضي - ممكن الملف الصوتي فيه مشكلة",
                code="EMPTY_TRANSCRIPTION"
            )

        text = text.strip()

        duration = int((time.time() - start_time) * 1000)
        log(f"<- Whisper OK | {len(text)} chars | {duration}ms")

        return EngineResult(
            success=True,
            data=text,
            provider="whisper",
            model=model,
            duration_ms=duration
        )

    except EngineError:
        raise
    except Exception as e:
        duration = int((time.time() - start_time) * 1000)
        raise EngineError(
            f"خطأ غير متوقع من Whisper: {str(e)[:500]}",
            code="UNEXPECTED_ERROR"
        )


def _transcribe_impl(audio_file: str, model: str, api_key: str, language: str = None) -> str:
    """تنفيذ Whisper Transcription"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    kwargs = {"model": model}
    if language:
        kwargs["language"] = language

    with open(audio_file, 'rb') as f:
        kwargs["file"] = f
        transcript = client.audio.transcriptions.create(**kwargs)

    return transcript.text


def transcribe_with_timestamps(audio_file: str, model: str = "whisper-1", language: str = None) -> EngineResult:
    """
    تحويل صوت لنص مع word-level timestamps عبر Whisper (OpenAI)
    يرجع list of dicts: [{"word": "...", "start": 0.0, "end": 0.5}, ...]
    """
    start_time = time.time()
    audio_file = _check_audio_file(audio_file)
    api_key = _check_api_key("OPENAI_API_KEY")

    log(f"→ Whisper+Timestamps | الملف: {os.path.basename(audio_file)} | الحجم: {os.path.getsize(audio_file)} bytes" + (f" | اللغة: {language}" if language else ""))

    try:
        words = _retry_call(
            lambda: _transcribe_timestamps_impl(audio_file, model, api_key, language=language),
            max_retries=3, base_delay=3.0, description="Whisper+Timestamps"
        )

        if not words:
            raise EngineError(
                "Whisper رجع نتيجة فاضية — ممكن الملف الصوتي فيه مشكلة",
                code="EMPTY_TRANSCRIPTION"
            )

        duration = int((time.time() - start_time) * 1000)
        log(f"<- Whisper+Timestamps OK | {len(words)} words | {duration}ms")

        return EngineResult(
            success=True,
            data=words,
            provider="whisper",
            model=model,
            duration_ms=duration
        )

    except EngineError:
        raise
    except Exception as e:
        duration = int((time.time() - start_time) * 1000)
        raise EngineError(
            f"خطأ غير متوقع من Whisper+Timestamps: {str(e)[:500]}",
            code="UNEXPECTED_ERROR"
        )


def _transcribe_timestamps_impl(audio_file: str, model: str, api_key: str, language: str = None) -> list:
    """تنفيذ Whisper Transcription مع word timestamps"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    kwargs = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
    }
    if language:
        kwargs["language"] = language

    with open(audio_file, 'rb') as f:
        kwargs["file"] = f
        transcript = client.audio.transcriptions.create(**kwargs)

    words = []
    if hasattr(transcript, 'words') and transcript.words:
        for w in transcript.words:
            words.append({
                "word": w.word.strip(),
                "start": w.start,
                "end": w.end,
            })

    return words

