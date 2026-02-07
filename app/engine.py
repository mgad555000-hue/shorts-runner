"""
محرك الذكاء الصناعي - Shorts Runner Engine
=============================================
مكتبة موحدة للتعامل مع كل موديلات الذكاء الصناعي.
الوصفات بتستدعي الدوال دي وبس - مش بتتعامل مع الموديلات مباشرة.

الدوال الستة:
1. generate()         - توليد نص من برومبت
2. batch_send()       - إرسال دفعة برومبتات
3. batch_retrieve()   - استقبال نتائج الدفعة
4. tts_elevenlabs()   - تحويل نص لصوت (ElevenLabs)
5. tts_vertex()       - تحويل نص لصوت (Vertex AI)
6. transcribe()       - تحويل صوت لنص (Whisper)
"""

import os
import sys
import time
import json
import traceback
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
    try:
        print(f"[{timestamp}] [ENGINE] {msg}")
    except UnicodeEncodeError:
        print(f"[{timestamp}] [ENGINE] {msg.encode('utf-8', errors='replace').decode('utf-8')}")


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
        raise EngineError(
            f"ملف الصوت فاضي من {source}",
            code="EMPTY_AUDIO_DATA"
        )
    if len(data) < 100:
        raise EngineError(
            f"ملف الصوت صغير جداً ({len(data)} bytes) من {source} - غالباً فيه مشكلة",
            code="AUDIO_TOO_SMALL"
        )
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
    if model_lower.startswith("gemini"):
        return "gemini"
    elif model_lower.startswith("gpt-") or model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
        return "openai"
    elif model_lower.startswith("claude"):
        return "claude"
    elif model_lower.startswith("glm"):
        return "glm"
    else:
        raise EngineError(
            f"موديل غير معروف: {model}. الموديلات المدعومة: gemini, gpt, claude, glm",
            code="UNKNOWN_MODEL"
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

def generate(prompt: str, model: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 8192) -> EngineResult:
    """
    الدالة 1: توليد نص من برومبت
    تدعم: Gemini, OpenAI, Claude, GLM
    """
    start_time = time.time()

    # فحوصات ما قبل الإرسال
    prompt = _check_prompt(prompt)
    model = _check_model(model)
    provider = detect_provider(model)
    api_key = _get_api_key_for_provider(provider)

    log(f"→ توليد نص | الموديل: {model} | المزود: {provider} | طول البرومبت: {len(prompt)}")

    # سيتم تنفيذ الكود الفعلي في المرحلة 2
    raise EngineError("دالة generate لم تُنفذ بعد - المرحلة 2", code="NOT_IMPLEMENTED")


def batch_send(prompts: list, model: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 8192, save_path: str = None) -> EngineResult:
    """
    الدالة 2: إرسال دفعة برومبتات
    تدعم: Gemini Batch, Claude Batch
    """
    start_time = time.time()

    # فحوصات
    prompts = _check_batch_prompts(prompts)
    model = _check_model(model)
    provider = detect_provider(model)
    api_key = _get_api_key_for_provider(provider)

    log(f"→ إرسال دفعة | الموديل: {model} | المزود: {provider} | العدد: {len(prompts)}")

    raise EngineError("دالة batch_send لم تُنفذ بعد - المرحلة 3", code="NOT_IMPLEMENTED")


def batch_retrieve(batch_info_path: str = None, batch_info: BatchInfo = None) -> EngineResult:
    """
    الدالة 3: استقبال نتائج الدفعة
    """
    start_time = time.time()

    # فحوصات
    info = batch_info or (BatchInfo.load(batch_info_path) if batch_info_path else None)
    info = _check_batch_info(info)

    log(f"→ استقبال دفعة | المزود: {info.provider} | المهمة: {info.job_id[:20]}...")

    raise EngineError("دالة batch_retrieve لم تُنفذ بعد - المرحلة 4", code="NOT_IMPLEMENTED")


def tts_elevenlabs(text: str, voice_id: str = None) -> EngineResult:
    """
    الدالة 4: تحويل نص لصوت عبر ElevenLabs
    """
    start_time = time.time()
    text = _check_text_for_tts(text)

    log(f"→ TTS ElevenLabs | طول النص: {len(text)}")

    raise EngineError("دالة tts_elevenlabs لم تُنفذ بعد - المرحلة 5", code="NOT_IMPLEMENTED")


def tts_vertex(text: str, voice: str = "Achird", project_id: str = None, location: str = "europe-west1") -> EngineResult:
    """
    الدالة 5: تحويل نص لصوت عبر Vertex AI
    """
    start_time = time.time()
    text = _check_text_for_tts(text)

    log(f"→ TTS Vertex AI | طول النص: {len(text)} | الصوت: {voice}")

    raise EngineError("دالة tts_vertex لم تُنفذ بعد - المرحلة 6", code="NOT_IMPLEMENTED")


def transcribe(audio_file: str) -> EngineResult:
    """
    الدالة 6: تحويل صوت لنص عبر Whisper
    """
    start_time = time.time()
    audio_file = _check_audio_file(audio_file)

    log(f"→ Whisper | الملف: {os.path.basename(audio_file)}")

    raise EngineError("دالة transcribe لم تُنفذ بعد - المرحلة 7", code="NOT_IMPLEMENTED")
