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
    try:
        print(f"[{timestamp}] [ENGINE] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{timestamp}] [ENGINE] {msg.encode('utf-8', errors='replace').decode('utf-8')}", flush=True)


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

def generate(prompt: str, model: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = None, thinking_budget: int = None, thinking_level: str = None) -> EngineResult:
    """
    الدالة 1: توليد نص من برومبت
    تدعم: Gemini, OpenAI, Claude, GLM, Vertex AI

    ملاحظة: لاستخدام Vertex AI، استخدم "vertex:gemini-2.5-pro" مثلاً
    """
    start_time = time.time()

    # فحوصات ما قبل الإرسال
    prompt = _check_prompt(prompt)
    model = _check_model(model)
    provider = detect_provider(model)

    # Vertex AI مش بيحتاج API key - بيستخدم Google Cloud credentials
    api_key = None if provider == "vertex" else _get_api_key_for_provider(provider)

    log(f"-> generate | model: {model} | provider: {provider} | prompt: {len(prompt)} chars")

    try:
        token_usage = {"input": 0, "output": 0, "thinking": 0, "total": 0}

        if provider == "gemini":
            result_tuple = _retry_call(
                lambda: _generate_gemini(prompt, model, api_key, system_prompt, temperature, max_tokens, thinking_budget, thinking_level),
                max_retries=3, base_delay=3.0, description=f"Gemini {model}"
            )
        elif provider == "vertex":
            # استخرج اسم الموديل الفعلي (بعد "vertex:")
            actual_model = model.split(":", 1)[1] if ":" in model else model
            result_tuple = _retry_call(
                lambda: _generate_vertex(prompt, actual_model, system_prompt, temperature, max_tokens),
                max_retries=3, base_delay=3.0, description=f"Vertex AI {actual_model}"
            )
        elif provider == "openai":
            result_tuple = _retry_call(
                lambda: _generate_openai(prompt, model, api_key, system_prompt, temperature, max_tokens),
                max_retries=3, base_delay=3.0, description=f"OpenAI {model}"
            )
        elif provider == "claude":
            result_tuple = _retry_call(
                lambda: _generate_claude(prompt, model, api_key, system_prompt, temperature, max_tokens),
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

def _generate_gemini(prompt: str, model: str, api_key: str, system_prompt: str, temperature: float, max_tokens: int, thinking_budget: int = None, thinking_level: str = None) -> str:
    """ربط Gemini عبر google.genai SDK الجديدة"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    config_params = {"temperature": temperature, "top_p": 0.95}
    if max_tokens:
        config_params["max_output_tokens"] = max_tokens
    # تحكم في التفكير — thinking_budget=0 له الأولوية القصوى (إلغاء التفكير)
    is_gemini_3 = any(x in model for x in ["gemini-3", "gemini-3.0", "gemini-3.1"])
    if thinking_budget == 0:
        # إلغاء التفكير تماماً — Gemini 3.x يستخدم "none"، الباقي thinking_budget=0
        if is_gemini_3:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_level="none")
            log(f"  [thinking] OFF (thinking_level=none for Gemini 3.x)")
        else:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            log(f"  [thinking] OFF (thinking_budget=0)")
    elif thinking_level and is_gemini_3:
        config_params["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    elif thinking_budget is not None:
        config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    elif thinking_level and not is_gemini_3:
        # fallback: حوّل thinking_level لـ thinking_budget للموديلات القديمة
        level_to_budget = {"low": 1024, "medium": 8192, "high": 24576}
        budget = level_to_budget.get(thinking_level.lower(), 1024)
        config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
    config = types.GenerateContentConfig(**config_params)
    if system_prompt:
        config.system_instruction = system_prompt

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    if not response or not response.text:
        raise ValueError("Gemini returned empty response")

    # تسجيل واستخراج استهلاك التوكنز
    _token_usage = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        um = response.usage_metadata
        _token_usage["input"] = getattr(um, 'prompt_token_count', 0) or 0
        _token_usage["output"] = getattr(um, 'candidates_token_count', 0) or 0
        _token_usage["thinking"] = getattr(um, 'thoughts_token_count', 0) or 0
        _token_usage["total"] = getattr(um, 'total_token_count', 0) or 0
        log(f"  [tokens] input={_token_usage['input']} output={_token_usage['output']} thinking={_token_usage['thinking']} total={_token_usage['total']}")

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
    _token_usage = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    if hasattr(response, 'usage') and response.usage:
        _token_usage["input"] = getattr(response.usage, 'prompt_tokens', 0) or 0
        _token_usage["output"] = getattr(response.usage, 'completion_tokens', 0) or 0
        _token_usage["total"] = _token_usage["input"] + _token_usage["output"]
        log(f"  [tokens] input={_token_usage['input']} output={_token_usage['output']} total={_token_usage['total']}")

    return response.choices[0].message.content, _token_usage


def _generate_claude(prompt: str, model: str, api_key: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
    """ربط Claude عبر anthropic SDK"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    kwargs = {
        "model": model,
        "max_tokens": max_tokens or 16384,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    message = client.messages.create(**kwargs)

    if not message.content or not message.content[0].text:
        raise ValueError("Claude returned empty response")

    # استخراج التوكنز
    _token_usage = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    if hasattr(message, 'usage') and message.usage:
        _token_usage["input"] = getattr(message.usage, 'input_tokens', 0) or 0
        _token_usage["output"] = getattr(message.usage, 'output_tokens', 0) or 0
        _token_usage["total"] = _token_usage["input"] + _token_usage["output"]
        log(f"  [tokens] input={_token_usage['input']} output={_token_usage['output']} total={_token_usage['total']}")

    return message.content[0].text, _token_usage


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
    _token_usage = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    usage = data.get("usage", {})
    if usage:
        _token_usage["input"] = usage.get("prompt_tokens", 0)
        _token_usage["output"] = usage.get("completion_tokens", 0)
        _token_usage["total"] = usage.get("total_tokens", 0) or (_token_usage["input"] + _token_usage["output"])
        log(f"  [tokens] input={_token_usage['input']} output={_token_usage['output']} total={_token_usage['total']}")

    return data["choices"][0]["message"]["content"], _token_usage


def _generate_vertex(prompt: str, model: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
    """ربط Vertex AI لتوليد النصوص (من الوثيقة - طريقة 6)"""
    import json
    import tempfile
    from google import genai
    from google.genai import types

    # جلب بيانات الاعتماد من المتغيرات البيئية
    project_id = os.getenv("GOOGLE_QUOTA_PROJECT", "")
    if not project_id:
        raise ValueError("GOOGLE_QUOTA_PROJECT غير محدد في .env")

    location = os.getenv("GOOGLE_LOCATION", "europe-west1")

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
    config = types.GenerateContentConfig(**config_params)
    if system_prompt:
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
    _token_usage = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        um = response.usage_metadata
        _token_usage["input"] = getattr(um, 'prompt_token_count', 0) or 0
        _token_usage["output"] = getattr(um, 'candidates_token_count', 0) or 0
        _token_usage["thinking"] = getattr(um, 'thoughts_token_count', 0) or 0
        _token_usage["total"] = getattr(um, 'total_token_count', 0) or 0
        log(f"  [tokens] input={_token_usage['input']} output={_token_usage['output']} thinking={_token_usage['thinking']} total={_token_usage['total']}")

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
    """رفع محتوى نصي إلى GCS وإرجاع gs:// URI"""
    from google.cloud import storage

    project_id = os.getenv("GOOGLE_QUOTA_PROJECT", "")
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content, content_type='application/json')

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
    """إرسال دفعة عبر Gemini Batch API (يحتاج GCS)"""
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
        is_gemini_3 = any(x in model for x in ["gemini-3", "gemini-3.0", "gemini-3.1"])
        if thinking_level and is_gemini_3:
            request["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
        elif thinking_budget is not None:
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        elif thinking_level and not is_gemini_3:
            level_to_budget = {"low": 1024, "medium": 8192, "high": 24576}
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": level_to_budget.get(thinking_level.lower(), 1024)}
        if system_prompt:
            request["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        # كل سطر في JSONL لازم يكون {"request": {...}}
        jsonl_lines.append(json.dumps({"request": request}, ensure_ascii=False))

    jsonl_content = "\n".join(jsonl_lines)

    # رفع JSONL إلى GCS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_blob_name = f"batch_input/gemini_{timestamp}.jsonl"
    input_uri = _upload_to_gcs(bucket_name, input_blob_name, jsonl_content)

    log(f"  رفع الطلبات إلى: {input_uri}")

    # إنشاء الـ Batch Job — نجرب الـ location الأصلي، ولو NOT_FOUND نجرب global
    # display_name يحتوي على run_id + recipe للتتبع في Google Cloud Console
    run_id = (labels or {}).get("run_id", "")
    recipe = (labels or {}).get("recipe", "")
    if run_id:
        job_name = f"mgr-{run_id[:8]}-{recipe[:20]}-{timestamp}" if recipe else f"mgr-{run_id[:8]}-{timestamp}"
    else:
        job_name = f"mgr-batch-{timestamp}"
    locations_to_try = [location, "global"] if location != "global" else ["global"]
    last_err = None

    # تنظيف Labels حسب شروط Google Cloud (أحرف صغيرة، أرقام، شرطات فقط، max 63 chars)
    job_labels = {}
    if labels:
        import re as _re
        for k, v in labels.items():
            clean_key = _re.sub(r'[^a-z0-9_-]', '_', str(k).lower())[:63]
            clean_val = _re.sub(r'[^a-z0-9_-]', '_', str(v).lower())[:63]
            if clean_key and clean_val:
                job_labels[clean_key] = clean_val
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
        is_gemini_3 = any(x in model for x in ["gemini-3", "gemini-3.0", "gemini-3.1"])
        if thinking_level and is_gemini_3:
            request["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
        elif thinking_budget is not None:
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        elif thinking_level and not is_gemini_3:
            level_to_budget = {"low": 1024, "medium": 8192, "high": 24576}
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

    response = httpx.post(url, headers=headers, json=payload, timeout=120.0)
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


def _batch_send_vertex(prompts: list, model: str, system_prompt: str, temperature: float, max_tokens: int, thinking_budget: int = None, thinking_level: str = None, labels: dict = None) -> BatchInfo:
    """إرسال دفعة عبر Vertex AI Batch Prediction (من الوثيقة - طريقة 7)"""
    import json
    from google.cloud import aiplatform

    # إعداد GCS
    project_id, location, bucket_name = _setup_gcs_credentials()

    # تهيئة Vertex AI
    aiplatform.init(project=project_id, location=location)

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
        is_gemini_3 = any(x in model for x in ["gemini-3", "gemini-3.0", "gemini-3.1"])
        if thinking_level and is_gemini_3:
            request["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
        elif thinking_budget is not None:
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        elif thinking_level and not is_gemini_3:
            level_to_budget = {"low": 1024, "medium": 8192, "high": 24576}
            request["generationConfig"]["thinkingConfig"] = {"thinkingBudget": level_to_budget.get(thinking_level.lower(), 1024)}
        if system_prompt:
            request["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        jsonl_lines.append(json.dumps({"request": request}, ensure_ascii=False))

    jsonl_content = "\n".join(jsonl_lines)

    # رفع إلى GCS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_blob_name = f"batch_input/vertex_{timestamp}.jsonl"
    input_uri = _upload_to_gcs(bucket_name, input_blob_name, jsonl_content)

    output_uri = f"gs://{bucket_name}/batch_output/vertex_{timestamp}/"

    log(f"  رفع الطلبات إلى: {input_uri}")

    # إنشاء مهمة Batch مع labels للتتبع في Google Cloud Billing
    _run_id = (labels or {}).get("run_id", "")
    _recipe = (labels or {}).get("recipe", "")
    if _run_id:
        job_display_name = f"mgr-{_run_id[:8]}-{_recipe[:20]}-{timestamp}" if _recipe else f"mgr-{_run_id[:8]}-{timestamp}"
    else:
        job_display_name = f"mgr-vertex-{timestamp}"
    job_labels = {}
    if labels:
        # تنظيف Labels حسب شروط Google Cloud (أحرف صغيرة، أرقام، شرطات سفلية فقط، max 63 chars)
        import re as _re
        for k, v in labels.items():
            clean_key = _re.sub(r'[^a-z0-9_-]', '_', str(k).lower())[:63]
            clean_val = _re.sub(r'[^a-z0-9_-]', '_', str(v).lower())[:63]
            if clean_key and clean_val:
                job_labels[clean_key] = clean_val
        if job_labels:
            log(f"  [labels] Vertex Batch Labels: {job_labels}")

    batch_job = aiplatform.BatchPredictionJob.create(
        job_display_name=job_display_name,
        model_name=f"publishers/google/models/{model}",
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=input_uri,
        gcs_destination_prefix=output_uri,
        sync=False,
        labels=job_labels if job_labels else None,
    )

    batch_info = BatchInfo(
        provider="vertex",
        model=model,
        job_id=batch_job.resource_name,
        job_name=batch_job.display_name,
        item_order=list(range(len(prompts))),
        items_count=len(prompts),
        created_at=datetime.now().isoformat(),
        status="submitted",
        extra={"gcs_output": output_uri, "input_uri": input_uri}
    )

    return batch_info


def batch_send(prompts: list, model: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 8192, save_path: str = None, method: str = "sdk", thinking_budget: int = None, thinking_level: str = None, labels: dict = None) -> EngineResult:
    """
    الدالة 2: إرسال دفعة برومبتات
    تدعم: Gemini Batch (SDK/REST), Claude Batch, Vertex AI Batch

    method: "sdk" (افتراضي), "rest" (Gemini REST API), "vertex" (Vertex AI Batch Prediction)
    """
    start_time = time.time()

    # فحوصات
    prompts = _check_batch_prompts(prompts)
    model = _check_model(model)
    provider = detect_provider(model)

    # Vertex لا يحتاج API key (سواء بالـ method أو بالـ provider)
    api_key = None if method == "vertex" or provider == "vertex" else _get_api_key_for_provider(provider)

    log(f"→ إرسال دفعة | الموديل: {model} | المزود: {provider} | الطريقة: {method} | العدد: {len(prompts)}")

    try:
        if provider == "gemini" and method == "sdk":
            batch_info = _retry_call(
                lambda: _batch_send_gemini(prompts, model, api_key, system_prompt, temperature, max_tokens, thinking_budget, thinking_level, labels=labels),
                max_retries=3, base_delay=3.0, description=f"Gemini Batch SDK {model}"
            )
        elif provider == "gemini" and method == "rest":
            batch_info = _retry_call(
                lambda: _batch_send_gemini_rest(prompts, model, api_key, system_prompt, temperature, max_tokens, thinking_budget, thinking_level),
                max_retries=3, base_delay=3.0, description=f"Gemini Batch REST {model}"
            )
        elif (provider == "gemini" or provider == "vertex") and method == "vertex":
            # Vertex AI Batch Prediction
            actual_model = model.split(":", 1)[1] if ":" in model else model
            batch_info = _retry_call(
                lambda: _batch_send_vertex(prompts, actual_model, system_prompt, temperature, max_tokens, thinking_budget, thinking_level, labels=labels),
                max_retries=3, base_delay=3.0, description=f"Vertex AI Batch {actual_model}"
            )
        elif provider == "claude":
            batch_info = _retry_call(
                lambda: _batch_send_claude(prompts, model, api_key, system_prompt, temperature, max_tokens),
                max_retries=3, base_delay=3.0, description=f"Claude Batch {model}"
            )
        else:
            raise EngineError(
                f"Batch غير مدعوم للمزود: {provider} بالطريقة: {method}",
                code="BATCH_NOT_SUPPORTED"
            )

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

    response = httpx.get(batch_url, headers=headers, timeout=60.0)
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


def _batch_retrieve_vertex(batch_info: BatchInfo) -> list:
    """استقبال نتائج دفعة من Vertex AI Batch Prediction (من الوثيقة - طريقة 7)"""
    import json
    from google.cloud import aiplatform
    from google.cloud import storage

    # إعداد GCS
    project_id, location, bucket_name = _setup_gcs_credentials()

    # تهيئة Vertex AI
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
    token_totals = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    for blob in blobs:
        if blob.name.endswith('.jsonl'):
            content = blob.download_as_text()
            for line in content.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    try:
                        text = data['prediction']['candidates'][0]['content']['parts'][0]['text']
                        results.append(text)
                        # استخراج التوكنز
                        usage = data.get('prediction', {}).get('usageMetadata', {})
                        if usage:
                            token_totals["input"] += usage.get('promptTokenCount', 0)
                            token_totals["output"] += usage.get('candidatesTokenCount', 0)
                            token_totals["thinking"] += usage.get('thoughtsTokenCount', 0)
                            token_totals["total"] += usage.get('totalTokenCount', 0)
                    except (KeyError, IndexError) as e:
                        log(f"[!] فشل استخراج نتيجة: {str(e)}")
                        results.append("")

    if token_totals["total"] > 0:
        log(f"  [tokens] إجمالي Vertex Batch: input={token_totals['input']} output={token_totals['output']} thinking={token_totals['thinking']} total={token_totals['total']}")
    batch_info._token_totals = token_totals

    return results


def batch_retrieve(batch_info_path: str = None, batch_info: BatchInfo = None) -> EngineResult:
    """
    الدالة 3: استقبال نتائج الدفعة
    تدعم: Gemini (SDK/REST), Claude, Vertex AI Batch
    """
    start_time = time.time()

    # فحوصات
    info = batch_info or (BatchInfo.load(batch_info_path) if batch_info_path else None)
    info = _check_batch_info(info)

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

    try:
        audio_data = _retry_call(
            lambda: _check_audio_data(_tts_vertex_impl(text, voice, project_id, location), "Vertex AI"),
            max_retries=3, base_delay=3.0, description="Vertex AI TTS"
        )

        duration = int((time.time() - start_time) * 1000)
        log(f"<- TTS Vertex AI OK | {len(audio_data)} bytes | {duration}ms")

        return EngineResult(
            success=True,
            data=audio_data,
            provider="vertex",
            duration_ms=duration
        )

    except EngineError:
        raise
    except Exception as e:
        duration = int((time.time() - start_time) * 1000)
        raise EngineError(
            f"خطأ غير متوقع من Vertex AI TTS: {str(e)[:500]}",
            code="UNEXPECTED_ERROR"
        )


def _tts_vertex_impl(text: str, voice: str, project_id: str, location: str) -> bytes:
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

    # تحويل النص لصوت
    response = client.models.generate_content(
        model="gemini-2.5-pro-tts",
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
