# وثيقة بناء تطبيق Shorts Runner

## القرارات المتفق عليها

### 1. نظام القنوات
- كل قناة مجلد منفصل في `data/channels/`
- القناة تتختار من dropdown في الواجهة
- كل وصفة ليها `input/` و `output/` جوه مجلد القناة
- إضافة قناة جديدة = مجلد جديد وخلاص
- الفيديوهات مربوطة من الهارد الخارجي (mount)

### 2. الوصفات
- التعليمات كلها جوه كود الوصفة (مش ملفات خارجية)
- التعليمات عامة تصلح لأي قناة
- مجلدات input و output فاضية تماماً (بس ملفات العمل)
- الوصفات مشتركة بين كل القنوات
- نبني الوصفات مرحلة مرحلة وبموافقة قبل كل وصفة

### 3. اختيار الموديل
- dropdown في الواجهة لاختيار الموديل قبل التشغيل
- الموديل مش ثابت جوه الوصفة - يتختار من الواجهة
- ده يدي مرونة لتجربة أي وصفة بأي موديل
- نظام الموديلات مفتوح - القائمة مخزنة في قاعدة البيانات
- لإضافة موديل جديد: نضيفه في القائمة ونكتب كود الاتصال مرة واحدة

---

## نتائج اختبار الموديلات (تم التجربة 2026-02-06)

### كل الموديلات المجربة والناجحة:

| الموديل | النتيجة | طريقة الربط | ملاحظة |
|---|---|---|---|
| gemini-2.5-flash | نجح | google.genai SDK | المكتبة الجديدة |
| gemini-2.5-pro | نجح | google.genai SDK | |
| gemini-3-pro-preview | نجح | google.genai SDK | |
| gpt-4o-mini | نجح | OpenAI SDK | يستخدم max_tokens |
| gpt-4.1-mini | نجح | OpenAI SDK | يستخدم max_tokens |
| gpt-4.1-nano | نجح | OpenAI SDK | يستخدم max_tokens |
| gpt-5-nano | نجح | OpenAI SDK | يستخدم max_completion_tokens (مهم!) |
| gpt-5-mini | نجح | OpenAI SDK | يستخدم max_completion_tokens |
| gpt-5 | نجح | OpenAI SDK | يستخدم max_completion_tokens |
| gpt-5.1 | نجح | OpenAI SDK | يستخدم max_completion_tokens |
| gpt-5.2 | نجح | OpenAI SDK | يستخدم max_completion_tokens |
| claude-sonnet-4-20250514 | نجح | Anthropic SDK | |
| claude-opus-4-20250514 | نجح | Anthropic SDK | |
| glm-4-plus | نجح | httpx REST | |
| gemini-2.5-pro-tts | نجح | Vertex AI | TTS فقط |

### اكتشاف مهم:
- مكتبة google.generativeai القديمة هتتوقف - لازم نستخدم google.genai الجديدة
- موديلات GPT-5 بتستخدم `max_completion_tokens` بدل `max_tokens` (لو استخدمت القديم هيدي خطأ)
- موديلات GPT-4 لسه بتستخدم `max_tokens`

---

## الموديلات المتاحة وطرق الربط الناجحة

### طريقة 1: Gemini عبر google.genai SDK (الجديدة - المعتمدة)
- **الموديلات:** gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview
- **المفتاح:** GEMINI_API_KEY
- **الكود الناجح:**
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=8192,
        system_instruction=SYSTEM_PROMPT,  # اختياري
    ),
)

result_text = response.text
```

### طريقة 1 (قديمة - ستتوقف): Gemini عبر google.generativeai
- **تحذير:** هذه المكتبة ستتوقف. استخدم google.genai بدلاً منها
- **الكود:**
```python
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    MODEL_NAME,
    safety_settings=safety_settings,
    system_instruction=SYSTEM_PROMPT
)

response = await model.generate_content_async(
    prompt,
    generation_config=genai.GenerationConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=8192,
    ),
)

result_text = response.text
```

### طريقة 2: OpenAI عبر OpenAI SDK (جديد - تم تجربته)
- **موديلات GPT-4:** gpt-4o-mini, gpt-4.1-mini, gpt-4.1-nano
- **موديلات GPT-5:** gpt-5-nano, gpt-5-mini, gpt-5, gpt-5.1, gpt-5.2
- **المفتاح:** OPENAI_API_KEY
- **الكود الناجح (GPT-4):**
```python
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model=MODEL_NAME,  # مثل gpt-4o-mini
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=8192,  # GPT-4 يستخدم max_tokens
)

result_text = response.choices[0].message.content
```

- **الكود الناجح (GPT-5) - مهم الفرق:**
```python
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model=MODEL_NAME,  # مثل gpt-5-mini
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_completion_tokens=8192,  # GPT-5 يستخدم max_completion_tokens (مش max_tokens!)
)

result_text = response.choices[0].message.content
```

- **دالة موحدة تتعامل مع الاتنين:**
```python
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

is_gpt5 = MODEL_NAME.startswith("gpt-5")
token_param = {"max_completion_tokens": 8192} if is_gpt5 else {"max_tokens": 8192}

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    **token_param,
)

result_text = response.choices[0].message.content
```

### طريقة 4: Gemini Batch عبر REST API
- **الموديلات:** gemini-3-pro-preview, gemini-2.5-pro
- **المفتاح:** GEMINI_API_KEY
- **الكود الناجح:**
```python
import requests as http_requests

url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchGenerateContent"

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": GEMINI_API_KEY
}

# إرسال
payload = {
    "batch": {
        "display_name": "job-name",
        "input_config": {
            "requests": {
                "requests": inline_requests  # قائمة الطلبات
            }
        }
    }
}
response = http_requests.post(url, headers=headers, json=payload, timeout=120)
job_name = response.json().get('name', '')

# استرجاع
batch_url = f"https://generativelanguage.googleapis.com/v1beta/{job_name}"
response = http_requests.get(batch_url, headers=headers, timeout=60)
```

### طريقة 5: Gemini Batch عبر google.genai SDK
- **الموديلات:** gemini-3-pro-preview
- **المفتاح:** GEMINI_API_KEY
- **الكود الناجح:**
```python
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

# إرسال
batch_job = client.batches.create(
    model=f"models/{MODEL}",
    src=inline_requests,
    config={'display_name': 'job-name'},
)

# استرجاع
batch_job = client.batches.get(name=job_name)
for response in batch_job.dest.inlined_responses:
    text = response.response.candidates[0].content.parts[0].text
```

### طريقة 6: Vertex AI (google.genai مع vertexai=True)
- **الموديلات:** gemini-2.5-pro, gemini-2.5-pro-tts
- **المفتاح:** Google Cloud ADC (بيانات اعتماد)
- **الكود الناجح:**
```python
from google import genai
from google.genai import types

# بيانات الاعتماد
CREDS = {
    "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
    "refresh_token": os.getenv("GOOGLE_REFRESH_TOKEN", ""),
    "quota_project_id": os.getenv("GOOGLE_QUOTA_PROJECT", ""),
    "type": "authorized_user",
    "universe_domain": "googleapis.com"
}
creds_file = os.path.join(tempfile.gettempdir(), 'gcp_creds.json')
with open(creds_file, 'w') as f:
    json.dump(CREDS, f)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# نص عادي
response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=16384)
)

# TTS (تحويل نص لصوت)
response = client.models.generate_content(
    model="gemini-2.5-pro-tts",
    contents=text,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Achird")
            )
        )
    )
)
audio_data = response.candidates[0].content.parts[0].inline_data.data
```

### طريقة 7: Vertex AI Batch Prediction
- **الموديلات:** gemini-2.5-pro
- **المفتاح:** Google Cloud ADC
- **الكود الناجح:**
```python
from google.cloud import aiplatform

aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION)

batch_job = aiplatform.BatchPredictionJob.create(
    job_display_name="job-name",
    model_name="publishers/google/models/gemini-2.5-pro",
    instances_format="jsonl",
    predictions_format="jsonl",
    gcs_source=input_uri,           # gs://bucket/input.jsonl
    gcs_destination_prefix=output_uri,  # gs://bucket/output/
    sync=False,
)
```

### طريقة 8: Claude عبر Anthropic SDK
- **الموديلات:** claude-opus-4-20250514, claude-sonnet-4-20250514
- **المفتاح:** CLAUDE_API_KEY
- **الكود الناجح:**
```python
import anthropic

# رسالة واحدة (async)
client = anthropic.AsyncAnthropic(api_key=CLAUDE_API_KEY)
message = await client.messages.create(
    model=MODEL_NAME,
    max_tokens=8192,
    temperature=0.7,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": prompt}],
)
result_text = message.content[0].text

# Batch
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
batch = client.batches.create(requests=[
    {
        "custom_id": "1",
        "params": {
            "model": MODEL_NAME,
            "max_tokens": 8192,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
    }
])

# استرجاع Batch
batch_status = client.batches.retrieve(batch_id)
for result in client.batches.results(batch_id):
    text = result.result.message.content[0].text
```

### طريقة 9: GLM عبر REST API
- **الموديلات:** glm-4-plus
- **المفتاح:** GLM_API_KEY
- **الكود الناجح:**
```python
import httpx

async with httpx.AsyncClient(timeout=180.0) as client:
    response = await client.post(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        headers={
            "Authorization": f"Bearer {GLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 8192,
        },
    )
    response.raise_for_status()
    result_text = response.json()["choices"][0]["message"]["content"]
```

---

## هيكل المجلدات

```
shorts-runner/
  data/
    channels/
      My_Kidney/
        videos_list.xlsx
        videos/              (mount من الهارد)
        وصفة_1/
          input/             (فاضي - بتحط فيه ملفاتك)
          output/            (فاضي - الوصفة بتكتب فيه)
        وصفة_2/
          input/
          output/
      Alhashab2000/
        videos_list.xlsx
        videos/
        ...
      Social_relations/
        ...
```

---

## الواجهة

- dropdown للقناة
- dropdown للوصفة
- dropdown للموديل (جديد - لم يُنفذ بعد)
- مجلدات input/output تتحدد تلقائي

---

## مراحل لم تُحدد بعد

- تفاصيل وصفات الشورتس (المراحل والخطوات)
- ترتيب بناء الوصفات
