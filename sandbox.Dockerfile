# صورة Sandbox مع مكتبات AI والأدوات المفيدة
FROM python:3.11-slim

# تثبيت أدوات النظام المطلوبة
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# تثبيت مكتبات Python الأساسية
RUN pip install --no-cache-dir \
    # === مكتبات AI ===
    openai \
    anthropic \
    google-generativeai \
    google-genai \
    google-cloud-aiplatform \
    google-cloud-storage \
    zhipuai \
    langchain \
    langchain-openai \
    langchain-anthropic \
    langchain-google-genai \
    # === معالجة الملفات ===
    python-docx \
    openpyxl \
    pandas \
    PyPDF2 \
    pdfplumber \
    python-pptx \
    Pillow \
    # === معالجة الصوت ===
    pydub \
    # === HTTP والتواصل ===
    requests \
    httpx \
    aiohttp \
    beautifulsoup4 \
    lxml \
    # === معالجة البيانات ===
    numpy \
    pyyaml \
    python-dotenv \
    jinja2 \
    markdown \
    # === أدوات مساعدة ===
    tqdm \
    rich \
    tenacity \
    backoff

# مجلد العمل
WORKDIR /mnt/output

# متغيرات بيئة افتراضية
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
