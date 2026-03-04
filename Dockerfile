# استخدام صورة بايثون رسمية كأساس
FROM python:3.11-slim

# تثبيت Docker CLI (لتنفيذ sandbox containers)
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    ffmpeg \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/* \
    && docker --version

# التأكد من صلاحيات الوصول لـ docker.sock (سيتم ربطه من docker-compose)
RUN groupadd -f docker || true

# تعيين مجلد العمل
WORKDIR /app

# نسخ ملف المتطلبات
COPY requirements.txt .

# تثبيت مكتبات بايثون
RUN pip install --no-cache-dir -r requirements.txt

# تثبيت مكتبات إضافية للتنفيذ المحلي (fallback)
RUN pip install --no-cache-dir \
    openpyxl \
    google-generativeai \
    google-cloud-aiplatform \
    google-cloud-storage \
    anthropic \
    httpx

# مكتبات المونتاج (رسم عناوين عربية + معالجة صور)
RUN pip install --no-cache-dir \
    Pillow \
    arabic-reshaper \
    python-bidi

# Playwright for thumbnail generation (high-quality Arabic text rendering)
RUN pip install --no-cache-dir playwright \
    && playwright install chromium \
    && playwright install-deps chromium

# تحميل خط عربي للعناوين
RUN mkdir -p /app/fonts \
    && curl -sL 'https://github.com/google/fonts/raw/main/ofl/notokuficarabic/static/NotoKufiArabic-Bold.ttf' \
       -o /app/fonts/NotoKufiArabic-Bold.ttf

# نسخ باقي ملفات التطبيق
COPY . .

# إنشاء مجلدات البيانات
RUN mkdir -p /app/data /app/utilities/out /app/static

# تعيين المنفذ
EXPOSE 8000

# تشغيل التطبيق
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
