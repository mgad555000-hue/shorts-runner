# خطة نقل وصفات Python Runner إلى MG Ranner
> تاريخ الإنشاء: 2026-02-21
> الحالة: **مكتمل** ✅ (2026-02-22)

---

## المرحلة 0 — التحضير (مرة واحدة) ✅

### 0.1 إضافة أكشن `remove_tashkeel` لـ recipe_runner.py
- [x] إضافة الدالة: `action_remove_tashkeel(step, ctx)`
- [x] المنطق: `re.compile(r'[\u064B-\u065F]')` — إزالة التشكيل من النص
- [x] تسجيل الأكشن في dict الـ ACTIONS
- [x] البارامترات: `input` (مطلوب) + `save_as` (اختياري)

### 0.2 إعادة بناء Docker
- [x] `docker-compose build shorts-app`
- [x] `docker-compose restart shorts-app`
- [x] التحقق: API يرد على `/api/utilities/recipes`

### 0.3 اختبار البنية التحتية
- [x] التحقق من أن batch_send يعمل (إرسال برومبت واحد اختباري)
- [x] التحقق من أن batch_retrieve يسترجع النتيجة
- [x] التحقق من أن format_text يطبق المسافات والتلوين

---

## المرحلة 1 — الوصفات الأساسية ✅

### وصفة 1: توليد سكريبتات (long) ✅ — DB ID=11
**المصدر:** `script_generator.py` (614 سطر)
**الأكشنات:** read_input → read_input → template → generate → format_text → save_docx

#### الخطوات:
- [x] 1.1 استخراج SCRIPT_INSTRUCTIONS من script_generator.py
- [x] 1.2 تعديل قسم المخرجات فقط لاستخدام ماركرز MG Ranner (`<<<SCRIPT_N>>>`)
- [x] 1.3 حفظ البرومبت في instructions.txt
- [x] 1.4 إنشاء topics.json (موضوعين اختباريين)
- [x] 1.5 كتابة ملف JSON الوصفة
- [x] 1.6 إنشاء مجلدات input/output للقنوات الـ 3
- [x] 1.7 تسجيل في DB (Python sqlite3)
- [x] 1.9 تشغيل الـ checklist الإلزامي (10 بنود) — 10/10 PASS
- [ ] 1.10 اختبار بموضوعين (في انتظار التشغيل)
- [ ] 1.11 مقارنة المخرج مع Python Runner

---

### وصفة 2: إزالة تشكيل ✅ — DB ID=12
**المصدر:** `remove_tashkeel_python.py` (478 سطر)
**الأكشنات:** read_docx → remove_tashkeel → save_docx

#### الخطوات:
- [x] 2.1 كتابة ملف JSON الوصفة (3 خطوات بسيطة)
- [x] 2.2 إنشاء مجلدات input/output للقنوات الـ 3
- [x] 2.3 تسجيل في DB
- [x] 2.5 تشغيل الـ checklist — 10/10 PASS
- [ ] 2.6 اختبار: إدخال ملف Word مشكّل → التحقق من إزالة التشكيل

---

### وصفة 3: معالجة اللهجة المصرية ✅ — DB ID=13
**المصدر:** `egyptian_dialect_processor_send.py` + `_retrieve.py` (1098 سطر)
**الأكشنات:** read_input → read_docx → template → generate → format_text → save_docx

#### الخطوات:
- [x] 3.1 استخراج EGYPTIAN_DIALECT_INSTRUCTIONS من الكود
- [x] 3.2 تعديل قسم المخرجات لماركرز MG Ranner
- [x] 3.3 حفظ في instructions.txt
- [x] 3.5 كتابة JSON الوصفة
- [x] 3.6 إنشاء مجلدات للقنوات الـ 3
- [x] 3.7 تسجيل في DB
- [x] 3.9 تشغيل الـ checklist — 10/10 PASS
- [ ] 3.10 اختبار بموضوعين

---

### وصفة 4: تجهيز نصوص للصوت (TTS Prep) ✅ — DB ID=14
**المصدر:** `tts_prep_send.py` + `_retrieve.py` (722 سطر)
**الأكشنات:** read_input → read_docx → template → generate → save_docx

#### الخطوات:
- [x] 4.1 استخراج TTS_PREP_INSTRUCTIONS من الكود
- [x] 4.2 تعديل قسم المخرجات لماركرز MG Ranner
- [x] 4.3 حفظ في instructions.txt
- [x] 4.5 كتابة JSON الوصفة
- [x] 4.6 إنشاء مجلدات للقنوات الـ 3
- [x] 4.7 تسجيل في DB
- [x] 4.9 تشغيل الـ checklist — 10/10 PASS
- [ ] 4.10 اختبار

---

## المرحلة 2 — الوصفات المتقدمة ✅

### وصفة 5: تحويل نص لصوت (TTS) ✅ — DB ID=15
**المصدر:** `tts_generator.py` + `tts_batch_generator.py` (903 سطر)
**الأكشنات:** read_docx (reconstruct_markers=true) → tts_multi

#### الخطوات:
- [x] 5.1 كتابة JSON الوصفة (خطوتين بسيطتين)
- [x] 5.2 إنشاء مجلدات للقنوات الـ 3
- [x] 5.3 تسجيل في DB
- [x] 5.5 تشغيل الـ checklist — 10/10 PASS
- [ ] 5.6 اختبار بموضوع واحد

---

### وصفة 6: فيديوهات توضيحية ✅ — DB ID=16
**المصدر:** `illustrative_videos_vertexai_send.py` + `_retrieve.py` (1256 سطر)
**الأكشنات:** read_input → read_excel → read_docx → template → generate → save_docx → copy_videos

#### الخطوات:
- [x] 6.1 استخراج VIDEOS_LIST_INSTRUCTIONS + GENERATION_PROMPT
- [x] 6.2 تعديل لماركرز MG Ranner
- [x] 6.3 حفظ في instructions.txt
- [x] 6.6 كتابة JSON الوصفة
- [x] 6.7 إنشاء مجلدات للقنوات الـ 3
- [x] 6.8 تسجيل في DB
- [x] 6.10 تشغيل الـ checklist — 10/10 PASS
- [ ] 6.11 اختبار بموضوع واحد
- [ ] 6.4 نسخ ملفات Excel القنوات (videos_list.xlsx) لمجلدات input — **مطلوب قبل التشغيل**

---

### وصفة 7: تدقيق محتوى ✅ — DB ID=17
**المصدر:** `content_validator.py` (655 سطر)
**الأكشنات:** read_input → read_docx(titles) → read_docx(intros) → read_docx(texts) → template → generate → save_file

#### الخطوات:
- [x] 7.1 استخراج VALIDATION_PROMPT
- [x] 7.2 حفظ في instructions.txt
- [x] 7.3 كتابة JSON الوصفة
- [x] 7.4 إنشاء مجلدات للقنوات الـ 3
- [x] 7.5 تسجيل في DB
- [x] 7.7 تشغيل الـ checklist — 10/10 PASS
- [ ] 7.8 اختبار بموضوع واحد

---

## القواعد الذهبية (تطبّق على كل وصفة)

1. **البرومبت = نسخة طبق الأصل** — أقل تعديل ممكن (بس قسم المخرجات يتغير للماركرز) ✅
2. **كل وصفة تتبني وتتختبر** قبل الانتقال للي بعدها ✅
3. **الـ checklist الإلزامي** (10 بنود من CLAUDE.md) لازم PASS ✅ (كل الـ 7 = 10/10)
4. **مجلدات للقنوات الـ 3** — My_Kidney + Alhashab2000 + Social_relations ✅
5. **DB registration:** Python sqlite3 مباشرة (مش curl) ✅
6. **Docker restart** بعد كل تعديل Python ✅
7. **المقارنة:** نفس المدخلات في التطبيقين ونقارن المخرجات — في انتظار التشغيل

---

## ملخص تقني

| الوصفة | DB ID | recipe_type | Temperature | format_text | الأكشنات |
|--------|-------|-------------|-------------|-------------|----------|
| توليد سكريبتات | 11 | long | 0.7 | نعم | read_input → template → generate → format_text → save_docx |
| إزالة تشكيل | 12 | long | — | لا | read_docx → remove_tashkeel → save_docx |
| معالجة لهجة مصرية | 13 | long | 0.3 | نعم | read_input → read_docx → template → generate → format_text → save_docx |
| تجهيز نصوص للصوت | 14 | long | 0.3 | لا | read_input → read_docx → template → generate → save_docx |
| تحويل نص لصوت | 15 | long | — | لا | read_docx → tts_multi |
| فيديوهات توضيحية | 16 | long | 0.3 | لا | read_input → read_excel → read_docx → template → generate → save_docx → copy_videos |
| تدقيق محتوى | 17 | long | 0.1 | لا | read_input → read_docx(3) → template → generate → save_file |

---

## ملاحظات للتشغيل

1. **وصفة 6 (فيديوهات توضيحية)**: محتاجة `videos_list.xlsx` في input كل قناة + مجلد videos فيه ملفات MP4
2. **وصفة 7 (تدقيق محتوى)**: محتاجة 3 ملفات Word في input: `titles.docx` + `intros.docx` + `texts.docx`
3. **الموديل الافتراضي**: يتحدد من إعدادات Docker (gemini-2.5-flash) — ممكن يتغير من الواجهة
4. **thinking_budget**: غير مدعوم حالياً في engine.py — ممكن يتضاف لاحقاً لو لزم
