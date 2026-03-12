"""
Docker Sandbox Executor
تنفيذ كود بايثون في حاوية معزولة
"""
import docker
import uuid
import os
import json
import shutil
from pathlib import Path
from typing import Tuple, Optional
import subprocess

# مسارات البيانات
DATA_ROOT = os.getenv("DATA_ROOT", "./data")
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "./utilities/out")

# اسم صورة الـ Sandbox (تحتوي على python-docx)
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "python-runner-sandbox")

# تحميل مفاتيح API من ملف .env
def load_env_file():
    """تحميل متغيرات البيئة من ملف .env (يُستدعى عند كل تشغيلة لقراءة أحدث القيم)"""
    env_vars = {}
    # جرب عدة مسارات للعثور على .env
    possible_paths = [
        Path(__file__).parent.parent / ".env",  # c:\python-runner\.env
        Path("c:/python-runner/.env"),
        Path(".env"),
    ]
    
    env_file = None
    for p in possible_paths:
        if p.exists():
            env_file = p
            break
    
    if env_file and env_file.exists():
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if value:  # فقط إذا كانت القيمة غير فارغة
                            env_vars[key] = value
        except Exception as e:
            print(f"[load_env_file] Error reading {env_file}: {e}")
    
    return env_vars

# تُستخدم كقيم افتراضية عند عدم التشغيل؛ عند التنفيذ نقرأ .env من جديد
ENV_VARS = load_env_file()


def _get_api_env():
    """قراءة .env الحالي ودمجه مع البيئة — يُستدعى عند كل تشغيلة حتى يصل المفتاح."""
    return load_env_file()

# إعداد اتصال Docker (يقرأ من متغير البيئة إن وجد)
DOCKER_HOST = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")

# التأكد من وجود مجلد الإخراج
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

# إعداد Docker client
docker_client = None
try:
    docker_client = docker.DockerClient(base_url=DOCKER_HOST)
    # التحقق من أن Docker client يعمل
    docker_client.ping()
    print(f"Docker client connected successfully (DOCKER_HOST={DOCKER_HOST})")
except Exception as e:
    docker_client = None
    print(f"Warning: Docker client initialization failed (DOCKER_HOST={DOCKER_HOST}): {e}")
    print("Make sure Docker daemon is running and DOCKER_HOST is correct.")


def create_sandbox_container(run_id: str, code: str, input_folder: str, content_type: str = "shorts") -> Tuple[bool, str, Optional[str]]:
    """
    إنشاء وتشغيل حاوية sandbox لتنفيذ الكود

    Returns:
        (success, output_path, error_message)
    """
    try:
        # إنشاء مجلد الإخراج حسب نوع المحتوى
        output_root = os.getenv("LONGS_OUTPUT_ROOT", "./longs/out") if content_type == "long" else OUTPUT_ROOT
        output_dir = Path(output_root) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # التأكد من وجود مجلد الإدخال
        input_dir = Path(DATA_ROOT) / input_folder
        if not input_dir.exists():
            return False, str(output_dir), f"مجلد الإدخال غير موجود: {input_folder}"

        # كشف JSON Pipeline
        is_json_pipeline = False
        if code and code.strip().startswith("{"):
            try:
                json.loads(code)
                is_json_pipeline = True
            except json.JSONDecodeError:
                pass

        if is_json_pipeline:
            # حفظ الـ config
            config_path = output_dir / "recipe_config.json"
            config_path.write_text(code, encoding="utf-8")
            print(f"[SANDBOX DEBUG] JSON Pipeline detected, config saved: {config_path.stat().st_size} bytes")
            # توليد wrapper script
            wrapper = '''import sys, json, os
sys.path.insert(0, '/app/app')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))
from recipe_runner import run_pipeline
config_path = os.path.join(os.environ.get("OUTPUT_DIR", "/mnt/output"), "recipe_config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
run_pipeline(config)
'''
            script_path = output_dir / "script.py"
            script_path.write_text(wrapper, encoding="utf-8")
            print(f"[SANDBOX DEBUG] Wrapper script written, file size: {script_path.stat().st_size}")
        else:
            # كتابة الكود في مجلد الإخراج
            script_path = output_dir / "script.py"
            print(f"[SANDBOX DEBUG] Code length to write: {len(code) if code else 0}")
            script_path.write_text(code, encoding="utf-8")
            print(f"[SANDBOX DEBUG] Script written, file size: {script_path.stat().st_size}")

        # مسارات ملفات السجل والـ manifest
        log_path = output_dir / "run_log.txt"
        manifest_path = output_dir / "result_manifest.json"

        # إذا كان Docker متاحاً نستخدمه، وإلا ننفذ محلياً داخل نفس الحاوية
        if docker_client is not None:
            try:
                container = docker_client.containers.run(
                    image=SANDBOX_IMAGE,  # صورة مخصصة فيها python-docx
                    command=["python", "/mnt/output/script.py"],
                    volumes={
                        str(input_dir.resolve()): {"bind": "/mnt/input", "mode": "ro"},
                        str(output_dir.resolve()): {"bind": "/mnt/output", "mode": "rw"},
                    },
                    environment={
                        "INPUT_DIR": "/mnt/input",
                        "OUTPUT_DIR": "/mnt/output",
                        "PYTHONUNBUFFERED": "1",
                        "MODEL_NAME": os.getenv("MODEL_NAME", ""),
                        "CHANNEL_NAME": os.getenv("CHANNEL_NAME", ""),
                        "CHANNEL_ROOT": os.getenv("CHANNEL_ROOT", ""),
                        "RECIPE_OUTPUT_DIR": os.getenv("RECIPE_OUTPUT_DIR", ""),
                        "TTS_PROVIDER": os.getenv("TTS_PROVIDER", "vertex"),
                        "TTS_VOICE_ID": _get_api_env().get("TTS_VOICE_ID", "Achird"),
                        "EXECUTION_MODE": os.getenv("EXECUTION_MODE", "instant"),
                        "TOPIC_IDS": os.getenv("TOPIC_IDS", ""),
                        # تمرير مفاتيح API (قراءة .env من المشروع عند كل تشغيلة)
                        "GEMINI_API_KEY": _get_api_env().get("GEMINI_API_KEY", ""),
                        "CLAUDE_API_KEY": _get_api_env().get("CLAUDE_API_KEY", ""),
                        "GLM_API_KEY": _get_api_env().get("GLM_API_KEY", ""),
                        "OPENAI_API_KEY": _get_api_env().get("OPENAI_API_KEY", ""),
                        "ELEVENLABS_API_KEY": _get_api_env().get("ELEVENLABS_API_KEY", ""),
                        "MINIMAX_API_KEY": _get_api_env().get("MINIMAX_API_KEY", ""),
                        "GOOGLE_CLIENT_ID": _get_api_env().get("GOOGLE_CLIENT_ID", ""),
                        "GOOGLE_CLIENT_SECRET": _get_api_env().get("GOOGLE_CLIENT_SECRET", ""),
                        "GOOGLE_REFRESH_TOKEN": _get_api_env().get("GOOGLE_REFRESH_TOKEN", ""),
                        "GOOGLE_QUOTA_PROJECT": _get_api_env().get("GOOGLE_QUOTA_PROJECT", ""),
                    },
                    network_disabled=False,  # تفعيل الإنترنت للتواصل مع APIs
                    mem_limit="2g",  # زيادة الذاكرة للمهام الكبيرة
                    cpu_count=4,  # زيادة المعالجات
                    pids_limit=200,  # زيادة عدد العمليات
                    tmpfs={"/tmp": "size=500m"},  # زيادة مساحة /tmp
                    detach=True,
                    remove=False,  # لا نحذف تلقائياً — نحتاج نقرأ اللوج أولاً
                    working_dir="/mnt/output",
                )

                # كتابة header اللوج فوراً + stream اللوج في real-time
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"=== Run ID: {run_id} ===\n")
                    f.write(f"Input Folder: {input_folder}\n")
                    f.write(f"Exit Code: running...\n\n")
                    f.write("=== Execution Logs ===\n")

                logs_lines = []
                with open(log_path, "a", encoding="utf-8") as f:
                    for chunk in container.logs(stream=True, follow=True):
                        line = chunk.decode("utf-8", errors="ignore")
                        logs_lines.append(line)
                        f.write(line)
                        f.flush()

                # انتظار انتهاء التنفيذ والحصول على exit code
                result = container.wait(timeout=21600)  # 6 ساعات
                logs = "".join(logs_lines)

                # تحديث header بـ exit code النهائي
                with open(log_path, "r", encoding="utf-8") as f:
                    content = f.read()
                content = content.replace("Exit Code: running...", f"Exit Code: {result['StatusCode']}", 1)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(content)

                # حذف الكونتينر بعد الانتهاء
                try:
                    container.remove()
                except Exception:
                    pass

                # إنشاء manifest
                exit_code = result["StatusCode"]
                success = exit_code == 0

                manifest = {
                    "run_id": run_id,
                    "input_folder": input_folder,
                    "status": "completed" if success else "failed",
                    "exit_code": exit_code,
                    "output_files": [
                        str(f.name) for f in output_dir.iterdir() if f.is_file()
                    ],
                    "logs_preview": logs[:500] if logs else "",
                }

                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)

                if not success:
                    return (
                        False,
                        str(output_dir),
                        f"فشل التنفيذ مع كود الخروج: {exit_code}\n\nLogs:\n{logs[:1000]}",
                    )

                return True, str(output_dir), None

            except docker.errors.ContainerError as e:
                error_msg = f"خطأ في الحاوية: {str(e)}"
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"Error: {error_msg}\n")
                return False, str(output_dir), error_msg

            except Exception as e:
                error_msg = f"خطأ غير متوقع أثناء استخدام Docker: {str(e)}"
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"Error: {error_msg}\n")
                return False, str(output_dir), error_msg

        # Fallback: تنفيذ محلي داخل نفس الحاوية بدون Docker
        try:
            api_env = _get_api_env()
            
            # تشخيص: عرض المفاتيح المقروءة من .env
            print(f"[SANDBOX DEBUG] .env path: {Path(__file__).parent.parent / '.env'}")
            print(f"[SANDBOX DEBUG] api_env keys: {list(api_env.keys())}")
            print(f"[SANDBOX DEBUG] OPENAI_API_KEY in api_env: {'Yes' if api_env.get('OPENAI_API_KEY') else 'No'}")
            
            # مسار بيانات اعتماد Google Cloud (ADC)
            appdata = os.environ.get('APPDATA', '')
            gcloud_dir = Path(appdata) / 'gcloud' if appdata else Path.home() / '.config' / 'gcloud'
            gcloud_creds = gcloud_dir / 'application_default_credentials.json'
            
            print(f"[SANDBOX DEBUG] Google ADC path: {gcloud_creds}, exists: {gcloud_creds.exists()}")
            print(f"[SANDBOX DEBUG] gcloud dir: {gcloud_dir}")
            
            # Use absolute paths
            abs_input_dir = input_dir.resolve()
            abs_output_dir = output_dir.resolve()
            
            env = os.environ.copy()
            env.update(
                {
                    "INPUT_DIR": str(abs_input_dir),
                    "OUTPUT_DIR": str(abs_output_dir),
                    "PYTHONUNBUFFERED": "1",
                    # تمرير اسم الموديل المختار من الواجهة
                    "MODEL_NAME": os.getenv("MODEL_NAME", ""),
                    "TTS_PROVIDER": os.getenv("TTS_PROVIDER", "vertex"),
                    "TTS_VOICE_ID": api_env.get("TTS_VOICE_ID", "Achird"),
                    "EXECUTION_MODE": os.getenv("EXECUTION_MODE", "instant"),
                    "TOPIC_IDS": os.getenv("TOPIC_IDS", ""),
                    # تمرير مفاتيح API (قراءة .env من المشروع عند كل تشغيلة)
                    "GEMINI_API_KEY": api_env.get("GEMINI_API_KEY", ""),
                    "CLAUDE_API_KEY": api_env.get("CLAUDE_API_KEY", ""),
                    "GLM_API_KEY": api_env.get("GLM_API_KEY", ""),
                    "OPENAI_API_KEY": api_env.get("OPENAI_API_KEY", ""),
                    "ELEVENLABS_API_KEY": api_env.get("ELEVENLABS_API_KEY", ""),
                    "MINIMAX_API_KEY": api_env.get("MINIMAX_API_KEY", ""),
                    "CHANNEL_NAME": os.getenv("CHANNEL_NAME", ""),
                    "CHANNEL_ROOT": os.getenv("CHANNEL_ROOT", ""),
                    "RECIPE_OUTPUT_DIR": os.getenv("RECIPE_OUTPUT_DIR", ""),
                    "GOOGLE_CLIENT_ID": api_env.get("GOOGLE_CLIENT_ID", ""),
                    "GOOGLE_CLIENT_SECRET": api_env.get("GOOGLE_CLIENT_SECRET", ""),
                    "GOOGLE_REFRESH_TOKEN": api_env.get("GOOGLE_REFRESH_TOKEN", ""),
                    "GOOGLE_QUOTA_PROJECT": api_env.get("GOOGLE_QUOTA_PROJECT", ""),
                    # بيانات اعتماد Google Cloud لـ Vertex AI
                    "GOOGLE_APPLICATION_CREDENTIALS": str(gcloud_creds) if gcloud_creds.exists() else "",
                    "CLOUDSDK_CONFIG": str(gcloud_dir) if gcloud_dir.exists() else "",
                    "APPDATA": appdata,  # Pass APPDATA for credential lookup
                }
            )

            # Use absolute paths to avoid path confusion
            abs_script_path = script_path.resolve()

            # كتابة header اللوج مبدئياً
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== Run ID: {run_id} (local execution) ===\n")
                f.write(f"Input Folder: {input_folder}\n")
                f.write(f"Exit Code: running...\n\n")
                f.write("=== Execution Logs ===\n")

            # تشغيل بـ Popen عشان نكتب اللوج أول بأول
            proc = subprocess.Popen(
                ["python", "-u", str(abs_script_path)],
                cwd=str(abs_output_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # قراءة وكتابة اللوج سطر بسطر
            logs_lines = []
            with open(log_path, "a", encoding="utf-8") as f:
                for line in proc.stdout:
                    logs_lines.append(line)
                    f.write(line)
                    f.flush()

            proc.wait(timeout=21600)  # 6 ساعات للباتش
            logs = "".join(logs_lines)

            # تحديث header بـ exit code النهائي
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace("Exit Code: running...", f"Exit Code: {proc.returncode}", 1)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(content)

            success = proc.returncode == 0

            manifest = {
                "run_id": run_id,
                "input_folder": input_folder,
                "status": "completed" if success else "failed",
                "exit_code": proc.returncode,
                "output_files": [
                    str(f.name) for f in output_dir.iterdir() if f.is_file()
                ],
                "logs_preview": logs[:500] if logs else "",
                "mode": "local",
            }

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            if not success:
                return (
                    False,
                    str(output_dir),
                    f"فشل التنفيذ (local) مع كود الخروج: {proc.returncode}\n\nLogs:\n{logs[:1000]}",
                )

            return True, str(output_dir), None

        except subprocess.TimeoutExpired as e:
            # كتابة سجل حتى في حالة timeout
            logs = e.stdout if e.stdout else ""
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== Run ID: {run_id} (local execution - TIMEOUT) ===\n")
                f.write(f"Input Folder: {input_folder}\n")
                f.write(f"Error: تجاوز الوقت المسموح (ساعة)\n\n")
                f.write("=== Execution Logs (partial) ===\n")
                f.write(str(logs) if logs else "لا توجد سجلات")
            return False, str(output_dir), f"تجاوز الوقت المسموح (ساعة). الرجاء تقليل عدد المواضيع عبر TEST_LIMIT"

        except Exception as e:
            # كتابة سجل حتى في حالة خطأ
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== Run ID: {run_id} (local execution - ERROR) ===\n")
                f.write(f"Input Folder: {input_folder}\n")
                f.write(f"Error: {str(e)}\n")
            return False, str(output_dir), f"خطأ في التنفيذ المحلي: {str(e)}"

    except Exception as e:
        return False, "", f"خطأ في إعداد Sandbox: {str(e)}"
