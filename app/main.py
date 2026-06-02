"""
Shorts Runner - FastAPI Backend
تطبيق تشغيل وصفات الفيديوهات القصيرة
نظام القنوات: كل قناة مجلد منفصل، كل وصفة ليها input/output جوه القناة
"""
import sys
if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import re
import uuid
import shutil
import asyncio
import zipfile
import io
import time
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import threading

from app.database import get_db, init_db, Recipe, Run, Setting, User, UserPermission, ApiUsage, SessionLocal
from app.models import (
    RecipeCreate, RecipeUpdate, RecipeResponse,
    RunCreate, RunResponse, PathResponse, CleanupResponse,
    SettingsResponse, SettingsUpdate,
    LoginRequest, LoginResponse, UserCreate, UserUpdate, UserResponse, PermissionUpdate
)
from app.sandbox import create_sandbox_container
from app.auth import (
    get_current_user, require_admin, hash_pin, verify_pin,
    create_token, seed_admin_user
)


# ========== الإعدادات ==========
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "6"))
CLEANUP_MAX_AGE_DAYS = int(os.getenv("CLEANUP_MAX_AGE_DAYS", "7"))
CLEANUP_KEEP_LAST_N = int(os.getenv("CLEANUP_KEEP_LAST_N", "50"))
MAX_CONCURRENT_RUNS = int(os.getenv("MAX_CONCURRENT_RUNS", "2"))
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")

DATA_ROOT = os.getenv("DATA_ROOT", "./data")
CHANNELS_ROOT = os.path.join(DATA_ROOT, "channels")
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "./shorts/out")
LONGS_OUTPUT_ROOT = os.getenv("LONGS_OUTPUT_ROOT", "./longs/out")

_env_lock = threading.Lock()
Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
Path(CHANNELS_ROOT).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
Path(LONGS_OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)


# ========== نظام القنوات ==========

_channels_cache = {"data": None, "time": 0}
_CHANNELS_CACHE_TTL = 30  # ثانية

def get_channels() -> List[str]:
    """قراءة القنوات المتاحة من المجلدات (مع cache 30 ثانية)"""
    now = time.time()
    if _channels_cache["data"] is not None and (now - _channels_cache["time"]) < _CHANNELS_CACHE_TTL:
        return _channels_cache["data"]
    if not os.path.exists(CHANNELS_ROOT):
        return []
    channels = []
    try:
        for entry in os.scandir(CHANNELS_ROOT):
            if entry.is_dir(follow_symlinks=False):
                channels.append(entry.name)
    except Exception:
        pass
    result = sorted(channels)
    _channels_cache["data"] = result
    _channels_cache["time"] = now
    return result


def validate_channel_name(channel: str):
    """التحقق من اسم القناة — منع path traversal"""
    if not channel or ".." in channel or "/" in channel or "\\" in channel:
        raise HTTPException(status_code=400, detail="اسم القناة غير صالح")


def get_channel_path(channel: str) -> Path:
    """مسار مجلد القناة"""
    return Path(CHANNELS_ROOT) / channel


def get_recipe_input_path(channel: str, recipe_name: str) -> Path:
    """مسار مجلد الإدخال لوصفة معينة في قناة معينة"""
    safe_recipe = sanitize_folder_name(recipe_name).replace(' ', '_')
    return get_channel_path(channel) / safe_recipe / "input"


def get_recipe_output_path(channel: str, recipe_name: str) -> Path:
    """مسار مجلد الإخراج لوصفة معينة في قناة معينة"""
    safe_recipe = sanitize_folder_name(recipe_name).replace(' ', '_')
    return get_channel_path(channel) / safe_recipe / "output"


def ensure_channel_recipe_folders(channel: str, recipe_name: str, db_input_folder: str = None):
    """إنشاء مجلدات input/output لوصفة في قناة — يستخدم input_folder من DB لو متاح"""
    if db_input_folder:
        ch = get_channel_path(channel)
        (ch / db_input_folder / "input").mkdir(parents=True, exist_ok=True)
        (ch / db_input_folder / "output").mkdir(parents=True, exist_ok=True)
    else:
        get_recipe_input_path(channel, recipe_name).mkdir(parents=True, exist_ok=True)
        get_recipe_output_path(channel, recipe_name).mkdir(parents=True, exist_ok=True)


# ========== التنظيف التلقائي ==========

async def cleanup_old_runs():
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)
            result = perform_cleanup()
            print(f"[Cleanup] deleted {result['deleted_runs']} runs, freed {result['freed_space_mb']:.2f} MB")
        except Exception as e:
            print(f"[Cleanup] error: {e}")


def cleanup_zombie_runs():
    """تنظيف الـ runs اللي فضلت 'running' بعد restart السيرفر"""
    db = SessionLocal()
    try:
        zombies = db.query(Run).filter(Run.status == "running").all()
        if zombies:
            for run in zombies:
                run.status = "failed"
                run.completed_at = datetime.utcnow()
                run.error_message = "توقف بسبب إعادة تشغيل السيرفر (zombie run cleanup)"
                if run.started_at:
                    run.execution_time_ms = int((run.completed_at - run.started_at).total_seconds() * 1000)
            db.commit()
            print(f"[Startup] تم تنظيف {len(zombies)} zombie run(s)")
        else:
            print(f"[Startup] لا يوجد zombie runs")
    except Exception as e:
        print(f"[Startup] خطأ في تنظيف zombie runs: {e}")
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_admin_user()
    cleanup_zombie_runs()
    cleanup_task = asyncio.create_task(cleanup_old_runs())
    print(f"[MG Ranner] Started on port 8001")
    print(f"[MG Ranner] Channels root: {CHANNELS_ROOT}")
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="MG Ranner", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== الإعدادات الديناميكية ==========

def get_dynamic_settings() -> dict:
    defaults = {
        "max_concurrent_runs": MAX_CONCURRENT_RUNS,
        "timeout_seconds": 3600,
        "mock_mode": MOCK_MODE,
        "cleanup_max_age_days": CLEANUP_MAX_AGE_DAYS,
        "cleanup_keep_last_n": CLEANUP_KEEP_LAST_N,
    }
    db = SessionLocal()
    try:
        settings = db.query(Setting).all()
        for s in settings:
            if s.key in defaults:
                if isinstance(defaults[s.key], bool):
                    defaults[s.key] = s.value.lower() in ("true", "1", "yes")
                elif isinstance(defaults[s.key], int):
                    try:
                        defaults[s.key] = int(s.value)
                    except ValueError:
                        pass
                else:
                    defaults[s.key] = s.value
    except Exception:
        pass
    finally:
        db.close()
    return defaults


def is_mock_mode() -> bool:
    return get_dynamic_settings()["mock_mode"]


def mock_execute(run_id: str, code: str, input_folder: str, content_type: str = "shorts"):
    output_root = LONGS_OUTPUT_ROOT if content_type == "long" else OUTPUT_ROOT
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "script.py").write_text(code, encoding="utf-8")
    time.sleep(1)
    log_content = f"=== Run ID: {run_id} (MOCK MODE) ===\nInput: {input_folder}\n[MOCK] Done\n"
    (output_dir / "run_log.txt").write_text(log_content, encoding="utf-8")
    manifest = {"run_id": run_id, "input_folder": input_folder, "status": "completed", "mock": True}
    (output_dir / "result_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return True, str(output_dir), None


# ========== أمان المسارات ==========

def validate_path(folder_name: str) -> str:
    if not folder_name or not isinstance(folder_name, str):
        raise HTTPException(status_code=400, detail="اسم المجلد مطلوب")
    folder_name = folder_name.strip()
    if ".." in folder_name:
        raise HTTPException(status_code=400, detail="مسار غير مسموح")
    if folder_name.startswith("/") or folder_name.startswith("\\"):
        raise HTTPException(status_code=400, detail="المسارات المطلقة غير مسموحة")
    if re.search(r'[<>"|?*]', folder_name):
        raise HTTPException(status_code=400, detail="المسار يحتوي على أحرف غير مسموحة")
    return folder_name


def sanitize_folder_name(name: str) -> str:
    safe_chars = set(' -_.')
    result = []
    for c in name:
        if c.isalnum() or c in safe_chars:
            result.append(c)
        elif '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F':
            result.append(c)
    return "".join(result).strip()


def check_concurrency(db: Session):
    settings = get_dynamic_settings()
    max_runs = settings["max_concurrent_runs"]
    running_count = db.query(Run).filter(Run.status == "running").count()
    if running_count >= max_runs:
        raise HTTPException(status_code=409, detail=f"يوجد {running_count} تشغيل جاري. الحد الأقصى: {max_runs}")


# ========== التنظيف ==========

def perform_cleanup(max_age_days: int = None, keep_last_n: int = None) -> dict:
    if max_age_days is None:
        max_age_days = CLEANUP_MAX_AGE_DAYS
    if keep_last_n is None:
        keep_last_n = CLEANUP_KEEP_LAST_N
    db = SessionLocal()
    deleted_runs = 0
    freed_space = 0
    errors = []
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        all_runs = db.query(Run).order_by(Run.created_at.desc()).all()
        for i, run in enumerate(all_runs):
            if i < keep_last_n:
                continue
            if run.created_at and run.created_at < cutoff_date:
                try:
                    output_dir = get_run_output_dir(run.run_id, run.output_relpath)
                    if output_dir.exists():
                        for f in output_dir.rglob('*'):
                            if f.is_file():
                                freed_space += f.stat().st_size
                        shutil.rmtree(output_dir)
                    db.delete(run)
                    deleted_runs += 1
                except Exception as e:
                    errors.append(str(e))
        db.commit()
    except Exception as e:
        errors.append(str(e))
    finally:
        db.close()
    return {"deleted_runs": deleted_runs, "freed_space_mb": freed_space / (1024*1024), "errors": errors, "settings": {"max_age_days": max_age_days, "keep_last_n": keep_last_n}}


_storage_stats_cache = {"data": None, "time": 0}
_STORAGE_STATS_CACHE_TTL = 60  # ثانية — مسح الملفات مرة كل دقيقة بدل كل 15 ثانية

def get_storage_stats() -> dict:
    now = time.time()
    cached = _storage_stats_cache

    # الإحصائيات السريعة (DB) — دايماً طازة
    db = SessionLocal()
    try:
        total_runs = db.query(Run).count()
        completed_runs = db.query(Run).filter(Run.status == "completed").count()
        failed_runs = db.query(Run).filter(Run.status == "failed").count()
        running_runs = db.query(Run).filter(Run.status == "running").count()
        cancelled_runs = db.query(Run).filter(Run.status == "cancelled").count()
        oldest_run = db.query(Run).order_by(Run.created_at.asc()).first()
        newest_run = db.query(Run).order_by(Run.created_at.desc()).first()
    finally:
        db.close()

    # إحصائيات الملفات (الثقيلة) — من الكاش لو متاح
    if cached["data"] is not None and (now - cached["time"]) < _STORAGE_STATS_CACHE_TTL:
        total_files = cached["data"]["total_files"]
        total_size = cached["data"]["total_size"]
    else:
        total_size = 0
        total_files = 0
        for out_root in [OUTPUT_ROOT, LONGS_OUTPUT_ROOT]:
            output_path = Path(out_root)
            if output_path.exists():
                for f in output_path.rglob('*'):
                    if f.is_file():
                        total_size += f.stat().st_size
                        total_files += 1
        _storage_stats_cache["data"] = {"total_files": total_files, "total_size": total_size}
        _storage_stats_cache["time"] = now

    return {
        "total_runs": total_runs, "completed_runs": completed_runs, "failed_runs": failed_runs,
        "running_runs": running_runs, "cancelled_runs": cancelled_runs,
        "pending_runs": total_runs - completed_runs - failed_runs - running_runs - cancelled_runs,
        "total_files": total_files, "total_size_mb": round(total_size / (1024*1024), 2),
        "oldest_run": oldest_run.created_at.isoformat() if oldest_run and oldest_run.created_at else None,
        "newest_run": newest_run.created_at.isoformat() if newest_run and newest_run.created_at else None,
        "max_concurrent_runs": get_dynamic_settings()["max_concurrent_runs"],
        "mock_mode": is_mock_mode(),
        "channels_count": len(get_channels()),
        "cleanup_settings": {"interval_hours": CLEANUP_INTERVAL_HOURS, "max_age_days": get_dynamic_settings()["cleanup_max_age_days"], "keep_last_n": get_dynamic_settings()["cleanup_keep_last_n"]}
    }


# ========== Auth Helper ==========

def get_user_recipe_ids(db: Session, user_id: int) -> list:
    perms = db.query(UserPermission).filter(UserPermission.user_id == user_id).all()
    return [p.recipe_id for p in perms]


def user_to_response(db: Session, user: User) -> UserResponse:
    recipe_ids = get_user_recipe_ids(db, user.id)
    return UserResponse(
        id=user.id, username=user.username, display_name=user.display_name,
        is_admin=bool(user.is_admin), is_active=bool(user.is_active),
        created_at=user.created_at, recipe_ids=recipe_ids
    )


# ========== API - Auth ==========

# Rate limiting for login (simple in-memory)
_login_attempts: dict = {}  # {username: (count, last_attempt_time)}

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    # Rate limit: max 5 attempts per 60 seconds per username
    now = time.time()
    # Prune old entries to prevent memory growth
    if len(_login_attempts) > 1000:
        expired = [k for k, (c, t) in _login_attempts.items() if now - t > 120]
        for k in expired:
            _login_attempts.pop(k, None)
    key = req.username[:100].lower()
    if key in _login_attempts:
        count, last_time = _login_attempts[key]
        if now - last_time < 60 and count >= 5:
            raise HTTPException(status_code=429, detail="محاولات كتير — استنى دقيقة")
        if now - last_time >= 60:
            _login_attempts[key] = (0, now)

    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_pin(req.pin, user.pin_hash):
        # Track failed attempt
        prev = _login_attempts.get(key, (0, now))
        _login_attempts[key] = (prev[0] + 1, now)
        raise HTTPException(status_code=401, detail="اسم المستخدم أو الرمز غلط")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="الحساب معطل")
    # Clear rate limit on success
    _login_attempts.pop(key, None)
    token = create_token(user.id, user.username, bool(user.is_admin))
    recipe_ids = get_user_recipe_ids(db, user.id)
    return LoginResponse(
        token=token, username=user.username, display_name=user.display_name,
        is_admin=bool(user.is_admin), recipe_ids=recipe_ids
    )


@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return user_to_response(db, current_user)


# ========== API - Admin: User Management ==========

@app.get("/api/admin/users", response_model=list[UserResponse])
async def list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [user_to_response(db, u) for u in users]


@app.post("/api/admin/users", response_model=UserResponse)
async def create_user(data: UserCreate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    if len(data.pin) != 4 or not data.pin.isdigit():
        raise HTTPException(status_code=400, detail="الرمز لازم يكون 4 أرقام")
    existing = db.query(User).filter(User.username == data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="اسم المستخدم موجود بالفعل")
    user = User(
        username=data.username, display_name=data.display_name,
        pin_hash=hash_pin(data.pin), is_admin=1 if data.is_admin else 0,
        is_active=1, created_at=datetime.utcnow(), updated_at=datetime.utcnow()
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user_to_response(db, user)


@app.put("/api/admin/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, data: UserUpdate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    if data.display_name is not None:
        user.display_name = data.display_name
    if data.pin is not None:
        if len(data.pin) != 4 or not data.pin.isdigit():
            raise HTTPException(status_code=400, detail="الرمز لازم يكون 4 أرقام")
        user.pin_hash = hash_pin(data.pin)
    if data.is_admin is not None:
        # Prevent removing admin from last admin
        if not data.is_admin and user.is_admin:
            admin_count = db.query(User).filter(User.is_admin == 1, User.is_active == 1).count()
            if admin_count <= 1:
                raise HTTPException(status_code=400, detail="مينفعش تشيل صلاحية المدير من آخر مدير")
        user.is_admin = 1 if data.is_admin else 0
    if data.is_active is not None:
        if not data.is_active and user.is_admin:
            admin_count = db.query(User).filter(User.is_admin == 1, User.is_active == 1).count()
            if admin_count <= 1:
                raise HTTPException(status_code=400, detail="مينفعش تعطل آخر مدير")
        user.is_active = 1 if data.is_active else 0
    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user_to_response(db, user)


@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    if user.is_admin:
        admin_count = db.query(User).filter(User.is_admin == 1, User.is_active == 1).count()
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="مينفعش تحذف آخر مدير")
    # Delete permissions
    db.query(UserPermission).filter(UserPermission.user_id == user_id).delete()
    db.delete(user)
    db.commit()
    return {"message": "تم حذف المستخدم"}


@app.put("/api/admin/users/{user_id}/permissions")
async def set_permissions(user_id: int, data: PermissionUpdate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    # Validate recipe_ids exist
    if data.recipe_ids:
        valid_ids = {r.id for r in db.query(Recipe.id).filter(Recipe.id.in_(data.recipe_ids)).all()}
        invalid = set(data.recipe_ids) - valid_ids
        if invalid:
            raise HTTPException(status_code=400, detail=f"وصفات غير موجودة: {sorted(invalid)}")
    # Replace all permissions (deduplicate)
    db.query(UserPermission).filter(UserPermission.user_id == user_id).delete()
    seen = set()
    for recipe_id in data.recipe_ids:
        if recipe_id in seen:
            continue
        seen.add(recipe_id)
        db.add(UserPermission(user_id=user_id, recipe_id=recipe_id, created_at=datetime.utcnow()))
    db.commit()
    return {"message": "تم تحديث الصلاحيات", "recipe_ids": data.recipe_ids}


@app.get("/api/admin/users/{user_id}/permissions")
async def get_permissions(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="المستخدم غير موجود")
    return {"recipe_ids": get_user_recipe_ids(db, user_id)}


# ========== API - القنوات ==========

@app.get("/api/channels")
async def list_channels(current_user: User = Depends(get_current_user)):
    """قائمة القنوات المتاحة"""
    channels = get_channels()
    result = []
    for ch in channels:
        ch_path = get_channel_path(ch)
        # عد المجلدات الفرعية (الوصفات)
        tasks = []
        try:
            for entry in os.scandir(ch_path):
                if entry.is_dir() and entry.name not in ('videos', 'videos_list'):
                    tasks.append(entry.name)
        except Exception:
            pass
        result.append({"name": ch, "tasks_count": len(tasks)})
    return {"channels": result}


@app.post("/api/channels")
async def create_channel(name: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """إنشاء قناة جديدة"""
    safe_name = sanitize_folder_name(name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="اسم القناة غير صالح")
    ch_path = get_channel_path(safe_name)
    if ch_path.exists():
        return {"message": "القناة موجودة بالفعل", "channel": safe_name, "created": False}
    ch_path.mkdir(parents=True, exist_ok=True)
    (ch_path / "videos").mkdir(exist_ok=True)
    # إنشاء مجلدات كل الوصفات الموجودة في القناة الجديدة
    for recipe in db.query(Recipe).all():
        ensure_channel_recipe_folders(safe_name, recipe.name, recipe.input_folder)
    return {"message": "تم إنشاء القناة", "channel": safe_name, "created": True}


@app.get("/api/channels/{channel}/tasks")
async def list_channel_tasks(channel: str, current_user: User = Depends(get_current_user)):
    """قائمة مجلدات المهام في قناة"""
    validate_channel_name(channel)
    ch_path = get_channel_path(channel)
    if not ch_path.exists():
        raise HTTPException(status_code=404, detail="القناة غير موجودة")
    tasks = []
    try:
        for entry in os.scandir(ch_path):
            if entry.is_dir() and entry.name not in ('videos',):
                has_input = (Path(entry.path) / "input").exists()
                has_output = (Path(entry.path) / "output").exists()
                tasks.append({"name": entry.name, "has_input": has_input, "has_output": has_output})
    except Exception:
        pass
    return {"channel": channel, "tasks": sorted(tasks, key=lambda x: x["name"])}


# ========== API - المسارات (متوافق مع الواجهة القديمة) ==========

@app.get("/api/utilities/paths", response_model=PathResponse)
async def get_paths(current_user: User = Depends(get_current_user)):
    """القنوات المتاحة كمجلدات"""
    channels = get_channels()
    return PathResponse(available_folders=channels, data_root=CHANNELS_ROOT)


@app.post("/api/utilities/folders")
async def create_folder(folder_name: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """إنشاء قناة جديدة"""
    safe_name = sanitize_folder_name(folder_name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="اسم غير صالح")
    ch_path = get_channel_path(safe_name)
    if ch_path.exists():
        return {"message": "القناة موجودة بالفعل", "folder": safe_name, "created": False}
    ch_path.mkdir(parents=True, exist_ok=True)
    (ch_path / "videos").mkdir(exist_ok=True)
    # إنشاء مجلدات كل الوصفات الموجودة في القناة الجديدة
    for recipe in db.query(Recipe).all():
        ensure_channel_recipe_folders(safe_name, recipe.name, recipe.input_folder)
    return {"message": "تم إنشاء القناة", "folder": safe_name, "created": True}


# ========== API - Recipes ==========

@app.post("/api/utilities/recipes/{recipe_id}/create-folder")
async def create_recipe_folders_for_all_channels(recipe_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """إنشاء مجلدات الوصفة في كل القنوات"""
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="الوصفة غير موجودة")
    channels = get_channels()
    created = []
    folder = recipe.input_folder or sanitize_folder_name(recipe.name).replace(' ', '_')
    for ch in channels:
        (get_channel_path(ch) / folder / "input").mkdir(parents=True, exist_ok=True)
        (get_channel_path(ch) / folder / "output").mkdir(parents=True, exist_ok=True)
        created.append(ch)
    if not recipe.input_folder:
        recipe.input_folder = folder
    db.commit()
    return {"message": f"تم إنشاء المجلدات في {len(created)} قناة", "channels": created}


@app.post("/api/utilities/create-all-recipe-folders")
async def create_all_recipe_folders(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """إنشاء مجلدات كل الوصفات في كل القنوات"""
    recipes = db.query(Recipe).all()
    channels = get_channels()
    count = 0
    for recipe in recipes:
        folder = recipe.input_folder or sanitize_folder_name(recipe.name).replace(' ', '_')
        for ch in channels:
            (get_channel_path(ch) / folder / "input").mkdir(parents=True, exist_ok=True)
            (get_channel_path(ch) / folder / "output").mkdir(parents=True, exist_ok=True)
            count += 1
        if not recipe.input_folder:
            recipe.input_folder = folder
    db.commit()
    return {"message": f"تم إنشاء {count} مجلد", "summary": {"recipes": len(recipes), "channels": len(channels), "folders_created": count}}


# ========== API - مسارات المجلدات ==========

@app.get("/api/channels/{channel}/recipe-path/{recipe_name}")
async def get_recipe_paths(channel: str, recipe_name: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """الحصول على مسارات input/output لوصفة في قناة — يستخدم input_folder من DB أولاً"""
    validate_channel_name(channel)
    # الأولوية لـ input_folder المسجّل في DB (نفس اللي execute_run بيستخدمه)
    db_recipe = db.query(Recipe).filter(Recipe.name == recipe_name).first()
    if db_recipe and db_recipe.input_folder:
        folder_name = db_recipe.input_folder
    else:
        folder_name = sanitize_folder_name(recipe_name).replace(' ', '_')
    channel_path = get_channel_path(channel)
    input_path = channel_path / folder_name / "input"
    output_path = channel_path / folder_name / "output"
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    return {
        "channel": channel,
        "recipe": recipe_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
    }


# ========== API - Runs ==========

@app.post("/api/utilities/runs", response_model=RunResponse)
async def create_run(run_data: RunCreate, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    validate_path(run_data.input_folder)
    check_concurrency(db)

    # Permission check for non-admin users
    if not current_user.is_admin:
        if not run_data.recipe_id:
            raise HTTPException(status_code=403, detail="لازم تختار وصفة — التشغيل اليدوي للمدير بس")
        allowed = get_user_recipe_ids(db, current_user.id)
        if run_data.recipe_id not in allowed:
            raise HTTPException(status_code=403, detail="مش مسموح لك تشغل الوصفة دي")

    run_id = str(uuid.uuid4())
    content_type = run_data.content_type or "shorts"
    if content_type == "long":
        output_relpath = f"longs/out/{run_id}"
    else:
        output_relpath = f"shorts/out/{run_id}"
    recipe_name = None
    code_to_run = run_data.code or ""

    if run_data.recipe_id:
        recipe = db.query(Recipe).filter(Recipe.id == run_data.recipe_id).first()
        if recipe:
            recipe_name = recipe.name
            # Saved recipes must always run server-side code, not stale code sent by an old browser tab.
            code_to_run = ""
            # "#" أو فاضي = مفيش كود حقيقي، نقرأ من ملف الوصفة
            if not code_to_run.strip() or code_to_run.strip().startswith("#"):
                # أولاً: قراءة الكود من ملف الوصفة (الأحدث دائماً)
                # بيدور في /app/recipes (Docker) أو المسار النسبي (local)
                recipe_base = recipe.input_folder or sanitize_folder_name(recipe.name).replace(' ', '_')
                local_recipes_dir = Path(__file__).resolve().parent.parent / "recipes"
                print(f"[RECIPE DEBUG] recipe.name=[{recipe.name}] input_folder=[{recipe.input_folder}] recipe_base=[{recipe_base}] local_dir=[{local_recipes_dir}]")
                # جرّب JSON أولاً (pipeline)، ثم Python
                recipe_filename = f"{recipe_base}.json"
                recipe_file = Path("/app/recipes") / recipe_filename
                print(f"[RECIPE DEBUG] Try Docker JSON: {recipe_file} exists={recipe_file.exists()}")
                if not recipe_file.exists():
                    recipe_file = local_recipes_dir / recipe_filename
                    print(f"[RECIPE DEBUG] Try Local JSON: {recipe_file} exists={recipe_file.exists()}")
                if not recipe_file.exists():
                    # fallback لـ Python
                    recipe_filename = f"{recipe_base}.py"
                    recipe_file = Path("/app/recipes") / recipe_filename
                    print(f"[RECIPE DEBUG] Try Docker PY: {recipe_file} exists={recipe_file.exists()}")
                    if not recipe_file.exists():
                        recipe_file = local_recipes_dir / recipe_filename
                        print(f"[RECIPE DEBUG] Try Local PY: {recipe_file} exists={recipe_file.exists()}")
                if recipe_file.exists():
                    code_to_run = recipe_file.read_text(encoding="utf-8")
                    print(f"[RECIPE DEBUG] Loaded {len(code_to_run)} chars from {recipe_file}")
                else:
                    code_to_run = recipe.code or ""
                    print(f"[RECIPE DEBUG] FALLBACK to recipe.code: [{code_to_run[:50]}]")

            # ربط input/output بالقناة والوصفة
            channel = run_data.input_folder  # اسم القناة
            recipe_folder = recipe.input_folder or sanitize_folder_name(recipe.name).replace(' ', '_')
            input_path = get_channel_path(channel) / recipe_folder / "input"
            output_path = get_channel_path(channel) / recipe_folder / "output"
            input_path.mkdir(parents=True, exist_ok=True)
            output_path.mkdir(parents=True, exist_ok=True)

            # الوصفة هتستقبل المسارات من البيئة
            # INPUT_DIR = channels/<channel>/<recipe>/input
            # OUTPUT_DIR = channels/<channel>/<recipe>/output
            # CHANNEL_NAME = <channel>
            # CHANNEL_ROOT = channels/<channel>

    db_run = Run(
        run_id=run_id, recipe_id=run_data.recipe_id, recipe_name=recipe_name,
        input_folder=run_data.input_folder, status="pending", output_relpath=output_relpath,
        user_id=current_user.id
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)

    # تمرير اسم القناة واسم الوصفة واسم الموديل ومزود الصوت مع التشغيل
    # تحويل topic_ids لـ string مفصولة بفواصل
    topic_ids_str = ""
    if run_data.topic_ids:
        topic_ids_str = ",".join(str(x) for x in run_data.topic_ids)

    background_tasks.add_task(
        execute_run,
        run_id=run_id, code=code_to_run,
        input_folder=run_data.input_folder,
        recipe_name=recipe_name,
        model_name=run_data.model_name or "gemini-2.5-flash",
        thinking_level=_normalize_thinking_level(run_data.thinking_level),
        tts_provider=run_data.tts_provider or "vertex",
        tts_model=run_data.tts_model or "gemini-2.5-pro-tts",
        execution_mode=run_data.execution_mode or "instant",
        topic_ids=topic_ids_str,
        content_type=content_type
    )
    return RunResponse.model_validate(db_run)


# ========== جدول أسعار التوكنز (لكل مليون توكن) ==========
# ========== نظام حساب التكلفة — مطابق لفواتير Google Cloud ==========
#
# المصادر الرسمية:
# - https://ai.google.dev/gemini-api/docs/pricing
# - https://cloud.google.com/vertex-ai/generative-ai/pricing
# آخر تحديث: 2026-03-28
#
# قواعد حساب التكلفة (مطابقة 100% لفواتير Google):
# 1. الأسعار بالدولار لكل مليون توكن (per million tokens) — Standard tier
# 2. Batch/Flex API = خصم 50% على كل أنواع التوكنز
# 3. Thinking tokens = بتتحسب بنفس سعر Output (مفيش سعر منفصل!)
#    - Google بيعاملها كـ output tokens في الفاتورة
#    - إحنا بنجمعها مع output عند الحساب
# 4. Google usageMetadata:
#    - promptTokenCount = input (يشمل system_prompt + user prompt)
#    - candidatesTokenCount = output (النص الفعلي المنتج)
#    - thoughtsTokenCount = thinking (التفكير — بيتحسب بسعر output)
#    - totalTokenCount = مجموع الثلاثة
# 5. Context tiers: بعض الموديلات (2.5 Pro, 3.1 Pro) سعرها بيزيد فوق 200K tokens

MODEL_PRICING = {
    # ========== Gemini models — أسعار Standard tier (per million tokens) ==========
    # input = سعر input tokens | output = سعر output + thinking tokens
    # thinking = نفس سعر output (Google بيحسبهم سوا في الفاتورة)

    # Gemini 2.5 Flash (سعر واحد — مفيش context tiers)
    "gemini-2.5-flash": {"input": 0.30, "cached_input": 0.03, "output": 2.50, "thinking": 2.50, "batch_discount": 0.5},

    # Gemini 2.5 Pro (فيه context tiers: ≤200K و >200K)
    "gemini-2.5-pro": {"input": 1.25, "cached_input": 0.125, "output": 10.00, "thinking": 10.00, "batch_discount": 0.5,
                        "input_200k": 2.50, "cached_input_200k": 0.25, "output_200k": 15.00, "thinking_200k": 15.00},

    # Gemini 2.0 Flash (legacy — deprecated)
    "gemini-2.0-flash": {"input": 0.15, "cached_input": 0.025, "output": 0.60, "thinking": 0.60, "batch_discount": 0.5},

    # Gemini 3.0 Flash (= Gemini 3 Flash Preview)
    "gemini-3.0-flash": {"input": 0.50, "cached_input": 0.05, "output": 3.00, "thinking": 3.00, "batch_discount": 0.5},
    "gemini-3-flash-preview": {"input": 0.50, "cached_input": 0.05, "output": 3.00, "thinking": 3.00, "batch_discount": 0.5},

    # Gemini 3.1 Pro Preview (فيه context tiers)
    "gemini-3.1-pro-preview": {"input": 2.00, "cached_input": 0.20, "output": 12.00, "thinking": 12.00, "batch_discount": 0.5,
                                "input_200k": 4.00, "cached_input_200k": 0.40, "output_200k": 18.00, "thinking_200k": 18.00},

    # Gemini 3.1 Flash Preview (سعر واحد)
    "gemini-3.1-flash-preview": {"input": 0.50, "cached_input": 0.05, "output": 3.00, "thinking": 3.00, "batch_discount": 0.5},

    # Gemini 3.1 Flash-Lite Preview
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "cached_input": 0.025, "output": 1.50, "thinking": 1.50, "batch_discount": 0.5},

    # Gemini 2.5 Flash-Lite
    "gemini-2.5-flash-lite": {"input": 0.10, "cached_input": 0.01, "output": 0.40, "thinking": 0.40, "batch_discount": 0.5},

    # ========== موديلات الصوت (TTS) — input = توكنز النص | output = توكنز الصوت ==========
    # ⚠️ أسعار TTS: input = سعر توكنز النص الداخل | output = سعر توكنز الصوت الخارج
    # الأسعار المنشورة من Google للـ Preview TTS (لكل مليون توكن):
    "gemini-2.5-flash-tts": {"input": 0.50, "output": 10.00, "thinking": 10.00, "batch_discount": 0.5},
    "gemini-2.5-pro-tts": {"input": 1.00, "output": 20.00, "thinking": 20.00, "batch_discount": 0.5},
    # gemini-3.1-flash-tts-preview: السعر الرسمي من Google (input نص $1 | output صوت $20) | 25 توكن/ثانية
    "gemini-3.1-flash-tts-preview": {"input": 1.00, "output": 20.00, "thinking": 20.00, "batch_discount": 0.5},

    # ========== OpenAI ==========
    "gpt-4o": {"input": 2.50, "output": 10.00, "thinking": 0, "batch_discount": 0.5},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "thinking": 0, "batch_discount": 0.5},

    # ========== Claude (Anthropic) — batch discount = 50% ==========
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "thinking": 15.00, "batch_discount": 0.5},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00, "thinking": 4.00, "batch_discount": 0.5},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int, thinking_tokens: int,
                   cached_tokens: int = 0, call_type: str = "direct") -> float:
    """حساب التكلفة بالدولار — مطابق لفواتير Google Cloud.

    القواعد:
    - Thinking tokens بتتحسب بنفس سعر Output (Google بيجمعهم في الفاتورة)
    - Batch API = خصم 50% على كل شيء
    - Context tiers (>200K) = سعر أعلى للموديلات المدعومة

    Args:
        model: اسم الموديل
        input_tokens: عدد توكنز المدخلات الخام (promptTokenCount) وقد يشمل cached tokens
        output_tokens: عدد توكنز المخرجات (candidatesTokenCount)
        thinking_tokens: عدد توكنز التفكير (thoughtsTokenCount) — بيتحسب بسعر output
        cached_tokens: cachedContentTokenCount — يُخصم من input العادي ويُحسب بسعر cache read
        call_type: "direct" أو "batch" — الباتش بيحصل على خصم 50%

    Returns:
        التكلفة بالدولار (6 خانات عشرية)
    """
    # بحث بالاسم الكامل أولاً ثم بأقرب تطابق
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        for key in MODEL_PRICING:
            if key in model or model in key:
                pricing = MODEL_PRICING[key]
                break
    if not pricing:
        print(f"[PRICING WARNING] موديل غير معروف: {model} — استخدام تقدير افتراضي!")
        pricing = {"input": 1.00, "cached_input": 1.00, "output": 5.00, "thinking": 5.00, "batch_discount": 0.5}

    # كشف context tier (>200K tokens) — لو الموديل بيدعم tiers
    # ⚠️ للباتش: مفيش 200K tier لأن input_tokens هنا مجموع كل البرومبتات
    # وكل برومبت لوحده أقل بكتير من 200K — جوجل بتحسب لكل برومبت
    use_200k_tier = False
    if "input_200k" in pricing and input_tokens > 200_000 and call_type != "batch":
        use_200k_tier = True

    if use_200k_tier:
        input_price = pricing["input_200k"]
        cached_input_price = pricing.get("cached_input_200k", pricing.get("cached_input", input_price))
        output_price = pricing["output_200k"]
        thinking_price = pricing["thinking_200k"]
    else:
        input_price = pricing["input"]
        cached_input_price = pricing.get("cached_input", input_price)
        output_price = pricing["output"]
        thinking_price = pricing["thinking"]

    cached_tokens = max(0, min(cached_tokens or 0, input_tokens or 0))
    uncached_input_tokens = max(0, (input_tokens or 0) - cached_tokens)

    # حساب التكلفة الأساسية
    cost = (
        (uncached_input_tokens / 1_000_000) * input_price +
        (cached_tokens / 1_000_000) * cached_input_price +
        ((output_tokens or 0) / 1_000_000) * output_price +
        ((thinking_tokens or 0) / 1_000_000) * thinking_price
    )

    # تطبيق خصم الباتش لو call_type = "batch"
    if call_type == "batch":
        discount = pricing.get("batch_discount", 0.5)
        cost *= discount

    return round(cost, 6)


def _save_usage_from_summary(db, run_id: str, output_dir: Path):
    """قراءة usage_summary.json وحفظ البيانات في جدول api_usage — مع حساب تكلفة دقيق"""
    usage_file = output_dir / "usage_summary.json"
    if not usage_file.exists():
        return False
    try:
        with open(usage_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary_run_id = summary.get("run_id")
        if summary_run_id and summary_run_id != run_id:
            print(f"[USAGE] رفض usage_summary غير مطابق: {summary_run_id[:8]} != {run_id[:8]} | {usage_file}")
            return False
        records = summary.get("records", [])
        if not records:
            print(f"[USAGE] usage_summary بلا records: {usage_file}")
            return False
        db.query(ApiUsage).filter(ApiUsage.run_id == run_id).delete(synchronize_session=False)
        for rec in records:
            rec_call_type = rec.get("call_type", "direct")
            cost = _estimate_cost(
                rec.get("model", ""),
                rec.get("input_tokens", 0),
                rec.get("output_tokens", 0),
                rec.get("thinking_tokens", 0),
                rec.get("cached_tokens", 0),
                call_type=rec_call_type,
            )
            usage = ApiUsage(
                run_id=run_id,
                step_id=rec.get("step_id", ""),
                call_type=rec_call_type,
                provider=rec.get("provider", ""),
                model=rec.get("model", ""),
                input_tokens=rec.get("input_tokens", 0),
                output_tokens=rec.get("output_tokens", 0),
                thinking_tokens=rec.get("thinking_tokens", 0),
                cached_tokens=rec.get("cached_tokens", 0),
                total_tokens=rec.get("total_tokens", 0),
                estimated_cost_usd=cost,
                send_run_id=rec.get("send_run_id"),
            )
            db.add(usage)
        db.commit()
        total_cost = sum(
            _estimate_cost(
                r.get("model", ""), r.get("input_tokens", 0),
                r.get("output_tokens", 0), r.get("thinking_tokens", 0),
                r.get("cached_tokens", 0),
                call_type=r.get("call_type", "direct")
            )
            for r in records
        )
        print(f"[USAGE] حفظ {len(records)} سجل استهلاك | التكلفة: ${total_cost:.4f} (cache-aware)")
        return True
    except Exception as e:
        db.rollback()
        print(f"[USAGE] خطأ في حفظ بيانات الاستهلاك: {e}")
        return False


def _normalize_thinking_level(value: str | None) -> str:
    value = (value or "none").strip().lower()
    return value if value in {"none", "low", "medium", "high"} else "none"


def execute_run(run_id: str, code: str, input_folder: str, recipe_name: str = None, model_name: str = "gemini-2.5-flash", thinking_level: str = "none", tts_provider: str = "vertex", tts_model: str = "gemini-2.5-pro-tts", execution_mode: str = "instant", topic_ids: str = "", content_type: str = "shorts"):
    db = SessionLocal()
    try:
        db_run = db.query(Run).filter(Run.run_id == run_id).first()
        if not db_run or db_run.status == "cancelled":
            return

        start_time = time.time()
        db_run.status = "running"
        db_run.started_at = datetime.utcnow()
        db.commit()

        # تحديد المسارات بناءً على القناة والوصفة
        channel = input_folder
        # استخدام input_folder من DB بدل sanitize(name) — لتجنب عدم تطابق الأسماء
        db_recipe = db.query(Recipe).filter(Recipe.name == recipe_name).first() if recipe_name else None
        if db_recipe and db_recipe.input_folder:
            actual_input = str(get_channel_path(channel) / db_recipe.input_folder / "input")
            actual_output_recipe = str(get_channel_path(channel) / db_recipe.input_folder / "output")
        elif recipe_name:
            actual_input = str(get_recipe_input_path(channel, recipe_name))
            actual_output_recipe = str(get_recipe_output_path(channel, recipe_name))
        else:
            actual_input = str(Path(CHANNELS_ROOT) / channel)
            actual_output_recipe = None
        channel_root = str(get_channel_path(channel))

        if db_recipe and db_recipe.id == 19:
            stale_markers = (
                "gemini-3.1-pro-preview" in code
                or '"thinking_level": "high"' in code
                or "'thinking_level': 'high'" in code
            )
            if stale_markers:
                raise RuntimeError("SAFETY_BLOCK_STALE_RECIPE_CODE: recipe 19 received old Pro/high code")

        # تمرير متغيرات بيئة (القناة + الموديل + مزود الصوت)
        # Lock to prevent race condition with concurrent runs
        with _env_lock:
            os.environ["CHANNEL_NAME"] = channel
            os.environ["CHANNEL_ROOT"] = channel_root
            os.environ["MODEL_NAME"] = model_name
            os.environ["THINKING_LEVEL"] = _normalize_thinking_level(thinking_level)
            os.environ["TTS_PROVIDER"] = tts_provider
            os.environ["TTS_MODEL"] = tts_model
            os.environ["EXECUTION_MODE"] = execution_mode
            os.environ["TOPIC_IDS"] = topic_ids
            os.environ["RUN_ID"] = run_id
            os.environ["RECIPE_NAME"] = recipe_name or ""
            if actual_output_recipe:
                os.environ["RECIPE_OUTPUT_DIR"] = actual_output_recipe

            if is_mock_mode():
                success, output_path, error_msg = mock_execute(run_id, code, actual_input, content_type)
            else:
                success, output_path, error_msg = create_sandbox_container(
                    run_id=run_id, code=code, input_folder=actual_input, content_type=content_type
                )

        execution_time_ms = int((time.time() - start_time) * 1000)

        # تحديث الحالة في DB فوراً — قبل نسخ الملفات عشان الواجهة تتحدث بسرعة
        db_run = db.query(Run).filter(Run.run_id == run_id).first()
        if db_run and db_run.status != "cancelled":
            db_run.status = "completed" if success else "failed"
            db_run.completed_at = datetime.utcnow()
            db_run.execution_time_ms = execution_time_ms
            if error_msg:
                db_run.error_message = error_msg[:2000]
            db.commit()

        # حفظ بيانات استهلاك التوكنز من usage_summary.json
        # مهم: نقرأ من sandbox أولاً (per-run) — مش من مجلد الوصفة المشترك
        # عشان مجلد الوصفة ممكن يكون فيه usage_summary من تشغيلة سابقة
        _usage_saved = False
        if output_path and (Path(output_path) / "usage_summary.json").exists():
            _usage_saved = _save_usage_from_summary(db, run_id, Path(output_path))
        if not _usage_saved and actual_output_recipe:
            # fallback: نقرأ من مجلد الوصفة بس نتحقق إن الـ run_id متطابق
            usage_file = Path(actual_output_recipe) / "usage_summary.json"
            if usage_file.exists():
                try:
                    import json as _json
                    with open(usage_file, "r", encoding="utf-8") as _f:
                        _summary = _json.load(_f)
                    if _summary.get("run_id") == run_id:
                        _usage_saved = _save_usage_from_summary(db, run_id, Path(actual_output_recipe))
                    else:
                        print(f"[USAGE] تخطي usage_summary — run_id مختلف: {_summary.get('run_id', '?')[:8]} != {run_id[:8]}")
                except Exception as _e:
                    print(f"[USAGE] خطأ في قراءة usage_summary: {_e}")

        # نسخ ملفات الإخراج من مجلد الوصفة للـ sandbox — عشان كل تشغيلة يكون عندها نسختها
        if success and actual_output_recipe and output_path:
            recipe_out = Path(actual_output_recipe).resolve()
            sandbox_out = Path(output_path).resolve()
            if recipe_out.exists() and sandbox_out.exists():
                skip_files = {"script.py", "recipe_config.json", "result_manifest.json", "run_log.txt", "usage_summary.json"}
                copied = 0
                for f in recipe_out.iterdir():
                    if f.is_file() and f.name not in skip_files:
                        dest = sandbox_out / f.name
                        if not dest.exists():
                            shutil.copy2(f, dest)
                            copied += 1
                out_files = [f.name for f in sandbox_out.iterdir() if f.is_file() and f.name not in skip_files]
                print(f"[OUTPUT] نسخ {copied} ملف من مجلد الوصفة للـ sandbox: {sandbox_out}")
                print(f"[OUTPUT] عدد ملفات الإخراج: {len(out_files)}")
                for fname in out_files[:20]:
                    fpath = sandbox_out / fname
                    print(f"[OUTPUT]   - {fname} ({fpath.stat().st_size} bytes)")

                # التحقق من WAV الفاسد
                for fname in out_files:
                    fpath = recipe_out / fname
                    if fpath.suffix.lower() == '.wav' and fpath.stat().st_size < 10240:
                        print(f"[OUTPUT] ⚠ WAV صغير جداً ({fpath.stat().st_size} bytes): {fname} — likely corrupt")
                        try:
                            fpath.unlink()
                        except Exception:
                            pass
        elif not success and output_path:
            print(f"[OUTPUT] الرن فشل، لن يتم نسخ ملفات قديمة من مجلد الوصفة إلى: {output_path}")

    except Exception as e:
        # Prevent stuck "running" runs on unexpected errors
        print(f"[execute_run] خطأ غير متوقع: {e}")
        try:
            db_run = db.query(Run).filter(Run.run_id == run_id).first()
            if db_run and db_run.status == "running":
                db_run.status = "failed"
                db_run.completed_at = datetime.utcnow()
                db_run.error_message = f"خطأ غير متوقع: {str(e)[:500]}"
                db.commit()
        except Exception as e2:
            print(f"[execute_run] فشل تحديث حالة الـ run: {e2}")
    finally:
        db.close()


@app.get("/api/utilities/runs", response_model=List[RunResponse])
async def list_runs(skip: int = 0, limit: int = 50, status: Optional[str] = Query(None), recipe_id: Optional[int] = Query(None), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(Run)
    if status:
        query = query.filter(Run.status == status)
    if recipe_id:
        query = query.filter(Run.recipe_id == recipe_id)
    return [RunResponse.model_validate(r) for r in query.order_by(Run.created_at.desc()).offset(skip).limit(limit).all()]


@app.get("/api/utilities/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    return RunResponse.model_validate(run)


@app.post("/api/utilities/runs/{run_id}/cancel")
async def cancel_run(run_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    if run.status in ("pending", "running"):
        run.status = "cancelled"
        run.completed_at = datetime.utcnow()
        db.commit()
        return {"success": True, "message": "تم الإلغاء"}
    raise HTTPException(status_code=400, detail=f"لا يمكن إلغاء تشغيل بحالة: {run.status}")


@app.get("/api/utilities/runs/{run_id}/log")
async def get_run_log(run_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    log_path = get_run_output_dir(run_id, run.output_relpath) / "run_log.txt"
    if not log_path.exists():
        return {"log": "لا يوجد سجل بعد"}
    try:
        return {"log": log_path.read_text(encoding="utf-8")}
    except Exception:
        return {"log": "خطأ في قراءة السجل"}


@app.get("/api/utilities/runs/{run_id}/manifest")
async def get_run_manifest(run_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    manifest_path = get_run_output_dir(run_id, run.output_relpath) / "result_manifest.json"
    if not manifest_path.exists():
        return {"manifest": None}
    return FileResponse(manifest_path, media_type="application/json")


@app.get("/api/utilities/runs/{run_id}/files")
async def list_run_files(run_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    output_dir = _get_recipe_output_for_run(run, db)
    if not output_dir.exists():
        return {"files": []}
    skip = {"script.py", "recipe_config.json", "result_manifest.json", "run_log.txt"}
    files = [{"name": f.name, "size": f.stat().st_size, "path": f"/api/utilities/runs/{run_id}/files/{f.name}"} for f in output_dir.iterdir() if f.is_file() and f.name not in skip]
    return {"files": sorted(files, key=lambda x: x["name"])}


@app.get("/api/utilities/runs/{run_id}/files/{filename}")
async def get_run_file(run_id: str, filename: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="اسم ملف غير صالح")
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    output_dir = _get_recipe_output_for_run(run, db)
    file_path = output_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="الملف غير موجود")
    return FileResponse(file_path)


@app.get("/api/utilities/runs/{run_id}/download")
async def download_run_outputs(run_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    output_dir = _get_recipe_output_for_run(run, db)
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="مجلد الإخراج غير موجود")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fp in output_dir.rglob('*'):
            if fp.is_file():
                zf.write(fp, fp.relative_to(output_dir))
    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="run_{run_id[:8]}.zip"'})


@app.get("/api/utilities/stats")
async def get_stats(current_user: User = Depends(get_current_user)):
    return get_storage_stats()


# ========== API تتبع التكاليف ==========

@app.get("/api/usage")
async def get_usage(
    days: int = Query(default=30, ge=1, le=365),
    run_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """جلب بيانات استهلاك التوكنز"""
    from sqlalchemy import func
    query = db.query(ApiUsage)
    if run_id:
        query = query.filter(ApiUsage.run_id == run_id)
    else:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = query.filter(ApiUsage.created_at >= cutoff)
    records = query.order_by(ApiUsage.created_at.desc()).all()

    # تجميعات
    total_input = sum(r.input_tokens for r in records)
    total_output = sum(r.output_tokens for r in records)
    total_thinking = sum(r.thinking_tokens for r in records)
    total_cached = sum((getattr(r, "cached_tokens", 0) or 0) for r in records)
    total_billable_input = max(0, total_input - total_cached)
    total_tokens = sum(r.total_tokens for r in records)
    total_cost = sum(r.estimated_cost_usd for r in records)

    # تجميع حسب الموديل
    by_model = {}
    for r in records:
        if r.model not in by_model:
            by_model[r.model] = {"input": 0, "cached": 0, "billable_input": 0, "output": 0, "thinking": 0, "total": 0, "cost": 0, "calls": 0, "batch_calls": 0, "direct_calls": 0}
        cached_tokens = getattr(r, "cached_tokens", 0) or 0
        by_model[r.model]["input"] += r.input_tokens
        by_model[r.model]["cached"] += cached_tokens
        by_model[r.model]["billable_input"] += max(0, r.input_tokens - cached_tokens)
        by_model[r.model]["output"] += r.output_tokens
        by_model[r.model]["thinking"] += r.thinking_tokens
        by_model[r.model]["total"] += r.total_tokens
        by_model[r.model]["cost"] += r.estimated_cost_usd
        by_model[r.model]["calls"] += 1
        if r.call_type == "batch":
            by_model[r.model]["batch_calls"] += 1
        else:
            by_model[r.model]["direct_calls"] += 1

    # تجميع حسب التشغيلة — استخدام send_run_id لو موجود (عشان يطابق جوجل)
    by_run = {}
    for r in records:
        # المفتاح = send_run_id (الإرسال) لو موجود، وإلا run_id العادي
        display_id = r.send_run_id if r.send_run_id else r.run_id
        if display_id not in by_run:
            by_run[display_id] = {"input_tokens": 0, "cached_tokens": 0, "billable_input_tokens": 0, "total_tokens": 0, "cost": 0, "calls": 0, "created_at": r.created_at.isoformat() if r.created_at else "", "receive_run_id": r.run_id}
        cached_tokens = getattr(r, "cached_tokens", 0) or 0
        by_run[display_id]["input_tokens"] += r.input_tokens
        by_run[display_id]["cached_tokens"] += cached_tokens
        by_run[display_id]["billable_input_tokens"] += max(0, r.input_tokens - cached_tokens)
        by_run[display_id]["total_tokens"] += r.total_tokens
        by_run[display_id]["cost"] += r.estimated_cost_usd
        by_run[display_id]["calls"] += 1

    return {
        "period_days": days,
        "totals": {
            "input_tokens": total_input,
            "cached_tokens": total_cached,
            "billable_input_tokens": total_billable_input,
            "output_tokens": total_output,
            "thinking_tokens": total_thinking,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(total_cost, 4),
            "api_calls": len(records),
        },
        "by_model": {k: {**v, "cost": round(v["cost"], 4)} for k, v in by_model.items()},
        "by_run": {k: {**v, "cost": round(v["cost"], 4)} for k, v in sorted(by_run.items(), key=lambda x: x[1]["cost"], reverse=True)[:50]},
        "records": [
            {
                "run_id": r.send_run_id if r.send_run_id else r.run_id,
                "receive_run_id": r.run_id if r.send_run_id else None,
                "step_id": r.step_id,
                "call_type": r.call_type,
                "provider": r.provider,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "cached_tokens": getattr(r, "cached_tokens", 0) or 0,
                "billable_input_tokens": max(0, r.input_tokens - (getattr(r, "cached_tokens", 0) or 0)),
                "output_tokens": r.output_tokens,
                "thinking_tokens": r.thinking_tokens,
                "total_tokens": r.total_tokens,
                "estimated_cost_usd": round(r.estimated_cost_usd, 6),
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in records[:200]
        ],
    }


@app.get("/api/usage/run/{run_id}")
async def get_run_usage(run_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """جلب استهلاك تشغيلة محددة — بيدور بالـ run_id أو send_run_id"""
    from sqlalchemy import or_
    records = db.query(ApiUsage).filter(
        or_(ApiUsage.run_id == run_id, ApiUsage.send_run_id == run_id)
    ).all()
    if not records:
        return {"run_id": run_id, "records": [], "totals": {"total_tokens": 0, "estimated_cost_usd": 0}}
    total_cost = sum(r.estimated_cost_usd for r in records)
    total_input = sum(r.input_tokens for r in records)
    total_cached = sum((getattr(r, "cached_tokens", 0) or 0) for r in records)
    # استخراج send_run_id (الإرسال اللي بيظهر في جوجل)
    send_ids = set(r.send_run_id for r in records if r.send_run_id)
    send_run_id = send_ids.pop() if len(send_ids) == 1 else (list(send_ids) if send_ids else None)
    return {
        "run_id": run_id,
        "send_run_id": send_run_id,
        "totals": {
            "input_tokens": total_input,
            "cached_tokens": total_cached,
            "billable_input_tokens": max(0, total_input - total_cached),
            "output_tokens": sum(r.output_tokens for r in records),
            "thinking_tokens": sum(r.thinking_tokens for r in records),
            "total_tokens": sum(r.total_tokens for r in records),
            "estimated_cost_usd": round(total_cost, 6),
            "api_calls": len(records),
        },
        "records": [
            {
                "step_id": r.step_id,
                "call_type": r.call_type,
                "provider": r.provider,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "cached_tokens": getattr(r, "cached_tokens", 0) or 0,
                "billable_input_tokens": max(0, r.input_tokens - (getattr(r, "cached_tokens", 0) or 0)),
                "output_tokens": r.output_tokens,
                "thinking_tokens": r.thinking_tokens,
                "total_tokens": r.total_tokens,
                "estimated_cost_usd": round(r.estimated_cost_usd, 6),
                "send_run_id": r.send_run_id,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in records
        ],
    }


@app.post("/api/usage/recalculate")
async def recalculate_costs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """إعادة حساب كل التكاليف بالمعادلة الصحيحة (بدون 200K tier للباتش)"""
    records = db.query(ApiUsage).all()
    updated = 0
    total_old = 0
    total_new = 0
    for r in records:
        old_cost = r.estimated_cost_usd
        new_cost = _estimate_cost(
            r.model, r.input_tokens, r.output_tokens, r.thinking_tokens,
            getattr(r, "cached_tokens", 0), call_type=r.call_type
        )
        if abs(old_cost - new_cost) > 0.0001:
            total_old += old_cost
            total_new += new_cost
            r.estimated_cost_usd = new_cost
            updated += 1
    db.commit()
    return {"updated": updated, "total_old": round(total_old, 4), "total_new": round(total_new, 4)}


@app.get("/api/usage/timeline")
async def get_usage_timeline(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=200, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Timeline محاسبي: يعرض كل runs حتى لو مفيش api_usage، ثم يركب عليها usage إن وجد."""
    from sqlalchemy import or_

    cutoff = datetime.utcnow() - timedelta(days=days)
    runs_recent = db.query(Run).filter(
        or_(Run.created_at >= cutoff, Run.completed_at >= cutoff)
    ).order_by(Run.created_at.desc()).all()
    records = db.query(ApiUsage).filter(ApiUsage.created_at >= cutoff).order_by(ApiUsage.created_at.desc()).all()

    runs_by_id = {r.run_id: r for r in db.query(Run).all()}

    uuid_re = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)

    def infer_billing_lookup_run_id(run: Run | None) -> str | None:
        if not run:
            return None
        try:
            output_dir = get_run_output_dir(run.run_id, run.output_relpath)
            metadata_path = output_dir / "batch_metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata.get("run_id") and metadata.get("run_id") != run.run_id:
                        return run.run_id
                except Exception:
                    return run.run_id
            else:
                return run.run_id
            info_path = output_dir / "batch_job_info.json"
            if not info_path.exists():
                return run.run_id
            info = json.loads(info_path.read_text(encoding="utf-8"))

            def from_text(value):
                if not value:
                    return None
                match = uuid_re.search(str(value).lower())
                return match.group(0) if match else None

            def from_extra(extra):
                if not isinstance(extra, dict):
                    return None
                for key in ("labels", "job_labels"):
                    labels = extra.get(key)
                    if isinstance(labels, dict) and labels.get("run_id"):
                        return labels.get("run_id")
                for key in ("gcs_output", "input_uri", "display_name", "job_name"):
                    found = from_text(extra.get(key))
                    if found:
                        return found
                return None

            found = from_extra(info.get("extra") or {})
            if found:
                return found
            for chunk in (info.get("extra") or {}).get("chunks") or []:
                found = from_extra((chunk or {}).get("extra") or {})
                if found:
                    return found
                found = from_text((chunk or {}).get("job_name") or (chunk or {}).get("job_id"))
                if found:
                    return found
        except Exception:
            pass
        return run.run_id

    def make_group(run_id: str, run_meta: Run | None, usage_source: str = "run_only"):
        seen = None
        if run_meta:
            seen = run_meta.completed_at or run_meta.started_at or run_meta.created_at
        return {
            "run_id": run_id,
            "billing_lookup_run_id": infer_billing_lookup_run_id(run_meta) or run_id,
            "linked_send_run_id": None,
            "recipe_name": run_meta.recipe_name if run_meta else "",
            "status": run_meta.status if run_meta else "",
            "started_at": run_meta.started_at.isoformat() if run_meta and run_meta.started_at else None,
            "model": "",
            "providers": set(),
            "models": set(),
            "input_tokens": 0,
            "cached_tokens": 0,
            "billable_input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "calls": 0,
            "batch_calls": 0,
            "direct_calls": 0,
            "first_seen": seen,
            "last_seen": seen,
            "usage_source": usage_source,
        }

    grouped = {}
    for run in runs_recent:
        grouped[run.run_id] = make_group(run.run_id, run, "run_only")

    for r in records:
        key = r.run_id
        if key not in grouped:
            run_meta = runs_by_id.get(key) or runs_by_id.get(r.send_run_id)
            grouped[key] = make_group(key, run_meta, "api_usage")
        g = grouped[key]
        g["usage_source"] = "api_usage"
        if r.send_run_id:
            g["billing_lookup_run_id"] = r.send_run_id
            g["linked_send_run_id"] = r.send_run_id
        else:
            g["billing_lookup_run_id"] = r.run_id
            g["linked_send_run_id"] = None
        g["providers"].add(r.provider or "")
        g["models"].add(r.model or "")
        if r.model and not g["model"]:
            g["model"] = r.model
        cached_tokens = getattr(r, "cached_tokens", 0) or 0
        g["input_tokens"] += r.input_tokens
        g["cached_tokens"] += cached_tokens
        g["billable_input_tokens"] += max(0, r.input_tokens - cached_tokens)
        g["output_tokens"] += r.output_tokens
        g["thinking_tokens"] += r.thinking_tokens
        g["total_tokens"] += r.total_tokens
        g["estimated_cost_usd"] += r.estimated_cost_usd
        g["calls"] += 1
        if r.call_type == "batch":
            g["batch_calls"] += 1
        else:
            g["direct_calls"] += 1
        if r.created_at and (not g["first_seen"] or r.created_at < g["first_seen"]):
            g["first_seen"] = r.created_at
        if r.created_at and (not g["last_seen"] or r.created_at > g["last_seen"]):
            g["last_seen"] = r.created_at

    rows = []
    linked_send_ids = {
        g["linked_send_run_id"]
        for g in grouped.values()
        if g.get("linked_send_run_id") and g.get("calls", 0) > 0
    }
    for g in grouped.values():
        if g["run_id"] in linked_send_ids and g.get("calls", 0) == 0:
            continue
        rows.append({
            "run_id": g["run_id"],
            "billing_lookup_run_id": g["billing_lookup_run_id"],
            "linked_send_run_id": g["linked_send_run_id"],
            "recipe_name": g["recipe_name"],
            "status": g["status"],
            "model": ",".join(sorted(g["models"])) or g["model"] or "—",
            "providers": ",".join(sorted(g["providers"])),
            "input_tokens": g["input_tokens"],
            "cached_tokens": g["cached_tokens"],
            "billable_input_tokens": g["billable_input_tokens"],
            "output_tokens": g["output_tokens"],
            "thinking_tokens": g["thinking_tokens"],
            "total_tokens": g["total_tokens"],
            "estimated_cost_usd": round(g["estimated_cost_usd"], 6),
            "calls": g["calls"],
            "batch_calls": g["batch_calls"],
            "direct_calls": g["direct_calls"],
            "has_usage": g["calls"] > 0,
            "usage_source": g["usage_source"],
            "first_seen": g["first_seen"].isoformat() if g["first_seen"] else None,
            "last_seen": g["last_seen"].isoformat() if g["last_seen"] else None,
        })
    rows.sort(key=lambda x: x["last_seen"] or "", reverse=True)
    rows = rows[:limit]

    totals = {
        "runs": len(rows),
        "estimated_cost_usd": round(sum(r["estimated_cost_usd"] for r in rows), 4),
        "total_tokens": sum(r["total_tokens"] for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "cached_tokens": sum(r["cached_tokens"] for r in rows),
        "billable_input_tokens": sum(r["billable_input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "thinking_tokens": sum(r["thinking_tokens"] for r in rows),
    }
    return {"period_days": days, "totals": totals, "runs": rows}


def _billing_export_candidates(client, dataset: str, override_pattern: str | None = None, prefer_detailed: bool = False):
    if override_pattern:
        kind = "detailed" if "resource" in override_pattern else "override"
        return [{"pattern": override_pattern, "kind": kind}]

    tables = list(client.list_tables(dataset))
    candidates = []
    if any(t.table_id.startswith("gcp_billing_export_resource_v1_") for t in tables):
        candidates.append({"pattern": "gcp_billing_export_resource_v1_*", "kind": "detailed"})
    if any(t.table_id.startswith("gcp_billing_export_v1_") for t in tables):
        candidates.append({"pattern": "gcp_billing_export_v1_*", "kind": "standard"})
    if not prefer_detailed:
        candidates.sort(key=lambda c: 0 if c["kind"] == "standard" else 1)
    return candidates


def _as_utc(dt):
    if not dt:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_datetime(value: str | None):
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise HTTPException(status_code=400, detail="since لازم يكون ISO datetime صالح")
    return _as_utc(parsed)


def _run_cost_ready_time(db: Session, run_id: str):
    """آخر وقت لازم BigQuery Billing Export يكون وصل له قبل عرض رقم فعلي للرن."""
    times = []

    def add_run_time(run):
        if not run:
            return
        dt = run.completed_at or run.started_at or run.created_at
        if dt:
            times.append(_as_utc(dt))

    add_run_time(db.query(Run).filter(Run.run_id == run_id).first())

    linked_usage = db.query(ApiUsage).filter(ApiUsage.send_run_id == run_id).all()
    linked_run_ids = {u.run_id for u in linked_usage if u.run_id}
    if linked_run_ids:
        for run in db.query(Run).filter(Run.run_id.in_(linked_run_ids)).all():
            add_run_time(run)
        for usage in linked_usage:
            if usage.created_at:
                times.append(_as_utc(usage.created_at))

    direct_usage = db.query(ApiUsage).filter(ApiUsage.run_id == run_id).all()
    for usage in direct_usage:
        if usage.created_at:
            times.append(_as_utc(usage.created_at))

    times = [t for t in times if t]
    return max(times) if times else None


def _pending_billing_response(run_id: str, cleaned_run_id: str, reason: str, latest_usage_dt=None, ready_time=None, actual=None, sources=None):
    return {
        "run_id": run_id,
        "cleaned_run_id": cleaned_run_id,
        "source": {"checked": [c["pattern"] for c in sources or []], "kind": "bigquery_label_lookup"},
        "actual_from_google": {
            "cost_usd": 0.0,
            "credits_usd": 0.0,
            "net_cost_usd": 0.0,
            "tax_rate": _billing_tax_rate(),
            "tax_usd": 0.0,
            "total_with_tax_usd": 0.0,
            "line_items": 0,
            "services": [],
            "skus": [],
        },
        "pending": True,
        "unmatched": False,
        "reason": reason,
        "diagnostics": {
            "billing_ready_time": ready_time.isoformat() if ready_time else None,
            "latest_usage": latest_usage_dt.isoformat() if latest_usage_dt else None,
            "partial_from_google": actual,
            "sources": sources or [],
        },
    }


def _billing_tax_rate() -> float:
    try:
        return max(0.0, float(os.getenv("BILLING_TAX_RATE", "0.14")))
    except ValueError:
        return 0.14


def _tax_fields(net_cost_usd: float, tax_rate: float | None = None, actual_tax_usd: float | None = None) -> dict:
    rate = _billing_tax_rate() if tax_rate is None else tax_rate
    tax = float(actual_tax_usd) if actual_tax_usd is not None else float(net_cost_usd or 0.0) * rate
    total = float(net_cost_usd or 0.0) + tax
    return {
        "tax_rate": rate,
        "tax_usd": round(tax, 6),
        "total_with_tax_usd": round(total, 6),
    }


def _billing_model_hint(sku: str | None, labels_text: str | None = None) -> str:
    text = f"{sku or ''} {labels_text or ''}".lower().replace("_", "-")
    checks = [
        ("gemini-3-flash", ("gemini3flash", "gemini-3-flash", "gemini 3 flash", "3.0 flash", "3 flash")),
        ("gemini-3-pro", ("gemini3pro", "gemini-3-pro", "gemini 3 pro", "gemini 3.0 pro", "3.0 pro")),
        ("gemini-2.5-pro", ("gemini25pro", "gemini-2.5-pro", "gemini 2.5 pro", "2.5 pro")),
        ("gemini-2.5-flash", ("gemini25flash", "gemini-2.5-flash", "gemini 2.5 flash", "2.5 flash")),
        ("gemini-2.0-flash", ("gemini20flash", "gemini-2.0-flash", "gemini 2.0 flash", "2.0 flash")),
    ]
    for hint, needles in checks:
        if any(n in text for n in needles):
            return hint
    if "gemini" in text:
        return "gemini"
    return ""


def _classify_unassigned_cost(service: str | None, sku: str | None, labels_text: str | None, cost_type: str | None = None):
    service_l = (service or "").lower()
    sku_l = (sku or "").lower()
    cost_type_l = (cost_type or "").lower()
    text = f"{service_l} {sku_l} {(labels_text or '').lower()}"

    if cost_type_l == "tax":
        return {
            "kind": "vat_tax",
            "label": "ضريبة VAT على الفاتورة",
            "explanation": "ده بند ضريبة من Google Billing، مش رن مستقل.",
            "action": "بيتضاف على تكلفة التشغيلات بنسبة الضريبة بدل ما يتساب كتكلفة غامضة.",
        }

    if "gemini api" in service_l:
        if "storage" in sku_l and "cache" in sku_l:
            return {
                "kind": "gemini_cache_storage",
                "label": "تخزين كاش Gemini بدون run_id",
                "explanation": "ده token-hours لتخزين كاش اتعمل من Gemini API قبل ما يبقى مربوط برن.",
                "action": "مستقبلا اتقفل جوه الرن لأن أي direct Google بقى ممنوع في وضع التتبع الصارم.",
            }
        if "cached" in sku_l or "cache" in sku_l:
            return {
                "kind": "gemini_cached_direct",
                "label": "استخدام كاش Gemini API بدون run_id",
                "explanation": "ده استهلاك كاش من Gemini API مباشر، ومش شايل run_id في فاتورة Google.",
                "action": "مستقبلا مش هيعدي من الرنات لأن direct Google بقى بيتمنع.",
            }
        return {
            "kind": "gemini_direct_unlabeled",
            "label": "Gemini API مباشر بدون run_id",
            "explanation": "ده طلب فوري على Gemini API، وGoogle ماحطتش عليه run_id.",
            "action": "مستقبلا أي طلب Google مباشر جوه run هيفشل بدل ما يعمل تكلفة غير مربوطة.",
        }

    if "vertex ai" in service_l:
        if "batch" in sku_l:
            return {
                "kind": "vertex_batch_unlabeled",
                "label": "Vertex Batch قديم بدون run_id",
                "explanation": "ده Batch Prediction من Vertex AI لكن من غير label يربطه برن محدد.",
                "action": "مستقبلا كل batch لازم يتبعت بالـ run_id label، فالرقم هيظهر على الرن نفسه.",
            }
        if "cache" in text:
            return {
                "kind": "vertex_cache_unlabeled",
                "label": "كاش Vertex بدون run_id",
                "explanation": "ده بند كاش من Vertex AI مش مربوط برن.",
                "action": "هنسيبه ظاهر هنا كتكلفة غير منسوبة لو Google ماقبلتش label عليه.",
            }
        return {
            "kind": "vertex_unlabeled",
            "label": "Vertex AI بدون run_id",
            "explanation": "ده بند Vertex AI غير حامل للـ run_id في Billing Export.",
            "action": "هنا بنحتاج نبص على الوقت والموديل ونرشح أقرب رن.",
        }

    if "bigquery" in service_l:
        return {
            "kind": "billing_tracking_bigquery",
            "label": "BigQuery تتبع الفاتورة",
            "explanation": "ده غالبا تكلفة قراءة Billing Export نفسها أو استعلامات التتبع.",
            "action": "مش تكلفة توليد رن، لكنها تكلفة تشغيل واجهة التكاليف.",
        }

    if "cloud logging" in service_l:
        return {
            "kind": "logging_overhead",
            "label": "Cloud Logging",
            "explanation": "ده بند لوجات من Google Cloud، مش رن توليد مباشر.",
            "action": "لو الرقم كبير نفلتره حسب اللوجات، لكنه مش تكلفة موديل.",
        }

    return {
        "kind": "other_google_cloud",
        "label": "مصدر Google Cloud آخر بدون run_id",
        "explanation": "ده بند فاتورة من Google Cloud مافيهوش run_id.",
        "action": "اتحدد بالخدمة والـ SKU والوقت في الجدول.",
    }


def _build_usage_candidate_index(db: Session, days: int):
    cutoff = datetime.utcnow() - timedelta(days=days + 3)
    records = db.query(ApiUsage).filter(ApiUsage.created_at >= cutoff).order_by(ApiUsage.created_at.desc()).all()
    runs_by_id = {r.run_id: r for r in db.query(Run).filter(Run.created_at >= cutoff).all()}

    grouped = {}
    for record in records:
        key = record.send_run_id if record.send_run_id else record.run_id
        run_meta = runs_by_id.get(key) or runs_by_id.get(record.run_id)
        group = grouped.setdefault(key, {
            "run_id": key,
            "recipe_name": run_meta.recipe_name if run_meta else "",
            "status": run_meta.status if run_meta else "",
            "models": set(),
            "providers": set(),
            "call_types": set(),
            "estimated_cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_tokens": 0,
            "calls": 0,
            "first_seen": record.created_at,
            "last_seen": record.created_at,
        })
        group["models"].add(record.model or "")
        group["providers"].add(record.provider or "")
        group["call_types"].add(record.call_type or "")
        group["estimated_cost_usd"] += float(record.estimated_cost_usd or 0.0)
        group["input_tokens"] += int(record.input_tokens or 0)
        group["output_tokens"] += int(record.output_tokens or 0)
        group["thinking_tokens"] += int(record.thinking_tokens or 0)
        group["cached_tokens"] += int(getattr(record, "cached_tokens", 0) or 0)
        group["calls"] += 1
        if record.created_at and (not group["first_seen"] or record.created_at < group["first_seen"]):
            group["first_seen"] = record.created_at
        if record.created_at and (not group["last_seen"] or record.created_at > group["last_seen"]):
            group["last_seen"] = record.created_at

    result = []
    for group in grouped.values():
        models = sorted(m for m in group["models"] if m)
        providers = sorted(p for p in group["providers"] if p)
        call_types = sorted(c for c in group["call_types"] if c)
        result.append({
            "run_id": group["run_id"],
            "recipe_name": group["recipe_name"],
            "status": group["status"],
            "models": models,
            "providers": providers,
            "call_types": call_types,
            "model_text": " ".join(models).lower().replace("_", "-"),
            "estimated_cost_usd": round(group["estimated_cost_usd"], 6),
            "input_tokens": group["input_tokens"],
            "output_tokens": group["output_tokens"],
            "thinking_tokens": group["thinking_tokens"],
            "cached_tokens": group["cached_tokens"],
            "calls": group["calls"],
            "first_seen": _as_utc(group["first_seen"]),
            "last_seen": _as_utc(group["last_seen"]),
        })
    return result


def _candidate_runs_for_unassigned(item: dict, usage_index: list):
    first_usage = item.get("_first_usage_dt")
    last_usage = item.get("_last_usage_dt") or first_usage
    if not first_usage:
        return []
    window_start = first_usage - timedelta(hours=12)
    window_end = (last_usage or first_usage) + timedelta(hours=36)
    model_hint = item.get("model_hint") or ""
    kind = (item.get("classification") or {}).get("kind", "")
    wants_batch = "batch" in kind
    wants_direct = "direct" in kind or "gemini_" in kind

    matches = []
    for run in usage_index:
        first_seen = run.get("first_seen")
        last_seen = run.get("last_seen") or first_seen
        if not first_seen:
            continue
        overlaps = first_seen <= window_end and last_seen >= window_start
        same_day = first_seen.date() <= (last_usage or first_usage).date() and last_seen.date() >= first_usage.date()
        if not overlaps and not same_day:
            continue

        score = 0
        reasons = []
        if overlaps:
            score += 4
            reasons.append("نفس نافذة الوقت")
        elif same_day:
            score += 2
            reasons.append("نفس اليوم")

        model_text = run.get("model_text") or ""
        if model_hint:
            if model_hint in model_text:
                score += 5
                reasons.append("نفس الموديل")
            elif model_hint.split("-")[0] in model_text:
                score += 1
                reasons.append("نفس عائلة الموديل")

        call_types = set(run.get("call_types") or [])
        providers = set(run.get("providers") or [])
        if wants_batch and "batch" in call_types:
            score += 3
            reasons.append("باتش")
        if wants_direct and "direct" in call_types:
            score += 2
            reasons.append("استدعاء مباشر")
        if "vertex" in providers and "vertex" in kind:
            score += 2
            reasons.append("Vertex")
        if "gemini" in providers and "gemini" in kind:
            score += 2
            reasons.append("Gemini")

        if score < 4:
            continue
        matches.append({
            "run_id": run["run_id"],
            "recipe_name": run.get("recipe_name") or "",
            "status": run.get("status") or "",
            "models": run.get("models") or [],
            "call_types": run.get("call_types") or [],
            "providers": run.get("providers") or [],
            "estimated_cost_usd": run.get("estimated_cost_usd") or 0.0,
            "first_seen": run["first_seen"].isoformat() if run.get("first_seen") else None,
            "last_seen": run["last_seen"].isoformat() if run.get("last_seen") else None,
            "score": score,
            "reasons": reasons[:4],
        })

    matches.sort(key=lambda m: (m["score"], m["estimated_cost_usd"]), reverse=True)
    top = matches[:5]
    if len(top) == 1 and top[0]["score"] >= 10:
        top[0]["confidence"] = "high"
    else:
        for m in top:
            m["confidence"] = "medium" if m["score"] >= 9 else "low"
    return top


@app.get("/api/usage/unassigned-audit")
async def get_unassigned_cost_audit(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=50, ge=1, le=500),
    min_cost: float = Query(default=0.000001, ge=0.0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """يكشف أي تكلفة في Google Billing Export مافيهاش run_id label.

    ده مش بيخمن الفاتورة: كل رقم هنا جاي من BigQuery Billing Export. الترشيحات المحلية
    هدفها تفسير البنود القديمة غير الموسومة فقط، وليست بديل عن run_id label.
    """
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: google-cloud-bigquery ({e})")

    project_id = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0008961174"
    dataset = os.getenv("BQ_BILLING_DATASET", "billing_export")
    override_pattern = os.getenv("BQ_BILLING_TABLE")

    try:
        client = bigquery.Client(project=project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery client init failed: {e}")

    try:
        candidates = _billing_export_candidates(client, dataset, override_pattern, prefer_detailed=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery list_tables failed (dataset={dataset}): {e}")

    if not candidates:
        return {
            "project_id": project_id,
            "dataset": dataset,
            "days": days,
            "source": None,
            "error": "لا يوجد جدول Billing Export في BigQuery.",
            "totals": {"net_cost_usd": 0.0, "groups": 0},
            "items": [],
        }

    usage_index = _build_usage_candidate_index(db, days)
    last_error = None

    for chosen in candidates:
        table_pattern = chosen["pattern"]
        table_ref = f"`{project_id}.{dataset}.{table_pattern}`"
        is_detailed = chosen.get("kind") == "detailed" or "resource" in table_pattern
        resource_select = (
            "IFNULL(resource.name, '') AS resource_name, IFNULL(resource.global_name, '') AS resource_global_name,"
            if is_detailed else
            "'' AS resource_name, '' AS resource_global_name,"
        )

        freshness = {}
        try:
            freshness_q = f"""
            SELECT MAX(export_time) AS latest_export, MAX(usage_end_time) AS latest_usage
            FROM {table_ref}
            """
            row = list(client.query(freshness_q).result())[0]
            freshness = {
                "latest_export_time": row["latest_export"].isoformat() if row["latest_export"] else None,
                "latest_usage_time": row["latest_usage"].isoformat() if row["latest_usage"] else None,
            }
        except Exception as e:
            freshness = {"freshness_error": str(e)}

        query = f"""
        WITH grouped AS (
          SELECT
            DATE(usage_start_time) AS day,
            service.description AS service,
            sku.description AS sku,
            project.id AS project_id,
            location.location AS location,
            {resource_select}
            ARRAY_TO_STRING(ARRAY(
              SELECT CONCAT(lbl.key, '=', lbl.value)
              FROM UNNEST(labels) lbl
              ORDER BY lbl.key
            ), ', ') AS labels_text,
            SUM(cost) AS cost_usd,
            SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS credits_usd,
            SUM(cost + IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS net_cost_usd,
            SUM(usage.amount) AS usage_amount,
            ANY_VALUE(usage.unit) AS usage_unit,
            ANY_VALUE(currency) AS currency,
            MIN(usage_start_time) AS first_usage,
            MAX(usage_end_time) AS last_usage,
            COUNT(*) AS line_items
          FROM {table_ref}
          WHERE usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
            AND NOT EXISTS (
              SELECT 1
              FROM UNNEST(labels) lbl
              WHERE lbl.key = 'run_id'
                AND IFNULL(lbl.value, '') != ''
            )
          GROUP BY day, service, sku, project_id, location, resource_name, resource_global_name, labels_text
        )
        SELECT
          *,
          SUM(net_cost_usd) OVER() AS total_unassigned_net_cost_usd,
          COUNT(*) OVER() AS total_groups
        FROM grouped
        WHERE ABS(net_cost_usd) >= @min_cost
        ORDER BY net_cost_usd DESC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("days", "INT64", days),
                bigquery.ScalarQueryParameter("min_cost", "FLOAT64", float(min_cost)),
                bigquery.ScalarQueryParameter("limit", "INT64", int(limit)),
            ]
        )

        try:
            rows = list(client.query(query, job_config=job_config).result())
        except Exception as e:
            last_error = f"{table_pattern}: {e}"
            continue

        items = []
        category_totals = {}
        total_unassigned = 0.0
        total_groups = 0
        for r in rows:
            classification = _classify_unassigned_cost(r["service"], r["sku"], r["labels_text"])
            model_hint = _billing_model_hint(r["sku"], r["labels_text"])
            net = float(r["net_cost_usd"] or 0.0)
            total_unassigned = float(r["total_unassigned_net_cost_usd"] or total_unassigned or 0.0)
            total_groups = int(r["total_groups"] or total_groups or 0)
            cat = category_totals.setdefault(classification["kind"], {
                "kind": classification["kind"],
                "label": classification["label"],
                "net_cost_usd": 0.0,
                "items": 0,
            })
            cat["net_cost_usd"] += net
            cat["items"] += 1
            item = {
                "day": r["day"].isoformat() if r["day"] else None,
                "service": r["service"],
                "sku": r["sku"],
                "project_id": r["project_id"],
                "location": r["location"],
                "resource_name": r["resource_name"],
                "resource_global_name": r["resource_global_name"],
                "labels_text": r["labels_text"],
                "cost_usd": round(float(r["cost_usd"] or 0.0), 6),
                "credits_usd": round(float(r["credits_usd"] or 0.0), 6),
                "net_cost_usd": round(net, 6),
                "usage_amount": float(r["usage_amount"] or 0.0),
                "usage_unit": r["usage_unit"],
                "currency": r["currency"],
                "first_usage": r["first_usage"].isoformat() if r["first_usage"] else None,
                "last_usage": r["last_usage"].isoformat() if r["last_usage"] else None,
                "line_items": int(r["line_items"] or 0),
                "classification": classification,
                "model_hint": model_hint,
                "_first_usage_dt": _as_utc(r["first_usage"]),
                "_last_usage_dt": _as_utc(r["last_usage"]),
            }
            item["candidate_runs"] = _candidate_runs_for_unassigned(item, usage_index)
            item.pop("_first_usage_dt", None)
            item.pop("_last_usage_dt", None)
            items.append(item)

        categories = []
        for cat in category_totals.values():
            cat["net_cost_usd"] = round(cat["net_cost_usd"], 6)
            categories.append(cat)
        categories.sort(key=lambda c: c["net_cost_usd"], reverse=True)
        tax_rate = _billing_tax_rate()
        tax_usd = total_unassigned * tax_rate

        return {
            "project_id": project_id,
            "dataset": dataset,
            "days": days,
            "limit": limit,
            "min_cost": min_cost,
            "source": {"kind": chosen.get("kind"), "pattern": table_pattern, **freshness},
            "totals": {
                "net_cost_usd": round(total_unassigned, 6),
                "tax_rate": tax_rate,
                "tax_usd": round(tax_usd, 6),
                "total_with_tax_usd": round(total_unassigned + tax_usd, 6),
                "groups": total_groups,
                "returned": len(items),
            },
            "categories": categories,
            "items": items,
            "note": "أي بند هنا رقم مؤكد من Google لكنه لا يحمل run_id label. الترشيحات للرنات القديمة تفسيرية فقط.",
        }

    raise HTTPException(status_code=500, detail=f"BigQuery unassigned audit failed: {last_error}")


@app.get("/api/usage/billing-delta")
async def get_billing_delta_since_baseline(
    since: str = Query(..., min_length=1),
    baseline_balance_usd: float = Query(default=0.0, ge=0.0),
    limit: int = Query(default=200, ge=1, le=500),
    min_cost: float = Query(default=0.000001, ge=0.0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """يراقب أي تكلفة ظهرت في Billing Export بعد لحظة baseline محددة."""
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: google-cloud-bigquery ({e})")

    since_dt = _parse_iso_datetime(since)
    if not since_dt:
        raise HTTPException(status_code=400, detail="since مطلوب")

    project_id = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0008961174"
    dataset = os.getenv("BQ_BILLING_DATASET", "billing_export")
    override_pattern = os.getenv("BQ_BILLING_TABLE")

    try:
        client = bigquery.Client(project=project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery client init failed: {e}")

    try:
        candidates = _billing_export_candidates(client, dataset, override_pattern, prefer_detailed=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery list_tables failed (dataset={dataset}): {e}")

    if not candidates:
        return {
            "project_id": project_id,
            "dataset": dataset,
            "baseline": {"since": since_dt.isoformat(), "balance_usd": baseline_balance_usd},
            "source": None,
            "error": "لا يوجد جدول Billing Export في BigQuery.",
            "totals": {
                "new_cost_usd": 0.0,
                "new_regular_cost_usd": 0.0,
                "actual_tax_cost_usd": 0.0,
                "estimated_tax_usd": 0.0,
                "tax_rate": _billing_tax_rate(),
                "tax_source": "estimated_egypt_vat",
                "new_cost_with_tax_usd": 0.0,
                "projected_balance_usd": round(baseline_balance_usd, 6),
                "attributed_cost_usd": 0.0,
                "attributed_cost_with_tax_usd": 0.0,
                "unassigned_cost_usd": 0.0,
                "unassigned_cost_with_tax_usd": 0.0,
            },
            "items": [],
        }

    usage_index = _build_usage_candidate_index(db, 365)
    run_meta_by_id = {r.run_id: r for r in db.query(Run).all()}
    last_error = None

    for chosen in candidates:
        table_pattern = chosen["pattern"]
        table_ref = f"`{project_id}.{dataset}.{table_pattern}`"
        is_detailed = chosen.get("kind") == "detailed" or "resource" in table_pattern
        resource_select = (
            "IFNULL(resource.name, '') AS resource_name, IFNULL(resource.global_name, '') AS resource_global_name,"
            if is_detailed else
            "'' AS resource_name, '' AS resource_global_name,"
        )

        freshness = {}
        try:
            freshness_q = f"""
            SELECT MAX(export_time) AS latest_export, MAX(usage_end_time) AS latest_usage
            FROM {table_ref}
            """
            row = list(client.query(freshness_q).result())[0]
            latest_usage = _as_utc(row["latest_usage"])
            freshness = {
                "latest_export_time": row["latest_export"].isoformat() if row["latest_export"] else None,
                "latest_usage_time": row["latest_usage"].isoformat() if row["latest_usage"] else None,
                "export_has_reached_baseline": bool(latest_usage and latest_usage >= since_dt),
            }
        except Exception as e:
            freshness = {"freshness_error": str(e), "export_has_reached_baseline": False}

        query = f"""
        WITH grouped AS (
          SELECT
            COALESCE((
              SELECT lbl.value
              FROM UNNEST(labels) lbl
              WHERE lbl.key = 'run_id'
                AND IFNULL(lbl.value, '') != ''
              LIMIT 1
            ), '') AS run_id,
            IFNULL(cost_type, 'regular') AS cost_type,
            DATE(usage_start_time) AS day,
            service.description AS service,
            sku.description AS sku,
            project.id AS project_id,
            location.location AS location,
            {resource_select}
            ARRAY_TO_STRING(ARRAY(
              SELECT CONCAT(lbl.key, '=', lbl.value)
              FROM UNNEST(labels) lbl
              ORDER BY lbl.key
            ), ', ') AS labels_text,
            SUM(cost) AS cost_usd,
            SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS credits_usd,
            SUM(cost + IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS net_cost_usd,
            SUM(usage.amount) AS usage_amount,
            ANY_VALUE(usage.unit) AS usage_unit,
            ANY_VALUE(currency) AS currency,
            MIN(usage_start_time) AS first_usage,
            MAX(usage_end_time) AS last_usage,
            COUNT(*) AS line_items
          FROM {table_ref}
          WHERE usage_start_time >= @since
          GROUP BY run_id, cost_type, day, service, sku, project_id, location, resource_name, resource_global_name, labels_text
        )
        SELECT
          *,
          SUM(IF(cost_type = 'tax', 0, net_cost_usd)) OVER() AS total_regular_net_cost_usd,
          SUM(IF(cost_type = 'tax', net_cost_usd, 0)) OVER() AS total_tax_net_cost_usd,
          SUM(IF(run_id != '' AND cost_type != 'tax', net_cost_usd, 0)) OVER() AS attributed_regular_net_cost_usd,
          SUM(IF(run_id = '' AND cost_type != 'tax', net_cost_usd, 0)) OVER() AS unassigned_regular_net_cost_usd,
          COUNT(*) OVER() AS total_groups
        FROM grouped
        WHERE ABS(net_cost_usd) >= @min_cost
        ORDER BY first_usage DESC, net_cost_usd DESC
        LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("since", "TIMESTAMP", since_dt),
                bigquery.ScalarQueryParameter("min_cost", "FLOAT64", float(min_cost)),
                bigquery.ScalarQueryParameter("limit", "INT64", int(limit)),
            ]
        )

        try:
            rows = list(client.query(query, job_config=job_config).result())
        except Exception as e:
            last_error = f"{table_pattern}: {e}"
            continue

        items = []
        total_regular = 0.0
        actual_tax = 0.0
        attributed = 0.0
        unassigned = 0.0
        groups = 0
        tax_rate = _billing_tax_rate()
        for r in rows:
            run_id = r["run_id"] or ""
            cost_type = r["cost_type"] or "regular"
            net = float(r["net_cost_usd"] or 0.0)
            total_regular = float(r["total_regular_net_cost_usd"] or total_regular or 0.0)
            actual_tax = float(r["total_tax_net_cost_usd"] or actual_tax or 0.0)
            attributed = float(r["attributed_regular_net_cost_usd"] or attributed or 0.0)
            unassigned = float(r["unassigned_regular_net_cost_usd"] or unassigned or 0.0)
            groups = int(r["total_groups"] or groups or 0)
            meta = run_meta_by_id.get(run_id)
            classification = (
                {"kind": "run_labeled", "label": "رن محدد بالـ run_id", "explanation": "ده بند Google مربوط برن صريح.", "action": "راجع الرن في Timeline."}
                if run_id and cost_type != "tax" else
                _classify_unassigned_cost(r["service"], r["sku"], r["labels_text"], cost_type)
            )
            item_tax = _tax_fields(net, tax_rate, actual_tax_usd=net if cost_type == "tax" else None)
            item = {
                "run_id": run_id or None,
                "run_recipe_name": meta.recipe_name if meta else "",
                "run_status": meta.status if meta else "",
                "cost_type": cost_type,
                "day": r["day"].isoformat() if r["day"] else None,
                "service": r["service"],
                "sku": r["sku"],
                "project_id": r["project_id"],
                "location": r["location"],
                "resource_name": r["resource_name"],
                "resource_global_name": r["resource_global_name"],
                "labels_text": r["labels_text"],
                "cost_usd": round(float(r["cost_usd"] or 0.0), 6),
                "credits_usd": round(float(r["credits_usd"] or 0.0), 6),
                "net_cost_usd": round(net, 6),
                "tax_rate": item_tax["tax_rate"],
                "tax_usd": item_tax["tax_usd"],
                "total_with_tax_usd": item_tax["total_with_tax_usd"],
                "usage_amount": float(r["usage_amount"] or 0.0),
                "usage_unit": r["usage_unit"],
                "currency": r["currency"],
                "first_usage": r["first_usage"].isoformat() if r["first_usage"] else None,
                "last_usage": r["last_usage"].isoformat() if r["last_usage"] else None,
                "line_items": int(r["line_items"] or 0),
                "classification": classification,
                "model_hint": _billing_model_hint(r["sku"], r["labels_text"]),
                "_first_usage_dt": _as_utc(r["first_usage"]),
                "_last_usage_dt": _as_utc(r["last_usage"]),
            }
            item["candidate_runs"] = [] if run_id or cost_type == "tax" else _candidate_runs_for_unassigned(item, usage_index)
            item.pop("_first_usage_dt", None)
            item.pop("_last_usage_dt", None)
            items.append(item)

        estimated_tax = actual_tax if abs(actual_tax) > 0.000001 else total_regular * tax_rate
        tax_source = "actual_export" if abs(actual_tax) > 0.000001 else "estimated_egypt_vat"
        total_with_tax = total_regular + estimated_tax

        return {
            "project_id": project_id,
            "dataset": dataset,
            "baseline": {"since": since_dt.isoformat(), "balance_usd": round(baseline_balance_usd, 6)},
            "source": {"kind": chosen.get("kind"), "pattern": table_pattern, **freshness},
            "totals": {
                "new_cost_usd": round(total_regular, 6),
                "new_regular_cost_usd": round(total_regular, 6),
                "actual_tax_cost_usd": round(actual_tax, 6),
                "estimated_tax_usd": round(estimated_tax, 6),
                "tax_rate": tax_rate,
                "tax_source": tax_source,
                "new_cost_with_tax_usd": round(total_with_tax, 6),
                "projected_balance_usd": round(baseline_balance_usd + total_with_tax, 6),
                "attributed_cost_usd": round(attributed, 6),
                "attributed_cost_with_tax_usd": round(attributed * (1 + tax_rate), 6) if tax_source != "actual_export" else round(attributed, 6),
                "unassigned_cost_usd": round(unassigned, 6),
                "unassigned_cost_with_tax_usd": round(unassigned * (1 + tax_rate), 6) if tax_source != "actual_export" else round(unassigned, 6),
                "groups": groups,
                "returned": len(items),
            },
            "items": items,
            "note": "ده فقط ما ظهر في Billing Export بعد baseline. لو export لسه ماوصلش للوقت ده، هيظهر صفر مؤقتا.",
        }

    raise HTTPException(status_code=500, detail=f"BigQuery billing delta failed: {last_error}")


@app.get("/api/usage/google-actual/{run_id}")
async def get_usage_google_actual(
    run_id: str,
    current_user: User = Depends(get_current_user),
):
    """يجلب tokenUsageStats الفعلية من Vertex AI Batch Jobs API لكل batch job مرتبط بـ run_id.
    ده مصدر Google المباشر — أعداد التوكينز اللي Google عدّتها فعلاً."""
    try:
        import google.auth
        import google.auth.transport.requests
        import httpx
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")

    project_id = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0008961174"
    locations = ["global", "us-central1", "us-east1", "europe-west4"]

    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        token = creds.token
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google auth failed: {e}")

    headers = {"Authorization": f"Bearer {token}"}
    short = run_id[:8]
    found_jobs = []
    locations_checked = []

    for loc in locations:
        url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{loc}/batchPredictionJobs"
        params = {"pageSize": 100, "filter": f'display_name:"mgr-{short}"'}
        try:
            with httpx.Client(timeout=20) as client:
                resp = client.get(url, headers=headers, params=params)
                locations_checked.append({"location": loc, "status": resp.status_code})
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for job in data.get("batchPredictionJobs", []):
                    dn = job.get("displayName", "")
                    if run_id in dn or short in dn:
                        usage = (job.get("completionStats") or {})
                        token_stats = job.get("tokenUsageStats") or {}
                        found_jobs.append({
                            "location": loc,
                            "job_name": job.get("name", ""),
                            "display_name": dn,
                            "state": job.get("state", ""),
                            "model": job.get("model", ""),
                            "create_time": job.get("createTime", ""),
                            "end_time": job.get("endTime", ""),
                            "completion_stats": usage,
                            "token_usage_stats": token_stats,
                        })
        except Exception as e:
            locations_checked.append({"location": loc, "error": str(e)})

    actual_input = 0
    actual_output = 0
    actual_total = 0
    for j in found_jobs:
        ts = j.get("token_usage_stats") or {}
        actual_input += int(ts.get("inputTokenCount", 0) or 0)
        actual_output += int(ts.get("outputTokenCount", 0) or 0)
        actual_total += int(ts.get("totalTokenCount", 0) or 0)

    actual_cost_usd = 0.0
    for j in found_jobs:
        model_name = (j.get("model") or "").split("/")[-1]
        ts = j.get("token_usage_stats") or {}
        i_tok = int(ts.get("inputTokenCount", 0) or 0)
        o_tok = int(ts.get("outputTokenCount", 0) or 0)
        actual_cost_usd += _estimate_cost(model_name, i_tok, o_tok, 0, call_type="batch")

    return {
        "run_id": run_id,
        "project_id": project_id,
        "locations_checked": locations_checked,
        "jobs_found": len(found_jobs),
        "actual_from_google": {
            "input_tokens": actual_input,
            "output_tokens": actual_output,
            "total_tokens": actual_total,
            "estimated_cost_usd": round(actual_cost_usd, 6),
        },
        "jobs": found_jobs,
    }


@app.get("/api/usage/bigquery-actual/{run_id}")
async def get_bigquery_actual_for_run(
    run_id: str,
    days: int = 90,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """يقرأ التكلفة الفعلية من BigQuery Billing Export لتشغيلة محددة عبر custom labels.

    الفكرة: لما الـ recipe بيشغّل generate، بيرسل label = {"run_id": "<cleaned>"} مع الطلب.
    BigQuery بيخزّن هذا الـ label في حقل labels (ARRAY<STRUCT<key, value>>).
    هذا الـ endpoint يستعلم عن كل cost لـ label.value = cleaned run_id → رقم 100% من Google.

    تحفظ: BigQuery Billing Export متأخر 24-48 ساعة. لو التشغيلة حديثة → أرقام صفرية.
    """
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: google-cloud-bigquery ({e})")

    # تنظيف run_id بنفس قاعدة Google Cloud labels (lowercase + alphanumeric + _ + -)
    import re as _re
    cleaned_run_id = _re.sub(r'[^a-z0-9_-]', '_', run_id.lower())[:63]

    project_id = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0008961174"
    dataset = os.getenv("BQ_BILLING_DATASET", "billing_export")
    override_pattern = os.getenv("BQ_BILLING_TABLE")
    run_meta = db.query(Run).filter(Run.run_id == run_id).first()

    try:
        client = bigquery.Client(project=project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery client init failed: {e}")

    candidates = []
    if override_pattern:
        candidates.append({"pattern": override_pattern, "kind": "override"})
    else:
        try:
            tables = list(client.list_tables(dataset))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"BigQuery list_tables failed: {e}")

        has_detailed = any(t.table_id.startswith("gcp_billing_export_resource_v1_") for t in tables)
        has_standard = any(t.table_id.startswith("gcp_billing_export_v1_") for t in tables)

        if has_detailed:
            candidates.append({"pattern": "gcp_billing_export_resource_v1_*", "kind": "detailed"})
        if has_standard:
            candidates.append({"pattern": "gcp_billing_export_v1_*", "kind": "standard"})

    if not candidates:
        return {
            "run_id": run_id,
            "cleaned_run_id": cleaned_run_id,
            "error": "لا يوجد جدول Billing Export في BigQuery",
            "actual_from_google": None,
        }

    # احسب حداثة الـ export عشان Pending يبقى له سبب حقيقي، مش حالة أبدية.
    for c in candidates:
        try:
            freshness_q = f"""
            SELECT MAX(export_time) AS latest_export, MAX(usage_end_time) AS latest_usage
            FROM `{project_id}.{dataset}.{c["pattern"]}`
            """
            freshness = list(client.query(freshness_q).result())[0]
            c["latest_export"] = freshness["latest_export"].isoformat() if freshness["latest_export"] else None
            c["latest_usage"] = freshness["latest_usage"].isoformat() if freshness["latest_usage"] else None
            c["_latest_usage_dt"] = freshness["latest_usage"]
        except Exception as e:
            c["freshness_error"] = str(e)
            c["_latest_usage_dt"] = None

    latest_usage_dt = None
    for c in candidates:
        dt = c.get("_latest_usage_dt")
        if dt and (latest_usage_dt is None or dt > latest_usage_dt):
            latest_usage_dt = dt
    public_sources = [
        {k: v for k, v in c.items() if not k.startswith("_")}
        for c in candidates
    ]
    billing_ready_time = _run_cost_ready_time(db, run_id)

    # نجرّب على كل جدول لحد ما نلاقي بيانات بالـ run_id label
    last_error = None
    for chosen in candidates:
        table_pattern = chosen["pattern"]
        table_ref = f"`{project_id}.{dataset}.{table_pattern}`"

        query = f"""
        SELECT
          SUM(cost) AS cost_usd,
          SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS credits_usd,
          ANY_VALUE(currency) AS currency,
          COUNT(*) AS line_items,
          ARRAY_AGG(DISTINCT service.description IGNORE NULLS LIMIT 10) AS services,
          ARRAY_AGG(DISTINCT sku.description IGNORE NULLS LIMIT 20) AS skus,
          MIN(usage_start_time) AS first_usage,
          MAX(usage_end_time) AS last_usage
        FROM {table_ref},
        UNNEST(labels) AS lbl
        WHERE lbl.key = 'run_id'
          AND lbl.value = @run_id
          AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_id", "STRING", cleaned_run_id),
                bigquery.ScalarQueryParameter("days", "INT64", days),
            ]
        )

        try:
            rows = list(client.query(query, job_config=job_config).result())
        except Exception as e:
            last_error = f"{table_pattern}: {e}"
            continue

        if not rows:
            continue
        row = rows[0]
        line_items = int(row["line_items"] or 0)
        if line_items == 0:
            # الجدول ده مفيهوش بيانات لهذه التشغيلة — جرّب التالي
            continue

        cost = float(row["cost_usd"] or 0.0)
        credits = float(row["credits_usd"] or 0.0)
        net = cost + credits
        tax = _tax_fields(net)
        actual = {
            "cost_usd": round(cost, 6),
            "credits_usd": round(credits, 6),
            "net_cost_usd": round(net, 6),
            "tax_rate": tax["tax_rate"],
            "tax_usd": tax["tax_usd"],
            "total_with_tax_usd": tax["total_with_tax_usd"],
            "currency": row["currency"],
            "line_items": line_items,
            "services": list(row["services"] or []),
            "skus": list(row["skus"] or []),
            "first_usage": row["first_usage"].isoformat() if row["first_usage"] else None,
            "last_usage": row["last_usage"].isoformat() if row["last_usage"] else None,
        }

        if billing_ready_time and latest_usage_dt and latest_usage_dt < billing_ready_time:
            reason = (
                f"Billing Export وجد جزء من بيانات الرن لكن لم يصل لنهاية الرن بعد: "
                f"آخر usage في BigQuery = {latest_usage_dt.isoformat()}، "
                f"ونهاية الرن/الاستقبال = {billing_ready_time.isoformat()}."
            )
            return _pending_billing_response(
                run_id, cleaned_run_id, reason,
                latest_usage_dt=latest_usage_dt,
                ready_time=billing_ready_time,
                actual=actual,
                sources=public_sources,
            )

        return {
            "run_id": run_id,
            "cleaned_run_id": cleaned_run_id,
            "source": {"kind": chosen["kind"], "pattern": table_pattern},
            "actual_from_google": actual,
        }

    # ما لقيناش بيانات بالـ label في أي جدول. هنا نفرّق بين تأخير BigQuery وبين run قديم بلا labels.
    export_is_behind_run = bool(billing_ready_time and latest_usage_dt and latest_usage_dt < billing_ready_time)
    if export_is_behind_run:
        reason = (
            f"Billing Export لسه متأخر: آخر usage في BigQuery = {latest_usage_dt.isoformat()}، "
            f"ونهاية الرن/الاستقبال = {billing_ready_time.isoformat()}."
        )
        pending = True
        unmatched = False
    else:
        reason = last_error or (
            "BigQuery Billing Export اتفحص لكن لم يجد label run_id لهذا الرن. "
            "ده يحدث مع التشغيلات القديمة أو direct calls قبل تمرير labels للفوترة."
        )
        pending = False
        unmatched = True

    return {
        "run_id": run_id,
        "cleaned_run_id": cleaned_run_id,
        "source": {"checked": [c["pattern"] for c in candidates], "kind": "bigquery_label_lookup"},
        "actual_from_google": {
            "cost_usd": 0.0,
            "credits_usd": 0.0,
            "net_cost_usd": 0.0,
            "tax_rate": _billing_tax_rate(),
            "tax_usd": 0.0,
            "total_with_tax_usd": 0.0,
            "line_items": 0,
            "services": [],
            "skus": [],
        },
        "pending": pending,
        "unmatched": unmatched,
        "reason": reason,
        "diagnostics": {
            "billing_ready_time": billing_ready_time.isoformat() if billing_ready_time else None,
            "latest_usage": latest_usage_dt.isoformat() if latest_usage_dt else None,
            "sources": public_sources,
        },
    }


@app.get("/api/usage/bigquery-actuals")
async def get_bigquery_actuals_for_runs(
    run_ids: str = Query(..., min_length=1),
    days: int = Query(default=90, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """تكلفة Google الفعلية لمجموعة runs في Query واحدة.

    يرجع رقم مؤكد فقط لو BigQuery Billing Export فيه label run_id مطابق.
    أي run بلا label يفضل "unmatched" ولا يتحول لتقدير.
    """
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: google-cloud-bigquery ({e})")

    import re as _re
    raw_ids = []
    seen = set()
    for rid in run_ids.split(","):
        rid = rid.strip()
        if rid and rid not in seen:
            raw_ids.append(rid)
            seen.add(rid)
    if not raw_ids:
        raise HTTPException(status_code=400, detail="run_ids فارغة")
    if len(raw_ids) > 200:
        raise HTTPException(status_code=400, detail="الحد الأقصى 200 run في الطلب الواحد")

    cleaned_map = {
        rid: _re.sub(r'[^a-z0-9_-]', '_', rid.lower())[:63]
        for rid in raw_ids
    }
    reverse_cleaned = {}
    for rid, cleaned in cleaned_map.items():
        reverse_cleaned.setdefault(cleaned, []).append(rid)

    project_id = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0008961174"
    dataset = os.getenv("BQ_BILLING_DATASET", "billing_export")
    override_pattern = os.getenv("BQ_BILLING_TABLE")

    try:
        client = bigquery.Client(project=project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery client init failed: {e}")

    candidates = []
    if override_pattern:
        candidates.append({"pattern": override_pattern, "kind": "override"})
    else:
        try:
            tables = list(client.list_tables(dataset))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"BigQuery list_tables failed: {e}")
        if any(t.table_id.startswith("gcp_billing_export_resource_v1_") for t in tables):
            candidates.append({"pattern": "gcp_billing_export_resource_v1_*", "kind": "detailed"})
        if any(t.table_id.startswith("gcp_billing_export_v1_") for t in tables):
            candidates.append({"pattern": "gcp_billing_export_v1_*", "kind": "standard"})

    if not candidates:
        return {
            "error": "لا يوجد جدول Billing Export في BigQuery",
            "results": {
                rid: {
                    "cleaned_run_id": cleaned_map[rid],
                    "pending": False,
                    "unmatched": True,
                    "reason": "لا يوجد جدول Billing Export في BigQuery",
            "actual_from_google": {"cost_usd": 0.0, "credits_usd": 0.0, "net_cost_usd": 0.0, "tax_rate": _billing_tax_rate(), "tax_usd": 0.0, "total_with_tax_usd": 0.0, "line_items": 0, "services": [], "skus": []},
                }
                for rid in raw_ids
            },
        }

    latest_usage_dt = None
    for c in candidates:
        try:
            freshness_q = f"""
            SELECT MAX(export_time) AS latest_export, MAX(usage_end_time) AS latest_usage
            FROM `{project_id}.{dataset}.{c["pattern"]}`
            """
            freshness = list(client.query(freshness_q).result())[0]
            c["latest_export"] = freshness["latest_export"].isoformat() if freshness["latest_export"] else None
            c["latest_usage"] = freshness["latest_usage"].isoformat() if freshness["latest_usage"] else None
            c["_latest_usage_dt"] = freshness["latest_usage"]
            if freshness["latest_usage"] and (latest_usage_dt is None or freshness["latest_usage"] > latest_usage_dt):
                latest_usage_dt = freshness["latest_usage"]
        except Exception as e:
            c["freshness_error"] = str(e)
            c["_latest_usage_dt"] = None

    public_sources = [{k: v for k, v in c.items() if not k.startswith("_")} for c in candidates]
    billing_ready_times = {rid: _run_cost_ready_time(db, rid) for rid in raw_ids}
    results = {}
    last_error = None
    chosen_source = None
    cleaned_values = sorted(set(cleaned_map.values()))

    for chosen in candidates:
        table_pattern = chosen["pattern"]
        table_ref = f"`{project_id}.{dataset}.{table_pattern}`"
        query = f"""
        SELECT
          lbl.value AS run_label,
          SUM(cost) AS cost_usd,
          SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS credits_usd,
          ANY_VALUE(currency) AS currency,
          COUNT(*) AS line_items,
          ARRAY_AGG(DISTINCT service.description IGNORE NULLS LIMIT 10) AS services,
          ARRAY_AGG(DISTINCT sku.description IGNORE NULLS LIMIT 20) AS skus,
          MIN(usage_start_time) AS first_usage,
          MAX(usage_end_time) AS last_usage
        FROM {table_ref},
        UNNEST(labels) AS lbl
        WHERE lbl.key = 'run_id'
          AND lbl.value IN UNNEST(@run_ids)
          AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
        GROUP BY run_label
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("run_ids", "STRING", cleaned_values),
                bigquery.ScalarQueryParameter("days", "INT64", days),
            ]
        )
        try:
            rows = list(client.query(query, job_config=job_config).result())
        except Exception as e:
            last_error = f"{table_pattern}: {e}"
            continue
        if not rows:
            continue

        chosen_source = {"kind": chosen["kind"], "pattern": table_pattern}
        for row in rows:
            cleaned = row["run_label"]
            line_items = int(row["line_items"] or 0)
            if line_items == 0:
                continue
            cost = float(row["cost_usd"] or 0.0)
            credits = float(row["credits_usd"] or 0.0)
            net = cost + credits
            tax = _tax_fields(net)
            actual = {
                "cost_usd": round(cost, 6),
                "credits_usd": round(credits, 6),
                "net_cost_usd": round(net, 6),
                "tax_rate": tax["tax_rate"],
                "tax_usd": tax["tax_usd"],
                "total_with_tax_usd": tax["total_with_tax_usd"],
                "currency": row["currency"],
                "line_items": line_items,
                "services": list(row["services"] or []),
                "skus": list(row["skus"] or []),
                "first_usage": row["first_usage"].isoformat() if row["first_usage"] else None,
                "last_usage": row["last_usage"].isoformat() if row["last_usage"] else None,
            }
            for original_id in reverse_cleaned.get(cleaned, []):
                billing_ready_time = billing_ready_times.get(original_id)
                if billing_ready_time and latest_usage_dt and latest_usage_dt < billing_ready_time:
                    reason = (
                        f"Billing Export found part of the run, but it has not reached the run/receive end yet: "
                        f"latest_usage={latest_usage_dt.isoformat()}, "
                        f"billing_ready_time={billing_ready_time.isoformat()}."
                    )
                    results[original_id] = _pending_billing_response(
                        original_id,
                        cleaned,
                        reason,
                        latest_usage_dt=latest_usage_dt,
                        ready_time=billing_ready_time,
                        actual=actual,
                        sources=public_sources,
                    )
                    continue
                results[original_id] = {
                    "cleaned_run_id": cleaned,
                    "source": chosen_source,
                    "pending": False,
                    "unmatched": False,
                    "actual_from_google": actual,
                }
        break

    run_meta = {r.run_id: r for r in db.query(Run).filter(Run.run_id.in_(raw_ids)).all()}
    public_sources = [{k: v for k, v in c.items() if not k.startswith("_")} for c in candidates]
    for rid in raw_ids:
        if rid in results:
            continue
        billing_ready_time = billing_ready_times.get(rid)
        meta = run_meta.get(rid)
        run_completed_at = None
        if meta:
            run_completed_at = meta.completed_at or meta.started_at or meta.created_at
            if run_completed_at and run_completed_at.tzinfo is None:
                run_completed_at = run_completed_at.replace(tzinfo=timezone.utc)
        if not run_completed_at and billing_ready_time:
            run_completed_at = billing_ready_time
        pending = bool(billing_ready_time and latest_usage_dt and latest_usage_dt < billing_ready_time)
        if pending:
            reason = (
                f"Billing Export لسه متأخر: آخر usage_end في BigQuery = {latest_usage_dt.isoformat()}، "
                f"ونهاية الرن/الاستقبال = {billing_ready_time.isoformat()}."
            )
        else:
            reason = last_error or (
                "لا يوجد run_id label لهذا الرن في BigQuery Billing Export. "
                "التكلفة المؤكدة لكل رن تتطلب Google Batch/Vertex labels فقط."
            )
        results[rid] = {
            "cleaned_run_id": cleaned_map[rid],
            "source": {"checked": [c["pattern"] for c in candidates], "kind": "bigquery_label_lookup"},
            "pending": pending,
            "unmatched": not pending,
            "reason": reason,
            "actual_from_google": {"cost_usd": 0.0, "credits_usd": 0.0, "net_cost_usd": 0.0, "tax_rate": _billing_tax_rate(), "tax_usd": 0.0, "total_with_tax_usd": 0.0, "line_items": 0, "services": [], "skus": []},
            "diagnostics": {
                "billing_ready_time": billing_ready_time.isoformat() if billing_ready_time else None,
                "run_completed_at": run_completed_at.isoformat() if run_completed_at else None,
                "latest_usage": latest_usage_dt.isoformat() if latest_usage_dt else None,
                "sources": public_sources,
            },
        }

    return {
        "project_id": project_id,
        "dataset": dataset,
        "days": days,
        "count": len(raw_ids),
        "source": chosen_source or {"checked": [c["pattern"] for c in candidates], "kind": "bigquery_label_lookup"},
        "results": results,
    }


@app.get("/api/usage/billing-actual")
async def get_billing_actual(
    days: int = 30,
    current_user: User = Depends(get_current_user),
):
    """يقرأ من BigQuery Billing Export الفعلي اليومي لخدمات Google Cloud.
    ده مصدر 100% من فاتورة Google — أرقام متطابقة مع اللي Google بيحاسبك عليه.

    بيحاول يقرأ من الجدولين (Standard + Detailed) ويختار اللي فيه بيانات أحدث.
    """
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: google-cloud-bigquery ({e})")

    project_id = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0008961174"
    dataset = os.getenv("BQ_BILLING_DATASET", "billing_export")
    override_pattern = os.getenv("BQ_BILLING_TABLE")

    try:
        client = bigquery.Client(project=project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery client init failed: {e}")

    # اكتشاف الجداول المتاحة (Standard أو Detailed) وأحدث export_time لكل منهم
    candidates: list = []
    if override_pattern:
        candidates = [{"pattern": override_pattern, "kind": "override"}]
    else:
        try:
            tables = list(client.list_tables(dataset))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"BigQuery list_tables failed (dataset={dataset}): {e}")

        has_standard = any(t.table_id.startswith("gcp_billing_export_v1_") for t in tables)
        has_detailed = any(t.table_id.startswith("gcp_billing_export_resource_v1_") for t in tables)
        if has_standard:
            candidates.append({"pattern": "gcp_billing_export_v1_*", "kind": "standard"})
        if has_detailed:
            candidates.append({"pattern": "gcp_billing_export_resource_v1_*", "kind": "detailed"})

    if not candidates:
        return {
            "project_id": project_id,
            "dataset": dataset,
            "days": days,
            "source": None,
            "error": "لا يوجد جدول Billing Export في BigQuery. فعّل Standard أو Detailed usage cost export من Google Cloud Console.",
            "totals": {"cost_usd": 0.0, "credits_usd": 0.0, "net_cost_usd": 0.0, "tax_rate": _billing_tax_rate(), "tax_usd": 0.0, "total_with_tax_usd": 0.0},
            "by_day": [],
            "items": [],
            "diagnostics": {"tables_checked": [c["pattern"] for c in candidates]},
        }

    # قياس "freshness" لكل جدول
    for c in candidates:
        try:
            q = f"""SELECT MAX(export_time) AS latest_export, MAX(usage_end_time) AS latest_usage
                    FROM `{project_id}.{dataset}.{c["pattern"]}`"""
            row = list(client.query(q).result())[0]
            c["latest_export"] = row["latest_export"].isoformat() if row["latest_export"] else None
            c["latest_usage"] = row["latest_usage"].isoformat() if row["latest_usage"] else None
        except Exception as e:
            c["error"] = str(e)
            c["latest_export"] = None

    # اختيار الجدول اللي فيه أحدث export_time
    valid = [c for c in candidates if c.get("latest_export")]
    valid.sort(key=lambda x: x["latest_export"], reverse=True)
    chosen = valid[0] if valid else candidates[0]
    table_pattern = chosen["pattern"]

    table_ref = f"`{project_id}.{dataset}.{table_pattern}`"
    query = f"""
    SELECT
      DATE(usage_start_time) AS day,
      IFNULL(cost_type, 'regular') AS cost_type,
      service.description AS service,
      sku.description AS sku,
      SUM(cost) AS cost_usd,
      SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS credits_usd,
      SUM(usage.amount) AS usage_amount,
      ANY_VALUE(usage.unit) AS usage_unit,
      ANY_VALUE(currency) AS currency
    FROM {table_ref}
    WHERE usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    GROUP BY day, cost_type, service, sku
    ORDER BY day DESC, cost_usd DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("days", "INT64", days)]
    )

    try:
        rows = list(client.query(query, job_config=job_config).result())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery query failed (table={table_pattern}): {e}")

    by_day: dict = {}
    items: list = []
    total_cost = 0.0
    total_credits = 0.0
    regular_net = 0.0
    actual_tax_net = 0.0
    tax_rate = _billing_tax_rate()
    for r in rows:
        day_str = r["day"].isoformat() if r["day"] else None
        cost_type = r["cost_type"] or "regular"
        cost = float(r["cost_usd"] or 0.0)
        credits = float(r["credits_usd"] or 0.0)
        net = cost + credits
        total_cost += cost
        total_credits += credits
        if cost_type == "tax":
            actual_tax_net += net
        else:
            regular_net += net
        item_tax = _tax_fields(net, tax_rate, actual_tax_usd=net if cost_type == "tax" else None)
        items.append({
            "day": day_str,
            "cost_type": cost_type,
            "service": r["service"],
            "sku": r["sku"],
            "cost_usd": round(cost, 6),
            "credits_usd": round(credits, 6),
            "net_cost_usd": round(net, 6),
            "tax_rate": item_tax["tax_rate"],
            "tax_usd": item_tax["tax_usd"],
            "total_with_tax_usd": item_tax["total_with_tax_usd"],
            "usage_amount": float(r["usage_amount"] or 0.0),
            "usage_unit": r["usage_unit"],
            "currency": r["currency"],
        })
        if day_str:
            d = by_day.setdefault(day_str, {"day": day_str, "cost_usd": 0.0, "credits_usd": 0.0, "net_cost_usd": 0.0, "regular_net_cost_usd": 0.0, "tax_cost_usd": 0.0})
            d["cost_usd"] = round(d["cost_usd"] + cost, 6)
            d["credits_usd"] = round(d["credits_usd"] + credits, 6)
            d["net_cost_usd"] = round(d["net_cost_usd"] + net, 6)
            if cost_type == "tax":
                d["tax_cost_usd"] = round(d["tax_cost_usd"] + net, 6)
            else:
                d["regular_net_cost_usd"] = round(d["regular_net_cost_usd"] + net, 6)

    estimated_tax = actual_tax_net if abs(actual_tax_net) > 0.000001 else regular_net * tax_rate
    tax_source = "actual_export" if abs(actual_tax_net) > 0.000001 else "estimated_egypt_vat"
    total_with_tax = regular_net + estimated_tax

    return {
        "project_id": project_id,
        "dataset": dataset,
        "days": days,
        "source": {
            "kind": chosen.get("kind"),
            "pattern": table_pattern,
            "latest_export_time": chosen.get("latest_export"),
            "latest_usage_time": chosen.get("latest_usage"),
        },
        "totals": {
            "cost_usd": round(total_cost, 6),
            "credits_usd": round(total_credits, 6),
            "net_cost_usd": round(total_cost + total_credits, 6),
            "regular_net_cost_usd": round(regular_net, 6),
            "actual_tax_cost_usd": round(actual_tax_net, 6),
            "tax_rate": tax_rate,
            "tax_source": tax_source,
            "tax_usd": round(estimated_tax, 6),
            "total_with_tax_usd": round(total_with_tax, 6),
        },
        "by_day": sorted(by_day.values(), key=lambda x: x["day"], reverse=True),
        "items": items,
        "diagnostics": {"candidates": candidates},
    }


def _model_family_from_app_name(model_name: str) -> str:
    m = (model_name or "").lower().replace("-", " ").replace("_", " ")
    if "3" in m and "flash" in m:
        return "gemini 3 flash"
    if ("3.1" in m or "3 " in m or m.endswith(" 3") or "gemini 3" in m) and "pro" in m:
        return "gemini 3 pro"
    if "2.5" in m and "pro" in m:
        return "gemini 2.5 pro"
    if "2.5" in m and "flash" in m:
        return "gemini 2.5 flash"
    return "other"


def _model_family_from_sku(sku: str) -> str:
    s = (sku or "").lower()
    if "gemini 3 flash" in s or "gemini 3.0 flash" in s:
        return "gemini 3 flash"
    if "gemini 3 pro" in s or "gemini 3.0 pro" in s:
        return "gemini 3 pro"
    if "gemini 2.5 pro" in s:
        return "gemini 2.5 pro"
    if "gemini 2.5 flash" in s:
        return "gemini 2.5 flash"
    return "other"


@app.get("/api/usage/run-billing-share")
async def get_run_billing_share(
    run_ids: str = Query(..., min_length=1),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """نصيب كل رن من فاتورة Google الفعلية في BigQuery.

    المنطق:
      1) من api_usage نطلع: لكل ساعة UTC × موديل، إجمالي توكنز كل رن.
      2) من BigQuery نطلع: لكل ساعة UTC × موديل، إجمالي تكلفة الموديل بعد credits.
      3) نصيب الرن = (توكنز الرن في الساعة × الموديل) / (إجمالي توكنز كل الرنز في الساعة × الموديل) × تكلفة BigQuery.
      4) مجموع نصائب الرن في كل الساعات = التكلفة الفعلية من جوجل.

    ده 100% مستند لفاتورة جوجل، مفيش تخمين في الأسعار.
    """
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: google-cloud-bigquery ({e})")

    from datetime import datetime, timedelta
    from sqlalchemy import text as _sa_text

    run_id_list = [r.strip() for r in run_ids.split(",") if r.strip()]
    if not run_id_list:
        raise HTTPException(status_code=400, detail="run_ids مطلوبة")
    if len(run_id_list) > 500:
        raise HTTPException(status_code=400, detail="حد أقصى 500 run_id في الطلب")

    # 1) Pull token usage data globally (we need all runs in the affected hours to compute share)
    rows = db.execute(_sa_text(
        "SELECT run_id, model, created_at, "
        "       IFNULL(input_tokens,0) AS i, IFNULL(output_tokens,0) AS o, "
        "       IFNULL(thinking_tokens,0) AS t "
        "FROM api_usage "
        "WHERE provider IN ('gemini','vertex') "
        "  AND created_at >= datetime('now','-120 days')"
    )).fetchall()

    def _parse_dt(s):
        if isinstance(s, datetime):
            return s
        try:
            return datetime.fromisoformat(str(s).replace(" ", "T"))
        except Exception:
            return None

    # (hour_utc, family) → {run_id: tokens, _total: tokens}
    bucket = {}
    requested = set(run_id_list)
    run_keys = {rid: set() for rid in run_id_list}

    for rid, model, created_at, inp, out, thk in rows:
        dt = _parse_dt(created_at)
        if not dt:
            continue
        family = _model_family_from_app_name(model)
        if family == "other":
            continue
        billable = (inp or 0) + (out or 0) + (thk or 0)
        if billable <= 0:
            continue
        hour = dt.replace(minute=0, second=0, microsecond=0)
        key = (hour, family)
        b = bucket.get(key)
        if b is None:
            b = {"_total": 0}
            bucket[key] = b
        b[rid] = b.get(rid, 0) + billable
        b["_total"] += billable
        if rid in requested:
            run_keys[rid].add(key)

    if not bucket:
        return {
            "results": {rid: {"actual_net_cost_usd": 0.0, "actual_total_with_tax_usd": 0.0,
                              "hours_count": 0, "status": "no_token_data", "breakdown": []}
                        for rid in run_id_list},
        }

    # 2) Build BigQuery query for the affected hour window
    project_id = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0008961174"
    dataset = os.getenv("BQ_BILLING_DATASET", "billing_export")

    all_hours = sorted({k[0] for k in bucket.keys()})
    min_hour = all_hours[0]
    max_hour = all_hours[-1] + timedelta(hours=1)

    try:
        bq = bigquery.Client(project=project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery client init failed: {e}")

    try:
        tables = list(bq.list_tables(dataset))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery list_tables failed: {e}")

    table_pattern = None
    for t in tables:
        if t.table_id.startswith("gcp_billing_export_resource_v1_"):
            table_pattern = "gcp_billing_export_resource_v1_*"
            break
    if not table_pattern:
        for t in tables:
            if t.table_id.startswith("gcp_billing_export_v1_"):
                table_pattern = "gcp_billing_export_v1_*"
                break
    if not table_pattern:
        raise HTTPException(status_code=500, detail="لا يوجد جدول Billing Export في BigQuery")

    table_ref = f"`{project_id}.{dataset}.{table_pattern}`"
    q = f"""
    SELECT
      TIMESTAMP_TRUNC(usage_start_time, HOUR) AS hour,
      LOWER(sku.description) AS sku,
      SUM(cost) AS cost_usd,
      SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)) AS credits_usd
    FROM {table_ref}
    WHERE usage_start_time >= TIMESTAMP(@start)
      AND usage_start_time < TIMESTAMP(@end)
      AND (service.description = 'Vertex AI' OR service.description = 'Gemini API')
    GROUP BY hour, sku
    """
    jc = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("start", "TIMESTAMP", min_hour),
        bigquery.ScalarQueryParameter("end", "TIMESTAMP", max_hour),
    ])

    try:
        bq_rows = list(bq.query(q, job_config=jc).result())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery query failed: {e}")

    try:
        latest_usage_row = list(bq.query(f"SELECT MAX(usage_end_time) AS m FROM {table_ref}").result())[0]
        latest_usage = latest_usage_row["m"]
    except Exception:
        latest_usage = None

    # (hour, family) → net_cost
    bq_cost_by_key = {}
    for r in bq_rows:
        family = _model_family_from_sku(r["sku"])
        if family == "other":
            continue
        net = float(r["cost_usd"] or 0) + float(r["credits_usd"] or 0)
        key = (r["hour"].replace(tzinfo=None), family)
        bq_cost_by_key[key] = bq_cost_by_key.get(key, 0.0) + net

    # 3) Per-run share
    tax_rate = _billing_tax_rate()
    results = {}
    for rid in run_id_list:
        total_net = 0.0
        breakdown = []
        hours_pending = 0
        hours_unmatched = 0
        for key in run_keys.get(rid, set()):
            hour, family = key
            b = bucket[key]
            run_tokens = b.get(rid, 0)
            total_tokens = b["_total"]
            if total_tokens <= 0:
                continue
            share = run_tokens / total_tokens
            bq_cost = bq_cost_by_key.get(key)
            if bq_cost is None:
                # Either pending export or matched 0 in BigQuery for this hour+family
                if latest_usage and hour.replace(tzinfo=None) >= latest_usage.replace(tzinfo=None):
                    hours_pending += 1
                else:
                    hours_unmatched += 1
                continue
            run_share = share * bq_cost
            total_net += run_share
            breakdown.append({
                "hour": hour.isoformat(),
                "model_family": family,
                "run_tokens": run_tokens,
                "hour_total_tokens": total_tokens,
                "share_pct": round(share * 100, 2),
                "hour_total_cost_usd": round(bq_cost, 6),
                "run_share_cost_usd": round(run_share, 6),
            })
        net_cost = round(total_net, 6)
        tax = round(net_cost * tax_rate, 6)
        total_with_tax = round(net_cost + tax, 6)

        status = "ok"
        hours_count = len(run_keys.get(rid, set()))
        if hours_count == 0:
            status = "no_token_data"
        elif hours_pending > 0 and len(breakdown) == 0:
            status = "pending_bigquery_export"
        elif hours_unmatched > 0 and len(breakdown) == 0:
            status = "no_billing_match"
        elif hours_pending > 0:
            status = "partial_pending"

        results[rid] = {
            "actual_net_cost_usd": net_cost,
            "tax_rate": tax_rate,
            "tax_usd": tax,
            "actual_total_with_tax_usd": total_with_tax,
            "hours_count": hours_count,
            "hours_pending_export": hours_pending,
            "hours_unmatched": hours_unmatched,
            "status": status,
            "breakdown": sorted(breakdown, key=lambda x: x["hour"]),
        }

    return {
        "source": {
            "table_pattern": table_pattern,
            "latest_usage": latest_usage.isoformat() if latest_usage else None,
        },
        "results": results,
    }


@app.post("/api/utilities/cleanup", response_model=CleanupResponse)
async def trigger_cleanup(max_age_days: int = None, keep_last_n: int = None, admin: User = Depends(require_admin)):
    return CleanupResponse(**perform_cleanup(max_age_days, keep_last_n))


@app.delete("/api/utilities/runs/{run_id}")
async def delete_run(run_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    output_dir = get_run_output_dir(run_id, run.output_relpath)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    db.delete(run)
    db.commit()
    return {"message": "تم الحذف", "run_id": run_id}


# ========== Recipes CRUD ==========

@app.post("/api/utilities/recipes", response_model=RecipeResponse)
async def create_recipe(recipe: RecipeCreate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    db_recipe = Recipe(**recipe.model_dump())
    if not db_recipe.input_folder:
        db_recipe.input_folder = sanitize_folder_name(db_recipe.name).replace(' ', '_')
    db.add(db_recipe)
    db.commit()
    db.refresh(db_recipe)
    # إنشاء مجلدات في كل القنوات
    for ch in get_channels():
        (get_channel_path(ch) / db_recipe.input_folder / "input").mkdir(parents=True, exist_ok=True)
        (get_channel_path(ch) / db_recipe.input_folder / "output").mkdir(parents=True, exist_ok=True)
    return RecipeResponse.model_validate(db_recipe)


@app.get("/api/utilities/recipes", response_model=List[RecipeResponse])
async def list_recipes(recipe_type: Optional[str] = Query(None), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(Recipe)
    if recipe_type:
        query = query.filter((Recipe.recipe_type == recipe_type) | (Recipe.recipe_type == "both"))
    return [RecipeResponse.model_validate(r) for r in query.order_by(Recipe.created_at.desc()).all()]


@app.get("/api/utilities/recipes/{recipe_id}", response_model=RecipeResponse)
async def get_recipe(recipe_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="الوصفة غير موجودة")
    return RecipeResponse.model_validate(recipe)


@app.put("/api/utilities/recipes/{recipe_id}", response_model=RecipeResponse)
async def update_recipe(recipe_id: int, recipe_update: RecipeUpdate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="الوصفة غير موجودة")
    for key, value in recipe_update.model_dump(exclude_unset=True).items():
        setattr(recipe, key, value)
    db.commit()
    db.refresh(recipe)
    return RecipeResponse.model_validate(recipe)


@app.delete("/api/utilities/recipes/{recipe_id}")
async def delete_recipe(recipe_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="الوصفة غير موجودة")
    # Clean up orphaned permissions
    db.query(UserPermission).filter(UserPermission.recipe_id == recipe_id).delete()
    db.delete(recipe)
    db.commit()
    return {"message": "تم الحذف"}


# ========== Settings ==========

@app.get("/api/utilities/settings")
async def get_settings(admin: User = Depends(require_admin)):
    return get_dynamic_settings()


@app.put("/api/utilities/settings")
async def update_settings(updates: SettingsUpdate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    for key, value in updates.model_dump(exclude_unset=True).items():
        existing = db.query(Setting).filter(Setting.key == key).first()
        str_value = str(value).lower() if isinstance(value, bool) else str(value)
        if existing:
            existing.value = str_value
        else:
            db.add(Setting(key=key, value=str_value))
    db.commit()
    return get_dynamic_settings()


# ========== مسارات المجلدات على الجهاز المضيف ==========

def get_run_output_dir(run_id: str, output_relpath: str = None) -> Path:
    """مجلد sandbox المؤقت (script.py و config فقط)"""
    if output_relpath and output_relpath.startswith("longs/"):
        return Path(LONGS_OUTPUT_ROOT) / run_id
    return Path(OUTPUT_ROOT) / run_id


def _get_recipe_output_for_run(run: "Run", db: "Session") -> Path:
    """مجلد الـ sandbox الخاص بالتشغيلة — كل تشغيلة ليها مجلد منفصل.
    بيرجع مجلد sandbox (per-run) عشان كل تشغيلة تعرض ملفاتها هي بالظبط."""
    return get_run_output_dir(run.run_id, run.output_relpath)

HOST_DATA_DIR = os.getenv("HOST_DATA_DIR", "C:/Users/w10/shorts-runner/data")

@app.post("/api/open-folder")
async def get_host_folder_path(docker_path: str, current_user: User = Depends(get_current_user)):
    """تحويل مسار Docker لمسار Windows وإنشاء المجلد"""
    if not docker_path or ".." in docker_path:
        raise HTTPException(status_code=400, detail="مسار غير صالح")
    HOST_ROOT = os.getenv("HOST_ROOT", "C:/Users/w10/shorts-runner")
    if docker_path.startswith("/app/data/"):
        host_path = docker_path.replace("/app/data/", HOST_DATA_DIR + "/")
    elif docker_path.startswith("/app/longs/out/"):
        host_path = docker_path.replace("/app/longs/out/", HOST_ROOT + "/longs/out/")
    elif docker_path.startswith("/app/shorts/out/"):
        host_path = docker_path.replace("/app/shorts/out/", HOST_ROOT + "/shorts/out/")
    else:
        raise HTTPException(status_code=400, detail="المسار لازم يبدأ بـ /app/data/ أو /app/longs/out/ أو /app/shorts/out/")
    host_path = host_path.replace("/", "\\")
    return {"success": True, "path": host_path}


@app.get("/api/utilities/runs/{run_id}/output-path")
async def get_run_output_path(run_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """إرجاع مسار مجلد المخرجات الحقيقي للوصفة (مش الـ sandbox)"""
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    # مسار مجلد المخرجات الحقيقي للوصفة
    channel = run.input_folder
    if run.recipe_id:
        recipe = db.query(Recipe).filter(Recipe.id == run.recipe_id).first()
        if recipe and recipe.input_folder:
            docker_path = str(get_channel_path(channel) / recipe.input_folder / "output")
            real_path = Path(docker_path)
            return {"output_path": docker_path, "exists": real_path.exists()}
    # fallback: sandbox path
    docker_path = f"/app/{run.output_relpath}"
    output_dir = get_run_output_dir(run.run_id, run.output_relpath)
    return {"output_path": str(docker_path), "exists": output_dir.exists()}


def get_output_root_for_content(content_type: str) -> str:
    """إرجاع مسار الإخراج حسب نوع المحتوى"""
    if content_type == "long":
        return LONGS_OUTPUT_ROOT
    return OUTPUT_ROOT


@app.get("/health")
async def health():
    return {"status": "ok"}


# ========== Static ==========

app.mount("/", StaticFiles(directory="static", html=True), name="static")
