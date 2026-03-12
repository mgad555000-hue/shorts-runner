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
from datetime import datetime, timedelta
import threading

from app.database import get_db, init_db, Recipe, Run, Setting, User, UserPermission, SessionLocal
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

def get_channels() -> List[str]:
    """قراءة القنوات المتاحة من المجلدات"""
    if not os.path.exists(CHANNELS_ROOT):
        return []
    channels = []
    try:
        for entry in os.scandir(CHANNELS_ROOT):
            if entry.is_dir(follow_symlinks=False):
                channels.append(entry.name)
    except Exception:
        pass
    return sorted(channels)


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


def ensure_channel_recipe_folders(channel: str, recipe_name: str):
    """إنشاء مجلدات input/output لوصفة في قناة"""
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


def get_storage_stats() -> dict:
    db = SessionLocal()
    try:
        total_runs = db.query(Run).count()
        completed_runs = db.query(Run).filter(Run.status == "completed").count()
        failed_runs = db.query(Run).filter(Run.status == "failed").count()
        running_runs = db.query(Run).filter(Run.status == "running").count()
        cancelled_runs = db.query(Run).filter(Run.status == "cancelled").count()
        total_size = 0
        total_files = 0
        for out_root in [OUTPUT_ROOT, LONGS_OUTPUT_ROOT]:
            output_path = Path(out_root)
            if output_path.exists():
                for f in output_path.rglob('*'):
                    if f.is_file():
                        total_size += f.stat().st_size
                        total_files += 1
        oldest_run = db.query(Run).order_by(Run.created_at.asc()).first()
        newest_run = db.query(Run).order_by(Run.created_at.desc()).first()
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
    finally:
        db.close()


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
        ensure_channel_recipe_folders(safe_name, recipe.name)
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
        ensure_channel_recipe_folders(safe_name, recipe.name)
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
async def get_recipe_paths(channel: str, recipe_name: str, current_user: User = Depends(get_current_user)):
    """الحصول على مسارات input/output لوصفة في قناة"""
    validate_channel_name(channel)
    input_path = get_recipe_input_path(channel, recipe_name)
    output_path = get_recipe_output_path(channel, recipe_name)
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
        tts_provider=run_data.tts_provider or "vertex",
        execution_mode=run_data.execution_mode or "instant",
        topic_ids=topic_ids_str,
        content_type=content_type
    )
    return RunResponse.model_validate(db_run)


def execute_run(run_id: str, code: str, input_folder: str, recipe_name: str = None, model_name: str = "gemini-2.5-flash", tts_provider: str = "vertex", execution_mode: str = "instant", topic_ids: str = "", content_type: str = "shorts"):
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

        # تمرير متغيرات بيئة (القناة + الموديل + مزود الصوت)
        # Lock to prevent race condition with concurrent runs
        with _env_lock:
            os.environ["CHANNEL_NAME"] = channel
            os.environ["CHANNEL_ROOT"] = channel_root
            os.environ["MODEL_NAME"] = model_name
            os.environ["TTS_PROVIDER"] = tts_provider
            os.environ["EXECUTION_MODE"] = execution_mode
            os.environ["TOPIC_IDS"] = topic_ids
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

        if output_path:
            output_dir = Path(output_path).resolve()
            print(f"[COPY] === بدء نسخ المخرجات ===")
            print(f"[COPY] المصدر: {output_dir} (exists={output_dir.exists()})")
            if output_dir.exists():
                all_files = list(output_dir.iterdir())
                print(f"[COPY] عدد الملفات في المصدر: {len(all_files)}")
                for item in all_files:
                    print(f"[COPY]   - {item.name} ({'dir' if item.is_dir() else f'{item.stat().st_size} bytes'})")

                skip_files = {"script.py", "result_manifest.json", "run_log.txt"}
                if actual_output_recipe:
                    recipe_out = Path(actual_output_recipe).resolve()
                    recipe_out.mkdir(parents=True, exist_ok=True)
                    print(f"[COPY] الوجهة: {recipe_out} (exists={recipe_out.exists()})")

                    copied_count = 0
                    for f in all_files:
                        if f.is_file() and f.name not in skip_files:
                            dest = recipe_out / f.name
                            try:
                                # لو الملف موجود ومقفول، امسحه الأول
                                if dest.exists():
                                    try:
                                        dest.unlink()
                                        print(f"[COPY] حذف ملف قديم: {dest.name}")
                                    except Exception:
                                        pass
                                shutil.copy2(str(f), str(dest))
                                copied_count += 1
                                print(f"[COPY] OK: {f.name} ({f.stat().st_size} bytes) -> {dest}")
                                if dest.exists():
                                    print(f"[COPY] تأكيد: {dest.name} موجود ({dest.stat().st_size} bytes)")
                                else:
                                    print(f"[COPY] تحذير: {dest.name} مش موجود بعد النسخ!")
                            except Exception as e:
                                print(f"[COPY] خطأ في نسخ {f.name}: {type(e).__name__}: {e}")
                                # محاولة تانية بطريقة مختلفة
                                try:
                                    with open(str(f), 'rb') as src, open(str(dest), 'wb') as dst:
                                        dst.write(src.read())
                                    copied_count += 1
                                    print(f"[COPY] OK (محاولة تانية): {f.name}")
                                except Exception as e2:
                                    # محاولة ثالثة: حفظ باسم جديد (timestamp)
                                    from datetime import datetime as _dt
                                    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                                    stem = f.stem
                                    suffix = f.suffix
                                    alt_name = f"{stem}_{ts}{suffix}"
                                    alt_dest = recipe_out / alt_name
                                    try:
                                        shutil.copy2(str(f), str(alt_dest))
                                        copied_count += 1
                                        print(f"[COPY] OK (اسم بديل): {f.name} -> {alt_name} ({alt_dest.stat().st_size} bytes)")
                                    except Exception as e3:
                                        print(f"[COPY] فشل نهائي: {f.name}: {e2} | بديل: {e3}")

                    for d in all_files:
                        if d.is_dir():
                            dest_dir = recipe_out / d.name
                            try:
                                shutil.copytree(str(d), str(dest_dir), dirs_exist_ok=True)
                                copied_count += 1
                                print(f"[COPY] OK dir: {d.name}")
                            except Exception as e:
                                print(f"[COPY] خطأ في نسخ مجلد {d.name}: {e}")

                    print(f"[COPY] === اكتمل: {copied_count} ملف/مجلد تم نسخهم ===")
                else:
                    print(f"[COPY] تحذير: actual_output_recipe فارغ - مفيش وجهة للنسخ!")
            else:
                print(f"[COPY] تحذير: مجلد المصدر مش موجود: {output_dir}")

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
    output_dir = get_run_output_dir(run_id, run.output_relpath)
    if not output_dir.exists():
        return {"files": []}
    files = [{"name": f.name, "size": f.stat().st_size, "path": f"/api/utilities/runs/{run_id}/files/{f.name}"} for f in output_dir.iterdir() if f.is_file()]
    return {"files": sorted(files, key=lambda x: x["name"])}


@app.get("/api/utilities/runs/{run_id}/files/{filename}")
async def get_run_file(run_id: str, filename: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="اسم ملف غير صالح")
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    file_path = get_run_output_dir(run_id, run.output_relpath) / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="الملف غير موجود")
    return FileResponse(file_path)


@app.get("/api/utilities/runs/{run_id}/download")
async def download_run_outputs(run_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    output_dir = get_run_output_dir(run_id, run.output_relpath)
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
    """تحديد مجلد الإخراج بناءً على output_relpath أو البحث في المسارين"""
    if output_relpath and output_relpath.startswith("longs/"):
        return Path(LONGS_OUTPUT_ROOT) / run_id
    return Path(OUTPUT_ROOT) / run_id

HOST_DATA_DIR = os.getenv("HOST_DATA_DIR", "C:/Users/w10/shorts-runner/data")

@app.post("/api/open-folder")
async def get_host_folder_path(docker_path: str, current_user: User = Depends(get_current_user)):
    """تحويل مسار Docker لمسار Windows وإنشاء المجلد"""
    if not docker_path or ".." in docker_path:
        raise HTTPException(status_code=400, detail="مسار غير صالح")
    if not docker_path.startswith("/app/data/"):
        raise HTTPException(status_code=400, detail="المسار لازم يبدأ بـ /app/data/")
    host_path = docker_path.replace("/app/data/", HOST_DATA_DIR + "/").replace("/", "\\")
    os.makedirs(docker_path, exist_ok=True)
    return {"success": True, "path": host_path}


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
