"""
Shorts Runner - FastAPI Backend
تطبيق تشغيل وصفات الفيديوهات القصيرة
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
from pathlib import Path
from datetime import datetime, timedelta

from app.database import get_db, init_db, Recipe, Run, Setting, SessionLocal
from app.models import (
    RecipeCreate, RecipeUpdate, RecipeResponse,
    RunCreate, RunResponse, PathResponse, CleanupResponse,
    SettingsResponse, SettingsUpdate
)
from app.sandbox import create_sandbox_container


# ========== الإعدادات ==========
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "6"))
CLEANUP_MAX_AGE_DAYS = int(os.getenv("CLEANUP_MAX_AGE_DAYS", "7"))
CLEANUP_KEEP_LAST_N = int(os.getenv("CLEANUP_KEEP_LAST_N", "50"))
MAX_CONCURRENT_RUNS = int(os.getenv("MAX_CONCURRENT_RUNS", "2"))
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")

DATA_ROOT = os.getenv("DATA_ROOT", "./data")
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "./shorts/out")

Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)


async def cleanup_old_runs():
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)
            result = perform_cleanup()
            print(f"[Cleanup] deleted {result['deleted_runs']} runs, freed {result['freed_space_mb']:.2f} MB")
        except Exception as e:
            print(f"[Cleanup] error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    cleanup_task = asyncio.create_task(cleanup_old_runs())
    print(f"[Shorts Runner] Started on port 8001")
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Shorts Runner", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


def mock_execute(run_id: str, code: str, input_folder: str):
    import json
    output_dir = Path(OUTPUT_ROOT) / run_id
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
            if run.created_at < cutoff_date:
                try:
                    output_dir = Path(OUTPUT_ROOT) / run.run_id
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
        output_path = Path(OUTPUT_ROOT)
        total_size = 0
        total_files = 0
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
            "oldest_run": oldest_run.created_at.isoformat() if oldest_run else None,
            "newest_run": newest_run.created_at.isoformat() if newest_run else None,
            "max_concurrent_runs": get_dynamic_settings()["max_concurrent_runs"],
            "mock_mode": is_mock_mode(),
            "cleanup_settings": {"interval_hours": CLEANUP_INTERVAL_HOURS, "max_age_days": get_dynamic_settings()["cleanup_max_age_days"], "keep_last_n": get_dynamic_settings()["cleanup_keep_last_n"]}
        }
    finally:
        db.close()


# ========== API ==========

@app.get("/api/utilities/paths", response_model=PathResponse)
async def get_paths():
    data_root_path = Path(DATA_ROOT)
    folders = []
    if data_root_path.exists():
        try:
            with os.scandir(data_root_path) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False) or entry.is_symlink():
                        folders.append(entry.name)
        except Exception as e:
            print(f"Error scanning DATA_ROOT: {e}")
    return PathResponse(available_folders=sorted(folders), data_root=DATA_ROOT)


@app.post("/api/utilities/folders")
async def create_folder(folder_name: str):
    safe_name = sanitize_folder_name(folder_name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="اسم المجلد غير صالح")
    folder_path = Path(DATA_ROOT) / safe_name
    if folder_path.exists():
        return {"message": "المجلد موجود بالفعل", "folder": safe_name, "created": False}
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        return {"message": "تم إنشاء المجلد بنجاح", "folder": safe_name, "created": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/utilities/recipes/{recipe_id}/create-folder")
async def create_recipe_folder(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="الوصفة غير موجودة")
    safe_name = sanitize_folder_name(recipe.name).replace(' ', '_')
    if not safe_name:
        safe_name = f"recipe_{recipe_id}"
    folder_path = Path(DATA_ROOT) / safe_name
    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        recipe.input_folder = safe_name
        db.commit()
        return {"message": "تم إنشاء المجلد", "folder": safe_name, "recipe_id": recipe_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/utilities/create-all-recipe-folders")
async def create_all_recipe_folders(db: Session = Depends(get_db)):
    recipes = db.query(Recipe).all()
    created, skipped, errors = [], [], []
    for recipe in recipes:
        if recipe.input_folder:
            if (Path(DATA_ROOT) / recipe.input_folder).exists():
                skipped.append({"id": recipe.id, "name": recipe.name})
                continue
        safe_name = sanitize_folder_name(recipe.name).replace(' ', '_') or f"recipe_{recipe.id}"
        try:
            (Path(DATA_ROOT) / safe_name).mkdir(parents=True, exist_ok=True)
            recipe.input_folder = safe_name
            created.append({"id": recipe.id, "name": recipe.name, "folder": safe_name})
        except Exception as e:
            errors.append({"id": recipe.id, "error": str(e)})
    db.commit()
    return {"created": created, "skipped": skipped, "errors": errors, "summary": {"total": len(recipes), "created": len(created), "skipped": len(skipped), "errors": len(errors)}}


@app.post("/api/utilities/runs", response_model=RunResponse)
async def create_run(run_data: RunCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    validate_path(run_data.input_folder)
    check_concurrency(db)
    run_id = str(uuid.uuid4())
    output_relpath = f"shorts/out/{run_id}"
    recipe_name = None
    code_to_run = run_data.code or ""
    if run_data.recipe_id:
        recipe = db.query(Recipe).filter(Recipe.id == run_data.recipe_id).first()
        if recipe:
            recipe_name = recipe.name
            if not code_to_run.strip():
                code_to_run = recipe.code or ""
    db_run = Run(run_id=run_id, recipe_id=run_data.recipe_id, recipe_name=recipe_name, input_folder=run_data.input_folder, status="pending", output_relpath=output_relpath)
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    background_tasks.add_task(execute_run, run_id=run_id, code=code_to_run, input_folder=run_data.input_folder)
    return RunResponse.model_validate(db_run)


def execute_run(run_id: str, code: str, input_folder: str):
    db = SessionLocal()
    try:
        db_run = db.query(Run).filter(Run.run_id == run_id).first()
        if not db_run or db_run.status == "cancelled":
            return
        start_time = time.time()
        db_run.status = "running"
        db_run.started_at = datetime.utcnow()
        db.commit()
        if is_mock_mode():
            success, output_path, error_msg = mock_execute(run_id, code, input_folder)
        else:
            success, output_path, error_msg = create_sandbox_container(run_id, code, input_folder)
        execution_time_ms = int((time.time() - start_time) * 1000)
        if success and output_path:
            output_dir = Path(output_path)
            if output_dir.exists():
                auto_files = {"script.py", "run_log.txt", "result_manifest.json", "batch_job_info.json"}
                actual_files = {f.name for f in output_dir.iterdir() if f.is_file()}
                if actual_files.issubset(auto_files) and not [d for d in output_dir.iterdir() if d.is_dir()]:
                    try:
                        shutil.rmtree(output_dir)
                    except Exception:
                        pass
        db_run = db.query(Run).filter(Run.run_id == run_id).first()
        if db_run and db_run.status != "cancelled":
            db_run.status = "completed" if success else "failed"
            db_run.completed_at = datetime.utcnow()
            db_run.execution_time_ms = execution_time_ms
            if error_msg:
                db_run.error_message = error_msg[:2000]
            db.commit()
    finally:
        db.close()


@app.get("/api/utilities/runs", response_model=List[RunResponse])
async def list_runs(skip: int = 0, limit: int = 50, status: Optional[str] = Query(None), recipe_id: Optional[int] = Query(None), db: Session = Depends(get_db)):
    query = db.query(Run)
    if status:
        query = query.filter(Run.status == status)
    if recipe_id:
        query = query.filter(Run.recipe_id == recipe_id)
    return [RunResponse.model_validate(r) for r in query.order_by(Run.created_at.desc()).offset(skip).limit(limit).all()]


@app.get("/api/utilities/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    return RunResponse.model_validate(run)


@app.post("/api/utilities/runs/{run_id}/cancel")
async def cancel_run(run_id: str, db: Session = Depends(get_db)):
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
async def get_run_log(run_id: str):
    log_path = Path(OUTPUT_ROOT) / run_id / "run_log.txt"
    if not log_path.exists():
        return {"log": "لا يوجد سجل بعد"}
    try:
        return {"log": log_path.read_text(encoding="utf-8")}
    except Exception:
        return {"log": "خطأ في قراءة السجل"}


@app.get("/api/utilities/runs/{run_id}/manifest")
async def get_run_manifest(run_id: str):
    manifest_path = Path(OUTPUT_ROOT) / run_id / "result_manifest.json"
    if not manifest_path.exists():
        return {"manifest": None}
    return FileResponse(manifest_path, media_type="application/json")


@app.get("/api/utilities/runs/{run_id}/files")
async def list_run_files(run_id: str):
    output_dir = Path(OUTPUT_ROOT) / run_id
    if not output_dir.exists():
        return {"files": []}
    files = [{"name": f.name, "size": f.stat().st_size, "path": f"/api/utilities/runs/{run_id}/files/{f.name}"} for f in output_dir.iterdir() if f.is_file()]
    return {"files": sorted(files, key=lambda x: x["name"])}


@app.get("/api/utilities/runs/{run_id}/files/{filename}")
async def get_run_file(run_id: str, filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="اسم ملف غير صالح")
    file_path = Path(OUTPUT_ROOT) / run_id / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="الملف غير موجود")
    return FileResponse(file_path)


@app.get("/api/utilities/runs/{run_id}/download")
async def download_run_outputs(run_id: str):
    output_dir = Path(OUTPUT_ROOT) / run_id
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
async def get_stats():
    return get_storage_stats()


@app.post("/api/utilities/cleanup", response_model=CleanupResponse)
async def trigger_cleanup(max_age_days: int = None, keep_last_n: int = None):
    return CleanupResponse(**perform_cleanup(max_age_days, keep_last_n))


@app.delete("/api/utilities/runs/{run_id}")
async def delete_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="التشغيل غير موجود")
    output_dir = Path(OUTPUT_ROOT) / run_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    db.delete(run)
    db.commit()
    return {"message": "تم الحذف", "run_id": run_id}


# ========== Recipes CRUD ==========

@app.post("/api/utilities/recipes", response_model=RecipeResponse)
async def create_recipe(recipe: RecipeCreate, db: Session = Depends(get_db)):
    db_recipe = Recipe(**recipe.model_dump())
    db.add(db_recipe)
    db.commit()
    db.refresh(db_recipe)
    return RecipeResponse.model_validate(db_recipe)


@app.get("/api/utilities/recipes", response_model=List[RecipeResponse])
async def list_recipes(db: Session = Depends(get_db)):
    return [RecipeResponse.model_validate(r) for r in db.query(Recipe).order_by(Recipe.created_at.desc()).all()]


@app.get("/api/utilities/recipes/{recipe_id}", response_model=RecipeResponse)
async def get_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="الوصفة غير موجودة")
    return RecipeResponse.model_validate(recipe)


@app.put("/api/utilities/recipes/{recipe_id}", response_model=RecipeResponse)
async def update_recipe(recipe_id: int, recipe_update: RecipeUpdate, db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="الوصفة غير موجودة")
    for key, value in recipe_update.model_dump(exclude_unset=True).items():
        setattr(recipe, key, value)
    db.commit()
    db.refresh(recipe)
    return RecipeResponse.model_validate(recipe)


@app.delete("/api/utilities/recipes/{recipe_id}")
async def delete_recipe(recipe_id: int, db: Session = Depends(get_db)):
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="الوصفة غير موجودة")
    db.delete(recipe)
    db.commit()
    return {"message": "تم الحذف"}


# ========== Settings ==========

@app.get("/api/utilities/settings")
async def get_settings():
    return get_dynamic_settings()


@app.put("/api/utilities/settings")
async def update_settings(updates: SettingsUpdate, db: Session = Depends(get_db)):
    for key, value in updates.model_dump(exclude_unset=True).items():
        existing = db.query(Setting).filter(Setting.key == key).first()
        str_value = str(value).lower() if isinstance(value, bool) else str(value)
        if existing:
            existing.value = str_value
        else:
            db.add(Setting(key=key, value=str_value))
    db.commit()
    return get_dynamic_settings()


# ========== Static ==========

app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}
