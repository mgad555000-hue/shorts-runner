"""
قاعدة بيانات SQLite للـ Recipes و Runs
مع دعم الترقية التلقائية للأعمدة الجديدة
"""
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Float, inspect, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import sys

# مسار قاعدة البيانات
if sys.platform == "win32":
    _db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "database.db")
    os.makedirs(os.path.dirname(_db_path), exist_ok=True)
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_db_path}")
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/data/database.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Recipe(Base):
    """قوالب الكود المحفوظة"""
    __tablename__ = "shorts_recipes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    code = Column(Text, nullable=False)
    input_folder = Column(String, nullable=True)
    recipe_type = Column(String, default="shorts")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Setting(Base):
    """إعدادات ديناميكية مخزنة في قاعدة البيانات"""
    __tablename__ = "shorts_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, nullable=False, index=True)
    value = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class User(Base):
    """المستخدمين"""
    __tablename__ = "auth_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    pin_hash = Column(String, nullable=False)
    is_admin = Column(Integer, default=0)  # 0/1
    is_active = Column(Integer, default=1)  # 0/1
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserPermission(Base):
    """صلاحيات المستخدمين على الوصفات"""
    __tablename__ = "auth_permissions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    recipe_id = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Run(Base):
    """سجل التشغيلات"""
    __tablename__ = "shorts_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, index=True, nullable=False)
    recipe_id = Column(Integer, nullable=True)
    recipe_name = Column(String, nullable=True)
    input_folder = Column(String, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    output_relpath = Column(String, nullable=False)
    user_id = Column(Integer, nullable=True)


class ApiUsage(Base):
    """سجل استهلاك التوكنز لكل API call"""
    __tablename__ = "api_usage"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, index=True, nullable=False)
    step_id = Column(String, nullable=True)
    call_type = Column(String, nullable=False)  # "direct" أو "batch"
    provider = Column(String, nullable=False)    # gemini, vertex, openai, claude
    model = Column(String, nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    thinking_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    estimated_cost_usd = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


def _migrate_db():
    """ترقية آمنة"""
    insp = inspect(engine)
    if insp.has_table("shorts_runs"):
        existing = {col["name"] for col in insp.get_columns("shorts_runs")}
        new_columns = {
            "started_at": "ALTER TABLE shorts_runs ADD COLUMN started_at DATETIME",
            "execution_time_ms": "ALTER TABLE shorts_runs ADD COLUMN execution_time_ms INTEGER",
        }
        with engine.connect() as conn:
            for col_name, sql in new_columns.items():
                if col_name not in existing:
                    conn.execute(text(sql))
                    conn.commit()
    if insp.has_table("shorts_recipes"):
        existing = {col["name"] for col in insp.get_columns("shorts_recipes")}
        if "recipe_type" not in existing:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE shorts_recipes ADD COLUMN recipe_type TEXT DEFAULT 'shorts'"))
                conn.commit()
    if insp.has_table("shorts_runs"):
        existing = {col["name"] for col in insp.get_columns("shorts_runs")}
        if "user_id" not in existing:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE shorts_runs ADD COLUMN user_id INTEGER"))
                conn.commit()


def init_db():
    Base.metadata.create_all(bind=engine)
    try:
        _migrate_db()
    except Exception as e:
        print(f"[DB Migration] Warning: {e}")
    # إنشاء جدول api_usage لو مش موجود (create_all بيعمله أوتوماتيك)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
