"""
نظام المصادقة - PIN Authentication + JWT
"""
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Request, HTTPException, Depends
from sqlalchemy.orm import Session
import hashlib
import hmac
from jose import jwt, JWTError

from app.database import get_db, User, Setting, SessionLocal

# JWT settings
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = int(os.getenv("TOKEN_EXPIRE_DAYS", "30"))


def get_jwt_secret() -> str:
    """Get or create JWT secret key from DB settings"""
    env_key = os.getenv("JWT_SECRET_KEY")
    if env_key:
        return env_key
    db = SessionLocal()
    try:
        setting = db.query(Setting).filter(Setting.key == "jwt_secret").first()
        if setting:
            return setting.value
        # Generate and store
        secret = secrets.token_hex(32)
        db.add(Setting(key="jwt_secret", value=secret))
        try:
            db.commit()
        except Exception:
            db.rollback()
            # Another worker might have created it
            setting = db.query(Setting).filter(Setting.key == "jwt_secret").first()
            if setting:
                return setting.value
            raise
        return secret
    finally:
        db.close()


def hash_pin(pin: str) -> str:
    """Hash a 4-digit PIN using HMAC-SHA256 with a salt"""
    salt = secrets.token_hex(16)
    h = hmac.new(salt.encode(), pin.encode(), hashlib.sha256).hexdigest()
    return f"{salt}${h}"


def verify_pin(plain_pin: str, hashed: str) -> bool:
    """Verify PIN against stored hash"""
    try:
        salt, h = hashed.split("$", 1)
        expected = hmac.new(salt.encode(), plain_pin.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(h, expected)
    except Exception:
        return False


def create_token(user_id: int, username: str, is_admin: bool) -> str:
    expire = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "username": username,
        "is_admin": is_admin,
        "exp": expire,
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Extract and validate user from Authorization header or cookie"""
    token = None

    # Try Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]

    # Try cookie
    if not token:
        token = request.cookies.get("auth_token")

    if not token:
        raise HTTPException(status_code=401, detail="غير مسجل الدخول")

    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
        user_id = int(payload.get("sub", 0))
    except (JWTError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="جلسة منتهية - سجل دخول مرة تانية")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="المستخدم غير موجود")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="الحساب معطل")

    return user


def require_admin(request: Request, db: Session = Depends(get_db)) -> User:
    """Same as get_current_user but also checks admin"""
    user = get_current_user(request, db)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="صلاحيات المدير مطلوبة")
    return user


def seed_admin_user():
    """Create default admin user if no users exist"""
    db = SessionLocal()
    try:
        count = db.query(User).count()
        if count == 0:
            default_pin = os.getenv("ADMIN_DEFAULT_PIN", "1234")
            admin = User(
                username="admin",
                display_name="المدير",
                pin_hash=hash_pin(default_pin),
                is_admin=1,
                is_active=1,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(admin)
            db.commit()
            print(f"[Auth] تم إنشاء حساب المدير الافتراضي — غيّر الرمز من لوحة التحكم")
    except Exception as e:
        print(f"[Auth] خطأ في إنشاء حساب المدير: {e}")
        db.rollback()
    finally:
        db.close()
