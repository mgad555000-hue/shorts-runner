"""
نماذج Pydantic للـ API
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class RecipeCreate(BaseModel):
    name: str
    description: Optional[str] = None
    code: str
    input_folder: Optional[str] = None


class RecipeUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    code: Optional[str] = None
    input_folder: Optional[str] = None


class RecipeResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    code: str
    input_folder: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class RunCreate(BaseModel):
    code: str = ""
    input_folder: str
    recipe_id: Optional[int] = None
    model_name: Optional[str] = "gemini-2.5-flash"


class RunResponse(BaseModel):
    id: int
    run_id: str
    recipe_id: Optional[int]
    recipe_name: Optional[str]
    input_folder: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    error_message: Optional[str]
    output_relpath: str

    class Config:
        from_attributes = True


class PathResponse(BaseModel):
    available_folders: List[str]
    data_root: str


class CleanupSettings(BaseModel):
    max_age_days: int
    keep_last_n: int


class CleanupResponse(BaseModel):
    deleted_runs: int
    freed_space_mb: float
    errors: List[str]
    settings: CleanupSettings


class SettingsResponse(BaseModel):
    max_concurrent_runs: int
    timeout_seconds: int
    mock_mode: bool
    cleanup_max_age_days: int
    cleanup_keep_last_n: int


class SettingsUpdate(BaseModel):
    max_concurrent_runs: Optional[int] = None
    timeout_seconds: Optional[int] = None
    mock_mode: Optional[bool] = None
    cleanup_max_age_days: Optional[int] = None
    cleanup_keep_last_n: Optional[int] = None
