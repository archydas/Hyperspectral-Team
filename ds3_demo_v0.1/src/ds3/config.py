import os
from pathlib import Path

def database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///./ds3.db")

def data_dir() -> Path:
    return Path(os.environ.get("DS3_DATA_DIR", "./data"))
