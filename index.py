# Vercel entry point: exposes the FastAPI app defined in ../app.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import app  # noqa: E402,F401
