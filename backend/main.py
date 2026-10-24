"""Root application entrypoint for the backend directory.

Allows running directly from backend/ with:
    uvicorn main:app --reload
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Expose app from app.main
from app.main import app
from app.config import settings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True, reload_dirs=["app"])
