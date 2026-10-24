"""FastAPI Application Entry Point for Fake Image Detection & Explainability."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import sys
from pathlib import Path

# Add backend and app dirs to sys.path so uvicorn main:app works from anywhere
_app_dir = Path(__file__).resolve().parent
_backend_dir = _app_dir.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

try:
    from app.config import settings
    from app.services.model_service import get_model_service
    from app.services.groq_agent import get_groq_agent
    from app.api.routes import router as api_router
except (ImportError, ValueError):
    try:
        from .config import settings
        from .services.model_service import get_model_service
        from .services.groq_agent import get_groq_agent
        from .api.routes import router as api_router
    except (ImportError, ValueError):
        from config import settings
        from services.model_service import get_model_service
        from services.groq_agent import get_groq_agent
        from api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app.main")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown events."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Preload PyTorch Model into memory
    model_service = get_model_service()
    try:
        model_service.load_model()
        logger.info("Deepfake detection model successfully preloaded into memory.")
    except Exception as e:
        logger.error(f"Failed to preload model: {e}. Model will attempt to load on first request.")

    # Check Groq configuration
    groq_agent = get_groq_agent()
    if groq_agent.is_configured():
        logger.info(f"Groq Explainability Agent active with model: {settings.GROQ_MODEL}")
    else:
        logger.warning(
            "GROQ_API_KEY is not configured. The agent will run with the internal "
            "forensic rules engine until GROQ_API_KEY is set in .env."
        )

    yield

    logger.info("Shutting down application resources.")


# Create FastAPI instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade FastAPI service combining a dual-branch neural network "
        "(ConvNeXt Spatial + SRM Frequency) and a Groq LLM Agent. Detects 4 classes: "
        "real, ai_generated, faceswap, and photoshop, followed by an authoritative "
        "2-line explainability report."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Mount user uploads directory for serving uploaded image evidence
UPLOADS_DIR = _app_dir / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

FRONTEND_DIST = _backend_dir.parent / "frontend" / "dist"

# Mount frontend assets if built
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend_assets")
elif STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, summary="Interactive Web Testing UI", include_in_schema=False)
@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serves the interactive React web interface for browser testing."""
    react_index = FRONTEND_DIST / "index.html"
    if react_index.exists():
        return FileResponse(react_index)
    static_index = STATIC_DIR / "index.html"
    if static_index.exists():
        return FileResponse(static_index)
    return HTMLResponse(
        content=(
            "<h1>ForensIQ API is Running</h1>"
            "<p>React frontend is active at <a href='http://localhost:5173'>http://localhost:5173</a>.</p>"
            "<p>API Docs: <a href='/docs'>/docs</a></p>"
        )
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
