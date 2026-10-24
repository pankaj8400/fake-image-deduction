"""API route definitions for fake image detection and explanation."""

import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, status, Depends, Query

try:
    from ..config import settings
    from ..models.schemas import (
        DetectionResponse, 
        HealthResponse, 
        ClassesListResponse, 
        ClassDetail,
        ChatRequest,
        ChatResponse,
        UploadedImagesResponse,
        SessionHistoryResponse,
        RegisterRequest,
        LoginRequest,
        AuthResponse,
        UserProfileResponse
    )
    from ..services.langgraph_agent import LangGraphForensicAgent, get_langgraph_agent
    from ..services.model_service import ModelService, get_model_service
    from ..services.database import register_user, authenticate_user, get_user_by_id, get_user_images_count
except (ImportError, ValueError):
    try:
        from app.config import settings
        from app.models.schemas import (
            DetectionResponse, 
            HealthResponse, 
            ClassesListResponse, 
            ClassDetail,
            ChatRequest,
            ChatResponse,
            UploadedImagesResponse,
            SessionHistoryResponse,
            RegisterRequest,
            LoginRequest,
            AuthResponse,
            UserProfileResponse
        )
        from app.services.langgraph_agent import LangGraphForensicAgent, get_langgraph_agent
        from app.services.model_service import ModelService, get_model_service
        from app.services.database import register_user, authenticate_user, get_user_by_id
    except (ImportError, ValueError):
        from config import settings
        from models.schemas import (
            DetectionResponse, 
            HealthResponse, 
            ClassesListResponse, 
            ClassDetail,
            ChatRequest,
            ChatResponse,
            UploadedImagesResponse,
            SessionHistoryResponse,
            RegisterRequest,
            LoginRequest,
            AuthResponse,
            UserProfileResponse
        )
        from services.langgraph_agent import LangGraphForensicAgent, get_langgraph_agent
        from services.model_service import ModelService, get_model_service
        from services.database import register_user, authenticate_user, get_user_by_id

router = APIRouter(prefix="/api", tags=["Detection & Explainability"])


# --- User Authentication Endpoints ---

@router.post(
    "/auth/register",
    response_model=AuthResponse,
    summary="Register New User into SQLite Vault",
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"]
)
async def api_register_user(req: RegisterRequest):
    """Registers a user into the SQLite database with password confirmation check."""
    if req.password != req.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match. Please verify your password confirmation."
        )

    try:
        user = register_user(req.username, req.password)
        token = f"tok_{user['id']}_{user['username']}"
        return AuthResponse(
            status="success",
            message=f"User '{user['username']}' successfully registered in SQLite database.",
            user_id=user["id"],
            username=user["username"],
            token=token
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(e)}")


@router.post(
    "/auth/login",
    response_model=AuthResponse,
    summary="Sign In User & Retrieve Vault Session Token",
    status_code=status.HTTP_200_OK,
    tags=["Authentication"]
)
async def api_login_user(req: LoginRequest):
    """Authenticates username and password against SQLite records."""
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password. Please verify your credentials."
        )

    token = f"tok_{user['id']}_{user['username']}"
    return AuthResponse(
        status="success",
        message="Authentication successful.",
        user_id=user["id"],
        username=user["username"],
        token=token
    )


@router.get(
    "/auth/me",
    response_model=UserProfileResponse,
    summary="Get Current User Profile",
    tags=["Authentication"]
)
async def get_me(
    x_user_id: Optional[int] = Header(None, description="Current authenticated user ID"),
    user_id: Optional[int] = None
):
    """Fetches user profile from SQLite."""
    uid = x_user_id or user_id or 1
    user = get_user_by_id(uid)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return UserProfileResponse(
        id=user["id"],
        username=user["username"],
        created_at=user["created_at"]
    )


# --- System Health & Metadata ---

@router.get("/health", response_model=HealthResponse, summary="Service Health & Status")
async def health_check(
    model_service: ModelService = Depends(get_model_service)
):
    """Returns the operational status of the model, device, and Groq agent."""
    resolved_path = settings.get_resolved_model_path()
    groq_configured = bool(settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY"))

    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
        model_path=str(resolved_path),
        device=str(model_service.device),
        groq_configured=groq_configured,
        groq_model=settings.GROQ_MODEL,
        supported_classes=settings.CLASSES,
        app_version=settings.APP_VERSION,
    )


@router.get("/classes", response_model=ClassesListResponse, summary="List Supported Classes")
async def get_supported_classes():
    """Returns metadata and descriptions for the 4 detectable classes."""
    classes = [
        ClassDetail(
            name="real",
            display_name="Authentic / Real",
            category="authentic",
            description="Pristine, unmanipulated image captured directly from physical camera sensors with organic PRNU noise."
        ),
        ClassDetail(
            name="ai_generated",
            display_name="AI-Generated",
            category="synthetic",
            description="Fully synthesized media generated by generative models (Diffusion, Midjourney, DALL-E, Stable Diffusion, GANs)."
        ),
        ClassDetail(
            name="faceswap",
            display_name="FaceSwap / Deepfake",
            category="manipulated",
            description="Facial replacement deepfake where a source identity is blended onto a target body with boundary seams."
        ),
        ClassDetail(
            name="photoshop",
            display_name="Photoshop / Digital Tampering",
            category="manipulated",
            description="Traditional 2D digital image editing, including cloning, content-aware fills, splicing, and airbrushing."
        ),
    ]
    return ClassesListResponse(supported_classes=classes)


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Analyze Image via LangGraph Pipeline with SQLite User Isolation",
    status_code=status.HTTP_200_OK,
)
async def detect_fake_image(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, WebP)"),
    session_id: Optional[str] = Form(None, description="Optional LangGraph session ID"),
    user_id: Optional[int] = Form(None, description="User ID for isolated evidence vault"),
    x_user_id: Optional[int] = Header(None, description="User ID from header"),
    groq_api_key: Optional[str] = Form(None, description="Optional Groq API key override"),
    x_groq_api_key: Optional[str] = Header(None, description="Optional Groq API key from header"),
    agent: LangGraphForensicAgent = Depends(get_langgraph_agent)
):
    """**LangGraph Forensic Agent Pipeline with Multi-Tenant SQLite Isolation**:
    
    1. **Vision Forensics Node**: Dual-branch model evaluates spatial (ConvNeXt) and frequency (SRM) signatures.
    2. **Explainability Node**: Groq LLM Specialist synthesizes authoritative forensic diagnostic rationale.
    3. **SQLite Vault Registration Node**: Persists image to disk and registers into user's private SQLite vault.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided in upload.")

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to read uploaded file: {str(e)}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    if len(image_bytes) > settings.MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {settings.MAX_IMAGE_SIZE_BYTES // (1024*1024)}MB."
        )

    effective_groq_key = groq_api_key or x_groq_api_key or settings.GROQ_API_KEY
    effective_user_id = user_id or x_user_id or 1

    try:
        result = agent.analyze_image(
            image_bytes=image_bytes,
            user_id=effective_user_id,
            session_id=session_id,
            custom_groq_api_key=effective_groq_key
        )
        return DetectionResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph inference error during analysis: {str(ex)}"
        )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask Follow-up Questions with User-Isolated LangGraph Memory",
    status_code=status.HTTP_200_OK,
)
async def forensic_chat_followup(
    req: ChatRequest,
    x_user_id: Optional[int] = Header(None),
    agent: LangGraphForensicAgent = Depends(get_langgraph_agent)
):
    """Processes conversational questions strictly scoped to the authenticated user's evidence vault."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")

    effective_user_id = req.user_id or x_user_id or 1
    effective_session_id = req.session_id or "default_session"
    effective_groq_key = req.groq_api_key or settings.GROQ_API_KEY

    try:
        result = agent.chat_followup(
            query=req.query.strip(),
            user_id=effective_user_id,
            session_id=effective_session_id,
            image_id=req.image_id,
            custom_groq_api_key=effective_groq_key
        )
        return ChatResponse(**result)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in LangGraph conversational follow-up: {str(ex)}"
        )


@router.get(
    "/images",
    response_model=UploadedImagesResponse,
    summary="Show Registered User-Uploaded Images (Isolated per User)",
    status_code=status.HTTP_200_OK,
)
async def get_uploaded_images(
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    filter_session: bool = Query(False, description="Filter images strictly to this session_id"),
    limit: Optional[int] = Query(None, description="Limit for pagination"),
    offset: int = Query(0, description="Offset for pagination"),
    x_user_id: Optional[int] = Header(None),
    agent: LangGraphForensicAgent = Depends(get_langgraph_agent)
):
    """Returns all images belonging to the specified user across all their sessions by default. Guarantees user isolation."""
    effective_user_id = user_id or x_user_id or 1
    images = agent.get_registered_images(
        user_id=effective_user_id,
        session_id=session_id,
        filter_by_session=filter_session,
        limit=limit,
        offset=offset
    )
    total = get_user_images_count(user_id=effective_user_id, session_id=session_id, filter_by_session=filter_session)
    user_info = get_user_by_id(effective_user_id)
    username = user_info["username"] if user_info else f"user_{effective_user_id}"

    return UploadedImagesResponse(
        session_id=session_id or "all_sessions",
        user_id=effective_user_id,
        username=username,
        total_images=total,
        images=images
    )


@router.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get User-Isolated LangGraph Checkpointed History & Evidence",
    status_code=status.HTTP_200_OK,
)
async def get_session_history(
    session_id: str,
    user_id: Optional[int] = None,
    x_user_id: Optional[int] = Header(None),
    agent: LangGraphForensicAgent = Depends(get_langgraph_agent)
):
    """Returns the message dialogue and registered images for this user and session."""
    effective_user_id = user_id or x_user_id or 1
    history = agent.get_session_history(user_id=effective_user_id, session_id=session_id)
    return SessionHistoryResponse(**history)


