"""End-to-End Verification Test for Fake Image Detection & Groq Agent."""

import io
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import torch

# Ensure backend directory is in path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import settings
from app.services.model_service import ModelService
from app.services.groq_agent import GroqExplainabilityAgent
from app.services.detector_agent import DetectorAgent
from app.models.architecture import HybridDeepfakeDetector


def generate_sample_image() -> bytes:
    """Generates a synthetic 256x256 test image in memory as JPEG bytes."""
    img = Image.new("RGB", (256, 256), color=(45, 85, 125))
    draw = ImageDraw.Draw(img)
    # Add high-contrast geometric and edge patterns
    draw.ellipse([40, 40, 216, 216], fill=(200, 160, 140), outline=(255, 255, 255), width=2)
    draw.rectangle([80, 80, 176, 176], fill=(120, 180, 200), outline=(0, 0, 0), width=1)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def run_tests():
    print("=" * 65)
    print("RUNNING FAKE IMAGE DETECTION & GROQ AGENT TEST SUITE")
    print("=" * 65)

    # 1. Config Check
    print("\n[Test 1] Checking configuration...")
    resolved_path = settings.get_resolved_model_path()
    print(f"  Model Path: {resolved_path}")
    print(f"  Model Exists: {resolved_path.exists()}")
    print(f"  Classes ({len(settings.CLASSES)}): {settings.CLASSES}")
    print(f"  Device: {settings.DEVICE}")
    assert resolved_path.exists(), f"Model file not found at {resolved_path}"
    assert len(settings.CLASSES) == 4, "Must have exactly 4 classes"
    print("  --> PASS: Configuration verified.")

    # 2. Architecture & Strict Weight Loading Check
    print("\n[Test 2] Loading PyTorch model & verifying parameter weights...")
    model_service = ModelService.get_instance()
    model_service.load_model()
    assert model_service.is_loaded, "Model failed to load"
    assert model_service.model is not None, "Model is None"
    print(f"  Model is active on: {model_service.device}")
    print("  --> PASS: Model successfully loaded into memory with strict matching.")

    # 3. Model Inference & Forensic Feature Extraction
    print("\n[Test 3] Testing dual-branch inference on sample image...")
    sample_bytes = generate_sample_image()
    detection = model_service.predict(sample_bytes)
    print(f"  Predicted Class: {detection['predicted_class']}")
    print(f"  Confidence: {detection['confidence'] * 100:.2f}%")
    print(f"  Is Fake: {detection['is_fake']}")
    print(f"  Class Breakdown: {detection['class_probabilities']}")
    print(f"  Forensic Metrics: {detection['forensics']}")

    assert detection["predicted_class"] in settings.CLASSES
    assert 0.0 <= detection["confidence"] <= 1.0
    assert len(detection["class_probabilities"]) == 4
    assert "srm_residual_energy" in detection["forensics"]
    print("  --> PASS: Dual-branch spatial/frequency inference verified.")

    # 4. Groq Explainability Agent Check
    print("\n[Test 4] Testing Groq Explainability Agent...")
    groq_agent = GroqExplainabilityAgent()
    print(f"  Groq API Configured: {groq_agent.is_configured()}")
    
    explanation_res = groq_agent.generate_explanation(
        predicted_class=detection["predicted_class"],
        confidence=detection["confidence"],
        class_probabilities=detection["class_probabilities"],
        forensics=detection["forensics"]
    )

    print(f"  Provider: {explanation_res['provider']}")
    print(f"  Line 1: {explanation_res['line_1']}")
    print(f"  Line 2: {explanation_res['line_2']}")
    assert len(explanation_res["line_1"]) > 10, "Line 1 is too short"
    assert len(explanation_res["line_2"]) > 10, "Line 2 is too short"
    print("  --> PASS: 2-line explainability generation verified.")

    # 5. Master Detector Agent End-to-End Orchestration
    print("\n[Test 5] Testing Master DetectorAgent and LangGraphForensicAgent workflow...")
    agent = DetectorAgent(model_service=model_service, groq_agent=groq_agent)
    final_response = agent.analyze_image(sample_bytes)
    print(f"  Status: {final_response.status}")
    print(f"  Class: {final_response.predicted_class} ({final_response.confidence * 100:.1f}%)")
    print(f"  Is Fake: {final_response.is_fake}")
    print(f"  Execution Time: {final_response.inference_time_ms} ms")
    assert final_response.status == "success"
    assert final_response.inference_time_ms > 0

    from app.services.langgraph_agent import get_langgraph_agent
    langgraph_agent = get_langgraph_agent()
    lg_result = langgraph_agent.analyze_image(sample_bytes, session_id="test_suite_sess")
    print(f"  LangGraph Image ID: {lg_result['image_id']}")
    print(f"  LangGraph Registered Images: {len(lg_result['uploaded_images'])}")
    assert lg_result["status"] == "success"
    assert lg_result["image_id"] is not None
    assert len(lg_result["uploaded_images"]) >= 1
    print("  --> PASS: Master Detector Agent and LangGraph Forensic Agent verified!")

    # 6. FastAPI Router Health Check
    print("\n[Test 6] Testing FastAPI routes via TestClient...")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200, f"Health check failed: {health.text}"
    health_json = health.json()
    print(f"  Health endpoint: {health_json}")
    assert health_json["model_loaded"] is True
    assert set(health_json["supported_classes"]) == {"real", "ai_generated", "faceswap", "photoshop"}

    classes_resp = client.get("/api/classes")
    assert classes_resp.status_code == 200
    print(f"  Classes endpoint returned {len(classes_resp.json()['supported_classes'])} classes.")

    # Test /api/detect endpoint with multipart upload and session memory
    detect_resp = client.post(
        "/api/detect",
        data={"session_id": "test_suite_sess"},
        files={"file": ("test.jpg", sample_bytes, "image/jpeg")}
    )
    assert detect_resp.status_code == 200, f"Detect failed: {detect_resp.text}"
    detect_json = detect_resp.json()
    print(f"  API /detect returned class: {detect_json['predicted_class']}")
    print(f"  API /detect image ID: {detect_json.get('image_id')}")
    print(f"  API /detect uploaded images: {len(detect_json.get('uploaded_images') or [])}")
    assert detect_json["session_id"] == "test_suite_sess"
    assert detect_json["image_id"] is not None

    # Test /api/chat follow-up with LangGraph memory
    chat_resp = client.post(
        "/api/chat",
        json={
            "session_id": "test_suite_sess",
            "query": "Show registered uploaded images and explain the classification."
        }
    )
    assert chat_resp.status_code == 200, f"Chat failed: {chat_resp.text}"
    chat_json = chat_resp.json()
    print(f"  API /chat answer: {chat_json['answer'][:90]}...")
    print(f"  API /chat returned registered images count: {len(chat_json.get('uploaded_images') or [])}")
    assert len(chat_json.get("uploaded_images") or []) >= 1

    # Test /api/images endpoint
    images_resp = client.get("/api/images?session_id=test_suite_sess")
    assert images_resp.status_code == 200
    images_json = images_resp.json()
    print(f"  API /images total: {images_json['total_images']}")
    assert images_json["total_images"] >= 1

    # Test /api/sessions/{session_id}/history endpoint
    hist_resp = client.get("/api/sessions/test_suite_sess/history")
    assert hist_resp.status_code == 200
    hist_json = hist_resp.json()
    print(f"  API /sessions/history message count: {len(hist_json['messages'])}")
    assert len(hist_json["messages"]) >= 1

    print("  --> PASS: Full FastAPI HTTP API endpoints operational!")

    print("\n" + "=" * 65)
    print("ALL 6 PRODUCTION TEST PHASES PASSED WITH 100% SUCCESS!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()

