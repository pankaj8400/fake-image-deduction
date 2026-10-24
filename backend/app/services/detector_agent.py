"""Master Detector Agent orchestrating model identification and Groq explainability."""

import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

try:
    from .model_service import ModelService, get_model_service
    from .groq_agent import GroqExplainabilityAgent, get_groq_agent
    from ..models.schemas import DetectionResponse, ClassConfidence, ForensicMetrics, ExplainabilityResult
except (ImportError, ValueError):
    try:
        from app.services.model_service import ModelService, get_model_service
        from app.services.groq_agent import GroqExplainabilityAgent, get_groq_agent
        from app.models.schemas import DetectionResponse, ClassConfidence, ForensicMetrics, ExplainabilityResult
    except (ImportError, ValueError):
        from services.model_service import ModelService, get_model_service
        from services.groq_agent import GroqExplainabilityAgent, get_groq_agent
        from models.schemas import DetectionResponse, ClassConfidence, ForensicMetrics, ExplainabilityResult


class DetectorAgent:
    """End-to-end agent coordinating image verification:
    
    1. Phase 1 - Classification & Forensic Feature Extraction:
       Runs image through the Dual-Branch Hybrid Deepfake model.
       Identifies class out of [real, ai_generated, faceswap, photoshop] with confidence.
    
    2. Phase 2 - Explainability Synthesis:
       Feeds the forensic evidence (SRM residual energy, spatial features, attention gate, confidence)
       into the Groq LLM Agent with a specialized forensic prompt.
       Generates a precise 2-line explanation justifying the classification.
    """

    def __init__(
        self,
        model_service: Optional[ModelService] = None,
        groq_agent: Optional[GroqExplainabilityAgent] = None
    ):
        self.model_service = model_service or get_model_service()
        self.groq_agent = groq_agent or get_groq_agent()

    def analyze_image(
        self,
        image_bytes: bytes,
        custom_groq_api_key: Optional[str] = None
    ) -> DetectionResponse:
        """Executes full two-step agent workflow on input image bytes."""
        start_time = time.perf_counter()

        # Step 1: Model Identification & Forensic Extraction
        detection = self.model_service.predict(image_bytes)

        # Step 2: Groq-Powered 2-Line Explanation
        explanation_data = self.groq_agent.generate_explanation(
            predicted_class=detection["predicted_class"],
            confidence=detection["confidence"],
            class_probabilities=detection["class_probabilities"],
            forensics=detection["forensics"],
            custom_api_key=custom_groq_api_key
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Assemble typed response
        probabilities = [ClassConfidence(**c) for c in detection["class_probabilities"]]
        forensic_metrics = ForensicMetrics(**detection["forensics"])
        explainability = ExplainabilityResult(**explanation_data)

        return DetectionResponse(
            status="success",
            predicted_class=detection["predicted_class"],
            confidence=detection["confidence"],
            is_fake=detection["is_fake"],
            class_probabilities=probabilities,
            forensics=forensic_metrics,
            explainability=explainability,
            inference_time_ms=elapsed_ms,
            timestamp=datetime.now(timezone.utc).isoformat()
        )


def get_detector_agent() -> DetectorAgent:
    """Dependency injection helper."""
    return DetectorAgent()
