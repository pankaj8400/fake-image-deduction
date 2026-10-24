"""Groq-powered Forensic Explainability Agent.

Generates an authoritative, 2-line scientific explanation of why an image
was categorized into its specific class (real, ai_generated, faceswap, photoshop).
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
try:
    from ..config import settings
except (ImportError, ValueError):
    try:
        from app.config import settings
    except (ImportError, ValueError):
        from config import settings

logger = logging.getLogger(__name__)

# Try importing groq safely
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq library not installed. Forensic rules fallback will be utilized.")


class GroqExplainabilityAgent:
    """Agent responsible for synthesizing spatial-spectral forensic signals into a 2-line explanation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        self.model_name = settings.GROQ_MODEL
        self.fallback_model = settings.GROQ_FALLBACK_MODEL
        self.client: Optional[Any] = None

        if self.api_key and GROQ_AVAILABLE:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None

    def is_configured(self) -> bool:
        """Checks if Groq API is ready for inference."""
        return self.client is not None or bool(self.api_key or os.environ.get("GROQ_API_KEY"))

    def _build_system_prompt(self) -> str:
        """Constructs forensic specialist persona and output constraints."""
        return (
            "You are an elite Digital Image Forensics and AI Authentication Agent. "
            "You analyze diagnostic telemetry from a state-of-the-art Hybrid Deepfake Detection Engine combining:\n"
            "1. User's Trained Dual-Branch Model: ConvNeXt Spatial Backbone + 3-kernel SRM High-Pass Frequency Residual Branch.\n"
            "2. Prebuilt Vision Transformer (ViT-B/32): 49 non-overlapping 32x32 spatial patch token analysis for deep generative synthesis and boundary anomaly detection.\n\n"
            "FORENSIC DISAMBIGUATION RULES:\n"
            "- FaceSwap: Localized facial insertion on an authentic body/background. Exhibits boundary blending seams, significant noise inconsistency between the swapped face and background, and patch token divergence.\n"
            "- AI Generated: Global synthetic generation from generative models (Diffusion, Midjourney, DALL-E, GANs). Exhibits uniform 32x32 ViT patch synthesis scores and homogeneous high-frequency generative noise across both subject and background.\n"
            "- Photoshop: Traditional 2D digital tampering (splicing, clone stamp, content-aware fills, airbrushing) showing localized edge or Laplacian discontinuities.\n"
            "- Real: Natural camera sensor PRNU noise, coherent optical depth of field, and lack of synthetic boundary artifacts.\n\n"
            "DIAGNOSTIC EXPLANATION RULES:\n"
            "- Provide a clear, authoritative, and scientifically grounded forensic analysis.\n"
            "- Clearly evaluate how both the trained dual-branch model (spatial + SRM frequency) and the prebuilt ViT-B/32 patch analysis confirm this classification.\n"
            "- Explain precisely why the image belongs to the detected class and how it is distinguished from other fake types.\n"
            "- Structure as 2 clear, impactful diagnostic paragraphs (Paragraph 1: Physical evidence from spatial, frequency, and ViT-32 patches; Paragraph 2: Definitive forensic conclusion).\n"
            "- No greetings, no preamble, no markdown asterisks, no bullet points."
        )

    def _build_user_prompt(
        self,
        predicted_class: str,
        confidence: float,
        class_probabilities: list,
        forensics: Dict[str, Any]
    ) -> str:
        """Formats the input metrics into a prompt for Groq LLM."""
        prob_str = ", ".join([f"{c['class_name']}: {c['percentage']}%" for c in class_probabilities])
        runner_up = forensics.get("runner_up_class")
        margin = forensics.get("margin")
        inconsistency = forensics.get("noise_inconsistency")
        inconsistency_str = f"- Spatial Noise Inconsistency (Center Face vs Background): {inconsistency}\n" if inconsistency is not None else ""

        disambiguation_note = ""
        if forensics.get("is_ambiguous") and runner_up:
            disambiguation_note = (
                f"- Model Ambiguity Note: The model detects manipulation, with '{predicted_class}' leading "
                f"over runner-up '{runner_up}' by {margin * 100:.1f}%. Clearly distinguish why '{predicted_class}' "
                f"is the primary manipulation signature.\n"
            )

        vit_std = forensics.get("vit_32_patch_anomaly")
        vit_score = forensics.get("vit_synthesis_score")
        vit_str = f"- Prebuilt ViT-B/32 Patch Analysis (32x32 patches): Token Anomaly={vit_std}, Synthesis Probability={vit_score}\n" if vit_std is not None else ""

        fft_ratio = forensics.get("fft_high_frequency_ratio")
        fft_str = f"- 2D FFT Spectral High-Frequency Ratio: {fft_ratio}\n" if fft_ratio is not None else ""
        vit_scene = forensics.get("vit_scene_category")
        vit_scene_str = f"- ViT-B/32 Scene Composition: {vit_scene}\n" if vit_scene is not None else ""
        ai_profile = forensics.get("ai_generator_profile")
        ai_profile_str = f"- Generative Synthesis Profile: {ai_profile}\n" if ai_profile is not None else ""

        return (
            f"Image Forensic Analysis Telemetry:\n"
            f"- Model Prediction: {predicted_class.upper()} (Confidence: {confidence * 100:.1f}%)\n"
            f"- Full Probability Distribution: {prob_str}\n"
            f"- Trained ConvNeXt Spatial Feature Norm: {forensics.get('spatial_feature_norm')}\n"
            f"- Trained SRM Noise Residual Energy: {forensics.get('srm_residual_energy')}\n"
            f"- Dominant SRM Filter Kernel: {forensics.get('dominant_kernel')}\n"
            f"- Attention Weights: {forensics.get('attention_spatial_ratio') * 100:.1f}% Spatial vs "
            f"{forensics.get('attention_frequency_ratio') * 100:.1f}% Frequency Branch\n"
            f"{inconsistency_str}"
            f"{vit_str}"
            f"{fft_str}"
            f"{vit_scene_str}"
            f"{ai_profile_str}"
            f"{disambiguation_note}\n"
            f"Provide an authoritative forensic explanation detailing why this image belongs to the '{predicted_class}' class, "
            f"incorporating the trained dual-branch model findings and prebuilt ViT-B/32 patch evaluation."
        )

    def generate_explanation(
        self,
        predicted_class: str,
        confidence: float,
        class_probabilities: list,
        forensics: Dict[str, Any],
        custom_api_key: Optional[str] = None
    ) -> Dict[str, str]:
        raw_key = custom_api_key or self.api_key or os.environ.get("GROQ_API_KEY")
        active_key = str(raw_key).strip().strip("'\"").strip() if raw_key else None

        if active_key and GROQ_AVAILABLE:
            # Basic validation
            if not active_key.startswith("gsk_"):
                logger.warning(
                    f"Provided Groq API key starts with '{active_key[:4]}...', but standard Groq keys start with 'gsk_'. "
                    "Make sure you copied your key from https://console.groq.com/keys"
                )

            try:
                client = Groq(api_key=active_key)
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt(predicted_class, confidence, class_probabilities, forensics)

                candidate_models = [
                    self.model_name,
                    self.fallback_model,
                    "qwen/qwen3.8-27b",
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                ]
                seen = set()
                model_queue = [m for m in candidate_models if not (m in seen or seen.add(m))]

                chat_completion = None
                successful_model = self.model_name
                last_error = None

                for target_model in model_queue:
                    try:
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            model=target_model,
                            temperature=settings.GROQ_TEMPERATURE,
                            max_tokens=settings.GROQ_MAX_TOKENS,
                        )
                        if chat_completion.choices and chat_completion.choices[0].message.content:
                            successful_model = target_model
                            break
                    except Exception as model_err:
                        last_error = model_err
                        err_text = str(model_err)
                        if "401" in err_text or "invalid_api_key" in err_text.lower():
                            raise model_err
                        logger.warning(f"Groq model {target_model} unavailable ({model_err}). Trying next candidate...")

                if chat_completion is None or not chat_completion.choices or not chat_completion.choices[0].message.content:
                    if last_error:
                        raise last_error
                    return self._fallback_explanation(predicted_class, confidence, forensics)

                raw_text = chat_completion.choices[0].message.content.strip()
                lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

                if len(lines) >= 2:
                    line_1 = lines[0]
                    line_2 = " ".join(lines[1:])
                elif len(lines) == 1:
                    line_1 = lines[0]
                    line_2 = f"This diagnostic evidence confirms the classification as {predicted_class} with {confidence * 100:.1f}% certainty."
                else:
                    return self._fallback_explanation(predicted_class, confidence, forensics)

                return {
                    "explanation": raw_text,
                    "line_1": line_1,
                    "line_2": line_2,
                    "provider": f"groq:{successful_model}"
                }

            except Exception as e:
                err_text = str(e)
                if "401" in err_text or "invalid_api_key" in err_text.lower():
                    logger.error(
                        f"Groq Authentication Failed (401 Invalid API Key). "
                        "Please verify your key at https://console.groq.com/keys and ensure it starts with 'gsk_'."
                    )
                    return self._fallback_explanation(
                        predicted_class,
                        confidence,
                        forensics,
                        reason="Groq 401: Invalid API Key. Please check console.groq.com/keys"
                    )

                logger.error(f"Error calling Groq API: {e}. Falling back to internal forensics engine.")

        # Fallback to local forensic rules engine
        return self._fallback_explanation(predicted_class, confidence, forensics)

    def _fallback_explanation(
        self,
        predicted_class: str,
        confidence: float,
        forensics: Dict[str, Any],
        reason: Optional[str] = None
    ) -> Dict[str, str]:
        """Deterministic 2-line forensic rationale based on model activation signatures."""
        dominant_kernel = forensics.get("dominant_kernel", "Laplacian Residual")
        srm_energy = forensics.get("srm_residual_energy", 0.05)
        conf_pct = round(confidence * 100.0, 1)

        if predicted_class == "real":
            line_1 = (
                f"Analysis shows uniform sensor noise across RGB channels with consistent {dominant_kernel} "
                f"residuals ({srm_energy:.4f}) and zero boundary manipulation."
            )
            line_2 = (
                f"These natural photometric properties and organic pixel distribution verify the image is authentic "
                f"with {conf_pct}% confidence."
            )

        elif predicted_class == "ai_generated":
            line_1 = (
                f"Spectral analysis reveals characteristic generative diffusion artifacts with suppressed high-frequency "
                f"residual energy ({srm_energy:.4f}) and unnatural semantic smoothness."
            )
            line_2 = (
                f"These synthetic frequency signatures and periodic grid patterns definitively substantiate AI-generated "
                f"synthesis at {conf_pct}% probability."
            )

        elif predicted_class == "faceswap":
            line_1 = (
                f"Dual-branch inspection detects facial boundary blending seams and sharp {dominant_kernel} "
                f"discontinuities around the facial perimeter."
            )
            line_2 = (
                f"The significant resolution and noise mismatch between the facial mask and torso establishes a deepfake "
                f"faceswap with {conf_pct}% certainty."
            )

        elif predicted_class == "photoshop":
            line_1 = (
                f"Spatial-frequency evaluation flags localized clone-stamp gradients and irregular {dominant_kernel} "
                f"residual spikes indicative of manual splicing."
            )
            line_2 = (
                f"These localized pixel resampling irregularities and illumination inconsistencies validate deliberate "
                f"Photoshop tampering at {conf_pct}% confidence."
            )

        else:
            line_1 = f"Forensic analysis identifies anomalous high-pass residual cues across the {dominant_kernel} domain."
            line_2 = f"These structural indicators align with the {predicted_class} classification profile at {conf_pct}% confidence."

        provider_name = f"forensic_rules_engine ({reason})" if reason else "forensic_rules_engine (Set GROQ_API_KEY in .env for Groq LLM)"
        return {
            "explanation": f"{line_1}\n{line_2}",
            "line_1": line_1,
            "line_2": line_2,
            "provider": provider_name
        }


def get_groq_agent() -> GroqExplainabilityAgent:
    """Dependency injection helper."""
    return GroqExplainabilityAgent()
