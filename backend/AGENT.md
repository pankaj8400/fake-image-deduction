# 🕵️ Backend AI Agent Architecture & Technical Specification

This document details the internal **AI Agent Systems** operating within the backend: the **`DetectorAgent`** (Master Forensic Orchestrator) and the **`GroqExplainabilityAgent`** (LLM Forensic Diagnostic Specialist).

---

## 🏛️ Agent System Architecture

```
                    ┌───────────────────────────────────┐
                    │      Input Image Byte Stream      │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │       Phase 1: DetectorAgent (Orchestrator)     │
             │   Calls: app.services.model_service.ModelService│
             └────────────────────────┬────────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        ┌─────────────────────┐               ┌─────────────────────┐
        │   Spatial Branch    │               │  Frequency Branch   │
        │   (ConvNeXt-Tiny)   │               │   (SRM Residuals)   │
        └──────────┬──────────┘               └──────────┬──────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │   Attention Gate Fusion   │
                        │ (Spatial/Spectral Weights)│
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │     Neural Forensics      │
                        │ [Class, Conf, SRM, Ratio] │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │     Phase 2: GroqExplainabilityAgent (LLM)      │
             │       app.services.groq_agent.GroqAgent         │
             ├─────────────────────────────────────────────────┤
             │  • Ingests Telemetry & Class Disambiguation     │
             │  • Evaluates Noise Inconsistency (Face vs Body) │
             │  • Selects Active Groq LLM (qwen3.8-27b)        │
             │  • Synthesizes Physical Evidence & Verdict      │
             └────────────────────────┬────────────────────────┘
                                      │
                                      ▼
             ┌─────────────────────────────────────────────────┐
             │            Typed DetectionResponse              │
             │  Status, Class, Conf, Forensics, Explainability │
             └─────────────────────────────────────────────────┘
```

---

## 🤖 1. Master Detector Agent (`DetectorAgent`)

- **File**: [`app/services/detector_agent.py`](file:///c:/Users/dell/Desktop/project/backend/app/services/detector_agent.py)
- **Role**: Master controller coordinating computer vision inference with language reasoning.

### Execution Workflow (`analyze_image`):
1. **Timestamping**: Starts high-resolution performance counter (`time.perf_counter()`).
2. **Vision Extraction**: Invokes `ModelService.predict(image_bytes)` to obtain:
   - Softmax probabilities across `real`, `photoshop`, `ai_generated`, `faceswap`.
   - SRM residual energy, dominant noise kernel, spatial attention ratios.
   - Spatial noise inconsistency between facial bounding region and background.
   - Ambiguity flags and margin against the runner-up class.
3. **Agent Hand-off**: Passes structured telemetry to `GroqExplainabilityAgent.generate_explanation()`.
4. **Typed Response Packaging**: Assembles the results into a validated Pydantic [`DetectionResponse`](file:///c:/Users/dell/Desktop/project/backend/app/models/schemas.py#L37).

### Programmatic Invocation:
```python
from app.services.detector_agent import get_detector_agent

agent = get_detector_agent()
with open("sample_evidence.jpg", "rb") as f:
    response = agent.analyze_image(f.read())

print(f"Verdict: {response.predicted_class} ({response.confidence * 100:.1f}%)")
print(f"Report: {response.explainability.explanation}")
```

---

## 🧠 2. Groq Explainability Agent (`GroqExplainabilityAgent`)

- **File**: [`app/services/groq_agent.py`](file:///c:/Users/dell/Desktop/project/backend/app/services/groq_agent.py)
- **Role**: Specialized forensic LLM agent that articulates physical evidence into an authoritative diagnostic report.

### Primary Responsibilities:
1. **Physical Artifact Translation**: Translates abstract tensor norms and frequency spikes into human-understandable forensic explanations (e.g. PRNU sensor noise, boundary seams, generative diffusion artifacts).
2. **Manipulation Disambiguation**: Differentiates close-call classes using spectral cues:
   - **`faceswap` vs `photoshop`**: Flags local facial perimeter blending and center vs background noise inconsistency.
   - **`ai_generated` vs `real`**: Flags homogeneous generative high-frequency suppression across the entire canvas vs organic camera sensor PRNU noise.
3. **Model Fallback Chain**:
   ```
   Primary: qwen/qwen3.8-27b
       └── Fallback 1: openai/gpt-oss-120b
           └── Fallback 2: openai/gpt-oss-20b
               └── Fallback 3: Deterministic Forensic Rules Engine (Zero crash guarantee)
   ```

---

## 📜 Agent Prompt Engineering

### System Prompt Specification:
```text
You are an elite Digital Image Forensics and AI Authentication Agent.
You analyze diagnostic telemetry from a state-of-the-art Hybrid Deepfake Detection Engine combining:
1. User's Trained Dual-Branch Model: ConvNeXt Spatial Backbone + 3-kernel SRM High-Pass Frequency Residual Branch.

FORENSIC DISAMBIGUATION RULES:
- FaceSwap: Localized facial insertion on an authentic body/background. Exhibits boundary blending seams, significant noise inconsistency between the swapped face and background.
- AI Generated: Global synthetic generation from generative models (Diffusion, Midjourney, DALL-E, GANs). Exhibits homogeneous high-frequency generative noise across both subject and background.
- Photoshop: Traditional 2D digital tampering (splicing, clone stamp, content-aware fills, airbrushing) showing localized edge or Laplacian discontinuities.
- Real: Natural camera sensor PRNU noise, coherent optical depth of field, and lack of synthetic boundary artifacts.

DIAGNOSTIC EXPLANATION RULES:
- Provide a clear, authoritative, and scientifically grounded forensic analysis.
- Clearly evaluate how the trained dual-branch model (spatial + SRM frequency) confirms this classification.
- Explain precisely why the image belongs to the detected class and how it is distinguished from other fake types.
- Structure as 2 clear, impactful diagnostic paragraphs (Paragraph 1: Physical evidence from spatial and frequency telemetry; Paragraph 2: Definitive forensic conclusion).
- No greetings, no preamble, no markdown asterisks, no bullet points.
```

### Dynamic User Prompt Telemetry Ingestion:
```text
Image Forensic Analysis Telemetry:
- Model Prediction: FACESWAP (Confidence: 94.2%)
- Full Probability Distribution: real: 4.65%, photoshop: 0.75%, ai_generated: 0.40%, faceswap: 94.20%
- Trained ConvNeXt Spatial Feature Norm: 5.17
- Trained SRM Noise Residual Energy: 0.0087
- Dominant SRM Filter Kernel: Cross-Star Residual
- Attention Weights: 49.7% Spatial vs 50.3% Frequency Branch
- Spatial Noise Inconsistency (Center Face vs Background): 0.6288
- Model Ambiguity Note: [Included if top margin is below threshold]

Provide an authoritative forensic explanation detailing why this image belongs to the 'faceswap' class, incorporating the trained dual-branch model findings.
```

---

## 📊 Extracted Forensic Telemetry Signals

The agent consumes and reports the following signals:

| Metric | Field | Normal Range | Forensic Significance |
| :--- | :--- | :--- | :--- |
| **SRM Residual Energy** | `srm_residual_energy` | `0.0001 - 0.0500` | High-pass filtered noise energy. Suppressed in diffusion models; elevated in uncompressed authentic images. |
| **Dominant Kernel** | `dominant_kernel` | `Laplacian`, `Edge`, `Star` | Identifies which directional SRM filter captured the strongest tampering residual. |
| **Spatial Attention Ratio**| `attention_spatial_ratio`| `0.0 - 1.0` | Proportion of attention assigned to ConvNeXt semantic features vs SRM frequency noise. |
| **Noise Inconsistency** | `noise_inconsistency` | `0.0 - 2.5` | Ratio between face and background noise variance. High ($> 0.8$) indicates FaceSwap identity insertion. |
| **Runner-Up Margin** | `margin` | `0.0 - 1.0` | Gap between rank-1 and rank-2 probabilities. Triggers disambiguation mode when $< 0.15$. |

---

## 🛡️ Deterministic Fallback Engine

If the external Groq API is offline, unconfigured, or rate-limited, the `GroqExplainabilityAgent` automatically engages its local deterministic rules engine:
- Generates scientifically structured diagnostic rationales customized to the predicted class and extracted SRM residuals.
- Appends provider indicator `forensic_rules_engine` so API consumers know the origin of the explanation.
- Ensures the system **never fails or throws an HTTP 500 error** due to third-party API downtime.

---

## 🧪 Testing the Agents

To run the end-to-end agent verification suite:
```powershell
cd backend
uv run python test_api.py
```

This verifies:
1. `ModelService` parameter loading and device allocation (`cpu`/`cuda`).
2. `DetectorAgent` orchestration and Pydantic schema validation.
3. `GroqExplainabilityAgent` LLM connectivity, temperature calibration, and fallback behaviors.
4. FastAPI route endpoints (`/api/detect`, `/api/health`, `/api/classes`).

---

## 🧠 3. LangGraph Forensic Agent (`LangGraphForensicAgent`)

The backend incorporates a **LangGraph StateGraph** pipeline equipped with **`MemorySaver` checkpointing** for multi-turn conversational forensic investigation and user image tracking:

```
                           ┌───────────────────────────┐
                           │      Incoming Payload     │
                           └─────────────┬─────────────┘
                                         │
                        [Has `query`?] ──┴── [Has `image_bytes`?]
                               │                     │
                               ▼                     ▼
             ┌───────────────────────────┐   ┌───────────────────────────┐
             │ conversational_followup   │   │ extract_vision_forensics  │
             │ (LangGraph Memory Node)   │   │ (ConvNeXt + SRM Residuals)│
             └─────────────┬─────────────┘   └─────────────┬─────────────┘
                           │                               ▼
                           │                 ┌───────────────────────────┐
                           │                 │ generate_explanation      │
                           │                 │ (Groq LLM Specialist)     │
                           │                 └─────────────┬─────────────┘
                           │                               ▼
                           │                 ┌───────────────────────────┐
                           │                 │ register_image_memory     │
                           │                 │ (Disk & Session Registry) │
                           │                 └─────────────┬─────────────┘
                           ▼                               ▼
                     ┌───────────┐                   ┌───────────┐
                     │    END    │                   │    END    │
                     └───────────┘                   └───────────┘
```

### Key Graph Nodes:
1. **`extract_vision_forensics`**: Evaluates the image via dual-branch model (ConvNeXt spatial + SRM frequency residuals).
2. **`generate_explanation`**: Calls the Groq LLM Specialist to synthesize a 2-line diagnostic report.
3. **`register_image_memory`**: Writes the image to disk (`/uploads/{id}.jpg`), saves durable JSON metadata (`/uploads/{id}.json`), stores it in `SessionImageRegistry`, and checkpoints state.
4. **`conversational_followup`**: Uses checkpointed conversation history and session image memory to answer user follow-up questions with full awareness of the active image and all registered session images.

### Dedicated Endpoints:
- `POST /api/detect`: Executes full LangGraph pipeline on image upload and registers session evidence.
- `POST /api/chat`: Handles follow-up conversational inquiries using LangGraph MemorySaver checkpointing.
- `GET /api/images`: Returns all registered uploaded images in the session.
- `GET /api/sessions/{session_id}/history`: Returns the conversation history and evidence records.

---

## 🔬 Related Neural Model Documentation

For detailed mathematical formulations of the ConvNeXt spatial backbone, 3-kernel SRM frequency filters, cross-attention fusion gates, and calibrated temperature scaling, see **[`MODEL.md`](MODEL.md)**.


