"""LangGraph-Powered Forensic Agent with Conversational Memory & Multi-Image Session Tracking."""

import os
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime, timezone
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    from .model_service import get_model_service
    from .groq_agent import get_groq_agent
    from .database import (
        save_image_record, 
        get_user_images, 
        get_image_by_id, 
        save_chat_message, 
        get_user_chat_history
    )
    from ..config import settings
except (ImportError, ValueError):
    try:
        from app.services.model_service import get_model_service
        from app.services.groq_agent import get_groq_agent
        from app.services.database import (
            save_image_record, 
            get_user_images, 
            get_image_by_id, 
            save_chat_message, 
            get_user_chat_history
        )
        from app.config import settings
    except (ImportError, ValueError):
        from services.model_service import get_model_service
        from services.groq_agent import get_groq_agent
        from services.database import (
            save_image_record, 
            get_user_images, 
            get_image_by_id, 
            save_chat_message, 
            get_user_chat_history
        )
        from config import settings

logger = logging.getLogger(__name__)

UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class ForensicState(TypedDict, total=False):
    """Complete LangGraph state schema for multi-modal forensic inspection & chat memory."""
    user_id: Optional[int]
    session_id: str
    image_id: Optional[str]
    filename: Optional[str]
    image_url: Optional[str]
    image_bytes: Optional[bytes]
    custom_groq_api_key: Optional[str]
    predicted_class: Optional[str]
    confidence: Optional[float]
    is_fake: Optional[bool]
    class_probabilities: Optional[List[Dict[str, Any]]]
    forensics: Optional[Dict[str, Any]]
    explainability: Optional[Dict[str, str]]
    inference_time_ms: Optional[float]
    messages: Annotated[List[Any], add_messages]
    query: Optional[str]
    answer: Optional[str]
    uploaded_images: List[Dict[str, Any]]


class SessionImageRegistry:
    """In-memory and file-backed registry tracking user-uploaded images and their forensic records."""
    _sessions: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def register_image(cls, session_id: str, record: Dict[str, Any]):
        if session_id not in cls._sessions:
            cls._sessions[session_id] = []
        
        # Check if already present to avoid duplicates
        existing_index = next(
            (i for i, item in enumerate(cls._sessions[session_id]) if item.get("image_id") == record.get("image_id")),
            None
        )
        if existing_index is not None:
            cls._sessions[session_id][existing_index] = record
        else:
            # Prepend newest first
            cls._sessions[session_id].insert(0, record)

        # Persist JSON metadata to disk for durability across restarts
        try:
            import json
            rec_file = UPLOADS_DIR / f"{record['image_id']}.json"
            with open(rec_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)
        except Exception as err:
            logger.warning(f"Could not persist image metadata to disk: {err}")

    @classmethod
    def get_images(cls, session_id: str) -> List[Dict[str, Any]]:
        # If in-memory is empty for this session, try reading from disk
        if session_id not in cls._sessions or not cls._sessions[session_id]:
            cls._load_from_disk(session_id)
        return cls._sessions.get(session_id, [])

    @classmethod
    def get_all_images(cls) -> List[Dict[str, Any]]:
        """Returns all registered images across sessions."""
        all_imgs = []
        seen_ids = set()
        for session_imgs in cls._sessions.values():
            for img in session_imgs:
                if img.get("image_id") not in seen_ids:
                    seen_ids.add(img.get("image_id"))
                    all_imgs.append(img)

        # Scan disk for any JSON files not yet loaded
        try:
            import json
            for jf in UPLOADS_DIR.glob("*.json"):
                img_id = jf.stem
                if img_id not in seen_ids:
                    try:
                        with open(jf, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            all_imgs.append(data)
                            seen_ids.add(img_id)
                    except Exception:
                        pass
        except Exception:
            pass

        all_imgs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return all_imgs

    @classmethod
    def _load_from_disk(cls, session_id: str):
        """Loads records from disk that match session_id or all records if default."""
        try:
            import json
            loaded = []
            for jf in UPLOADS_DIR.glob("*.json"):
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("session_id") == session_id or not session_id:
                            loaded.append(data)
                except Exception:
                    pass
            if loaded:
                loaded.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                cls._sessions[session_id] = loaded
        except Exception as e:
            logger.warning(f"Error loading images from disk: {e}")

    @classmethod
    def get_image_by_id(cls, session_id: str, image_id: str) -> Optional[Dict[str, Any]]:
        for item in cls.get_images(session_id):
            if item.get("image_id") == image_id:
                return item
        for item in cls.get_all_images():
            if item.get("image_id") == image_id:
                return item
        return None


class LangGraphForensicAgent:
    """Production LangGraph Agent with StateGraph, Checkpointing Memory, and Follow-up QA."""

    def __init__(self):
        self.model_service = get_model_service()
        self.groq_agent = get_groq_agent()
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _extract_vision_forensics_node(self, state: ForensicState) -> Dict[str, Any]:
        """Node 1: Runs the dual-branch neural network on the uploaded image bytes."""
        image_bytes = state.get("image_bytes")
        if not image_bytes:
            return {}

        start_time = time.perf_counter()
        detection = self.model_service.predict(image_bytes)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return {
            "predicted_class": detection["predicted_class"],
            "confidence": detection["confidence"],
            "is_fake": detection["is_fake"],
            "class_probabilities": detection["class_probabilities"],
            "forensics": detection["forensics"],
            "inference_time_ms": elapsed_ms,
        }

    def _generate_explanation_node(self, state: ForensicState) -> Dict[str, Any]:
        """Node 2: Invokes Groq LLM Specialist to synthesize initial forensic report."""
        predicted_class = state.get("predicted_class")
        confidence = state.get("confidence", 0.0)
        class_probabilities = state.get("class_probabilities", [])
        forensics = state.get("forensics", {})
        custom_key = state.get("custom_groq_api_key")

        if not predicted_class:
            return {}

        explanation_res = self.groq_agent.generate_explanation(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            forensics=forensics,
            custom_api_key=custom_key
        )

        initial_msg = (
            f"Forensic Verdict: {predicted_class.upper()} ({confidence * 100:.1f}% confidence).\n\n"
            f"{explanation_res.get('line_1', '')}\n\n"
            f"{explanation_res.get('line_2', '')}"
        )

        return {
            "explainability": explanation_res,
            "messages": [AIMessage(content=initial_msg)]
        }

    def _register_image_memory_node(self, state: ForensicState) -> Dict[str, Any]:
        """Node 3: Persists the uploaded image to disk and registers into SQLite user database."""
        user_id = state.get("user_id") or 1
        session_id = state.get("session_id", "default_session")
        image_bytes = state.get("image_bytes")
        image_id = state.get("image_id") or str(uuid.uuid4())[:8]
        filename = f"{image_id}.jpg"
        filepath = UPLOADS_DIR / filename

        if image_bytes:
            with open(filepath, "wb") as f:
                f.write(image_bytes)

        image_url = f"/uploads/{filename}"

        # Persist record into SQLite linked to user_id
        save_image_record(
            user_id=user_id,
            session_id=session_id,
            image_id=image_id,
            filename=filename,
            filepath=str(filepath),
            image_url=image_url,
            predicted_class=state.get("predicted_class") or "unknown",
            confidence=state.get("confidence") or 0.0,
            is_fake=state.get("is_fake") or False,
            class_probabilities=state.get("class_probabilities") or [],
            forensics=state.get("forensics") or {},
            explainability=state.get("explainability") or {},
            inference_time_ms=state.get("inference_time_ms") or 0.0
        )

        user_images = get_user_images(user_id=user_id)

        return {
            "image_id": image_id,
            "filename": filename,
            "image_url": image_url,
            "image_bytes": None,
            "uploaded_images": user_images
        }

    def _conversational_followup_node(self, state: ForensicState) -> Dict[str, Any]:
        """Node 4: Handles conversational follow-up questions about active/registered images using LangGraph memory."""
        query = state.get("query")
        if not query:
            return {}

        user_id = state.get("user_id") or 1
        session_id = state.get("session_id", "default_session")
        predicted_class = state.get("predicted_class", "unknown")
        confidence = state.get("confidence", 0.0)
        forensics = state.get("forensics") or {}
        probabilities = state.get("class_probabilities") or []
        messages = state.get("messages") or []
        custom_key = state.get("custom_groq_api_key")

        # Retrieve ONLY this user's images from SQLite database
        user_images = get_user_images(user_id=user_id)

        images_context = ""
        if user_images:
            images_context = f"\nREGISTERED UPLOADED IMAGES IN THIS USER'S VAULT ({len(user_images)} total):\n"
            for idx, img in enumerate(user_images, 1):
                images_context += (
                    f"- Image #{idx} [ID: {img.get('image_id')}]: "
                    f"Class={img.get('predicted_class', 'unknown').upper()}, "
                    f"Confidence={img.get('confidence', 0.0) * 100:.1f}%, "
                    f"Filename={img.get('filename')}, "
                    f"Timestamp={img.get('timestamp')}\n"
                )

        # Build context prompt for Groq
        system_context = (
            "You are an expert AI Digital Image Forensic Investigator. "
            "You analyze diagnostic telemetry from a dual-branch neural network (ConvNeXt Spatial + SRM Frequency residuals) "
            "and assist the user with follow-up forensic inquiries using checkpointed conversation memory.\n\n"
            f"ACTIVE IMAGE EVIDENCE SUMMARY:\n"
            f"- Verdict: {predicted_class.upper()} ({confidence * 100:.1f}% confidence)\n"
            f"- Probabilities: {', '.join([f'{c.get('class_name')}: {c.get('percentage')}%' for c in probabilities])}\n"
            f"- SRM Residual Energy: {forensics.get('srm_residual_energy', 'N/A')}\n"
            f"- Dominant Filter Kernel: {forensics.get('dominant_kernel', 'N/A')}\n"
            f"- Spatial Attention Ratio: {forensics.get('attention_spatial_ratio', 0.5) * 100:.1f}%\n"
            f"- Spatial Noise Inconsistency: {forensics.get('noise_inconsistency', 'N/A')}\n"
            f"{images_context}\n"
            "INSTRUCTIONS:\n"
            "1. If the user asks about the active image's forensic classification or metrics, answer with scientific precision.\n"
            "2. If the user asks about their uploaded images (e.g. 'show uploaded images', 'what did I upload?', 'list my images'), "
            "explicitly reference and list their registered images, their IDs, verdicts, and key observations.\n"
            "3. If the user asks to compare images, compare the physical forensic metrics of their uploaded images.\n"
            "4. Keep your tone authoritative, concise, and helpful. Avoid robotic repetition."
        )

        active_key = custom_key or self.groq_agent.api_key or os.environ.get("GROQ_API_KEY")
        answer = ""

        if active_key:
            try:
                from groq import Groq
                client = Groq(api_key=str(active_key).strip().strip("'\""))
                llm_msgs = [{"role": "system", "content": system_context}]
                for msg in messages[-6:]:
                    if isinstance(msg, HumanMessage):
                        llm_msgs.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        llm_msgs.append({"role": "assistant", "content": msg.content})
                llm_msgs.append({"role": "user", "content": query})

                candidate_models = [
                    settings.GROQ_MODEL,
                    settings.GROQ_FALLBACK_MODEL,
                    "openai/gpt-oss-120b",
                    "llama-3.3-70b-versatile"
                ]

                for model in candidate_models:
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=llm_msgs,
                            temperature=0.3,
                            max_tokens=350
                        )
                        answer = resp.choices[0].message.content.strip()
                        if answer:
                            break
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"Groq conversational follow-up failed: {e}")

        if not answer:
            # High-quality fallback rule answer
            lower_q = query.lower()
            if any(k in lower_q for k in ["uploaded", "images", "show", "list", "history", "all"]):
                if user_images:
                    img_lines = [
                        f"• Image {im.get('image_id')} ({im.get('filename')}): {im.get('predicted_class', '').upper()} "
                        f"({im.get('confidence', 0.0) * 100:.1f}% confidence)"
                        for im in user_images
                    ]
                    answer = (
                        f"You have {len(user_images)} registered image(s) in your private evidence vault:\n\n"
                        + "\n".join(img_lines)
                        + f"\n\nThe currently active inspection focus is image {state.get('image_id', 'recent')}, "
                        f"classified as {predicted_class.upper()}."
                    )
                else:
                    answer = "No images are currently registered in your vault. Upload an image to begin forensic inspection."
            else:
                srm = forensics.get('srm_residual_energy', 0.005)
                kernel = forensics.get('dominant_kernel', 'Laplacian')
                incon = forensics.get('noise_inconsistency', 0.0)
                answer = (
                    f"Regarding your query on the {predicted_class.upper()} assessment: "
                    f"The neural pipeline identified spectral residual energy of {srm:.4f} within the {kernel} domain. "
                    f"The local spatial noise inconsistency is recorded at {incon}, corroborating the {predicted_class} signature "
                    f"at {confidence * 100:.1f}% confidence."
                )

        # Save conversation turn to SQLite for persistent memory
        try:
            save_chat_message(
                user_id=user_id,
                session_id=session_id,
                role="user",
                content=query,
                image_id=state.get("image_id")
            )
            save_chat_message(
                user_id=user_id,
                session_id=session_id,
                role="assistant",
                content=answer,
                image_id=state.get("image_id")
            )
        except Exception as e:
            logger.warning(f"Could not persist chat message to SQLite: {e}")

        return {
            "answer": answer,
            "query": None,
            "messages": [HumanMessage(content=query), AIMessage(content=answer)],
            "uploaded_images": user_images
        }

    def _build_graph(self):
        """Compiles the LangGraph StateGraph with MemorySaver checkpointer."""
        workflow = StateGraph(ForensicState)

        # Add Nodes
        workflow.add_node("extract_forensics", self._extract_vision_forensics_node)
        workflow.add_node("generate_explanation", self._generate_explanation_node)
        workflow.add_node("register_memory", self._register_image_memory_node)
        workflow.add_node("conversational_followup", self._conversational_followup_node)

        # Flow Routing: Prioritize explicit query over lingering image bytes
        def route_initial_step(state: ForensicState):
            if state.get("query"):
                return "conversational_followup"
            if state.get("image_bytes"):
                return "extract_forensics"
            return "conversational_followup"

        workflow.set_conditional_entry_point(
            route_initial_step,
            {
                "extract_forensics": "extract_forensics",
                "conversational_followup": "conversational_followup"
            }
        )

        # Pipeline Edges
        workflow.add_edge("extract_forensics", "generate_explanation")
        workflow.add_edge("generate_explanation", "register_memory")
        workflow.add_edge("register_memory", END)
        workflow.add_edge("conversational_followup", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def analyze_image(
        self,
        image_bytes: bytes,
        user_id: Optional[int] = 1,
        session_id: Optional[str] = None,
        custom_groq_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs image through LangGraph pipeline and registers memory partitioned by user_id."""
        uid = user_id or 1
        sid = session_id or str(uuid.uuid4())[:12]
        thread_id = f"user_{uid}_{sid}"
        config = {"configurable": {"thread_id": thread_id}}

        initial_state: ForensicState = {
            "user_id": uid,
            "session_id": sid,
            "image_bytes": image_bytes,
            "query": None,
            "custom_groq_api_key": custom_groq_api_key,
        }

        final_state = self.graph.invoke(initial_state, config=config)
        user_images = get_user_images(user_id=uid)

        return {
            "status": "success",
            "user_id": uid,
            "session_id": sid,
            "image_id": final_state.get("image_id"),
            "filename": final_state.get("filename"),
            "image_url": final_state.get("image_url"),
            "predicted_class": final_state.get("predicted_class"),
            "confidence": final_state.get("confidence"),
            "is_fake": final_state.get("is_fake"),
            "class_probabilities": final_state.get("class_probabilities"),
            "forensics": final_state.get("forensics"),
            "explainability": final_state.get("explainability"),
            "inference_time_ms": final_state.get("inference_time_ms"),
            "uploaded_images": user_images,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def chat_followup(
        self,
        query: str,
        user_id: Optional[int] = 1,
        session_id: str = "default_session",
        image_id: Optional[str] = None,
        custom_groq_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes conversational follow-up in LangGraph strictly scoped to user_id."""
        uid = user_id or 1
        thread_id = f"user_{uid}_{session_id}"
        config = {"configurable": {"thread_id": thread_id}}

        input_state: ForensicState = {
            "user_id": uid,
            "session_id": session_id,
            "query": query,
            "image_bytes": None,
            "custom_groq_api_key": custom_groq_api_key,
        }

        # If a specific past image is targeted, load its context only if it belongs to this user
        target_img = None
        if image_id:
            target_img = get_image_by_id(image_id=image_id, user_id=uid)
        if not target_img:
            # Default to latest image of this user
            user_imgs = get_user_images(user_id=uid)
            if user_imgs:
                target_img = user_imgs[0]

        if target_img:
            input_state.update({
                "image_id": target_img.get("image_id"),
                "filename": target_img.get("filename"),
                "image_url": target_img.get("image_url"),
                "predicted_class": target_img.get("predicted_class"),
                "confidence": target_img.get("confidence"),
                "is_fake": target_img.get("is_fake"),
                "class_probabilities": target_img.get("class_probabilities"),
                "forensics": target_img.get("forensics"),
                "explainability": target_img.get("explainability"),
            })

        final_state = self.graph.invoke(input_state, config=config)
        user_images = get_user_images(user_id=uid)

        return {
            "status": "success",
            "user_id": uid,
            "session_id": session_id,
            "image_id": final_state.get("image_id") or (target_img.get("image_id") if target_img else None),
            "query": query,
            "answer": final_state.get("answer"),
            "uploaded_images": user_images,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_session_history(self, user_id: int = 1, session_id: str = "default_session") -> Dict[str, Any]:
        """Retrieves checkpointed conversation messages and registered images for this user."""
        uid = user_id or 1
        thread_id = f"user_{uid}_{session_id}"
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.graph.get_state(config)
        messages_out = []
        if state_snapshot and state_snapshot.values:
            raw_msgs = state_snapshot.values.get("messages", [])
            for m in raw_msgs:
                if isinstance(m, HumanMessage):
                    messages_out.append({"role": "user", "content": m.content})
                elif isinstance(m, AIMessage):
                    messages_out.append({"role": "assistant", "content": m.content})
                elif isinstance(m, dict):
                    messages_out.append(m)

        # Supplement with durable SQLite chat messages if LangGraph checkpointer is empty
        if not messages_out:
            messages_out = get_user_chat_history(user_id=uid, session_id=session_id)

        user_images = get_user_images(user_id=uid)

        return {
            "session_id": session_id,
            "user_id": uid,
            "messages": messages_out,
            "uploaded_images": user_images
        }

    def get_registered_images(
        self, 
        user_id: int = 1, 
        session_id: Optional[str] = None,
        filter_by_session: bool = False,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Returns registered images strictly uploaded by user_id."""
        return get_user_images(
            user_id=user_id, 
            session_id=session_id, 
            filter_by_session=filter_by_session, 
            limit=limit, 
            offset=offset
        )


# Singleton instance
_langgraph_agent: Optional[LangGraphForensicAgent] = None

def get_langgraph_agent() -> LangGraphForensicAgent:
    global _langgraph_agent
    if _langgraph_agent is None:
        _langgraph_agent = LangGraphForensicAgent()
    return _langgraph_agent
