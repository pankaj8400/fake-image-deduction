"""SQLite Database Service for ForensIQ: User Authentication, Session Memory & Multi-Tenant Image Vault."""

import os
import json
import sqlite3
import hashlib
import secrets
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "forensiq.db"


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database with row factory."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db():
    """Initializes tables and indices for users, sessions, images, and chat history."""
    conn = get_db_connection()
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_fake INTEGER NOT NULL,
                    class_probabilities_json TEXT NOT NULL,
                    forensics_json TEXT NOT NULL,
                    explainability_json TEXT NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    image_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_images_user ON images(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_images_session ON images(user_id, session_id);
                CREATE INDEX IF NOT EXISTS idx_chat_user_session ON chat_messages(user_id, session_id, created_at ASC);
            """)
        logger.info(f"ForensIQ SQLite database successfully initialized at {DB_PATH}")
    finally:
        conn.close()


def _hash_password(password: str, salt_hex: Optional[str] = None) -> Tuple[str, str]:
    """Hashes password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
    if not salt_hex:
        salt = secrets.token_bytes(16)
        salt_hex = salt.hex()
    else:
        salt = bytes.fromhex(salt_hex)

    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return derived.hex(), salt_hex


# --- User Management & Authentication ---

def register_user(username: str, password: str) -> Dict[str, Any]:
    """Registers a new user into SQLite with salted PBKDF2 hash."""
    uname = username.strip()
    if len(uname) < 3:
        raise ValueError("Username must be at least 3 characters long.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")

    pwd_hash, salt_hex = _hash_password(password)
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (uname, pwd_hash, salt_hex, now_iso)
            )
            user_id = cursor.lastrowid
            return {
                "id": user_id,
                "username": uname,
                "created_at": now_iso
            }
    except sqlite3.IntegrityError:
        raise ValueError(f"Username '{uname}' is already registered. Please choose another username or sign in.")
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Validates user credentials against SQLite records."""
    uname = username.strip()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash, salt, created_at FROM users WHERE username = ?", (uname,))
        row = cursor.fetchone()
        if not row:
            return None

        user_id, db_username, db_hash, db_salt, created_at = row
        calc_hash, _ = _hash_password(password, db_salt)
        if secrets.compare_digest(calc_hash, db_hash):
            return {
                "id": user_id,
                "username": db_username,
                "created_at": created_at
            }
        return None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves user profile by ID."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieves user profile by username."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, created_at FROM users WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


# --- Image Vault & Evidence Management (User-Isolated) ---

def save_image_record(
    user_id: int,
    session_id: str,
    image_id: str,
    filename: str,
    filepath: str,
    image_url: str,
    predicted_class: str,
    confidence: float,
    is_fake: bool,
    class_probabilities: list,
    forensics: dict,
    explainability: dict,
    inference_time_ms: float
) -> Dict[str, Any]:
    """Persists an image detection record in SQLite linked strictly to the user."""
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    try:
        with conn:
            # Ensure session is registered
            conn.execute(
                """
                INSERT INTO sessions (session_id, user_id, created_at, last_active)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET last_active = excluded.last_active
                """,
                (session_id, user_id, now_iso, now_iso)
            )

            # Insert or replace image record
            conn.execute(
                """
                INSERT OR REPLACE INTO images (
                    image_id, user_id, session_id, filename, filepath, image_url,
                    predicted_class, confidence, is_fake, class_probabilities_json,
                    forensics_json, explainability_json, inference_time_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    user_id,
                    session_id,
                    filename,
                    filepath,
                    image_url,
                    predicted_class,
                    float(confidence),
                    1 if is_fake else 0,
                    json.dumps(class_probabilities),
                    json.dumps(forensics),
                    json.dumps(explainability),
                    float(inference_time_ms),
                    now_iso
                )
            )

        return {
            "image_id": image_id,
            "user_id": user_id,
            "session_id": session_id,
            "filename": filename,
            "image_url": image_url,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_fake": is_fake,
            "class_probabilities": class_probabilities,
            "forensics": forensics,
            "explainability": explainability,
            "inference_time_ms": inference_time_ms,
            "timestamp": now_iso
        }
    finally:
        conn.close()


def get_user_images_count(user_id: int, session_id: Optional[str] = None, filter_by_session: bool = False) -> int:
    """Returns total count of images uploaded by user_id."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if filter_by_session and session_id:
            cursor.execute("SELECT COUNT(*) FROM images WHERE user_id = ? AND session_id = ?", (user_id, session_id))
        else:
            cursor.execute("SELECT COUNT(*) FROM images WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def get_user_images(
    user_id: int, 
    session_id: Optional[str] = None,
    filter_by_session: bool = False,
    limit: Optional[int] = None,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Retrieves all images uploaded by user_id across all sessions by default. Strict isolation: NEVER shows other users' images."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM images WHERE user_id = ?"
        params: List[Any] = [user_id]

        if filter_by_session and session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        images = []
        for r in rows:
            images.append({
                "image_id": r["image_id"],
                "user_id": r["user_id"],
                "session_id": r["session_id"],
                "filename": r["filename"],
                "image_url": r["image_url"],
                "predicted_class": r["predicted_class"],
                "confidence": r["confidence"],
                "is_fake": bool(r["is_fake"]),
                "class_probabilities": json.loads(r["class_probabilities_json"]),
                "forensics": json.loads(r["forensics_json"]),
                "explainability": json.loads(r["explainability_json"]),
                "inference_time_ms": r["inference_time_ms"],
                "timestamp": r["created_at"]
            })
        return images
    finally:
        conn.close()


def get_image_by_id(image_id: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an image record by ID, optionally validating user_id ownership."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if user_id is not None:
            cursor.execute("SELECT * FROM images WHERE image_id = ? AND user_id = ?", (image_id, user_id))
        else:
            cursor.execute("SELECT * FROM images WHERE image_id = ?", (image_id,))

        r = cursor.fetchone()
        if not r:
            return None

        return {
            "image_id": r["image_id"],
            "user_id": r["user_id"],
            "session_id": r["session_id"],
            "filename": r["filename"],
            "image_url": r["image_url"],
            "predicted_class": r["predicted_class"],
            "confidence": r["confidence"],
            "is_fake": bool(r["is_fake"]),
            "class_probabilities": json.loads(r["class_probabilities_json"]),
            "forensics": json.loads(r["forensics_json"]),
            "explainability": json.loads(r["explainability_json"]),
            "inference_time_ms": r["inference_time_ms"],
            "timestamp": r["created_at"]
        }
    finally:
        conn.close()


# --- Chat History & LangGraph Memory Persistence (User-Isolated) ---

def save_chat_message(
    user_id: int,
    session_id: str,
    role: str,
    content: str,
    image_id: Optional[str] = None
) -> Dict[str, Any]:
    """Saves a conversation message in SQLite linked strictly to the user and session."""
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_messages (user_id, session_id, image_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, session_id, image_id, role, content, now_iso)
            )
            return {
                "id": cursor.lastrowid,
                "user_id": user_id,
                "session_id": session_id,
                "image_id": image_id,
                "role": role,
                "content": content,
                "created_at": now_iso
            }
    finally:
        conn.close()


def get_user_chat_history(user_id: int, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves conversation history for user_id and session_id from SQLite."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT role, content, image_id, created_at 
            FROM chat_messages 
            WHERE user_id = ? AND session_id = ? 
            ORDER BY created_at ASC 
            LIMIT ?
            """,
            (user_id, session_id, limit)
        )
        rows = cursor.fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "image_id": r["image_id"],
                "timestamp": r["created_at"]
            }
            for r in rows
        ]
    finally:
        conn.close()


# Initialize database automatically upon module import
init_db()
