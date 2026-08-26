"""ค่าตั้งต้นทั้งหมดของระบบ อ่านจาก environment variables (ดู .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Settings", "get_settings", "load_dotenv"]


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """โหลดไฟล์ .env แบบง่าย ๆ โดยไม่ต้องพึ่ง python-dotenv.

    ค่าที่ตั้งไว้ใน environment อยู่แล้วจะไม่ถูกเขียนทับ
    """
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _s(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # ── สมอง ────────────────────────────────────────────────────────────
    model: str = field(default_factory=lambda: _s("THAIVOICE_MODEL", "claude-opus-5"))
    effort: str = field(default_factory=lambda: _s("THAIVOICE_EFFORT", "low"))
    max_tokens: int = field(default_factory=lambda: _i("THAIVOICE_MAX_TOKENS", 1024))
    assistant_name: str = field(
        default_factory=lambda: _s("THAIVOICE_ASSISTANT_NAME", "ใจ")
    )
    memory_model: str = field(
        default_factory=lambda: _s("THAIVOICE_MEMORY_MODEL", "")
        or _s("THAIVOICE_MODEL", "claude-opus-5")
    )

    # ── ความจำ ──────────────────────────────────────────────────────────
    db_path: Path = field(
        default_factory=lambda: Path(_s("THAIVOICE_DB", "data/memory.db"))
    )
    history_turns: int = field(default_factory=lambda: _i("THAIVOICE_HISTORY_TURNS", 16))
    summarize_every: int = field(default_factory=lambda: _i("THAIVOICE_SUMMARIZE_EVERY", 20))

    # ── เสียงเข้า ───────────────────────────────────────────────────────
    stt_backend: str = field(default_factory=lambda: _s("THAIVOICE_STT", "faster-whisper"))
    whisper_model: str = field(default_factory=lambda: _s("THAIVOICE_WHISPER_MODEL", "large-v3"))
    whisper_device: str = field(default_factory=lambda: _s("THAIVOICE_WHISPER_DEVICE", "auto"))
    whisper_compute: str = field(default_factory=lambda: _s("THAIVOICE_WHISPER_COMPUTE", "default"))

    # ── เสียงออก ────────────────────────────────────────────────────────
    tts_backend: str = field(default_factory=lambda: _s("THAIVOICE_TTS", "edge"))
    tts_voice_female: str = field(
        default_factory=lambda: _s("THAIVOICE_TTS_VOICE_FEMALE", "th-TH-PremwadeeNeural")
    )
    tts_voice_male: str = field(
        default_factory=lambda: _s("THAIVOICE_TTS_VOICE_MALE", "th-TH-NiwatNeural")
    )
    tts_rate: str = field(default_factory=lambda: _s("THAIVOICE_TTS_RATE", "+8%"))

    # ── จำลายเสียง ──────────────────────────────────────────────────────
    speaker_backend: str = field(
        default_factory=lambda: _s("THAIVOICE_SPEAKER_BACKEND", "resemblyzer")
    )
    speaker_threshold: float = field(
        default_factory=lambda: _f("THAIVOICE_SPEAKER_THRESHOLD", 0.75)
    )

    # ── ไมโครโฟน ────────────────────────────────────────────────────────
    sample_rate: int = 16000
    vad_aggressiveness: int = field(
        default_factory=lambda: _i("THAIVOICE_VAD_AGGRESSIVENESS", 2)
    )
    silence_ms: int = field(default_factory=lambda: _i("THAIVOICE_SILENCE_MS", 800))
    max_utterance_s: int = field(default_factory=lambda: _i("THAIVOICE_MAX_UTTERANCE_S", 30))

    # ── เซิร์ฟเวอร์ ─────────────────────────────────────────────────────
    host: str = field(default_factory=lambda: _s("THAIVOICE_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _i("THAIVOICE_PORT", 8080))
    cors_origins: str = field(default_factory=lambda: _s("THAIVOICE_CORS_ORIGINS", ""))

    @property
    def cors_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if not raw:
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


_cached: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """คืนค่า Settings (แคชไว้) — เรียก reload=True เมื่อเปลี่ยน env กลางคัน."""
    global _cached
    if _cached is None or reload:
        load_dotenv()
        _cached = Settings()
    return _cached
