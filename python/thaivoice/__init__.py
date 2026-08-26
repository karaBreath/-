"""thaivoice — ระบบสนทนาด้วยเสียงภาษาไทยที่จดจำผู้สนทนาได้

ใช้งานเร็วสุด::

    from thaivoice import create_session

    session = create_session()
    result = session.exchange("สวัสดีครับ ผมชื่อเดช")
    print(result.reply)
"""

from __future__ import annotations

from .audio import AudioPlayer, Microphone, audio_available
from .brain import BrainEvent, ThaiBrain
from .config import Settings, get_settings, load_dotenv
from .extraction import MemoryExtractor
from .memory import Fact, MemoryStore, Speaker, Turn
from .session import ConversationSession, ExchangeResult, SessionEvent
from .speaker import SpeakerIdentifier, load_embedder
from .stt import Transcription, load_stt, pcm_to_wav, wav_to_pcm
from .tts import Speech, load_tts

__version__ = "0.1.0"

__all__ = [
    "AudioPlayer",
    "BrainEvent",
    "ConversationSession",
    "ExchangeResult",
    "Fact",
    "MemoryExtractor",
    "MemoryStore",
    "Microphone",
    "SessionEvent",
    "Settings",
    "Speaker",
    "SpeakerIdentifier",
    "Speech",
    "ThaiBrain",
    "Transcription",
    "Turn",
    "audio_available",
    "create_session",
    "get_settings",
    "load_dotenv",
    "load_embedder",
    "load_stt",
    "load_tts",
    "pcm_to_wav",
    "wav_to_pcm",
]


def create_session(
    settings: Settings | None = None,
    *,
    store: MemoryStore | None = None,
    client=None,
    session_id: str | None = None,
    with_tts: bool = True,
    with_memory_extraction: bool = True,
) -> ConversationSession:
    """ประกอบระบบทั้งหมดให้พร้อมใช้ในบรรทัดเดียว

    ทุกชิ้นส่วนที่เป็น optional (จำเสียง, สังเคราะห์เสียง, สกัดความจำ) จะถูกข้าม
    อย่างเงียบ ๆ ถ้าติดตั้งไม่ครบ ระบบยังคุยได้เสมอ

    ส่ง ``client`` เข้ามาได้ถ้าต้องการใช้ไคลเอนต์ Anthropic ที่ตั้งค่าเอง
    (หรือของปลอมในเทสต์)
    """
    settings = settings or get_settings()
    store = store or MemoryStore(settings.db_path)
    brain = ThaiBrain(store, client=client, settings=settings)
    identifier = SpeakerIdentifier(
        store,
        load_embedder(settings.speaker_backend),
        threshold=settings.speaker_threshold,
    )
    extractor = (
        MemoryExtractor(store, brain.client, settings) if with_memory_extraction else None
    )
    return ConversationSession(
        store=store,
        brain=brain,
        identifier=identifier,
        extractor=extractor,
        tts=load_tts(settings) if with_tts else None,
        settings=settings,
        session_id=session_id,
    )
