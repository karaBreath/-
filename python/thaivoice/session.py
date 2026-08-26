"""ตัวประสานงานหนึ่งบทสนทนา — รู้ว่าใครพูด แล้วคิด พูดตอบ และจำ

ลำดับการทำงานของหนึ่งเทิร์น:

1. **ระบุตัวผู้พูด** จากลายเสียง ถ้าไม่รู้จักก็ดูว่าเขาบอกชื่อมาไหม
2. **ตรวจคำสั่งเกี่ยวกับความจำ** เช่น "ลืมทุกอย่างเกี่ยวกับฉัน" — จัดการตรง ๆ
   ไม่ผ่านโมเดล เพราะการลบข้อมูลส่วนบุคคลต้องเชื่อถือได้ 100 เปอร์เซ็นต์
3. **สตรีมคำตอบ** จาก Claude พร้อมความจำของคนคนนั้น
4. **พูดออกมาทีละประโยค** ทันทีที่ประโยคจบ ไม่รอคำตอบครบก้อน
5. **บันทึกบทสนทนา** และสั่งสกัดความจำเบื้องหลัง
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .brain import BrainEvent, ThaiBrain
from .config import Settings, get_settings
from .extraction import MemoryExtractor
from .memory import MemoryStore, Speaker
from .speaker import Identification, SpeakerIdentifier
from .thai_text import detect_particle, particle_for_gender
from .tts import Speech, TextToSpeech

log = logging.getLogger("thaivoice.session")

__all__ = [
    "ConversationSession",
    "SessionEvent",
    "ExchangeResult",
    "detect_forget_all",
]

# คำสั่งลบความจำ — จัดการด้วยกฎ ไม่ปล่อยให้โมเดลตัดสิน
_FORGET_ALL = re.compile(
    r"(ลืม(ทุกอย่าง|ทุกเรื่อง|ฉัน|ผม|หนู|เรา)|ลบ(ความจำ|ข้อมูล|ประวัติ)"
    r"|เคลียร์(ความจำ|ข้อมูล)|forget everything|delete my data)"
)

_CONFIRM_FORGET = "ลบความจำเกี่ยวกับคุณทั้งหมดแล้ว{particle} เริ่มรู้จักกันใหม่ได้เลย{particle}"


def detect_forget_all(text: str) -> bool:
    """ตรวจว่าผู้ใช้สั่งให้ลืมทุกอย่างหรือไม่

    >>> detect_forget_all("ลืมทุกอย่างเกี่ยวกับฉันหน่อย")
    True
    >>> detect_forget_all("วันนี้กินอะไรดี")
    False
    """
    return bool(_FORGET_ALL.search(text or ""))


@dataclass
class SessionEvent:
    """เหตุการณ์ระหว่างหนึ่งเทิร์น

    * ``speaker``  — ระบุตัวผู้พูดได้แล้ว
    * ``delta``    — ข้อความที่โมเดลกำลังพิมพ์ (สำหรับแสดงบนจอ)
    * ``chunk``    — ประโยคที่พร้อมพูด พร้อมเสียงใน ``speech`` (ถ้ามี)
    * ``done``     — จบเทิร์น ``text`` คือคำตอบเต็ม
    """

    type: str
    text: str = ""
    speaker: Speaker | None = None
    speech: Speech | None = None
    identification: Identification | None = None


@dataclass
class ExchangeResult:
    transcript: str
    reply: str
    speaker: Speaker | None = None
    identification: Identification | None = None
    chunks: list[str] = field(default_factory=list)


class ConversationSession:
    """หนึ่งบทสนทนา — ใช้ซ้ำได้หลายเทิร์นและหลายคนสลับกันพูด"""

    def __init__(
        self,
        store: MemoryStore,
        brain: ThaiBrain,
        identifier: SpeakerIdentifier,
        extractor: MemoryExtractor | None = None,
        tts: TextToSpeech | None = None,
        settings: Settings | None = None,
        session_id: str | None = None,
    ) -> None:
        self.store = store
        self.brain = brain
        self.identifier = identifier
        self.extractor = extractor
        self.tts = tts
        self.settings = settings or get_settings()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.current_speaker: Speaker | None = None

    # ── ระบุตัวผู้พูด ───────────────────────────────────────────────────
    def identify(self, transcript: str, pcm: bytes | None) -> Identification:
        """หาว่าใครพูดประโยคนี้ และอัปเดตคำลงท้ายที่ควรใช้กับเขา"""
        ident = self.identifier.resolve(
            pcm, self.settings.sample_rate, transcript
        )

        if ident.speaker is None and self.current_speaker is not None and pcm is None:
            # ไม่มีเสียงให้เทียบ (เช่นพิมพ์เข้ามา) — ถือว่ายังเป็นคนเดิมในบทสนทนานี้
            ident = Identification(
                speaker=self.current_speaker, method="fallback", score=0.0
            )

        speaker = ident.speaker
        if speaker is not None:
            # ผู้ใช้เผยเพศจากคำลงท้ายเมื่อไหร่ ก็ปรับคำลงท้ายของบอทตามทันที
            gender = detect_particle(transcript)
            if gender and not speaker.gender:
                updated = self.store.update_speaker(
                    speaker.id, gender=gender, particle=particle_for_gender(gender)
                )
                speaker = updated or speaker
                ident.speaker = speaker
            self.current_speaker = speaker

            # เจอเสียงตรงกับคนเดิม -> เสริมลายเสียงให้แม่นขึ้นเรื่อย ๆ
            if pcm and ident.method == "voice" and ident.confident:
                self.identifier.enroll(speaker, pcm, self.settings.sample_rate)
        return ident

    def register_speaker(
        self, name: str, pcm: bytes | None = None, gender: str | None = None
    ) -> Speaker:
        """ลงทะเบียนคนใหม่ด้วยตนเอง แล้วผูกลายเสียงถ้ามีเสียงตัวอย่าง"""
        existing = self.store.find_speaker_by_name(name)
        speaker = existing or self.store.create_speaker(
            name, gender=gender, particle=particle_for_gender(gender)
        )
        if pcm:
            self.identifier.enroll(speaker, pcm, self.settings.sample_rate)
        self.current_speaker = speaker
        return speaker

    # ── หนึ่งเทิร์น ─────────────────────────────────────────────────────
    def stream_exchange(
        self,
        transcript: str,
        pcm: bytes | None = None,
        speaker: Speaker | None = None,
        speak: bool = True,
    ) -> Iterator[SessionEvent]:
        """ประมวลผลหนึ่งเทิร์นแบบสตรีม"""
        transcript = transcript.strip()
        if not transcript:
            return

        if speaker is not None:
            self.current_speaker = speaker
            ident = Identification(speaker=speaker, method="fallback")
        else:
            ident = self.identify(transcript, pcm)
            speaker = ident.speaker

        yield SessionEvent("speaker", speaker=speaker, identification=ident)

        # คำสั่งลบความจำ — ทำเองไม่ผ่านโมเดล เพื่อให้มั่นใจว่าลบจริง
        if speaker is not None and detect_forget_all(transcript):
            yield from self._handle_forget(speaker, transcript, speak)
            return

        if speaker is not None:
            self.store.record_turn(speaker.id, self.session_id, "user", transcript)

        reply = ""
        chunks: list[str] = []
        for event in self.brain.stream(
            transcript, speaker, voice_enabled=self.identifier.enabled
        ):
            if event.type == "delta":
                yield SessionEvent("delta", text=event.text, speaker=speaker)
            elif event.type == "chunk":
                chunks.append(event.text)
                yield SessionEvent(
                    "chunk",
                    text=event.text,
                    speaker=speaker,
                    speech=self._speak(event.text, speaker) if speak else None,
                )
            elif event.type == "done":
                reply = event.text

        if speaker is not None and reply:
            self.store.record_turn(speaker.id, self.session_id, "assistant", reply)
            self._remember(speaker)

        yield SessionEvent("done", text=reply, speaker=speaker, identification=ident)

    def exchange(
        self,
        transcript: str,
        pcm: bytes | None = None,
        speaker: Speaker | None = None,
        speak: bool = True,
        on_event: Callable[[SessionEvent], None] | None = None,
    ) -> ExchangeResult:
        """ประมวลผลหนึ่งเทิร์นแบบรอผลลัพธ์ครบ"""
        result = ExchangeResult(transcript=transcript, reply="")
        for event in self.stream_exchange(transcript, pcm, speaker, speak):
            if on_event:
                on_event(event)
            if event.type == "speaker":
                result.speaker = event.speaker
                result.identification = event.identification
            elif event.type == "chunk":
                result.chunks.append(event.text)
            elif event.type == "done":
                result.reply = event.text
                result.speaker = event.speaker
        return result

    # ── ภายใน ───────────────────────────────────────────────────────────
    def _handle_forget(
        self, speaker: Speaker, transcript: str, speak: bool
    ) -> Iterator[SessionEvent]:
        removed = self.store.forget_all_facts(speaker.id)
        log.info("ลบความจำ %d รายการของ speaker %s", removed, speaker.id)
        particle = speaker.particle or "ครับ"
        reply = _CONFIRM_FORGET.format(particle=particle)
        self.store.record_turn(speaker.id, self.session_id, "user", transcript)
        self.store.record_turn(speaker.id, self.session_id, "assistant", reply)
        yield SessionEvent(
            "chunk",
            text=reply,
            speaker=speaker,
            speech=self._speak(reply, speaker) if speak else None,
        )
        yield SessionEvent("done", text=reply, speaker=speaker)

    def _speak(self, text: str, speaker: Speaker | None) -> Speech | None:
        if self.tts is None or not text.strip():
            return None
        try:
            return self.tts.synthesize(text)
        except Exception:
            log.warning("สังเคราะห์เสียงไม่สำเร็จ", exc_info=True)
            return None

    def _remember(self, speaker: Speaker) -> None:
        if self.extractor is None:
            return
        turns = self.store.recent_turns(speaker.id, limit=6)
        self.extractor.schedule(speaker, turns)
        self.extractor.maybe_summarize(speaker)
