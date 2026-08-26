"""สมองของระบบ — เรียก Claude แล้วสตรีมคำตอบภาษาไทยออกมาเป็นท่อน ๆ

จุดสำคัญของบทสนทนาด้วยเสียงคือ latency ระบบจึงสตรีมคำตอบและปล่อยออกมาเป็น
"ท่อนที่พูดได้" ทันทีที่จบประโยค แทนที่จะรอคำตอบครบทั้งก้อนแล้วค่อยเริ่มพูด
ทำให้ผู้ใช้ได้ยินเสียงตอบเร็วขึ้นมาก

โครงสร้าง prompt ที่ส่งไป:

    system[0]  ← กฎการสนทนาที่คงที่ (ทำเครื่องหมายให้แคชได้)
    system[1]  ← ความจำเกี่ยวกับคนที่กำลังคุยอยู่ (เปลี่ยนทุกเทิร์น)
    messages   ← บทสนทนาย้อนหลังของ "คนคนนี้" + สิ่งที่เพิ่งพูด
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import anthropic

from .config import Settings, get_settings
from .memory import MemoryStore, Speaker, Turn
from .prompts import base_system, build_memory_block, unknown_speaker_block
from .thai_text import SpeechChunker, clean_for_speech

log = logging.getLogger("thaivoice.brain")

__all__ = [
    "ThaiBrain",
    "BrainEvent",
    "MissingCredentialsError",
    "REFUSAL_REPLY",
    "refusal_reply",
]

# beta flag สำหรับ server-side fallback: ถ้าคำขอถูกปฏิเสธด้วยตัวจำแนกความปลอดภัย
# เซิร์ฟเวอร์จะสลับไปโมเดลที่เหมาะสมให้เอง แทนที่บทสนทนาจะเงียบไปเฉย ๆ
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

def refusal_reply(particle: str = "ค่ะ") -> str:
    """ข้อความปฏิเสธ — ต้องใช้คำลงท้ายของผู้ช่วย ไม่ใช่ฮาร์ดโค้ดเป็นเพศหญิง

    ถ้าตั้งผู้ช่วยเป็นผู้ชาย เสียงจะเป็นเสียงผู้ชายแต่พูดว่า "ค่ะ"
    ซึ่งเป็นความไม่เข้าคู่ที่ prompt เองเรียกว่า "สะดุดหูทันทีที่ได้ยิน"
    """
    return f"ขอโทษ{particle} เรื่องนี้ตอบให้ไม่ได้ ลองถามเรื่องอื่นได้เลย{particle}"


REFUSAL_REPLY = refusal_reply()

_NO_CREDENTIALS_HINT = (
    "ยังไม่ได้ตั้งค่าคีย์สำหรับเรียก Claude\n"
    "  วิธีแก้: ใส่ ANTHROPIC_API_KEY ในไฟล์ .env หรือรัน `ant auth login`\n"
    "  ตรวจสถานะทั้งระบบได้ด้วย: thaivoice doctor"
)


class MissingCredentialsError(RuntimeError):
    """ไม่พบคีย์หรือโปรไฟล์สำหรับเรียก Claude"""


@dataclass
class BrainEvent:
    """เหตุการณ์ระหว่างสตรีมคำตอบ

    * ``delta``    — ข้อความชิ้นเล็กตามที่โมเดลพิมพ์ออกมา (ใช้แสดงบนจอ)
    * ``chunk``    — ประโยคที่สมบูรณ์ ทำความสะอาดแล้ว พร้อมส่งให้ TTS พูด
    * ``done``     — จบแล้ว ``text`` คือคำตอบเต็ม
    """

    type: str
    text: str


class ThaiBrain:
    """ห่อ Claude API ให้กลายเป็นคู่สนทนาภาษาไทยที่จำผู้พูดได้"""

    def __init__(
        self,
        store: MemoryStore,
        client: anthropic.Anthropic | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        # ไม่ส่ง api_key ตรง ๆ — SDK จะหาจาก ANTHROPIC_API_KEY หรือโปรไฟล์ ant auth
        self.client = client or anthropic.Anthropic()
        self._base_system = base_system(
            self.settings.assistant_name, self.settings.assistant_particle
        )
        self._use_fallbacks = True

    # ── ประกอบ prompt ───────────────────────────────────────────────────
    def build_system(self, speaker: Speaker | None, voice_enabled: bool) -> list[dict]:
        particle = self.settings.assistant_particle
        if speaker is None:
            memory = unknown_speaker_block(voice_enabled, particle)
        else:
            memory = build_memory_block(
                speaker,
                self.store.facts_for(speaker.id),
                (self.store.latest_summary(speaker.id) or (None, 0))[0],
                self.store.stats(speaker.id),
                assistant_particle=particle,
            )
        return [
            # ส่วนคงที่มาก่อนเสมอ และทำเครื่องหมายแคชไว้ — cache เป็นการจับคู่
            # ส่วนหน้าของ prompt ถ้าสลับลำดับกับความจำ แคชจะพังทุกเทิร์น
            {
                "type": "text",
                "text": self._base_system,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": memory},
        ]

    def build_messages(
        self,
        speaker: Speaker | None,
        user_text: str,
        extra_history: Sequence[Turn] | None = None,
    ) -> list[dict]:
        """ประกอบรายการข้อความที่จะส่งให้โมเดล

        ``extra_history`` เป็น ``None`` แปลว่า "ไปโหลดเอง" ส่วนลิสต์ว่างแปลว่า
        "ไม่มีประวัติจริง ๆ" ต้องแยกสองกรณีนี้ ไม่งั้นผู้เรียกที่ตั้งใจส่งประวัติว่าง
        จะได้ประวัติที่โหลดเองซ้อนมา ซึ่งรวมถึงเทิร์นที่เพิ่งบันทึกไปเมื่อครู่
        ทำให้ข้อความล่าสุดถูกส่งซ้ำสองครั้ง
        """
        if extra_history is None:
            history: list[Turn] = (
                self.store.recent_turns(speaker.id, limit=self.settings.history_turns)
                if speaker is not None
                else []
            )
        else:
            history = list(extra_history)

        messages: list[dict] = [
            {"role": t.role, "content": t.content} for t in history if t.content.strip()
        ]
        # API บังคับว่าข้อความแรกต้องเป็นของผู้ใช้
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        messages.append({"role": "user", "content": user_text})
        return messages

    # ── สตรีมคำตอบ ──────────────────────────────────────────────────────
    def stream(
        self,
        user_text: str,
        speaker: Speaker | None = None,
        *,
        voice_enabled: bool = True,
        history: Sequence[Turn] | None = None,
    ) -> Iterator[BrainEvent]:
        """สตรีมคำตอบออกมาเป็น BrainEvent

        ท่อนที่ ``type == "chunk"`` คือสิ่งที่ควรส่งเข้า TTS ทันที
        """
        params: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": self.build_system(speaker, voice_enabled),
            "messages": self.build_messages(speaker, user_text, history),
            "output_config": {"effort": self.settings.effort},
        }

        chunker = SpeechChunker()
        parts: list[str] = []
        emitted = False

        for stream_ctx in self._stream_attempts(params):
            try:
                with stream_ctx() as stream:
                    for delta in stream.text_stream:
                        if not delta:
                            continue
                        emitted = True
                        parts.append(delta)
                        yield BrainEvent("delta", delta)
                        for chunk in chunker.feed(delta):
                            spoken = clean_for_speech(chunk)
                            if spoken:
                                yield BrainEvent("chunk", spoken)
                    final = stream.get_final_message()
                break
            except anthropic.BadRequestError as exc:
                # beta flag ใช้ไม่ได้กับ endpoint/บัญชีนี้ -> ลองใหม่แบบไม่ใช้
                if emitted or not self._use_fallbacks:
                    raise
                log.info("ปิด server-side fallback แล้วลองใหม่: %s", exc)
                self._use_fallbacks = False
                continue
            except anthropic.AuthenticationError as exc:
                raise MissingCredentialsError(_NO_CREDENTIALS_HINT) from exc
            except TypeError as exc:
                # SDK โยน TypeError เมื่อหาคีย์ไม่เจอ ซึ่งอ่านแล้วไม่รู้ว่าต้องทำอะไรต่อ
                if "authentication" in str(exc).lower():
                    raise MissingCredentialsError(_NO_CREDENTIALS_HINT) from exc
                raise
        else:  # pragma: no cover - ลองครบทุกทางแล้วยังไม่สำเร็จ
            raise RuntimeError("เรียกโมเดลไม่สำเร็จ")

        for chunk in chunker.flush():
            spoken = clean_for_speech(chunk)
            if spoken:
                yield BrainEvent("chunk", spoken)

        text = "".join(parts).strip()
        if not text and getattr(final, "stop_reason", None) == "refusal":
            text = refusal_reply(self.settings.assistant_particle)
            yield BrainEvent("chunk", text)
        yield BrainEvent("done", text)

    def reply(
        self,
        user_text: str,
        speaker: Speaker | None = None,
        *,
        voice_enabled: bool = True,
        history: Sequence[Turn] | None = None,
    ) -> str:
        """เวอร์ชันไม่สตรีม — สะดวกสำหรับเทสต์และ API แบบ request/response"""
        text = ""
        for event in self.stream(
            user_text, speaker, voice_enabled=voice_enabled, history=history
        ):
            if event.type == "done":
                text = event.text
        return text

    # ── ภายใน ───────────────────────────────────────────────────────────
    def _stream_attempts(self, params: dict[str, Any]):
        """คืนลิสต์วิธีเรียก เรียงตามลำดับที่จะลอง

        ลองแบบเปิด server-side fallback ก่อน (กันคำขอถูกปฏิเสธแล้วเงียบ)
        ถ้า beta ใช้ไม่ได้ ค่อยถอยไปใช้ endpoint ปกติ
        """
        attempts = []
        if self._use_fallbacks:
            attempts.append(
                lambda: self.client.beta.messages.stream(
                    betas=[_FALLBACK_BETA], fallbacks="default", **params
                )
            )
        attempts.append(lambda: self.client.messages.stream(**params))
        return attempts
