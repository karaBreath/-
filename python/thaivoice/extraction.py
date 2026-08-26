"""สกัดความจำจากบทสนทนา และสรุปบทสนทนาเป็นระยะ

ทำงานเป็น "งานเบื้องหลัง" หลังจากตอบผู้ใช้ไปแล้ว เพื่อไม่ให้ผู้ใช้ต้องรอ
(latency ในบทสนทนาด้วยเสียงคือทุกอย่าง)

ใช้ structured outputs เพื่อให้ผลลัพธ์เป็น JSON ที่ผ่านการตรวจสอบ schema แล้ว
ไม่ต้องมานั่งแกะข้อความอิสระ
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .memory import MemoryStore, Speaker, Turn

log = logging.getLogger("thaivoice.extraction")

__all__ = ["MemoryUpdate", "MemoryExtractor", "ExtractedFact"]

CATEGORIES = (
    "ข้อมูลส่วนตัว",
    "งาน",
    "ความชอบ",
    "ความสัมพันธ์",
    "แผนการ",
    "สุขภาพ",
    "อื่น ๆ",
)


class ExtractedFact(BaseModel):
    """ข้อเท็จจริงหนึ่งข้อที่ควรจำไว้เกี่ยวกับผู้สนทนา"""

    key: str = Field(description="หัวข้อสั้น ๆ เป็นภาษาไทย เช่น 'อาชีพ' 'เมืองที่อยู่'")
    value: str = Field(description="เนื้อหาของข้อเท็จจริง เป็นภาษาไทย สั้นที่สุดเท่าที่ยังเข้าใจ")
    category: Literal[
        "ข้อมูลส่วนตัว", "งาน", "ความชอบ", "ความสัมพันธ์", "แผนการ", "สุขภาพ", "อื่น ๆ"
    ]
    confidence: float = Field(description="ความมั่นใจ 0.0 ถึง 1.0")


class MemoryUpdate(BaseModel):
    """สิ่งที่ควรอัปเดตในความจำหลังบทสนทนารอบนี้"""

    facts: list[ExtractedFact] = Field(description="ข้อเท็จจริงใหม่หรือที่เปลี่ยนไป")
    forget_keys: list[str] = Field(
        description="คีย์ที่ผู้ใช้ขอให้ลืม หรือที่รู้แล้วว่าไม่จริง"
    )
    display_name: str = Field(description="ชื่อจริงถ้าเพิ่งรู้ ไม่งั้นใส่สตริงว่าง")
    nickname: str = Field(description="ชื่อเล่นถ้าเพิ่งรู้ ไม่งั้นใส่สตริงว่าง")
    gender: Literal["male", "female", "unknown"] = Field(
        description="เพศที่เดาจากคำลงท้าย ผม/ครับ = male, หนู/ดิฉัน/ค่ะ = female"
    )


_EXTRACT_SYSTEM = """\
คุณคือระบบจัดการความจำของผู้ช่วยสนทนาภาษาไทย

หน้าที่: อ่านบทสนทนารอบล่าสุด แล้วดึงเฉพาะสิ่งที่ "ควรจำข้ามบทสนทนา" ออกมา

จำ:
- ข้อเท็จจริงถาวรหรือกึ่งถาวรเกี่ยวกับผู้ใช้ เช่น ชื่อ อาชีพ ที่อยู่ ครอบครัว
  ความชอบ ความไม่ชอบ ภาษาที่ใช้ เป้าหมาย แผนที่วางไว้ ข้อจำกัดด้านสุขภาพ
- สิ่งที่ผู้ใช้บอกให้จำไว้ตรง ๆ
- วิธีที่เขาอยากให้เราคุยด้วย เช่น ให้เรียกชื่อเล่น ให้ตอบสั้น ๆ

ห้ามจำ:
- เนื้อหาที่ใช้ได้แค่ในบทสนทนานี้ เช่น คำถามความรู้ทั่วไป การขอให้คำนวณ
- สิ่งที่ผู้ช่วยพูดเอง ให้ดูเฉพาะข้อมูลที่มาจากผู้ใช้
- การคาดเดาที่ไม่มีหลักฐานในบทสนทนา
- ข้อมูลอ่อนไหวที่ผู้ใช้ไม่ได้ตั้งใจบอก เช่น รหัสผ่าน เลขบัตร เลขบัญชี

กติกา:
- key และ value เป็นภาษาไทย สั้น กระชับ ไม่ใส่คำลงท้าย ครับ/ค่ะ
- ใช้ key เดิมถ้าเป็นการอัปเดตของเดิม เพื่อให้ค่าใหม่ทับค่าเก่าได้ถูกต้อง
- ถ้าผู้ใช้บอกว่าข้อมูลเดิมไม่จริงแล้ว หรือขอให้ลืม ให้ใส่ key นั้นใน forget_keys
- ถ้าไม่มีอะไรน่าจำ ให้คืน facts เป็นลิสต์ว่าง อย่าฝืนหาเรื่องมาจำ
"""

_SUMMARY_SYSTEM = """\
คุณคือระบบสรุปความจำของผู้ช่วยสนทนาภาษาไทย

เขียนสรุปบทสนทนาสะสมกับผู้ใช้คนนี้ใหม่ทั้งหมด โดยรวมสรุปเดิมเข้ากับบทสนทนาใหม่
เขียนเป็นภาษาไทย ความยาวไม่เกิน 150 คำ เน้นเรื่องที่ยังมีผลต่อการคุยครั้งหน้า
เช่น เรื่องที่ค้างอยู่ สิ่งที่เขากำลังทำ อารมณ์โดยรวมของการคุย
ไม่ต้องใส่ข้อเท็จจริงที่ถูกเก็บแยกไว้แล้ว (ชื่อ อาชีพ ความชอบ)
ตอบมาเป็นข้อความสรุปอย่างเดียว ไม่ต้องมีหัวข้อหรือคำนำ
"""


def _format_turns(turns: Sequence[Turn]) -> str:
    lines = []
    for turn in turns:
        who = "ผู้ใช้" if turn.role == "user" else "ผู้ช่วย"
        lines.append(f"{who}: {turn.content}")
    return "\n".join(lines)


class MemoryExtractor:
    """สกัดและบันทึกความจำแบบไม่บล็อกบทสนทนา"""

    def __init__(
        self,
        store: MemoryStore,
        client,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.settings = settings or get_settings()
        # เธรดเดียวพอ — ให้งานสกัดความจำเรียงคิวกัน ไม่แย่ง rate limit กับบทสนทนา
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thaivoice-mem")
        self._lock = threading.Lock()

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    # ── API หลัก ────────────────────────────────────────────────────────
    def schedule(self, speaker: Speaker, turns: Sequence[Turn]) -> None:
        """สั่งให้สกัดความจำเบื้องหลัง (ไม่รอผล)"""
        snapshot = list(turns)
        self._pool.submit(self._safe_run, speaker, snapshot)

    def run_now(self, speaker: Speaker, turns: Sequence[Turn]) -> MemoryUpdate | None:
        """สกัดแบบรอผล — ใช้ในเทสต์และสคริปต์ที่ต้องการผลทันที"""
        return self._extract_and_apply(speaker, list(turns))

    # ── ภายใน ───────────────────────────────────────────────────────────
    def _safe_run(self, speaker: Speaker, turns: list[Turn]) -> None:
        try:
            self._extract_and_apply(speaker, turns)
        except Exception:  # งานเบื้องหลังต้องไม่ทำให้บทสนทนาล่ม
            log.exception("สกัดความจำไม่สำเร็จ")

    def _extract_and_apply(
        self, speaker: Speaker, turns: list[Turn]
    ) -> MemoryUpdate | None:
        if not turns:
            return None

        known = self.store.facts_for(speaker.id, limit=40)
        known_text = "\n".join(f"- {f.key}: {f.value}" for f in known) or "(ยังไม่มี)"
        prompt = (
            f"ความจำที่มีอยู่แล้วเกี่ยวกับผู้ใช้คนนี้:\n{known_text}\n\n"
            f"บทสนทนารอบล่าสุด:\n{_format_turns(turns)}"
        )

        update = self._call_model(prompt)
        if update is None:
            return None

        with self._lock:
            self._apply(speaker, update, turns)
        return update

    def _call_model(self, prompt: str) -> MemoryUpdate | None:
        """เรียกโมเดลด้วย structured output — ถอยไปใช้ JSON schema ดิบถ้าจำเป็น"""
        kwargs = dict(
            model=self.settings.memory_model,
            max_tokens=2000,
            system=_EXTRACT_SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            response = self.client.messages.parse(output_format=MemoryUpdate, **kwargs)
            parsed = getattr(response, "parsed_output", None)
            if isinstance(parsed, MemoryUpdate):
                return parsed
        except Exception:
            log.debug("messages.parse ใช้ไม่ได้ ถอยไปใช้ json_schema ดิบ", exc_info=True)

        try:
            response = self.client.messages.create(
                output_config={
                    "effort": "low",
                    "format": {
                        "type": "json_schema",
                        "schema": MemoryUpdate.model_json_schema(),
                    },
                },
                **{k: v for k, v in kwargs.items() if k != "output_config"},
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            return MemoryUpdate.model_validate(json.loads(text))
        except Exception:
            log.warning("สกัดความจำไม่สำเร็จทั้งสองวิธี", exc_info=True)
            return None

    def _apply(
        self, speaker: Speaker, update: MemoryUpdate, turns: list[Turn]
    ) -> None:
        source_turn_id = turns[-1].id if turns else None

        for key in update.forget_keys:
            if key.strip():
                self.store.forget_fact(speaker.id, key.strip())

        for fact in update.facts:
            self.store.upsert_fact(
                speaker.id,
                fact.key,
                fact.value,
                category=fact.category,
                confidence=max(0.0, min(1.0, fact.confidence)),
                source_turn_id=source_turn_id,
            )

        # อัปเดตโปรไฟล์ — เขียนเฉพาะช่องที่ยังว่างอยู่ ไม่ทับของที่ผู้ใช้ตั้งเอง
        profile: dict[str, str] = {}
        if update.display_name.strip() and speaker.display_name.startswith("ผู้ใช้"):
            profile["display_name"] = update.display_name.strip()
        if update.nickname.strip() and not speaker.nickname:
            profile["nickname"] = update.nickname.strip()
        if update.gender != "unknown" and not speaker.gender:
            from .thai_text import particle_for_gender

            profile["gender"] = update.gender
            profile["particle"] = particle_for_gender(update.gender)
        if profile:
            self.store.update_speaker(speaker.id, **profile)

    # ── บทสรุปสะสม ──────────────────────────────────────────────────────
    def maybe_summarize(self, speaker: Speaker) -> None:
        """สรุปบทสนทนาสะสมทุก ๆ N เทิร์น (ทำเบื้องหลัง)"""
        total = self.store.turn_count(speaker.id)
        if total and total % max(2, self.settings.summarize_every) == 0:
            self._pool.submit(self._safe_summarize, speaker)

    def _safe_summarize(self, speaker: Speaker) -> None:
        try:
            self._summarize(speaker)
        except Exception:
            log.exception("สรุปบทสนทนาไม่สำเร็จ")

    def _summarize(self, speaker: Speaker) -> None:
        previous = self.store.latest_summary(speaker.id)
        after = previous[1] if previous else 0
        turns = self.store.recent_turns(speaker.id, limit=60, after_turn_id=after)
        if not turns:
            return
        prompt = (
            f"สรุปเดิม:\n{previous[0] if previous else '(ยังไม่มี)'}\n\n"
            f"บทสนทนาใหม่:\n{_format_turns(turns)}"
        )
        response = self.client.messages.create(
            model=self.settings.memory_model,
            max_tokens=1000,
            system=_SUMMARY_SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        if text:
            self.store.save_summary(speaker.id, text, turns[-1].id)
