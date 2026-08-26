"""ของปลอมสำหรับเทสต์ — ทำให้ทดสอบทั้งเส้นทางได้โดยไม่ต้องเรียก Claude API จริง"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from thaivoice.config import Settings  # noqa: E402
from thaivoice.memory import MemoryStore  # noqa: E402


# ── Claude ปลอม ─────────────────────────────────────────────────────────────
@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeMessage:
    content: list[FakeTextBlock]
    stop_reason: str = "end_turn"


class FakeStream:
    """เลียนแบบ context manager ที่ ``client.messages.stream()`` คืนมา"""

    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self._text = text
        self._stop_reason = stop_reason

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    @property
    def text_stream(self):
        # ปล่อยทีละไม่กี่ตัวอักษร เพื่อให้ตัวตัดประโยคได้ทำงานจริง
        step = 7
        for i in range(0, len(self._text), step):
            yield self._text[i : i + step]

    def get_final_message(self) -> FakeMessage:
        return FakeMessage([FakeTextBlock(self._text)], self._stop_reason)


@dataclass
class FakeMessages:
    parent: "FakeAnthropic"

    def stream(self, **kwargs: Any) -> FakeStream:
        self.parent.calls.append(kwargs)
        if self.parent.fail_with is not None:
            raise self.parent.fail_with
        return FakeStream(self.parent.next_reply(), self.parent.stop_reason)

    def create(self, **kwargs: Any) -> FakeMessage:
        self.parent.calls.append(kwargs)
        if self.parent.fail_with is not None:
            raise self.parent.fail_with
        return FakeMessage([FakeTextBlock(self.parent.next_reply())])

    def parse(self, **kwargs: Any):
        self.parent.calls.append(kwargs)
        parsed = self.parent.parsed_outputs.pop(0) if self.parent.parsed_outputs else None
        if parsed is None:
            raise RuntimeError("ไม่ได้ตั้งค่า parsed_output ไว้")

        class Parsed:
            parsed_output = parsed

        return Parsed()


@dataclass
class FakeBeta:
    parent: "FakeAnthropic"

    @property
    def messages(self) -> FakeMessages:
        return FakeMessages(self.parent)


class FakeAnthropic:
    """Claude ปลอมที่ตอบตามสคริปต์ที่กำหนดไว้ล่วงหน้า"""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = replies or ["สวัสดีครับ ยินดีที่ได้รู้จักครับ"]
        self.parsed_outputs: list[Any] = []
        self.calls: list[dict] = []
        self.stop_reason = "end_turn"
        # ตั้งเป็น exception เพื่อจำลองว่าโมเดลล่ม
        self.fail_with: BaseException | None = None
        self._index = 0

    def next_reply(self) -> str:
        if not self.replies:
            return ""
        reply = self.replies[min(self._index, len(self.replies) - 1)]
        self._index += 1
        return reply

    @property
    def messages(self) -> FakeMessages:
        return FakeMessages(self)

    @property
    def beta(self) -> FakeBeta:
        return FakeBeta(self)


# ── ตัวสร้างลายเสียงปลอม ────────────────────────────────────────────────────
@dataclass
class FakeEmbedder:
    """แปลงเสียงเป็นเวกเตอร์แบบกำหนดเองได้ ใช้ทดสอบการจำเสียง"""

    name: str = "fake"
    table: dict[bytes, list[float]] = field(default_factory=dict)

    def embed(self, pcm: bytes, sample_rate: int) -> list[float]:
        if pcm in self.table:
            return self.table[pcm]
        # ค่าเริ่มต้น: ใช้ค่าไบต์แรก ๆ เป็นเวกเตอร์ เพื่อให้เสียงต่างกันได้เวกเตอร์ต่างกัน
        head = pcm[:3].ljust(3, b"\x00")
        return [byte / 255 for byte in head]


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def store() -> MemoryStore:
    memory = MemoryStore(":memory:")
    yield memory
    memory.close()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        db_path=tmp_path / "memory.db",
        model="claude-opus-5",
        effort="low",
        speaker_backend="none",
        tts_backend="none",
        stt_backend="external",
    )


@pytest.fixture
def fake_client() -> FakeAnthropic:
    return FakeAnthropic()
