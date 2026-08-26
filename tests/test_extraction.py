"""ทดสอบการสกัดความจำจากบทสนทนา"""

import json

import pytest
from conftest import FakeAnthropic

from thaivoice.extraction import ExtractedFact, MemoryExtractor, MemoryUpdate
from thaivoice.memory import MemoryStore


@pytest.fixture
def extractor(store: MemoryStore, fake_client: FakeAnthropic, settings) -> MemoryExtractor:
    made = MemoryExtractor(store, fake_client, settings)
    yield made
    made.shutdown(wait=False)


def _turns(store: MemoryStore, speaker_id: int) -> list:
    store.record_turn(speaker_id, "s1", "user", "ผมเป็นสถาปนิก อยู่เชียงใหม่ครับ")
    store.record_turn(speaker_id, "s1", "assistant", "ดีจังครับ")
    return store.recent_turns(speaker_id)


class TestApplyUpdate:
    def test_บันทึกข้อเท็จจริงที่สกัดได้(self, store, extractor, fake_client):
        speaker = store.create_speaker("เดช")
        fake_client.parsed_outputs = [
            MemoryUpdate(
                facts=[
                    ExtractedFact(key="อาชีพ", value="สถาปนิก", category="งาน", confidence=0.9),
                    ExtractedFact(
                        key="เมืองที่อยู่", value="เชียงใหม่", category="ข้อมูลส่วนตัว", confidence=0.8
                    ),
                ],
                forget_keys=[],
                display_name="",
                nickname="",
                gender="unknown",
            )
        ]

        extractor.run_now(speaker, _turns(store, speaker.id))

        facts = {f.key: f.value for f in store.facts_for(speaker.id)}
        assert facts == {"อาชีพ": "สถาปนิก", "เมืองที่อยู่": "เชียงใหม่"}

    def test_ลบคีย์ที่ผู้ใช้บอกให้ลืม(self, store, extractor, fake_client):
        speaker = store.create_speaker("เดช")
        store.upsert_fact(speaker.id, "อาชีพ", "หมอ")
        fake_client.parsed_outputs = [
            MemoryUpdate(
                facts=[], forget_keys=["อาชีพ"], display_name="", nickname="", gender="unknown"
            )
        ]

        extractor.run_now(speaker, _turns(store, speaker.id))
        assert store.facts_for(speaker.id) == []

    def test_เติมชื่อเล่นและเพศเมื่อยังว่าง(self, store, extractor, fake_client):
        speaker = store.create_speaker("ผู้ใช้ใหม่")
        fake_client.parsed_outputs = [
            MemoryUpdate(
                facts=[],
                forget_keys=[],
                display_name="เดชา",
                nickname="เดช",
                gender="male",
            )
        ]

        extractor.run_now(speaker, _turns(store, speaker.id))

        updated = store.get_speaker(speaker.id)
        assert updated.display_name == "เดชา"
        assert updated.nickname == "เดช"
        assert updated.particle == "ครับ"

    def test_ไม่ทับชื่อเล่นที่ผู้ใช้ตั้งไว้แล้ว(self, store, extractor, fake_client):
        speaker = store.create_speaker("เดชา", nickname="พี่เดช")
        fake_client.parsed_outputs = [
            MemoryUpdate(
                facts=[], forget_keys=[], display_name="", nickname="เดช", gender="unknown"
            )
        ]

        extractor.run_now(speaker, _turns(store, speaker.id))
        assert store.get_speaker(speaker.id).nickname == "พี่เดช"

    def test_บีบความมั่นใจให้อยู่ในช่วง_0_ถึง_1(self, store, extractor, fake_client):
        speaker = store.create_speaker("เดช")
        fake_client.parsed_outputs = [
            MemoryUpdate(
                facts=[ExtractedFact(key="ก", value="ข", category="อื่น ๆ", confidence=5.0)],
                forget_keys=[],
                display_name="",
                nickname="",
                gender="unknown",
            )
        ]

        extractor.run_now(speaker, _turns(store, speaker.id))
        assert store.facts_for(speaker.id)[0].confidence == 1.0


class TestFallbacks:
    def test_ถอยไปใช้_json_schema_เมื่อ_parse_ใช้ไม่ได้(self, store, extractor, fake_client):
        """SDK บางรุ่นไม่มี messages.parse — ต้องยังสกัดความจำได้"""
        speaker = store.create_speaker("เดช")
        fake_client.parsed_outputs = []  # ทำให้ parse() โยน error
        fake_client.replies = [
            json.dumps(
                {
                    "facts": [
                        {"key": "สัตว์เลี้ยง", "value": "แมวสองตัว", "category": "ข้อมูลส่วนตัว",
                         "confidence": 0.9}
                    ],
                    "forget_keys": [],
                    "display_name": "",
                    "nickname": "",
                    "gender": "unknown",
                },
                ensure_ascii=False,
            )
        ]

        update = extractor.run_now(speaker, _turns(store, speaker.id))

        assert update is not None
        assert store.facts_for(speaker.id)[0].value == "แมวสองตัว"

    def test_โมเดลตอบเป็นขยะแล้วไม่ทำให้ระบบล่ม(self, store, extractor, fake_client):
        speaker = store.create_speaker("เดช")
        fake_client.parsed_outputs = []
        fake_client.replies = ["นี่ไม่ใช่ JSON เลย"]

        assert extractor.run_now(speaker, _turns(store, speaker.id)) is None
        assert store.facts_for(speaker.id) == []

    def test_ไม่มีบทสนทนาก็ไม่เรียกโมเดล(self, store, extractor, fake_client):
        speaker = store.create_speaker("เดช")
        assert extractor.run_now(speaker, []) is None
        assert fake_client.calls == []
