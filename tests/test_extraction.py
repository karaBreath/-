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


class TestStrictSchema:
    """schema ที่ส่งไปกับ structured outputs ต้องเข้มงวดพอ ไม่งั้นถูกปฏิเสธเป็น 400"""

    def test_ทุกอ็อบเจ็กต์ต้องปิดฟิลด์เกิน(self):
        from thaivoice.extraction import MemoryUpdate, strict_json_schema

        schema = strict_json_schema(MemoryUpdate)

        def objects(node, path="root"):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    yield path, node
                for key, value in node.items():
                    yield from objects(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    yield from objects(value, f"{path}[{index}]")

        found = list(objects(schema))
        assert found, "ต้องมีอ็อบเจ็กต์อย่างน้อยหนึ่งตัว"
        for path, node in found:
            assert node.get("additionalProperties") is False, path
            assert node.get("required"), path

    def test_ไม่ทำลาย_schema_เดิม(self):
        from thaivoice.extraction import MemoryUpdate, strict_json_schema

        schema = strict_json_schema(MemoryUpdate)
        assert set(schema["properties"]) == {
            "facts",
            "forget_keys",
            "display_name",
            "nickname",
            "gender",
        }


class Testความจำที่ถูกลบต้องไม่ฟื้นคืนมา:
    """งานเบื้องหลังถ่ายภาพบทสนทนาไว้แล้วเรียกโมเดล

    ระหว่างนั้นผู้ใช้อาจสั่งลบความจำและได้ยินว่า "ลบเรียบร้อยแล้ว"
    พองานกลับมา มันเขียนข้อเท็จจริงที่สกัดจากบทสนทนาที่ถูกลบไปแล้วกลับเข้าไป
    ระบบจึงบอกผู้ใช้ว่าข้อมูลหายไปทั้งที่ยังอยู่
    """

    def test_ลบระหว่างสกัดแล้วต้องทิ้งผล(self, store, settings):
        import threading

        speaker = store.create_speaker("เดช")
        store.record_turn(speaker.id, "s", "user", "ผมเป็นหมอครับ")
        turns = store.recent_turns(speaker.id)

        เริ่มแล้ว = threading.Event()
        ลบเสร็จ = threading.Event()

        class Clientที่ค้าง(FakeAnthropic):
            def _hold(self):
                เริ่มแล้ว.set()
                ลบเสร็จ.wait(timeout=5)

        client = Clientที่ค้าง()
        client.parsed_outputs = [
            MemoryUpdate(
                facts=[
                    ExtractedFact(
                        key="อาชีพ", value="หมอ", category="งาน", confidence=0.9
                    )
                ],
                forget_keys=[],
                display_name="",
                nickname="",
                gender="unknown",
            )
        ]
        extractor = MemoryExtractor(store, client, settings)
        original = extractor._call_model

        def slow(prompt):
            client._hold()
            return original(prompt)

        extractor._call_model = slow
        extractor.schedule(speaker, turns)

        assert เริ่มแล้ว.wait(timeout=5), "งานเบื้องหลังต้องเริ่มแล้ว"
        removed = store.forget_everything(speaker.id)
        assert store.facts_for(speaker.id) == []
        ลบเสร็จ.set()
        extractor.shutdown(wait=True)

        assert store.facts_for(speaker.id) == [], (
            f"ข้อเท็จจริงฟื้นคืนมาหลังบอกผู้ใช้ว่าลบแล้ว (ลบไป {removed})"
        )

    def test_สกัดตามปกติยังเขียนได้(self, store, settings):
        speaker = store.create_speaker("เดช")
        store.record_turn(speaker.id, "s", "user", "ผมเป็นหมอครับ")
        client = FakeAnthropic()
        client.parsed_outputs = [
            MemoryUpdate(
                facts=[
                    ExtractedFact(
                        key="อาชีพ", value="หมอ", category="งาน", confidence=0.9
                    )
                ],
                forget_keys=[],
                display_name="",
                nickname="",
                gender="unknown",
            )
        ]
        extractor = MemoryExtractor(store, client, settings)

        extractor.schedule(speaker, store.recent_turns(speaker.id))
        extractor.shutdown(wait=True)

        assert [(f.key, f.value) for f in store.facts_for(speaker.id)] == [("อาชีพ", "หมอ")]
