"""ทดสอบทั้งเส้นทางของหนึ่งเทิร์น — รู้ว่าใครพูด คิด ตอบ แล้วจำ

ใช้ Claude ปลอม จึงรันได้โดยไม่ต้องต่ออินเทอร์เน็ตและไม่มีค่าใช้จ่าย
"""

import pytest
from conftest import FakeAnthropic, FakeEmbedder

from thaivoice.brain import ThaiBrain
from thaivoice.memory import MemoryStore
from thaivoice.session import ConversationSession, detect_forget_all
from thaivoice.speaker import SpeakerIdentifier


@pytest.fixture
def session(store: MemoryStore, fake_client: FakeAnthropic, settings) -> ConversationSession:
    brain = ThaiBrain(store, client=fake_client, settings=settings)
    identifier = SpeakerIdentifier(store, FakeEmbedder(), threshold=0.9, margin=0.02)
    return ConversationSession(
        store=store, brain=brain, identifier=identifier, settings=settings, session_id="test"
    )


class TestExchange:
    def test_เทิร์นแรกสร้างผู้สนทนาจากชื่อที่บอก(self, session, store):
        result = session.exchange("สวัสดีครับ ผมชื่อเดชครับ", speak=False)

        assert result.speaker is not None
        assert result.speaker.display_name == "เดช"
        assert result.reply
        assert store.turn_count(result.speaker.id) == 2  # ผู้ใช้ + ผู้ช่วย

    def test_ปรับคำลงท้ายตามเพศที่เดาได้(self, session, store):
        result = session.exchange("สวัสดีค่ะ หนูชื่อแนนค่ะ", speak=False)
        assert result.speaker.particle == "ค่ะ"

    def test_จำเสียงได้ในเทิร์นถัดไป(self, session):
        voice = b"\x44\x55\x66" * 90
        first = session.exchange("ผมชื่อเดชครับ", pcm=voice, speak=False)
        second = session.exchange("วันนี้อากาศดีนะ", pcm=voice, speak=False)

        assert second.speaker.id == first.speaker.id
        assert second.identification.method == "voice"

    def test_คนละเสียงคือคนละคน(self, session, store):
        session.exchange("ผมชื่อเอกครับ", pcm=b"\xff\x00\x00" * 90, speak=False)
        result = session.exchange("หนูชื่อบีค่ะ", pcm=b"\x00\x00\xff" * 90, speak=False)

        assert len(store.list_speakers()) == 2
        assert result.speaker.display_name == "บี"

    def test_ข้อความว่างไม่ทำอะไร(self, session):
        result = session.exchange("   ", speak=False)
        assert result.reply == ""

    def test_ปล่อยท่อนเสียงระหว่างตอบ(self, session, fake_client):
        fake_client.replies = ["สวัสดีครับ ยินดีที่ได้รู้จักนะครับ วันนี้มีอะไรให้ช่วยไหมครับ"]
        result = session.exchange("สวัสดี", speak=False)
        assert len(result.chunks) >= 2, "ควรตัดเป็นหลายท่อนเพื่อเริ่มพูดได้เร็ว"


class TestMemoryInPrompt:
    def test_ความจำถูกส่งเข้า_prompt(self, session, store, fake_client):
        speaker = session.register_speaker("เดช")
        store.upsert_fact(speaker.id, "อาชีพ", "สถาปนิก", category="งาน")

        session.exchange("จำได้ไหมว่าผมทำงานอะไร", speaker=speaker, speak=False)

        system_blocks = fake_client.calls[-1]["system"]
        memory_block = system_blocks[1]["text"]
        assert "เดช" in memory_block
        assert "สถาปนิก" in memory_block

    def test_ส่วนคงที่มาก่อนและทำเครื่องหมายแคชไว้(self, session, fake_client):
        session.exchange("สวัสดี", speak=False)
        system_blocks = fake_client.calls[-1]["system"]

        assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in system_blocks[1], "ส่วนที่เปลี่ยนทุกเทิร์นต้องไม่ถูกแคช"

    def test_บทสนทนาย้อนหลังถูกส่งไปด้วย(self, session, fake_client):
        speaker = session.register_speaker("เดช")
        session.exchange("ผมชอบกาแฟ", speaker=speaker, speak=False)
        session.exchange("แล้วชาล่ะ", speaker=speaker, speak=False)

        messages = fake_client.calls[-1]["messages"]
        assert messages[0]["role"] == "user"
        assert any("กาแฟ" in m["content"] for m in messages)
        assert messages[-1]["content"] == "แล้วชาล่ะ"

    def test_ความจำแยกกันระหว่างคน(self, session, store, fake_client):
        เดช = session.register_speaker("เดช")
        store.upsert_fact(เดช.id, "ความลับ", "ชอบกินทุเรียน")
        แนน = session.register_speaker("แนน")

        session.exchange("สวัสดี", speaker=แนน, speak=False)
        memory_block = fake_client.calls[-1]["system"][1]["text"]
        assert "ทุเรียน" not in memory_block, "ความจำของคนอื่นต้องไม่รั่วมา"


class TestForgetCommand:
    def test_สั่งลืมแล้วลบจริงโดยไม่ผ่านโมเดล(self, session, store, fake_client):
        speaker = session.register_speaker("เดช")
        store.upsert_fact(speaker.id, "อาชีพ", "หมอ")
        calls_before = len(fake_client.calls)

        result = session.exchange("ลืมทุกอย่างเกี่ยวกับฉันเลย", speaker=speaker, speak=False)

        assert store.facts_for(speaker.id) == []
        assert "ลบความจำ" in result.reply
        assert len(fake_client.calls) == calls_before, "ไม่ควรเรียกโมเดลสำหรับคำสั่งลบ"

    def test_ยังจำตัวคนไว้แม้ลบข้อเท็จจริง(self, session, store):
        speaker = session.register_speaker("เดช")
        session.exchange("ลบความจำที", speaker=speaker, speak=False)
        assert store.get_speaker(speaker.id) is not None

    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("ลืมทุกอย่างเกี่ยวกับฉัน", True),
            ("ลบความจำหน่อย", True),
            ("เคลียร์ข้อมูลให้ที", True),
            ("forget everything", True),
            ("วันนี้กินอะไรดี", False),
            ("ผมลืมกุญแจไว้ที่บ้าน", False),
        ],
    )
    def test_ตรวจจับคำสั่งลืม(self, utterance, expected):
        assert detect_forget_all(utterance) is expected


class TestRegistration:
    def test_ลงทะเบียนคนใหม่ด้วยตนเอง(self, session, store):
        speaker = session.register_speaker("มะลิ", pcm=b"\x22\x33\x44" * 60, gender="female")

        assert speaker.particle == "ค่ะ"
        assert store.has_voiceprint(speaker.id, "fake")

    def test_ลงทะเบียนชื่อซ้ำคืนคนเดิม(self, session, store):
        first = session.register_speaker("มะลิ")
        second = session.register_speaker("มะลิ")
        assert first.id == second.id
        assert len(store.list_speakers()) == 1
