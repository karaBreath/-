"""ทดสอบการจดจำว่า "ใครกำลังพูด" """

import pytest
from conftest import FakeEmbedder

from thaivoice.memory import MemoryStore
from thaivoice.speaker import SpeakerIdentifier, cosine_similarity, extract_name_claim


class TestNameClaim:
    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("ผมชื่อสมชายครับ", "สมชาย"),
            ("หนูชื่อแนนค่ะ", "แนน"),
            ("ฉันชื่อว่ามะลิจ้า", "มะลิ"),
            ("เรียกว่าโบทก็ได้", "โบท"),
            ("นี่ต้นเองนะ", "ต้น"),
            ("ผมชื่อ ต้น ครับ", "ต้น"),
            ("My name is Alex", "Alex"),
        ],
    )
    def test_ดึงชื่อจากประโยคแนะนำตัว(self, utterance, expected):
        assert extract_name_claim(utterance) == expected

    @pytest.mark.parametrize(
        "utterance",
        ["อยากกินอะไรดี", "วันนี้อากาศเป็นยังไงบ้าง", "", "ชื่ออะไร"],
    )
    def test_ไม่เจอชื่อในประโยคทั่วไป(self, utterance):
        assert extract_name_claim(utterance) is None


class TestCosine:
    def test_ค่าพื้นฐาน(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_อินพุตไม่ถูกต้องคืนศูนย์แทนที่จะพัง(self):
        assert cosine_similarity([], [1]) == 0.0
        assert cosine_similarity([1, 2], [1]) == 0.0
        assert cosine_similarity([0, 0], [1, 1]) == 0.0


class TestIdentify:
    @pytest.fixture
    def identifier(self, store: MemoryStore) -> SpeakerIdentifier:
        return SpeakerIdentifier(store, FakeEmbedder(), threshold=0.9, margin=0.02)

    def test_จำเสียงคนเดิมได้(self, store: MemoryStore, identifier: SpeakerIdentifier):
        speaker = store.create_speaker("เดช")
        voice = b"\xff\x00\x00" * 100
        identifier.enroll(speaker, voice, 16000)

        result = identifier.identify(voice, 16000)
        assert result.confident
        assert result.speaker is not None and result.speaker.id == speaker.id
        assert result.method == "voice"

    def test_เสียงที่ไม่รู้จักถือเป็นคนใหม่(self, store: MemoryStore, identifier: SpeakerIdentifier):
        speaker = store.create_speaker("เดช")
        identifier.enroll(speaker, b"\xff\x00\x00" * 100, 16000)

        result = identifier.identify(b"\x00\x00\xff" * 100, 16000)
        assert result.is_new
        assert result.speaker is None

    def test_ยังไม่มีใครในระบบ(self, identifier: SpeakerIdentifier):
        result = identifier.identify(b"\x10\x20\x30" * 100, 16000)
        assert result.is_new and result.speaker is None

    def test_เสียงคล้ายกันสองคนต้องไม่สลับตัว(self, store: MemoryStore):
        """ถ้าอันดับหนึ่งกับอันดับสองคะแนนใกล้กันเกินไป ให้ถือว่าไม่มั่นใจ"""
        identifier = SpeakerIdentifier(store, FakeEmbedder(), threshold=0.5, margin=0.5)
        a = store.create_speaker("เอ")
        b = store.create_speaker("บี")
        identifier.enroll(a, b"\x10\x10\x10" * 50, 16000)
        identifier.enroll(b, b"\x11\x11\x11" * 50, 16000)

        result = identifier.identify(b"\x10\x10\x11" * 50, 16000)
        assert result.is_new, "คะแนนใกล้กันเกินไป ไม่ควรฟันธง"


class TestResolve:
    def test_รู้จักจากชื่อที่บอกเมื่อจำเสียงไม่ได้(self, store: MemoryStore):
        identifier = SpeakerIdentifier(store, None)
        result = identifier.resolve(None, 16000, "สวัสดีครับ ผมชื่อเดชครับ")

        assert result.speaker is not None
        assert result.speaker.display_name == "เดช"
        assert result.speaker.particle == "ครับ"  # เดาคำลงท้ายจาก "ครับ"
        assert result.method == "name"
        assert result.is_new

    def test_บอกชื่อเดิมซ้ำต้องไม่สร้างคนใหม่(self, store: MemoryStore):
        identifier = SpeakerIdentifier(store, None)
        first = identifier.resolve(None, 16000, "ผมชื่อเดชครับ")
        second = identifier.resolve(None, 16000, "ผมชื่อเดชครับ")

        assert first.speaker.id == second.speaker.id
        assert second.is_new is False
        assert len(store.list_speakers()) == 1

    def test_บอกชื่อแล้วผูกลายเสียงให้อัตโนมัติ(self, store: MemoryStore):
        """ครั้งหน้าต้องจำเสียงได้เองโดยไม่ต้องบอกชื่อซ้ำ"""
        identifier = SpeakerIdentifier(store, FakeEmbedder(), threshold=0.9, margin=0.02)
        voice = b"\x33\x66\x99" * 80

        first = identifier.resolve(voice, 16000, "หนูชื่อแนนค่ะ")
        assert first.speaker is not None

        again = identifier.resolve(voice, 16000, "วันนี้อากาศดีจัง")
        assert again.confident
        assert again.speaker.id == first.speaker.id
        assert again.method == "voice"

    def test_ไม่รู้จักและไม่บอกชื่อ(self, store: MemoryStore):
        identifier = SpeakerIdentifier(store, None)
        result = identifier.resolve(None, 16000, "วันนี้กินอะไรดี")
        assert result.speaker is None and result.is_new

    def test_ปิดการจำเสียงแล้วยังทำงานได้(self, store: MemoryStore):
        identifier = SpeakerIdentifier(store, None)
        assert identifier.enabled is False
        assert identifier.backend_name == "none"
        assert identifier.identify(b"x" * 100, 16000).speaker is None
