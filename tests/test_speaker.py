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
            ("ผมชื่อ ต้น ครับ", "ต้น"),
            ("My name is Alex", "Alex"),
            # ชื่อที่ขึ้นต้นเหมือนคำนำหน้า ต้องไม่ถูกตัดหัวทิ้ง
            ("ผมชื่ออาทิตย์ครับ", "อาทิตย์"),
            ("ดิฉันชื่ออารีย์ค่ะ", "อารีย์"),
            ("ผมชื่อคุณากรครับ", "คุณากร"),
            ("เรียกผมว่าพี่เดช", "เดช"),
            ("ชื่อผมสมชายครับ", "สมชาย"),
            ("ชื่อจริงว่าอนุชาครับ", "อนุชา"),
            ("ผมคือสมชายครับ", "สมชาย"),
        ],
    )
    def test_ดึงชื่อจากประโยคแนะนำตัว(self, utterance, expected):
        assert extract_name_claim(utterance) == expected

    @pytest.mark.parametrize(
        "utterance",
        [
            # กฎที่เดาว่า "คำระหว่างสรรพนามกับคำลงท้ายคือชื่อ" ถูกถอดออกแล้ว
            # เพราะมันจับประโยคธรรมดานับไม่ถ้วนเป็นชื่อ แล้วสร้างตัวตนผิด
            # พร้อมผูกลายเสียงของผู้ใช้เข้ากับตัวตนนั้นถาวร
            "ผมหิวครับ",
            "ผมสบายดีครับ",
            "ผมโอเคครับ",
            "ผมเหนื่อยครับ",
            "หนูง่วงค่ะ",
            "ผมขอโทษครับ",
            "ผมป่วยครับ",
            "หนูเบื่อค่ะ",
            "ผมยุ่งอยู่ครับ",
            "เราพร้อมครับ",
            "ผมกลับก่อนนะครับ",
            "นี่เอกสารนะครับ",
            "นี่ปัญหานะ",
            "นี่ราคานะครับ",
        ],
    )
    def test_ประโยคธรรมดาต้องไม่กลายเป็นชื่อ(self, utterance):
        assert extract_name_claim(utterance) is None

    @pytest.mark.parametrize(
        "answer,expected",
        [
            ("เดช", "เดช"),
            ("เดชครับ", "เดช"),
            ("แนนค่ะ", "แนน"),
            ("อาทิตย์ครับ", "อาทิตย์"),
        ],
    )
    def test_ตอบชื่อเปล่าได้เมื่อบอทเพิ่งถามชื่อ(self, answer, expected):
        """เดาว่าเป็นชื่อได้ปลอดภัย เพราะเรารู้ว่าเราเพิ่งถามอะไรไป"""
        assert extract_name_claim(answer, expecting_name=True) == expected
        assert extract_name_claim(answer) is None, "นอกบริบทนั้นต้องไม่เดา"

    @pytest.mark.parametrize(
        "answer",
        ["ไม่บอก", "ขอไม่บอกได้ไหม", "เรียกว่าอะไรก็ได้", "ผมไม่อยากบอกชื่อหรอกครับเพราะมันยาว"],
    )
    def test_เลี่ยงไม่บอกชื่อต้องไม่ถูกตีความเป็นชื่อ(self, answer):
        assert extract_name_claim(answer, expecting_name=True) is None

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


class Testชื่อเล่นสองตัวอักษร:
    """โอ มด นก ปอ เอ บี เจ กบ เป็นชื่อเล่นไทยที่พบบ่อยที่สุดกลุ่มหนึ่ง

    ของเดิมกัน "นะ" ด้วยความยาวขั้นต่ำ 3 ทำให้ "ผมชื่อโอนะครับ" ได้ "โอนะ"
    แล้วบอทเรียกเขาว่า "คุณโอนะ" ไปตลอด และเป็นคนละตัวตนกับ "ผมชื่อโอครับ"
    """

    @pytest.mark.parametrize(
        "phrase,name",
        [
            ("ผมชื่อโอนะครับ", "โอ"),
            ("หนูชื่อมดนะคะ", "มด"),
            ("ผมชื่อนกนะครับ", "นก"),
            ("เรียกผมว่าปอนะครับ", "ปอ"),
            ("ชื่อบีนะคะ", "บี"),
            ("ผมชื่อกบนะครับ", "กบ"),
        ],
    )
    def test_ตัดนะออกจากชื่อสองตัวอักษร(self, phrase, name):
        assert extract_name_claim(phrase) == name

    def test_ชื่อที่ลงท้ายด้วยนะจริงต้องไม่ถูกตัด(self):
        assert extract_name_claim("ผมชื่อมานะครับ") == "มานะ"

    def test_ชื่อยาวยังทำงานเหมือนเดิม(self):
        assert extract_name_claim("ผมชื่อสมชายนะครับ") == "สมชาย"


class Testคำตอบสั้นตอนบอทถามชื่อ:
    """``expecting_name`` เป็นบริบทเดียวที่ยอมรับคำเปล่า ๆ เป็นชื่อ

    การเช็คแค่ "ไม่อยู่ในบัญชีดำ" หลวมเกินไป — ตัวเลข วันเดือนปี และคำ
    อุทานผ่านหมด แล้วกลายเป็นตัวตนถาวรของผู้ใช้
    """

    @pytest.mark.parametrize(
        "answer",
        ["35", "3.5", "5555", "ก.พ.", "อายุ 30", "ทำไมต้องรู้", "เดี๋ยวก่อน",
         "กำลังขับรถ", "หิวข้าว", "hello", "หมอ", "กรุงเทพ", "ครับ", "ไม่บอก"],
    )
    def test_answerที่ไม่ใช่ชื่อต้องถูกปฏิเสธ(self, answer):
        assert extract_name_claim(answer, expecting_name=True) is None

    @pytest.mark.parametrize(
        "answer,name",
        [("เดชครับ", "เดช"), ("นก", "นก"), ("โอค่ะ", "โอ"), ("สมชาย", "สมชาย")],
    )
    def test_ชื่อจริงยังผ่าน(self, answer, name):
        assert extract_name_claim(answer, expecting_name=True) == name


class Testคนชื่อซ้ำกัน:
    """การแยกตัวตนผิดแพงกว่าการรวมผิดมาก

    รวมผิดแก้ได้ด้วยการพูดใหม่ แต่แยกผิดคือความจำหายทั้งก้อน แถวใหม่ได้กุญแจ
    แยกจึงหาไม่เจออีกเลย และตัวอย่างเสียงที่ต่ำกว่าเกณฑ์นิดเดียวเกิดได้ง่ายมาก
    (เป็นหวัด ไมค์คนละตัว อยู่ในที่เสียงดัง)
    """

    @staticmethod
    def _มุม(องศา: float) -> list[float]:
        import math

        return [math.cos(math.radians(องศา)), math.sin(math.radians(องศา))]

    def _ระบบ(self, store, คู่):
        return SpeakerIdentifier(store, FakeEmbedder(table=คู่))

    def test_เสียงเปลี่ยนไปนิดหน่อยต้องไม่แยกเป็นคนใหม่(self, store):
        # cos ≈ 0.64 — ต่ำกว่าเกณฑ์จำเสียงแต่ยังสูงกว่าเกณฑ์แยกตัวตน
        ระบบ = self._ระบบ(store, {b"normal": self._มุม(0), b"cold": self._มุม(50)})
        first = ระบบ.resolve(b"normal", 16000, "ผมชื่อสมชายครับ")
        store.upsert_fact(first.speaker.id, "เงินเดือน", "80000")

        again = ระบบ.resolve(b"cold", 16000, "ผมชื่อสมชายครับ")

        assert again.speaker.id == first.speaker.id
        assert store.facts_for(again.speaker.id), "ความจำต้องยังอยู่"

    def test_เสียงเปลี่ยนคาบเส้นต้องไม่เอาไปสะสมเป็นลายเสียง(self, store):
        ระบบ = self._ระบบ(store, {b"normal": self._มุม(0), b"cold": self._มุม(50)})
        first = ระบบ.resolve(b"normal", 16000, "ผมชื่อสมชายครับ")
        เดิม = store.voiceprint_for(first.speaker.id, "fake")

        ระบบ.resolve(b"cold", 16000, "ผมชื่อสมชายครับ")

        assert store.voiceprint_for(first.speaker.id, "fake") == เดิม, (
            "ถ้าเผลอเป็นคนละคนจริง การสะสมจะทำให้ลายเสียงเพี้ยนไปทั้งคู่"
        )

    def test_คนแปลกหน้าที่ชื่อซ้ำต้องเป็นคนละตัวตน(self, store):
        ระบบ = self._ระบบ(store, {b"a": self._มุม(0), b"b": self._มุม(89)})
        first = ระบบ.resolve(b"a", 16000, "ผมชื่อสมชายครับ")
        store.upsert_fact(first.speaker.id, "โรคประจำตัว", "เบาหวาน")

        second = ระบบ.resolve(b"b", 16000, "ผมชื่อสมชายครับ")

        assert second.speaker.id != first.speaker.id
        assert store.facts_for(second.speaker.id) == []

    def test_คนที่สองต้องไม่ถูกสร้างซ้ำทุกครั้งที่กลับมา(self, store):
        """ของเดิมเทียบกับแถวเดียว จึงสะสมแถวใหม่ไปเรื่อย ๆ ทุกวันที่กลับมาคุย"""
        ระบบ = self._ระบบ(store, {b"a": self._มุม(0), b"b": self._มุม(89)})
        ระบบ.resolve(b"a", 16000, "ผมชื่อสมชายครับ")
        second = ระบบ.resolve(b"b", 16000, "ผมชื่อสมชายครับ")

        for _ in range(5):
            กลับมา = ระบบ.resolve(b"b", 16000, "ผมชื่อสมชายครับ")
            assert กลับมา.speaker.id == second.speaker.id

        assert len(store.find_speakers_by_name("สมชาย")) == 2
