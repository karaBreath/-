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


class Testชื่อที่ลงท้ายด้วยนะจริง:
    """ตัวแก้รอบสามตัด "นะ" ออกเสมอโดยมีบัญชีรายชื่อยกเว้นแค่สามชื่อ

    วัฒนะ พัฒนะ ชัยวัฒนะ เป็นชื่อที่พบบ่อย พอถูกตัดเหลือ "วัฒ" "พัฒ" TTS
    จะอ่านว่า "วัด" "พัด" แล้วบอทเรียกเขาแบบนั้นไปตลอด บัญชีรายชื่อขยายตามไม่ทัน
    """

    @pytest.mark.parametrize(
        "phrase,name",
        [
            ("ผมชื่อวัฒนะครับ", "วัฒนะ"),
            ("ผมชื่อพัฒนะครับ", "พัฒนะ"),
            ("ผมชื่อชนะครับ", "ชนะ"),
            ("ผมชื่อธนะครับ", "ธนะ"),
            ("ผมชื่อมานะครับ", "มานะ"),
        ],
    )
    def test_ไม่ตัดนะออกจากชื่อจริง(self, phrase, name):
        assert extract_name_claim(phrase) == name

    @pytest.mark.parametrize(
        "phrase,name",
        [
            ("ผมชื่อโอนะครับ", "โอ"),
            ("หนูชื่อมดนะคะ", "มด"),
            ("ผมชื่อสมชายนะครับ", "สมชาย"),
            ("ผมชื่อเบียร์นะครับ", "เบียร์"),
        ],
    )
    def test_ยังตัดนะที่เป็นคำลงท้ายได้(self, phrase, name):
        assert extract_name_claim(phrase) == name


class Testสรรพนามนำหน้าชื่อ:
    """"ผมเบียร์ครับ" คือคำตอบที่เป็นธรรมชาติที่สุดของคำถาม "เรียกว่าอะไรดีคะ"

    ถ้าไม่ตัดสรรพนามออก ตัวตนของเขาจะชื่อ "ผมเบียร์" ถาวร
    """

    @pytest.mark.parametrize(
        "answer,name",
        [
            ("ผมเบียร์ครับ", "เบียร์"),
            ("หนูมิ้นค่ะ", "มิ้น"),
            ("กระผมสมชายครับ", "สมชาย"),
            ("ดิฉันมาลีค่ะ", "มาลี"),
            ("ผมนายสมศักดิ์ครับ", "สมศักดิ์"),
        ],
    )
    def test_ตัดสรรพนามออกจากคำตอบ(self, answer, name):
        assert extract_name_claim(answer, expecting_name=True) == name

    @pytest.mark.parametrize(
        "answer", ["ผมหิวครับ", "ผมง่วงครับ", "ผมลืมบอกไปครับ", "หนูเหนื่อยค่ะ"]
    )
    def test_สรรพนามต้องไม่บังคำที่ไม่ใช่ชื่อ(self, answer):
        assert extract_name_claim(answer, expecting_name=True) is None


class Testเพศที่ยังไม่รู้:
    """ของเดิมเก็บ "ครับ" ให้ทุกคนที่ยังไม่ได้ลงท้ายอะไร

    แล้ว prompt ก็บอกโมเดลว่า "คู่สนทนาลงท้ายว่าครับ" ทั้งที่เขาไม่เคยพูดคำนั้น
    ผู้หญิงจึงถูกยืนยันว่าเป็นผู้ชาย โมเดลก็เลือกคำเรียกผิด
    """

    def test_ไม่รู้เพศต้องไม่เดาว่าเป็นผู้ชาย(self, store):
        ระบบ = SpeakerIdentifier(store, None)
        result = ระบบ.resolve(None, 16000, "ชื่อแนน")
        assert result.speaker.gender is None
        assert result.speaker.particle is None

    @pytest.mark.parametrize(
        "utterance,gender",
        [
            ("หนูชื่อฝ้าย", "female"),
            ("ดิฉันชื่อวรรณ", "female"),
            ("ผมชื่อเดช", "male"),
            ("หนูชื่อมาลีค่ะ", "female"),
        ],
    )
    def test_สรรพนามบอกเพศได้พอๆกับคำลงท้าย(self, store, utterance, gender):
        ระบบ = SpeakerIdentifier(store, None)
        assert ระบบ.resolve(None, 16000, utterance).speaker.gender == gender


class Testคำนำที่กำกวม:
    """ผ่อนให้ยาวกว่าคำนำได้สามตัวอักษรหลวมเกินไปมาก

    "ชื่อลูกค้า" "ชื่อเพลงนี้" "ชื่อร้านชัย" "ชื่อหมาผม" กลายเป็นชื่อคนหมด
    แล้วลายเสียงของผู้ใช้จะถูกผูกกับตัวตนปลอมนั้นถาวร
    """

    @pytest.mark.parametrize(
        "utterance",
        ["ชื่อลูกค้า", "ชื่อเพลงนี้", "ชื่อร้านชัย", "ชื่อหมาผม", "ชื่อเมนูนี้", "ชื่อเพื่อนผม"],
    )
    def test_ต้องไม่กลายเป็นชื่อคน(self, utterance):
        assert extract_name_claim(utterance) is None

    @pytest.mark.parametrize(
        "utterance,name",
        [
            ("ผมชื่อแมวครับ", "แมว"),
            ("หนูชื่อเพลงค่ะ", "เพลง"),
            ("ผมชื่อลูกน้ำครับ", "ลูกน้ำ"),
            ("ผมชื่อปลาครับ", "ปลา"),
        ],
    )
    def test_ชื่อเล่นที่เป็นคำธรรมดายังใช้ได้(self, utterance, name):
        assert extract_name_claim(utterance) == name


class Testคำตอบที่เป็นประโยคไม่ใช่ชื่อ:
    @pytest.mark.parametrize(
        "answer",
        ["อยู่บ้าน", "ขับรถอยู่", "เพิ่งตื่น", "นั่งเล่น", "สบายดี", "ก็ดีนะ", "แป๊บนึง"],
    )
    def test_ต้องถูกปฏิเสธ(self, answer):
        assert extract_name_claim(answer, expecting_name=True) is None

    @pytest.mark.parametrize("answer", ["O'Brien", "Anne-Marie", "มิ้น"])
    def test_ชื่อที่มีเครื่องหมายในตัวยังผ่าน(self, answer):
        assert extract_name_claim(answer, expecting_name=True) == answer


class Testคำนำหน้าทางวิชาชีพ:
    """"คุณดร.สมชาย" ผิดหลักภาษา และ TTS อ่าน "ดร." ทีละตัวอักษร"""

    @pytest.mark.parametrize(
        "utterance,name",
        [
            ("ผมชื่อดร.สมชายครับ", "สมชาย"),
            ("ผมชื่อผศ.ดร.สมชายครับ", "สมชาย"),
            ("ผมชื่อนพ.สมชายครับ", "สมชาย"),
            ("หนูชื่อพญ.มาลีค่ะ", "มาลี"),
        ],
    )
    def test_ตัดคำนำหน้าออก(self, utterance, name):
        assert extract_name_claim(utterance) == name

    def test_ชื่อที่ขึ้นต้นคล้ายคำนำหน้าต้องไม่ถูกตัด(self):
        assert extract_name_claim("ผมชื่อคุณากรครับ") == "คุณากร"


class Testชื่อเล่นที่ขึ้นต้นด้วยคำกำกวม:
    """ลูกปัด ลูกแพร ลูกพีช เป็นชื่อเล่นผู้หญิงไทยที่พบบ่อยที่สุดกลุ่มหนึ่ง

    กฎเข้มที่เพิ่มรอบสี่ทิ้งพวกเธอทั้งหมด แม้ในประโยคแนะนำตัวที่ชัดเจน
    ผู้ใช้จึงไม่ถูกสร้างตัวตนเลย และบอทจะถามชื่อซ้ำทุกเทิร์น
    """

    @pytest.mark.parametrize(
        "utterance,name",
        [
            ("ดิฉันชื่อลูกปัดค่ะ", "ลูกปัด"),
            ("หนูชื่อลูกแพรค่ะ", "ลูกแพร"),
            ("ดิฉันชื่อยาใจค่ะ", "ยาใจ"),
            ("ผมชื่อวงศ์ครับ", "วงศ์"),
            ("เรียกผมว่าลูกปัดก็ได้", "ลูกปัด"),
        ],
    )
    def test_มีสรรพนามยืนยันแล้วต้องผ่าน(self, utterance, name):
        assert extract_name_claim(utterance) == name

    @pytest.mark.parametrize(
        "utterance", ["ชื่อลูกค้า", "ชื่อเพลงนี้", "ชื่อร้านชัย", "ชื่อหมาผม"]
    )
    def test_ไม่มีสรรพนามยืนยันยังต้องถูกปฏิเสธ(self, utterance):
        assert extract_name_claim(utterance) is None

    def test_ตอนบอทเพิ่งถามชื่อก็ไม่ต้องเข้ม(self):
        assert extract_name_claim("ลูกปัดค่ะ", expecting_name=True) == "ลูกปัด"

    @pytest.mark.parametrize(
        "answer,name",
        [("มานะครับ", "มานะ"), ("ปรีดีครับ", "ปรีดี"), ("รอฮีมครับ", "รอฮีม")],
    )
    def test_ชื่อจริงต้องไม่ถูกกฎท้ายคำโยนทิ้ง(self, answer, name):
        """_clean_name อุตส่าห์เก็บ "มานะ" ไว้ แล้วกฎท้ายคำมาโยนทิ้ง"""
        assert extract_name_claim(answer, expecting_name=True) == name

    def test_ณัฐต้องไม่ติดนะ(self):
        """"ณัฐ" "รัฐ" พบบ่อยกว่า "วัฒนะ" มาก"""
        assert extract_name_claim("ผมชื่อณัฐนะครับ") == "ณัฐ"
        assert extract_name_claim("ผมชื่อวัฒนะครับ") == "วัฒนะ"


class Testสำนวนอธิบายไม่ใช่การแนะนำตัว:
    """"เรียกว่า" เป็นสำนวนอธิบายที่ใช้บ่อยที่สุดสำนวนหนึ่งในภาษาไทย

    รอบห้าผ่อนการกันคำกำกวมให้ทุกกฎที่ไม่ใช่กฎ "ชื่อX" เปล่า ๆ ซึ่งรวมกฎ
    "เรียกว่า" ที่ไม่มีสรรพนามยืนยันเข้าไปด้วย
    """

    @pytest.mark.parametrize(
        "utterance",
        [
            "โรคนี้เรียกว่าโรคเบาหวานครับ",
            "เพลงนี้เรียกว่าเพลงลูกทุ่งครับ",
            "ที่นี่เรียกว่าถนนข้าวสารครับ",
        ],
    )
    def test_ต้องไม่กลายเป็นชื่อคน(self, utterance):
        assert extract_name_claim(utterance) is None

    @pytest.mark.parametrize(
        "utterance,name",
        [
            ("เรียกผมว่าเดชก็ได้", "เดช"),
            ("เรียกผมว่าลูกปัดก็ได้", "ลูกปัด"),
            ("เรียกว่าโบทครับ", "โบท"),
            ("เรียกผมว่าอาทิตย์ก็ได้ครับ", "อาทิตย์"),
        ],
    )
    def test_การแนะนำตัวจริงยังผ่าน(self, utterance, name):
        assert extract_name_claim(utterance) == name


class Testคำบอกลาไม่ใช่ชื่อ:
    """รอบห้าถอด "นะ" ออกจากบัญชีคำท้ายเพื่อกู้ "มานะ" แล้วเปิดประตูให้
    ทุกคำตอบสั้นที่ลงท้ายแบบนั้นกลายเป็นชื่อ"""

    @pytest.mark.parametrize(
        "answer", ["ไว้ก่อนนะ", "โชคดีนะ", "พรุ่งนี้นะ", "สักครู่นะ", "ไปก่อนนะ"]
    )
    def test_ต้องถูกปฏิเสธ(self, answer):
        assert extract_name_claim(answer, expecting_name=True) is None

    @pytest.mark.parametrize("answer", ["มานะ", "ปรีดี", "ณัฐ", "ลูกปัด", "เดช"])
    def test_ชื่อจริงยังผ่าน(self, answer):
        assert extract_name_claim(answer, expecting_name=True) == answer

    def test_ทางลัดของตัวตัดคำนำหน้าต้องไม่ข้ามด่านตรวจ(self):
        """`return None if "." in title else name` ข้ามการตรวจความยาวและคำต้องห้าม"""
        assert extract_name_claim("คุณาไม่ว่างค่ะ", expecting_name=True) is None


class Testชื่อยาวและยศ:
    """เพดานการจับต้องเผื่อคำลงท้ายที่ติดมาด้วย

    ไม่งั้นชื่อยาว 12-15 ตัวอักษรถูกตัดกลางคำลงท้าย ตัวตัดคำลงท้ายจำไม่ได้
    แล้วเศษที่เหลืออ่านออกเสียงไม่ได้เลย ("กิตติศักดิ์นะคร")
    """

    @pytest.mark.parametrize(
        "utterance,name",
        [
            ("ผมชื่อกิตติศักดิ์นะครับ", "กิตติศักดิ์"),
            ("ผมชื่อสมชายแซ่ลิ้มครับ", "สมชายแซ่ลิ้ม"),
            ("ผมชื่อพงศ์พัฒน์ชัยครับ", "พงศ์พัฒน์ชัย"),
        ],
    )
    def test_ชื่อยาวต้องไม่ถูกตัดกลางคำลงท้าย(self, utterance, name):
        assert extract_name_claim(utterance) == name

    def test_ยศทหารตำรวจต้องถูกตัดออก(self):
        """คำนำหน้าแทนที่ "คุณ" ไม่ใช่ซ้อนใต้มัน — "คุณพล.ต.ต.วิชัย" ผิดหลักภาษา"""
        assert extract_name_claim("ผมชื่อ พล.ต.ต.วิชัย ครับ") == "วิชัย"
        assert extract_name_claim("ผมชื่ออาจารย์สมพงษ์ครับ") == "สมพงษ์"

    def test_ยศล้วนๆไม่ใช่ชื่อ(self):
        """ช่องว่างคั่นทำให้กฎจับได้แค่ยศ แล้วบอทเรียกเขาว่า "คุณร.ต.อ" """
        assert extract_name_claim("ผมชื่อ ร.ต.อ. สมศักดิ์ ครับ") is None

    @pytest.mark.parametrize("utterance", ["ผมชื่อหน่อยครับ", "หนูชื่อหน่อยค่ะ"])
    def test_หน่อยเป็นชื่อเล่นไทยที่ใช้จริง(self, utterance):
        assert extract_name_claim(utterance) == "หน่อย"

    @pytest.mark.parametrize("utterance", ["ขอชื่อหน่อยครับ", "ช่วยบอกหน่อย"])
    def test_คำขอที่ลงท้ายด้วยหน่อยยังไม่ใช่ชื่อ(self, utterance):
        assert extract_name_claim(utterance) is None


class Testกฎที่มีสรรพนามยืนยัน:
    """ตอนเพิ่มกฎจาก 8 เป็น 10 ข้อ ดัชนีชุด "มีสรรพนามยืนยัน" ไม่ได้ถูกปรับตาม

    กฎ "ชื่อเล่นชื่อX" จึงกลายเป็นแบบเข้ม แล้วชื่อเล่นที่ขึ้นต้นด้วยคำกำกวม
    ถูกปฏิเสธหมด ผู้ใช้กลุ่มนั้นไม่ถูกสร้างตัวตนเลย
    """

    @pytest.mark.parametrize(
        "utterance,name",
        [
            ("ชื่อเล่นลูกปัดค่ะ", "ลูกปัด"),
            ("ชื่อเล่นว่าลูกปัดค่ะ", "ลูกปัด"),
            ("ชื่อเล่นเพลงพิณค่ะ", "เพลงพิณ"),
            ("ชื่อเล่นแมวเหมียวค่ะ", "แมวเหมียว"),
            ("เรียกหนูว่าลูกปัดก็ได้ค่ะ", "ลูกปัด"),
        ],
    )
    def test_การแนะนำชื่อเล่นต้องไม่ถูกกันด้วยคำกำกวม(self, utterance, name):
        assert extract_name_claim(utterance) == name

    @pytest.mark.parametrize(
        "utterance",
        ["ชื่อลูกค้า", "โรคนี้เรียกว่าโรคเบาหวานครับ", "เพลงนี้เรียกว่าเพลงลูกทุ่งครับ"],
    )
    def test_สำนวนอธิบายยังถูกกันอยู่(self, utterance):
        assert extract_name_claim(utterance) is None


class Testสำนวนที่หน้าตาเหมือนการบอกชื่อ:
    """ประโยคไทยธรรมดาต้องไม่กลายเป็นตัวตนใหม่

    การสร้างตัวตนปลอมแพงมาก — ลายเสียงของผู้ใช้ถูกผูกกับตัวตนนั้นถาวร
    ความจำจริงของเขากำพร้า และครั้งหน้าบอทจะทักเขาด้วยชื่อที่ไม่ใช่ชื่อเขา
    """

    @pytest.mark.parametrize(
        "text",
        [
            # สำนวน "เรียก <คน> มา" — สั่งให้ใครมา ไม่ใช่การบอกชื่อ
            "เรียกเพื่อนมาก็ได้ครับ",
            "เรียกช่างมาก็ได้ครับ",
            "เรียกหมอมาก็ได้",
            "เรียกแท็กซี่มาก็ได้",
            "เรียกตำรวจมาก็ได้ครับ",
            "เรียกน้องมาช่วยก็ได้ครับ",
            "เรียกคนมาซ่อมก็ได้",
            "เรียกรถมารับก็ได้ครับ",
            "เรียกทีมมาช่วยก็ได้ครับ",
            # วลีขยายที่จับเกินขอบชื่อ
            "ชื่อเล่นลูกชายผมครับ",
            "ชื่อเล่นแมวบ้านผม",
            "ชื่อเล่นน้องชายผมครับ",
            "ชื่อเล่นร้านนี้ครับ",
            "ชื่อเล่นเพลงนี้ครับ",
            "ชื่อเล่นหมาตัวนี้",
        ],
    )
    def test_ต้องไม่ถูกอ่านเป็นชื่อ(self, text):
        assert extract_name_claim(text) is None, text

    @pytest.mark.parametrize(
        "text,expected",
        [
            # ชื่อเล่นจริงที่ขึ้นต้นด้วยคำกำกวม ต้องยังใช้ได้
            ("ชื่อเล่นลูกปัดค่ะ", "ลูกปัด"),
            ("ชื่อเล่นแมวเหมียวค่ะ", "แมวเหมียว"),
            ("ชื่อเล่นเพลงพิณค่ะ", "เพลงพิณ"),
            ("เรียกหนูว่าลูกปัดก็ได้ค่ะ", "ลูกปัด"),
            ("เรียกผมว่าเดชก็ได้ครับ", "เดช"),
            ("เรียกหนูว่าลูกปัดก็ได้ค่ะ", "ลูกปัด"),
            # ชื่อที่มีพยางค์ "มา"/"ไป" อยู่จริง ต้องไม่ถูกด่านกันสำนวนเผลอตัดทิ้ง
            ("ผมชื่อมาลีครับ", "มาลี"),
            ("หนูชื่ออุมาค่ะ", "อุมา"),
            ("ดิฉันชื่อปัทมาค่ะ", "ปัทมา"),
            ("หนูชื่อไปรยาค่ะ", "ไปรยา"),
            # ชื่อที่ขึ้นต้นด้วยพยางค์เดียวกับสรรพนาม
            ("ผมชื่อเขมิกาครับ", "เขมิกา"),
        ],
    )
    def test_การแนะนำตัวจริงต้องยังใช้ได้(self, text, expected):
        assert extract_name_claim(text) == expected


class Testกฎเรียกแบบเปล่าถูกถอดออก:
    """"เรียก X ก็ได้" (ไม่มีสรรพนาม ไม่มี "ว่า") กลืนประโยคธรรมดาเป็นชื่อคน

    "เรียก" เป็นคำกริยาที่ใช้บ่อยมาก เคยพยายามกันด้วยการปฏิเสธชื่อที่มีพยางค์
    "มา" (สำนวน "เรียก X มา") ซึ่งล้มเหลว — แยก "เรียกช่างมา" ออกจาก "เรียกอุมา"
    ไม่ได้ถ้าไม่มีพจนานุกรมชื่อคน
    """

    @pytest.mark.parametrize(
        "text",
        [
            "เรียกช่างก็ได้ครับ",
            "เรียกแท็กซี่ก็ได้ครับ",
            "เรียกประชุมก็ได้ครับ",
            "เรียกเก็บเงินก็ได้ครับ",
            "เรียกรถก็ได้ครับ",
            "เรียกแม่ก็ได้ครับ",
            "เรียกลิฟต์ก็ได้ครับ",
            "เรียกเพื่อนมาก็ได้ครับ",
            "เรียกช่างมาก็ได้ครับ",
            "เรียกน้องมาช่วยก็ได้ครับ",
        ],
    )
    def test_ประโยคธรรมดาต้องไม่กลายเป็นชื่อ(self, text):
        assert extract_name_claim(text) is None, text

    @pytest.mark.parametrize(
        "name",
        [
            # ชื่อไทยจริงที่มีพยางค์ "มา" — ตัวกันแบบเดิมปฏิเสธทั้งหมด
            "อุมา", "ปัทมา", "ศุภมาส", "สมหมาย", "ชุติมา", "สุมาลี",
            "สมาน", "วิมาลา", "มาลี", "ไปรยา", "อุมาพร", "ธัญมาส",
        ],
    )
    def test_ชื่อไทยที่มีพยางค์มาต้องใช้ได้ทุกรูป(self, name):
        assert extract_name_claim(f"หนูชื่อ{name}ค่ะ") == name
        assert extract_name_claim(f"เรียกหนูว่า{name}ก็ได้ค่ะ") == name
        assert extract_name_claim(f"เรียก{name}ก็ได้ค่ะ", expecting_name=True) == name

    def test_ทางถอยต้องไม่สร้างชื่อที่ขึ้นต้นด้วยเรียก(self):
        """เมื่อกฎทั่วไปไม่รับ ทางถอยเคยแปลทั้งประโยคเป็นชื่อ "เรียกสมหมาย"

        แล้วผูกกับลายเสียงถาวร บอทเรียกเขาว่า "คุณเรียกสมหมาย" ตลอดไป
        """
        for text in ["เรียกสมหมายก็ได้ครับ", "เรียกอุมาก็ได้ค่ะ"]:
            for expecting in (True, False):
                got = extract_name_claim(text, expecting_name=expecting)
                assert got is None or not got.startswith("เรียก"), (text, got)

    @pytest.mark.parametrize(
        "text",
        [
            "เพื่อนผมชื่อเล่นว่าสมชายครับ",
            "แมวผมชื่อเล่นว่าเหมียวครับ",
            "ร้านนี้ชื่อเล่นว่าเจ๊หมวยครับ",
            "ลูกผมชื่อเล่นว่าปุ๊กครับ",
        ],
    )
    def test_ชื่อเล่นของคนอื่นต้องไม่กลายเป็นตัวตนผู้พูด(self, text):
        """กฎ "ชื่อเล่น…" เคยไม่มีขอบซ้าย ต่างจากกฎอื่นทุกข้อ"""
        assert extract_name_claim(text) is None, text

    def test_ชื่อเล่นของตัวเองต้องยังใช้ได้(self):
        assert extract_name_claim("ชื่อเล่นลูกปัดค่ะ") == "ลูกปัด"
        assert extract_name_claim("หนูชื่อเล่นว่าลูกปัดค่ะ") == "ลูกปัด"


class Testทางถอยตอนบอทถามชื่อต้องไม่รับคำสั่ง:
    """ไม่ใช่ทุกคำตอบหลังคำถาม "ชื่ออะไร" จะเป็นชื่อ

    การตัดคำกริยา "เรียก" ทิ้งเฉย ๆ ทำให้ประโยคที่ยาวเกินเพดานหดลงมาผ่านด่านได้
    """

    @pytest.mark.parametrize(
        "text",
        [
            "เรียกช่างมาหน่อยครับ",
            "เรียกแท็กซี่ให้หน่อย",
            "เรียกรถพยาบาลด่วน",
            "เรียกพยาบาลหน่อยค่ะ",
            "เรียกลิฟต์ให้หน่อย",
            "เรียกเก็บเงินหน่อย",
            "เรียกน้องมาช่วยก็ได้ครับ",
        ],
    )
    def test_คำสั่งต้องไม่กลายเป็นชื่อแม้ตอนบอทเพิ่งถามชื่อ(self, text):
        assert extract_name_claim(text, expecting_name=True) is None, text

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("เรียกลูกปัดก็ได้ค่ะ", "ลูกปัด"),
            ("เรียกเดชก็ได้ครับ", "เดช"),
            ("เรียกอุมาก็ได้ค่ะ", "อุมา"),
            ("เรียกว่าลูกปัดก็ได้ค่ะ", "ลูกปัด"),
            ("ลูกปัดค่ะ", "ลูกปัด"),
        ],
    )
    def test_คำตอบที่เป็นชื่อจริงต้องยังใช้ได้(self, text, expected):
        assert extract_name_claim(text, expecting_name=True) == expected

    def test_ลำดับของคำกริยาที่ตัดต้องลองคำยาวก่อน(self):
        """ถ้าลอง "เรียก" ก่อน "เรียกว่า" จะเหลือ "ว่าลูกปัด" """
        assert extract_name_claim("เรียกว่าลูกปัดก็ได้ค่ะ", expecting_name=True) == "ลูกปัด"


class Testสำนวนอธิบายต้องไม่แย่งตัวตน:
    """"เรียกว่า" เป็นสำนวนอธิบายที่ใช้บ่อยที่สุดสำนวนหนึ่งในภาษาไทย

    ตัวอย่างในคอมเมนต์เดิม ("โรคนี้เรียกว่าโรคเบาหวาน") รอดเพราะบังเอิญ "โรค"
    อยู่ในบัญชีคำกำกวมเท่านั้น คำนามอื่นทะลุหมด
    """

    @pytest.mark.parametrize(
        "text",
        [
            "ยาตัวนี้เรียกว่าพาราเซตามอลครับ",
            "อาหารจานนี้เรียกว่าผัดกะเพราครับ",
            "ปรากฏการณ์นี้เรียกว่าเอลนีโญครับ",
            "อาการนี้เรียกว่าไมเกรนครับ",
            "ตำแหน่งนี้เรียกว่าซีทีโอครับ",
            # รูปที่ลงท้ายด้วย "ก็ได้" เข้ากฎคนละข้อ ต้องมีขอบซ้ายเหมือนกัน
            "ยาตัวนี้เรียกว่าพาราเซตามอลก็ได้ครับ",
            "อาการนี้เรียกว่าไมเกรนก็ได้ครับ",
            "สภาพนี้เรียกว่าโอเวอร์โหลดก็ได้",
        ],
    )
    def test_ประโยคอธิบายต้องไม่กลายเป็นชื่อ(self, text):
        assert extract_name_claim(text) is None, text

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("สวัสดีครับ เรียกว่าเปิ้ลก็ได้", "เปิ้ล"),
            ("เรียกหนูว่าลูกปัดก็ได้ค่ะ", "ลูกปัด"),
        ],
    )
    def test_การแนะนำตัวจริงต้องยังใช้ได้(self, text, expected):
        assert extract_name_claim(text) == expected


class Testคำขอต้องไม่กว้างจนกินชื่อจริง:
    """การเทียบแบบสตริงย่อยทำให้คำสั้นอย่าง "ที" กินพยางค์ของชื่อไทย

    เป็นความผิดพลาดแบบเดียวกับที่เคยเกิดตอนใส่ "มา" ลงบัญชีคำต้องห้าม
    """

    @pytest.mark.parametrize(
        "name", ["ประทีป", "ทีปกร", "ปทีป", "ทีน่า", "มาร์ที", "วีระ", "สาธิต"]
    )
    def test_ชื่อที่มีพยางค์เดียวกับคำขอต้องใช้ได้(self, name):
        assert extract_name_claim(f"เรียก{name}ก็ได้ครับ", expecting_name=True) == name
        assert extract_name_claim(f"หนูชื่อ{name}ค่ะ") == name

    @pytest.mark.parametrize(
        "text",
        [
            "เรียกช่างมาหน่อยครับ",
            "เรียกแท็กซี่ให้หน่อย",
            "เรียกรถพยาบาลด่วน",
            "เรียกช่างมาซิ",
            "เรียกหมอมาช่วยที",
        ],
    )
    def test_คำสั่งต้องยังถูกปฏิเสธ(self, text):
        assert extract_name_claim(text, expecting_name=True) is None, text
