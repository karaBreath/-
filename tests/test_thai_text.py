"""ทดสอบยูทิลิตี้ภาษาไทย — ส่วนที่กฎภาษาไทยอยู่ตรงนี้ทั้งหมด"""

from thaivoice.thai_text import (
    SpeechChunker,
    clean_for_speech,
    detect_particle,
    normalize_transcript,
    particle_for_gender,
    read_digits,
    split_sentences,
    thai_number_to_words,
    thai_ratio,
)


class TestCleanForSpeech:
    def test_ถอด_markdown_ที่อ่านออกเสียงไม่ได้(self):
        assert clean_for_speech("**สวัสดี**") == "สวัสดี"
        assert clean_for_speech("# หัวข้อใหญ่") == "หัวข้อใหญ่"
        assert clean_for_speech("- ข้อหนึ่ง") == "ข้อหนึ่ง"
        assert clean_for_speech("1. ข้อแรก") == "ข้อแรก"

    def test_แทนโค้ดบล็อกด้วยคำอธิบายสั้น(self):
        spoken = clean_for_speech("ลองดูนี่\n```python\nprint(1)\n```")
        assert "print" not in spoken
        assert "โค้ด" in spoken

    def test_เก็บเฉพาะข้อความของลิงก์(self):
        assert clean_for_speech("[เว็บไซต์](https://example.com)") == "เว็บไซต์"
        assert "example.com" not in clean_for_speech("ดูที่ https://example.com นะ")

    def test_แปลงหน่วยและเลขไทย(self):
        assert "เปอร์เซ็นต์" in clean_for_speech("50%")
        assert "กิโลเมตร" in clean_for_speech("ระยะ 5 km")
        assert clean_for_speech("๑๒๓") == "123"

    def test_ตัดอิโมจิออก(self):
        assert clean_for_speech("ดีใจจัง 😀🎉") == "ดีใจจัง"

    def test_ข้อความว่างไม่พัง(self):
        assert clean_for_speech("") == ""
        assert clean_for_speech("   ") == ""


class TestNumbers:
    def test_คำอ่านตัวเลขพื้นฐาน(self):
        assert thai_number_to_words(0) == "ศูนย์"
        assert thai_number_to_words(5) == "ห้า"
        assert thai_number_to_words(10) == "สิบ"

    def test_กฎ_เอ็ด_และ_ยี่สิบ(self):
        # ภาษาไทยอ่าน 11 ว่า "สิบเอ็ด" ไม่ใช่ "สิบหนึ่ง" และ 20 ว่า "ยี่สิบ"
        assert thai_number_to_words(11) == "สิบเอ็ด"
        assert thai_number_to_words(20) == "ยี่สิบ"
        assert thai_number_to_words(21) == "ยี่สิบเอ็ด"
        assert thai_number_to_words(101) == "หนึ่งร้อยเอ็ด"

    def test_หลักใหญ่(self):
        assert thai_number_to_words(2568) == "สองพันห้าร้อยหกสิบแปด"
        assert thai_number_to_words(1_000_000) == "หนึ่งล้าน"
        assert thai_number_to_words(1_000_001) == "หนึ่งล้านหนึ่ง"

    def test_เลขติดลบ(self):
        assert thai_number_to_words(-5) == "ลบห้า"

    def test_อ่านทีละตัวสำหรับเบอร์โทร(self):
        assert read_digits("081-234") == "ศูนย์ แปด หนึ่ง สอง สาม สี่"
        assert read_digits("ไม่มีเลข") == ""


class TestParticles:
    def test_ตรวจเพศจากคำลงท้าย(self):
        assert detect_particle("ผมชื่อสมชายครับ") == "male"
        assert detect_particle("หนูชื่อแนนค่ะ") == "female"
        assert detect_particle("เอาอันนี้") is None

    def test_ดูคำลงท้ายที่อยู่ท้ายสุด(self):
        # "เขาบอกว่าครับ" กลางประโยคไม่ควรชนะคำลงท้ายจริงที่ท้ายสุด
        assert detect_particle("เมื่อกี้เขาพูดว่าครับ แต่หนูว่าไม่ใช่ค่ะ") == "female"

    def test_คำลงท้ายของบอท(self):
        assert particle_for_gender("male") == "ครับ"
        assert particle_for_gender("female") == "ค่ะ"
        assert particle_for_gender(None) == "ครับ"


class TestSentenceSplitting:
    def test_ตัดที่คำลงท้ายสุภาพ(self):
        parts = split_sentences("สวัสดีครับ วันนี้อากาศดีมากเลยนะครับ")
        assert len(parts) >= 1
        assert "สวัสดีครับ" in parts[0]

    def test_ตัดที่ขึ้นบรรทัดใหม่(self):
        parts = split_sentences("บรรทัดแรกยาวพอสมควร\nบรรทัดที่สองก็ยาวพอสมควร")
        assert len(parts) == 2

    def test_ข้อความว่างคืนลิสต์ว่าง(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []


class TestSpeechChunker:
    """ตัวตัดท่อนคือหัวใจของการทำให้เสียงตอบเร็ว — ต้องไม่ทำข้อความหาย"""

    def test_ปล่อยประโยคทันทีที่จบ(self):
        chunker = SpeechChunker()
        chunks = list(chunker.feed("สวัสดีครับ ยินดีที่ได้รู้จักนะครับ "))
        assert chunks
        assert chunks[0].endswith("ครับ")

    def test_ไม่ทำข้อความหายระหว่างตัด(self):
        source = "สวัสดีครับ วันนี้เป็นยังไงบ้างครับ ผมชื่อใจนะครับ ยินดีที่ได้รู้จัก"
        chunker = SpeechChunker()
        collected = []
        for index in range(0, len(source), 5):
            collected.extend(chunker.feed(source[index : index + 5]))
        collected.extend(chunker.flush())
        # เทียบโดยไม่สนช่องว่าง เพราะตัวตัดจะ trim ขอบแต่ละท่อน
        assert "".join(collected).replace(" ", "") == source.replace(" ", "")

    def test_flush_คืนเศษที่ค้างอยู่(self):
        chunker = SpeechChunker()
        list(chunker.feed("สั้น"))
        assert list(chunker.flush()) == ["สั้น"]

    def test_บังคับตัดเมื่อยาวเกินไป(self):
        # ข้อความไทยยาว ๆ ที่ไม่มีเว้นวรรคเลย ต้องไม่ค้างในบัฟเฟอร์ตลอดกาล
        chunker = SpeechChunker(min_chars=10, max_chars=40)
        chunks = list(chunker.feed("ก" * 100))
        assert chunks, "ต้องบังคับตัดเมื่อเกิน max_chars"
        assert all(len(chunk) <= 40 for chunk in chunks)


class TestTranscriptCleanup:
    def test_ตัดคำซ้ำที่ASRหลอน(self):
        assert normalize_transcript("ขอบคุณ ขอบคุณ ขอบคุณ ขอบคุณ") == "ขอบคุณ"

    def test_ยุบช่องว่างและจุดไข่ปลา(self):
        assert normalize_transcript("สวัสดี    ครับ...") == "สวัสดี ครับ"


def test_สัดส่วนอักขระไทย():
    assert thai_ratio("สวัสดี") == 1.0
    assert thai_ratio("hello") == 0.0
    assert thai_ratio("") == 0.0
    assert 0.0 < thai_ratio("hello สวัสดี") < 1.0


class TestSegmenterEngine:
    """เส้นทาง pythainlp เคยล้มแบบเงียบ ๆ เพราะ engine เริ่มต้น (crfcut) ต้องการ
    python-crfsuite ที่ไม่ได้ติดมาด้วย เทสต์ชุดนี้กันไม่ให้กลับไปพังเงียบอีก
    """

    def test_คืนชื่อ_engine_หรือ_None_โดยไม่โยน_error(self):
        from thaivoice.thai_text import _SENT_ENGINES, thai_segmenter_engine

        engine = thai_segmenter_engine(force_probe=True)
        assert engine is None or engine in _SENT_ENGINES

    def test_ผลถูกแคชไว้ไม่ต้องตรวจซ้ำ(self):
        from thaivoice import thai_text

        first = thai_text.thai_segmenter_engine(force_probe=True)
        assert thai_text._sent_engine_probed is True
        assert thai_text.thai_segmenter_engine() == first

    def test_ถอยไปใช้กฎสำรองเมื่อ_pythainlp_ใช้ไม่ได้(self, monkeypatch):
        from thaivoice import thai_text

        monkeypatch.setattr(thai_text, "_sent_engine", None)
        monkeypatch.setattr(thai_text, "_sent_engine_probed", True)

        parts = thai_text.split_sentences("สวัสดีครับ วันนี้อากาศดีมากเลยนะครับ ผมชื่อโบท")
        assert parts, "ต้องยังตัดประโยคได้แม้ไม่มี pythainlp"
        assert "".join(parts).replace(" ", "") == (
            "สวัสดีครับวันนี้อากาศดีมากเลยนะครับผมชื่อโบท"
        )

    def test_engine_ที่พังต้องถูกข้ามไปตัวถัดไป(self, monkeypatch):
        """จำลอง crfcut ที่ขาด dependency — ต้องเลื่อนไปใช้ engine ถัดไป ไม่ใช่ยอมแพ้"""
        import sys
        import types

        from thaivoice import thai_text

        calls = []

        def fake_sent_tokenize(text, engine="crfcut"):
            calls.append(engine)
            if engine == "crfcut":
                raise ModuleNotFoundError("No module named 'pycrfsuite'")
            return [p for p in text.split() if p]

        module = types.ModuleType("pythainlp.tokenize")
        module.sent_tokenize = fake_sent_tokenize
        parent = types.ModuleType("pythainlp")
        parent.tokenize = module
        monkeypatch.setitem(sys.modules, "pythainlp", parent)
        monkeypatch.setitem(sys.modules, "pythainlp.tokenize", module)
        monkeypatch.setattr(thai_text, "_sent_engine", None)
        monkeypatch.setattr(thai_text, "_sent_engine_probed", False)

        engine = thai_text.thai_segmenter_engine(force_probe=True)

        assert calls[0] == "crfcut", "ต้องลอง engine ที่แม่นที่สุดก่อน"
        assert engine == "thaisum", "crfcut พังแล้วต้องเลื่อนไปตัวถัดไป"

    def test_pythainlp_พังทั้งหมดต้องคืน_None(self, monkeypatch):
        import sys
        import types

        from thaivoice import thai_text

        def always_fails(text, engine="crfcut"):
            raise RuntimeError("พังทุก engine")

        module = types.ModuleType("pythainlp.tokenize")
        module.sent_tokenize = always_fails
        parent = types.ModuleType("pythainlp")
        parent.tokenize = module
        monkeypatch.setitem(sys.modules, "pythainlp", parent)
        monkeypatch.setitem(sys.modules, "pythainlp.tokenize", module)
        monkeypatch.setattr(thai_text, "_sent_engine_probed", False)

        assert thai_text.thai_segmenter_engine(force_probe=True) is None
        # ยังต้องตัดประโยคได้ด้วยกฎสำรอง
        assert thai_text.split_sentences("สวัสดีครับ ผมชื่อโบทนะครับ")
