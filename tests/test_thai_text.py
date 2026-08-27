"""ทดสอบยูทิลิตี้ภาษาไทย — ส่วนที่กฎภาษาไทยอยู่ตรงนี้ทั้งหมด"""

import unicodedata

import pytest

from thaivoice.thai_text import (
    SpeechChunker,
    clean_for_speech,
    detect_particle,
    expand_numbers_for_speech,
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
        assert clean_for_speech("๑๒๓") == "หนึ่งร้อยยี่สิบสาม"
        assert clean_for_speech("๑๒๓", expand_numbers=False) == "123"

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
        # กฎ เอ็ด ต้องข้ามการเรียกซ้ำของหลักล้านไปด้วย
        assert thai_number_to_words(1_000_001) == "หนึ่งล้านเอ็ด"

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


class TestTranscriptKeepsNumbers:
    """ตัวจับคำซ้ำของ ASR เคยกินตัวเลขซ้ำในก้อนเดียวกัน ทำลายข้อมูลผู้ใช้

    เดิมใช้ ``(\\S+?)(?:\\s*\\1){2,}`` ซึ่งจับได้แม้แต่ตัวอักษรเดียวและไม่ต้องมี
    ช่องว่างคั่น ผลคือเบอร์โทร ราคา และปี พังทุกครั้งที่ผู้ใช้พูด
    """

    @pytest.mark.parametrize(
        "spoken",
        [
            "เบอร์ผม 0811111111 ครับ",
            "โทร 0899999999",
            "ราคา 1000 บาท",
            "ปี 2222",
            "ห้อง 555",
            "เลขที่ 11111",
            "บัญชี 1234567890",
        ],
    )
    def test_ตัวเลขต้องไม่ถูกแตะ(self, spoken):
        assert normalize_transcript(spoken) == spoken

    def test_ยังยุบคำซ้ำที่ASRหลอนได้อยู่(self):
        assert normalize_transcript("ขอบคุณ ขอบคุณ ขอบคุณ ขอบคุณ") == "ขอบคุณ"
        assert normalize_transcript("สวัสดี สวัสดี สวัสดี ครับ") == "สวัสดี ครับ"

    def test_คำซ้ำสองครั้งไม่ถือว่าหลอน(self):
        # พูดซ้ำสองครั้งเป็นเรื่องปกติของภาษาพูด
        assert normalize_transcript("ไม่ ไม่") == "ไม่ ไม่"


class TestParticleCollisions:
    """คำลงท้ายไทยพ้องกับคำธรรมดาเยอะมาก จึงต้องดูเฉพาะท้ายประโยค"""

    @pytest.mark.parametrize(
        "spoken",
        [
            "ค่าไฟเดือนนี้แพงมากเลย",   # ค่า = ค่าใช้จ่าย
            "คะแนนสอบออกหรือยัง",       # คะ ใน คะแนน
            "เสื้อตัวนี้คับไปหน่อย",      # คับ = แน่น
            "เสื้อคับ",
            "กางเกงคับ",
            "งานนี้ไม่มีค่า",
            "สิ่งนี้มีคุณค่า",
            "ผมมีอาการปวดขา",           # ขา = อวัยวะ
            "ขายของออนไลน์",            # ขา ใน ขาย
            "เสื้อสีขาวสวยดี",           # ขา ใน ขาว
            "วันนี้อากาศดี",
        ],
    )
    def test_ไม่เดาเพศจากคำที่พ้องกัน(self, spoken):
        assert detect_particle(spoken) is None

    @pytest.mark.parametrize(
        "spoken,expected",
        [
            ("ผมชื่อสมชายครับ", "male"),
            ("ผมชื่อสมชายครับผม", "male"),
            ("หนูชื่อแนนค่ะ", "female"),
            ("ไปไหนมาคะ", "female"),
            ("ได้เลยครับ.", "male"),
        ],
    )
    def test_คำลงท้ายท้ายประโยคยังจับได้(self, spoken, expected):
        assert detect_particle(spoken) == expected

    def test_คำลงท้ายที่เป็นการยกคำพูดคนอื่นต้องไม่นับ(self):
        assert detect_particle("เขาพูดว่าค่ะ") is None
        assert detect_particle("เมื่อกี้เขาบอกว่าครับ") is None

    def test_จ้ะจ๊ะไม่บอกเพศ(self):
        # จ้ะ/จ๊ะ/จ้า ใช้ได้ทั้งสองเพศ การเดาเพศจากคำพวกนี้ผิดบ่อยกว่าถูก
        assert detect_particle("ไปไหนจ๊ะ") is None
        assert detect_particle("ขอบคุณจ้า") is None


class TestNumberEdRule:
    """กฎ หนึ่ง -> เอ็ด ต้องข้ามการเรียกซ้ำของหลักล้านไปด้วย"""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (1_000_001, "หนึ่งล้านเอ็ด"),
            (2_000_001, "สองล้านเอ็ด"),
            (10_000_001, "สิบล้านเอ็ด"),
            (21_000_001, "ยี่สิบเอ็ดล้านเอ็ด"),
            (1_000_000_001, "หนึ่งพันล้านเอ็ด"),
            (1_000_011, "หนึ่งล้านสิบเอ็ด"),
            (1_100_000, "หนึ่งล้านหนึ่งแสน"),
            (1_000_000, "หนึ่งล้าน"),
            (21_000_000, "ยี่สิบเอ็ดล้าน"),
        ],
    )
    def test_คำอ่านหลักล้าน(self, value, expected):
        assert thai_number_to_words(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (100, "หนึ่งร้อย"),
            (101, "หนึ่งร้อยเอ็ด"),
            (110, "หนึ่งร้อยสิบ"),
            (111, "หนึ่งร้อยสิบเอ็ด"),
            (120, "หนึ่งร้อยยี่สิบ"),
            (121, "หนึ่งร้อยยี่สิบเอ็ด"),
            (1001, "หนึ่งพันเอ็ด"),
            (1010, "หนึ่งพันสิบ"),
            (1100, "หนึ่งพันหนึ่งร้อย"),
            (10000, "หนึ่งหมื่น"),
            (100000, "หนึ่งแสน"),
        ],
    )
    def test_คำอ่านหลักรองลงมา(self, value, expected):
        assert thai_number_to_words(value) == expected


class TestNumberExpansion:
    def test_อ่านจำนวนเป็นคำ(self):
        assert expand_numbers_for_speech("ราคา 199 บาท") == "ราคา หนึ่งร้อยเก้าสิบเก้า บาท"
        assert "สามสิบหก" in expand_numbers_for_speech("อุณหภูมิ 36 องศา")

    def test_เบอร์โทรอ่านทีละตัว(self):
        spoken = expand_numbers_for_speech("โทร 0812345678")
        assert "ศูนย์ แปด หนึ่ง" in spoken
        assert "แปดร้อย" not in spoken

    def test_เลขยาวอ่านทีละตัว(self):
        assert "หนึ่ง สอง สาม" in expand_numbers_for_speech("บัญชี 12345678")

    def test_ไม่แตะ_IP(self):
        assert expand_numbers_for_speech("ไอพี 192.168.1.1") == "ไอพี 192.168.1.1"

    def test_ไม่แตะเลขที่ติดกับตัวอักษร(self):
        assert expand_numbers_for_speech("รุ่น A4 ครับ") == "รุ่น A4 ครับ"

    def test_ทศนิยม(self):
        assert "จุด" in expand_numbers_for_speech("3.5 กิโล")


class TestUnitBoundaries:
    """หน่วยภาษาอังกฤษต้องเทียบแบบมีขอบเขตคำ ไม่งั้นไปกินตัวอักษรกลางคำ"""

    @pytest.mark.parametrize(
        "text",
        ["เปิดบัญชี KBank ได้ที่สาขา", "ผมสอบ MBTI ได้ INFP", "ชุมชน LGBT"],
    )
    def test_ไม่แปลงตัวย่อกลางคำ(self, text):
        spoken = clean_for_speech(text)
        assert "กิโลไบต์" not in spoken
        assert "เมกะไบต์" not in spoken
        assert "กิกะไบต์" not in spoken

    def test_ยังแปลงหน่วยที่ยืนเดี่ยว(self):
        assert "กิโลเมตร" in clean_for_speech("ระยะ 5 km")
        assert "กิกะไบต์" in clean_for_speech("ไฟล์ 5 GB")

    def test_องศาเซลเซียสต้องไม่หายไปกับตัวกรองสัญลักษณ์(self):
        # ° เป็นอักขระหมวด So ถ้ากรองทิ้งก่อนแปลงหน่วย จะเหลือแค่ C ลอย ๆ
        spoken = clean_for_speech("อุณหภูมิ 25°C ครับ")
        assert "องศาเซลเซียส" in spoken
        assert "C" not in spoken


class TestChunkerDoesNotBreakWords:
    def test_ไม่ตัดกลางคลัสเตอร์อักขระไทย(self):
        """ตัดคั่นวรรณยุกต์จะเหลือเครื่องหมายลอย ๆ ที่ TTS อ่านเป็นพยางค์ประหลาด"""
        import unicodedata

        source = "ผมขออธิบายแบบนี้นะครับการที่เราจะทำให้ระบบเสียงตอบสนองได้เร็วนั้นต้องเริ่มพูดตั้งแต่ประโยคแรกไม่ต้องรอให้โมเดลเขียนคำตอบจนจบ"
        chunker = SpeechChunker(min_chars=20, max_chars=60)
        chunks = list(chunker.feed(source)) + list(chunker.flush())

        assert len(chunks) > 1, "ข้อความยาวขนาดนี้ต้องถูกตัด"
        for chunk in chunks:
            assert unicodedata.category(chunk[0]) != "Mn", (
                f"ท่อนขึ้นต้นด้วยวรรณยุกต์ลอย: {chunk[:10]!r}"
            )
            assert chunk[-1] not in "เแโใไ", f"ท่อนจบด้วยสระหน้าที่ไม่มีพยัญชนะ: {chunk[-10:]!r}"
        assert "".join(chunks).replace(" ", "") == source

    def test_ขึ้นบรรทัดใหม่ตัดได้แม้ท่อนสั้น(self):
        chunker = SpeechChunker(min_chars=24)
        chunks = list(chunker.feed("สั้นมาก\nบรรทัดถัดไปที่ยาวกว่ามากพอสมควรเลยนะครับ "))
        assert chunks[0] == "สั้นมาก"
        for chunk in chunks:
            assert "\n" not in chunk


def test_ท่อนสุดท้ายที่สั้นถูกรวมกับท่อนก่อนหน้า():
    # ไม่งั้น "ครับ" จะถูกพูดเดี่ยว ๆ ห้อยท้าย
    parts = split_sentences("วันนี้อากาศดีมากเลยนะ ครับ")
    assert parts[-1] != "ครับ"


class TestChunkerNeverHangs:
    """ตัวตัดท่อนต้องคืบหน้าเสมอ ไม่งั้นเธรดจะหมุนอยู่กับที่ตลอดกาล

    เคยพังจริงกับข้อความที่ซ้อนวรรณยุกต์ไทยติดกันเป็นร้อยตัว: จุดตัดถูกเลื่อนถอย
    หลังจนถึงศูนย์ บัฟเฟอร์ไม่สั้นลง แล้ว feed() ก็หมุนไม่จบ ซึ่งแขวนทั้งคำขอ
    """

    @pytest.mark.parametrize(
        "evil",
        [
            "ก" + "่" * 300 + "ข" * 50,
            "่" * 500,
            "เ" * 400,
            "เเเเ" + "้" * 200 + "ก" * 100,
            "​" * 300 + "สวัสดี",
        ],
    )
    def test_ข้อความประหลาดต้องไม่ทำให้วนไม่จบ(self, evil):
        chunker = SpeechChunker(min_chars=20, max_chars=60)
        chunks = []
        for index, chunk in enumerate(chunker.feed(evil)):
            chunks.append(chunk)
            assert index < 2000, "ปล่อยท่อนออกมาไม่หยุด — น่าจะวนไม่จบ"
        chunks.extend(chunker.flush())
        assert "".join(chunks).replace(" ", "") == evil.replace(" ", "")

    def test_ข้อความยาวมากยังทำงานเร็ว(self):
        import time

        chunker = SpeechChunker()
        start = time.time()
        chunks = list(chunker.feed("สวัสดีครับ วันนี้อากาศดีมาก " * 2000))
        chunks.extend(chunker.flush())
        assert time.time() - start < 5.0
        assert chunks


class TestChunkerLatency:
    """เหตุผลทั้งหมดของการสตรีมคือได้ยินเสียงแรกเร็ว ถ้าคำตอบกลายเป็นท่อนเดียว
    ระบบก็ต้องรอโมเดลพิมพ์จนจบก่อนถึงจะเริ่มพูด ซึ่งเท่ากับไม่ได้สตรีมเลย
    """

    REPLIES = [
        "สวัสดีค่ะคุณเดช ยินดีที่ได้คุยกันอีกนะคะ วันนี้งานออกแบบเป็นยังไงบ้างคะ",
        "ได้เลยค่ะ อย่างแรกคือเรื่องงบประมาณ ถัดมาคือกำหนดส่งงาน สุดท้ายคือทีมที่จะมาช่วย",
        "ตอนนี้ที่เชียงใหม่อุณหภูมิประมาณ 24 องศาค่ะ อากาศเย็นสบายกำลังดีเลย",
        "จำได้ค่ะ คุณเดชเป็นสถาปนิกอยู่เชียงใหม่ แล้วก็ชอบกาแฟลาเต้ร้อนใช่ไหมคะ",
    ]

    @staticmethod
    def _stream(reply: str, step: int = 5):
        chunker = SpeechChunker()
        first_at = None
        received = 0
        chunks = []
        for index in range(0, len(reply), step):
            piece = reply[index : index + step]
            received += len(piece)
            for chunk in chunker.feed(piece):
                if first_at is None:
                    first_at = received
                chunks.append(chunk)
        chunks.extend(chunker.flush())
        return first_at, chunks

    @pytest.mark.parametrize("reply", REPLIES)
    def test_ต้องเริ่มพูดได้ก่อนคำตอบจะจบ(self, reply):
        first_at, chunks = self._stream(reply)
        assert first_at is not None, "ไม่มีท่อนไหนถูกปล่อยระหว่างสตรีมเลย"
        assert first_at < len(reply) * 0.75, (
            f"ต้องรอถึง {first_at}/{len(reply)} อักขระจึงเริ่มพูดได้ ซึ่งช้าเกินไป"
        )
        assert len(chunks) >= 2

    @pytest.mark.parametrize("reply", REPLIES)
    def test_ข้อความต้องครบไม่หายระหว่างตัด(self, reply):
        _first, chunks = self._stream(reply)
        assert "".join(chunks).replace(" ", "") == reply.replace(" ", "")

    def test_ประโยคสั้นที่จบในตัวถูกพูดทันที(self):
        # "ได้เลยค่ะ" เป็นประโยคสมบูรณ์ พูดออกไปเลยได้ ไม่ต้องรอประโยคถัดไป
        first_at, chunks = self._stream("ได้เลยค่ะ แล้วเดี๋ยวจะจัดการให้เรียบร้อยนะคะ")
        assert chunks[0] == "ได้เลยค่ะ"
        assert first_at <= 12


class TestTimeAndPhoneReading:
    """รูปแบบตัวเลขที่คนไทยเขียนจริง ซึ่งตัวอ่านตัวเลขทั่วไปอ่านผิดทั้งหมด"""

    @pytest.mark.parametrize(
        "text,expected_fragment",
        [
            # คนไทยเขียนเวลาด้วยจุดเป็นปกติ ถ้าอ่านเป็นทศนิยมจะฟังไม่รู้เรื่อง
            ("นัดกัน 20.30 น.", "ยี่สิบนาฬิกาสามสิบนาที"),
            ("เจอกัน 9.00 น.", "เก้านาฬิกา"),
            ("นัดกัน 20:30", "ยี่สิบนาฬิกาสามสิบนาที"),
            ("เวลา 0:05", "ศูนย์นาฬิกาห้านาที"),
        ],
    )
    def test_อ่านเวลา(self, text, expected_fragment):
        assert expected_fragment in expand_numbers_for_speech(text)

    def test_เวลาแบบจุดต้องไม่ถูกอ่านเป็นทศนิยม(self):
        assert "จุด" not in expand_numbers_for_speech("นัดกัน 20.30 น.")

    def test_ทศนิยมที่ไม่ใช่เวลายังอ่านเป็นทศนิยม(self):
        assert "จุด" in expand_numbers_for_speech("สูง 3.5 เมตร")

    @pytest.mark.parametrize(
        "text",
        ["โทร 081-234-5678", "โทร. 02 123 4567", "เบอร์ +66 81 234 5678", "โทร 0812345678"],
    )
    def test_เบอร์โทรอ่านทีละตัว(self, text):
        spoken = expand_numbers_for_speech(text)
        # ถ้าอ่านเป็นจำนวนจะมีคำว่า ร้อย พัน หมื่น โผล่มา
        assert not any(word in spoken for word in ("ร้อย", "พัน", "หมื่น")), spoken
        assert "ศูนย์" in spoken or "หก หก" in spoken

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ราคา 1,234 บาท", "ราคา หนึ่งพันสองร้อยสามสิบสี่ บาท"),
            ("ประชากร 1,000,000 คน", "ประชากร หนึ่งล้าน คน"),
            ("ยอด 12,345,678 บาท", None),
        ],
    )
    def test_เลขคั่นจุลภาคอ่านเป็นจำนวนเดียว(self, text, expected):
        spoken = expand_numbers_for_speech(text)
        assert "," not in spoken, spoken
        if expected:
            assert spoken == expected

    def test_อ่านวันที่แบบทับ(self):
        spoken = expand_numbers_for_speech("นัดวันที่ 15/8/2568")
        assert "สิงหาคม" in spoken
        assert "/" not in spoken

    def test_วันที่ผิดรูปต้องไม่ถูกเดาเป็นวันเดือนปี(self):
        # เดือนที่ 13 ไม่มีจริง อย่าเดา — แต่ก็ต้องไม่ปล่อย "/" ให้ TTS อ่าน
        spoken = expand_numbers_for_speech("1/13/2568")
        assert "/" not in spoken
        assert "มกราคม" not in spoken and "ธันวาคม" not in spoken

    def test_เลขบ้านและคะแนนไม่ถูกเข้าใจผิดเป็นเบอร์โทร(self):
        assert "ศูนย์ หนึ่ง" not in expand_numbers_for_speech("คะแนน 2-1")
        assert "สอง" in expand_numbers_for_speech("คะแนน 2-1")

    def test_รวมกันทั้งประโยค(self):
        spoken = clean_for_speech("ประชุม 20.30 น. ที่ห้อง 302 ราคา 1,500 บาท")
        assert "ยี่สิบนาฬิกาสามสิบนาที" in spoken
        assert "หนึ่งพันห้าร้อย" in spoken
        assert "," not in spoken


class TestSpokenNumberForms:
    """รูปแบบตัวเลขที่คนไทยเขียนจริง ตรวจว่าพูดออกมาแล้วความหมายไม่เพี้ยน"""

    @pytest.mark.parametrize(
        "text,expected",
        [
            # ขีดกลางอ่านออกเสียงไม่ได้ TTS จะกลืนหาย ทำให้ "3-5 คน" ฟังเป็น "สามห้าคน"
            ("มากันประมาณ 3-5 คน", "สามถึงห้า"),
            ("อ่านหน้า 12-15", "สิบสองถึงสิบห้า"),
            ("คะแนน 2-1", "สองต่อหนึ่ง"),
            ("สกอร์ 3-0", "สามต่อศูนย์"),
            ("บ้านเลขที่ 99/1", "เก้าสิบเก้าทับหนึ่ง"),
        ],
    )
    def test_ช่วงตัวเลขสกอร์และเลขที่(self, text, expected):
        spoken = expand_numbers_for_speech(text)
        assert expected in spoken, spoken
        assert "-" not in spoken and "/" not in spoken

    @pytest.mark.parametrize(
        "text",
        ["ห้อง 2105", "ชั้น 305", "เบอร์ 1234", "รหัส 4589", "ที่นั่ง 12345"],
    )
    def test_เลขรหัสอ่านทีละตัว(self, text):
        # คนไทยอ่าน "ห้อง 2105" ว่า "ห้องสองหนึ่งศูนย์ห้า" ไม่ใช่ "สองพันหนึ่งร้อยห้า"
        spoken = expand_numbers_for_speech(text)
        assert not any(word in spoken for word in ("ร้อย", "พัน", "หมื่น")), spoken

    def test_เลขทั่วไปยังอ่านเป็นจำนวน(self):
        assert "ยี่สิบห้า" in expand_numbers_for_speech("อายุ 25 ปี")
        assert "สองพันห้าร้อยหกสิบแปด" in expand_numbers_for_speech("ปี 2568")

    @pytest.mark.parametrize(
        "text",
        ["เลขบัญชี 123-4-56789-0", "บัตร 1234-5678-9012-3456", "เบอร์ 081-234-5678"],
    )
    def test_เลขรหัสยาวที่มีตัวคั่นอ่านทีละตัว(self, text):
        spoken = expand_numbers_for_speech(text)
        assert not any(word in spoken for word in ("ร้อย", "พัน", "หมื่น")), spoken

    def test_เลขคั่นจุลภาคที่มีทศนิยม(self):
        # เดิมกลุ่มจุลภาคกินแต่ส่วนจำนวนเต็ม เหลือ ".50" เป็นตัวเลขดิบไปถึง TTS
        # และเดิมอ่าน "จุดห้าศูนย์บาท" ซึ่งฟังออกทันทีว่าเครื่องอ่าน
        spoken = expand_numbers_for_speech("ราคา 1,234.50 บาท")
        assert spoken == "ราคา หนึ่งพันสองร้อยสามสิบสี่บาทห้าสิบสตางค์"

    def test_ทศนิยมที่ไม่ใช่เงินยังอ่านเป็นจุด(self):
        spoken = expand_numbers_for_speech("สูง 1,234.50 เมตร")
        assert spoken == "สูง หนึ่งพันสองร้อยสามสิบสี่ จุด ห้า ศูนย์ เมตร"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("นัดกันตอน 20.30 นะคะ", "ยี่สิบนาฬิกาสามสิบนาที"),
            ("เจอกัน 19.00 ที่ร้าน", "สิบเก้านาฬิกา"),
            ("ประชุมเวลา 8.30", "แปดนาฬิกาสามสิบนาที"),
        ],
    )
    def test_เวลาแบบจุดที่ไม่มี_น_แต่มีคำบอกเวลา(self, text, expected):
        assert expected in expand_numbers_for_speech(text)

    @pytest.mark.parametrize("text", ["สูง 3.50 เมตร", "กว้าง 2.75 เมตร", "หนัก 1.20 กิโล"])
    def test_ทศนิยมที่ไม่มีบริบทเวลายังเป็นทศนิยม(self, text):
        spoken = expand_numbers_for_speech(text)
        assert "นาฬิกา" not in spoken, spoken
        assert "จุด" in spoken

    def test_ไม่เหลือ_น_ลอยหลังเวลา(self):
        assert expand_numbers_for_speech("ประชุม 9:00 น.").strip().endswith("เก้านาฬิกา")


class TestChunkerKeepsWordsIntact:
    """จุดตัดต้องไม่ทำให้คำหรือความหมายขาด"""

    CASES = [
        "เขาเป็นคนคับแคบมากเลยค่ะ ไม่ค่อยแบ่งใครเลย",
        "ตอนนี้ยังไม่ทราบค่าใช้จ่ายทั้งหมดค่ะ",
        "เดือนนี้ค่าไฟแพงกว่าเดือนที่แล้วเยอะเลยค่ะ",
        "บริษัทกำลังหาผู้รับจ้างรายใหม่อยู่ค่ะ",
        "เขาเกิดเมื่อ พ.ศ. 2540 ที่จังหวัดเชียงใหม่ค่ะ",
        "ประชุมพรุ่งนี้เวลา 13.30 น. ที่ห้องใหญ่ค่ะ",
        "ราคารวมทั้งหมด 1,234.50 บาทค่ะ เดี๋ยวส่งใบเสร็จให้นะคะ",
    ]

    @pytest.mark.parametrize("reply", CASES)
    @pytest.mark.parametrize("step", [1, 3, 5, 7, 11])
    def test_ข้อความครบทุกจังหวะการสตรีม(self, reply, step):
        chunker = SpeechChunker()
        chunks = []
        for index in range(0, len(reply), step):
            chunks.extend(chunker.feed(reply[index : index + step]))
        chunks.extend(chunker.flush())
        assert "".join(chunks).replace(" ", "") == reply.replace(" ", "")

    @pytest.mark.parametrize("step", [1, 3, 5, 7, 11])
    def test_คำลงท้ายที่เป็นคำย่อยต้องไม่ทำให้ตัดกลางคำ(self, step):
        """"คับ" ใน "คับแคบ" และ "ค่า" ใน "ค่าไฟ" เคยกลายเป็นจุดตัด

        เกิดเฉพาะระหว่างสตรีมเมื่อบัฟเฟอร์จบกลางคำนั้นพอดี จึงเกิดแบบสุ่ม
        ตามจังหวะที่ข้อความไหลมา และตามหาสาเหตุจากรายงานบัคได้ยากมาก
        """
        reply = "เขาเป็นคนคับแคบมากเลยค่ะ"
        chunker = SpeechChunker()
        chunks = []
        for index in range(0, len(reply), step):
            chunks.extend(chunker.feed(reply[index : index + step]))
        chunks.extend(chunker.flush())
        assert all("คนคับ" not in chunk or "คับแคบ" in chunk for chunk in chunks), chunks

    @pytest.mark.parametrize("step", [1, 3, 5, 7])
    def test_ตัวย่อและเวลาไม่ถูกตัดกลาง(self, step):
        reply = "ประชุมเวลา 13.30 น. ที่ตึก พ.ศ. ใหม่ค่ะ"
        chunker = SpeechChunker()
        chunks = []
        for index in range(0, len(reply), step):
            chunks.extend(chunker.feed(reply[index : index + step]))
        chunks.extend(chunker.flush())
        for chunk in chunks:
            assert not chunk.endswith("พ."), chunk
            assert not chunk.endswith("13.30"), chunk

    def test_เวลายังถูกแปลงถูกหลังตัดท่อน(self):
        reply = "ประชุมพรุ่งนี้เวลา 13.30 น. ที่ห้องใหญ่ค่ะ"
        chunker = SpeechChunker()
        chunks = list(chunker.feed(reply)) + list(chunker.flush())
        spoken = " ".join(clean_for_speech(chunk) for chunk in chunks)
        assert "สิบสามนาฬิกาสามสิบนาที" in spoken, spoken


class Testการอ่านตัวเลขที่เคยพลาด:
    """ทุกเคสในนี้เคยอ่านผิดจริง — ไม่ใช่กรณีสมมติ"""

    @pytest.mark.parametrize(
        "text,expected",
        [
            # ขีดกลางระหว่างเลขคั่นจุลภาคเคยถูกผ่ากลาง เหลือ "หนึ่ง," ลอยอยู่
            ("ราคา 1,200-1,500 บาท", "ราคา หนึ่งพันสองร้อยถึงหนึ่งพันห้าร้อย บาท"),
            # รายการตัวเลขคั่นด้วยช่องว่างเคยถูกเหมาเป็นเบอร์โทรก้อนเดียว
            ("ราคา 1500 2000 3000 บาท", "ราคา หนึ่งพันห้าร้อย สองพัน สามพัน บาท"),
            # เลขคั่นจุลภาคคือจำนวน ไม่ใช่รหัส เคยอ่านทีละตัวเพราะยาวเกิน 8 หลัก
            ("ยอด 12,345,678 บาท", "ยอด สิบสองล้านสามแสนสี่หมื่นห้าพันหกร้อยเจ็ดสิบแปด บาท"),
            # "ประมาณ" ไม่ใช่คำบอกเวลา เคยอ่านเป็น "สามนาฬิกาห้าสิบนาที"
            ("สูงประมาณ 3.50 เมตร", "สูงประมาณ สาม จุด ห้า ศูนย์ เมตร"),
            # ภาษาไทยเขียนติดกัน กฎ lookbehind เดิมจึงแทบไม่เคยทำงาน
            ("ไปห้อง 2105", "ไปห้อง สอง หนึ่ง ศูนย์ ห้า"),
            # เลขที่บ้านต้องอ่าน "ทับ" ไม่ใช่ทิ้งเครื่องหมายไว้
            ("บ้านเลขที่ 123/45", "บ้านเลขที่ หนึ่งร้อยยี่สิบสามทับสี่สิบห้า"),
            # "ผล" ลอย ๆ เคยไปติด "ผลไม้" ทำให้อ่านเป็นสกอร์
            ("ผลไม้ 3-5 ชิ้น", "ผลไม้ สามถึงห้า ชิ้น"),
            ("ผลบอล 3-5", "ผลบอล สามต่อห้า"),
            # เบอร์บริการสั้น ๆ ต้องอ่านทีละตัว
            ("โทร 1668", "โทร หนึ่ง หก หก แปด"),
            # ขีดกลางหน้าเลข TTS อ่านไม่ออก ความหมายเลยกลับด้าน
            ("อุณหภูมิ -5.5 องศา", "อุณหภูมิ ลบห้า จุด ห้า องศา"),
        ],
    )
    def test_อ่านถูกต้อง(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    def test_ปีหลายปีติดกันต้องไม่กลายเป็นเบอร์โทร(self):
        got = expand_numbers_for_speech("ปี 2566 2567 2568")
        assert "สองพันห้าร้อยหกสิบหก" in got
        assert "สองพันห้าร้อยหกสิบแปด" in got

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("โทร 081-234-5678", "โทร ศูนย์ แปด หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด"),
            ("นัดกัน 20.30 น.", "นัดกัน ยี่สิบนาฬิกาสามสิบนาที"),
            ("เจอกัน 19.00", "เจอกัน สิบเก้านาฬิกา"),
            ("วันที่ 15/8/2568", "วันที่ สิบห้า สิงหาคม สองพันห้าร้อยหกสิบแปด"),
            ("ราคา 1,234.50 บาท", "ราคา หนึ่งพันสองร้อยสามสิบสี่บาทห้าสิบสตางค์"),
        ],
    )
    def test_ของที่เคยถูกต้องต้องยังถูกต้อง(self, text, expected):
        assert expand_numbers_for_speech(text) == expected


class Testตัวย่อและสัญลักษณ์:
    def test_ตัวย่อไทยต้องไม่ถูกตัดเป็นคนละประโยค(self):
        """"ผศ. ดร. สมชาย" ถูกตัดหลัง "ดร." ทำให้ TTS หยุดกลางชื่อคน"""
        chunker = SpeechChunker(min_chars=8)
        out = []
        for ch in "ผศ. ดร. สมชาย เป็นอาจารย์":
            out += list(chunker.feed(ch))
        out += list(chunker.flush())
        assert out[0].startswith("ผศ. ดร. สมชาย")

    def test_จุดจบประโยคจริงยังตัดได้(self):
        chunker = SpeechChunker(min_chars=8)
        out = []
        for ch in "อากาศดี. ไปเที่ยวกัน":
            out += list(chunker.feed(ch))
        out += list(chunker.flush())
        assert out == ["อากาศดี.", "ไปเที่ยวกัน"]

    @pytest.mark.parametrize(
        "text,expected",
        [
            # ภาษาไทยพูดหน่วยเงินหลังจำนวนเสมอ
            ("ราคา ฿500", "ราคา ห้าร้อย บาท"),
            ("ราคา $20", "ราคา ยี่สิบ ดอลลาร์"),
            ("ราคา 1,200฿", "ราคา หนึ่งพันสองร้อย บาท"),
            # เครื่องหมายที่ไม่ติดกับตัวเลขไม่ใช่ราคา อ่านว่า "ดอลลาร์" ตรงนั้นแย่กว่าเงียบ
            ("ใช้ `echo $HOME` ดู", "ใช้ echo HOME ดู"),
            ("คิดเป็น 3×4=12", "คิดเป็น สาม คูณ สี่ เท่ากับ สิบสอง"),
            ("อุณหภูมิ 25±2 องศา", "อุณหภูมิ ยี่สิบห้า บวกลบ สอง องศา"),
            # ตัวย่อเดือนถูกอ่านทีละพยางค์ ฟังไม่ออกว่าเดือนอะไร
            ("ก.ค. นี้", "กรกฎาคม นี้"),
            ("วันที่ 5 มี.ค. 2568", "วันที่ ห้า มีนาคม สองพันห้าร้อยหกสิบแปด"),
        ],
    )
    def test_สัญลักษณ์ที่TTSอ่านไม่ออกต้องถูกแปลง(self, text, expected):
        assert clean_for_speech(text) == expected


class Testเวลาและเศษส่วน:
    """ทุกเคสมาจากการตรวจภาษาไทยรอบสี่"""

    @pytest.mark.parametrize(
        "text,expected",
        [
            # เวลาทำการคือวิธีเขียนเวลาที่พบบ่อยที่สุด แต่เคยอ่านเป็นทศนิยม
            # "สิบจุดศูนย์ศูนย์" ซึ่งไม่ใช่ภาษาไทยเลย
            ("ตั้งแต่ 9.00 ถึง 17.00", "ตั้งแต่ เก้านาฬิกาถึงสิบเจ็ดนาฬิกา"),
            ("ร้านเปิด 10.00-22.00 น.", "ร้านเปิด สิบนาฬิกาถึงยี่สิบสองนาฬิกา"),
            # ขีดกลางระหว่างเวลาสองตัวเคยค้างไว้ให้ TTS กลืนหาย
            ("9:00-17:00", "เก้านาฬิกาถึงสิบเจ็ดนาฬิกา"),
            (
                "เวลา 13.30-14.30 น.",
                "เวลา สิบสามนาฬิกาสามสิบนาทีถึงสิบสี่นาฬิกาสามสิบนาที",
            ),
            # คนไทยเขียนเวลาสิ้นวันแบบนี้เป็นปกติ
            ("24.00 น.", "ยี่สิบสี่นาฬิกา"),
        ],
    )
    def test_ช่วงเวลา(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            # "ทับ" ใช้กับที่อยู่เท่านั้น เศษส่วนและคะแนนใช้ "ส่วน"
            ("แบ่งคนละ 1/2", "แบ่งคนละ หนึ่งส่วนสอง"),
            ("สอบได้ 18/20", "สอบได้ สิบแปดจากยี่สิบ"),
            ("บ้านเลขที่ 123/45", "บ้านเลขที่ หนึ่งร้อยยี่สิบสามทับสี่สิบห้า"),
            ("ชั้น 3/1", "ชั้น สามทับหนึ่ง"),
            ("วันที่ 5/12", "วันที่ ห้า ธันวาคม"),
        ],
    )
    def test_เครื่องหมายทับ(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            # คำบอกบริบทเคยต้องติดกับตัวเลข กฎจึงทำงานเฉพาะสำนวนสั้นที่สุด
            ("ห้องประชุม 2105", "ห้องประชุม สอง หนึ่ง ศูนย์ ห้า"),
            ("รหัสพนักงาน 123456", "รหัสพนักงาน หนึ่ง สอง สาม สี่ ห้า หก"),
            # "+" เคยหายไปเงียบ ๆ เหมือน "-" ก่อนแก้รอบสาม
            ("+15 เปอร์เซ็นต์", "บวกสิบห้า เปอร์เซ็นต์"),
            # เงินต้องอ่านเป็นบาท/สตางค์
            ("ค่าอาหาร 250.75 บาท", "ค่าอาหาร สองร้อยห้าสิบบาทเจ็ดสิบห้าสตางค์"),
            ("จ่าย 100.00 บาท", "จ่าย หนึ่งร้อยบาทถ้วน"),
        ],
    )
    def test_รหัสเครื่องหมายและเงิน(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    def test_เลขบัตรประชาชนสิบสามหลักต้องอ่านทีละตัว(self):
        """ของเดิมจำกัดที่ 12 หลัก เลข 13 หลักจึงหลุดไปให้ TTS อ่านดิบ ๆ"""
        spoken = expand_numbers_for_speech("เลขบัตรประชาชน 1234567890123")
        assert not any(ch.isdigit() for ch in spoken), spoken


class Testรหัสประจำตัวและช่วงราคา:
    def test_เลขประจำตัวที่คั่นด้วยช่องว่างต้องอ่านทีละตัว(self):
        """เลขบัตรประชาชนไทยเขียนเป็น "1 2345 67890 12 3" และไม่ขึ้นต้นด้วย 0

        กฎ "ต้องมีขีดหรือขึ้นต้นด้วย 0" ที่เพิ่มรอบสามจึงพลาดเสมอ
        """
        for text in (
            "บัตรประชาชน 1 2345 67890 12 3",
            "เลขบัตร 4111 1111 1111 1111",
            "เลขบัญชี 123 456 7890",
        ):
            spoken = expand_numbers_for_speech(text)
            assert not any(ch.isdigit() for ch in spoken), spoken

    def test_รายการตัวเลขธรรมดายังไม่ถูกเหมาเป็นเบอร์(self):
        assert "สองพันห้าร้อยหกสิบหก" in expand_numbers_for_speech("ปี 2566 2567 2568")
        assert "หนึ่งพันห้าร้อย" in expand_numbers_for_speech("ราคา 1500 2000 3000 บาท")

    @pytest.mark.parametrize(
        "text,expected",
        [
            # ห้าหลักเคยตกไปให้กฎเบอร์โทร ซึ่งเห็นขีดกลางของช่วงเป็นหลักฐาน
            ("ราคา 10000-20000 บาท", "ราคา หนึ่งหมื่นถึงสองหมื่น บาท"),
            (
                "เงินเดือน 15000-25000 บาท",
                "เงินเดือน หนึ่งหมื่นห้าพันถึงสองหมื่นห้าพัน บาท",
            ),
        ],
    )
    def test_ช่วงราคาห้าหลัก(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    def test_ยศทหารต้องไม่ถูกตัดกลาง(self):
        """รายการที่มีจุดอยู่ในตัวกันได้แค่จุดที่สอง จุดแรกของ "พล.อ." จึงหลุด"""
        chunker = SpeechChunker(min_chars=8)
        out = []
        for ch in "ครับผม พล.อ. ประยุทธ์ เดินทางไปประชุมครับ":
            out += list(chunker.feed(ch))
        out += list(chunker.flush())
        assert out[0] == "ครับผม พล.อ."


class Testตัวย่อหน่วยไทย:
    """TTS อ่านตัวย่อหน่วยเป็นชื่อตัวอักษร ("กอกอ") ส่วน kg/cm ถูกขยายอยู่แล้ว"""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("น้ำหนัก 3.2 กก.", "น้ำหนัก สาม จุด สอง กิโลกรัม"),
            ("ส่วนสูง 50 ซม.", "ส่วนสูง ห้าสิบ เซนติเมตร"),
            ("ระยะทาง 12 กม.", "ระยะทาง สิบสอง กิโลเมตร"),
            ("ใช้เวลา 2 ชม.", "ใช้เวลา สอง ชั่วโมง"),
            ("พื้นที่ 40 ตร.ม.", "พื้นที่ สี่สิบ ตารางเมตร"),
        ],
    )
    def test_ขยายเป็นคำเต็ม(self, text, expected):
        assert clean_for_speech(text) == expected

    def test_คำลงท้ายเดี่ยวต้องไม่ถูกส่งไปพูดแยก(self):
        """"ค่ะ" คำเดียวที่ถูกสังเคราะห์เสียงแยกฟังเหมือนคนละประโยค"""
        chunker = SpeechChunker()
        out = []
        for ch in clean_for_speech("ส่วนสูง 50 ซม. ค่ะ ยินดีที่ได้รู้จักนะคะ"):
            out += list(chunker.feed(ch))
        out += list(chunker.flush())
        assert all(len(chunk) > 4 for chunk in out), out


class Testสะพานคำระหว่างคำบอกบริบทกับตัวเลข:
    """การยอมให้มีคำไทยคั่นทำให้กฎกินประโยคธรรมดาเข้ามาด้วย

    "โต๊ะที่นั่งกันเมื่อวาน 1234" ไม่ใช่รหัสโต๊ะ — "ที่นั่ง" เป็นคำบอกบริบท
    แล้ว "กันเมื่อวาน" ถูกนับเป็นส่วนของคำนามประสม
    """

    @pytest.mark.parametrize(
        "text",
        [
            "โต๊ะที่นั่งกันเมื่อวาน 1234",
            "เบอร์ที่ให้ไว้เมื่อวาน 1234",
            "ชั้นของเขาอยู่ที่ 250",
        ],
    )
    def test_คำเชื่อมที่เปิดอนุประโยคต้องไม่ถูกข้าม(self, text):
        spoken = expand_numbers_for_speech(text)
        assert " หนึ่ง สอง " not in spoken, spoken

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ห้องประชุม 2105", "ห้องประชุม สอง หนึ่ง ศูนย์ ห้า"),
            ("รหัสพนักงาน 123456", "รหัสพนักงาน หนึ่ง สอง สาม สี่ ห้า หก"),
            ("เลขที่ใบสั่งซื้อ 778899", "เลขที่ใบสั่งซื้อ เจ็ด เจ็ด แปด แปด เก้า เก้า"),
        ],
    )
    def test_คำนามประสมยังข้ามได้(self, text, expected):
        assert expand_numbers_for_speech(text) == expected


class Testหน่วยที่ตามหลังตัวเลข:
    """หน่วยที่ตามหลังพิสูจน์ว่าเป็น "จำนวน" ไม่ใช่รหัสหรือเวลา

    แม่นกว่าการดูคำที่อยู่ข้างหน้ามาก เพราะคำนำหน้าอย่าง "ห้อง" "ตู้" "เวลา"
    "ตี" "ออก" ล้วนเป็นคำธรรมดาที่โผล่ในประโยคไหนก็ได้
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ห้องละ 1500 บาท", "ห้องละ หนึ่งพันห้าร้อย บาท"),
            ("ตู้เย็นราคา 8990 บาท", "ตู้เย็นราคา แปดพันเก้าร้อยเก้าสิบ บาท"),
            ("เที่ยวบินดีเลย์ 120 นาที", "เที่ยวบินดีเลย์ หนึ่งร้อยยี่สิบ นาที"),
            ("ประชุมใช้เวลา 1.30 ชั่วโมง", "ประชุมใช้เวลา หนึ่ง จุด สาม ศูนย์ ชั่วโมง"),
            ("เริ่มต้นที่ 1.20 เมตร", "เริ่มต้นที่ หนึ่ง จุด สอง ศูนย์ เมตร"),
            # เลขกลม ๆ ที่ลงท้ายด้วยศูนย์หลายตัวไม่ใช่รหัสแน่นอน
            ("ประชากร 70000000 คน", "ประชากร เจ็ดสิบล้าน คน"),
        ],
    )
    def test_มีหน่วยตามหลังต้องอ่านเป็นจำนวน(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ห้องประชุม 2105", "ห้องประชุม สอง หนึ่ง ศูนย์ ห้า"),
            # "ภายใน" มี "ใน" อยู่ข้างใน กฎเดิมจึงกันผิดตัว
            ("เบอร์ภายใน 1234", "เบอร์ภายใน หนึ่ง สอง สาม สี่"),
            ("เจอกัน 19.00", "เจอกัน สิบเก้านาฬิกา"),
            # "ที่" เป็นทั้งลักษณนามและคำบุพบท
            ("เจอกัน 19.00 ที่ร้าน", "เจอกัน สิบเก้านาฬิกา ที่ร้าน"),
        ],
    )
    def test_ไม่มีหน่วยตามหลังยังเป็นรหัสหรือเวลา(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            # ทวิภาคเคยหลุดถึง TTS ดิบ ๆ
            ("อัตราส่วน 1:2", "อัตราส่วน หนึ่งต่อสอง"),
            ("สัดส่วน 60:40", "สัดส่วน หกสิบต่อสี่สิบ"),
            # คะแนนสอบพูดว่า "จาก" ไม่ใช่ "ส่วน" ซึ่งเป็นการอ่านเศษส่วน
            ("คะแนน 18/20", "คะแนน สิบแปดจากยี่สิบ"),
            ("แบ่งคนละ 1/2", "แบ่งคนละ หนึ่งส่วนสอง"),
            ("บ้านเลขที่ 123/45", "บ้านเลขที่ หนึ่งร้อยยี่สิบสามทับสี่สิบห้า"),
            # ".5 บาท" คือห้าสิบสตางค์ ไม่ใช่ห้าสตางค์
            ("ราคา 99.5 บาท", "ราคา เก้าสิบเก้าบาทห้าสิบสตางค์"),
        ],
    )
    def test_อัตราส่วนคะแนนและเงิน(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ราคา 50 บาท/กก.", "ราคา ห้าสิบ บาท ต่อ กิโลกรัม"),
            ("ความเร็ว 90 กม./ชม.", "ความเร็ว เก้าสิบ กิโลเมตร ต่อ ชั่วโมง"),
            ("พ.ศ. 2568", "พุทธศักราช สองพันห้าร้อยหกสิบแปด"),
            ("สูง 1.75 ม.", "สูง หนึ่ง จุด เจ็ด ห้า เมตร"),
            # ตัวย่อหน่วยต้องตามหลังตัวเลขเท่านั้น
            ("บ.ก. บอกว่าให้แก้", "บ.ก. บอกว่าให้แก้"),
            ("มีส้ม กล้วย ฯลฯ", "มีส้ม กล้วย และอื่น ๆ"),
        ],
    )
    def test_หน่วยศักราชและตัวย่อ(self, text, expected):
        assert clean_for_speech(text) == expected


class Testเบอร์บ้านต่างจังหวัดและช่วงทศนิยม:
    @pytest.mark.parametrize(
        "text,expected",
        [
            # 0XX-XXXXXX คือเบอร์บ้านต่างจังหวัด ไม่ใช่ช่วง 38 ถึง 123456
            ("ติดต่อ 038-123456 ครับ", "ติดต่อ ศูนย์ สาม แปด หนึ่ง สอง สาม สี่ ห้า หก ครับ"),
            ("โทร 053-123456", "โทร ศูนย์ ห้า สาม หนึ่ง สอง สาม สี่ ห้า หก"),
        ],
    )
    def test_เบอร์ที่ขึ้นต้นด้วยศูนย์ต้องไม่เป็นช่วง(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["น้ำหนัก 1.50-2.30 กิโลกรัม", "เกรดเฉลี่ย 3.00-4.00", "ราคา 1.50-2.30 ดอลลาร์"],
    )
    def test_ช่วงทศนิยมต้องไม่กลายเป็นเวลา(self, text):
        """เลขสองตัวคั่นด้วยขีดไม่ใช่หลักฐานในตัวเองว่าเป็นเวลา"""
        assert "นาฬิกา" not in expand_numbers_for_speech(text)

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ร้านเปิด 10.00-22.00 น.", "ร้านเปิด สิบนาฬิกาถึงยี่สิบสองนาฬิกา"),
            ("ตั้งแต่ 9.00 ถึง 17.00", "ตั้งแต่ เก้านาฬิกาถึงสิบเจ็ดนาฬิกา"),
            ("9:00-17:00", "เก้านาฬิกาถึงสิบเจ็ดนาฬิกา"),
        ],
    )
    def test_ช่วงเวลาจริงยังทำงาน(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    def test_เลขยาวเกินขีดจำกัดของ_int_ต้องไม่โยน(self):
        """Python แปลงสตริงเป็น int ได้ไม่เกิน 4300 หลัก"""
        for text in ("9" * 5000 + ".50 บาท", "1," + "234," * 2000 + "567"):
            expand_numbers_for_speech(text)
            clean_for_speech(text)


class Testขอบเขตของหน่วยที่ตามหลัง:
    """ภาษาไทยไม่เว้นวรรคระหว่างคำ หน่วยจึงต้องจบตรงนั้นจริง ๆ

    "ห้อง 2105 คนนี้" มี "คน" ตามหลังตัวเลข ทั้งที่คำจริงคือ "คนนี้"
    ซึ่งไม่ใช่ลักษณนาม เลขห้องเลยถูกอ่านเป็นจำนวน
    """

    @pytest.mark.parametrize(
        "text",
        [
            "ห้อง 2105 คนนี้",
            "ห้อง 2105 วันนี้",
            "ห้อง 2105 ที่ดี",
            "รหัส 123456 ครั้งนี้",
            "เบอร์ 1234 ปีที่แล้ว",
        ],
    )
    def test_พยางค์แรกของคำอื่นไม่ใช่หน่วย(self, text):
        assert "พัน" not in expand_numbers_for_speech(text), text

    @pytest.mark.parametrize(
        "text,expected",
        [
            # ยอมให้ตามด้วยคำลงท้ายได้ คนไทยเขียนติดกันเป็นปกติ
            ("ห้องละ 1500 บาทครับ", "ห้องละ หนึ่งพันห้าร้อย บาทครับ"),
            ("พัสดุน้ำหนัก 500 กรัมครับ", "พัสดุน้ำหนัก ห้าร้อย กรัมครับ"),
            ("ที่นั่งว่างอีก 300 ที่", "ที่นั่งว่างอีก สามร้อย ที่"),
            ("อายุ 30 ปี", "อายุ สามสิบ ปี"),
        ],
    )
    def test_หน่วยจริงยังทำงาน(self, text, expected):
        assert expand_numbers_for_speech(text) == expected


class Testหน่วยสองชั้น:
    """ภาษาไทยไม่เว้นวรรค กฎขอบเขตจึงต้องแบ่งเป็นสองชั้น

    เทียบแบบ "ขึ้นต้นด้วย" เฉย ๆ ทำให้ "ห้อง 2105 คนนี้" เห็น "คน" เป็นลักษณนาม
    ส่วนการบังคับว่าต้องจบตรงนั้นพอดีทำให้ "2500 บาทรวมอาหารเช้า" ไม่เห็น "บาท"
    แล้วราคาถูกท่องทีละหลัก ซึ่งแย่กว่ากันมาก
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ห้องพักคืนละ 2500 บาทรวมอาหารเช้า", "ห้องพักคืนละ สองพันห้าร้อย บาทรวมอาหารเช้า"),
            ("ที่นั่ง 350 บาทต่อที่", "ที่นั่ง สามร้อยห้าสิบ บาทต่อที่"),
            (
                "ชั้นวางยาว 120 เซนติเมตรกว่า ๆ",
                "ชั้นวางยาว หนึ่งร้อยยี่สิบ เซนติเมตรกว่า ๆ",
            ),
            ("ออกกำลัง 1.30 ชั่วโมงทุกวัน", "ออกกำลัง หนึ่ง จุด สาม ศูนย์ ชั่วโมงทุกวัน"),
        ],
    )
    def test_หน่วยที่ตามด้วยคำอื่นยังนับเป็นหน่วย(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["ห้อง 2105 คนนี้", "ห้อง 2105 วันนี้", "ห้อง 2105 ตัวอย่าง", "รหัส 123456 ครั้งนี้"],
    )
    def test_พยางค์แรกของคำอื่นยังไม่ใช่หน่วย(self, text):
        assert "พัน" not in expand_numbers_for_speech(text), text


class Testเครื่องหมายทับและอัตราส่วน:
    @pytest.mark.parametrize(
        "text,expected",
        [
            # คะแนนสอบ: ตัวล่างเป็นเลขเต็มสิบขึ้นไป
            ("ได้ 15/20 ข้อ", "ได้ สิบห้าจากยี่สิบ ข้อ"),
            ("ทำได้ 7/10", "ทำได้ เจ็ดจากสิบ"),
            # เศษส่วนจริง: ตัวล่างเล็ก
            ("แบ่งคนละ 1/2", "แบ่งคนละ หนึ่งส่วนสอง"),
            ("ราคาลด 1/3", "ราคาลด หนึ่งส่วนสาม"),
            # ตัวบนมากกว่าตัวล่าง = ค่าคู่ ไม่ใช่เศษส่วน
            ("ความดัน 120/80", "ความดัน หนึ่งร้อยยี่สิบทับแปดสิบ"),
            ("บ้านเลขที่ 123/45", "บ้านเลขที่ หนึ่งร้อยยี่สิบสามทับสี่สิบห้า"),
        ],
    )
    def test_เลือกคำอ่านให้ถูกบริบท(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    def test_ทับหลายชั้นต้องไม่หลุดถึง_TTS(self):
        assert "/" not in expand_numbers_for_speech("ราคา 100/200/300 บาท")

    @pytest.mark.parametrize(
        "text,expected",
        [
            # เลขไทยอยู่ในช่วง [ก-๙] กฎ "ต่อ" เดิมจึงกินเครื่องหมายทับไปก่อน
            ("บ้านเลขที่ ๙๙/๑", "บ้านเลขที่ เก้าสิบเก้าทับหนึ่ง"),
            ("วันที่ ๑๕/๘/๒๕๖๘", "วันที่ สิบห้า สิงหาคม สองพันห้าร้อยหกสิบแปด"),
            # "/" ที่แปลว่า "หรือ" ต้องไม่กลายเป็น "ต่อ"
            ("ชาย/หญิง", "ชาย/หญิง"),
            ("และ/หรือ", "และ/หรือ"),
            # "/" ที่แปลว่า "ต่อ" จริง ๆ ต้องยังทำงาน
            ("ราคา 50 บาท/กก.", "ราคา ห้าสิบ บาท ต่อ กิโลกรัม"),
            ("ขนาด 30x40 ซม.", "ขนาด สามสิบ คูณ สี่สิบ เซนติเมตร"),
        ],
    )
    def test_เลขไทยและทับที่แปลว่าหรือ(self, text, expected):
        assert clean_for_speech(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ยา 500 มก.", "ยา ห้าร้อย มิลลิกรัม"),
            ("น้ำ 500 มล.", "น้ำ ห้าร้อย มิลลิลิตร"),
            ("พื้นที่ 50 ตร.ว.", "พื้นที่ ห้าสิบ ตารางวา"),
        ],
    )
    def test_ตัวย่อหน่วยที่เพิ่มเข้ามา(self, text, expected):
        assert clean_for_speech(text) == expected

    def test_รหัสที่มีตัวอักษรละตินนำหน้า(self):
        assert expand_numbers_for_speech("เที่ยวบิน TG 123") == "เที่ยวบิน TG หนึ่ง สอง สาม"


class Testรหัสยาวและช่วงที่ไม่ใช่เวลา:
    @pytest.mark.parametrize(
        "text", ["ติดต่อ 66812345000", "OTP 12345000", "รหัสอ้างอิง 98765000"]
    )
    def test_รหัสที่ลงท้ายด้วยศูนย์ยังเป็นรหัส(self, text):
        """"ลงท้ายด้วย 000" ไม่พอ — 66812345000 คือเบอร์มือถือที่มีรหัสประเทศนำ"""
        spoken = expand_numbers_for_speech(text)
        assert "ล้าน" not in spoken and "หมื่น" not in spoken, spoken

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ประชากร 70000000 คน", "ประชากร เจ็ดสิบล้าน คน"),
            ("ยอดขาย 12500000 บาท", "ยอดขาย สิบสองล้านห้าแสน บาท"),
        ],
    )
    def test_จำนวนกลมยังอ่านเป็นจำนวน(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "วันที่ 05-10 มิถุนายน",
            "น้ำหนัก 1.50-2.30 กิโลกรัม",
            "เกรดเฉลี่ย 3.00-4.00",
            "ราคา 1.50-2.30 ดอลลาร์",
        ],
    )
    def test_ช่วงที่ไม่ใช่เวลาต้องไม่ทิ้งขีดกลางไว้(self, text):
        """ขีดกลางอ่านออกเสียงไม่ได้ TTS จะกลืนหาย ความหมายของช่วงหายไปด้วย"""
        spoken = expand_numbers_for_speech(text)
        assert "-" not in spoken, spoken
        assert "ถึง" in spoken, spoken
        assert "นาฬิกา" not in spoken, spoken


class Testความเร็วของตัวตัดท่อน:
    """feed() ก้อนใหญ่ก้อนเดียวเคยเป็น O(n²)

    _find_cut สแกนบัฟเฟอร์ทั้งก้อนทุกรอบ ทั้งที่จุดตัดไม่มีทางอยู่เลย max_chars
    ข้อความห้าหมื่นอักขระจึงใช้เวลาหกวินาที
    """

    def test_ข้อความยาวมากต้องเร็ว(self):
        import time

        source = "สวัสดีครับ วันนี้อากาศดีมาก " * 2000
        chunker = SpeechChunker()
        start = time.time()
        chunks = list(chunker.feed(source)) + list(chunker.flush())
        elapsed = time.time() - start

        assert elapsed < 1.0, f"ใช้เวลา {elapsed:.2f}s"
        assert "".join(chunks).replace(" ", "") == source.replace(" ", "")

    def test_ท่อนต้องไม่ยาวเกิน_max_chars_แม้ป้อนก้อนเดียว(self):
        """การสแกนไม่จำกัดเคยคืนจุดตัดที่อยู่เลย max_chars ไปไกล"""
        chunker = SpeechChunker(min_chars=10, max_chars=40)
        source = "กขคง " * 200
        chunks = list(chunker.feed(source)) + list(chunker.flush())
        assert all(len(chunk) <= 40 for chunk in chunks), [
            len(c) for c in chunks if len(c) > 40
        ]


class Testหน่วยและเลขที่เอกสาร:
    @pytest.mark.parametrize(
        "text,expected",
        [
            # ศูนย์นำหน้าของวันที่เป็นแค่การเติมให้เต็มหลัก ไม่มีใครออกเสียง
            ("วันที่ 05-10 มิถุนายน", "วันที่ ห้าถึงสิบ มิถุนายน"),
            ("วันที่ 01-05 พฤษภาคม", "วันที่ หนึ่งถึงห้า พฤษภาคม"),
            # เลขที่เอกสารที่ลงท้ายด้วยปี พ.ศ. คือ "ทับ"
            ("เทอม 1/2567", "เทอม หนึ่งทับสองพันห้าร้อยหกสิบเจ็ด"),
            ("ไตรมาส 3/2567", "ไตรมาส สามทับสองพันห้าร้อยหกสิบเจ็ด"),
            # ลำดับที่ X จากทั้งหมด Y ไม่ใช่เศษส่วน
            ("งวดที่ 1/12", "งวดที่ หนึ่งจากสิบสอง"),
            ("หน้า 1/20", "หน้า หนึ่งจากยี่สิบ"),
            # เศษส่วนจริงยังเป็น "ส่วน"
            ("แบ่งคนละ 1/2", "แบ่งคนละ หนึ่งส่วนสอง"),
        ],
    )
    def test_เลือกคำอ่านตามบริบท(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["จองห้อง 1203 เดือนหน้านะครับ", "จองห้อง 1203 สัปดาห์หน้านะครับ"],
    )
    def test_เดือนหน้าสัปดาห์หน้าไม่ใช่ลักษณนาม(self, text):
        """"เดือนหน้า" "สัปดาห์หน้า" พบบ่อยกว่าการใช้เป็นลักษณนามมาก"""
        assert "พัน" not in expand_numbers_for_speech(text), text

    @pytest.mark.parametrize(
        "text,expected",
        [
            # หน่วยที่ตามหลังหนักกว่ากฎ "ยาวเกินแปดหลักให้อ่านทีละตัว"
            ("ยอดขาย 12345678 บาท", "ยอดขาย สิบสองล้านสามแสนสี่หมื่นห้าพันหกร้อยเจ็ดสิบแปด บาท"),
            ("ทำงาน 9.00 ถึง 17.00", "ทำงาน เก้านาฬิกาถึงสิบเจ็ดนาฬิกา"),
        ],
    )
    def test_หน่วยที่ตามหลังชนะกฎความยาว(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    def test_เบอร์ยาวยังอ่านทีละตัว(self):
        assert "ล้าน" not in expand_numbers_for_speech("เบอร์ 66812345000")

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ทานยา 2 เม็ด/วัน", "ทานยา สอง เม็ด ต่อ วัน"),
            ("น้ำตาล 110 มก./ดล.", "น้ำตาล หนึ่งร้อยสิบ มิลลิกรัม ต่อ เดซิลิตร"),
            ("ยา 500 mg", "ยา ห้าร้อย มิลลิกรัม"),
            ("มีที่นั่ง 250 ที่นั่ง", "มีที่นั่ง สองร้อยห้าสิบ ที่นั่ง"),
        ],
    )
    def test_หน่วยทางการแพทย์และอัตราส่วน(self, text, expected):
        assert clean_for_speech(text) == expected


class Testเครื่องหมายนาฬิกาต้องมาก่อนหน่วย:
    """`น.` ท้ายช่วงเป็นหลักฐานชิ้นเดียวที่กำกวมไม่ได้

    ของเดิมเช็คหน่วยที่ตามหลังก่อน "ทำงาน 8.00-17.00 น. ชั่วโมงพัก..." จึงเห็น
    "ชั่วโมง" แล้วอ่านเป็นทศนิยม ทั้งยังกลืน "น." หายไปเพราะ regex กินเข้ามาแล้ว
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                "ทำงาน 8.00-17.00 น. ชั่วโมงพัก",
                "ทำงาน แปดนาฬิกาถึงสิบเจ็ดนาฬิกา ชั่วโมงพัก",
            ),
            (
                "สอบ 13.00-15.00 น. คะแนนเต็ม",
                "สอบ สิบสามนาฬิกาถึงสิบห้านาฬิกา คะแนนเต็ม",
            ),
        ],
    )
    def test_มีนหน่วยที่ตามหลังต้องแพ้(self, text, expected):
        assert expand_numbers_for_speech(text) == expected

    def test_ไม่มีนก็ยังให้หน่วยชี้ขาดเหมือนเดิม(self):
        got = expand_numbers_for_speech("วิ่ง 8.00-17.00 กิโลเมตร")
        assert "นาฬิกา" not in got, got

    def test_นต้องไม่หายไปจากผลลัพธ์(self):
        """แม้เส้นทางที่คืนเป็น "ช่วง" เฉย ๆ ก็ห้ามกลืน `น.` ทิ้ง"""
        for text in ["ทำงาน 8.00-17.00 น. ชั่วโมงพัก", "20-21 น."]:
            got = expand_numbers_for_speech(text)
            assert "จุด" not in got or "น." in got, got


class Testตัดท่อนระหว่างสตรีมต้องไม่ทำความหมายพัง:
    """จุดตัดที่บังคับตอนบัฟเฟอร์ชนเพดานเคยตัดโดยไม่ตรวจอะไรเลย

    ตัวช่วยกันตัดกลางคลัสเตอร์ต้องมองเห็นอักขระ *ถัดจาก* จุดตัดจึงจะตัดสินได้
    ตอนบัฟเฟอร์ยาวเท่าเพดานพอดียังไม่มีตัวถัดไป มันจึงคืนจุดเดิมทันที
    """

    @staticmethod
    def _สตรีม(text: str, ทีละ: int = 1) -> list[str]:
        chunker = SpeechChunker()
        out: list[str] = []
        for i in range(0, len(text), ทีละ):
            out.extend(chunker.feed(text[i : i + ทีละ]))
        out.extend(chunker.flush())
        return out

    def test_ตัวเลขต้องไม่ถูกตัดครึ่ง(self):
        text = (
            "แจ้งยอดค่าใช้จ่ายของโครงการนะคะค่าออกแบบ 152,000 บาท"
            "ค่าก่อสร้าง 1,845,000 บาทค่าตกแต่งที่พักอาศัยภายใน 673,500 บาท"
            "ค่าเฟอร์นิเจอร์ 428,900 บาทรวมทั้งหมด 3,099,400 บาทค่ะ"
        )
        ท่อน = self._สตรีม(text)
        assert any("3,099,400" in c for c in ท่อน), ท่อน
        # จุลภาคที่หลุดถึง TTS แปลว่าตัวแปลงตัวเลขมองไม่เห็นรูปที่มันรู้จักแล้ว
        for c in ท่อน:
            assert "จุลภาค" not in clean_for_speech(c), c

    def test_ท่อนต้องไม่ขึ้นต้นด้วยเครื่องหมายท้ายคำ(self):
        text = (
            "แจ้งยอดค่าใช้จ่ายของโครงการนะคะค่าออกแบบ 152,000 บาท"
            "ค่าก่อสร้าง 1,845,000 บาทค่าตกแต่งห้องภายใน 673,500 บาท"
            "ค่าเฟอร์นิเจอร์ 428,900 บาทรวมทั้งหมด 3,099,400 บาทค่ะ"
        )
        for c in self._สตรีม(text):
            assert c[0] not in "ะำๅๆฯ", repr(c)
            assert unicodedata.category(c[0]) != "Mn", repr(c)

    @pytest.mark.parametrize("ทีละ", [1, 2, 3, 7, 40])
    def test_ป้อนทีละกี่ตัวก็ต้องได้ผลเหมือนกัน(self, ทีละ):
        text = "ยอดรวม 3,099,400 บาท ค่าส่ง 250 บาท รวมสุทธิ 3,099,650 บาทค่ะ"
        ท่อน = self._สตรีม(text, ทีละ)
        assert "".join(ท่อน).replace(" ", "") == text.replace(" ", "")

    def test_ตัวเลขที่ยาวกว่าท่อนทั้งท่อนต้องยังคืบหน้า(self):
        """กรณีที่ไม่มีทางเลือกอื่น — ห้ามวนไม่จบ และห้ามได้ท่อนสั้นจู๋"""
        ท่อน = self._สตรีม("9" * 400)
        assert "".join(ท่อน) == "9" * 400
        assert all(len(c) > 1 for c in ท่อน[:-1]), [len(c) for c in ท่อน]


class Testตัวย่อไทยต้องไม่ถูกตัดกลางคำ:
    """`_SENT_END` ยิงใส่จุดที่มีพยัญชนะสองตัวนำหน้า

    "มี.ค." "ก.พ." รอดเพราะตัวหน้าจุดเป็นสระหรือพยัญชนะตัวเดียว แต่ "เม.ย."
    "ตร.ม." "ลบ.ซม." ไม่รอด — พังเฉพาะบางตัวจึงจับได้ยากกว่า
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ประชุมวันที่ 5 เม.ย. 2569 นะครับ", "เมษายน"),
            ("ห้องนี้พื้นที่ 120 ตร.ม. ราคาสูง", "ตารางเมตร"),
            ("รถวิ่งเร็ว 90 กม./ชม. ครับ", "กิโลเมตร ต่อ ชั่วโมง"),
            ("ปริมาตร 500 ลบ.ซม. ครับ", "ลูกบาศก์เซนติเมตร"),
            ("เกิดปี พ.ศ. 2540 ครับ", "พุทธศักราช"),
        ],
    )
    def test_ต้องไม่ถูกตัดแล้วอ่านทีละพยางค์(self, text, expected):
        assert split_sentences(text) == [text], split_sentences(text)
        assert expected in clean_for_speech(text)

    def test_ตัวย่อทุกตัวในตารางต้องไม่ถูกตัด(self):
        """ไล่พิมพ์มือแล้วรอบหน้าที่เพิ่มหน่วยใหม่จะพังซ้ำที่เดิม

        ตรวจที่พฤติกรรมจริง ไม่ใช่ที่หน้าตาของรายการ — ตัวย่อที่มีพยัญชนะตัวเดียว
        หน้าจุด ("ก.พ.") ไม่เคยเสี่ยงอยู่แล้ว เพราะตัวตรวจจุดจบต้องการอักขระคำ
        สองตัวหน้าจุด
        """
        from thaivoice.thai_text import _MONTH_ABBR, _THAI_UNIT_ABBR

        for abbreviation in list(_MONTH_ABBR) + list(_THAI_UNIT_ABBR):
            text = f"ค่า 5 {abbreviation} ต่อไป"
            assert split_sentences(text) == [text], abbreviation

    def test_จุดจบประโยคจริงต้องยังถูกตัด(self):
        """stem ที่ยาวตัวอักษรเดียวทำให้ทุกคำที่ลงท้ายด้วยตัวนั้นถูกนับเป็นตัวย่อ

        "อากาศดีมาก." จึงไม่ถูกตัดอีกเลย ซึ่งทำให้ท่อนแรกของเสียงมาช้าลง
        """
        from thaivoice.thai_text import _sentence_ends

        for text in ["อากาศดีมาก. พรุ่งนี้ฝนตก", "งานเสร็จครบ. ไปกันเลย"]:
            assert list(_sentence_ends(text)), text

    def test_ท่อนแรกต้องมาถึงเร็วเหมือนเดิม(self):
        """จุดจบประโยคที่หายไปทำให้ตัวตัดท่อนต้องรอจนกว่าจะเจอเว้นวรรค"""
        text = (
            "ผมไปทำงานมาทั้งวันเลยครับวันนี้ประชุมสามรอบติดกันเลยรู้สึกเหนื่อยมาก. "
            "พรุ่งนี้ตั้งใจว่าจะพักผ่อนให้เต็มที่"
        )
        chunker = SpeechChunker()
        out: list[str] = []
        for i in range(0, len(text), 4):
            out.extend(chunker.feed(text[i : i + 4]))
        assert out, "ต้องได้ท่อนแรกก่อนข้อความจบ"
        # ท่อนแรกต้องจบที่จุด ไม่ใช่ไหลต่อไปจนเจอเว้นวรรคหรือชนเพดาน
        assert out[0].rstrip().endswith("."), out[0]


class Testทับที่แปลว่าต่อกับทับที่แปลว่าหรือ:
    @pytest.mark.parametrize(
        "text",
        ["กรอกวัน/เดือน/ปีเกิด", "ระบุสถานที่/เวลานัดหมาย", "แบบรายวัน/รายเดือน"],
    )
    def test_ไม่มีจำนวนกำกับต้องไม่อ่านว่าต่อ(self, text):
        """"วัน/เดือน/ปีเกิด" เป็นสำนวนไทยที่พบทุกวัน "/" ตรงนั้นแปลว่า "หรือ" """
        assert "ต่อ" not in clean_for_speech(text), clean_for_speech(text)

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ซื้อ 50 บาท/กก.", "ห้าสิบ บาท ต่อ กิโลกรัม"),
            ("ทานยา 2 เม็ด/วัน", "สอง เม็ด ต่อ วัน"),
            ("ราคา 1,200 บาท/ตร.ม.", "หนึ่งพันสองร้อย บาท ต่อ ตารางเมตร"),
        ],
    )
    def test_มีจำนวนกำกับต้องอ่านว่าต่อและตัวเลขต้องไม่หาย(self, text, expected):
        assert expected in clean_for_speech(text)


class Testช่วงตัวเลขกับสกอร์การแข่งขัน:
    @pytest.mark.parametrize(
        "text",
        ["ได้คะแนนสอบระหว่าง 85-90 คะแนน", "คะแนนเฉลี่ยอยู่ที่ 70-80"],
    )
    def test_คำบอกช่วงต้องชนะบริบทคะแนน(self, text):
        """"ต่อ" แปลว่าสกอร์การแข่งขัน ความหมายคนละเรื่องกับช่วง"""
        got = clean_for_speech(text)
        assert "ถึง" in got and "ต่อ" not in got, got

    @pytest.mark.parametrize("text", ["ทีมชนะ 3-1", "ผลบอล 2-1"])
    def test_สกอร์จริงต้องยังอ่านว่าต่อ(self, text):
        assert "ต่อ" in clean_for_speech(text)


class Testวันที่รูปอื่นและเลขที่หนังสือ:
    @pytest.mark.parametrize(
        "text,expected",
        [
            # รูปที่คนไทยเขียนบ่อยที่สุดรูปหนึ่ง เคยตกไปให้กฎ "ทับ" ของเลขที่บ้าน
            ("นัดวันที่ 15/8/68 นะคะ", "สิบห้า สิงหาคม หก แปด"),
            # ขีดกลางเคยค้างทั้งสองตัว TTS กลืนหาย เหลือตัวเลขติดกันหมดความหมาย
            ("ครบกำหนด 2568-08-15", "สิบห้า สิงหาคม สองพันห้าร้อยหกสิบแปด"),
            ("วันที่ 15/8/2568", "สิบห้า สิงหาคม สองพันห้าร้อยหกสิบแปด"),
        ],
    )
    def test_ต้องอ่านเป็นวันเดือนปี(self, text, expected):
        assert expected in clean_for_speech(text)

    @pytest.mark.parametrize(
        "text,expected",
        [("บ้านเลขที่ 123/45", "ทับ"), ("ได้ 18/20", "จาก"), ("3/4 ถ้วย", "ส่วน")],
    )
    def test_รูปที่ไม่ใช่วันที่ต้องไม่เปลี่ยน(self, text, expected):
        assert expected in clean_for_speech(text)

    def test_เลขที่หนังสือราชการต้องอ่านว่าทับ(self):
        """"ศธ 0506/ว 123" — ข้างหลังทับเป็นอักษรไทยตัวเดียว กฎเลข/เลขไม่ครอบ"""
        got = clean_for_speech("หนังสือที่ ศธ 0506/ว 123")
        assert "ทับ" in got and "/" not in got, got


class Testทับที่หลุดเพราะลำดับการแปลง:
    """กฎ "ต่อ" ต้องรันอีกรอบหลังแปลงตัวย่อหน่วย

    รายการหน่วยมี "บาท" แต่ไม่มี "บ." พอ "บ." ถูกแปลงเป็น "บาท" ทีหลัง
    กฎก็ผ่านไปแล้ว เหลือ "/" ดิบ ๆ ให้ TTS และ "ต่อ" หายทั้งคำ
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ราคา 38.50 บ./ลิตร", "ต่อ ลิตร"),
            ("แบตเตอรี่ 5000 mAh/ชม.", "ต่อ ชั่วโมง"),
            ("ยา 500 mg/วัน", "ต่อ วัน"),
        ],
    )
    def test_ต้องไม่เหลือทับดิบ(self, text, expected):
        got = clean_for_speech(text)
        assert expected in got and "/" not in got, got


class Testความเร็วของตัวจับจำนวนเงิน:
    def test_สตริงตัวเลขยาวต้องไม่ทำให้ค้าง(self):
        """\\d[\\d,]* ที่ไม่มีเพดานทำให้ทุกตำแหน่งกินตัวเลขทั้งพรืดแล้วถอยกลับ

        เส้นทางหลักตัดเป็นท่อนสั้นอยู่แล้ว แต่ tts.py รับข้อความอะไรก็ได้
        ที่ผู้เรียกส่งมา
        """
        import time

        เริ่ม = time.monotonic()
        clean_for_speech("9" * 50000)
        ใช้เวลา = time.monotonic() - เริ่ม
        assert ใช้เวลา < 3.0, f"ใช้เวลา {ใช้เวลา:.1f} วินาที"


class Testหน่วยสองฝั่งทับต้องถูกอ่านครบ:
    """กฎ "ต่อ" กับการขยายตัวย่อสลับลำดับกันไม่ได้

    หน่วยที่อยู่ *หลัง* ทับเพิ่งมีคำว่า "ต่อ" มานำหน้าตอนกฎรอบสองทำงาน
    ถ้าไม่ขยายตัวย่อปิดท้ายอีกรอบ TTS จะอ่าน "กอ กอ" แทน "กิโลกรัม"
    """

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ราคา 25 บ./กก.", "ยี่สิบห้า บาท ต่อ กิโลกรัม"),
            ("ค่าส่ง 30 บ./ชม.", "สามสิบ บาท ต่อ ชั่วโมง"),
            ("อัตรา 12 ลบ.ม./ชั่วโมง", "สิบสอง ลูกบาศก์เมตร ต่อ ชั่วโมง"),
            ("อัตรา 12 มม./ชม.", "สิบสอง มิลลิเมตร ต่อ ชั่วโมง"),
        ],
    )
    def test_ทั้งสองฝั่งต้องเป็นชื่อเต็ม(self, text, expected):
        got = clean_for_speech(text)
        assert expected in got and "/" not in got, got


class Testเลขที่บ้านที่หน้าตาเหมือนวันที่:
    @pytest.mark.parametrize(
        "text",
        ["บ้านเลขที่ 12/5/45", "ห้อง 3/4/12", "ที่อยู่ 9/4/76", "เลขที่ 12/5/2545"],
    )
    def test_บริบทที่อยู่ต้องชนะการอ่านเป็นวันที่(self, text):
        got = clean_for_speech(text)
        assert "ทับ" in got, got
        assert "มกราคม" not in got and "พฤษภาคม" not in got and "เมษายน" not in got

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("นัดวันที่ 15/8/68 นะคะ", "สิงหาคม"),
            ("วันที่ 15/8/2568", "สิงหาคม"),
            ("ประชุม 5/4/69", "เมษายน"),
        ],
    )
    def test_วันที่จริงต้องยังอ่านเป็นวันเดือนปี(self, text, expected):
        assert expected in clean_for_speech(text)


class Testสกอร์ที่ใช้คำว่าอยู่ที่:
    """"อยู่ที่" กับ "ช่วง" ใช้บอกสกอร์พอ ๆ กับใช้บอกช่วง"""

    @pytest.mark.parametrize(
        "text", ["สกอร์ตอนนี้อยู่ที่ 2-1", "คะแนนตอนนี้อยู่ที่ 3-2"]
    )
    def test_สกอร์ต้องอ่านว่าต่อ(self, text):
        got = clean_for_speech(text)
        assert "ต่อ" in got and "ถึง" not in got, got

    @pytest.mark.parametrize(
        "text", ["ได้คะแนนสอบระหว่าง 85-90 คะแนน", "คะแนนเฉลี่ย 70-80"]
    )
    def test_ช่วงจริงต้องยังอ่านว่าถึง(self, text):
        got = clean_for_speech(text)
        assert "ถึง" in got and "ต่อ" not in got, got


class Testวันที่ที่เป็นไปไม่ได้ต้องไม่ถูกอ่านเป็นวันที่:
    @pytest.mark.parametrize("text", ["15/13/68", "32/8/68", "0/8/68", "15/0/68"])
    def test_เดือนหรือวันเกินช่วงต้องตกไปเป็นทับ(self, text):
        got = clean_for_speech(text)
        assert "ทับ" in got, got

    @pytest.mark.parametrize("text", ["15/8/68", "1/1/70", "31/12/99"])
    def test_วันที่ที่เป็นไปได้ต้องอ่านเป็นวันเดือนปี(self, text):
        got = clean_for_speech(text)
        assert "ทับ" not in got, got


class Testที่ในฐานะลักษณนาม:
    """"ที่" ลงท้ายคำนามไทยเยอะ แต่ข้อบังคับเรื่องจำนวนนำหน้าแยกได้แล้ว"""

    @pytest.mark.parametrize(
        "text,expected",
        [("นั่งได้ 20 ที่/คัน", "ยี่สิบ ที่ ต่อ คัน"), ("มี 4 ที่/โต๊ะ", "สี่ ที่ ต่อ โต๊ะ")],
    )
    def test_มีจำนวนนำหน้าต้องอ่านว่าต่อ(self, text, expected):
        got = clean_for_speech(text)
        assert expected in got and "/" not in got, got

    @pytest.mark.parametrize(
        "text", ["ระบุสถานที่/เวลานัดหมาย", "เลือกที่/วันที่สะดวก"]
    )
    def test_ไม่มีจำนวนต้องไม่อ่านว่าต่อ(self, text):
        assert "ต่อ" not in clean_for_speech(text), clean_for_speech(text)


class Testเส้นทางที่ใช้ตัวตัดคำของ_pythainlp:
    """เส้นทางนี้ไม่เคยถูกรันในชุดทดสอบเลย เพราะ pythainlp ไม่ได้ติดตั้ง

    แต่มันทำงานจริงในเครื่องที่ติดตั้งไว้ ต้องยิงด้วยของปลอมอย่างน้อยหนึ่งครั้ง
    ไม่งั้นโค้ดที่ผู้ใช้จริงรันอยู่ไม่เคยถูกทดสอบเลย
    """

    @pytest.fixture
    def ตัวตัดคำปลอม(self, monkeypatch):
        import sys
        import types

        import thaivoice.thai_text as tt

        def word_tokenize(text):
            """ตัดทุก 5 ตัวอักษร — พอให้เห็นว่าเส้นทางนี้ถูกใช้จริง"""
            return [text[i : i + 5] for i in range(0, len(text), 5)]

        pythainlp = types.ModuleType("pythainlp")
        tokenize = types.ModuleType("pythainlp.tokenize")
        tokenize.word_tokenize = word_tokenize
        pythainlp.tokenize = tokenize
        monkeypatch.setitem(sys.modules, "pythainlp", pythainlp)
        monkeypatch.setitem(sys.modules, "pythainlp.tokenize", tokenize)
        monkeypatch.setattr(tt, "_word_tokenizer_ok", True)
        yield
        monkeypatch.setattr(tt, "_word_tokenizer_ok", None)

    def test_เส้นทางนี้ถูกใช้จริง(self, ตัวตัดคำปลอม):
        from thaivoice.thai_text import thai_word_tokenizer_available

        assert thai_word_tokenizer_available()

    def test_ตัวเลขต้องไม่ถูกตัดครึ่งในเส้นทางนี้ด้วย(self, ตัวตัดคำปลอม):
        """ตัวตัดคำปลอมตัดทุก 5 ตัวอักษร ซึ่งตกกลางตัวเลขแน่นอน"""
        # วางตัวเลขให้คร่อมเพดาน 160 ตัวอักษรพอดี จุดตัดที่ตัวตัดคำเสนอ
        # (ทุก 5 ตัวอักษร) จึงตกกลางตัวเลขแน่นอน
        text = "ก" * 156 + "3,099,400" + "ก" * 60
        chunker = SpeechChunker()
        out: list[str] = []
        for ch in text:
            out.extend(chunker.feed(ch))
        out.extend(chunker.flush())

        assert "".join(out) == text
        joined = "".join(out)
        pos = 0
        for chunk in out[:-1]:
            pos += len(chunk)
            ก่อน, หลัง = joined[pos - 1], joined[pos]
            assert not (
                ก่อน in "0123456789,." and หลัง in "0123456789,."
            ), f"ตัดกลางตัวเลขที่ {joined[pos - 8 : pos + 4]!r}"
