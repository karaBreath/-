"""ทดสอบยูทิลิตี้ภาษาไทย — ส่วนที่กฎภาษาไทยอยู่ตรงนี้ทั้งหมด"""

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
            ("ขอบคุณค่า", "female"),
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

    def test_ไม่แตะเวลาและ_IP(self):
        assert expand_numbers_for_speech("นัดกัน 20:30 นะ") == "นัดกัน 20:30 นะ"
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
