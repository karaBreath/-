"""ยูทิลิตี้ภาษาไทยสำหรับระบบสนทนาด้วยเสียง

ภาษาไทยมีปัญหาเฉพาะตัวที่ระบบเสียงทั่วไปจัดการไม่ได้:

1. **ไม่มีเว้นวรรคระหว่างคำ และไม่มีเครื่องหมายจบประโยค** — จะตัดข้อความที่ไหล
   ออกมาจากโมเดลเป็นท่อน ๆ เพื่อส่งให้ TTS พูดทันที (ลด latency) ไม่ได้ตรง ๆ
   ต้องใช้กฎเว้นวรรค + คำลงท้าย + ความยาว และห้ามตัดกลางคำหรือกลางคลัสเตอร์
2. **คำลงท้าย ครับ/ค่ะ** — บอกเพศของ *ผู้พูด* ไม่ใช่ผู้ฟัง และคำลงท้ายหลายคำ
   ไปพ้องกับคำธรรมดา (ค่า ขา คะแนน คับ) จึงต้องดูเฉพาะท้ายประโยค
3. **markdown อ่านออกเสียงไม่ได้** — ดอกจัน หัวข้อ โค้ดบล็อก ต้องถูกถอดทิ้ง
4. **ตัวเลข** — TTS ไทยอ่านเลขยาว ๆ ผิดบ่อย จึงแปลงเป็นคำอ่านให้ก่อน และอ่าน
   เบอร์โทรทีละตัว

ทุกฟังก์ชันในไฟล์นี้เป็น pure function ไม่พึ่ง network จึงทดสอบได้ตรง ๆ
ถ้าติดตั้ง ``pythainlp`` ไว้ จะใช้ตัวตัดประโยคและตัวตัดคำของ pythainlp
เพื่อความแม่นยำขึ้น
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterator

__all__ = [
    "THAI_PARTICLES",
    "SpeechChunker",
    "clean_for_speech",
    "detect_particle",
    "expand_numbers_for_speech",
    "particle_for_gender",
    "read_digits",
    "split_sentences",
    "thai_number_to_words",
    "thai_segmenter_engine",
    "thai_ratio",
    "thai_word_tokenizer_available",
    "normalize_transcript",
]

# ── ตารางคำอ่านตัวเลข ────────────────────────────────────────────────────────
_DIGITS = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]
_PLACES = ["", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน"]
_THAI_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("๐๑๒๓๔๕๖๗๘๙")}

# คำลงท้ายสุภาพที่ "บอกเพศของผู้พูดได้จริง"
#
# ตั้งใจไม่ใส่ จ้ะ/จ๊ะ/จ้า เพราะเป็นคำลงท้ายที่ใช้ได้ทั้งสองเพศ (พ่อพูดกับลูกก็ใช้
# จ๊ะ) การเดาเพศจากคำพวกนี้ผิดบ่อยกว่าถูก และไม่ใส่ ขา เพราะพ้องกับคำว่าขาที่
# เป็นอวัยวะ ซึ่งพบบ่อยกว่าการใช้เป็นคำลงท้ายมาก
THAI_PARTICLES: dict[str, tuple[str, ...]] = {
    "male": ("ครับผม", "ครับ", "คร้าบ", "คับ", "ครัช"),
    "female": ("ค่ะ", "คะ", "ค่า"),
}

# คำลงท้ายที่ใช้เป็น "จุดตัดประโยค" ได้ — กว้างกว่าตารางเดาเพศ เพราะแค่ต้องการ
# รู้ว่าประโยคจบ ไม่ได้ต้องการรู้เพศ
_BREAK_PARTICLES = (
    "ครับผม", "ครับ", "คร้าบ", "คับ", "ครัช", "ค่ะ", "คะ", "ค่า",
    "นะคะ", "นะครับ", "จ้ะ", "จ๊ะ", "จ้า", "ฮะ",
)

_THAI_RANGE = re.compile(r"[฀-๿]")

# สระหน้า — เขียนไว้ก่อนพยัญชนะที่ออกเสียงจริง จึงห้ามตัดหลังตัวพวกนี้
_LEADING_VOWELS = "เแโใไ"

# หน่วย/ตัวย่อที่ TTS ไทยมักอ่านผิด
#
# ตัวย่อภาษาอังกฤษต้องเทียบแบบมีขอบเขตคำ ไม่งั้น "KBank" จะกลายเป็น
# "กิโลไบต์ank" และ "LGBT" จะกลายเป็น "L กิกะไบต์ T"
_ASCII_UNITS = {
    "km/h": "กิโลเมตรต่อชั่วโมง",
    "km": "กิโลเมตร",
    "cm": "เซนติเมตร",
    "mm": "มิลลิเมตร",
    "kg": "กิโลกรัม",
    "GB": "กิกะไบต์",
    "MB": "เมกะไบต์",
    "KB": "กิโลไบต์",
    "TB": "เทระไบต์",
}
_ASCII_UNIT_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(u) for u in sorted(_ASCII_UNITS, key=len, reverse=True)) + r")(?![A-Za-z])"
)

# สัญลักษณ์ที่ต้องแปลง *ก่อน* กรองอักขระประเภทสัญลักษณ์ทิ้ง
# (° เป็นหมวด So ถ้ากรองก่อนจะไม่เหลืออะไรให้เทียบ)
_SYMBOL_UNITS = [
    ("°C", " องศาเซลเซียส "),
    ("°F", " องศาฟาเรนไฮต์ "),
    ("°", " องศา "),
    ("%", " เปอร์เซ็นต์ "),
    ("&", " และ "),
    ("@", " แอท "),
]


# ── ทำความสะอาดข้อความก่อนอ่านออกเสียง ──────────────────────────────────────
def clean_for_speech(text: str, expand_numbers: bool = True) -> str:
    """ถอด markdown / emoji / URL ออก แล้วคืนข้อความที่ TTS อ่านแล้วฟังรู้เรื่อง

    ขึ้นบรรทัดใหม่ถูกเก็บไว้ เพราะเป็นจุดตัดประโยคที่เชื่อถือได้ที่สุดของภาษาไทย

    >>> clean_for_speech("**สวัสดี** ครับ\\n- ข้อหนึ่ง")
    'สวัสดี ครับ\\nข้อหนึ่ง'
    """
    if not text:
        return ""

    s = text

    # โค้ดบล็อก -> บอกสั้น ๆ ว่ามีโค้ด แทนที่จะอ่านทั้งก้อน
    s = re.sub(r"```[\s\S]*?```", " (มีโค้ดแสดงอยู่บนหน้าจอ) ", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)

    # ลิงก์ markdown [ข้อความ](url) -> เก็บเฉพาะข้อความ
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    # URL เปล่า ๆ
    s = re.sub(r"https?://\S+", " ลิงก์ ", s)

    # ตัวหนา/ตัวเอียง/ขีดฆ่า
    s = re.sub(r"(\*\*\*|\*\*|\*|___|__|~~)", "", s)
    # หัวข้อ / บุลเล็ต / เลขข้อ ที่ต้นบรรทัด
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
    s = re.sub(r"(?m)^\s*[-*+•]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+[.)]\s+", "", s)
    # เส้นคั่นและตารางแบบ markdown
    s = re.sub(r"(?m)^\s*([-*_]\s*){3,}$", " ", s)
    s = s.replace("|", " ")

    # แปลงสัญลักษณ์ที่มีความหมายก่อน แล้วค่อยกรองสัญลักษณ์ที่เหลือทิ้ง
    for symbol, spoken in _SYMBOL_UNITS:
        s = s.replace(symbol, spoken)

    # emoji และสัญลักษณ์ที่อ่านไม่ได้
    s = "".join(
        ch for ch in s if unicodedata.category(ch) not in {"So", "Sk", "Cf", "Cs"}
    )

    # หน่วยภาษาอังกฤษ — ต้องมีขอบเขตคำ ไม่งั้นไปกินตัวอักษรกลางคำ
    s = _ASCII_UNIT_RE.sub(lambda m: f" {_ASCII_UNITS[m.group(1)]} ", s)

    # เลขไทย -> เลขอารบิก แล้วค่อยแปลงเป็นคำอ่าน
    s = s.translate(_THAI_DIGIT_MAP)
    if expand_numbers:
        s = expand_numbers_for_speech(s)

    # ยุบช่องว่าง
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


# ── ทำความสะอาดผลถอดเสียง ───────────────────────────────────────────────────
#
# ตัวจับคำซ้ำต้องยึด "ทั้งคำ" ที่คั่นด้วยช่องว่างเท่านั้น
#
# ของเดิมใช้ ``(\S+?)(?:\s*\1){2,}`` ซึ่ง \S+? แบบ non-greedy จับได้แม้แต่ตัวอักษร
# เดียว และ \s* ยอมให้ไม่มีช่องว่างคั่น ผลคือมันไปยุบเลขซ้ำในตัวเลขก้อนเดียว:
# "เบอร์ 0811111111" กลายเป็น "เบอร์ 081" และ "ราคา 1000 บาท" กลายเป็น
# "ราคา 10 บาท" ซึ่งทำลายข้อมูลผู้ใช้ทุกครั้งที่พูดเบอร์โทร ราคา หรือปี
_ASR_REPEAT = re.compile(r"(?<!\S)(\S{2,}?)(?:[ \t]+\1){2,}(?!\S)")


def _collapse_repeat(match: re.Match[str]) -> str:
    token = match.group(1)
    # ตัวเลขล้วนอาจเป็นข้อมูลจริง (รหัส เบอร์ ราคา) อย่าไปยุบ
    if token.isdigit():
        return match.group(0)
    return token


def normalize_transcript(text: str) -> str:
    """เก็บกวาดข้อความที่ได้จาก STT ก่อนส่งให้โมเดล

    Whisper ภาษาไทยชอบแถมช่องว่างเกิน จุดไข่ปลา และวนซ้ำทั้งคำท้ายประโยค

    >>> normalize_transcript("ขอบคุณ ขอบคุณ ขอบคุณ ขอบคุณ")
    'ขอบคุณ'
    >>> normalize_transcript("เบอร์ผม 0811111111 ครับ")
    'เบอร์ผม 0811111111 ครับ'
    """
    if not text:
        return ""
    s = text.strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\.{3,}", " ", s)
    s = _ASR_REPEAT.sub(_collapse_repeat, s)
    return re.sub(r"[ \t]+", " ", s).strip()


def thai_ratio(text: str) -> float:
    """สัดส่วนอักขระไทยต่ออักขระที่ไม่ใช่ช่องว่าง (0.0-1.0)"""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _THAI_RANGE.match(c)) / len(chars)


# ── คำลงท้ายสุภาพ ───────────────────────────────────────────────────────────
#
# ต้องดูเฉพาะ "ท้ายประโยค" เท่านั้น เพราะคำลงท้ายไทยพ้องกับคำธรรมดาเยอะมาก:
# ค่า (ค่าไฟ) ขา (ปวดขา) คะ (คะแนน) คับ (เสื้อคับ) ถ้าค้นแบบ substring
# "ค่าไฟเดือนนี้แพงมาก" จะถูกตัดสินว่าเป็นผู้หญิงพูด
_PARTICLE_TAIL = re.compile(
    r"(?P<lead>.{0,3}?)"
    r"(?P<particle>" + "|".join(
        re.escape(p) for group in (
            THAI_PARTICLES["male"], THAI_PARTICLES["female"]
        ) for p in group
    ) + r")"
    r"\s*[.!?…\"'”’)\]]*$"
)

_QUOTE_MARKERS = ("ว่า", "บอกว่า", "พูดว่า")


def detect_particle(text: str) -> str | None:
    """เดาเพศของ *ผู้พูด* จากคำลงท้ายท้ายประโยค คืน ``"male"`` / ``"female"`` / ``None``

    >>> detect_particle("ผมชื่อสมชายครับ")
    'male'
    >>> detect_particle("ค่าไฟเดือนนี้แพงมากเลย")
    """
    if not text:
        return None
    tail = text.strip()
    match = _PARTICLE_TAIL.search(tail)
    if not match:
        return None

    # คำลงท้ายที่ตามหลัง "ว่า" มักเป็นการยกคำพูดคนอื่น ไม่ใช่คำลงท้ายของผู้พูดเอง
    lead = match.group("lead")
    if any(lead.endswith(marker) for marker in _QUOTE_MARKERS):
        return None

    particle = match.group("particle")
    for gender, particles in THAI_PARTICLES.items():
        if particle in particles:
            return gender
    return None


def particle_for_gender(gender: str | None) -> str:
    """คำลงท้ายที่ *คนเพศนั้น* ใช้เมื่อพูด — ค่าเริ่มต้นเป็น 'ครับ'

    หมายเหตุสำคัญ: ในภาษาไทยคำลงท้ายบอกเพศของ "คนพูด" ไม่ใช่ "คนฟัง" ฟังก์ชันนี้
    จึงใช้ตอบคำถามว่า "คนคนนี้พูดลงท้ายว่าอะไร" เท่านั้น ห้ามเอาไปใช้เลือกคำลงท้าย
    ให้บอทจากเพศของผู้ฟัง ไม่งั้นบอทเสียงผู้หญิงจะลงท้ายว่า "ครับ"
    """
    return "ค่ะ" if gender == "female" else "ครับ"


# ── คำอ่านตัวเลข ────────────────────────────────────────────────────────────
def thai_number_to_words(n: int) -> str:
    """แปลงจำนวนเต็มเป็นคำอ่านภาษาไทย

    >>> thai_number_to_words(21)
    'ยี่สิบเอ็ด'
    >>> thai_number_to_words(1000001)
    'หนึ่งล้านเอ็ด'
    """
    return _read_number(n, False)


def _read_number(n: int, has_higher_place: bool) -> str:
    """อ่านตัวเลข โดย ``has_higher_place`` บอกว่ามีหลักที่ใหญ่กว่านำหน้าอยู่แล้วไหม

    จำเป็นเพราะกฎ "หนึ่ง -> เอ็ด" ขึ้นกับว่ามีหลักสูงกว่านำอยู่หรือไม่ และหลักล้าน
    ใช้การเรียกซ้ำ ทำให้เศษที่เหลือมองไม่เห็นหลักล้านที่นำหน้า
    ถ้าไม่ส่งค่านี้ไปด้วย 1,000,001 จะอ่านว่า "หนึ่งล้านหนึ่ง" แทน "หนึ่งล้านเอ็ด"
    """
    if n < 0:
        return "ลบ" + _read_number(-n, has_higher_place)
    if n == 0:
        return "" if has_higher_place else "ศูนย์"
    if n >= 1_000_000:
        head, tail = divmod(n, 1_000_000)
        out = _read_number(head, False) + "ล้าน"
        return out + (_read_number(tail, True) if tail else "")

    digits = str(n)
    out: list[str] = []
    length = len(digits)
    for i, ch in enumerate(digits):
        d = int(ch)
        place = length - i - 1
        if d == 0:
            continue
        if place == 1:  # หลักสิบ
            out.append("ยี่สิบ" if d == 2 else ("สิบ" if d == 1 else _DIGITS[d] + "สิบ"))
        elif place == 0 and d == 1 and (length > 1 or has_higher_place):
            out.append("เอ็ด")  # ลงท้ายด้วยหนึ่งเมื่อมีหลักสูงกว่านำ -> เอ็ด
        else:
            out.append(_DIGITS[d] + _PLACES[place])
    return "".join(out)


def read_digits(s: str) -> str:
    """อ่านตัวเลขทีละตัว เหมาะกับเบอร์โทร/รหัส

    >>> read_digits("081-234")
    'ศูนย์ แปด หนึ่ง สอง สาม สี่'
    """
    return " ".join(_DIGITS[int(c)] for c in s if c.isdigit())


# ตัวเลขที่ยืนเดี่ยว ๆ — ไม่ติดกับตัวอักษร ไม่ใช่ส่วนของเวลา (20:30) หรือ IP
_STANDALONE_NUMBER = re.compile(r"(?<![\w.:])(\d{1,12})(?:\.(\d{1,6}))?(?![\w:])(?!\.\d)")

# ยาวขนาดนี้คนไทยอ่านทีละตัว ไม่อ่านเป็นจำนวน
_DIGIT_BY_DIGIT_LENGTH = 8


def expand_numbers_for_speech(text: str) -> str:
    """แปลงตัวเลขอารบิกเป็นคำอ่านภาษาไทย เพื่อให้ TTS อ่านถูก

    * เลขที่ขึ้นต้นด้วย 0 หรือยาวตั้งแต่ 8 หลัก -> อ่านทีละตัว (เบอร์โทร รหัส เลขบัญชี)
    * เลขทศนิยม -> อ่าน "จุด" แล้วอ่านหลังจุดทีละตัว
    * เวลาแบบ 20:30 และเลขที่ติดกับตัวอักษร -> ปล่อยไว้ ไม่แตะ

    >>> expand_numbers_for_speech("ราคา 199 บาท")
    'ราคา หนึ่งร้อยเก้าสิบเก้า บาท'
    >>> expand_numbers_for_speech("โทร 0812345678")
    'โทร ศูนย์ แปด หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด'
    >>> expand_numbers_for_speech("นัดสองทุ่ม 20:30 นะ")
    'นัดสองทุ่ม 20:30 นะ'
    """

    def replace(match: re.Match[str]) -> str:
        whole, decimal = match.group(1), match.group(2)
        if whole.startswith("0") and len(whole) > 1 or len(whole) >= _DIGIT_BY_DIGIT_LENGTH:
            spoken = read_digits(whole)
        else:
            spoken = thai_number_to_words(int(whole))
        if decimal:
            spoken = f"{spoken} จุด {read_digits(decimal)}"
        # ไม่เติมช่องว่างรอบคำอ่าน — ภาษาไทยเขียนติดกันอยู่แล้ว ("ราคา199บาท" ->
        # "ราคาหนึ่งร้อยเก้าสิบเก้าบาท") และถ้าเดิมมีช่องว่างก็ยังอยู่ครบ
        return spoken

    return _STANDALONE_NUMBER.sub(replace, text)


# ── ตัดประโยค ───────────────────────────────────────────────────────────────
_SENT_END = re.compile(r"[.!?…‽]+[\s\"'”’)\]]*")
_PARTICLE_BREAK = re.compile(
    r"(?:" + "|".join(re.escape(p) for p in _BREAK_PARTICLES) + r")(?=\s|$)"
)

# ตัวตัดประโยคของ pythainlp เรียงตามความแม่นยำ
#
# ระวัง: engine เริ่มต้นของ pythainlp คือ "crfcut" ซึ่งต้องติดตั้ง ``python-crfsuite``
# เพิ่มต่างหาก ไม่ได้ติดมากับ ``pip install pythainlp`` ถ้าไม่เช็คตรงนี้ ระบบจะเงียบ ๆ
# ถอยไปใช้กฎสำรองทั้งที่ผู้ใช้ติดตั้ง pythainlp ไปแล้ว
_SENT_ENGINES = ("crfcut", "thaisum", "whitespace+newline")

_sent_engine: str | None = None
_sent_engine_probed = False
_word_tokenizer_ok: bool | None = None


def thai_segmenter_engine(force_probe: bool = False) -> str | None:
    """ชื่อ engine ตัดประโยคของ pythainlp ที่ใช้ได้จริง หรือ ``None`` ถ้าใช้ไม่ได้เลย

    ผลถูกแคชไว้ เพราะการทดลองเรียก engine ที่ขาด dependency จะโยน exception
    ทุกครั้งที่เรียก ซึ่งแพงเกินไปสำหรับงานที่ทำทุกเทิร์นของบทสนทนา
    """
    global _sent_engine, _sent_engine_probed
    if _sent_engine_probed and not force_probe:
        return _sent_engine

    _sent_engine_probed = True
    _sent_engine = None
    try:
        from pythainlp.tokenize import sent_tokenize  # type: ignore
    except Exception:
        return None

    probe = "ทดสอบระบบตัดประโยคภาษาไทยครับ วันนี้อากาศดีมากเลยนะครับ"
    for engine in _SENT_ENGINES:
        try:
            if sent_tokenize(probe, engine=engine):
                _sent_engine = engine
                break
        except Exception:
            continue
    return _sent_engine


def thai_word_tokenizer_available(force_probe: bool = False) -> bool:
    """ตัวตัดคำของ pythainlp ใช้ได้ไหม (ใช้กันไม่ให้ตัดกลางคำตอนบังคับตัดท่อน)"""
    global _word_tokenizer_ok
    if _word_tokenizer_ok is not None and not force_probe:
        return _word_tokenizer_ok
    try:
        from pythainlp.tokenize import word_tokenize  # type: ignore

        _word_tokenizer_ok = bool(word_tokenize("ทดสอบตัดคำ"))
    except Exception:
        _word_tokenizer_ok = False
    return _word_tokenizer_ok


def _pythainlp_sentences(text: str) -> list[str] | None:
    engine = thai_segmenter_engine()
    if engine is None:
        return None
    try:
        from pythainlp.tokenize import sent_tokenize  # type: ignore

        parts = [p.strip() for p in sent_tokenize(text, engine=engine)]
        return [p for p in parts if p] or None
    except Exception:
        return None


def _rule_based_split(line: str) -> list[str]:
    """ตัดหนึ่งบรรทัดด้วยกฎ: เครื่องหมายจบประโยค และคำลงท้ายสุภาพ"""
    parts: list[str] = []
    cursor = 0
    marks = sorted(
        {m.end() for m in _SENT_END.finditer(line)}
        | {m.end() for m in _PARTICLE_BREAK.finditer(line)}
    )
    for end in marks:
        chunk = line[cursor:end].strip()
        if chunk:
            parts.append(chunk)
        cursor = end
    tail = line[cursor:].strip()
    if tail:
        parts.append(tail)
    return parts


def split_sentences(text: str, min_chars: int = 12) -> list[str]:
    """ตัดข้อความไทยเป็นประโยคสำหรับส่งให้ TTS ทีละท่อน

    ตัดที่ขึ้นบรรทัดใหม่ก่อนเสมอ แล้วค่อยตัดในแต่ละบรรทัดด้วย pythainlp (ถ้าใช้ได้)
    ไม่งั้นถอยไปใช้กฎ: เครื่องหมายจบประโยค และคำลงท้ายสุภาพ

    เหตุที่ต้องตัดบรรทัดเองก่อน: ตัวตัดประโยคของ pythainlp (crfcut) ไม่ถือว่า
    ขึ้นบรรทัดใหม่เป็นจุดจบประโยค จะคืนข้อความที่มี ``\\n`` ติดมาทั้งก้อน
    ซึ่ง TTS จะอ่านรวดเดียวไม่เว้นจังหวะ ทั้งที่ขึ้นบรรทัดใหม่คือจุดตัดที่
    เชื่อถือได้ที่สุดของภาษาไทย
    """
    text = text.strip()
    if not text:
        return []

    parts: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts.extend(_pythainlp_sentences(line) or _rule_based_split(line))
    return _merge_short(parts, min_chars)


def _merge_short(parts: list[str], min_chars: int) -> list[str]:
    """รวมท่อนที่สั้นเกินไปเข้ากับท่อนข้างเคียง

    ท่อนสั้น ๆ ทำให้เสียงพูดขาดเป็นห้วง ๆ ท่อนที่สั้นกว่า ``min_chars`` จะถูกรวม
    กับท่อนถัดไป และถ้าท่อนสุดท้ายยังสั้นอยู่ก็รวมย้อนกลับเข้าท่อนก่อนหน้า
    ไม่งั้นคำอย่าง "ครับ" จะถูกพูดเดี่ยว ๆ ห้อยท้าย
    """
    out: list[str] = []
    for part in parts:
        if out and len(out[-1]) < min_chars:
            out[-1] = f"{out[-1]} {part}".strip()
        else:
            out.append(part)
    if len(out) > 1 and len(out[-1]) < min_chars:
        tail = out.pop()
        out[-1] = f"{out[-1]} {tail}".strip()
    return out


# ── ตัดท่อนแบบไม่ให้ขาดกลางคำ ───────────────────────────────────────────────
def _cluster_safe_cut(text: str, index: int) -> int:
    """เลื่อนจุดตัดถอยหลังจนไม่ขาดกลางคลัสเตอร์อักขระไทย

    ภาษาไทยเขียนสระบน สระล่าง และวรรณยุกต์ซ้อนบนพยัญชนะ ถ้าตัดคั่นกลาง
    เครื่องหมายพวกนี้จะเหลือวรรณยุกต์ลอย ๆ ที่ TTS อ่านเป็นพยางค์ประหลาด
    และสระหน้า (เ แ โ ใ ไ) เขียนก่อนพยัญชนะที่ออกเสียงจริง จึงห้ามตัดหลังมัน
    """
    if index <= 0 or index >= len(text):
        return index
    i = index
    while i > 0 and unicodedata.category(text[i]) == "Mn":
        i -= 1
    while i > 0 and text[i - 1] in _LEADING_VOWELS:
        i -= 1
    return i


def _word_safe_cut(text: str, low: int, high: int) -> int:
    """หาจุดตัดที่ไม่ขาดกลางคำ ในช่วง [low, high]

    ใช้ตัวตัดคำของ pythainlp ถ้ามี เพราะภาษาไทยไม่มีเว้นวรรคระหว่างคำ การตัดที่
    ตำแหน่งความยาวดิบ ๆ จะทำให้คำอย่าง "เขียน" ขาดเป็น "เขี" + "ยน" ซึ่งคนฟัง
    จับได้ทันที ถ้าไม่มีตัวตัดคำ อย่างน้อยก็ไม่ตัดกลางคลัสเตอร์
    """
    if high <= low:
        return _cluster_safe_cut(text, high)

    if thai_word_tokenizer_available():
        try:
            from pythainlp.tokenize import word_tokenize  # type: ignore

            # ต้องตัดคำจากข้อความที่ยาว *เกิน* จุดตัดไปหน่อย ไม่งั้นปลายที่เราตัดเอง
            # จะกลายเป็นขอบคำปลอม แล้วเราก็จะเลือกจุดนั้นซึ่งอยู่กลางคำพอดี
            window = text[: min(len(text), high + 32)]
            best = 0
            position = 0
            for token in word_tokenize(window):
                position += len(token)
                if position > high:
                    break
                if position >= low:
                    best = position
            if best:
                return best
        except Exception:
            pass
    return _cluster_safe_cut(text, high)


class SpeechChunker:
    """ตัวสะสมข้อความที่ไหลมาจากโมเดล แล้วปล่อยออกเป็นท่อนที่ "พูดได้"

    หัวใจของการทำให้บทสนทนาด้วยเสียงไม่หน่วง: เริ่มพูดประโยคแรกทันทีที่จบ
    ประโยค แทนที่จะรอคำตอบครบทั้งก้อน

        chunker = SpeechChunker()
        for delta in stream:
            for chunk in chunker.feed(delta):
                speak(chunk)
        for chunk in chunker.flush():
            speak(chunk)

    ภาษาไทยไม่มีจุดจบประโยค จึงตัดด้วยลำดับความสำคัญนี้:
    ขึ้นบรรทัดใหม่ > เครื่องหมายจบประโยค > คำลงท้ายสุภาพ > เว้นวรรคเมื่อยาวพอ
    และเมื่อจำเป็นต้องบังคับตัดเพราะยาวเกินไป จะตัดที่ขอบคำเสมอ
    """

    def __init__(self, min_chars: int = 24, max_chars: int = 160) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""

    def feed(self, delta: str) -> Iterator[str]:
        """ป้อนข้อความชิ้นใหม่ แล้ว yield ท่อนที่พร้อมพูด"""
        self._buf += delta
        while True:
            cut = self._find_cut(self._buf)
            if cut is None:
                return
            chunk, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if chunk:
                yield chunk

    def flush(self) -> Iterator[str]:
        """ปล่อยข้อความที่ค้างอยู่ทั้งหมด (เรียกตอนสตรีมจบ)"""
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            yield rest

    def _find_cut(self, buf: str) -> int | None:
        if len(buf) < self.min_chars:
            return None

        nl = buf.find("\n")
        if nl >= 0:
            # ขึ้นบรรทัดใหม่คือจุดตัดที่ชัดที่สุด ใช้ได้แม้ท่อนจะสั้นกว่า min_chars
            return nl + 1

        for match in _SENT_END.finditer(buf):
            if match.end() >= self.min_chars:
                return match.end()

        for match in _PARTICLE_BREAK.finditer(buf):
            if match.end() >= self.min_chars:
                return match.end()

        # ภาษาไทยใช้เว้นวรรคแทนการจบวลี — ตัดที่เว้นวรรคเมื่อบัฟเฟอร์ยาวพอแล้ว
        if len(buf) >= self.max_chars:
            space = buf.rfind(" ", self.min_chars, self.max_chars)
            if space > 0:
                return space + 1
            return _word_safe_cut(buf, self.min_chars, self.max_chars)
        return None
