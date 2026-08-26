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
# ไม่ใส่ "คับ" และ "ค่า" ด้วยเหตุผลเดียวกัน แม้จะใช้เป็นคำลงท้ายจริง แต่พ้องกับ
# คำธรรมดาที่จบประโยคได้บ่อยกว่ามาก ("เสื้อคับ" "กางเกงคับ" "ไม่มีค่า" "คุณค่า")
# การเดาเพศผิดทำให้ระบบเรียกขานผู้ใช้ผิดไปตลอด
THAI_PARTICLES: dict[str, tuple[str, ...]] = {
    "male": ("ครับผม", "ครับ", "คร้าบ", "ครัช"),
    "female": ("ค่ะ", "คะ"),
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
    # สกุลเงินและตัวดำเนินการ — หมวด Sc/Sm ไม่ถูกกรองทิ้ง จึงค้างให้ TTS
    # อ่านเป็นพยางค์แปลก ๆ หรือเงียบไปเฉย ๆ ทำให้ความหมายหาย
    ("±", " บวกลบ "),
    ("×", " คูณ "),
    ("÷", " หาร "),
    ("≈", " ประมาณ "),
    ("≤", " ไม่เกิน "),
    ("≥", " ไม่น้อยกว่า "),
    ("=", " เท่ากับ "),
    # "ราคา 100 → 80 บาท" คือการเปลี่ยนค่า ไม่ใช่ปลายทาง
    ("→", " เป็น "),
    ("←", " "),
    ("½", " ครึ่ง "),
    ("¼", " หนึ่งส่วนสี่ "),
    ("¾", " สามส่วนสี่ "),
]

# สัญลักษณ์สกุลเงิน — ภาษาไทยพูดหน่วยเงิน *หลัง* จำนวนเสมอ
# "฿500" คือ "ห้าร้อยบาท" ไม่ใช่ "บาทห้าร้อย"
_CURRENCY = {"฿": "บาท", "$": "ดอลลาร์", "€": "ยูโร", "£": "ปอนด์", "¥": "เยน"}
_CURRENCY_CLASS = "[" + "".join(re.escape(c) for c in _CURRENCY) + "]"
_AMOUNT = r"\d[\d,]*(?:\.\d+)?"
_CURRENCY_BEFORE = re.compile(rf"({_CURRENCY_CLASS})\s*({_AMOUNT})")
_CURRENCY_AFTER = re.compile(rf"({_AMOUNT})\s*({_CURRENCY_CLASS})")

# ตัวย่อหน่วยไทย — TTS อ่านเป็นชื่อตัวอักษร ("กอกอ" แทน "กิโลกรัม")
# ส่วน kg/cm ภาษาอังกฤษถูกขยายอยู่แล้ว จึงไม่สม่ำเสมอ
_THAI_UNIT_ABBR = {
    "กก.": "กิโลกรัม", "ก.ก.": "กิโลกรัม", "กม.": "กิโลเมตร", "ก.ม.": "กิโลเมตร",
    "ซม.": "เซนติเมตร", "ซ.ม.": "เซนติเมตร", "มม.": "มิลลิเมตร",
    "ชม.": "ชั่วโมง", "ช.ม.": "ชั่วโมง", "นาที": "นาที",
    "ลบ.ม.": "ลูกบาศก์เมตร", "ตร.ม.": "ตารางเมตร", "ตร.กม.": "ตารางกิโลเมตร",
    "บ.": "บาท", "ล้านบ.": "ล้านบาท", "ม.": "เมตร",
    # มก./มล. เลี่ยงไม่ได้ในบทสนทนาเรื่องยาและการทำอาหาร
    "มก.": "มิลลิกรัม", "มล.": "มิลลิลิตร", "ตร.ว.": "ตารางวา",
    "วิ.": "วินาที", "ลบ.ซม.": "ลูกบาศก์เซนติเมตร", "ไมโครกรัม": "ไมโครกรัม",
}
# ต้องตามหลังตัวเลขเท่านั้น ไม่งั้น "บ.ก. บอกว่า" กลายเป็น "บาท ก. บอกว่า"
_THAI_UNIT_RE = re.compile(
    # ตามหลังตัวเลข หรือตามหลัง "ต่อ" ที่เพิ่งแปลงมาจากเครื่องหมายทับ
    r"(?:(?<=\d)|(?<=ต่อ))(\s*)("
    + "|".join(re.escape(a) for a in sorted(_THAI_UNIT_ABBR, key=len, reverse=True))
    + r")"
)

# หน่วยที่ตามด้วย "/" แล้วตามด้วยหน่วยอีกตัว = "ต่อ"
_PER_UNITS = (
    "บาท", "กก.", "กิโลกรัม", "กรัม", "มก.", "มล.", "ลิตร", "ซีซี",
    "กม.", "กิโลเมตร", "ม.", "เมตร", "ซม.", "เซนติเมตร", "ตร.ม.", "ไมล์",
    "ชม.", "ชั่วโมง", "นาที", "วินาที", "วัน", "เดือน", "ปี", "คน", "ครั้ง",
    "หน่วย", "ชิ้น", "ที่นั่ง", "ห้อง", "ตัว", "ใบ", "เที่ยว",
)
_UNIT_PER_RE = re.compile(
    r"((?:" + "|".join(re.escape(u) for u in sorted(_PER_UNITS, key=len, reverse=True))
    + r"))\s*/\s*(?=[ก-๛])"
)

# ตัวย่อศักราช — มาก่อนตัวเลข ไม่ใช่ตามหลัง จึงต้องแยกกฎ
_ERA_ABBR = {"พ.ศ.": "พุทธศักราช", "ค.ศ.": "คริสต์ศักราช", "ฮ.ศ.": "ฮิจเราะห์ศักราช"}
_ERA_RE = re.compile(
    "(?:" + "|".join(re.escape(a) for a in _ERA_ABBR) + r")(?=\s*\d)"
)

# ตัวย่อเดือนไทย — TTS อ่านเป็นพยางค์เดี่ยว ๆ ("ก. ค.") ซึ่งฟังไม่ออกว่าเดือนอะไร
_MONTH_ABBR = {
    "ม.ค.": "มกราคม", "ก.พ.": "กุมภาพันธ์", "มี.ค.": "มีนาคม", "เม.ย.": "เมษายน",
    "พ.ค.": "พฤษภาคม", "มิ.ย.": "มิถุนายน", "ก.ค.": "กรกฎาคม", "ส.ค.": "สิงหาคม",
    "ก.ย.": "กันยายน", "ต.ค.": "ตุลาคม", "พ.ย.": "พฤศจิกายน", "ธ.ค.": "ธันวาคม",
}
_MONTH_ABBR_RE = re.compile(
    "|".join(re.escape(a) for a in sorted(_MONTH_ABBR, key=len, reverse=True))
)


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
    # อย่าบอกว่ามีหน้าจอ — prompt ห้ามรับปากสิ่งที่ทำไม่ได้ในช่องทางเสียง
    s = re.sub(r"```[\s\S]*?```", " (ตรงนี้เป็นโค้ด ขอข้ามการอ่านนะ) ", s)
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
    # ตารางแบบ markdown — ต้องเอาขีดตั้งออกก่อน ไม่งั้นเส้นคั่น "|---|---|"
    # จะไม่เข้าเงื่อนไขเส้นคั่น แล้วหลุดไปให้ TTS อ่านว่า "ขีดขีดขีด"
    s = s.replace("|", " ")
    s = re.sub(r"(?m)^[\s\-*_]{3,}$", " ", s)
    s = s.replace("~", " ")

    # สกุลเงินต้องย้ายไปหลังจำนวนก่อนที่ตัวกรองสัญลักษณ์จะลบเครื่องหมายทิ้ง
    s = _CURRENCY_BEFORE.sub(lambda m: f" {m.group(2)} {_CURRENCY[m.group(1)]} ", s)
    s = _CURRENCY_AFTER.sub(lambda m: f" {m.group(1)} {_CURRENCY[m.group(2)]} ", s)
    # เครื่องหมายที่ไม่ติดกับตัวเลขมักไม่ใช่ราคา ("echo $HOME", "a$b")
    # อ่านว่า "ดอลลาร์" ตรงนั้นแย่กว่าเงียบไปเฉย ๆ ยกเว้น ฿ ที่ไม่มีความหมายอื่น
    # แป้นพิมพ์ไทยพิมพ์ "x" ง่ายกว่า "×" มาก
    s = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " คูณ ", s)
    s = s.replace("฿", " บาท ")
    s = re.sub(_CURRENCY_CLASS, " ", s)

    # "50 บาท/กก." อ่านว่า "ห้าสิบบาทต่อกิโลกรัม"
    #
    # ต้องผูกกับ *หน่วย* ที่อยู่ข้างหน้าเท่านั้น การแทนที่ "/" ทุกตัวที่ขนาบ
    # ด้วยอักษรไทยผิดสองทาง: เลขไทยอยู่ในช่วง [ก-๙] ด้วย "บ้านเลขที่ ๙๙/๑"
    # จึงกลายเป็น "เก้าสิบเก้าต่อหนึ่ง" และ "/" ที่แปลว่า "หรือ" ก็โดนด้วย
    # ("ชาย/หญิง" -> "ชายต่อหญิง")
    s = _UNIT_PER_RE.sub(lambda m: f"{m.group(1)} ต่อ ", s)
    # TTS ส่วนใหญ่ข้าม "ฯลฯ" ไปเฉย ๆ หรืออ่านเป็นพยางค์
    s = s.replace("ฯลฯ", " และอื่น ๆ ")

    # ตัวย่อเดือนและหน่วยต้องแปลงก่อนแตะจุด ไม่งั้นเหลือ "ก. ค." ให้อ่านทีละพยางค์
    s = _MONTH_ABBR_RE.sub(lambda m: f" {_MONTH_ABBR[m.group(0)]} ", s)
    s = _THAI_UNIT_RE.sub(lambda m: f" {_THAI_UNIT_ABBR[m.group(2)]} ", s)
    s = _ERA_RE.sub(lambda m: f" {_ERA_ABBR[m.group(0)]} ", s)

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


# เวลา — ต้องจับก่อนตัวเลขทั่วไป
#
# คนไทยเขียนเวลาด้วย "จุด" เป็นปกติ ("20.30 น.") ถ้าปล่อยให้ตัวอ่านตัวเลขทั่วไป
# จัดการ มันจะอ่านเป็นทศนิยมว่า "ยี่สิบจุดสามศูนย์" ซึ่งฟังไม่รู้เรื่องเลย
# ส่วนรูปแบบที่ใช้ทวิภาค (20:30) ถือเป็นเวลาเสมอ
# กิน "น." ที่ตามมาด้วย ไม่งั้นจะเหลือ "น." ลอยให้ TTS อ่านเป็นพยางค์
_TIME_COLON = re.compile(r"(?<![\w.:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])(?:\s*น\.)?")
# ยอมรับ 24.00 ด้วย — คนไทยเขียนเวลาสิ้นวันแบบนี้เป็นปกติ
_TIME_DOT = re.compile(r"(?<![\w.:])([01]?\d|2[0-4])\.([0-5]\d)\s*น\.")

# ช่วงเวลา — ต้องจับก่อนกฎเวลาเดี่ยว
#
# "ร้านเปิด 10.00-22.00 น." และ "ตั้งแต่ 9.00 ถึง 17.00" คือวิธีเขียนเวลาทำการ
# ที่พบบ่อยที่สุด แต่ _TIME_CONTEXT มองย้อนหลังแค่ 24 ตัวอักษรและไม่มีคำพวกนี้
# ผลคืออ่านเป็นทศนิยม "สิบจุดศูนย์ศูนย์" ซึ่งไม่ใช่ภาษาไทยเลย
# ที่สำคัญกว่านั้น การมีเวลาสองตัวคั่นด้วยขีดหรือ "ถึง" เป็นหลักฐานในตัวเอง
# ว่าทั้งคู่เป็นเวลา ไม่ต้องพึ่งคำแวดล้อม
_TIME_PART = r"(?:[01]?\d|2[0-4])[.:][0-5]\d"
_TIME_RANGE = re.compile(
    rf"(?<![\w.:])({_TIME_PART})\s*(?:-|ถึง|–|—)\s*({_TIME_PART})(?![\d\w])(\s*น\.)?"
)

# คำที่บอกว่าเลขคู่นี้เป็นช่วง "เวลา" ไม่ใช่ช่วงปริมาณ
#
# การมีเวลาสองตัวคั่นด้วยขีดไม่ใช่หลักฐานในตัวเองอย่างที่เคยคิด — "1.50-2.30
# กิโลกรัม" "เกรดเฉลี่ย 3.00-4.00" ก็หน้าตาเหมือนกันเป๊ะ ต้องมีอย่างน้อย
# หนึ่งสัญญาณ: รูปทวิภาค, มี "น." ต่อท้าย, หรือมีคำบอกเวลานำหน้า
_TIME_RANGE_CONTEXT = re.compile(
    r"(?:เวลา|ตอน|นัด|เจอกัน|ประชุม|เริ่ม|เลิก|ออก|เปิด|ปิด|สอบ|ทำการ|ตั้งแต่|"
    r"บ่าย|เช้า|เย็น|ค่ำ|เที่ยง|ดึก|โมง|ทุ่ม|ตี|รอบ|กะ|เวร)"
)

# คนไทยเขียนเวลาแบบจุดโดยไม่มี "น." บ่อยมาก ("เจอกัน 19.00", "นัดตอน 8.30")
# แต่ "3.50 เมตร" ก็หน้าตาเหมือนกันเป๊ะ จึงต้องดูคำแวดล้อมประกอบ
_TIME_DOT_BARE = re.compile(r"(?<![\w.:])([01]?\d|2[0-3])\.([0-5]\d)(?![\d\w])(?!\.\d)")
#
# ตั้งใจ *ไม่* ใส่ "ประมาณ" "ราว" "ถึง" "ก่อน" "หลัง" เพราะมันใช้กับปริมาณอะไรก็ได้
# "สูงประมาณ 3.50 เมตร" เคยถูกอ่านเป็น "สามนาฬิกาห้าสิบนาที เมตร"
_TIME_CONTEXT = re.compile(
    r"(?:เวลา|ตอน|นัด|เจอกัน|ประชุม|เริ่ม|เลิก|ออก|บ่าย|เช้า|เย็น|ค่ำ|เที่ยง|"
    r"ดึก|โมง|ทุ่ม|ตี)"
)

# เบอร์โทรที่คั่นด้วยขีดหรือช่องว่าง (081-234-5678, 02 123 4567, +66 81 234 5678)
# ต้องจับก่อน ไม่งั้นแต่ละกลุ่มจะถูกอ่านเป็นจำนวน ("หนึ่ง-สองร้อยสามสิบสี่-ห้าพัน...")
_MAYBE_PHONE = re.compile(r"(?<![\w])(\+?\d[\d\- ]{7,17}\d)(?![\w])")

# คำที่ตามหลังตัวเลขแล้วพิสูจน์ว่ามันเป็น "จำนวน" ไม่ใช่รหัสหรือเวลา
#
# นี่คือสัญญาณที่เชื่อถือได้ที่สุด และแม่นกว่าการดูคำที่อยู่ *ข้างหน้า* มาก
# เพราะคำนำหน้าอย่าง "ห้อง" "ตู้" "เที่ยวบิน" "เวลา" "ตี" "ออก" ล้วนเป็นคำ
# ธรรมดาที่โผล่ในประโยคไหนก็ได้ ("ห้องละ 1500 บาท" ไม่ใช่เลขห้อง
# "ประชุมใช้เวลา 1.30 ชั่วโมง" ไม่ใช่เวลานาฬิกา) แต่ไม่มีใครเขียนหน่วยวัด
# ต่อท้ายรหัส
# หน่วยที่ตามหลังตัวเลขแล้วยืนยันว่าเป็น "จำนวน" ไม่ใช่รหัสหรือเวลา
#
# ต้องแบ่งสองชั้น เพราะภาษาไทยไม่เว้นวรรคระหว่างคำ การเทียบแบบ "ขึ้นต้นด้วย"
# เฉย ๆ ทำให้ "ห้อง 2105 คนนี้" เห็น "คน" เป็นลักษณนาม ส่วนการบังคับว่าต้อง
# จบตรงนั้นพอดีทำให้ "2500 บาทรวมอาหารเช้า" ไม่เห็น "บาท" เป็นหน่วย แล้ว
# ราคาถูกท่องทีละหลัก ซึ่งแย่กว่ากันมาก
#
# ชั้นที่หนึ่ง: หน่วยที่ไม่มีทางเป็นพยางค์แรกของคำธรรมดา — เทียบแบบขึ้นต้นได้เลย
_QUANTITY_UNITS_STRONG = (
    "บาทถ้วน", "บาท", "สตางค์", "ดอลลาร์", "ยูโร", "เยน", "ปอนด์",
    "กิโลกรัม", "มิลลิกรัม", "ไมโครกรัม", "กรัม", "ขีด", "ตัน",
    "มิลลิเมตร", "เซนติเมตร", "กิโลเมตร", "ตารางเมตร", "ตารางวา",
    "ลูกบาศก์เมตร", "เมตร", "ไมล์", "นิ้ว", "ฟุต", "ไร่",
    "มิลลิลิตร", "ลิตร", "ซีซี",
    "วินาที", "นาที", "ชั่วโมง", "สัปดาห์", "เดือน",
    "องศา", "เปอร์เซ็นต์", "%",
    "ล้าน", "แสน", "หมื่น", "คะแนน", "ชิ้น", "เล่ม", "แผ่น",
    "กล่อง", "ถุง", "ขวด", "แก้ว",
)

# ชั้นที่สอง: คำที่เป็นพยางค์แรกของคำธรรมดาได้ ("คนนี้" "วันนี้" "ที่ร้าน"
# "ตัวอย่าง" "ครั้งนี้" "หลังจาก" "พันธุ์") — ต้องจบตรงนั้นจริง ๆ
_QUANTITY_UNITS_WEAK = (
    "คน", "ครั้ง", "ที่", "ห้อง", "ตัว", "ใบ", "อัน", "เท่า", "ปี", "วัน",
    "ระดับ", "หลัง", "คัน", "จาน", "พัน", "ดาว", "วา", "งาน", "ข้อ",
    "กก.", "ซม.", "กม.", "มม.", "ชม.", "มก.", "มล.", "ม.", "ตร.ม.", "ตร.ว.",
)


def _alt(words: "tuple[str, ...]") -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


# คนไทยเขียนคำลงท้ายติดหน่วยเป็นปกติ ("ห้าร้อยกรัมครับ") จึงยอมให้ตามด้วยได้
_QUANTITY_UNIT = re.compile(
    r"^\s*(?:"
    + _alt(_QUANTITY_UNITS_STRONG)
    + r"|(?:"
    + _alt(_QUANTITY_UNITS_WEAK)
    + r")(?:(?![ก-๙])|(?=ครับ|ค่ะ|คะ|นะ)))"
)

# คำนำหน้าที่บอกว่าตัวเลขข้างหลังเป็นรหัสประจำตัว ไม่ใช่จำนวน
_ID_CONTEXT = re.compile(
    r"(?:เลขบัตร|บัตรประชาชน|บัตรเครดิต|เลขบัญชี|เลขที่บัญชี|บัญชีเลขที่|"
    r"เลขประจำตัว|เลขพัสดุ|เลขที่พัสดุ|หมายเลข|เลขทะเบียน|passport|พาสปอร์ต)"
)

# วันที่แบบ 15/8/2568 — เครื่องหมายทับอ่านออกเสียงไม่ได้
_SLASH_DATE = re.compile(r"(?<![\w/])(\d{1,2})/(\d{1,2})/(\d{4})(?![\w/])")

_THAI_MONTHS = (
    "", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
)

# ช่วงตัวเลขและสกอร์ — ขีดกลางอ่านออกเสียงไม่ได้ TTS จะกลืนหายไปเฉย ๆ
# ทำให้ "3-5 คน" ฟังเป็น "สามห้าคน" ซึ่งความหมายเปลี่ยน
# ต้องรับรูปที่มีจุลภาคด้วย ("1,200-1,500 บาท") ไม่งั้นจะเหลือขีดกลางค้างอยู่
# ระหว่างคำอ่านสองก้อน ซึ่ง TTS กลืนหายไปเฉย ๆ
# ต้องรับถึงหกหลัก ไม่งั้น "ราคา 10000-20000 บาท" ตกไปให้กฎเบอร์โทรจัดการ
# ซึ่งเห็นขีดกลางของช่วงเป็นหลักฐานว่าเป็นเบอร์ แล้วอ่านทีละตัวยาวเหยียด
# ห้ามขึ้นต้นด้วยศูนย์ — ช่วงตัวเลขไม่มีทางเขียนแบบนั้น แต่เบอร์โทรไทยขึ้นต้น
# ด้วยศูนย์เสมอ "038-123456" คือเบอร์บ้านต่างจังหวัด ไม่ใช่ช่วง 38 ถึง 123456
_RANGE_TOKEN = r"(?:\d{1,3}(?:,\d{3})+|0|[1-9]\d{0,5})"
_NUMBER_RANGE = re.compile(
    rf"(?<![\w.,-])({_RANGE_TOKEN})\s*-\s*({_RANGE_TOKEN})(?![\d\w,-])"
)
_SCORE_CONTEXT = re.compile(
    r"(?:คะแนน|สกอร์|ผลบอล|ผลการแข่ง|ผลแข่ง|ชนะ|แพ้|เสมอ)"
)

# ตัวเลขคั่นด้วยทับตั้งแต่สามตัวขึ้นไป
_SLASH_CHAIN = re.compile(r"(?<![\w/])\d{1,5}(?:/\d{1,5}){2,}(?![\d\w/])")

# อัตราส่วนที่ใช้ทวิภาค — "1:2" "60:40" ไม่ใช่เวลา แต่ก็ไม่ถูกกฎไหนจับเลย
# ทวิภาคจึงหลุดถึง TTS ดิบ ๆ ซึ่งเป็นสิ่งที่โมดูลนี้มีไว้เพื่อป้องกัน
_RATIO_COLON = re.compile(r"(?<![\w.:])(\d{1,4}):(\d{1,4})(?![\d:])")

# เลขที่มีเครื่องหมายทับแต่ไม่ใช่วันที่
#
# "ทับ" ใช้กับที่อยู่และเลขห้องเท่านั้น ("บ้านเลขที่ 99/1" "ชั้น 3/1")
# ของเดิมอ่าน "ทับ" กับทุกกรณี ทำให้เศษส่วนและคะแนนผิดหมด:
# "แบ่งคนละ 1/2" -> "หนึ่งทับสอง", "สอบได้ 18/20" -> "สิบแปดทับยี่สิบ"
_SLASH_PAIR = re.compile(r"(?<![\w/])(\d{1,5})/(\d{1,4})(?![\d\w/])")
_ADDRESS_CONTEXT = re.compile(
    r"(?:บ้านเลขที่|เลขที่|ที่อยู่|ห้อง|ชั้น|ซอย|ถนน|หมู่|อาคาร|ตึก|ยูนิต)"
)
# วันที่แบบไม่มีปี ("วันที่ 5/12") — ต้องมีคำว่าวันที่/เดือนนำหน้าจึงจะแน่ใจ
_SLASH_DAY_MONTH = re.compile(
    r"(?<=วันที่)\s*(\d{1,2})/(\d{1,2})(?![\d\w/])"
)

# เลขรหัสที่มีคำบอกบริบทนำหน้า — คนไทยอ่านทีละตัว ไม่อ่านเป็นจำนวน
# ("ห้อง 2105" คือ "ห้องสองหนึ่งศูนย์ห้า" ไม่ใช่ "ห้องสองพันหนึ่งร้อยห้า")
# ห้าม (?<![\w]) นำหน้า — ภาษาไทยเขียนติดกันไม่มีช่องว่าง "ไปห้อง 2105" จึงมี
# "ป" อยู่หน้า "ห้อง" ซึ่งเป็น \w ทำให้กฎนี้แทบไม่เคยทำงานเลย
# ต้องไม่กินเลขที่ตามด้วย "/" ("บ้านเลขที่ 123/45") ให้ _SLASH_PAIR จัดการต่อ
# ยอมให้มีคำไทยคั่นระหว่างคำบอกบริบทกับตัวเลขได้ ("ห้องประชุม 2105",
# "รหัสพนักงาน 123456", "เลขที่ใบสั่งซื้อ 778899") ของเดิมบังคับให้ติดกัน
# กฎจึงทำงานเฉพาะกับสำนวนที่สั้นที่สุดเท่านั้น
_CODE_NUMBER = re.compile(
    r"(ห้อง|ชั้น|ที่นั่ง|เบอร์|รหัส|ตู้|บ้านเลขที่|โต๊ะ|เที่ยวบิน|ล็อค|ล็อก"
    r"|โทร|สายด่วน|ไปรษณีย์|ตู้ ?ปณ|เลขที่|OTP|otp|พัสดุ)"
    # คำที่คั่นได้ต้องเป็นส่วนของคำนามประสม ("ห้องประชุม" "รหัสพนักงาน")
    # ไม่ใช่คำเชื่อมที่เปิดอนุประโยค ("โต๊ะที่นั่งกันเมื่อวาน 1234" ไม่ใช่รหัสโต๊ะ)
    # กันเฉพาะตอน *ขึ้นต้น* สะพาน ไม่ใช่ทุกตำแหน่ง ไม่งั้น "เบอร์ภายใน 1234"
    # ก็โดนไปด้วย เพราะ "ภายใน" มี "ใน" อยู่ข้างใน
    r"((?!ที่|กัน|เมื่อ|ของ|ใน|และ|แล้ว|นี้|นั้น|คือ|เป็น|กับ|ละ)[ก-๙]{0,12})"
    # ยอมให้มีตัวอักษรละตินคั่นด้วย — เลขเที่ยวบิน อาคาร และรุ่นสินค้าใช้กัน
    # เป็นปกติ ("เที่ยวบิน TG 123" อ่านว่า "ทีจี หนึ่ง สอง สาม")
    r"\s*([A-Za-z]{1,4}\s*)?(\d{3,6})(?![\d\w/])"
)

# เลขที่คั่นหลักพันด้วยจุลภาค — ต้องจับก่อนเช่นกัน ไม่งั้น "1,234" จะถูกอ่านเป็น
# "หนึ่ง" กับ "สองร้อยสามสิบสี่" คนละก้อน
_GROUPED_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+)(?:\.(\d{1,2}))?(?![\d,])")

# ตัวเลขที่ยืนเดี่ยว ๆ — ไม่ติดกับตัวอักษร ไม่ใช่ส่วนของ IP
# เลขบัตรประชาชนไทยยาว 13 หลักพอดี บัตรเครดิต 16 หลัก ของเดิมจำกัดที่ 12
# เลขพวกนี้จึงหลุดไปให้ TTS อ่านดิบ ๆ ทั้งก้อน
_STANDALONE_NUMBER = re.compile(
    r"(?<![\w.:])([-+]?)(\d{1,24})(?:\.(\d{1,6}))?(?![\w:])(?!\.\d)"
)

# จำนวนเงินที่มีทศนิยมสองตำแหน่งและตามด้วย "บาท"
#
# "1,234.50 บาท" อ่านว่า "หนึ่งพันสองร้อยสามสิบสี่บาทห้าสิบสตางค์"
# ไม่ใช่ "จุดห้าศูนย์บาท" ซึ่งฟังออกทันทีว่าเครื่องอ่าน
# รับทศนิยมหนึ่งหรือสองตำแหน่ง — "99.5 บาท" คือเก้าสิบเก้าบาทห้าสิบสตางค์
_BAHT_SATANG = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})*|\d+)\.(\d{1,2})\s*บาท")

# ยาวขนาดนี้คนไทยอ่านทีละตัว ไม่อ่านเป็นจำนวน
_DIGIT_BY_DIGIT_LENGTH = 8


# Python แปลงสตริงเป็น int ได้ไม่เกิน 4300 หลัก เกินกว่านั้นโยน ValueError
# ซึ่งไม่ควรหลุดออกจากฟังก์ชันที่มีหน้าที่แค่ทำข้อความให้อ่านออกเสียงได้
_MAX_INT_DIGITS = 4000


def _speak_quantity(token: str) -> str:
    """อ่านตัวเลขที่รู้แน่ว่าเป็นจำนวน — จุลภาคคือหลักฐานว่าไม่ใช่รหัส"""
    digits = token.replace(",", "")
    if "," in token and len(digits) <= _MAX_INT_DIGITS:
        return thai_number_to_words(int(digits))
    return _speak_integer(digits)


def _speak_integer(digits: str) -> str:
    """อ่านสตริงตัวเลข — ยาวหรือขึ้นต้นด้วยศูนย์ให้อ่านทีละตัว"""
    if digits.startswith("0") and len(digits) > 1:
        return read_digits(digits)
    # เลขกลม ๆ ที่ลงท้ายด้วยศูนย์ตั้งแต่สามตัวเป็นจำนวนแน่นอน ไม่ใช่รหัส
    # "ประชากร 70000000 คน" เคยถูกท่องทีละหลักเพราะยาวเกินแปดหลัก
    if len(digits) > _MAX_INT_DIGITS:
        return read_digits(digits)
    if len(digits) >= _DIGIT_BY_DIGIT_LENGTH and not digits.endswith("000"):
        return read_digits(digits)
    return thai_number_to_words(int(digits))


def _speak_time(hour: str, minute: str) -> str:
    """อ่านเวลาแบบทางการ — ชัดเจนและไม่กำกวมเรื่องเช้า/บ่าย"""
    spoken = f"{thai_number_to_words(int(hour))}นาฬิกา"
    if int(minute):
        spoken += f"{thai_number_to_words(int(minute))}นาที"
    return spoken


def _speak_phone(match: re.Match[str], id_context: bool = False) -> str:
    """อ่านเบอร์โทรทีละตัว ถ้าดูแล้วเป็นเบอร์โทรจริง ไม่งั้นปล่อยไว้"""
    raw = match.group(1)
    digits = re.sub(r"\D", "", raw)
    # เบอร์โทร เลขบัญชี เลขบัตร และเลขประจำตัว ล้วนยาวและอ่านทีละตัวทั้งหมด
    # ไม่ต้องแยกประเภท แค่ยาวพอและมีตัวคั่นก็พอ
    long_enough = 9 <= len(digits) <= 24
    # ต้องมีร่องรอยว่าเป็นเบอร์จริง ไม่งั้น "ราคา 1500 2000 3000 บาท" หรือ
    # "ปี 2566 2567 2568" จะถูกเหมารวมเป็นเบอร์เดียวแล้วอ่านทีละตัวยาวเหยียด
    #
    # ``id_context`` คือคำนำหน้าอย่าง "เลขบัตรประชาชน" ซึ่งเป็นหลักฐานเพียงพอ
    # ในตัวเอง — เลขบัตรประชาชนไทยเขียนเป็น "1 2345 67890 12 3" คั่นด้วยช่องว่าง
    # และไม่ได้ขึ้นต้นด้วย 0 กฎ "ต้องมีขีดหรือขึ้นต้นด้วย 0" จึงพลาดเสมอ
    looks_like_phone = "-" in raw or raw[0] in "+0" or id_context
    return read_digits(digits) if long_enough and looks_like_phone else raw


def _speak_date(match: re.Match[str]) -> str:
    day, month, year = (int(g) for g in match.groups())
    if not 1 <= month <= 12:
        return match.group(0)
    return (
        f"{thai_number_to_words(day)} {_THAI_MONTHS[month]} "
        f"{thai_number_to_words(year)}"
    )


def expand_numbers_for_speech(text: str) -> str:
    """แปลงตัวเลขอารบิกเป็นคำอ่านภาษาไทย เพื่อให้ TTS อ่านถูก

    * เวลา (20:30 หรือ 20.30 น.) -> อ่านเป็นนาฬิกา/นาที
    * วันที่ (15/8/2568) -> อ่านเป็นวันเดือนปี
    * เบอร์โทรที่มีขีดคั่น -> อ่านทีละตัว
    * เลขคั่นจุลภาค (1,234) -> อ่านเป็นจำนวนเดียว
    * เลขที่ขึ้นต้นด้วย 0 หรือยาวตั้งแต่ 8 หลัก -> อ่านทีละตัว (เบอร์โทร รหัส เลขบัญชี)
    * เลขทศนิยม -> อ่าน "จุด" แล้วอ่านหลังจุดทีละตัว
    * เลขที่ติดกับตัวอักษรหรือเป็นส่วนของ IP -> ปล่อยไว้ ไม่แตะ

    >>> expand_numbers_for_speech("ราคา 199 บาท")
    'ราคา หนึ่งร้อยเก้าสิบเก้า บาท'
    >>> expand_numbers_for_speech("นัดกัน 20.30 น.")
    'นัดกัน ยี่สิบนาฬิกาสามสิบนาที'
    >>> expand_numbers_for_speech("ราคา 1,234 บาท")
    'ราคา หนึ่งพันสองร้อยสามสิบสี่ บาท'
    """
    text = _SLASH_DATE.sub(_speak_date, text)

    def time_range(match: re.Match[str]) -> str:
        if _QUANTITY_UNIT.match(text[match.end() :]):
            return match.group(0)
        window = text[max(0, match.start() - 24) : match.start()]
        is_time = (
            ":" in match.group(0)
            or match.group(3)
            or _TIME_RANGE_CONTEXT.search(window)
        )
        if not is_time:
            return match.group(0)
        first, second = (re.split(r"[.:]", g) for g in match.groups()[:2])
        return (
            f"{_speak_time(first[0], first[1])}ถึง"
            f"{_speak_time(second[0], second[1])}"
        )

    def baht_satang(match: re.Match[str]) -> str:
        whole = match.group(1).replace(",", "")
        if len(whole) > _MAX_INT_DIGITS:
            return match.group(0)
        baht = thai_number_to_words(int(whole))
        # ".5" คือห้าสิบสตางค์ ไม่ใช่ห้าสตางค์
        satang = int(match.group(2).ljust(2, "0"))
        if not satang:
            return f"{baht}บาทถ้วน"
        return f"{baht}บาท{thai_number_to_words(satang)}สตางค์"

    text = _BAHT_SATANG.sub(baht_satang, text)
    text = _TIME_RANGE.sub(time_range, text)
    text = _TIME_DOT.sub(lambda m: _speak_time(m.group(1), m.group(2)), text)
    text = _TIME_COLON.sub(lambda m: _speak_time(m.group(1), m.group(2)), text)

    def bare_time(match: re.Match[str]) -> str:
        # หน่วยที่ตามหลังชนะคำแวดล้อมเสมอ — "ประชุมใช้เวลา 1.30 ชั่วโมง"
        # ไม่ใช่เวลานาฬิกา และ "เริ่มต้นที่ 1.20 เมตร" ก็ไม่ใช่
        if _QUANTITY_UNIT.match(text[match.end() :]):
            return match.group(0)
        window = text[max(0, match.start() - 24) : match.start()]
        if _TIME_CONTEXT.search(window):
            return _speak_time(match.group(1), match.group(2))
        return match.group(0)

    text = _TIME_DOT_BARE.sub(bare_time, text)

    def range_or_score(match: re.Match[str]) -> str:
        window = text[max(0, match.start() - 20) : match.start()]
        joiner = "ต่อ" if _SCORE_CONTEXT.search(window) else "ถึง"
        return (
            f"{_speak_quantity(match.group(1))}{joiner}"
            f"{_speak_quantity(match.group(2))}"
        )

    def phone(match: re.Match[str]) -> str:
        window = text[max(0, match.start() - 28) : match.start()]
        return _speak_phone(match, id_context=bool(_ID_CONTEXT.search(window)))

    # ช่วงตัวเลขต้องมาก่อนกฎเบอร์โทร ไม่งั้น "10000-20000" จะถูกมองว่าเป็นเบอร์
    # เพราะมีขีดกลางอยู่ แล้วอ่านทีละตัวยาวเหยียด
    text = _NUMBER_RANGE.sub(range_or_score, text)
    text = _MAYBE_PHONE.sub(phone, text)
    def code_number(match: re.Match[str]) -> str:
        # หน่วยที่ตามหลังพิสูจน์ว่าเป็นจำนวน ("ห้องละ 1500 บาท" ไม่ใช่เลขห้อง)
        if _QUANTITY_UNIT.match(text[match.end() :]):
            return match.group(0)
        latin = (match.group(3) or "").strip()
        prefix = f"{match.group(1)}{match.group(2)}"
        if latin:
            prefix += f" {latin}"
        return f"{prefix} {read_digits(match.group(4))}"

    text = _CODE_NUMBER.sub(code_number, text)

    def slash_pair(match: re.Match[str]) -> str:
        window = text[max(0, match.start() - 24) : match.start()]
        left, right = _speak_integer(match.group(1)), _speak_integer(match.group(2))
        if _ADDRESS_CONTEXT.search(window):
            return f"{left}ทับ{right}"
        top, bottom = int(match.group(1)), int(match.group(2))
        # ตัวบนมากกว่าตัวล่างไม่ใช่เศษส่วน — เป็นค่าคู่อย่างความดันโลหิต
        # ซึ่งคนไทยอ่านว่า "หนึ่งร้อยยี่สิบทับแปดสิบ"
        if top > bottom:
            return f"{left}ทับ{right}"
        # คะแนนสอบพูดว่า "สิบห้าจากยี่สิบ" ไม่ใช่ "สิบห้าส่วนยี่สิบ"
        # ซึ่งเป็นการอ่านเศษส่วนทางคณิตศาสตร์ เศษส่วนจริงตัวล่างมักเล็ก (1/2 3/4)
        # ส่วนคะแนนตัวล่างเป็นเลขเต็มสิบขึ้นไป
        if _SCORE_CONTEXT.search(window) or (top > 1 and bottom >= 10):
            return f"{left}จาก{right}"
        return f"{left}ส่วน{right}"

    def day_month(match: re.Match[str]) -> str:
        day, month = int(match.group(1)), int(match.group(2))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return match.group(0)
        return f" {thai_number_to_words(day)} {_THAI_MONTHS[month]}"

    text = _RATIO_COLON.sub(
        lambda m: f"{_speak_integer(m.group(1))}ต่อ{_speak_integer(m.group(2))}", text
    )
    text = _SLASH_DAY_MONTH.sub(day_month, text)
    text = _SLASH_PAIR.sub(slash_pair, text)
    # ทับหลายชั้นไม่เข้าเงื่อนไขกฎคู่ เพราะ lookaround ทั้งสองข้างล้มเหลว
    # "/" จึงหลุดถึง TTS ดิบ ๆ ซึ่งเป็นสิ่งที่โมดูลนี้มีไว้ป้องกัน
    text = _SLASH_CHAIN.sub(lambda m: m.group(0).replace("/", " ทับ "), text)

    def grouped(match: re.Match[str]) -> str:
        # จุลภาคคั่นหลักพันคือหลักฐานว่าเป็น "จำนวน" ไม่ใช่รหัส จึงอ่านเป็นจำนวน
        # เสมอ ไม่ต้องผ่าน _speak_integer ที่จะอ่าน "12,345,678" ทีละตัว
        whole = match.group(1).replace(",", "")
        if len(whole) > _MAX_INT_DIGITS:
            return match.group(0)
        spoken = thai_number_to_words(int(whole))
        if match.group(2):
            spoken += f" จุด {read_digits(match.group(2))}"
        return spoken

    text = _GROUPED_NUMBER.sub(grouped, text)

    def replace(match: re.Match[str]) -> str:
        sign, whole, decimal = match.group(1), match.group(2), match.group(3)
        # เครื่องหมายหน้าเลข TTS อ่านไม่ออก "อุณหภูมิ -5 องศา" จึงฟังเป็น
        # "ห้าองศา" ซึ่งความหมายกลับด้าน "+15%" ก็เสียเครื่องหมายไปเหมือนกัน
        prefix = {"-": "ลบ", "+": "บวก"}.get(sign, "")
        spoken = prefix + _speak_integer(whole)
        if decimal:
            spoken = f"{spoken} จุด {read_digits(decimal)}"
        # ไม่เติมช่องว่างรอบคำอ่าน — ภาษาไทยเขียนติดกันอยู่แล้ว ("ราคา199บาท" ->
        # "ราคาหนึ่งร้อยเก้าสิบเก้าบาท") และถ้าเดิมมีช่องว่างก็ยังอยู่ครบ
        return spoken

    return _STANDALONE_NUMBER.sub(replace, text)


# ── ตัดประโยค ───────────────────────────────────────────────────────────────
# เครื่องหมายจบประโยค
#
# จุดต้องมีตัวอักษรอย่างน้อยสองตัวนำหน้า ไม่งั้นจะไปตัดตัวย่อภาษาไทยที่ใช้จุด
# เป็นปกติ ("พ.ศ." -> "พ." + "ศ.") และต้องไม่มีตัวเลขขนาบ ไม่งั้นจะตัดเวลา
# ("13.30") และราคา ("1,234.50") ซึ่งนอกจากฟังไม่รู้เรื่องแล้ว ยังทำให้ตัวแปลง
# ตัวเลขทำงานไม่ได้ด้วย เพราะระบบตัดท่อนก่อนแล้วค่อยแปลงตัวเลขทีละท่อน
#
# ตัวย่อไทยยาวสองสามตัวอักษรที่ลงท้ายด้วยจุดมีเยอะมาก ("ผศ." "ดร." "บมจ." "ปตท."
# "รศ." "นพ." "พญ." "จำกัด" ฯลฯ) กฎ "สองตัวอักษรนำหน้าจุด" จึงตัดกลางชื่อคน
# ("ผศ. ดร. สมชาย" -> "ผศ. ดร." + "สมชาย") ทำให้ TTS หยุดผิดที่และเสียงขาด
# จึงต้องไม่ตัดเมื่อจุดนั้นตามหลังตัวย่อที่รู้จัก
_THAI_ABBREVIATIONS = (
    # "พล" ต้องอยู่ด้วย ไม่ใช่แค่ "พล.อ" — รายการที่มีจุดอยู่ในตัวกันได้แค่จุด
    # *ที่สอง* จุดแรกของ "พล.อ. ประยุทธ์" จึงยังตัดกลางยศอยู่
    "ผศ", "รศ", "ดร", "นพ", "พญ", "ทพ", "บมจ", "บจก", "หจก", "ปตท",
    "กฟผ", "รร", "รพ", "พล", "ร.ต", "ร.ท", "ร.อ", "พล.ต.อ", "พล.ต.ท",
    "พ.ต", "พ.ท", "พ.อ", "พล.ต", "พล.ท", "พล.อ", "น.ส", "ด.ช", "ด.ญ",
)
_SENT_END = re.compile(r"(?:(?<=[^\W\d_]{2})\.|[!?…‽])+[\s\"'”’)\]]*")


def _sentence_ends(text: str) -> "Iterator[int]":
    """หาตำแหน่งจบประโยค โดยข้ามจุดที่เป็นส่วนของตัวย่อ

    เคยลองใส่ lookbehind ของตัวย่อทุกตัวลงใน regex เอง แต่ Python บังคับให้
    lookbehind กว้างคงที่ จึงต้องต่อกันเป็นลูกโซ่ยี่สิบกว่าตัว ซึ่งทำให้การตัด
    ข้อความยาว ๆ ช้าลงหลายเท่า (เทสต์ความเร็วจับได้) การกรองทีหลังเสียเวลา
    เฉพาะตอนที่เจอจุดจริง ๆ เท่านั้น
    """
    for match in _SENT_END.finditer(text):
        head = text[: match.start()]
        if head and text[match.start()] == "." and _ends_with_abbreviation(head):
            continue
        yield match.end()


def _ends_with_abbreviation(head: str) -> bool:
    """ข้อความก่อนจุดลงท้ายด้วยตัวย่อที่รู้จักไหม ("ผศ" "ดร" "บมจ" "พล.อ")"""
    return any(head.endswith(abbr) for abbr in _THAI_ABBREVIATIONS)
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
        set(_sentence_ends(line))
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
    # ถ้าถอยจนถึงต้นข้อความ แปลว่าทั้งช่วงเป็นเครื่องหมายผสมล้วน (ข้อความที่จงใจ
    # ซ้อนวรรณยุกต์เป็นร้อยตัว) ให้ตัดตรงจุดเดิมแทน ไม่งั้นจะไม่มีความคืบหน้าเลย
    return i if i > 0 else index


def _word_safe_cut(text: str, low: int, high: int) -> int:
    """หาจุดตัดที่ไม่ขาดกลางคำ — คืนค่าในช่วง [1, high]

    พยายามให้อยู่ในช่วง [low, high] แต่ถ้าไม่มีขอบคำในช่วงนั้นเลย จะยอมคืนค่า
    ที่น้อยกว่า low เพื่อไม่ให้ตัดกลางคำ ผลคือได้ท่อนสั้นกว่าที่ตั้งใจ ซึ่งดีกว่า
    ท่อนที่คำขาดครึ่ง

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
    # จุดตัดต้องมากกว่า 0 เสมอ ไม่งั้นผู้เรียกจะตัดข้อความไม่ออกและวนไม่จบ
    return max(1, _cluster_safe_cut(text, high))


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

    def __init__(
        self, min_chars: int = 24, max_chars: int = 160, phrase_min_chars: int = 8
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        # จุดตัดที่ "แน่ใจ" ว่าจบวลีจริง (เครื่องหมายจบประโยคหรือคำลงท้ายสุภาพ)
        # ใช้เกณฑ์ความยาวต่ำกว่าได้ เพราะคำอย่าง "ได้เลยค่ะ" หรือ "จำได้ค่ะ"
        # เป็นประโยคสมบูรณ์ในตัวเอง พูดออกไปทันทีได้เลยและฟังเป็นธรรมชาติ
        # ซึ่งช่วยให้ผู้ฟังได้ยินเสียงแรกเร็วขึ้นมาก
        self.phrase_min_chars = phrase_min_chars
        self._buf = ""

    def feed(self, delta: str) -> Iterator[str]:
        """ป้อนข้อความชิ้นใหม่ แล้ว yield ท่อนที่พร้อมพูด"""
        self._buf += delta
        while True:
            cut = self._find_cut(self._buf)
            if cut is None:
                return
            # ตาข่ายกันวนไม่จบ: จุดตัดต้องเดินหน้าอย่างน้อยหนึ่งตัวอักษรเสมอ
            # ถ้าจุดตัดเป็นศูนย์ บัฟเฟอร์จะไม่สั้นลง แล้วลูปนี้จะหมุนตลอดกาล
            # (เกิดได้จริงกับข้อความที่ซ้อนวรรณยุกต์ไทยติดกันเป็นร้อยตัว)
            cut = max(1, min(cut, len(self._buf)))
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
        if len(buf) < self.phrase_min_chars:
            return None

        nl = buf.find("\n")
        if nl >= 0:
            # ขึ้นบรรทัดใหม่คือจุดตัดที่ชัดที่สุด ใช้ได้แม้ท่อนจะสั้น
            return nl + 1

        for end in _sentence_ends(buf):
            if end >= self.phrase_min_chars:
                return end

        for match in _PARTICLE_BREAK.finditer(buf):
            # ต้องมีช่องว่างตามหลังจริง ๆ ในบัฟเฟอร์ ไม่ใช่แค่ "จบบัฟเฟอร์"
            #
            # คำลงท้ายหลายคำเป็นคำย่อยของคำธรรมดา (ค่า ใน ค่าไฟ, คับ ใน คับแคบ,
            # จ้า ใน จ้าง) ระหว่างสตรีมบัฟเฟอร์จะจบกลางคำเหล่านั้นชั่วคราว
            # ถ้ายอมให้ $ นับเป็นจุดตัด ระบบจะตัด "เขาเป็นคนคับ" ออกไปพูดก่อน
            # แล้วค่อยพูด "แคบมาก" ตามมา ซึ่งคนฟังจับได้ทันที และเกิดแบบสุ่ม
            # ตามจังหวะที่ข้อความไหลมา จุดจบจริงถูกจัดการโดย flush() อยู่แล้ว
            if self.phrase_min_chars <= match.end() < len(buf):
                return match.end()

        if len(buf) < self.min_chars:
            return None

        # ภาษาไทยใช้เว้นวรรคคั่น "วลี" ไม่ใช่คั่นคำ จุดเว้นวรรคจึงเป็นจุดพักเสียง
        # ที่ฟังแล้วเป็นธรรมชาติ และมักเป็นจุดตัดเดียวที่มีในคำตอบที่ไม่มีคำลงท้าย
        # กลางประโยค ถ้ารอถึง max_chars อย่างเดิม คำตอบยาว 80 อักขระจะกลายเป็น
        # ท่อนเดียว แปลว่าต้องรอโมเดลพิมพ์จนจบก่อนถึงจะเริ่มพูด ซึ่งทำลายเหตุผล
        # ทั้งหมดของการสตรีม
        # อย่าตัดคั่นตัวเลขกับสิ่งที่อยู่ติดกัน ทั้งคำนำหน้า ("ชั้น 4") และหน่วย
        # ที่ตามมา ("13.30 น.") กรณีหลังสำคัญเป็นพิเศษ เพราะการตัดตรงนั้นทำให้
        # ตัวแปลงเวลามองไม่เห็น "น." แล้วอ่าน 13.30 เป็นทศนิยมแทน
        # ต้องเห็นตัวอักษรถัดจากเว้นวรรคก่อนเสมอ ถ้าเว้นวรรคอยู่ท้ายบัฟเฟอร์พอดี
        # เรายังไม่รู้ว่าอะไรตามมา การตัดตรงนั้นทำให้ "13.30 น." ขาดจากกัน
        # แล้วตัวแปลงเวลามองไม่เห็น "น." จึงอ่านเป็นทศนิยมแทน
        # อย่าตัดจนเหลือหางสั้นจู๋ ("... เซนติเมตร" | "ค่ะ") — คำลงท้ายคำเดียว
        # ที่ถูกส่งไปสังเคราะห์เสียงแยกต่างหากฟังเหมือนคนละประโยค
        # ระหว่างสตรีมยังมีข้อความตามมาอีก รอให้หางยาวพอแล้วค่อยตัด
        space = buf.find(" ", self.min_chars)
        while space > 0 and space + 1 < len(buf):
            if buf[space + 1].isdigit() or buf[space - 1].isdigit():
                space = buf.find(" ", space + 1)
                continue
            if len(buf) - (space + 1) < self.phrase_min_chars:
                space = buf.find(" ", space + 1)
                continue
            return space + 1

        # ไม่มีเว้นวรรคที่ใช้ได้ — บังคับตัดที่ขอบคำเมื่อยาวเกินไป
        if len(buf) >= self.max_chars:
            return _word_safe_cut(buf, self.min_chars, self.max_chars)
        return None
