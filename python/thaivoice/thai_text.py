"""ยูทิลิตี้ภาษาไทยสำหรับระบบสนทนาด้วยเสียง

ภาษาไทยมีปัญหาเฉพาะตัวที่ระบบเสียงทั่วไปจัดการไม่ได้:

1. **ไม่มีเว้นวรรคระหว่างคำ และไม่มีเครื่องหมายจบประโยค** — จะตัดข้อความที่ไหล
   ออกมาจากโมเดลเป็นท่อน ๆ เพื่อส่งให้ TTS พูดทันที (ลด latency) ไม่ได้ตรง ๆ
   ต้องใช้กฎเว้นวรรค + คำลงท้าย + ความยาว
2. **คำลงท้าย ครับ/ค่ะ** — เป็นตัวบอกเพศผู้พูด และถ้าตอบผิดจะฟังดูแปลกมาก
3. **markdown อ่านออกเสียงไม่ได้** — ดอกจัน หัวข้อ โค้ดบล็อก ต้องถูกถอดทิ้ง
4. **ตัวเลข** — TTS ไทยอ่านเลขเรียง (เบอร์โทร/ปี) ผิดบ่อย ต้องแปลงเป็นคำอ่าน

ทุกฟังก์ชันในไฟล์นี้เป็น pure function ไม่พึ่ง network จึงทดสอบได้ตรง ๆ
ถ้าติดตั้ง ``pythainlp`` ไว้ จะใช้ตัวตัดประโยคของ pythainlp เพื่อความแม่นยำขึ้น
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
    "particle_for_gender",
    "read_digits",
    "split_sentences",
    "thai_number_to_words",
    "thai_ratio",
    "normalize_transcript",
]

# ── ตารางคำอ่านตัวเลข ────────────────────────────────────────────────────────
_DIGITS = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]
_PLACES = ["", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน"]
_THAI_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("๐๑๒๓๔๕๖๗๘๙")}

# คำลงท้ายสุภาพ — ใช้ทั้งตรวจจับเพศผู้พูด และเลือกคำลงท้ายของบอท
THAI_PARTICLES: dict[str, tuple[str, ...]] = {
    "male": ("ครับ", "คับ", "ครัช", "ฮะ", "คร้าบ"),
    "female": ("ค่ะ", "คะ", "ค่า", "จ้ะ", "จ๊ะ", "ขา"),
}

_THAI_RANGE = re.compile(r"[฀-๿]")

# หน่วย/ตัวย่อที่ TTS ไทยมักอ่านผิด
_UNIT_MAP = {
    "km/h": "กิโลเมตรต่อชั่วโมง",
    "km": "กิโลเมตร",
    "cm": "เซนติเมตร",
    "mm": "มิลลิเมตร",
    "kg": "กิโลกรัม",
    "GB": "กิกะไบต์",
    "MB": "เมกะไบต์",
    "KB": "กิโลไบต์",
    "TB": "เทระไบต์",
    "%": "เปอร์เซ็นต์",
    "°C": "องศาเซลเซียส",
    "&": "และ",
    "@": "แอท",
}


# ── ทำความสะอาดข้อความก่อนอ่านออกเสียง ──────────────────────────────────────
def clean_for_speech(text: str) -> str:
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

    # emoji และสัญลักษณ์ที่อ่านไม่ได้
    s = "".join(
        ch for ch in s if unicodedata.category(ch) not in {"So", "Sk", "Cf", "Cs"}
    )

    # หน่วยที่อ่านผิดบ่อย
    for unit, spoken in _UNIT_MAP.items():
        s = s.replace(unit, f" {spoken} ")

    # เลขไทย -> เลขอารบิก (TTS ไทยอ่านเลขอารบิกได้ดีกว่า)
    s = s.translate(_THAI_DIGIT_MAP)

    # ยุบช่องว่าง
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def normalize_transcript(text: str) -> str:
    """เก็บกวาดข้อความที่ได้จาก STT ก่อนส่งให้โมเดล

    Whisper ภาษาไทยชอบแถมช่องว่างเกิน, จุดไข่ปลา, และซ้ำคำท้ายประโยค
    """
    if not text:
        return ""
    s = text.strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\.{3,}", " ", s)
    # ตัดคำเดิมที่ซ้ำติดกันเกิน 2 ครั้ง (อาการ hallucination ของ ASR)
    s = re.sub(r"(\S+?)(?:\s*\1){2,}", r"\1", s)
    return s.strip()


def thai_ratio(text: str) -> float:
    """สัดส่วนอักขระไทยต่ออักขระที่ไม่ใช่ช่องว่าง (0.0-1.0)"""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _THAI_RANGE.match(c)) / len(chars)


# ── คำลงท้ายสุภาพ ───────────────────────────────────────────────────────────
def detect_particle(text: str) -> str | None:
    """เดาเพศผู้พูดจากคำลงท้าย คืน ``"male"`` / ``"female"`` / ``None``

    ดูเฉพาะช่วงท้ายข้อความ เพราะ "ครับ" กลางประโยคอาจเป็นการยกคำพูดคนอื่น
    """
    if not text:
        return None
    tail = text.strip()[-24:]
    best: tuple[int, str] | None = None
    for gender, particles in THAI_PARTICLES.items():
        for p in particles:
            idx = tail.rfind(p)
            if idx >= 0 and (best is None or idx > best[0]):
                best = (idx, gender)
    return best[1] if best else None


def particle_for_gender(gender: str | None) -> str:
    """คำลงท้ายที่บอทควรใช้ — ค่าเริ่มต้นเป็น 'ครับ' ตามธรรมเนียมผู้ช่วย"""
    return "ค่ะ" if gender == "female" else "ครับ"


# ── คำอ่านตัวเลข ────────────────────────────────────────────────────────────
def thai_number_to_words(n: int) -> str:
    """แปลงจำนวนเต็มเป็นคำอ่านภาษาไทย

    >>> thai_number_to_words(21)
    'ยี่สิบเอ็ด'
    >>> thai_number_to_words(1000000)
    'หนึ่งล้าน'
    """
    if n < 0:
        return "ลบ" + thai_number_to_words(-n)
    if n == 0:
        return "ศูนย์"
    if n >= 1_000_000:
        head, tail = divmod(n, 1_000_000)
        out = thai_number_to_words(head) + "ล้าน"
        return out + (thai_number_to_words(tail) if tail else "")

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
        elif place == 0 and d == 1 and length > 1:  # ลงท้ายด้วยหนึ่ง -> เอ็ด
            out.append("เอ็ด")
        else:
            out.append(_DIGITS[d] + _PLACES[place])
    return "".join(out)


def read_digits(s: str) -> str:
    """อ่านตัวเลขทีละตัว เหมาะกับเบอร์โทร/รหัส

    >>> read_digits("081-234")
    'ศูนย์ แปด หนึ่ง สอง สาม สี่'
    """
    return " ".join(_DIGITS[int(c)] for c in s if c.isdigit())


# ── ตัดประโยค ───────────────────────────────────────────────────────────────
_SENT_END = re.compile(r"[.!?…‽]+[\s\"'”’)\]]*")
_PARTICLE_BREAK = re.compile(
    r"(?:" + "|".join(re.escape(p) for g in THAI_PARTICLES.values() for p in g) + r")(?=\s|$)"
)


def _pythainlp_sentences(text: str) -> list[str] | None:
    try:
        from pythainlp.tokenize import sent_tokenize  # type: ignore
    except Exception:
        return None
    try:
        parts = [p.strip() for p in sent_tokenize(text, engine="crfcut")]
        return [p for p in parts if p] or None
    except Exception:
        return None


def split_sentences(text: str, min_chars: int = 12) -> list[str]:
    """ตัดข้อความไทยเป็นประโยคสำหรับส่งให้ TTS ทีละท่อน

    ใช้ pythainlp ถ้ามี ไม่งั้นถอยไปใช้กฎ: ขึ้นบรรทัดใหม่ / เครื่องหมายจบประโยค /
    คำลงท้ายสุภาพ เป็นจุดตัด
    """
    text = text.strip()
    if not text:
        return []

    via_lib = _pythainlp_sentences(text)
    if via_lib:
        return _merge_short(via_lib, min_chars)

    parts: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
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
    return _merge_short(parts, min_chars)


def _merge_short(parts: list[str], min_chars: int) -> list[str]:
    """รวมท่อนที่สั้นเกินไปเข้ากับท่อนถัดไป — ท่อนสั้นทำให้เสียงพูดขาดเป็นห้วง ๆ"""
    out: list[str] = []
    for part in parts:
        if out and len(out[-1]) < min_chars:
            out[-1] = f"{out[-1]} {part}".strip()
        else:
            out.append(part)
    return out


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
        if nl >= self.min_chars:
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
            return space + 1 if space > 0 else self.max_chars
        return None
