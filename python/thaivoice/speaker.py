"""จดจำ "ใครกำลังพูด" จากลายเสียง (speaker recognition)

หลักการ: แปลงเสียงพูดเป็นเวกเตอร์ลายเสียง (embedding) แล้วเทียบ cosine similarity
กับลายเสียงของทุกคนที่เคยคุยกัน ถ้าคล้ายพอและทิ้งห่างอันดับสองมากพอ ก็ถือว่าเป็น
คนเดิม ถ้าไม่ใช่ ระบบจะถามชื่อแล้วลงทะเบียนเสียงใหม่

ตัวสร้าง embedding เป็นแบบถอดเปลี่ยนได้:

* ``resemblyzer``  — ค่าเริ่มต้น น้ำหนักเบา รันบน CPU ได้
* ``speechbrain``  — แม่นกว่า (ECAPA-TDNN) แต่กินทรัพยากรมากกว่า
* ``none``         — ปิดการจำเสียง ใช้การบอกชื่อแทน

ถ้าไม่มีไลบรารีใดเลย ระบบยังทำงานได้ปกติ เพียงแต่ต้องระบุตัวตนด้วยการบอกชื่อ
(``extract_name_claim`` จับประโยคแนะนำตัวภาษาไทยให้อัตโนมัติ)
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .memory import MemoryStore, Speaker

__all__ = [
    "Embedder",
    "Identification",
    "SpeakerIdentifier",
    "cosine_similarity",
    "extract_name_claim",
    "load_embedder",
]


# ── หาชื่อจากประโยคแนะนำตัวภาษาไทย ──────────────────────────────────────────
#
# กฎเหล็กของส่วนนี้: **ยอมพลาดดีกว่าเดาผิด**
#
# ถ้าจับชื่อผิด ระบบจะสร้างตัวตนใหม่ขึ้นมาแล้วผูกลายเสียงของผู้ใช้เข้ากับตัวตนนั้น
# ความจำจริงของเขาจะถูกทิ้งค้างไว้ และครั้งต่อไประบบจะทักเขาด้วยชื่อผิด
# ส่วนการ "พลาด" แค่ทำให้ต้องบอกชื่ออีกครั้ง ซึ่งเสียหายน้อยกว่ามาก
#
# จึงบังคับว่าต้องขึ้นต้น *วรรค* ด้วยสรรพนามบุรุษที่หนึ่งหรือคำว่า "ชื่อ" เท่านั้น
# ภาษาไทยไม่เว้นวรรคระหว่างคำแต่เว้นระหว่างวรรคตอน กฎนี้จึงยอมรับ
# "สวัสดีครับ ผมชื่อเดชครับ" แต่ตัด "เพื่อนผมชื่อสมชาย" และ "แมวผมชื่อมิ้นท์" ทิ้ง
# เพราะสรรพนามในสองประโยคหลังเป็นเจ้าของ ไม่ได้ขึ้นต้นวรรค

_PRONOUN = r"(?:ผม|ดิฉัน|ฉัน|หนู|กระผม|ข้าพเจ้า|เรา|อั๊ว)"
_NAME_CHARS = r"[ก-๙A-Za-z][ก-๙A-Za-z0-9.'-]{0,14}"

_NAME_PATTERNS = [
    # "ชื่อเล่นชื่อบิ๊กครับ" / "ชื่อเล่นว่าบิ๊ก" — ต้องมาก่อนกฎทั่วไป
    re.compile(rf"ชื่อเล่น(?:ชื่อ|ว่า|คือ)?\s*({_NAME_CHARS})"),
    # "ผมชื่อสมชายครับ" / "หนูชื่อว่าแนนค่ะ"
    # ลำดับใน (?:...) สำคัญ — ต้องลอง "จริงว่า" ก่อน "ว่า" ไม่งั้น "ชื่อจริงว่าอนุชา"
    # จะจับได้เป็น "จริงว่าอนุชา" แล้วถูกปฏิเสธเพราะขึ้นต้นด้วย "จริง"
    re.compile(rf"(?:^|\s){_PRONOUN}\s*ชื่อ(?:จริงว่า|ว่า)?\s*({_NAME_CHARS})"),
    # "ชื่อเดชครับ" / "ชื่อผมสมชายครับ" — ต้องอยู่ต้นวรรคเท่านั้น
    re.compile(rf"(?:^|\s)ชื่อ(?:จริงว่า|ว่า)?\s*(?:{_PRONOUN})?\s*({_NAME_CHARS})"),
    # "เรียกผมว่าพี่เดช" / "เรียกว่าโบท"
    re.compile(rf"เรียก(?:ผม|ฉัน|หนู|เรา|ดิฉัน)?ว่า\s*({_NAME_CHARS})"),
    # "เรียกเดชก็ได้"
    re.compile(rf"เรียก\s*({_NAME_CHARS}?)\s*ก็ได้"),
    # "ผมคือสมชายครับ"
    re.compile(rf"(?:^|\s){_PRONOUN}\s*คือ\s*({_NAME_CHARS})"),
    #
    # เคยมีอีกสองกฎตรงนี้ที่ถูก *ถอดออก* โดยตั้งใจ:
    #
    #   "นี่<X>เองนะ"          และ   "<สรรพนาม><X><คำลงท้าย>"
    #
    # ทั้งคู่เดาว่าคำที่อยู่ตรงกลางคือชื่อ ซึ่งเป็นจริงกับ "ผมเดชครับ" แต่ก็เป็นจริง
    # กับประโยคธรรมดาอีกนับไม่ถ้วน: "ผมหิวครับ" -> ชื่อ "หิว", "ผมขอโทษครับ" ->
    # ชื่อ "ขอโทษ", "หนูง่วงค่ะ" -> ชื่อ "ง่วง", "นี่เอกสารนะครับ" -> ชื่อ "เอกสาร"
    # แต่ละครั้งคือการสร้างตัวตนใหม่แล้วผูกลายเสียงของผู้ใช้เข้ากับตัวตนนั้น
    # ความจำจริงของเขาถูกทิ้งค้าง และครั้งหน้าระบบจะทักเขาว่า "คุณหิว"
    #
    # การแยกสองกรณีนี้ออกจากกันต้องใช้พจนานุกรมชื่อคน ซึ่งไม่มีทางครบ (ชื่อเล่นไทย
    # จำนวนมากเป็นคำธรรมดาอยู่แล้ว เช่น ต้น น้ำ ก้อง) จึงเลือกตัดทิ้ง แล้วไปรับ
    # "ชื่อเปล่า ๆ" เฉพาะตอนที่บอทเพิ่งถามชื่อแทน (ดู expecting_name)
    # ภาษาอังกฤษ — ต้องขึ้นต้นด้วยตัวใหญ่ ไม่งั้น "I'm going" จะกลายเป็นชื่อ "going"
    # ไม่ใช้ re.IGNORECASE เพราะต้องบังคับให้ *ชื่อ* ขึ้นต้นด้วยตัวใหญ่
    # ("I'm going" จะได้ไม่กลายเป็นชื่อ "going") จึงเขียนตัวเลือกตัวพิมพ์เอง
    re.compile(
        r"(?:\b[Mm]y name is\b|\b[Ii]\s*'?m\b|\b[Ii] am\b|\b[Cc]all me\b)\s+"
        r"([A-Z][A-Za-z.'-]{1,19})"
    ),
]

# คำที่มักติดมาท้ายชื่อเวลาพูด ต้องตัดทิ้งเพราะภาษาไทยไม่เว้นวรรคระหว่างคำ
# "นะ" เดี่ยว ๆ ไม่อยู่ในลิสต์นี้ เพราะเป็นพยางค์ของชื่อไทยจริง ("มานะ" "อนงค์นะ")
# ตัดเฉพาะตอนที่มันเกาะอยู่กับคำลงท้ายจริง ("นะครับ" "นะคะ") ซึ่งชัดเจนว่าไม่ใช่ชื่อ
_NAME_SUFFIXES = (
    "นะครับผม", "นะครับ", "นะคะ", "นะค่ะ", "นะฮะ",
    "ครับผม", "ครับ", "คร้าบ", "คับ", "ค่ะ", "คะ", "ค่า", "ขา", "จ้า", "จ๊ะ", "จ้ะ",
    "ฮะ", "เอง", "ก็ได้", "ได้เลย", "แหละ", "ล่ะ", "เลย", "อ่ะ",
)

# คำนำหน้าที่ไม่ใช่ส่วนหนึ่งของชื่อ
#
# ตั้งใจไม่ใส่คำสั้น ๆ อย่าง "อา" "น้า" "ป้า" เพราะมันไปกินต้นชื่อจริง:
# "อาทิตย์" -> "ทิตย์", "อารีย์" -> "รีย์", "อาร์ม" -> "ร์ม" (ขึ้นต้นด้วยวรรณยุกต์
# ซึ่งออกเสียงไม่ได้เลย) ส่วนคำยาวกว่านั้นชนกับชื่อจริงน้อยมาก
_NAME_TITLES = ("นางสาว", "นาย", "นาง", "คุณ", "พี่", "น้อง")

# อักขระที่ขึ้นต้นคำไทยไม่ได้ (สระตาม วรรณยุกต์ และเครื่องหมายกำกับ)
# ถ้าตัดคำนำหน้าแล้วเหลือขึ้นต้นด้วยตัวพวกนี้ แปลว่าเราตัดกลางคำ ไม่ใช่ตัดคำนำหน้า
# เช่น "คุณากร" ตัด "คุณ" ออกจะเหลือ "ากร" ซึ่งอ่านไม่ได้
_CANNOT_START_WORD = "าิีึืุูัํ็์ฺ่้๊๋ๅๆ"

# ถ้าชื่อที่จับได้ "ขึ้นต้นด้วย" คำพวกนี้ แปลว่าไม่ใช่การแนะนำตัว
# (เช่น "ชื่อเสียง" = reputation, "ชื่อไฟล์" = filename)
_NOT_A_NAME_PREFIXES = (
    "ชื่อ", "เสียง", "เล่น", "จริง", "ไฟล์", "ร้าน", "ผู้", "หนัง", "บริษัท", "เพลง",
    "หนังสือ", "ยา", "โรค", "ถนน", "ซอย", "เมนู", "สินค้า", "แบรนด์", "ทีม",
    "วง", "ตัวละคร", "เรื่อง", "ลูก", "แมว", "หมา", "เพื่อน", "เขา", "เธอ",
    "อะไร", "นี้", "นั้น", "บัญชี", "โดเมน", "จังหวัด", "โรงเรียน", "เดียว",
    "ตาม", "ที่", "ของ",
)

# ถ้าในชื่อมีคำพวกนี้อยู่ แปลว่าจับเกินไปโดนคำอื่นเข้ามาด้วย
# ระวัง: ห้ามใส่พยางค์ที่พบในชื่อคนไทยจริง ๆ ลงในลิสต์นี้
# เคยใส่ "มา" "จำ" "ไป" ไว้ ทำให้ชื่อธรรมดาอย่าง มาลี มานี สมหมาย จำเนียร ไปรยา
# ถูกปฏิเสธทิ้งหมด ซึ่งแย่กว่าปัญหาที่ตั้งใจกันตั้งแต่แรก
_NAME_REJECT_SUBSTRINGS = (
    "อะไร", "ไหม", "มั้ย", "เหรอ", "ยังไง", "คือ", "ว่า", "แล้ว",
    "หน่อย", "ด้วย", "ไม่",
    # คำเชื่อม — ถ้าติดมาแปลว่าจับเลยขอบชื่อไปโดนประโยคถัดไปแล้ว
    "แต่", "และ", "หรือ", "กับ",
)

# คำที่เป็นคำตอบสั้น ๆ ได้ แต่ไม่มีทางเป็นชื่อคน
# ใช้เฉพาะตอนที่บอทเพิ่งถามชื่อ ซึ่งเป็นบริบทเดียวที่เรายอมรับ "คำเปล่า ๆ" เป็นชื่อ
_NOT_A_NAME_ANSWERS = {
    "ครับ", "ครับผม", "ค่ะ", "คะ", "ค่า", "จ้า", "จ๊ะ", "ฮะ", "โอเค", "ตกลง",
    "ใช่", "ไม่", "ไม่ใช่", "ได้", "ไม่ได้", "เอา", "ไม่เอา", "อะไร", "ไง",
    "ขอโทษ", "ขอบคุณ", "สวัสดี", "หิว", "ง่วง", "เหนื่อย", "ป่วย", "เบื่อ",
    "หยุด", "รอ", "แป๊บ", "เดี๋ยว", "อืม", "เออ", "หา", "อะ", "นะ", "จบ",
    "ไม่บอก", "ลับ", "ไม่รู้", "ทำไม", "ยังไง", "ok", "yes", "no", "okay",
    "hello", "hi", "hey", "what", "why", "sure", "thanks", "sorry", "please",
    # อาชีพและสถานที่ที่คนมักตอบแทนชื่อเมื่อเลี่ยงไม่บอก
    "หมอ", "ครู", "อาจารย์", "พยาบาล", "ตำรวจ", "ทหาร", "วิศวกร", "นักเรียน",
    "นักศึกษา", "พนักงาน", "แม่บ้าน", "คนไทย", "กรุงเทพ", "เชียงใหม่", "ภูเก็ต",
    "ขอนแก่น", "ไทย", "คนขับ",
}

# ชื่อจริงที่ลงท้ายด้วย "นะ" — กันไม่ให้ถูกตัดเป็น "มา"
_NAMES_ENDING_IN_NA = {"มานะ", "ปัญญะ", "วีระนะ"}

# คำที่ขึ้นต้นคำตอบซึ่งบอกชัดว่าไม่ใช่ชื่อ ใช้เฉพาะตอนตีความคำตอบเปล่า ๆ
# (ทุกคำยาว >= 3 ตัวอักษร เพื่อไม่ให้ไปกินต้นชื่อจริง)
_NOT_A_NAME_STARTS = (
    "ทำไม", "เดี๋ยว", "กำลัง", "อยาก", "ต้อง", "ช่วย", "ขอโทษ", "ขอบ",
    "เมื่อ", "ตอนนี้", "วันนี้", "อาจ", "คงจะ", "น่าจะ", "เพราะ", "ตอบ",
    "บอกไม่", "ไม่อยาก", "ไม่บอก", "เอาไว้", "แล้วแต่",
    "หิว", "ง่วง", "เหนื่อย", "ปวด", "เบื่อ", "ไม่ใช่", "ไม่ได้",
)

_STOPWORD_NAMES = {
    "อะไร", "ไหน", "ใคร", "นี้", "นั้น", "เธอ", "คุณ", "เขา", "มัน", "เรา",
    "ชื่อ", "ตัว", "คน", "งาน", "วันนี้",
}


def _clean_name(raw: str) -> str | None:
    """ตัดคำลงท้าย/คำนำหน้าออกจากชื่อที่จับได้ แล้วตรวจว่าน่าเชื่อถือไหม"""
    name = raw.strip(" .,!?\"'")

    # ตัดคำลงท้ายที่ติดมา (วนซ้ำเผื่อซ้อนกัน เช่น "นะครับ")
    changed = True
    while changed and name:
        changed = False
        for suffix in _NAME_SUFFIXES:
            if not name.endswith(suffix):
                continue
            rest = name[: -len(suffix)].strip()
            # คำลงท้ายที่ขึ้นต้นด้วย "นะ" คลุมเครือ เพราะ "นะ" เป็นพยางค์ของชื่อได้
            # ("มานะครับ" ควรได้ "มานะ") แต่การกันไว้ด้วยความยาวขั้นต่ำทำให้ชื่อเล่น
            # สองตัวอักษรซึ่งพบบ่อยมาก (โอ มด นก ปอ เอ บี เจ กบ) กลายเป็น "โอนะ"
            # "มดนะ" ติดตัวไปตลอด — พลาดบ่อยกว่ากันเยอะ จึงตัด "นะ" ออกเสมอ
            # ยกเว้นชื่อไม่กี่ชื่อที่ลงท้ายด้วย "นะ" จริง ๆ
            if suffix.startswith("นะ") and (rest + "นะ") in _NAMES_ENDING_IN_NA:
                continue
            if len(rest) >= 2:
                name = rest
                changed = True
                break

    # ตัดคำนำหน้า เช่น "พี่เดช" -> "เดช"
    for title in _NAME_TITLES:
        if len(name) > len(title) + 1 and name.startswith(title):
            rest = name[len(title) :].strip()
            # ถ้าส่วนที่เหลือขึ้นต้นด้วยเครื่องหมายผสมหรือสระหน้าที่ไม่มีพยัญชนะ
            # แปลว่าเราตัดกลางคำ ไม่ใช่ตัดคำนำหน้า
            if (
                len(rest) >= 2
                and unicodedata.category(rest[0]) != "Mn"
                and rest[0] not in _CANNOT_START_WORD
            ):
                name = rest
            break

    if len(name) < 2 or len(name) > 15:
        return None
    if name in _STOPWORD_NAMES:
        return None
    if any(name.startswith(prefix) for prefix in _NOT_A_NAME_PREFIXES):
        return None
    if any(bad in name for bad in _NAME_REJECT_SUBSTRINGS):
        return None
    if name.lower() in _NOT_A_NAME_ANSWERS:
        return None
    return name


def extract_name_claim(text: str, expecting_name: bool = False) -> str | None:
    """ดึงชื่อจากประโยคแนะนำตัว เช่น "ผมชื่อสมชายครับ" -> "สมชาย"

    คืน ``None`` เมื่อไม่มั่นใจ ซึ่งเป็นค่าที่ปลอดภัยกว่าเสมอ

    ``expecting_name`` ใช้เมื่อบอท *เพิ่งถามชื่อไปหมาด ๆ* เท่านั้น ในบริบทนั้น
    คำตอบสั้น ๆ อย่าง "เดช" คือชื่อแน่นอน ซึ่งเป็นการเดาที่ปลอดภัยเพราะเรารู้ว่า
    เราเพิ่งถามอะไรไป ต่างจากการเดาจากประโยคลอย ๆ ที่ผิดได้ง่ายมาก

    >>> extract_name_claim("ผมชื่อสมชายครับ")
    'สมชาย'
    >>> extract_name_claim("เพื่อนผมชื่อสมชาย")
    >>> extract_name_claim("ผมหิวครับ")
    >>> extract_name_claim("เดชครับ", expecting_name=True)
    'เดช'
    """
    if not text:
        return None
    for pattern in _NAME_PATTERNS:
        for match in pattern.finditer(text):
            name = _clean_name(match.group(1))
            if name:
                return name

    if expecting_name:
        stripped = text.strip()
        # คำตอบต้องสั้นและเป็นคำเดียว ไม่งั้นน่าจะเป็นประโยคอื่นที่พูดต่อ
        if len(stripped) <= 24 and len(stripped.split()) <= 2:
            candidate = _clean_name(stripped.replace(" ", ""))
            # คำตอบสั้น ๆ ส่วนใหญ่ไม่ใช่ชื่อ ("ครับ" "ขอโทษ" "ไม่บอก")
            # _clean_name กรองให้ชั้นหนึ่งแล้ว ตรวจซ้ำหลังตัดคำลงท้ายอีกที
            if candidate and _looks_like_a_name(candidate):
                return candidate
    return None


def _looks_like_a_name(candidate: str) -> bool:
    """ตรวจว่าคำตอบเปล่า ๆ พอจะเป็นชื่อคนได้ไหม

    ใช้เฉพาะตอน ``expecting_name`` ซึ่งเป็นบริบทที่ยอมรับคำเดี่ยว ๆ เป็นชื่อ
    การเช็คแค่ "ไม่อยู่ในบัญชีดำ" หลวมเกินไป — "35" "3.5" "อายุ30" "hello"
    ผ่านหมด แล้วกลายเป็นตัวตนถาวรของผู้ใช้ ที่นี่จึงเรียกร้องหลักฐานเชิงบวก
    """
    if candidate.lower() in _NOT_A_NAME_ANSWERS:
        return False
    # ชื่อคนไม่มีตัวเลขและไม่มีเครื่องหมายวรรคตอน ("35" "3.5" "ก.พ." "อายุ30")
    if any(ch.isdigit() for ch in candidate):
        return False
    if any(ch in candidate for ch in ".,:;/\\()[]{}<>@#$%^&*+=|~`\"'!?"):
        return False
    if any(candidate.startswith(bad) for bad in _NOT_A_NAME_STARTS):
        return False
    # ต้องมีพยัญชนะไทยหรือตัวอักษรละตินอย่างน้อยหนึ่งตัว
    return any("ก" <= ch <= "ฮ" or ch.isalpha() for ch in candidate)


# ── ตัวสร้าง embedding ──────────────────────────────────────────────────────
@runtime_checkable
class Embedder(Protocol):
    """อินเทอร์เฟซของตัวแปลงเสียงเป็นเวกเตอร์ลายเสียง"""

    name: str

    def embed(self, pcm: bytes, sample_rate: int) -> list[float]:
        """รับ PCM 16-bit mono คืนเวกเตอร์ลายเสียงที่ normalize แล้ว"""
        ...


class ResemblyzerEmbedder:
    """ลายเสียงด้วย Resemblyzer (GE2E) — เบาและเร็วพอสำหรับงานเรียลไทม์"""

    name = "resemblyzer"

    def __init__(self) -> None:
        from resemblyzer import VoiceEncoder  # type: ignore

        self._encoder = VoiceEncoder()

    def embed(self, pcm: bytes, sample_rate: int) -> list[float]:
        import numpy as np  # type: ignore
        from resemblyzer import preprocess_wav  # type: ignore

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        wav = preprocess_wav(samples, source_sr=sample_rate)
        vec = self._encoder.embed_utterance(wav)
        return [float(x) for x in vec]


class SpeechBrainEmbedder:
    """ลายเสียงด้วย SpeechBrain ECAPA-TDNN — แม่นกว่าแต่หนักกว่า"""

    name = "speechbrain"

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        import torch  # type: ignore  # noqa: F401
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        self._model = EncoderClassifier.from_hparams(source=source)

    def embed(self, pcm: bytes, sample_rate: int) -> list[float]:
        import numpy as np  # type: ignore
        import torch  # type: ignore

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples).unsqueeze(0)
        with torch.no_grad():
            vec = self._model.encode_batch(tensor).squeeze().cpu().numpy()
        return [float(x) for x in vec]


def load_embedder(backend: str) -> Embedder | None:
    """สร้างตัวแปลงลายเสียงตามชื่อ backend — คืน ``None`` ถ้าใช้ไม่ได้

    ตั้งใจไม่ให้ throw เพื่อให้ระบบยังคุยได้แม้ยังไม่ได้ติดตั้งไลบรารีจำเสียง
    """
    backend = (backend or "none").strip().lower()
    if backend in {"none", "off", "disabled", ""}:
        return None
    try:
        if backend == "resemblyzer":
            return ResemblyzerEmbedder()
        if backend == "speechbrain":
            return SpeechBrainEmbedder()
    except Exception:
        return None
    return None


# ── เทียบความคล้าย ──────────────────────────────────────────────────────────
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """ความคล้ายเชิงโคไซน์ (-1 ถึง 1) — คืน 0.0 ถ้าเวกเตอร์ไม่เข้ากันหรือมีค่าเสีย

    ต้องกัน NaN/inf ให้ดี เพราะการเทียบกับ NaN ให้ผลเป็น False ทุกกรณี ถ้าปล่อยให้
    NaN หลุดออกไป การเรียงลำดับผู้พูดจะเพี้ยนทั้งชุด และ "คนที่คล้ายที่สุด" อาจ
    กลายเป็นแถวที่เสียหาย ทำให้ระบบจำใครไม่ได้เลยทั้งที่ลายเสียงตรงเป๊ะ
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        if not (math.isfinite(x) and math.isfinite(y)):
            return 0.0
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    score = dot / (math.sqrt(na) * math.sqrt(nb))
    return score if math.isfinite(score) else 0.0


@dataclass
class Identification:
    """ผลการระบุตัวผู้พูด"""

    speaker: Speaker | None
    score: float = 0.0
    runner_up: float = 0.0
    is_new: bool = False
    method: str = "none"  # voice | name | fallback | none

    @property
    def confident(self) -> bool:
        return self.speaker is not None and not self.is_new


class SpeakerIdentifier:
    """ระบุตัวผู้พูดจากเสียง และลงทะเบียนคนใหม่

    ``threshold`` คือความคล้ายขั้นต่ำที่จะยอมรับว่าเป็นคนเดิม
    ``margin`` คือระยะห่างขั้นต่ำจากอันดับสอง — กันกรณีเสียงคล้ายกันสองคน
    ทำให้ระบบสลับตัวคนกลางบทสนทนา
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder | None,
        *,
        threshold: float = 0.75,
        margin: float = 0.06,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self.margin = margin

    @property
    def enabled(self) -> bool:
        return self.embedder is not None

    @property
    def backend_name(self) -> str:
        return self.embedder.name if self.embedder else "none"

    def identify(self, pcm: bytes, sample_rate: int) -> Identification:
        """ระบุว่าเสียงนี้เป็นของใคร"""
        if self.embedder is None:
            return Identification(speaker=None, method="none")
        try:
            vec = self.embedder.embed(pcm, sample_rate)
        except Exception:
            return Identification(speaker=None, method="none")
        return self.identify_embedding(vec)

    def identify_embedding(self, vec: Sequence[float]) -> Identification:
        """เทียบเวกเตอร์ลายเสียงกับทุกคนในฐานความจำ"""
        backend = self.backend_name
        scored = sorted(
            (
                (cosine_similarity(vec, stored), speaker_id)
                for speaker_id, stored in self.store.all_voiceprints(backend)
            ),
            reverse=True,
        )
        if not scored:
            return Identification(speaker=None, is_new=True, method="voice")

        best_score, best_id = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0

        if best_score >= self.threshold and (best_score - runner_up) >= self.margin:
            speaker = self.store.get_speaker(best_id)
            if speaker is not None:
                self.store.touch_speaker(best_id)
                return Identification(
                    speaker=speaker,
                    score=best_score,
                    runner_up=runner_up,
                    method="voice",
                )
        return Identification(
            speaker=None, score=best_score, runner_up=runner_up, is_new=True, method="voice"
        )

    def _is_different_person(
        self, speaker: Speaker, pcm: bytes | None, sample_rate: int
    ) -> bool:
        """เสียงที่ได้ยินตอนนี้ขัดกับลายเสียงของคนที่ชื่อนี้หรือเปล่า

        ตอบ ``True`` เฉพาะเมื่อมีหลักฐานชัดว่าเป็นคนละคน คือมีทั้งเสียงตัวอย่าง
        และลายเสียงเดิมให้เทียบ แล้วความคล้ายต่ำกว่าเกณฑ์อย่างมีนัย
        """
        if pcm is None or not self.enabled:
            return False
        stored = self.store.voiceprint_for(speaker.id, self.backend_name)
        if stored is None:
            return False
        try:
            assert self.embedder is not None
            current = self.embedder.embed(pcm, sample_rate)
        except Exception:
            return False
        # เผื่อระยะห่างจากเกณฑ์ไว้ ไม่ตัดสินว่าคนละคนจากความต่างเพียงเล็กน้อย
        return cosine_similarity(current, stored) < (self.threshold - self.margin)

    def enroll(
        self,
        speaker: Speaker,
        pcm: bytes,
        sample_rate: int,
    ) -> bool:
        """ผูกเสียงนี้เข้ากับคนคนนี้ (เรียกซ้ำได้ ระบบจะเฉลี่ยสะสมให้แม่นขึ้น)"""
        if self.embedder is None:
            return False
        try:
            vec = self.embedder.embed(pcm, sample_rate)
            self.store.save_voiceprint(speaker.id, vec, self.backend_name)
            return True
        except Exception:
            return False

    def resolve(
        self,
        pcm: bytes | None,
        sample_rate: int,
        transcript: str,
        *,
        expecting_name: bool = False,
    ) -> Identification:
        """ระบุตัวตนโดยใช้ทั้งเสียงและเนื้อความ

        ลำดับการตัดสิน:

        1. ถ้าจำเสียงได้ชัดเจน -> ใช้คนนั้น
        2. ถ้าในประโยคมีการบอกชื่อ -> ใช้ชื่อนั้น (สร้างใหม่ถ้ายังไม่มี) และ
           ผูกลายเสียงเข้ากับชื่อนั้นทันที เพื่อครั้งหน้าจะจำได้เอง
        3. ถ้าไม่เข้าเงื่อนไขไหนเลย -> คืนผลว่าเป็นคนใหม่ ให้ชั้นบนถามชื่อต่อ
        """
        ident = Identification(speaker=None, is_new=True, method="none")
        if pcm is not None and self.enabled:
            ident = self.identify(pcm, sample_rate)
            if ident.confident:
                return ident

        claimed = extract_name_claim(transcript, expecting_name=expecting_name)
        if claimed:
            speaker = self.store.find_speaker_by_name(claimed)
            if speaker is not None and self._is_different_person(speaker, pcm, sample_rate):
                # ชื่อซ้ำกันแต่เสียงไม่ใช่คนเดิม — "สมชาย" มีได้หลายคน
                # ถ้ายุบเป็นคนเดียวกัน คนที่สองจะได้อ่านความจำของคนแรก
                # และลายเสียงของทั้งคู่จะถูกเฉลี่ยรวมกันจนจำใครไม่ได้เลย
                from .thai_text import detect_particle, particle_for_gender

                gender = detect_particle(transcript)
                speaker = self.store.create_speaker(
                    claimed,
                    gender=gender,
                    particle=particle_for_gender(gender),
                    allow_duplicate_name=True,
                )
                if pcm is not None and self.enabled:
                    self.enroll(speaker, pcm, sample_rate)
                return Identification(speaker=speaker, is_new=True, method="name")
            if speaker is None:
                from .thai_text import detect_particle, particle_for_gender

                gender = detect_particle(transcript)
                # ต้องใช้ get_or_create ไม่ใช่ create ตรง ๆ ไม่งั้นคำขอที่แข่งกัน
                # สร้างชื่อเดียวกันพร้อมกันจะชน UNIQUE index แล้วกลายเป็น 500
                speaker, created = self.store.get_or_create_speaker(
                    claimed,
                    gender=gender,
                    particle=particle_for_gender(gender),
                )
            else:
                created = False
                self.store.touch_speaker(speaker.id)
            if pcm is not None and self.enabled:
                self.enroll(speaker, pcm, sample_rate)
            return Identification(
                speaker=speaker,
                score=ident.score,
                is_new=created,
                method="name",
            )

        return ident
