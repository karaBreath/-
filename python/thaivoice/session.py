"""ตัวประสานงานหนึ่งบทสนทนา — รู้ว่าใครพูด แล้วคิด พูดตอบ และจำ

ลำดับการทำงานของหนึ่งเทิร์น:

1. **ระบุตัวผู้พูด** จากลายเสียง ถ้าไม่รู้จักก็ดูว่าเขาบอกชื่อมาไหม
2. **ตรวจคำสั่งเกี่ยวกับความจำ** เช่น "ลืมทุกอย่างเกี่ยวกับฉัน" — จัดการตรง ๆ
   ไม่ผ่านโมเดล เพราะการลบข้อมูลส่วนบุคคลต้องเชื่อถือได้ 100 เปอร์เซ็นต์
   และต้องผ่านการยืนยันหนึ่งครั้งเสมอ
3. **สตรีมคำตอบ** จาก Claude พร้อมความจำของคนคนนั้น
4. **พูดออกมาทีละประโยค** ทันทีที่ประโยคจบ ไม่รอคำตอบครบก้อน
5. **บันทึกบทสนทนา** และสั่งสกัดความจำเบื้องหลัง
"""

from __future__ import annotations

import logging
import re
import uuid
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

from .brain import ThaiBrain
from .config import Settings, get_settings
from .extraction import MemoryExtractor
from .memory import MemoryStore, Speaker, Turn
from .speaker import Identification, SpeakerIdentifier
from .thai_text import detect_particle, particle_for_gender
from .tts import Speech, TextToSpeech

log = logging.getLogger("thaivoice.session")

# คำขอลบความจำต้องยืนยันภายในเวลานี้ ไม่งั้นถือว่าเลิกล้ม
# ป้องกันคำว่า "ยืนยัน" ที่พูดขึ้นมาลอย ๆ อีกครึ่งชั่วโมงต่อมาไปลบข้อมูลทิ้ง
FORGET_CONFIRM_TIMEOUT = 120.0

__all__ = [
    "ConversationSession",
    "SessionEvent",
    "ExchangeResult",
    "detect_forget_all",
    "is_affirmative",
]

# ── ตรวจจับคำสั่งลบความจำ ───────────────────────────────────────────────────
#
# การลบข้อมูลส่วนบุคคลผิดพลาดเสียหายกว่าการไม่ลบมาก จึงต้องเข้มงวดสองชั้น:
# ชั้นแรกคือรูปประโยคต้องเป็นคำสั่งลบ "ความจำ" จริง ๆ ไม่ใช่แค่มีคำว่า ลืม/ลบ
# ชั้นที่สองคือต้องยืนยันอีกครั้งก่อนลบจริง (ดู ConversationSession)
#
# ของเดิมจับแค่คำว่า "ลืมฉัน" ทำให้ประโยคอย่าง "อย่าลืมฉันนะ" (คำบอกรักทั่วไป)
# และ "ไม่ต้องลบความจำนะ" (สั่งห้ามลบ!) ลบความจำทั้งหมดทันที
_SELF = r"(?:ฉัน|ผม|หนู|เรา|ดิฉัน|กระผม)"
_FORGET_VERB = r"(?:ลืม|ลบ|ล้าง|เคลียร์)"
_MEMORY_OBJECT = (
    r"(?:ความจำ"
    rf"|ข้อมูล(?:ของ)?{_SELF}"
    rf"|(?:ทุกอย่าง|ทุกเรื่อง)(?:ที่รู้)?เกี่ยวกับ{_SELF}"
    rf"|ที่รู้เกี่ยวกับ{_SELF}"
    r"|ประวัติการ(?:คุย|สนทนา)"
    r")"
)
_FORGET_INTENT = re.compile(_FORGET_VERB + r".{0,10}?" + _MEMORY_OBJECT)
_FORGET_EN = re.compile(
    r"\b(?:forget everything|delete my (?:data|memory)|erase my (?:data|memory))\b",
    re.IGNORECASE,
)
# คำห้าม/คำปฏิเสธที่อยู่ "ก่อน" คำกริยา แปลว่าผู้ใช้สั่งไม่ให้ลบ
_NEGATOR_BEFORE = re.compile(r"(?:อย่าเพิ่ง|อย่า|ไม่ต้อง|ห้าม|ยังไม่)\s*$")
# ประโยคคำถามไม่ใช่คำสั่ง — "ลบข้อมูลยังไง" คือขอวิธี ไม่ใช่ขอให้ลบ
_QUESTION_MARKER = re.compile(
    r"(?:ยังไง|ทำไง|ยังงัย|อย่างไร|วิธี|ไหม|มั้ย|หรือเปล่า|เหรอ|ทำไม|ไง\b)"
)
# "ผมลืม..." คือผู้พูดลืมเอง ไม่ใช่สั่งให้ระบบลืม
_FIRST_PERSON_FORGOT = re.compile(_SELF + r"\s*(?:ก็|เลย)?\s*ลืม")

# บอทเพิ่งถามชื่อไปหรือเปล่า — ตรวจจากข้อความที่ *เราเอง* พูด จึงเชื่อถือได้
# ใช้เพื่อยอมรับคำตอบสั้น ๆ อย่าง "เดช" ว่าเป็นชื่อ ซึ่งเป็นการเดาที่ปลอดภัย
# เพราะเรารู้ว่าเราเพิ่งถามอะไรไป
# ตั้งใจให้แคบมาก — ครอบเฉพาะสำนวนที่เราสั่งให้บอทใช้ถามชื่อคู่สนทนาเท่านั้น
# ของเดิมจับคำว่า "ชื่ออะไร" ลอย ๆ ซึ่งติดกับคำถามอย่าง "หมาคุณชื่ออะไรคะ" หรือ
# "ร้านนั้นชื่ออะไรคะ" ด้วย พอติดแล้วคำตอบสั้น ๆ ของผู้ใช้จะถูกตีความเป็นชื่อเขาเอง
#
# รอบก่อนยังหลวมอยู่: ทุกแพตเทิร์นถูก search แบบไม่ยึดขอบ ทำให้
# "ขอชื่อร้านหน่อยครับ" "หมาคุณชื่ออะไรคะ" "แนะนำตัวเลือกที่ดีที่สุด"
# "แนะนำตัวแทนจำหน่าย" "แล้วเพลงนั้นเรียกว่าอะไรคะ" ติดหมด พอติดแล้ว
# คำตอบถัดไปของผู้ใช้ ("ครัวคุณต๋อย") จะกลายเป็น *ชื่อของเขาเอง* — ตัวตนถูก
# สลับ ความจำเดิมกำพร้า และลายเสียงถูกผูกกับตัวตนปลอม
#
# กติกาที่ใช้ตอนนี้
#   * ขอบซ้าย: ห้ามมีตัวอักษรไทย/ละตินนำหน้า (กัน "หมาคุณชื่ออะไร")
#   * ขอบขวา: ต้องตามด้วยคำขอ/คำลงท้าย/จบประโยค (กัน "ขอชื่อร้าน")
#
# ขอบซ้ายแบบ "ห้ามมีตัวอักษรไทยนำหน้า" เพียงอย่างเดียวแน่นเกินไป — ภาษาไทย
# ไม่เว้นวรรคระหว่างคำ กฎจึงยิงใส่ประโยคปกติแทบทุกประโยค
# "แล้วเรียกว่าอะไรดีครับ" "รบกวนขอชื่อหน่อยครับ" "ช่วยแนะนำตัวหน่อยสิครับ"
# หลุดหมด พอหลุดแล้วผู้ใช้จะไม่ถูกสร้างตัวตนเลย ไม่มีความจำ ไม่มีลายเสียง
# และบอทจะถามชื่อซ้ำไปเรื่อย ๆ ทุกเทิร์น
#
# จึงยอมให้มีคำเชื่อม/คำขอร้องนำหน้าได้ตามรายการนี้ ซึ่งไม่มีคำไหนเป็นคำนาม
# ที่ "เป็นเจ้าของชื่อ" ได้เลย ("หมา" "เพลง" "ลูก" "บริษัท" จึงยังถูกกันอยู่)
_NAME_Q_PREFIXES = (
    "แล้ว", "รบกวน", "ช่วย", "ขอ", "ว่า", "ก็", "จะ", "งั้น", "ทีนี้",
    "เดี๋ยว", "แต่", "และ", "ผม", "ดิฉัน", "ฉัน", "หนู", "เรา", "อ้อ", "เอ",
)
_NAME_Q_LEFT = (
    "(?:(?<![ก-๛A-Za-z])"
    + "".join(f"|(?<={prefix})" for prefix in _NAME_Q_PREFIXES)
    + ")"
)
# ขอบขวาต้องครอบ *ทุก* ทางเลือก ไม่ใช่ทางเลือกสุดท้ายทางเดียว
# (เคยเขียนต่อท้ายก้อน alternation ซึ่ง Python ผูกให้กับทางเลือกสุดท้ายเท่านั้น
# "ผมยังไม่ทราบชื่อร้านนั้นครับ" จึงยังติดอยู่)
_NAME_Q_END = (
    r"(?=[\s\?\.!,]|ครับ|ค่ะ|คะ|นะ|ได้ไหม|ได้มั้ย|หรือ|เลย|สิ|ที|ด้วย|$)"
)
_NAME_Q_ALTERNATIVES = (
    # ระบุสรรพนามชัดเจน — ปลอดภัยพอโดยไม่ต้องยึดขอบซ้าย
    r"เรียก(?:คุณ|ผม|ฉัน|หนู|เธอ)ว่าอะไร(?:ดี)?",
    # "เรียกว่าอะไรดี" ลอย ๆ ต้องขึ้นต้นวรรค ไม่งั้นเป็นการถามชื่อของสิ่งอื่น
    _NAME_Q_LEFT + r"เรียกว่าอะไร(?:ดี)?",
    # "คุณชื่ออะไร" ต้องขึ้นต้นวรรค หรือตามหลัง "ว่า"
    # ("ยังไม่ได้ถามเลยว่าคุณชื่ออะไร") ไม่งั้นจะไปติด "หมาคุณชื่ออะไรคะ"
    r"(?:(?<=ว่า)|" + _NAME_Q_LEFT + r")คุณชื่ออะไร",
    # ยอมให้มีสรรพนามบุรุษที่หนึ่งนำหน้าได้ ("ผมยังไม่ทราบชื่อคุณเลยครับ")
    _NAME_Q_LEFT + r"(?:ผม|ดิฉัน|ฉัน|หนู|เรา)?ยังไม่(?:ทราบ|รู้จัก)ชื่อ(?:คุณ|ท่าน|เธอ)?",
    # "ขอชื่อ/ขอทราบชื่อ" ต้องตามด้วยคำขอทันที ห้ามมีคำนามคั่น
    _NAME_Q_LEFT
    + r"ขอ(?:ทราบ)?ชื่อ(?:คุณ|ท่าน|เธอ)?(?:หน่อย|ด้วย|ได้ไหม|ได้มั้ย|สักหน่อย)",
    # "แนะนำตัว" ต้องไม่ต่อด้วยคำอื่น (แนะนำตัวเลือก/ตัวแทน/ตัวเอง)
    _NAME_Q_LEFT + r"แนะนำตัว(?:สัก)?(?:กัน)?(?:หน่อย|เลย|ให้ฟัง|ให้หน่อย)?",
)
_ASKED_FOR_NAME = re.compile(
    "(?:" + "|".join(_NAME_Q_ALTERNATIVES) + ")" + _NAME_Q_END
)

# หมายเหตุ: ห้ามใช้ \b กับคำไทย เพราะคำไทยจำนวนมากลงท้ายด้วยวรรณยุกต์ (เช่น "ใช่"
# "ไม่") ซึ่งเป็นอักขระผสมที่ Python ไม่นับเป็น word character ทำให้ \b ไม่ match
# จึงใช้การยึดต้นและท้ายข้อความแทน
#
# คำ "ตอบรับ" ต้องตรงทั้งประโยค ไม่ใช่แค่ขึ้นต้น เพราะมันไปสั่งลบข้อมูลที่เอากลับ
# ไม่ได้ ถ้าจับแค่ต้นประโยค คำสั่งอย่าง "ลบข้อความล่าสุด" หรือ "ใช่ไหมว่าต้องทำ
# แบบนี้" จะกลายเป็นการยืนยันลบความจำทั้งหมด
_AFFIRMATIVE_TAIL = r"(?:ครับผม|ครับ|ค่ะ|คะ|เลย|นะ|แล้ว|แหละ|สิ|ล่ะ|จ้า|\.|!)*"
_AFFIRMATIVE = re.compile(
    # ตั้งใจ *ไม่* รับ "ครับ"/"ค่ะ" เดี่ยว ๆ แม้จะเป็นวิธีตอบรับที่คนไทยใช้บ่อยที่สุด
    # เพราะมันเป็นคำรับคำทั่วไปที่พูดแทรกได้ตลอด และระบบถอดเสียงก็มักแถมคำลงท้าย
    # ห้อยท้ายมาเอง ด่านนี้เป็นด่านสุดท้ายก่อนลบข้อมูลถาวร จึงต้องขอคำที่ชัดเจน
    # ซึ่งเป็นคำเดียวกับที่บอทบอกให้พูด
    r"^\s*(?:ใช่|ยืนยัน|ลบเลย|ลบได้เลย|ลบ|เอาเลย|เอาสิ|ตกลง|โอเค|โอเก|จัดไป|"
    r"แน่ใจ|ได้เลย|ทำเลย|จัดการเลย|ลุยเลย|"
    r"yes|yeah|yep|y|ok|okay|confirm|sure)"
    r"\s*" + _AFFIRMATIVE_TAIL + r"\s*$",
    re.IGNORECASE,
)

# คำ "ปฏิเสธ" จับแค่ขึ้นต้นก็พอ เพราะการเข้าใจผิดว่าปฏิเสธแปลว่าไม่ลบ
# ซึ่งเป็นทางที่ปลอดภัยกว่าเสมอ
_NEGATIVE = re.compile(
    r"^\s*(?:ไม่|อย่า|ยกเลิก|พอ|หยุด|เดี๋ยวก่อน|ยัง|แป๊บ|ขอคิด|ไว้ก่อน|"
    r"เปลี่ยนใจ|ทีหลัง|ช้าก่อน)"
    r"|^\s*(?:no|nope|cancel|stop|wait)\b",
    re.IGNORECASE,
)


def detect_forget_all(text: str) -> bool:
    """ตรวจว่าผู้ใช้ *สั่ง* ให้ลบความจำทั้งหมดหรือไม่

    ตั้งใจให้เข้มงวด — พลาดดีกว่าลบข้อมูลของคนอื่นทิ้งโดยไม่ได้ตั้งใจ

    >>> detect_forget_all("ลืมทุกอย่างเกี่ยวกับฉันหน่อย")
    True
    >>> detect_forget_all("อย่าลืมฉันนะ")
    False
    >>> detect_forget_all("ไม่ต้องลบความจำนะ")
    False
    >>> detect_forget_all("ลบข้อมูลในมือถือยังไง")
    False
    """
    if not text:
        return False
    if _QUESTION_MARKER.search(text):
        return False
    if _FIRST_PERSON_FORGOT.search(text):
        return False

    match = _FORGET_INTENT.search(text) or _FORGET_EN.search(text)
    if match is None:
        return False
    if _NEGATOR_BEFORE.search(text[: match.start()]):
        return False
    return True


# คำนำหน้าที่แปลว่าชื่อนั้นมีคำเรียกอยู่แล้ว ไม่ต้องเติม "คุณ" ซ้ำ
# ตั้งใจไม่ใส่ "อา" "น้า" "ป้า" — มันไปกินต้นชื่อจริง ("อาทิตย์" "อารีย์" "อาร์ม"
# "น้ำ" "ป้อม") แล้วบอทจะเรียกชื่อเปล่า ๆ ไม่มี "คุณ" ซึ่งฟังห้วนมากจากเสียงบริการ
# speaker.py กันคำสั้นพวกนี้ออกจาก _NAME_TITLES ด้วยเหตุผลเดียวกัน
_HAS_TITLE = ("คุณ", "พี่", "น้อง", "นาย", "นาง", "นางสาว", "ลุง")


def _address(name: str) -> str:
    """เติม "คุณ" นำหน้าชื่อ เว้นแต่ชื่อนั้นมีคำเรียกอยู่แล้ว (กัน "คุณพี่เดช")"""
    return name if name.startswith(_HAS_TITLE) else f"คุณ{name}"


def _pending_summary(stats: "SpeakerStats") -> str:
    """สรุปว่ากำลังจะลบอะไร — ต้องไม่นับศูนย์พร้อมลักษณนาม

    ข้อความนี้ถูกลืมไว้ตอนเขียน _removed_summary ใหม่ ผลคือสองประโยคที่อยู่
    ติดกันขัดกันเอง: ตอนถามบอกว่า "สิ่งที่จำไว้ ศูนย์ เรื่อง" แล้วตอนตอบบอกว่า
    "ไม่มีอะไรให้ลบอยู่แล้ว"
    """
    parts: list[str] = []
    if stats.facts > 0:
        amount = "เรื่องเดียว" if stats.facts == 1 else f" {stats.facts} เรื่อง"
        parts.append(f"สิ่งที่จำไว้{amount}")
    if stats.turns > 0:
        amount = "ข้อความเดียว" if stats.turns == 1 else f" {stats.turns} ข้อความ"
        parts.append(f"บทสนทนา{amount}")
    if not parts:
        return "ทั้งหมด"
    if len(parts) == 1:
        return f"ทั้งหมด ทั้ง{parts[0]} และบทสรุปที่เคยสรุปไว้"
    return f"ทั้งหมด ทั้ง{parts[0]} ทั้ง{parts[1]} และบทสรุปที่เคยสรุปไว้"


def _removed_summary(removed: dict[str, int], particle: str = "ค่ะ") -> str:
    """สรุปว่าลบอะไรไปบ้าง โดยไม่พูดถึงของที่ไม่มี

    ไม่มีคนไทยคนไหนนับศูนย์พร้อมลักษณนาม ("สิ่งที่จำไว้ศูนย์เรื่อง")
    และของที่มีอย่างเดียวก็พูดว่า "เรื่องเดียว" ไม่ใช่ "หนึ่งเรื่อง"

    เมื่อไม่มีอะไรให้ลบ ต้องไม่ขึ้นต้นว่า "ลบให้เรียบร้อยแล้ว" แล้วค่อยบอกว่า
    ไม่มีอะไร ซึ่งอ่านเหมือนแก้คำพูดตัวเอง
    """
    parts: list[str] = []
    for count, noun, classifier in (
        (removed.get("facts", 0), "สิ่งที่จำไว้", "เรื่อง"),
        (removed.get("summaries", 0), "บทสรุป", "ชุด"),
        (removed.get("turns", 0), "บทสนทนาเก่า", "ข้อความ"),
    ):
        if count <= 0:
            continue
        amount = f"{classifier}เดียว" if count == 1 else f" {count} {classifier}"
        parts.append(f"{noun}{amount}")
    if not parts:
        return f"ไม่มีอะไรให้ลบอยู่แล้ว{particle}"
    # สามรายการขึ้นไปต้องมี "ทั้ง" นำทุกตัว ไม่งั้นตัวกลางลอยไม่มีคำเชื่อม
    if len(parts) == 1:
        listed = parts[0]
    elif len(parts) == 2:
        listed = f"ทั้ง{parts[0]} และ{parts[1]}"
    else:
        listed = "ทั้ง" + " ทั้ง".join(parts[:-1]) + " และ" + parts[-1]
    return f"ลบ{listed}เรียบร้อยแล้ว{particle}"


def _soft(particle: str) -> str:
    """รูปของคำลงท้ายเมื่อตามหลัง "นะ" — ภาษาไทยใช้ "นะคะ" ไม่ใช่ "นะค่ะ" """
    return "คะ" if particle == "ค่ะ" else particle


def is_affirmative(text: str) -> bool | None:
    """ตอบรับ (``True``) ตอบปฏิเสธ (``False``) หรือไม่ใช่ทั้งสองอย่าง (``None``)"""
    if not text:
        return None
    if _NEGATIVE.search(text):
        return False
    if _AFFIRMATIVE.search(text):
        return True
    return None


@dataclass
class SessionEvent:
    """เหตุการณ์ระหว่างหนึ่งเทิร์น

    * ``speaker``  — ระบุตัวผู้พูดได้แล้ว
    * ``delta``    — ข้อความที่โมเดลกำลังพิมพ์ (สำหรับแสดงบนจอ)
    * ``chunk``    — ประโยคที่พร้อมพูด พร้อมเสียงใน ``speech`` (ถ้ามี)
    * ``done``     — จบเทิร์น ``text`` คือคำตอบเต็ม
    """

    type: str
    text: str = ""
    speaker: Speaker | None = None
    speech: Speech | None = None
    identification: Identification | None = None


@dataclass
class ExchangeResult:
    transcript: str
    reply: str
    speaker: Speaker | None = None
    identification: Identification | None = None
    chunks: list[str] = field(default_factory=list)


class ConversationSession:
    """หนึ่งบทสนทนา — ใช้ซ้ำได้หลายเทิร์นและหลายคนสลับกันพูด

    ``sticky_speaker`` บอกว่าเมื่อระบุตัวผู้พูดไม่ได้ (เช่นมีแต่ข้อความ ไม่มีเสียง)
    ควรถือว่ายังเป็นคนเดิมของบทสนทนานี้หรือไม่

    * ``True``  — เหมาะกับแอปที่มีผู้ใช้คนเดียวต่อหนึ่ง session เช่นโหมด CLI
      หรือหน้าเว็บที่เปิดในเครื่องของเจ้าตัว
    * ``False`` — **จำเป็นสำหรับเซิร์ฟเวอร์ที่ใช้ session ร่วมกัน** ไม่งั้นคำขอของ
      คนที่สองซึ่งไม่ได้ระบุตัวตนจะสวมรอยเป็นคนแรก และจะได้เห็นความจำของคนแรก
      ทั้งหมด รวมถึงลบความจำของคนแรกได้ด้วย
    """

    def __init__(
        self,
        store: MemoryStore,
        brain: ThaiBrain,
        identifier: SpeakerIdentifier,
        extractor: MemoryExtractor | None = None,
        tts: TextToSpeech | None = None,
        settings: Settings | None = None,
        session_id: str | None = None,
        sticky_speaker: bool = True,
    ) -> None:
        self.store = store
        self.brain = brain
        self.identifier = identifier
        self.extractor = extractor
        self.tts = tts
        self.settings = settings or get_settings()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.sticky_speaker = sticky_speaker
        self.current_speaker: Speaker | None = None
        # id ของคนที่ขอลบความจำและกำลังรอการยืนยัน พร้อมเวลาที่ขอ
        # คำขอลบที่รอยืนยันอยู่ — ต้องเก็บ *ต่อคน* ไม่ใช่ช่องเดียว
        #
        # หนึ่ง session รับได้หลายคน (คุยกันหลายคนหน้าไมค์ตัวเดียว) ของเดิมใช้
        # ช่องเดียว คำขอของคนที่สองจึงทับของคนแรกทิ้งเงียบ ๆ คนแรกที่ถูกบอกให้
        # พูดว่า "ยืนยัน" พูดตามแล้วไม่มีอะไรเกิดขึ้น โมเดลตอบเองว่าจัดการให้แล้ว
        self._pending_forget: dict[int, float] = {}
        # คนที่ถูกเตือนไปแล้วครั้งหนึ่งว่าให้พูดว่า "ยืนยัน"
        # เตือนซ้ำไม่จบทำให้คุยเรื่องอื่นไม่ได้เลย
        self._forget_nudged: set[int] = set()
        # เทิร์นก่อนหน้าบอทถามชื่อไปหรือเปล่า
        self._asked_for_name = False

    # ── ระบุตัวผู้พูด ───────────────────────────────────────────────────
    def identify(
        self, transcript: str, pcm: bytes | None, sample_rate: int | None = None
    ) -> Identification:
        """หาว่าใครพูดประโยคนี้ และอัปเดตคำลงท้ายที่คนนั้นใช้"""
        rate = sample_rate or self.settings.sample_rate
        ident = self.identifier.resolve(
            pcm, rate, transcript, expecting_name=self._asked_for_name
        )

        if ident.speaker is None and pcm is None and self.sticky_speaker:
            # ไม่มีเสียงให้เทียบ (เช่นพิมพ์เข้ามา) — ถือว่ายังเป็นคนเดิมในบทสนทนานี้
            # ทำได้เฉพาะเมื่อ session นี้เป็นของคนเดียวเท่านั้น
            remembered = self._live_current_speaker()
            if remembered is not None:
                ident = Identification(speaker=remembered, method="fallback", score=0.0)

        speaker = ident.speaker
        if speaker is not None:
            # ผู้ใช้เผยเพศจากคำลงท้ายเมื่อไหร่ ก็บันทึกไว้ใช้เรียกขานให้ถูก
            gender = detect_particle(transcript)
            if gender and not speaker.gender:
                updated = self.store.update_speaker(
                    speaker.id, gender=gender, particle=particle_for_gender(gender)
                )
                speaker = updated or speaker
                ident.speaker = speaker
            self.current_speaker = speaker

            # เจอเสียงตรงกับคนเดิม -> เสริมลายเสียงให้แม่นขึ้นเรื่อย ๆ
            if pcm and ident.method == "voice" and ident.confident:
                self.identifier.enroll(speaker, pcm, rate)
        return ident

    def _live_current_speaker(self) -> Speaker | None:
        """คนปัจจุบันที่ยัง *มีอยู่จริง* ในฐานข้อมูล

        ถ้าคนคนนั้นถูกลบไปแล้ว (เช่นผู้ใช้กดลบตัวเองแล้วคุยต่อ) การใช้ค่าเก่าค้าง
        จะทำให้บันทึกบทสนทนาชนกับ foreign key แล้วโยน error ออกไปเป็น 500
        """
        if self.current_speaker is None:
            return None
        if not self.store.speaker_exists(self.current_speaker.id):
            log.info("ผู้สนทนา %s ถูกลบไปแล้ว ล้างสถานะ", self.current_speaker.id)
            self._pending_forget.pop(self.current_speaker.id, None)
            self._forget_nudged.discard(self.current_speaker.id)
            self.current_speaker = None
        return self.current_speaker

    def register_speaker(
        self,
        name: str,
        pcm: bytes | None = None,
        gender: str | None = None,
        sample_rate: int | None = None,
    ) -> Speaker:
        """ลงทะเบียนคนใหม่ด้วยตนเอง แล้วผูกลายเสียงถ้ามีเสียงตัวอย่าง"""
        speaker, _created = self.store.get_or_create_speaker(
            name, gender=gender, particle=particle_for_gender(gender)
        )
        if pcm:
            self.identifier.enroll(speaker, pcm, sample_rate or self.settings.sample_rate)
        self.current_speaker = speaker
        return speaker

    # ── หนึ่งเทิร์น ─────────────────────────────────────────────────────
    def stream_exchange(
        self,
        transcript: str,
        pcm: bytes | None = None,
        speaker: Speaker | None = None,
        speak: bool = True,
        sample_rate: int | None = None,
    ) -> Iterator[SessionEvent]:
        """ประมวลผลหนึ่งเทิร์นแบบสตรีม"""
        transcript = transcript.strip()
        if not transcript:
            return

        if speaker is not None:
            self.current_speaker = speaker
            ident = Identification(speaker=speaker, method="fallback")
        else:
            ident = self.identify(transcript, pcm, sample_rate)
            speaker = ident.speaker

        yield SessionEvent("speaker", speaker=speaker, identification=ident)

        # คำสั่งลบความจำ — ทำเองไม่ผ่านโมเดล เพื่อให้มั่นใจว่าลบจริง
        handled = self._handle_memory_command(transcript, speaker, speak)
        if handled is not None:
            yield from handled
            return

        # ต้องอ่านประวัติ *ก่อน* บันทึกเทิร์นล่าสุด ไม่งั้นข้อความเดียวกันจะถูกส่ง
        # ให้โมเดลสองครั้ง (ครั้งหนึ่งจากประวัติ อีกครั้งจากข้อความปัจจุบัน)
        history: Sequence[Turn] = (
            self.store.recent_turns(speaker.id, limit=self.settings.history_turns)
            if speaker is not None
            else []
        )
        if speaker is not None:
            self.store.record_turn(speaker.id, self.session_id, "user", transcript)

        reply = ""
        for event in self.brain.stream(
            transcript,
            speaker,
            voice_enabled=self.identifier.enabled,
            history=history,
        ):
            if event.type == "delta":
                yield SessionEvent("delta", text=event.text, speaker=speaker)
            elif event.type == "chunk":
                yield SessionEvent(
                    "chunk",
                    text=event.text,
                    speaker=speaker,
                    speech=self._speak(event.text) if speak else None,
                )
            elif event.type == "done":
                reply = event.text

        self._note_assistant_reply(reply)
        if speaker is not None and reply:
            self.store.record_turn(speaker.id, self.session_id, "assistant", reply)
            self._remember(speaker)

        yield SessionEvent("done", text=reply, speaker=speaker, identification=ident)

    def exchange(
        self,
        transcript: str,
        pcm: bytes | None = None,
        speaker: Speaker | None = None,
        speak: bool = True,
        on_event: Callable[[SessionEvent], None] | None = None,
        sample_rate: int | None = None,
    ) -> ExchangeResult:
        """ประมวลผลหนึ่งเทิร์นแบบรอผลลัพธ์ครบ"""
        result = ExchangeResult(transcript=transcript, reply="")
        for event in self.stream_exchange(
            transcript, pcm, speaker, speak, sample_rate=sample_rate
        ):
            if on_event:
                on_event(event)
            if event.type == "speaker":
                result.speaker = event.speaker
                result.identification = event.identification
            elif event.type == "chunk":
                result.chunks.append(event.text)
            elif event.type == "done":
                result.reply = event.text
                result.speaker = event.speaker
        return result

    # ── คำสั่งเกี่ยวกับความจำ ───────────────────────────────────────────
    def _handle_memory_command(
        self, transcript: str, speaker: Speaker | None, speak: bool
    ) -> Iterator[SessionEvent] | None:
        """จัดการคำสั่งลบความจำ คืน ``None`` ถ้าไม่ใช่คำสั่งประเภทนี้"""
        particle = self.settings.assistant_particle

        # เก็บกวาดคำขอที่หมดเวลาไปแล้วของทุกคน
        now = time.time()
        self._pending_forget = {
            person: asked
            for person, asked in self._pending_forget.items()
            if now - asked <= FORGET_CONFIRM_TIMEOUT
        }

        # คนที่กำลังพูดมีคำขอค้างอยู่ไหม — คำขอของคนอื่นต้องไม่ถูกแตะ
        if speaker is not None and speaker.id in self._pending_forget:
            answer = is_affirmative(transcript)
            if answer is True:
                del self._pending_forget[speaker.id]
                self._forget_nudged.discard(speaker.id)
                removed = self.store.forget_everything(speaker.id)
                log.info("ลบความจำของ speaker %s: %s", speaker.id, removed)
                return self._say(
                    f"{_removed_summary(removed, particle)} "
                    f"เริ่มรู้จักกันใหม่ได้เลย{particle}",
                    speaker,
                    speak,
                    user_text=transcript,
                )
            if answer is False:
                del self._pending_forget[speaker.id]
                self._forget_nudged.discard(speaker.id)
                return self._say(
                    f"ได้{particle} งั้นไม่ลบนะ{_soft(particle)} ความจำทั้งหมดยังอยู่ครบ{particle}",
                    speaker,
                    speak,
                    user_text=transcript,
                )
            # ตอบเป็นอย่างอื่น -> ยังไม่ลบ และบอกให้ชัดว่าไม่ได้ลบอะไร
            # ของเดิมเคลียร์ _pending_forget ทิ้ง *ก่อน* จะบอกให้ผู้ใช้พูดว่า
            # "ยืนยัน" ผู้ใช้พูดตามแล้วไม่มีอะไรเกิดขึ้น โมเดลตอบเองว่า
            # "จัดการให้แล้วครับ" ทั้งที่ข้อมูลยังอยู่ครบ — คำสั่งลบที่ผู้ใช้
            # เชื่อว่าทำไปแล้วแต่ไม่ได้ทำ เป็นบั๊กที่ยอมไม่ได้
            # จึงคงสถานะรอไว้และรีเซ็ตนาฬิกาให้ตอบทันเสมอ — แต่เตือนได้ครั้งเดียว
            #
            # ถ้าเตือนทุกครั้งที่ตอบไม่ชัด ผู้ใช้ที่เปลี่ยนเรื่องไปแล้วจะคุยเรื่องอื่น
            # ไม่ได้เลย ทุกประโยคจะถูกตอบด้วยข้อความเดิมไปตลอด
            if speaker.id in self._forget_nudged:
                del self._pending_forget[speaker.id]
                self._forget_nudged.discard(speaker.id)
                return None
            self._forget_nudged.add(speaker.id)
            self._pending_forget[speaker.id] = time.time()
            return self._say(
                f'ยังไม่ได้ลบอะไรนะ{_soft(particle)} ถ้าจะลบจริง ๆ พูดว่า "ยืนยัน" '
                f"ได้เลย{particle}",
                speaker,
                speak,
                user_text=transcript,
            )

        return self._new_forget_request(transcript, speaker, speak)

    def _new_forget_request(
        self, transcript: str, speaker: Speaker | None, speak: bool
    ) -> Iterator[SessionEvent] | None:
        """รับคำขอลบความจำอันใหม่ คืน ``None`` ถ้าประโยคนี้ไม่ใช่คำขอลบ"""
        particle = self.settings.assistant_particle
        if not detect_forget_all(transcript):
            return None

        if speaker is None:
            # ตอบตามจริง ดีกว่ารับปากแล้วไม่ได้ลบอะไรเลย
            return self._say(
                f"ตอนนี้ยังไม่รู้ว่าคุยอยู่กับใคร เลยยังไม่มีความจำอะไรให้ลบ{particle}",
                None,
                speak,
                user_text=transcript,
            )

        self._pending_forget[speaker.id] = time.time()
        self._forget_nudged.discard(speaker.id)
        stats = self.store.stats(speaker.id)
        return self._say(
            f"ขอยืนยันก่อน{particle} จะลบความจำเกี่ยวกับ{_address(speaker.call_name)}"
            f"{_pending_summary(stats)} "
            f"ลบแล้วกู้คืนไม่ได้ "
            f"ส่วนเสียงที่ใช้จำว่าเป็นคุณจะยังอยู่ ถ้าอยากลบด้วยต้องลบทั้งบัญชี "
            f'ถ้าแน่ใจ พูดว่า "ยืนยัน" ได้เลย{particle}',
            speaker,
            speak,
            user_text=transcript,
        )

    def _say(
        self,
        text: str,
        speaker: Speaker | None,
        speak: bool,
        user_text: str | None = None,
    ) -> Iterator[SessionEvent]:
        """ตอบข้อความที่ระบบเขียนเอง (ไม่ผ่านโมเดล) และบันทึกลงบทสนทนา

        ``user_text`` คือประโยคที่ทำให้เกิดคำตอบนี้ ต้องบันทึกด้วย ไม่งั้น
        บทสนทนาจะมีแต่คำตอบลอย ๆ ("ลบให้เรียบร้อยแล้วค่ะ") โดยไม่มีคำขอ
        และเทิร์นถัดไปที่ส่งให้โมเดลจะเห็น assistant ติดกันหลายอันรวด
        """

        def generate() -> Iterator[SessionEvent]:
            self._note_assistant_reply(text)
            if speaker is not None and self.store.speaker_exists(speaker.id):
                if user_text and user_text.strip():
                    self.store.record_turn(
                        speaker.id, self.session_id, "user", user_text
                    )
                self.store.record_turn(speaker.id, self.session_id, "assistant", text)
            yield SessionEvent(
                "chunk",
                text=text,
                speaker=speaker,
                speech=self._speak(text) if speak else None,
            )
            yield SessionEvent("done", text=text, speaker=speaker)

        return generate()

    def _note_assistant_reply(self, reply: str) -> None:
        """จำไว้ว่าเทิร์นนี้บอทถามชื่อหรือเปล่า เพื่อใช้ตีความคำตอบเทิร์นถัดไป"""
        self._asked_for_name = bool(reply) and bool(_ASKED_FOR_NAME.search(reply))

    def _speak(self, text: str) -> Speech | None:
        if self.tts is None or not text.strip():
            return None
        try:
            return self.tts.synthesize(text)
        except Exception:
            log.warning("สังเคราะห์เสียงไม่สำเร็จ", exc_info=True)
            return None

    def _remember(self, speaker: Speaker) -> None:
        if self.extractor is None:
            return
        turns = self.store.recent_turns(speaker.id, limit=6)
        self.extractor.schedule(speaker, turns)
        self.extractor.maybe_summarize(speaker)
