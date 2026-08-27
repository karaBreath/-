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
# เพดานต้องเผื่อคำลงท้ายที่ติดมาด้วย ไม่งั้นชื่อยาว 12-15 ตัวอักษรจะถูกตัด
# กลางคำลงท้าย ("กิตติศักดิ์นะคร") ซึ่งตัวตัดคำลงท้ายจำไม่ได้อีกแล้ว
# แล้วเศษที่เหลืออ่านออกเสียงไม่ได้เลย ตัวกรองความยาวจริงอยู่ที่ _clean_name
_NAME_CHARS = r"[ก-๙A-Za-z][ก-๙A-Za-z0-9.'-]{0,24}"

# ธงของแต่ละกฎติดมากับตัวกฎเอง ไม่ใช่เก็บเป็นชุดดัชนีแยก
#
# ของเดิมเก็บเป็น frozenset ของดัชนี ซึ่งพังเงียบ ๆ สองรอบติดกันเมื่อมีคนเพิ่ม
# หรือถอดกฎแล้วลืมปรับดัชนีตาม ครั้งหนึ่งทำให้ชื่อเล่นอย่าง "ลูกปัด" ถูกปฏิเสธ
# ทั้งหมด อีกครั้งทำให้ประโยคธรรมดากลายเป็นตัวตนใหม่ การผูกธงไว้กับกฎปิด
# ความผิดพลาดประเภทนี้ถาวร
@dataclass(frozen=True)
class _NameRule:
    pattern: re.Pattern[str]
    # ผ่อนการกันคำกำกวมได้ไหม — ได้เฉพาะกฎที่มีสรรพนามหรือคำว่า "ชื่อเล่น"
    # ยืนยันว่าเป็นการแนะนำตัวจริง ("ดิฉันชื่อลูกปัดค่ะ" เป็นชื่อคนแน่นอน)
    # ส่วน "เรียกว่า" ลอย ๆ เป็นสำนวนอธิบายที่ใช้บ่อยที่สุดสำนวนหนึ่งในภาษาไทย
    # ("โรคนี้เรียกว่าโรคเบาหวานครับ" ไม่ใช่การบอกชื่อตัวเอง) จึงต้องเข้ม
    confirmed: bool = False


_NAME_RULES = [
    # "ชื่อเล่นชื่อบิ๊กครับ" / "หนูชื่อเล่นว่าบิ๊ก" — ต้องมาก่อนกฎทั่วไป
    #
    # ต้องมีขอบซ้ายเหมือนกฎอื่นทุกข้อ ไม่งั้น "เพื่อนผมชื่อเล่นว่าสมชายครับ"
    # และ "แมวผมชื่อเล่นว่าเหมียวครับ" กลายเป็นชื่อของ *ผู้พูด*
    _NameRule(
        re.compile(rf"(?:^|\s)(?:{_PRONOUN})?\s*ชื่อเล่น(?:ชื่อ|ว่า|คือ)?\s*({_NAME_CHARS})"),
        confirmed=True,
    ),
    # "ผมชื่อสมชายครับ" / "หนูชื่อว่าแนนค่ะ"
    # ลำดับใน (?:...) สำคัญ — ต้องลอง "จริงว่า" ก่อน "ว่า" ไม่งั้น "ชื่อจริงว่าอนุชา"
    # จะจับได้เป็น "จริงว่าอนุชา" แล้วถูกปฏิเสธเพราะขึ้นต้นด้วย "จริง"
    _NameRule(
        re.compile(rf"(?:^|\s){_PRONOUN}\s*ชื่อ(?:จริงว่า|ว่า)?\s*({_NAME_CHARS})"),
        confirmed=True,
    ),
    # "ชื่อเดชครับ" / "ชื่อผมสมชายครับ" — ต้องอยู่ต้นวรรคเท่านั้น
    _NameRule(
        re.compile(rf"(?:^|\s)ชื่อ(?:จริงว่า|ว่า)?\s*(?:{_PRONOUN})?\s*({_NAME_CHARS})")
    ),
    # "เรียกผมว่าอาทิตย์ก็ได้ครับ" — ต้องมาก่อนกฎ "เรียก...ว่า" ทั่วไป และต้อง
    # ไม่โลภ ไม่งั้นจะกิน "ก็ได้ครับ" เข้ามาด้วยจนได้ชื่อ "อาทิตย์ก็ได้ครั"
    #
    # สรรพนามคือสิ่งที่ยืนยันว่าเป็นการแนะนำตัว ไม่ใช่การอธิบายหรือการสั่ง
    _NameRule(
        re.compile(rf"เรียก(?:ผม|ฉัน|หนู|เรา|ดิฉัน)ว่า\s*({_NAME_CHARS}?)\s*ก็ได้"),
        confirmed=True,
    ),
    _NameRule(
        re.compile(rf"เรียก(?:ผม|ฉัน|หนู|เรา|ดิฉัน)ว่า\s*({_NAME_CHARS})"),
        confirmed=True,
    ),
    _NameRule(re.compile(rf"เรียกว่า\s*({_NAME_CHARS}?)\s*ก็ได้")),
    _NameRule(re.compile(rf"เรียกว่า\s*({_NAME_CHARS})")),
    #
    # เคยมีกฎ "เรียก X ก็ได้" (ไม่มีสรรพนาม ไม่มี "ว่า") ตรงนี้ ซึ่งถูก *ถอดออก*
    # โดยตั้งใจ
    #
    # "เรียก" เป็นคำกริยาธรรมดาที่ใช้บ่อยมาก กฎนี้จึงกลืนประโยคทั่วไปเป็นชื่อคน
    # นับไม่ถ้วน: "เรียกช่างก็ได้ครับ" -> ชื่อ "ช่าง", "เรียกแท็กซี่ก็ได้" ->
    # "แท็กซี่", "เรียกประชุมก็ได้" -> "ประชุม", "เรียกเก็บเงินก็ได้" ->
    # "เก็บเงิน", "เรียกรถ/ลิฟต์/แม่/เพื่อน..." ทุกอันสร้างตัวตนปลอมที่ลายเสียง
    # ของผู้ใช้ถูกผูกเข้าไปถาวร
    #
    # เคยพยายามกันด้วยการปฏิเสธชื่อที่มีพยางค์ "มา"/"ไป" (สำนวน "เรียก X มา")
    # ซึ่งล้มเหลว: แยก "เรียกช่างมา" ออกจาก "เรียกอุมา" ไม่ได้ถ้าไม่มี
    # พจนานุกรมชื่อคน ผลคือชื่อไทยจริงกว่า 25 ชื่อถูกปฏิเสธ (อุมา ปัทมา
    # ชุติมา สุมาลี สมหมาย ศุภมาส วิมาลา ...) แล้วทางถอย expecting_name
    # ก็ไปแปลทั้งประโยคเป็นชื่อ "เรียกสมหมาย" ต่ออีกที — แย่กว่าเดิมสองชั้น
    #
    # รูปที่มีสรรพนามหรือ "ว่า" คั่น ("เรียกหนูว่าลูกปัดก็ได้ค่ะ") ยังใช้ได้
    # ตามปกติ และตอนที่บอทเพิ่งถามชื่อไปหมาด ๆ ทางถอย expecting_name รับ
    # คำตอบสั้น ๆ อยู่แล้ว ซึ่งเป็นเส้นทางที่ผู้ใช้ใหม่เดินจริง
    #
    # "ผมคือสมชายครับ"
    _NameRule(
        re.compile(rf"(?:^|\s){_PRONOUN}\s*คือ\s*({_NAME_CHARS})"), confirmed=True
    ),
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
    _NameRule(
        re.compile(
            r"(?:\b[Mm]y name is\b|\b[Ii]\s*'?m\b|\b[Ii] am\b|\b[Cc]all me\b)\s+"
            r"([A-Z][A-Za-z.'-]{1,19})"
        ),
        confirmed=True,
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
# รวมคำนำหน้าทางวิชาชีพที่ลงท้ายด้วยจุด ("ดร.สมชาย" -> "สมชาย")
# ถ้าปล่อยติดมา บอทจะเรียกว่า "คุณดร.สมชาย" ซึ่งผิดหลักภาษา และ TTS
# ก็อ่าน "ดร." ทีละตัวอักษร
# เรียงยาวไปสั้น เพราะตัวตัดหยุดที่ตัวแรกที่ตรง
# ยศตำรวจ/ทหารกับคำนำหน้าอาชีพต้องอยู่ด้วย ไม่งั้นบอทจะเรียกว่า "คุณร.ต.อ"
# ซึ่งผิดหลักภาษา (คำนำหน้าแทนที่ "คุณ" ไม่ใช่ซ้อนใต้มัน) และ TTS ก็อ่าน
# ตัวย่อทีละตัวอักษร
_NAME_TITLES = (
    "พล.ต.ท.", "พล.ต.ต.", "พล.ต.อ.", "พล.อ.", "พล.ท.", "พล.ต.",
    "ร.ต.อ.", "ร.ต.ท.", "ร.ต.ต.", "พ.ต.อ.", "พ.ต.ท.", "พ.ต.ต.",
    "ผศ.ดร.", "รศ.ดร.", "ศ.ดร.", "ผศ.", "รศ.", "ศ.", "ดร.", "นพ.", "พญ.", "ทพ.",
    "อาจารย์", "นางสาว", "นาย", "นาง", "คุณ", "พี่", "น้อง",
)

# อักขระที่ขึ้นต้นคำไทยไม่ได้ (สระตาม วรรณยุกต์ และเครื่องหมายกำกับ)
# ถ้าตัดคำนำหน้าแล้วเหลือขึ้นต้นด้วยตัวพวกนี้ แปลว่าเราตัดกลางคำ ไม่ใช่ตัดคำนำหน้า
# เช่น "คุณากร" ตัด "คุณ" ออกจะเหลือ "ากร" ซึ่งอ่านไม่ได้
_CANNOT_START_WORD = "าิีึืุูัํ็์ฺ่้๊๋ๅๆ"

# ถ้าชื่อที่จับได้ "ขึ้นต้นด้วย" คำพวกนี้ แปลว่าไม่ใช่การแนะนำตัว
# (เช่น "ชื่อเสียง" = reputation, "ชื่อไฟล์" = filename)
# ไม่มีคำไหนในนี้เป็นชื่อเล่นไทยที่ใช้จริง จึงปฏิเสธได้ทันที
_NOT_A_NAME_PREFIXES = (
    # "เรียก" เป็นคำกริยา ไม่ใช่ต้นชื่อ ถ้าไม่กันไว้ ทางถอย expecting_name จะ
    # แปลทั้งประโยค "เรียกสมหมายก็ได้ครับ" เป็นชื่อ "เรียกสมหมาย" แล้วผูกกับ
    # ลายเสียงถาวร
    "เรียก",
    "ชื่อ", "เสียง", "เล่น", "จริง", "ไฟล์", "ผู้", "บริษัท",
    "หนังสือ", "สินค้า", "แบรนด์", "ตัวละคร",
    "อะไร", "นี้", "นั้น", "บัญชี", "โดเมน", "จังหวัด", "โรงเรียน", "เดียว",
    "ตาม", "ที่", "ของ",
)

# คำที่เป็นได้ทั้งชื่อเล่นจริงและคำนำของสิ่งอื่น
#
# "แมว" "เพลง" "ปลา" "ลูกน้ำ" "ต้น" "น้ำ" ล้วนเป็นชื่อเล่นไทยที่พบบ่อยมาก
# การปฏิเสธแบบขึ้นต้นเลยทำให้คนกลุ่มนี้แนะนำตัวไม่ได้เลย
#
# เคยผ่อนด้วยการยอมให้ยาวกว่าคำนำได้ถึงสามตัวอักษร ซึ่งหลวมเกินไปมาก —
# "ชื่อลูกค้า" "ชื่อเพลงนี้" "ชื่อร้านชัย" "ชื่อหมาผม" กลายเป็นชื่อคนหมด
# แล้วลายเสียงของผู้ใช้จะถูกผูกกับตัวตนปลอมนั้นถาวร
# ตอนนี้ต้อง *ตรงกันพอดี* เท่านั้น พร้อมบัญชีชื่อเล่นที่ยาวกว่าคำนำจริง ๆ
_AMBIGUOUS_NAME_PREFIXES = (
    "แมว", "หมา", "เพลง", "ลูก", "ยา", "ร้าน", "ทีม", "วง", "เรื่อง",
    "ถนน", "ซอย", "เมนู", "โรค", "หนัง", "เพื่อน", "เขา", "เธอ",
)
_AMBIGUOUS_NAME_ALLOW = {
    "ลูกน้ำ", "ลูกเกด", "ลูกตาล", "ลูกแก้ว", "ลูกหมี", "ลูกปลา", "ลูกอม",
    "ลูกหว้า", "ลูกหมู", "แมวน้ำ", "เพลงไพเราะ",
}

# คำที่ถ้าไปอยู่ *ท้าย* ชื่อ แปลว่าจับเกินขอบไปโดนวลีขยาย ไม่ใช่ชื่อคนแล้ว
#
# "ชื่อเล่นลูกชายผมครับ" -> "ลูกชายผม" (ลูกชายของผม ไม่ใช่ชื่อเขา)
# "ชื่อเล่นร้านนี้ครับ"  -> "ร้านนี้"  (ร้านนี้ ไม่ใช่ชื่อใคร)
#
# ตรวจที่ท้ายเท่านั้น ไม่ใช่ทั้งคำ เพราะชื่อไทยจริงขึ้นต้นด้วยพยางค์พวกนี้ได้
# ("ผมเดชครับ" ไม่เกี่ยว, "เขมิกา" ขึ้นต้นด้วย "เข" ไม่ใช่ "เขา")
# และต้องยาวกว่าคำนั้นจริง ๆ เพราะ "หนู" "เขา" เดี่ยว ๆ เป็นชื่อเล่นได้
_NOT_A_NAME_ENDINGS = (
    "ผม", "ฉัน", "ดิฉัน", "หนู", "เรา", "เขา", "เธอ", "ท่าน",
    "นี้", "นั้น", "นี่", "นั่น", "โน้น", "ไหน",
)

# ถ้าในชื่อมีคำพวกนี้อยู่ แปลว่าจับเกินไปโดนคำอื่นเข้ามาด้วย
# ระวัง: ห้ามใส่พยางค์ที่พบในชื่อคนไทยจริง ๆ ลงในลิสต์นี้
# เคยใส่ "มา" "จำ" "ไป" ไว้ ทำให้ชื่อธรรมดาอย่าง มาลี มานี สมหมาย จำเนียร ไปรยา
# ถูกปฏิเสธทิ้งหมด ซึ่งแย่กว่าปัญหาที่ตั้งใจกันตั้งแต่แรก
_NAME_REJECT_SUBSTRINGS = (
    "อะไร", "ไหม", "มั้ย", "เหรอ", "ยังไง", "คือ", "ว่า", "แล้ว",
    # ตั้งใจไม่ใส่ "หน่อย" — เป็นชื่อเล่นไทยที่ใช้จริง และคอมเมนต์ข้างบนก็ห้าม
    # ใส่พยางค์ที่พบในชื่อคนจริงไว้แล้ว คำขอที่ลงท้ายด้วย "หน่อย" ถูกกันด้วย
    # _NOT_A_NAME_STARTS และกฎท้ายคำอยู่แล้ว
    "ด้วย", "ไม่",
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
    "แป๊บนึง", "สักครู่", "เดี๋ยวนะ", "ไม่แน่ใจ", "ก็ได้", "แล้วแต่",
    "hello", "hi", "hey", "what", "why", "sure", "thanks", "sorry", "please",
    # อาชีพและสถานที่ที่คนมักตอบแทนชื่อเมื่อเลี่ยงไม่บอก
    "หมอ", "ครู", "อาจารย์", "พยาบาล", "ตำรวจ", "ทหาร", "วิศวกร", "นักเรียน",
    "นักศึกษา", "พนักงาน", "แม่บ้าน", "คนไทย", "กรุงเทพ", "เชียงใหม่", "ภูเก็ต",
    "ขอนแก่น", "ไทย", "คนขับ",
}

# ชื่อจริงที่ลงท้ายด้วย "นะ" — กันไม่ให้ถูกตัดเป็น "มา"
_NAMES_ENDING_IN_NA = {"มานะ", "ปัญญะ", "วีระนะ", "อนงค์นะ"}

# พยัญชนะที่แทบไม่เคยเป็นตัวสะกดของคำที่จบในตัวเอง
#
# ถ้าตัด "นะ" ออกแล้วเหลือคำที่ลงท้ายด้วยตัวพวกนี้ แปลว่าตัดกลางคำแน่นอน
# "วัฒนะ" -> "วัฒ" และ "พัฒนะ" -> "พัฒ" ซึ่งอ่านออกเสียงว่า "วัด" "พัด"
# แล้วบอทจะเรียกเขาว่า "คุณวัฒ" ไปตลอด บัญชีรายชื่อล้วน ๆ ขยายตามไม่ทัน
# (วัฒนะ พัฒนะ ชัยวัฒนะ ล้วนเป็นชื่อที่พบบ่อย) จึงใช้กฎการสะกดแทน
# ตั้งใจไม่ใส่ "ฐ" — "ณัฐ" "รัฐ" เป็นคำที่จบในตัวเองและพบบ่อยกว่า "วัฒนะ" มาก
_RARE_FINAL_CONSONANTS = "ฒฑฌฎฏฆฬ"

# คำที่ขึ้นต้นคำตอบซึ่งบอกชัดว่าไม่ใช่ชื่อ ใช้เฉพาะตอนตีความคำตอบเปล่า ๆ
# (ทุกคำยาว >= 3 ตัวอักษร เพื่อไม่ให้ไปกินต้นชื่อจริง)
_NOT_A_NAME_STARTS = (
    "ทำไม", "เดี๋ยว", "กำลัง", "อยาก", "ต้อง", "ช่วย", "ขอโทษ", "ขอบ",
    "เมื่อ", "ตอนนี้", "วันนี้", "อาจ", "คงจะ", "น่าจะ", "เพราะ", "ตอบ",
    "บอกไม่", "ไม่อยาก", "ไม่บอก", "เอาไว้", "แล้วแต่",
    "หิว", "ง่วง", "เหนื่อย", "ปวด", "เบื่อ", "ไม่ใช่", "ไม่ได้",
    "ลืม", "จำไม่", "ยังไม่", "ขอเวลา", "แค่", "เปล่า",
    "อยู่", "ขับ", "นั่ง", "นอน", "เพิ่ง", "สบาย", "กำลัง", "เดิน", "กิน",
    "ทำงาน", "ประชุม", "ยุ่ง", "ว่าง", "ฟัง", "พูด", "คิด", "ก็ดี", "เฉย",
    # ตั้งใจไม่ใส่ "รอ" — "รอฮีม" "รอมฎอน" เป็นชื่อจริง
)

# คำที่ลงท้ายคำตอบซึ่งบอกว่าเป็นประโยค ไม่ใช่ชื่อ
#
# ตั้งใจไม่ใส่ "นะ" กับ "ดี" เดี่ยว ๆ — มันขัดกับ _NAMES_ENDING_IN_NA ที่อุตส่าห์
# เก็บ "มานะ" ไว้ แล้วมาโยนทิ้งตรงนี้ และยังกิน "ปรีดี" "สมฤดี" ไปด้วย
_NOT_A_NAME_ENDS = ("อยู่", "แล้ว", "มาก", "จัง", "ครับ", "ค่ะ", "ๆ")

# แต่ "ก่อนนะ" "ดีนะ" "นี้นะ" เป็นคำบอกลา/คำอวยพร ไม่ใช่ชื่อแน่นอน
# การถอด "นะ" ออกจากบัญชีข้างบนเปิดให้ทุกคำตอบสั้นที่ลงท้ายแบบนี้กลายเป็นชื่อ
# แล้วผู้ใช้ที่เลี่ยงไม่บอกชื่อจะได้ตัวตนชื่อ "ไว้ก่อนนะ" ติดตัวไปตลอด
_NOT_A_NAME_ENDS_LONG = (
    "ก่อนนะ", "ดีนะ", "นี้นะ", "ครู่นะ", "แล้วนะ", "เลยนะ", "เถอะนะ",
    "ไปนะ", "มานะครับ", "ละนะ", "เนอะ", "จ้ะ", "จ้า",
)

# สรรพนามบุรุษที่หนึ่งที่คนไทยใส่นำหน้าชื่อเวลาตอบคำถาม "ชื่ออะไร"
#
# "ผมเบียร์ครับ" คือคำตอบที่เป็นธรรมชาติที่สุดของคำถามนี้ ถ้าไม่ตัดสรรพนามออก
# ตัวตนของเขาจะชื่อ "ผมเบียร์" ถาวร แล้วบอทเรียกเขาว่า "คุณผมเบียร์"
# เรียงจากยาวไปสั้น เพื่อให้ "กระผม" ถูกตัดก่อน "ผม"
_LEADING_PRONOUNS = ("กระผม", "ข้าพเจ้า", "ดิฉัน", "ผม", "ฉัน", "หนู", "เรา")

# คำกริยาที่นำหน้าชื่อได้เฉพาะตอนตอบคำถาม "ชื่ออะไร" — ดู extract_name_claim
_LEADING_CALL_VERBS = ("เรียกว่า", "เรียก")

_STOPWORD_NAMES = {
    "อะไร", "ไหน", "ใคร", "นี้", "นั้น", "เธอ", "คุณ", "เขา", "มัน", "เรา",
    "ชื่อ", "ตัว", "คน", "งาน", "วันนี้",
}


# สรรพนามบุรุษที่หนึ่งบอกเพศของผู้พูดพอ ๆ กับคำลงท้าย
# "หนูชื่อฝ้าย" ไม่มีคำลงท้ายเลย แต่ "หนู" บอกชัดว่าเป็นผู้หญิง
# extraction.py สอนโมเดลกฎนี้อยู่แล้ว มีแต่เส้นทางกฎที่ยังไม่รู้
_PRONOUN_GENDER = (
    ("กระผม", "male"),
    ("ผม", "male"),
    ("ดิฉัน", "female"),
    ("หนู", "female"),
)


def _gender_from_pronoun(text: str) -> str | None:
    """เดาเพศจากสรรพนามบุรุษที่หนึ่ง คืน ``None`` เมื่อไม่มีเบาะแส"""
    for pronoun, gender in _PRONOUN_GENDER:
        if pronoun in text:
            return gender
    return None


def _clean_name(raw: str, strict_ambiguous: bool = True) -> str | None:
    """ตัดคำลงท้าย/คำนำหน้าออกจากชื่อที่จับได้ แล้วตรวจว่าน่าเชื่อถือไหม

    ``strict_ambiguous`` ปฏิเสธชื่อที่ขึ้นต้นด้วยคำกำกวม (แมว เพลง ลูก ยา วง)
    ใช้เฉพาะกับกฎที่ไม่มีสรรพนามยืนยัน — ไม่งั้น "ดิฉันชื่อลูกปัดค่ะ" ซึ่งเป็น
    การแนะนำตัวชัดเจนจะถูกทิ้ง แล้วผู้ใช้จะไม่ถูกสร้างตัวตนเลย
    ลูกปัด ลูกแพร ลูกพีช เป็นชื่อเล่นผู้หญิงไทยที่พบบ่อยที่สุดกลุ่มหนึ่ง
    """
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
            if suffix.startswith("นะ") and (
                (rest + "นะ") in _NAMES_ENDING_IN_NA
                or (rest and rest[-1] in _RARE_FINAL_CONSONANTS)
            ):
                continue
            if len(rest) >= 2:
                name = rest
                changed = True
                break

    # ตัดคำนำหน้า เช่น "พี่เดช" -> "เดช" (วนซ้ำเผื่อซ้อนกัน "ผศ.ดร.สมชาย")
    for _ in range(3):
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
                # ส่วนที่เหลือขึ้นต้นด้วยเครื่องหมายผสม แปลว่าตัดกลางคำ
                # ห้าม return ตรงนี้ — ด่านตรวจที่เหลือ (ความยาว คำต้องห้าม
                # คำที่ไม่ใช่ชื่อ) จะถูกข้ามไปทั้งหมด
                if "." in title:
                    return None
                break
        else:
            break

    if len(name) < 2 or len(name) > 15:
        return None
    # คำนำหน้าล้วน ๆ ไม่ใช่ชื่อ ("ผมชื่อ ร.ต.อ. สมศักดิ์ ครับ" มีช่องว่างคั่น
    # กฎจึงจับได้แค่ยศ แล้วบอทก็เรียกเขาว่า "คุณร.ต.อ")
    if name in _NAME_TITLES or f"{name}." in _NAME_TITLES:
        return None
    if name in _STOPWORD_NAMES:
        return None
    if any(name.startswith(prefix) for prefix in _NOT_A_NAME_PREFIXES):
        return None
    if (
        strict_ambiguous
        and name not in _AMBIGUOUS_NAME_ALLOW
        and any(
            name.startswith(prefix) and name != prefix
            for prefix in _AMBIGUOUS_NAME_PREFIXES
        )
    ):
        return None
    if any(
        name.endswith(ending) and name != ending for ending in _NOT_A_NAME_ENDINGS
    ):
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
    for rule in _NAME_RULES:
        for match in rule.pattern.finditer(text):
            name = _clean_name(match.group(1), strict_ambiguous=not rule.confirmed)
            if name:
                return name

    if expecting_name:
        stripped = text.strip()
        # คำตอบต้องสั้นและเป็นคำเดียว ไม่งั้นน่าจะเป็นประโยคอื่นที่พูดต่อ
        if len(stripped) <= 24 and len(stripped.split()) <= 2:
            body = stripped.replace(" ", "")
            # "เรียกลูกปัดก็ได้ค่ะ" เป็นคำตอบที่เป็นธรรมชาติมากเมื่อบอทเพิ่งถาม
            # ชื่อไป กฎทั่วไปรับรูปนี้ไม่ได้ (คำว่า "เรียก" กลืนประโยคธรรมดา
            # นับไม่ถ้วน) แต่ตรงนี้เรารู้ว่าเราเพิ่งถามอะไรไป จึงตัดคำกริยาทิ้ง
            # แล้วรับส่วนที่เหลือได้ — ไม่มีใครตอบคำถาม "ชื่ออะไร" ด้วยการสั่ง
            # ให้เรียกช่างมา
            for verb in _LEADING_CALL_VERBS:
                if body.startswith(verb) and len(body) - len(verb) >= 2:
                    body = body[len(verb) :]
                    break
            for pronoun in _LEADING_PRONOUNS:
                if body.startswith(pronoun) and len(body) - len(pronoun) >= 2:
                    # ยอมรับความกำกวม: "หนูดีค่ะ" อาจแปลว่า "ชื่อดี" หรือ "ชื่อหนูดี"
                    # ก็ได้ แต่ "หนู" เป็นสรรพนามบ่อยกว่าเป็นต้นชื่อมาก
                    body = body[len(pronoun) :]
                    break
            # บอทเพิ่งถามชื่อไปหมาด ๆ คำตอบจึงเป็นชื่อแน่นอนพอที่จะไม่ต้องเข้ม
            # กับคำกำกวม ("ลูกปัดค่ะ" คือชื่อ ไม่ใช่ "ลูก" อะไรสักอย่าง)
            candidate = _clean_name(body, strict_ambiguous=False)
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
    if any(ch in candidate for ch in ".,:;/\\()[]{}<>@#$%^&*+=|~`\"!?"):
        return False
    if any(candidate.startswith(bad) for bad in _NOT_A_NAME_STARTS):
        return False
    # ชื่อคนไทยไม่ลงท้ายด้วยคำขยายหรือคำช่วย ("ขับรถอยู่" "ก็ดีนะ" "เฉย ๆ")
    if len(candidate) > 3 and candidate.endswith(_NOT_A_NAME_ENDS):
        return False
    if candidate.endswith(_NOT_A_NAME_ENDS_LONG):
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

    def _match_by_name(
        self, claimed: str, pcm: bytes | None, sample_rate: int
    ) -> tuple[Speaker | None, bool, bool]:
        """เลือกว่า "สมชาย" ที่พูดอยู่ตอนนี้คือสมชายคนไหน

        คืน ``(speaker, fork, sure)``

        * ``speaker`` — คนที่ควรใช้ หรือ ``None`` ถ้ายังไม่มีใครชื่อนี้
        * ``fork`` — มีคนชื่อนี้อยู่ แต่มีหลักฐานว่าเป็นคนละคน ต้องสร้างใหม่
        * ``sure`` — มั่นใจพอจะเอาเสียงนี้ไปสะสมเป็นลายเสียงของเขาไหม

        ของเดิมเทียบกับแถวเดียวที่ ``find_speaker_by_name`` คืนมา และตัดสินว่า
        "คนละคน" จากตัวอย่างเสียงชิ้นเดียวที่ต่ำกว่า ``threshold - margin``
        ผลคือเป็นหวัด เปลี่ยนไมค์ หรืออยู่ในที่เสียงดังครั้งเดียว ก็ถูกแยกเป็น
        ตัวตนใหม่ ความจำหายหมด และแถวใหม่ได้กุญแจแยก จึงหาไม่เจออีกเลย
        แถมยังสะสมแถวซ้ำไปเรื่อย ๆ เพราะไม่เคยเทียบกับแถวที่ชื่อซ้ำกันแถวอื่น
        """
        candidates = self.store.find_speakers_by_name(claimed)
        if not candidates:
            return None, False, True
        if pcm is None or not self.enabled:
            # ไม่มีเสียงให้เทียบ (เช่นคุยผ่านตัวหนังสือ) — ใช้คนที่เพิ่งคุยล่าสุด
            # ยอมรับว่านี่คือช่องโหว่ที่แก้ด้วยข้อมูลเท่านี้ไม่ได้
            return candidates[0], False, True
        try:
            assert self.embedder is not None
            current = self.embedder.embed(pcm, sample_rate)
        except Exception:
            return candidates[0], False, False

        backend = self.backend_name
        best: tuple[float, Speaker] | None = None
        without_print: Speaker | None = None
        for candidate in candidates:
            stored = self.store.voiceprint_for(candidate.id, backend)
            if stored is None:
                if without_print is None:
                    without_print = candidate
                continue
            score = cosine_similarity(current, stored)
            if best is None or score > best[0]:
                best = (score, candidate)

        if best is None:
            # ทุกคนที่ชื่อนี้ยังไม่มีลายเสียง — พิสูจน์ไม่ได้ว่าคนละคน
            return candidates[0], False, True

        score, matched = best
        if score >= (self.threshold - self.margin):
            return matched, False, True
        if without_print is not None:
            # ยังมีคนชื่อนี้ที่ไม่มีลายเสียงให้เทียบ อาจเป็นเขาก็ได้
            return without_print, False, True
        if score >= self.fork_threshold:
            # คาบเส้น — เป็นได้ทั้งคนเดิมที่เสียงเปลี่ยน (เป็นหวัด ไมค์คนละตัว
            # ที่เสียงดัง) และคนใหม่ที่บังเอิญชื่อซ้ำ การแยกตัวตนผิดทำให้ความจำ
            # หายทั้งก้อน จึงเลือกใช้คนเดิมไว้ก่อน แต่ไม่เอาเสียงนี้ไปสะสม
            # เพราะถ้าเป็นคนละคนจริงจะทำให้ลายเสียงเพี้ยนไปทั้งคู่
            return matched, False, False
        return matched, True, True

    @property
    def fork_threshold(self) -> float:
        """ต่ำกว่านี้จึงจะเชื่อว่าเป็นคนละคนที่บังเอิญชื่อซ้ำกัน

        ตั้งห่างจากเกณฑ์จำเสียงมากกว่าเดิมสามเท่า เพราะการแยกตัวตนผิดแพงกว่า
        การรวมผิดมาก — รวมผิดแก้ได้ด้วยการพูดใหม่ แยกผิดคือความจำหายทั้งก้อน
        """
        return self.threshold - 3 * self.margin

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
            from .thai_text import detect_particle, particle_for_gender

            speaker, fork, sure = self._match_by_name(claimed, pcm, sample_rate)
            gender = detect_particle(transcript) or _gender_from_pronoun(transcript)
            # ถ้ายังไม่รู้เพศ ต้องปล่อยว่างไว้ ไม่ใช่เดาเป็นผู้ชาย
            # ของเดิมเก็บ "ครับ" ให้ทุกคนที่ยังไม่ได้ลงท้ายอะไร แล้ว prompt
            # ก็บอกโมเดลว่า "คู่สนทนาลงท้ายว่าครับ" ทั้งที่เขาไม่เคยพูดคำนั้น
            particle = particle_for_gender(gender) if gender else None
            if fork:
                # ชื่อซ้ำกันแต่เสียงไม่ใช่คนเดิม — "สมชาย" มีได้หลายคน
                # ถ้ายุบเป็นคนเดียวกัน คนที่สองจะได้อ่านความจำของคนแรก
                # และลายเสียงของทั้งคู่จะถูกเฉลี่ยรวมกันจนจำใครไม่ได้เลย
                speaker = self.store.create_speaker(
                    claimed,
                    gender=gender,
                    particle=particle,
                    allow_duplicate_name=True,
                )
                created = True
            elif speaker is None:
                # ต้องใช้ get_or_create ไม่ใช่ create ตรง ๆ ไม่งั้นคำขอที่แข่งกัน
                # สร้างชื่อเดียวกันพร้อมกันจะชน UNIQUE index แล้วกลายเป็น 500
                speaker, created = self.store.get_or_create_speaker(
                    claimed,
                    gender=gender,
                    particle=particle,
                )
            else:
                created = False
                self.store.touch_speaker(speaker.id)
            if sure and pcm is not None and self.enabled:
                self.enroll(speaker, pcm, sample_rate)
            return Identification(
                speaker=speaker,
                score=ident.score,
                is_new=created,
                method="name",
            )

        return ident
