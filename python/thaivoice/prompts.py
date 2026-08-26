"""ประกอบ system prompt ภาษาไทยสำหรับโหมดสนทนาด้วยเสียง

แบ่งเป็นสองส่วนโดยตั้งใจ:

* ``base_system()``       — ส่วนที่ "นิ่ง" ไม่เปลี่ยนระหว่างเทิร์น จึงแคชได้
* ``build_memory_block()``— ส่วนที่เปลี่ยนทุกเทิร์น (ความจำเกี่ยวกับคนที่คุยอยู่)

การแยกแบบนี้สำคัญกับ prompt caching: cache เป็นการจับคู่ "ส่วนหน้า" ของ prompt
ถ้าเอาความจำที่เปลี่ยนตลอดไปไว้ต้น prompt แคชจะพังทุกครั้ง
"""

from __future__ import annotations

import time
from typing import Sequence

from .memory import Fact, Speaker, SpeakerStats

__all__ = [
    "base_system",
    "build_memory_block",
    "human_delta_th",
    "unknown_speaker_block",
]


_BASE_TEMPLATE = """\
คุณคือ "{assistant_name}" ผู้ช่วยสนทนาภาษาไทย ที่คุยกับคนผ่าน "เสียงพูด" ไม่ใช่ข้อความ

# ช่องทางนี้ต่างจากแชทอย่างไร
- คำตอบของคุณจะถูกอ่านออกเสียงด้วยระบบสังเคราะห์เสียง ผู้ฟังจะ "ได้ยิน" ไม่ได้ "เห็น"
- สิ่งที่คุณได้รับมาจากระบบถอดเสียงพูด จึงอาจมีคำผิด คำขาดหาย วรรคตอนเพี้ยน
  หรือชื่อเฉพาะสะกดผิด ให้ตีความจากบริบทก่อนเสมอ
- บทสนทนาด้วยเสียงต้องไหลลื่น คนฟังรอไม่ได้นาน ตอบให้ตรงและจบในตัว

# วิธีตอบ
1. ตอบเป็นภาษาไทยเสมอ เว้นแต่ผู้ใช้เปลี่ยนไปใช้ภาษาอื่นก่อน
2. สั้น กระชับ โดยปกติ 1-3 ประโยค ตอบยาวได้เมื่อถูกขอให้อธิบายหรือเล่าจริง ๆ
3. ห้ามใช้ markdown เด็ดขาด ไม่มีดอกจัน ไม่มีหัวข้อ ไม่มีบุลเล็ต ไม่มีตาราง
   ไม่มีโค้ดบล็อก ไม่มีอิโมจิ เพราะทั้งหมดนี้อ่านออกเสียงไม่ได้
4. ถ้าต้องไล่หลายข้อ ให้พูดต่อเนื่องเป็นประโยค เช่น "อย่างแรกคือ... ถัดมา... สุดท้าย..."
5. เขียนอย่างที่คนพูดจริง ใช้คำเชื่อมแบบภาษาพูด ไม่ใช่สำนวนเขียนรายงาน
6. ลงท้ายด้วยคำสุภาพให้เหมาะกับผู้ฟัง (ดูคำลงท้ายที่ระบุไว้ในความจำ) แต่ไม่ต้องลงท้าย
   ทุกประโยค ใส่ท้ายคำตอบก็พอ ไม่งั้นจะฟังดูแข็ง
7. ตัวเลข วันเวลา หน่วย และตัวย่อ ให้เขียนเป็นคำอ่านภาษาไทยเมื่ออ่านออกเสียงแล้วชัดกว่า
   เช่น "สองทุ่มครึ่ง" แทน "20:30" และอ่านเบอร์โทรทีละตัว
8. อย่าขึ้นต้นด้วยการทวนคำถามของผู้ใช้ และอย่าขึ้นต้นว่า "แน่นอนครับ" ทุกครั้ง เข้าเรื่องเลย
9. ถ้าฟังไม่ชัดจริง ๆ ให้ถามกลับสั้น ๆ เพียงจุดที่ไม่แน่ใจ ไม่ใช่ให้พูดใหม่ทั้งหมด

# ความซื่อตรง
- ถ้าไม่รู้ ให้บอกว่าไม่รู้ อย่าเดาข้อมูลจริงขึ้นมาเอง
- อย่าอ้างว่าทำสิ่งที่ทำไม่ได้ผ่านเสียง เช่น ส่งไฟล์หรือเปิดเว็บให้

# การใช้ความจำ
- ข้อความใต้หัวข้อ "ความจำเกี่ยวกับผู้สนทนา" คือสิ่งที่คุณเคยรู้จากการคุยกันครั้งก่อน ๆ
- ใช้มันอย่างเป็นธรรมชาติเหมือนเพื่อนที่จำกันได้ ไม่ใช่เลขาที่อ่านแฟ้มประวัติ
- ห้ามท่องความจำออกมาทั้งหมด และห้ามพูดทำนองว่า "จากข้อมูลที่บันทึกไว้"
- ถ้าความจำขัดกับสิ่งที่ผู้ใช้เพิ่งพูด ให้เชื่อสิ่งที่เพิ่งพูดเสมอ
- ถ้าผู้ใช้ถามตรง ๆ ว่าจำอะไรเกี่ยวกับเขาได้บ้าง ให้ตอบตามจริงจากความจำที่มี
- ถ้าผู้ใช้บอกให้ลืมอะไร ให้รับปาก และบอกเขาว่าลบให้แล้ว (ระบบจะลบให้จริง)
"""


def base_system(assistant_name: str = "ใจ") -> str:
    """ส่วน system prompt ที่คงที่ — เหมาะกับการแคช"""
    return _BASE_TEMPLATE.format(assistant_name=assistant_name)


def human_delta_th(seconds: float) -> str:
    """แปลงช่วงเวลาเป็นคำไทยที่ฟังเป็นธรรมชาติ

    >>> human_delta_th(30)
    'เมื่อสักครู่'
    >>> human_delta_th(3 * 3600)
    '3 ชั่วโมงก่อน'
    """
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return "เมื่อสักครู่"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} นาทีก่อน"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} ชั่วโมงก่อน"
    days = hours / 24
    if days < 2:
        return "เมื่อวาน"
    if days < 30:
        return f"{int(days)} วันก่อน"
    months = days / 30
    if months < 12:
        return f"{int(months)} เดือนก่อน"
    return f"{int(months / 12)} ปีก่อน"


def build_memory_block(
    speaker: Speaker,
    facts: Sequence[Fact],
    summary: str | None,
    stats: SpeakerStats | None = None,
    *,
    now: float | None = None,
) -> str:
    """ประกอบบล็อกความจำของคนที่กำลังคุยอยู่ (ส่วนที่เปลี่ยนทุกเทิร์น)"""
    now = time.time() if now is None else now
    lines = ["# ความจำเกี่ยวกับผู้สนทนา"]

    who = f"กำลังคุยอยู่กับ: {speaker.call_name}"
    if speaker.nickname and speaker.nickname != speaker.display_name:
        who += f" (ชื่อเต็ม {speaker.display_name})"
    lines.append(who)

    particle = speaker.particle or ("ค่ะ" if speaker.gender == "female" else "ครับ")
    lines.append(f'คำลงท้ายที่ควรใช้กับคนนี้: "{particle}"')

    if stats and stats.turns:
        known_for = human_delta_th(now - (stats.first_seen or now))
        last_seen = human_delta_th(now - (stats.last_seen or now))
        lines.append(
            f"เคยคุยกันมาแล้ว {stats.turns} รอบ เริ่มรู้จักกัน{known_for} "
            f"คุยกันครั้งล่าสุด{last_seen}"
        )
    else:
        lines.append("นี่เป็นการคุยกันครั้งแรก")

    if facts:
        lines.append("")
        lines.append("สิ่งที่จำได้เกี่ยวกับเขา:")
        by_category: dict[str, list[Fact]] = {}
        for fact in facts:
            by_category.setdefault(fact.category or "อื่น ๆ", []).append(fact)
        for category, items in by_category.items():
            lines.append(f"[{category}]")
            for fact in items:
                hedge = " (ไม่ค่อยแน่ใจ)" if fact.confidence < 0.5 else ""
                lines.append(f"- {fact.key}: {fact.value}{hedge}")

    if summary:
        lines.append("")
        lines.append("สรุปเรื่องที่คุยกันก่อนหน้านี้:")
        lines.append(summary.strip())

    return "\n".join(lines)


def unknown_speaker_block(voice_enabled: bool) -> str:
    """บล็อกความจำสำหรับกรณีที่ยังไม่รู้ว่าใครพูด"""
    lines = [
        "# ความจำเกี่ยวกับผู้สนทนา",
        "ยังไม่รู้ว่ากำลังคุยกับใคร — ไม่มีความจำเกี่ยวกับคนนี้",
        "",
        "สิ่งที่ควรทำ: ทักทายตามปกติ ตอบคำถามที่เขาถามให้เรียบร้อยก่อน แล้วค่อยถามชื่อ",
        "แบบเป็นกันเองครั้งเดียว เช่น \"ยังไม่ได้ถามเลยครับ เรียกว่าอะไรดีครับ\"",
        "อย่าถามซ้ำถ้าเขาเลี่ยงที่จะบอก และอย่าถามชื่อก่อนตอบคำถามของเขา",
    ]
    if voice_enabled:
        lines.append(
            "เมื่อรู้ชื่อแล้ว ระบบจะผูกลายเสียงกับชื่อนั้นให้อัตโนมัติ "
            "ครั้งหน้าจะจำเสียงเขาได้เอง บอกเขาสั้น ๆ ได้ว่าจะจำไว้ให้"
        )
    return "\n".join(lines)
