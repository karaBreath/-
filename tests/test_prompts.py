"""ทดสอบการประกอบ prompt ภาษาไทย"""

import time

from thaivoice.memory import MemoryStore
from thaivoice.prompts import (
    base_system,
    build_memory_block,
    human_delta_th,
    unknown_speaker_block,
)


class TestBaseSystem:
    def test_มีกฎสำคัญของโหมดเสียงครบ(self):
        prompt = base_system("ใจ")
        assert "ใจ" in prompt
        assert "markdown" in prompt
        assert "ภาษาไทย" in prompt

    def test_ข้อความคงที่ทุกครั้ง(self):
        """ส่วนนี้ต้องไม่มีอะไรเปลี่ยน ไม่งั้น prompt cache จะพังทุกเทิร์น"""
        assert base_system("ใจ") == base_system("ใจ")


class TestMemoryBlock:
    def test_บอกชื่อและคำลงท้าย(self, store: MemoryStore):
        speaker = store.create_speaker("เดชา", nickname="เดช", particle="ครับ")
        block = build_memory_block(speaker, [], None, store.stats(speaker.id))

        assert "เดช" in block
        assert "เดชา" in block
        assert "ครับ" in block

    def test_จัดกลุ่มข้อเท็จจริงตามหมวด(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        store.upsert_fact(speaker.id, "อาชีพ", "หมอ", category="งาน")
        store.upsert_fact(speaker.id, "ชอบกิน", "ส้มตำ", category="ความชอบ")

        block = build_memory_block(speaker, store.facts_for(speaker.id), None)
        assert "[งาน]" in block
        assert "[ความชอบ]" in block

    def test_เตือนเมื่อไม่ค่อยมั่นใจ(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        store.upsert_fact(speaker.id, "อาจจะ", "ไม่แน่", confidence=0.2)

        block = build_memory_block(speaker, store.facts_for(speaker.id), None)
        assert "ไม่ค่อยแน่ใจ" in block

    def test_บอกว่าเป็นการคุยครั้งแรก(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        block = build_memory_block(speaker, [], None, store.stats(speaker.id))
        assert "ครั้งแรก" in block

    def test_ใส่บทสรุปเมื่อมี(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        block = build_memory_block(speaker, [], "คุยเรื่องแผนไปเชียงใหม่")
        assert "เชียงใหม่" in block

    def test_คำลงท้ายเดาจากเพศเมื่อไม่ได้ตั้งไว้(self, store: MemoryStore):
        speaker = store.create_speaker("แนน", gender="female")
        assert 'ค่ะ' in build_memory_block(speaker, [], None)


class TestUnknownSpeaker:
    def test_สั่งให้ถามชื่อหลังตอบคำถามก่อน(self):
        block = unknown_speaker_block(voice_enabled=False)
        assert "ยังไม่รู้ว่ากำลังคุยกับใคร" in block
        assert "ตอบคำถามที่เขาถามให้เรียบร้อยก่อน" in block

    def test_บอกเรื่องการจำเสียงเมื่อเปิดใช้(self):
        assert "ลายเสียง" in unknown_speaker_block(voice_enabled=True)
        assert "ลายเสียง" not in unknown_speaker_block(voice_enabled=False)


class TestHumanDelta:
    def test_ช่วงเวลาต่าง_ๆ(self):
        assert human_delta_th(10) == "เมื่อสักครู่"
        assert human_delta_th(600) == "10 นาทีก่อน"
        assert human_delta_th(3 * 3600) == "3 ชั่วโมงก่อน"
        assert human_delta_th(30 * 3600) == "เมื่อวาน"
        assert human_delta_th(5 * 86400) == "5 วันก่อน"
        assert human_delta_th(90 * 86400) == "3 เดือนก่อน"
        assert human_delta_th(400 * 86400) == "1 ปีก่อน"

    def test_เวลาติดลบไม่พัง(self):
        assert human_delta_th(-100) == "เมื่อสักครู่"


class Testถ้อยคำในบล็อกความจำ:
    def test_ต้องไม่มีคำว่าเมื่อซ้อนกัน(self, store):
        """human_delta_th คืนค่าที่ขึ้นต้นด้วย "เมื่อ" อยู่แล้วในบางกรณี

        การเติมซ้ำได้ "เริ่มรู้จักกันเมื่อ เมื่อสักครู่" ซึ่งอ่านออกเสียงแล้วสะดุด
        และเป็นตัวอย่างภาษาที่ไม่ดีให้โมเดลเลียนแบบ
        """
        speaker = store.create_speaker("เดช")
        store.record_turn(speaker.id, "s", "user", "สวัสดี")

        block = build_memory_block(speaker, [], None, store.stats(speaker.id))

        assert "เมื่อ เมื่อ" not in block
        assert "เริ่มรู้จักกันเมื่อสักครู่" in block

    def test_ช่วงเวลาที่ไม่มีเมื่อนำหน้าต้องได้เมื่อ(self, store):
        import time

        speaker = store.create_speaker("เดช")
        store.record_turn(speaker.id, "s", "user", "สวัสดี")
        store._conn.execute(
            "UPDATE turns SET created_at = ?", (time.time() - 3 * 3600,)
        )
        store._conn.commit()

        block = build_memory_block(speaker, [], None, store.stats(speaker.id))

        assert "เริ่มรู้จักกันเมื่อ 3 ชั่วโมงก่อน" in block


class Testกฎภาษาไทยใน_system_prompt:
    """prompt คือที่ที่สอนโมเดลว่าจะพูดไทยยังไง ภาษาผิดตรงนี้ผิดไปทุกที่"""

    def test_ต้องสอนกฎนะคะไม่ใช่นะค่ะ(self):
        """เป็นจุดที่ผิดบ่อยที่สุดในภาษาไทย และ TTS อ่านความต่างออกมาตรง ๆ"""
        prompt = base_system("ใจ", "ค่ะ")
        assert "นะคะ" in prompt
        assert "นะค่ะ" in prompt, "ต้องยกตัวอย่างสิ่งที่ห้ามเขียนด้วย"
        assert 'ห้ามเขียน "นะค่ะ"' in prompt

    def test_ต้องบอกสรรพนามของผู้ช่วยให้ตรงเพศกับคำลงท้าย(self):
        """ของเดิมบอกแต่คำลงท้าย โมเดลจึงพูด "ผมจะช่วยดูให้ค่ะ" เป็นประจำ"""
        assert '"เรา"' in base_system("ใจ", "ค่ะ")
        assert '"ผม"' in base_system("ใจ", "ครับ")

    def test_คำลงท้ายผู้ชายต้องไม่ถูกสอนกฎของผู้หญิง(self):
        prompt = base_system("ใจ", "ครับ")
        # ยกเว้นบรรทัดที่ยกตัวอย่าง "ผิด" ให้ดู ซึ่งต้องมีทั้งสองเพศ
        body = "\n".join(
            line for line in prompt.splitlines() if "ตรงข้าม" not in line
        )
        assert "คะ" not in body.replace("ครับ", "")
