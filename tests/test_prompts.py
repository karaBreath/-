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
