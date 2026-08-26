"""ทดสอบชั้นความจำ — ความถูกต้องตรงนี้คือหัวใจของ "จดจำผู้สนทนาได้" """

import pytest

from thaivoice.memory import MemoryStore


class TestSpeakers:
    def test_สร้างและอ่านผู้สนทนา(self, store: MemoryStore):
        speaker = store.create_speaker("เดชา", nickname="เดช", gender="male", particle="ครับ")
        assert speaker.id > 0
        assert speaker.call_name == "เดช"  # ใช้ชื่อเล่นถ้ามี

        loaded = store.get_speaker(speaker.id)
        assert loaded is not None
        assert loaded.display_name == "เดชา"

    def test_ชื่อที่ใช้เรียกถอยไปใช้ชื่อจริงเมื่อไม่มีชื่อเล่น(self, store: MemoryStore):
        speaker = store.create_speaker("สมชาย")
        assert speaker.call_name == "สมชาย"

    def test_หาคนจากชื่อไม่สนช่องว่างและตัวพิมพ์(self, store: MemoryStore):
        store.create_speaker("Nan", nickname="แนน")
        assert store.find_speaker_by_name("  nan  ") is not None
        assert store.find_speaker_by_name("แนน") is not None
        assert store.find_speaker_by_name("ไม่มีคนนี้") is None

    def test_อัปเดตเฉพาะคอลัมน์ที่อนุญาต(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        updated = store.update_speaker(speaker.id, gender="male", particle="ครับ")
        assert updated is not None and updated.gender == "male"

        # คีย์แปลกปลอมต้องถูกเมิน ไม่ใช่ทำให้ SQL พัง
        safe = store.update_speaker(speaker.id, **{"id": 999, "evil": "DROP TABLE"})
        assert safe is not None and safe.id == speaker.id

    def test_ลบคนแล้วความจำหายตามทั้งหมด(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        store.upsert_fact(speaker.id, "อาชีพ", "หมอ")
        store.record_turn(speaker.id, "s1", "user", "สวัสดี")

        assert store.delete_speaker(speaker.id) is True
        assert store.get_speaker(speaker.id) is None
        assert store.facts_for(speaker.id) == []
        assert store.turn_count(speaker.id) == 0


class TestFacts:
    def test_คีย์เดิมถูกทับด้วยค่าใหม่(self, store: MemoryStore):
        """ถ้าผู้ใช้ย้ายบ้าน ค่าเก่าต้องหายไป ไม่ใช่เก็บสองค่าที่ขัดกัน"""
        speaker = store.create_speaker("เดช")
        store.upsert_fact(speaker.id, "เมืองที่อยู่", "กรุงเทพ")
        store.upsert_fact(speaker.id, "เมืองที่อยู่", "เชียงใหม่")

        facts = store.facts_for(speaker.id)
        assert len(facts) == 1
        assert facts[0].value == "เชียงใหม่"

    def test_เรียงตามความมั่นใจ(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        store.upsert_fact(speaker.id, "ไม่ค่อยแน่", "ก", confidence=0.2)
        store.upsert_fact(speaker.id, "แน่มาก", "ข", confidence=0.95)
        assert store.facts_for(speaker.id)[0].key == "แน่มาก"

    def test_ข้ามค่าว่าง(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        store.upsert_fact(speaker.id, "", "ค่า")
        store.upsert_fact(speaker.id, "คีย์", "   ")
        assert store.facts_for(speaker.id) == []

    def test_ลืมทีละข้อและลืมทั้งหมด(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        store.upsert_fact(speaker.id, "ก", "1")
        store.upsert_fact(speaker.id, "ข", "2")

        assert store.forget_fact(speaker.id, "ก") is True
        assert store.forget_fact(speaker.id, "ไม่มีคีย์นี้") is False
        assert store.forget_all_facts(speaker.id) == 1
        assert store.facts_for(speaker.id) == []


class TestVoiceprints:
    def test_เฉลี่ยลายเสียงสะสม(self, store: MemoryStore):
        """ยิ่งคุยบ่อย ลายเสียงยิ่งควรนิ่งขึ้น — จึงเฉลี่ยแบบถ่วงน้ำหนัก"""
        speaker = store.create_speaker("เดช")
        store.save_voiceprint(speaker.id, [0.0, 0.0, 0.0], "fake")
        store.save_voiceprint(speaker.id, [1.0, 1.0, 1.0], "fake")

        prints = store.all_voiceprints("fake")
        assert len(prints) == 1
        assert prints[0][1] == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)

    def test_มิติไม่ตรงกันต้องแจ้งเตือน(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        store.save_voiceprint(speaker.id, [0.1, 0.2], "fake")
        with pytest.raises(ValueError, match="มิติ"):
            store.save_voiceprint(speaker.id, [0.1, 0.2, 0.3], "fake")

    def test_แยกกันตาม_backend(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        store.save_voiceprint(speaker.id, [0.1], "fake")
        assert store.has_voiceprint(speaker.id, "fake") is True
        assert store.has_voiceprint(speaker.id, "อื่น") is False
        assert store.all_voiceprints("อื่น") == []

    def test_เวกเตอร์ว่างต้องปฏิเสธ(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        with pytest.raises(ValueError):
            store.save_voiceprint(speaker.id, [], "fake")


class TestTurns:
    def test_บทสนทนาเรียงจากเก่าไปใหม่(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        for i in range(5):
            store.record_turn(speaker.id, "s1", "user", f"ข้อความที่ {i}")

        turns = store.recent_turns(speaker.id, limit=3)
        assert [t.content for t in turns] == ["ข้อความที่ 2", "ข้อความที่ 3", "ข้อความที่ 4"]

    def test_บทบาทต้องถูกต้อง(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        with pytest.raises(ValueError, match="role"):
            store.record_turn(speaker.id, "s1", "system", "ไม่ควรผ่าน")

    def test_แยกความจำระหว่างคน(self, store: MemoryStore):
        """คนละคนต้องไม่เห็นบทสนทนาของกันและกัน"""
        a = store.create_speaker("เอ")
        b = store.create_speaker("บี")
        store.record_turn(a.id, "s1", "user", "ความลับของเอ")

        assert store.recent_turns(b.id) == []
        assert store.turn_count(a.id) == 1


class TestSummaries:
    def test_อ่านบทสรุปล่าสุด(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        store.save_summary(speaker.id, "สรุปเก่า", 10)
        store.save_summary(speaker.id, "สรุปใหม่", 20)

        latest = store.latest_summary(speaker.id)
        assert latest == ("สรุปใหม่", 20)

    def test_ยังไม่มีบทสรุป(self, store: MemoryStore):
        speaker = store.create_speaker("เดช")
        assert store.latest_summary(speaker.id) is None


def test_สถิติผู้สนทนา(store: MemoryStore):
    speaker = store.create_speaker("เดช")
    store.record_turn(speaker.id, "s1", "user", "หนึ่ง")
    store.record_turn(speaker.id, "s1", "assistant", "สอง")
    store.upsert_fact(speaker.id, "ก", "1")

    stats = store.stats(speaker.id)
    assert stats.turns == 2
    assert stats.facts == 1
    assert stats.first_seen is not None
