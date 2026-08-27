"""ทดสอบชั้นความจำ — ความถูกต้องตรงนี้คือหัวใจของ "จดจำผู้สนทนาได้" """

import sqlite3

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
        store.save_voiceprint(speaker.id, [0.2, 0.2, 0.2], "fake")
        store.save_voiceprint(speaker.id, [0.8, 0.8, 0.8], "fake")

        prints = store.all_voiceprints("fake")
        assert len(prints) == 1
        assert prints[0][1] == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)

    @pytest.mark.parametrize(
        "broken",
        [
            [float("nan")] * 3,
            [float("inf"), 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )
    def test_ปฏิเสธลายเสียงที่เสีย(self, store: MemoryStore, broken):
        """ลายเสียง NaN หนึ่งอันเคยทำให้ระบบจำเสียงใครไม่ได้เลยทั้งระบบ

        เพราะการเทียบกับ NaN ให้ผลเป็นเท็จทุกกรณี และค่าเฉลี่ยสะสมของคนคนนั้น
        จะกลายเป็น NaN ถาวร เกิดได้จริงเมื่อส่งไฟล์เสียง 0 เฟรมหรือเสียงเงียบสนิท
        """
        speaker = store.create_speaker("เดช")
        with pytest.raises(ValueError):
            store.save_voiceprint(speaker.id, broken, "fake")
        assert store.all_voiceprints("fake") == []

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


class Testการอัปเกรดฐานข้อมูลเก่า:
    """schema เวอร์ชัน 1 ไม่มี UNIQUE index จึงมีชื่อซ้ำกันได้จริง

    ของเดิมสร้าง index ทับโดยไม่แยกกุญแจก่อน ฐานข้อมูลจึงเปิดไม่ได้อีกเลย
    แล้วยัง leak connection ที่ค้าง transaction ไว้ ทำให้เปิดครั้งต่อไปเจอ
    "database is locked"
    """

    def _v1(self, path):
        conn = sqlite3.connect(str(path))
        conn.executescript(
            """
            CREATE TABLE speakers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              display_name TEXT NOT NULL, nickname TEXT, gender TEXT,
              particle TEXT, language TEXT DEFAULT 'th', notes TEXT,
              meta_json TEXT DEFAULT '{}', created_at REAL, last_seen_at REAL);
            """
        )
        for name in ("สมชาย", "  สมชาย  ", "สมชาย", "นก"):
            conn.execute(
                "INSERT INTO speakers (display_name, created_at, last_seen_at)"
                " VALUES (?, 1, 1)",
                (name,),
            )
        conn.commit()
        conn.close()

    def test_ชื่อซ้ำจากเวอร์ชัน1ต้องเปิดได้และหาเจอครบ(self, tmp_path):
        path = tmp_path / "v1.db"
        self._v1(path)

        store = MemoryStore(path)
        try:
            assert store.find_speaker_by_name("สมชาย") is not None
            assert store.find_speaker_by_name("นก") is not None
            assert len(store.find_speakers_by_name("สมชาย")) == 3
            keys = [
                row["name_key"]
                for row in store._conn.execute(
                    "SELECT name_key FROM speakers"
                ).fetchall()
            ]
            assert all(k for k in keys), "ทุกแถวต้องมีกุญแจ ไม่ใช่ NULL ค้างไว้"
            assert len(set(keys)) == len(keys)
        finally:
            store.close()

    def test_เปิดซ้ำต้องไม่พังและไม่ล็อก(self, tmp_path):
        path = tmp_path / "v1.db"
        self._v1(path)
        for _ in range(3):
            store = MemoryStore(path)
            assert store.find_speaker_by_name("นก") is not None
            store.close()


class Testกุญแจของคนชื่อซ้ำต้องพิมพ์ไม่ได้:
    def test_ชื่อที่มีเครื่องหมายชาร์ปไม่ไปโดนตัวตนคนอื่น(self, store):
        first = store.create_speaker("สมชาย")
        second = store.create_speaker("สมชาย", allow_duplicate_name=True)
        store.upsert_fact(second.id, "โรคประจำตัว", "เบาหวาน")

        assert first.id != second.id
        # ของเดิมกุญแจเป็น "สมชาย#2" ซึ่งผู้ใช้พิมพ์เข้ามาได้ แล้วได้ตัวตน
        # (พร้อมความจำเรื่องโรคประจำตัว) ของคนอื่นไปเลย
        assert store.find_speaker_by_name("สมชาย#2") is None
        found, created = store.get_or_create_speaker("สมชาย#2")
        assert created is True
        assert found.id not in (first.id, second.id)

    def test_เปลี่ยนชื่อไปชนกุญแจคนอื่นต้องไม่ระเบิด(self, store):
        a = store.create_speaker("สมชาย")
        b = store.create_speaker("มาลี")

        updated = store.update_speaker(b.id, display_name="สมชาย")

        assert updated is not None
        assert updated.display_name == "สมชาย"
        assert store.get_speaker(a.id) is not None
        assert {s.id for s in store.find_speakers_by_name("สมชาย")} == {a.id, b.id}


class Testชื่อว่างตอนเปลี่ยนชื่อ:
    """create_speaker กันไว้แล้ว แต่ update_speaker ไม่กัน

    ผลคือได้แถวที่ name_key ว่าง ซึ่ง partial index ไม่คุม หาไม่เจอด้วยชื่อ
    อีกเลย และบอทจะเรียกเขาว่า "คุณ" เฉย ๆ
    """

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_ต้องปฏิเสธชื่อว่าง(self, store, bad):
        speaker = store.create_speaker("เดช")
        with pytest.raises(ValueError):
            store.update_speaker(speaker.id, display_name=bad)
        assert store.find_speaker_by_name("เดช") is not None

    def test_เปลี่ยนชื่อปกติยังทำได้(self, store):
        speaker = store.create_speaker("เดช")
        updated = store.update_speaker(speaker.id, display_name="  สมชาย  ")
        assert updated is not None and updated.display_name == "สมชาย"
        assert store.find_speaker_by_name("สมชาย").id == speaker.id


class Testตารางรุ่นความจำ:
    """โตไม่หยุดได้ถ้าเลื่อนรุ่นทุกครั้งที่มีคำขอลบ แม้ id นั้นจะไม่มีอยู่จริง

    DELETE /api/speakers/{id} ไม่ต้องยืนยันตัวตน จึงเป็นตัวป้อนที่ดี
    """

    def test_ลบ_id_ที่ไม่มีอยู่ต้องไม่สร้างรายการใหม่(self, store):
        for bogus in range(1000, 1200):
            assert store.delete_speaker(bogus) is False
        assert store._memory_epoch == {}

    def test_ลบคนที่มีอยู่จริงยังเลื่อนรุ่น(self, store):
        speaker = store.create_speaker("เดช")
        before = store.memory_epoch(speaker.id)
        assert store.delete_speaker(speaker.id) is True
        assert store.memory_epoch(speaker.id) > before


class Testเขียนได้ต่อเมื่อรุ่นความจำยังเดิม:
    """`apply_if_epoch` ต้องกันคำสั่งลบไม่ให้แทรกกลางการเขียน

    การ "ตรวจแล้วค่อยเขียน" แยกกันไม่อะตอมมิก งานสกัดเบื้องหลังเคยตรวจใต้ lock
    ของตัวเอง ซึ่งคนละตัวกับที่ `forget_everything` ถือ คำสั่งลบจึงแทรกได้ทั้ง
    ระหว่างตรวจกับเขียน และระหว่างการเขียนข้อเท็จจริงสองข้อ
    """

    def test_รุ่นตรงกันต้องเขียนได้(self, store):
        speaker = store.create_speaker("เดช")
        epoch = store.memory_epoch(speaker.id)
        assert store.apply_if_epoch(
            speaker.id, epoch, lambda: store.upsert_fact(speaker.id, "อาชีพ", "หมอ")
        )
        assert store.facts_for(speaker.id)

    def test_รุ่นเปลี่ยนแล้วต้องไม่เขียน(self, store):
        speaker = store.create_speaker("เดช")
        epoch = store.memory_epoch(speaker.id)
        store.forget_everything(speaker.id)
        assert not store.apply_if_epoch(
            speaker.id, epoch, lambda: store.upsert_fact(speaker.id, "อาชีพ", "หมอ")
        )
        assert store.facts_for(speaker.id) == []

    def test_คนถูกลบไปแล้วต้องไม่เขียน(self, store):
        speaker = store.create_speaker("เดช")
        store.delete_speaker(speaker.id)
        # ส่ง epoch เป็น None เพื่อให้เหลือแต่ด่าน "คนนี้ยังอยู่ไหม" ทำงานจริง
        # ไม่งั้นด่านรุ่นความจำจะบังมันไว้ (delete_speaker เลื่อนรุ่นไปแล้ว)
        # แล้วเทสต์นี้จะผ่านแม้ถอดด่านนั้นออกทั้งอัน
        assert not store.apply_if_epoch(
            speaker.id, None, lambda: store.upsert_fact(speaker.id, "อาชีพ", "หมอ")
        )

    def test_คำสั่งลบต้องแทรกกลางการเขียนไม่ได้(self, store):
        """เธรดหนึ่งเขียน 30 ข้อ อีกเธรดสั่งลบ — ต้องไม่เหลือรอดแม้ข้อเดียว"""
        import threading

        speaker = store.create_speaker("เดช")
        epoch = store.memory_epoch(speaker.id)
        กำลังเขียน = threading.Event()

        def write():
            for i in range(30):
                store.upsert_fact(speaker.id, f"k{i}", "ค่า")
                กำลังเขียน.set()

        worker = threading.Thread(
            target=lambda: store.apply_if_epoch(speaker.id, epoch, write)
        )
        worker.start()
        assert กำลังเขียน.wait(timeout=5)
        store.forget_everything(speaker.id)
        worker.join(timeout=5)

        assert store.facts_for(speaker.id) == [], "ความจำรอดจากคำสั่งลบ"


class Testลบทุกอย่างต้องล้างโปรไฟล์ที่โมเดลเดาไว้ด้วย:
    """ชื่อเล่นและเพศถูกเติมโดยโมเดลที่อ่านจากบทสนทนา

    จึงเป็นสิ่งที่สกัดมาจากบทสนทนาที่กำลังจะถูกลบ ของเดิมปล่อยไว้ บอทจึงยัง
    เรียกผู้ใช้ด้วยชื่อเล่นที่มันเดาเอาเองหลังผู้ใช้สั่งลบทุกอย่าง
    """

    def test_ชื่อเล่นและเพศต้องถูกล้าง(self, store):
        speaker = store.create_speaker("เดช")
        store.update_speaker(speaker.id, nickname="ชาย", gender="male", particle="ครับ")
        store.upsert_fact(speaker.id, "อาชีพ", "หมอ")

        store.forget_everything(speaker.id)

        after = store.get_speaker(speaker.id)
        assert after.nickname is None
        assert after.gender is None
        assert after.particle is None

    def test_ชื่อที่ใช้เรียกยังเป็นชื่อเดิม(self, store):
        """ยังรู้จักตัวเขาอยู่ — display_name คือตัวตนที่เราบอกว่ายังจำได้"""
        speaker = store.create_speaker("เดช")
        store.update_speaker(speaker.id, nickname="ชาย")
        store.forget_everything(speaker.id)
        assert store.get_speaker(speaker.id).call_name == "เดช"

    def test_stats_นับบทสรุปด้วย(self, store):
        speaker = store.create_speaker("เดช")
        assert store.stats(speaker.id).summaries == 0
        store.save_summary(speaker.id, "สรุป", 1)
        assert store.stats(speaker.id).summaries == 1
