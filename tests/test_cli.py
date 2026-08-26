"""ทดสอบเครื่องมือบรรทัดคำสั่ง

เป็นหน้าตาหลักที่ผู้ใช้เจอ แต่เดิมไม่มีเทสต์เลยสักข้อ
"""

import pytest
from conftest import FakeAnthropic, FakeEmbedder

from thaivoice import cli
from thaivoice.brain import ThaiBrain
from thaivoice.memory import MemoryStore
from thaivoice.session import ConversationSession
from thaivoice.speaker import SpeakerIdentifier
from thaivoice.stt import pcm_to_wav


class CliEnv:
    """สภาพแวดล้อมจำลองสำหรับ CLI

    แต่ละคำสั่งเปิดและปิดฐานข้อมูลของตัวเอง (ซึ่งถูกต้องแล้ว) เทสต์จึงต้องเปิด
    ตัวใหม่ทุกครั้งที่จะตรวจผล ห้ามถือ store ตัวเดียวค้างไว้
    """

    def __init__(self, db, settings, client):
        self.db = db
        self.settings = settings
        self.client = client

    def store(self) -> MemoryStore:
        return MemoryStore(self.db)


@pytest.fixture
def env(monkeypatch, tmp_path, settings, fake_client: FakeAnthropic):
    """ชี้ทุกคำสั่งไปที่ฐานข้อมูลชั่วคราว และใช้ Claude ปลอม"""
    db = tmp_path / "memory.db"
    live = settings.__class__(
        db_path=db,
        speaker_backend="none",
        tts_backend="none",
        stt_backend="external",
    )
    monkeypatch.setattr(cli, "get_settings", lambda *a, **k: live)

    def fake_create_session(*args, **kwargs):
        store = MemoryStore(db)
        return ConversationSession(
            store=store,
            brain=ThaiBrain(store, client=fake_client, settings=live),
            identifier=SpeakerIdentifier(store, FakeEmbedder(), threshold=0.9),
            settings=live,
            session_id="cli-test",
        )

    monkeypatch.setattr(cli, "create_session", fake_create_session)
    return CliEnv(db, live, fake_client)


class TestDoctor:
    def test_รายงานสภาพระบบได้โดยไม่พัง(self, env, capsys):
        assert cli.main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "ตรวจสภาพระบบ" in out
        assert "ความจำ" in out
        assert "โมเดล" in out

    def test_ไม่พ่น_traceback_เมื่อ_backend_ยังไม่ได้ติดตั้ง(self, env, capsys):
        cli.main(["doctor"])
        captured = capsys.readouterr()
        assert "Traceback" not in captured.out
        assert "Traceback" not in captured.err


class TestSpeakers:
    def test_ยังไม่รู้จักใคร(self, env, capsys):
        assert cli.main(["speakers"]) == 0
        assert "ยังไม่รู้จักใคร" in capsys.readouterr().out

    def test_แสดงรายชื่อและคอลัมน์ไม่เบี้ยว(self, env, capsys):
        store = env.store()
        store.create_speaker("เดชา", nickname="เดช", particle="ครับ")
        store.create_speaker("มะลิ", particle="ค่ะ")
        store.close()

        assert cli.main(["speakers"]) == 0
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        assert any("เดช" in l for l in lines)
        assert any("มะลิ" in l for l in lines)


class TestMemory:
    def test_ไม่พบผู้สนทนา(self, env, capsys):
        assert cli.main(["memory", "999"]) == 1
        assert "ไม่พบผู้สนทนา" in capsys.readouterr().out

    def test_แสดงความจำครบทุกชั้น(self, env, capsys):
        store = env.store()
        speaker = store.create_speaker("เดช", particle="ครับ")
        store.upsert_fact(speaker.id, "อาชีพ", "สถาปนิก", category="งาน")
        store.record_turn(speaker.id, "s", "user", "สวัสดีครับ")
        store.save_summary(speaker.id, "คุยเรื่องแผนไปเชียงใหม่", 1)
        store.close()

        assert cli.main(["memory", str(speaker.id), "--turns", "5"]) == 0
        out = capsys.readouterr().out
        assert "สถาปนิก" in out
        assert "เชียงใหม่" in out
        assert "สวัสดีครับ" in out


class TestForget:
    def test_ลบทั้งคนต้องยืนยัน(self, env, monkeypatch, capsys):
        speaker = env.store().create_speaker("เดช")
        monkeypatch.setattr("builtins.input", lambda *_: "ไม่")

        assert cli.main(["forget", str(speaker.id)]) == 1
        assert env.store().get_speaker(speaker.id) is not None
        assert "ยกเลิก" in capsys.readouterr().out

    def test_ลบทั้งคนเมื่อยืนยัน(self, env, capsys):
        speaker = env.store().create_speaker("เดช")
        assert cli.main(["forget", str(speaker.id), "--yes"]) == 0
        assert env.store().get_speaker(speaker.id) is None

    def test_ลบเฉพาะความจำแต่ยังรู้จักตัวคน(self, env, capsys):
        store = env.store()
        speaker = store.create_speaker("เดช")
        store.upsert_fact(speaker.id, "อาชีพ", "หมอ")
        store.record_turn(speaker.id, "s", "user", "ความลับ")
        store.save_summary(speaker.id, "สรุป", 1)
        store.close()

        assert cli.main(["forget", str(speaker.id), "--memory-only", "--yes"]) == 0

        check = env.store()
        assert check.get_speaker(speaker.id) is not None, "ยังต้องรู้จักตัวเขา"
        assert check.facts_for(speaker.id) == []
        assert check.recent_turns(speaker.id) == []
        assert check.latest_summary(speaker.id) is None
        assert "ลบความจำ" in capsys.readouterr().out


class TestChat:
    def test_พิมพ์คุยแล้วจำผู้พูดได้(self, env, monkeypatch, capsys):
        lines = iter(["สวัสดีครับ ผมชื่อเดชครับ", "/ออก"])
        monkeypatch.setattr("builtins.input", lambda *_: next(lines))

        assert cli.main(["chat"]) == 0
        out = capsys.readouterr().out
        assert "เดช" in out
        assert env.store().find_speaker_by_name("เดช") is not None

    def test_ปิดด้วย_EOF_ได้(self, env, monkeypatch):
        def raise_eof(*_):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert cli.main(["chat"]) == 0

    def test_ไม่มีคีย์ต้องบอกวิธีแก้แทน_traceback(self, env, monkeypatch, capsys):
        from thaivoice.brain import MissingCredentialsError

        def boom(*_args, **_kwargs):
            raise MissingCredentialsError("ยังไม่ได้ตั้งค่าคีย์")

        monkeypatch.setattr(cli, "cmd_speakers", boom)
        assert cli.main(["speakers"]) == 1
        assert "ยังไม่ได้ตั้งค่าคีย์" in capsys.readouterr().out


class TestEnroll:
    def test_ไม่มีตัวจำลายเสียงต้องบอกวิธีติดตั้ง(self, env, monkeypatch, tmp_path, capsys):
        wav = tmp_path / "a.wav"
        wav.write_bytes(pcm_to_wav(b"\x11\x22" * 800, 16000))

        original = cli.create_session

        def no_embedder(*args, **kwargs):
            session = original(*args, **kwargs)
            session.identifier = SpeakerIdentifier(session.store, None)
            return session

        monkeypatch.setattr(cli, "create_session", no_embedder)
        assert cli.main(["enroll", "--name", "เดช", "--wav", str(wav)]) == 1
        assert "pip install" in capsys.readouterr().out

    def test_สอนลายเสียงจากไฟล์(self, env, tmp_path, capsys):
        wav = tmp_path / "a.wav"
        wav.write_bytes(pcm_to_wav(b"\x33\x44" * 800, 16000))

        assert cli.main(["enroll", "--name", "เดช", "--wav", str(wav)]) == 0
        speaker = env.store().find_speaker_by_name("เดช")
        assert speaker is not None
        assert env.store().has_voiceprint(speaker.id, "fake")


def test_ตัวช่วยจัดคอลัมน์นับความกว้างจริงของอักษรไทย():
    """สระบนสระล่างและวรรณยุกต์ไม่กินความกว้างบนจอ ถ้าใช้ len() ตารางจะเบี้ยว"""
    assert cli._width("เดช") == 3
    assert cli._width("ก") == 1
    assert cli._width("ก่") == 1, "วรรณยุกต์ไม่กินความกว้าง"
    assert cli._width("น้ำ") == 2
    assert len(cli._pad("ก่", 5)) == 6, "ต้องเติมช่องว่างชดเชยอักขระผสม"
