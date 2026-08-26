"""ทดสอบตัวเล่นเสียง — เน้นเรื่องการเก็บกวาดไฟล์ชั่วคราวและการหยุดกลางคัน

เครื่องทดสอบส่วนใหญ่ไม่มีทั้งลำโพงและ ffplay จึงแทนโปรแกรมเล่นเสียงด้วยคำสั่ง
มาตรฐานของระบบ เพื่อทดสอบวงจรชีวิตของกระบวนการและไฟล์ได้จริงโดยไม่ต้องมีเสียง
"""

import shutil
import time
from pathlib import Path

import pytest

from thaivoice.audio import AudioPlayer, audio_available


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.fixture
def player(monkeypatch, tmp_path):
    """ตัวเล่นเสียงปลอมที่ใช้ `cat` แทนโปรแกรมเล่นเสียงจริง"""
    cat = shutil.which("cat")
    if cat is None:  # pragma: no cover
        pytest.skip("ไม่มีคำสั่ง cat")

    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    made = AudioPlayer()
    monkeypatch.setattr(made, "_player", (cat, []))
    return made


def _temp_files(tmp_path: Path) -> list[Path]:
    return [p for p in tmp_path.iterdir() if p.suffix == ".mp3"]


class TestPlayback:
    def test_เล่นแบบรอจนจบแล้วลบไฟล์ชั่วคราว(self, player, tmp_path):
        assert player.play("เสียงปลอม".encode().ljust(64, b"\x00")) is True
        assert _temp_files(tmp_path) == []
        assert player.playing is False

    def test_เล่นแบบไม่รอก็ต้องลบไฟล์ชั่วคราวเหมือนกัน(self, player, tmp_path):
        """เคยมีบัค: โหมดไม่บล็อกคืนค่าทันทีโดยไม่ลบไฟล์ ทำให้ไฟล์กองอยู่ตลอดกาล"""
        assert player.play(b"x" * 64, block=False) is True

        assert _wait_until(lambda: _temp_files(tmp_path) == []), (
            f"ไฟล์ชั่วคราวยังค้างอยู่: {_temp_files(tmp_path)}"
        )
        assert _wait_until(lambda: player.playing is False)

    def test_เล่นหลายครั้งไม่ทิ้งขยะสะสม(self, player, tmp_path):
        for _ in range(10):
            player.play(b"y" * 32, block=False)
        assert _wait_until(lambda: _temp_files(tmp_path) == [], timeout=10)

    def test_เสียงว่างไม่สร้างไฟล์(self, player, tmp_path):
        assert player.play(b"") is False
        assert _temp_files(tmp_path) == []

    def test_ไม่มีโปรแกรมเล่นเสียงต้องไม่พังและไม่สร้างไฟล์(self, monkeypatch, tmp_path):
        import tempfile

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        quiet = AudioPlayer()
        monkeypatch.setattr(quiet, "_player", None)

        assert quiet.available is False
        assert quiet.play(b"x" * 32) is False
        assert _temp_files(tmp_path) == []

    def test_เริ่มโปรแกรมเล่นเสียงไม่สำเร็จก็ต้องลบไฟล์(self, monkeypatch, tmp_path):
        import tempfile

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        broken = AudioPlayer()
        monkeypatch.setattr(broken, "_player", ("/ไม่มีโปรแกรมนี้จริง", []))

        assert broken.play(b"x" * 32) is False
        assert _temp_files(tmp_path) == [], "ล้มเหลวแล้วต้องไม่ทิ้งไฟล์ค้างไว้"


class TestStop:
    def test_หยุดกลางคันได้(self, monkeypatch, tmp_path):
        """จำลองการพูดแทรก: ผู้ใช้พูดขึ้นมาระหว่างบอทกำลังพูด"""
        sleep = shutil.which("sleep")
        if sleep is None:  # pragma: no cover
            pytest.skip("ไม่มีคำสั่ง sleep")

        import tempfile

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        long_player = AudioPlayer()
        # `sleep 30 <ไฟล์>` จะเล่นค้างไว้จนกว่าจะถูกสั่งหยุด
        monkeypatch.setattr(long_player, "_player", (sleep, ["30"]))

        assert long_player.play(b"z" * 32, block=False) is True
        assert _wait_until(lambda: long_player.playing is True)

        long_player.stop()

        assert _wait_until(lambda: long_player.playing is False)
        assert _wait_until(lambda: _temp_files(tmp_path) == []), "หยุดแล้วต้องลบไฟล์ด้วย"

    def test_สั่งหยุดตอนไม่ได้เล่นอยู่ก็ไม่พัง(self, player):
        player.stop()
        player.stop()
        assert player.playing is False


def test_รายงานสถานะไมโครโฟนโดยไม่พังบนเครื่องที่ไม่มีเสียง():
    ok, why = audio_available()
    assert isinstance(ok, bool)
    assert ok or why, "ถ้าใช้ไม่ได้ต้องบอกเหตุผลเสมอ"
