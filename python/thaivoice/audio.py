"""ไมโครโฟนและลำโพง — จับเสียงพูดด้วย VAD และเล่นเสียงตอบพร้อมรองรับการพูดแทรก

รายละเอียดที่สำคัญกับภาษาไทยโดยเฉพาะ:

* **Pre-roll** — เก็บเสียงย้อนหลังราว 300 มิลลิวินาทีก่อนจุดที่ VAD ตัดสินว่า
  "เริ่มพูดแล้ว" เพราะถ้าพยางค์แรกถูกตัดหาย เสียงวรรณยุกต์จะเพี้ยนและ ASR
  จะถอดผิดทั้งคำ (เช่น "ข้าว" กลายเป็น "อ้าว")
* **หน่วงก่อนตัดจบ** — คนไทยเว้นจังหวะกลางประโยคบ่อย ค่าเริ่มต้น 800 มิลลิวินาที
  จึงเหมาะกว่า 400-500 ที่นิยมใช้กับภาษาอังกฤษ
* **พูดแทรก (barge-in)** — ถ้าผู้ใช้พูดขึ้นระหว่างที่บอทกำลังพูด ให้หยุดเสียงบอททันที
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Iterator

from .config import Settings, get_settings

log = logging.getLogger("thaivoice.audio")

__all__ = ["Microphone", "AudioPlayer", "audio_available", "FRAME_MS"]

FRAME_MS = 30  # webrtcvad รับเฉพาะเฟรม 10, 20 หรือ 30 มิลลิวินาที
PREROLL_MS = 300


def audio_available() -> tuple[bool, str]:
    """เช็คว่าเครื่องนี้ใช้ไมโครโฟนได้ไหม คืน (ใช้ได้, เหตุผล)"""
    try:
        import sounddevice  # type: ignore # noqa: F401
    except Exception as exc:
        return False, f"ยังไม่ได้ติดตั้ง sounddevice ({exc})"
    try:
        import webrtcvad  # type: ignore # noqa: F401
    except Exception as exc:
        return False, f"ยังไม่ได้ติดตั้ง webrtcvad ({exc})"
    try:
        import sounddevice as sd  # type: ignore

        if not any(d["max_input_channels"] > 0 for d in sd.query_devices()):
            return False, "ไม่พบอุปกรณ์รับเสียงเข้า (ไมโครโฟน)"
    except Exception as exc:
        return False, f"เข้าถึงอุปกรณ์เสียงไม่ได้ ({exc})"
    return True, ""


class Microphone:
    """จับเสียงพูดทีละประโยคจากไมโครโฟน"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.sample_rate = self.settings.sample_rate
        self.frame_bytes = int(self.sample_rate * FRAME_MS / 1000) * 2

        import sounddevice as sd  # type: ignore
        import webrtcvad  # type: ignore

        self._sd = sd
        self._vad = webrtcvad.Vad(self.settings.vad_aggressiveness)
        self._stop = threading.Event()

    # ── ระดับล่าง: สตรีมเฟรมเสียง ──────────────────────────────────────
    def frames(self) -> Iterator[tuple[bytes, bool]]:
        """สตรีมเฟรมเสียงพร้อมธงว่าเฟรมนั้นมีเสียงพูดไหม"""
        with self._sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_bytes // 2,
            dtype="int16",
            channels=1,
        ) as stream:
            while not self._stop.is_set():
                data, _overflowed = stream.read(self.frame_bytes // 2)
                frame = bytes(data)
                if len(frame) < self.frame_bytes:
                    continue
                try:
                    voiced = self._vad.is_speech(frame, self.sample_rate)
                except Exception:
                    voiced = False
                yield frame, voiced

    def stop(self) -> None:
        """สั่งให้หยุดฟัง — เรียก :meth:`resume` เพื่อกลับมาฟังใหม่ได้"""
        self._stop.set()

    def resume(self) -> None:
        """ยกเลิกคำสั่งหยุด ทำให้เริ่มฟังใหม่ได้

        ของเดิมไม่มีทางล้างธงนี้ ไมโครโฟนที่ถูกสั่งหยุดแล้วจึงใช้ต่อไม่ได้อีกเลย
        """
        self._stop.clear()

    # ── ระดับบน: จับทีละประโยค ─────────────────────────────────────────
    def record_utterance(
        self,
        on_speech_start: Callable[[], None] | None = None,
        start_frames: int = 3,
    ) -> bytes | None:
        """รอจนผู้ใช้พูดจบหนึ่งประโยค แล้วคืน PCM 16-bit

        คืน ``None`` ถ้าถูกสั่งหยุดก่อนได้ยินเสียงพูด
        """
        preroll = deque(maxlen=max(1, PREROLL_MS // FRAME_MS))
        collected: list[bytes] = []
        voiced_run = 0
        silence_run = 0
        started = False

        silence_frames = max(1, self.settings.silence_ms // FRAME_MS)
        max_frames = max(1, self.settings.max_utterance_s * 1000 // FRAME_MS)

        for frame, voiced in self.frames():
            if not started:
                preroll.append(frame)
                voiced_run = voiced_run + 1 if voiced else 0
                if voiced_run >= start_frames:
                    started = True
                    collected.extend(preroll)  # ใส่ pre-roll ไม่ให้พยางค์แรกหาย
                    if on_speech_start:
                        on_speech_start()
                continue

            collected.append(frame)
            silence_run = 0 if voiced else silence_run + 1
            if silence_run >= silence_frames or len(collected) >= max_frames:
                break

        if not collected:
            return None
        return b"".join(collected)

    def wait_for_speech(self, timeout_frames: int = 0, start_frames: int = 2) -> bool:
        """รอจนตรวจพบเสียงพูด — ใช้ตรวจจับการพูดแทรกระหว่างบอทกำลังพูด"""
        voiced_run = 0
        seen = 0
        for _frame, voiced in self.frames():
            seen += 1
            voiced_run = voiced_run + 1 if voiced else 0
            if voiced_run >= start_frames:
                return True
            if timeout_frames and seen >= timeout_frames:
                return False
        return False


class AudioPlayer:
    """เล่นเสียงตอบ และหยุดกลางคันได้เมื่อผู้ใช้พูดแทรก

    ใช้โปรแกรมเล่นเสียงของระบบ (ffplay / mpv / mpg123 / afplay) เพื่อเลี่ยงการ
    ผูกกับไลบรารีถอดรหัส mp3 ตัวใดตัวหนึ่ง
    """

    _CANDIDATES = (
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("mpv", ["--no-video", "--really-quiet"]),
        ("mpg123", ["-q"]),
        ("afplay", []),
        ("aplay", ["-q"]),
    )

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._player = self._find_player()

    @staticmethod
    def _find_player() -> tuple[str, list[str]] | None:
        for name, args in AudioPlayer._CANDIDATES:
            path = shutil.which(name)
            if path:
                return path, args
        return None

    @property
    def available(self) -> bool:
        return self._player is not None

    def play(self, audio: bytes, suffix: str = ".mp3", block: bool = True) -> bool:
        """เล่นเสียง

        ``block=True``  รอจนเล่นจบ คืน ``True`` ถ้าเล่นครบ ``False`` ถ้าถูกหยุดกลางคัน
        ``block=False`` สั่งเล่นแล้วคืนทันที คืน ``True`` ถ้าเริ่มเล่นได้

        ทั้งสองแบบลบไฟล์เสียงชั่วคราวให้เสมอ — โหมดไม่บล็อกใช้เธรดเก็บกวาดคอยรอ
        ให้กระบวนการเล่นจบก่อนแล้วค่อยลบ ถ้าลบทันทีบางโปรแกรมจะอ่านไฟล์ไม่ทัน
        """
        if not audio or self._player is None:
            return False
        binary, args = self._player
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio)
            path = Path(tmp.name)

        try:
            process = subprocess.Popen(
                [binary, *args, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            log.warning("เริ่มเล่นเสียงไม่สำเร็จ", exc_info=True)
            path.unlink(missing_ok=True)
            return False

        with self._lock:
            self._process = process

        if not block:
            threading.Thread(
                target=self._reap, args=(process, path), daemon=True
            ).start()
            return True

        try:
            return process.wait() == 0
        except Exception:
            log.warning("รอเสียงเล่นจบไม่สำเร็จ", exc_info=True)
            return False
        finally:
            self._release(process, path)

    def _reap(self, process: subprocess.Popen, path: Path) -> None:
        """รอให้กระบวนการเล่นเสียงจบแล้วเก็บกวาดไฟล์ชั่วคราว (ใช้กับโหมดไม่บล็อก)"""
        try:
            process.wait()
        except Exception:
            pass
        finally:
            self._release(process, path)

    def _release(self, process: subprocess.Popen, path: Path) -> None:
        """ล้างสถานะและลบไฟล์ชั่วคราว — ปลอดภัยเมื่อถูกเรียกซ้ำ"""
        with self._lock:
            # อย่าล้างทับ process ตัวใหม่ที่เพิ่งเริ่มเล่นหลังจากตัวนี้ถูกหยุด
            if self._process is process:
                self._process = None
        path.unlink(missing_ok=True)

    def stop(self) -> None:
        """หยุดเสียงที่กำลังเล่นทันที (ใช้ตอนผู้ใช้พูดแทรก)"""
        with self._lock:
            process = self._process
            self._process = None
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None
