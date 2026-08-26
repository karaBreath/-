"""สังเคราะห์เสียงพูดภาษาไทย (text-to-speech)

เสียงไทยที่ฟังเป็นธรรมชาติที่สุดและใช้ได้ฟรีคือเสียง Neural ของ Microsoft
ผ่าน ``edge-tts`` (``th-TH-PremwadeeNeural`` หญิง / ``th-TH-NiwatNeural`` ชาย)
จึงตั้งเป็นค่าเริ่มต้น

backend ที่มี:

* ``edge``    — edge-tts เสียง Neural ไทย คุณภาพดีที่สุดในบรรดาตัวเลือกฟรี
* ``gtts``    — Google Translate TTS สำรอง ติดตั้งง่าย แต่เสียงแข็งกว่า
* ``azure``   — Azure Speech ใช้เสียงเดียวกับ edge แต่มี SLA และปรับแต่งได้ละเอียด
* ``browser`` — ไม่สังเคราะห์ที่เซิร์ฟเวอร์ ให้เบราว์เซอร์พูดเองด้วย Web Speech API
* ``none``    — ปิดเสียง (โหมดข้อความล้วน)

ข้อความจะถูกส่งผ่าน :func:`thaivoice.thai_text.clean_for_speech` ก่อนเสมอ
เพื่อถอด markdown และอิโมจิที่ TTS อ่านไม่ได้ออกไป
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .config import Settings, get_settings
from .thai_text import clean_for_speech

log = logging.getLogger("thaivoice.tts")

__all__ = ["Speech", "TextToSpeech", "load_tts", "voice_for_assistant"]


@dataclass
class Speech:
    """เสียงที่สังเคราะห์ได้ — ``audio`` เป็น ``None`` เมื่อให้ไคลเอนต์พูดเอง"""

    audio: bytes | None
    mime: str = "audio/mpeg"
    voice: str = ""
    text: str = ""

    def __bool__(self) -> bool:
        return bool(self.audio)


@runtime_checkable
class TextToSpeech(Protocol):
    name: str
    mime: str

    def synthesize(self, text: str, voice: str | None = None) -> Speech: ...


def _run_async(coro):
    """รัน coroutine ได้ทั้งจากโค้ด sync และจากในเธรดที่มี event loop อยู่แล้ว"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # มี loop ทำงานอยู่ในเธรดนี้ -> ย้ายไปรันในเธรดใหม่ กัน "loop already running"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class EdgeTTS:
    """เสียง Neural ภาษาไทยผ่าน edge-tts (ต้องต่ออินเทอร์เน็ต)"""

    name = "edge"
    mime = "audio/mpeg"

    def __init__(self, voice: str = "th-TH-PremwadeeNeural", rate: str = "+0%") -> None:
        import edge_tts  # type: ignore

        self._edge = edge_tts
        self.voice = voice
        self.rate = rate

    async def asynthesize(self, text: str, voice: str | None = None) -> Speech:
        spoken = clean_for_speech(text)
        if not spoken:
            return Speech(audio=None, mime=self.mime, voice=voice or self.voice)
        chosen = voice or self.voice
        communicate = self._edge.Communicate(spoken, chosen, rate=self.rate)
        buffer = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.extend(chunk["data"])
        return Speech(audio=bytes(buffer), mime=self.mime, voice=chosen, text=spoken)

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        return _run_async(self.asynthesize(text, voice))


class GttsTTS:
    """สำรองด้วย gTTS — ติดตั้งง่าย ใช้ได้ทันที แต่ปรับแต่งเสียงไม่ได้"""

    name = "gtts"
    mime = "audio/mpeg"

    def __init__(self) -> None:
        from gtts import gTTS  # type: ignore

        self._gtts = gTTS

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        import io

        spoken = clean_for_speech(text)
        if not spoken:
            return Speech(audio=None, mime=self.mime)
        buffer = io.BytesIO()
        self._gtts(text=spoken, lang="th", slow=False).write_to_fp(buffer)
        return Speech(audio=buffer.getvalue(), mime=self.mime, voice="gtts-th", text=spoken)


class AzureTTS:
    """เสียงไทยผ่าน Azure Speech"""

    name = "azure"
    mime = "audio/mpeg"

    def __init__(self, key: str, region: str, voice: str = "th-TH-PremwadeeNeural") -> None:
        import azure.cognitiveservices.speech as speechsdk  # type: ignore

        self._sdk = speechsdk
        self._config = speechsdk.SpeechConfig(subscription=key, region=region)
        self._config.speech_synthesis_voice_name = voice
        self._config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
        )
        self.voice = voice

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        spoken = clean_for_speech(text)
        if not spoken:
            return Speech(audio=None, mime=self.mime)
        if voice and voice != self._config.speech_synthesis_voice_name:
            self._config.speech_synthesis_voice_name = voice
        synth = self._sdk.SpeechSynthesizer(speech_config=self._config, audio_config=None)
        result = synth.speak_text_async(spoken).get()
        if result.reason != self._sdk.ResultReason.SynthesizingAudioCompleted:
            return Speech(audio=None, mime=self.mime, text=spoken)
        return Speech(
            audio=bytes(result.audio_data),
            mime=self.mime,
            voice=voice or self.voice,
            text=spoken,
        )


class ClientSideTTS:
    """ไม่สังเคราะห์เสียงที่เซิร์ฟเวอร์ — ส่งข้อความให้ไคลเอนต์พูดเอง"""

    name = "browser"
    mime = "text/plain"

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        return Speech(audio=None, mime=self.mime, voice=voice or "", text=clean_for_speech(text))


class SilentTTS(ClientSideTTS):
    """โหมดข้อความล้วน ไม่มีเสียง"""

    name = "none"


def voice_for_assistant(settings: Settings) -> str:
    """เสียงของบอท — ขึ้นกับเพศของ *ตัวบอทเอง* ไม่ใช่ของผู้ฟัง

    ของเดิมเลือกเสียงจากเพศของคู่สนทนา ซึ่งผิดหลักภาษาไทย (คำลงท้ายและน้ำเสียง
    บอกเพศของคนพูด) และยังเทียบกับค่า "male_voice" ที่ไม่มีอยู่จริงในระบบ
    ทำให้ค่า THAIVOICE_TTS_VOICE_MALE ไม่เคยถูกใช้เลย
    """
    return settings.assistant_voice


def load_tts(settings: Settings | None = None) -> TextToSpeech | None:
    """สร้าง backend สังเคราะห์เสียงตามการตั้งค่า

    ถ้า ``edge`` ใช้ไม่ได้ จะถอยไป ``gtts`` ให้อัตโนมัติ เพื่อไม่ให้ระบบเงียบสนิท
    """
    settings = settings or get_settings()
    backend = settings.tts_backend.strip().lower()
    try:
        if backend in {"edge", "edge-tts"}:
            return EdgeTTS(voice_for_assistant(settings), settings.tts_rate)
        if backend == "gtts":
            return GttsTTS()
        if backend == "azure":
            import os

            return AzureTTS(
                os.environ.get("AZURE_SPEECH_KEY", ""),
                os.environ.get("AZURE_SPEECH_REGION", "southeastasia"),
                voice_for_assistant(settings),
            )
        if backend == "browser":
            return ClientSideTTS()
        if backend in {"none", "off", "silent"}:
            return SilentTTS()
    except Exception:
        log.warning("โหลด TTS backend %r ไม่สำเร็จ", backend, exc_info=True)

    if backend != "gtts":
        try:
            log.info("ถอยไปใช้ gTTS แทน %r", backend)
            return GttsTTS()
        except Exception:
            log.warning("gTTS ก็ใช้ไม่ได้ — ระบบจะทำงานแบบไม่มีเสียง")
    return SilentTTS()
