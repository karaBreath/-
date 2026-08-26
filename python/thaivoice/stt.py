"""ถอดเสียงพูดภาษาไทยเป็นข้อความ (speech-to-text)

รองรับหลาย backend เพราะแต่ละที่มีข้อจำกัดต่างกัน:

* ``faster-whisper`` — รันบนเครื่องตัวเอง ไม่ส่งเสียงออกนอก เหมาะกับความเป็นส่วนตัว
  ภาษาไทยควรใช้โมเดล ``large-v3`` ขึ้นไป โมเดลเล็กผิดเยอะจนคุยไม่รู้เรื่อง
* ``google`` / ``azure`` — บริการคลาวด์ แม่นและเร็ว แต่ต้องส่งเสียงออกนอกเครื่อง
* ``external`` — ฝั่งไคลเอนต์ถอดเสียงมาเองแล้ว (เช่นเบราว์เซอร์ใช้ Web Speech API
  ที่รองรับ th-TH) เซิร์ฟเวอร์แค่รับข้อความ

ทุก backend รับ PCM 16-bit mono และคืน :class:`Transcription`
"""

from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .config import Settings, get_settings
from .thai_text import normalize_transcript, thai_ratio

log = logging.getLogger("thaivoice.stt")

__all__ = [
    "Transcription",
    "SpeechToText",
    "load_stt",
    "looks_like_thai",
    "pcm_to_wav",
    "wav_to_pcm",
]

# บอกใบ้ Whisper ว่ากำลังฟังภาษาไทยแบบสนทนา ช่วยลดการถอดเป็นคำทับศัพท์แปลก ๆ
THAI_PROMPT = "บทสนทนาภาษาไทยแบบเป็นกันเอง มีคำลงท้าย ครับ ค่ะ นะคะ"


@dataclass
class Transcription:
    text: str
    language: str = "th"
    confidence: float = 0.0
    duration: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.text.strip())


@runtime_checkable
class SpeechToText(Protocol):
    name: str

    def transcribe(self, pcm: bytes, sample_rate: int) -> Transcription: ...


# ── ตัวช่วยแปลงรูปแบบเสียง ──────────────────────────────────────────────────
def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """ห่อ PCM 16-bit ด้วยหัวไฟล์ WAV"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm(data: bytes) -> tuple[bytes, int]:
    """แกะไฟล์ WAV เป็น (PCM 16-bit mono, sample_rate)"""
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"รองรับเฉพาะ PCM 16-bit ไม่ใช่ {width * 8}-bit")
    if channels > 1:  # รวมช่องสัญญาณเป็น mono
        import array

        samples = array.array("h")
        samples.frombytes(frames)
        mono = array.array(
            "h",
            [
                sum(samples[i : i + channels]) // channels
                for i in range(0, len(samples), channels)
            ],
        )
        frames = mono.tobytes()
    return frames, rate


# ── backends ────────────────────────────────────────────────────────────────
class FasterWhisperSTT:
    """ถอดเสียงบนเครื่องด้วย faster-whisper (CTranslate2)"""

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        from faster_whisper import WhisperModel  # type: ignore

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, pcm: bytes, sample_rate: int) -> Transcription:
        import numpy as np  # type: ignore

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != 16000:
            samples = _resample(samples, sample_rate, 16000)
        segments, info = self._model.transcribe(
            samples,
            language="th",
            beam_size=5,
            vad_filter=True,
            initial_prompt=THAI_PROMPT,
            condition_on_previous_text=False,  # กันอาการวนซ้ำประโยคเดิมของ Whisper
        )
        segments = list(segments)
        text = normalize_transcript("".join(s.text for s in segments))
        confidence = 0.0
        if segments:
            import math

            confidence = sum(math.exp(s.avg_logprob) for s in segments) / len(segments)
        return Transcription(
            text=text,
            language=getattr(info, "language", "th"),
            confidence=confidence,
            duration=len(pcm) / 2 / sample_rate,
        )


class GoogleSTT:
    """ถอดเสียงด้วย Google Cloud Speech-to-Text (th-TH)"""

    name = "google"

    def __init__(self) -> None:
        from google.cloud import speech  # type: ignore

        self._client = speech.SpeechClient()
        self._speech = speech

    def transcribe(self, pcm: bytes, sample_rate: int) -> Transcription:
        speech = self._speech
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="th-TH",
            enable_automatic_punctuation=True,
            model="latest_long",
        )
        response = self._client.recognize(
            config=config, audio=speech.RecognitionAudio(content=pcm)
        )
        parts, confidences = [], []
        for result in response.results:
            if result.alternatives:
                parts.append(result.alternatives[0].transcript)
                confidences.append(result.alternatives[0].confidence)
        return Transcription(
            text=normalize_transcript(" ".join(parts)),
            confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            duration=len(pcm) / 2 / sample_rate,
        )


class AzureSTT:
    """ถอดเสียงด้วย Azure Speech (th-TH)"""

    name = "azure"

    def __init__(self, key: str, region: str) -> None:
        import azure.cognitiveservices.speech as speechsdk  # type: ignore

        self._sdk = speechsdk
        self._config = speechsdk.SpeechConfig(subscription=key, region=region)
        self._config.speech_recognition_language = "th-TH"

    def transcribe(self, pcm: bytes, sample_rate: int) -> Transcription:
        sdk = self._sdk
        stream = sdk.audio.PushAudioInputStream(
            sdk.audio.AudioStreamFormat(samples_per_second=sample_rate, bits_per_sample=16, channels=1)
        )
        stream.write(pcm)
        stream.close()
        recognizer = sdk.SpeechRecognizer(
            speech_config=self._config,
            audio_config=sdk.audio.AudioConfig(stream=stream),
        )
        result = recognizer.recognize_once()
        text = result.text if result.reason == sdk.ResultReason.RecognizedSpeech else ""
        return Transcription(
            text=normalize_transcript(text), duration=len(pcm) / 2 / sample_rate
        )


class ExternalSTT:
    """ไม่ถอดเสียงเอง — ไคลเอนต์ส่งข้อความที่ถอดแล้วมาให้

    ใช้กับเบราว์เซอร์ที่ใช้ Web Speech API (รองรับ th-TH) ถอดเสียงฝั่งหน้าเว็บ
    """

    name = "external"

    def transcribe(self, pcm: bytes, sample_rate: int) -> Transcription:
        raise RuntimeError(
            "backend 'external' ไม่ถอดเสียงเอง — ให้ไคลเอนต์ส่งข้อความที่ถอดแล้วมา"
        )


def _resample(samples, src_rate: int, dst_rate: int):
    """รีแซมเปิลแบบเชิงเส้น — พอสำหรับงานเสียงพูด และไม่ต้องพึ่ง scipy"""
    import numpy as np  # type: ignore

    if src_rate == dst_rate:
        return samples
    duration = len(samples) / src_rate
    target_len = int(duration * dst_rate)
    if target_len <= 0:
        return samples
    src_idx = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(src_idx, np.arange(len(samples)), samples).astype("float32")


def load_stt(settings: Settings | None = None) -> SpeechToText | None:
    """สร้าง backend ถอดเสียงตามการตั้งค่า — คืน ``None`` ถ้าใช้ไม่ได้"""
    settings = settings or get_settings()
    backend = settings.stt_backend.strip().lower()
    try:
        if backend in {"faster-whisper", "whisper"}:
            return FasterWhisperSTT(
                settings.whisper_model, settings.whisper_device, settings.whisper_compute
            )
        if backend == "google":
            return GoogleSTT()
        if backend == "azure":
            import os

            return AzureSTT(
                os.environ.get("AZURE_SPEECH_KEY", ""),
                os.environ.get("AZURE_SPEECH_REGION", "southeastasia"),
            )
        if backend in {"external", "browser", "none"}:
            return ExternalSTT()
    except Exception:
        log.warning("โหลด STT backend %r ไม่สำเร็จ", backend, exc_info=True)
        return None
    log.warning("ไม่รู้จัก STT backend %r", backend)
    return None


def looks_like_thai(transcription: Transcription, threshold: float = 0.5) -> bool:
    """เช็คว่าผลถอดเสียงเป็นภาษาไทยจริงไหม (กันกรณี ASR หลุดไปภาษาอื่น)"""
    return thai_ratio(transcription.text) >= threshold
