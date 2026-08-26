"""จดจำ "ใครกำลังพูด" จากลายเสียง (speaker recognition)

หลักการ: แปลงเสียงพูดเป็นเวกเตอร์ลายเสียง (embedding) แล้วเทียบ cosine similarity
กับลายเสียงของทุกคนที่เคยคุยกัน ถ้าคล้ายพอและทิ้งห่างอันดับสองมากพอ ก็ถือว่าเป็น
คนเดิม ถ้าไม่ใช่ ระบบจะถามชื่อแล้วลงทะเบียนเสียงใหม่

ตัวสร้าง embedding เป็นแบบถอดเปลี่ยนได้:

* ``resemblyzer``  — ค่าเริ่มต้น น้ำหนักเบา รันบน CPU ได้
* ``speechbrain``  — แม่นกว่า (ECAPA-TDNN) แต่กินทรัพยากรมากกว่า
* ``none``         — ปิดการจำเสียง ใช้การบอกชื่อแทน

ถ้าไม่มีไลบรารีใดเลย ระบบยังทำงานได้ปกติ เพียงแต่ต้องระบุตัวตนด้วยการบอกชื่อ
(``extract_name_claim`` จับประโยคแนะนำตัวภาษาไทยให้อัตโนมัติ)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .memory import MemoryStore, Speaker

__all__ = [
    "Embedder",
    "Identification",
    "SpeakerIdentifier",
    "cosine_similarity",
    "extract_name_claim",
    "load_embedder",
]


# ── หาชื่อจากประโยคแนะนำตัวภาษาไทย ──────────────────────────────────────────
_NAME_PATTERNS = [
    re.compile(
        r"(?:ผม|ดิฉัน|ฉัน|หนู|เรา|กระผม|ข้าพเจ้า|อั๊ว)?\s*"
        r"(?:ชื่อว่า|ชื่อ|เรียกว่า|เรียกผมว่า|เรียกหนูว่า|เรียกฉันว่า)\s*"
        r"([฀-๿A-Za-z][฀-๿A-Za-z0-9.'-]{0,19})"
    ),
    re.compile(r"^\s*นี่\s*([฀-๿A-Za-z][฀-๿A-Za-z0-9.'-]{0,19})\s*(?:เอง|นะ|น่ะ)"),
    # คนไทยจำนวนมากแนะนำตัวเป็นภาษาอังกฤษปนไทย
    re.compile(
        r"(?:my name is|i\s*'?m|i am|call me)\s+([A-Za-z][A-Za-z.'-]{1,19})",
        re.IGNORECASE,
    ),
]

# คำที่มักติดมาท้ายชื่อเวลาพูด ต้องตัดทิ้งเพราะภาษาไทยไม่เว้นวรรคระหว่างคำ
_NAME_SUFFIXES = (
    "ครับผม", "ครับ", "คร้าบ", "คับ", "ค่ะ", "คะ", "ค่า", "ขา", "จ้า", "จ๊ะ", "จ้ะ",
    "ฮะ", "นะ", "น่ะ", "เอง", "ก็ได้", "ได้เลย", "แหละ", "ล่ะ", "ครับ.", "ค่ะ.",
)

_STOPWORD_NAMES = {"อะไร", "ไหน", "ใคร", "อะไรดี", "นี้", "นั้น", "เธอ", "คุณ"}


def extract_name_claim(text: str) -> str | None:
    """ดึงชื่อจากประโยคแนะนำตัว เช่น "ผมชื่อสมชายครับ" -> "สมชาย"

    คืน ``None`` ถ้าไม่พบชื่อที่น่าเชื่อถือ

    >>> extract_name_claim("ผมชื่อสมชายครับ")
    'สมชาย'
    >>> extract_name_claim("อยากกินอะไรดี")
    """
    if not text:
        return None
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = match.group(1).strip(" .,!?\"'")
        # ตัดคำลงท้าย/คำเสริมที่ติดมากับชื่อ (วนซ้ำเผื่อซ้อนกัน เช่น "นะครับ")
        changed = True
        while changed and name:
            changed = False
            for suffix in _NAME_SUFFIXES:
                if len(name) > len(suffix) and name.endswith(suffix):
                    name = name[: -len(suffix)].strip()
                    changed = True
        if len(name) >= 2 and name not in _STOPWORD_NAMES:
            return name
    return None


# ── ตัวสร้าง embedding ──────────────────────────────────────────────────────
@runtime_checkable
class Embedder(Protocol):
    """อินเทอร์เฟซของตัวแปลงเสียงเป็นเวกเตอร์ลายเสียง"""

    name: str

    def embed(self, pcm: bytes, sample_rate: int) -> list[float]:
        """รับ PCM 16-bit mono คืนเวกเตอร์ลายเสียงที่ normalize แล้ว"""
        ...


class ResemblyzerEmbedder:
    """ลายเสียงด้วย Resemblyzer (GE2E) — เบาและเร็วพอสำหรับงานเรียลไทม์"""

    name = "resemblyzer"

    def __init__(self) -> None:
        from resemblyzer import VoiceEncoder  # type: ignore

        self._encoder = VoiceEncoder()

    def embed(self, pcm: bytes, sample_rate: int) -> list[float]:
        import numpy as np  # type: ignore
        from resemblyzer import preprocess_wav  # type: ignore

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        wav = preprocess_wav(samples, source_sr=sample_rate)
        vec = self._encoder.embed_utterance(wav)
        return [float(x) for x in vec]


class SpeechBrainEmbedder:
    """ลายเสียงด้วย SpeechBrain ECAPA-TDNN — แม่นกว่าแต่หนักกว่า"""

    name = "speechbrain"

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        import torch  # type: ignore  # noqa: F401
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        self._model = EncoderClassifier.from_hparams(source=source)

    def embed(self, pcm: bytes, sample_rate: int) -> list[float]:
        import numpy as np  # type: ignore
        import torch  # type: ignore

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples).unsqueeze(0)
        with torch.no_grad():
            vec = self._model.encode_batch(tensor).squeeze().cpu().numpy()
        return [float(x) for x in vec]


def load_embedder(backend: str) -> Embedder | None:
    """สร้างตัวแปลงลายเสียงตามชื่อ backend — คืน ``None`` ถ้าใช้ไม่ได้

    ตั้งใจไม่ให้ throw เพื่อให้ระบบยังคุยได้แม้ยังไม่ได้ติดตั้งไลบรารีจำเสียง
    """
    backend = (backend or "none").strip().lower()
    if backend in {"none", "off", "disabled", ""}:
        return None
    try:
        if backend == "resemblyzer":
            return ResemblyzerEmbedder()
        if backend == "speechbrain":
            return SpeechBrainEmbedder()
    except Exception:
        return None
    return None


# ── เทียบความคล้าย ──────────────────────────────────────────────────────────
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """ความคล้ายเชิงโคไซน์ (-1 ถึง 1) — คืน 0.0 ถ้าเวกเตอร์ไม่เข้ากัน"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Identification:
    """ผลการระบุตัวผู้พูด"""

    speaker: Speaker | None
    score: float = 0.0
    runner_up: float = 0.0
    is_new: bool = False
    method: str = "none"  # voice | name | fallback | none

    @property
    def confident(self) -> bool:
        return self.speaker is not None and not self.is_new


class SpeakerIdentifier:
    """ระบุตัวผู้พูดจากเสียง และลงทะเบียนคนใหม่

    ``threshold`` คือความคล้ายขั้นต่ำที่จะยอมรับว่าเป็นคนเดิม
    ``margin`` คือระยะห่างขั้นต่ำจากอันดับสอง — กันกรณีเสียงคล้ายกันสองคน
    ทำให้ระบบสลับตัวคนกลางบทสนทนา
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder | None,
        *,
        threshold: float = 0.75,
        margin: float = 0.06,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self.margin = margin

    @property
    def enabled(self) -> bool:
        return self.embedder is not None

    @property
    def backend_name(self) -> str:
        return self.embedder.name if self.embedder else "none"

    def identify(self, pcm: bytes, sample_rate: int) -> Identification:
        """ระบุว่าเสียงนี้เป็นของใคร"""
        if self.embedder is None:
            return Identification(speaker=None, method="none")
        try:
            vec = self.embedder.embed(pcm, sample_rate)
        except Exception:
            return Identification(speaker=None, method="none")
        return self.identify_embedding(vec)

    def identify_embedding(self, vec: Sequence[float]) -> Identification:
        """เทียบเวกเตอร์ลายเสียงกับทุกคนในฐานความจำ"""
        backend = self.backend_name
        scored = sorted(
            (
                (cosine_similarity(vec, stored), speaker_id)
                for speaker_id, stored in self.store.all_voiceprints(backend)
            ),
            reverse=True,
        )
        if not scored:
            return Identification(speaker=None, is_new=True, method="voice")

        best_score, best_id = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0

        if best_score >= self.threshold and (best_score - runner_up) >= self.margin:
            speaker = self.store.get_speaker(best_id)
            if speaker is not None:
                self.store.touch_speaker(best_id)
                return Identification(
                    speaker=speaker,
                    score=best_score,
                    runner_up=runner_up,
                    method="voice",
                )
        return Identification(
            speaker=None, score=best_score, runner_up=runner_up, is_new=True, method="voice"
        )

    def enroll(
        self,
        speaker: Speaker,
        pcm: bytes,
        sample_rate: int,
    ) -> bool:
        """ผูกเสียงนี้เข้ากับคนคนนี้ (เรียกซ้ำได้ ระบบจะเฉลี่ยสะสมให้แม่นขึ้น)"""
        if self.embedder is None:
            return False
        try:
            vec = self.embedder.embed(pcm, sample_rate)
            self.store.save_voiceprint(speaker.id, vec, self.backend_name)
            return True
        except Exception:
            return False

    def resolve(
        self,
        pcm: bytes | None,
        sample_rate: int,
        transcript: str,
        *,
        default_name: str = "ผู้ใช้ใหม่",
    ) -> Identification:
        """ระบุตัวตนโดยใช้ทั้งเสียงและเนื้อความ

        ลำดับการตัดสิน:

        1. ถ้าจำเสียงได้ชัดเจน -> ใช้คนนั้น
        2. ถ้าในประโยคมีการบอกชื่อ -> ใช้ชื่อนั้น (สร้างใหม่ถ้ายังไม่มี) และ
           ผูกลายเสียงเข้ากับชื่อนั้นทันที เพื่อครั้งหน้าจะจำได้เอง
        3. ถ้าไม่เข้าเงื่อนไขไหนเลย -> คืนผลว่าเป็นคนใหม่ ให้ชั้นบนถามชื่อต่อ
        """
        ident = Identification(speaker=None, is_new=True, method="none")
        if pcm is not None and self.enabled:
            ident = self.identify(pcm, sample_rate)
            if ident.confident:
                return ident

        claimed = extract_name_claim(transcript)
        if claimed:
            speaker = self.store.find_speaker_by_name(claimed)
            created = speaker is None
            if speaker is None:
                from .thai_text import detect_particle, particle_for_gender

                gender = detect_particle(transcript)
                speaker = self.store.create_speaker(
                    claimed,
                    gender=gender,
                    particle=particle_for_gender(gender),
                )
            else:
                self.store.touch_speaker(speaker.id)
            if pcm is not None and self.enabled:
                self.enroll(speaker, pcm, sample_rate)
            return Identification(
                speaker=speaker,
                score=ident.score,
                is_new=created,
                method="name",
            )

        return ident
