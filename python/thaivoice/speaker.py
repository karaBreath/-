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
#
# กฎเหล็กของส่วนนี้: **ยอมพลาดดีกว่าเดาผิด**
#
# ถ้าจับชื่อผิด ระบบจะสร้างตัวตนใหม่ขึ้นมาแล้วผูกลายเสียงของผู้ใช้เข้ากับตัวตนนั้น
# ความจำจริงของเขาจะถูกทิ้งค้างไว้ และครั้งต่อไประบบจะทักเขาด้วยชื่อผิด
# ส่วนการ "พลาด" แค่ทำให้ต้องบอกชื่ออีกครั้ง ซึ่งเสียหายน้อยกว่ามาก
#
# จึงบังคับว่าต้องขึ้นต้น *วรรค* ด้วยสรรพนามบุรุษที่หนึ่งหรือคำว่า "ชื่อ" เท่านั้น
# ภาษาไทยไม่เว้นวรรคระหว่างคำแต่เว้นระหว่างวรรคตอน กฎนี้จึงยอมรับ
# "สวัสดีครับ ผมชื่อเดชครับ" แต่ตัด "เพื่อนผมชื่อสมชาย" และ "แมวผมชื่อมิ้นท์" ทิ้ง
# เพราะสรรพนามในสองประโยคหลังเป็นเจ้าของ ไม่ได้ขึ้นต้นวรรค

_PRONOUN = r"(?:ผม|ดิฉัน|ฉัน|หนู|กระผม|ข้าพเจ้า|เรา|อั๊ว)"
_NAME_CHARS = r"[ก-๙A-Za-z][ก-๙A-Za-z0-9.'-]{0,14}"

_NAME_PATTERNS = [
    # "ชื่อเล่นชื่อบิ๊กครับ" / "ชื่อเล่นว่าบิ๊ก" — ต้องมาก่อนกฎทั่วไป
    re.compile(rf"ชื่อเล่น(?:ชื่อ|ว่า|คือ)?\s*({_NAME_CHARS})"),
    # "ผมชื่อสมชายครับ" / "หนูชื่อว่าแนนค่ะ"
    re.compile(rf"(?:^|\s){_PRONOUN}\s*ชื่อ(?:ว่า|จริงว่า)?\s*({_NAME_CHARS})"),
    # "ชื่อเดชครับ" — ต้องอยู่ต้นประโยคเท่านั้น
    re.compile(rf"(?:^|\s)ชื่อ(?:ว่า)?\s*({_NAME_CHARS})"),
    # "เรียกผมว่าพี่เดช" / "เรียกว่าโบท"
    re.compile(rf"เรียก(?:ผม|ฉัน|หนู|เรา|ดิฉัน)?ว่า\s*({_NAME_CHARS})"),
    # "เรียกเดชก็ได้"
    re.compile(rf"เรียก\s*({_NAME_CHARS}?)\s*ก็ได้"),
    # "นี่ต้นเองนะ"
    re.compile(rf"(?:^|\s)นี่\s*({_NAME_CHARS}?)\s*(?:เอง|นะ|น่ะ)"),
    # "ผมเดชครับ" — สรรพนาม + ชื่อ + คำลงท้าย โดยไม่มีคำว่าชื่อ
    re.compile(
        rf"(?:^|\s){_PRONOUN}\s*({_NAME_CHARS}?)\s*"
        r"(?:นะ)?(?:ครับผม|ครับ|ค่ะ|คะ)\s*[.!?]?\s*$"
    ),
    # ภาษาอังกฤษ — ต้องขึ้นต้นด้วยตัวใหญ่ ไม่งั้น "I'm going" จะกลายเป็นชื่อ "going"
    # ไม่ใช้ re.IGNORECASE เพราะต้องบังคับให้ *ชื่อ* ขึ้นต้นด้วยตัวใหญ่
    # ("I'm going" จะได้ไม่กลายเป็นชื่อ "going") จึงเขียนตัวเลือกตัวพิมพ์เอง
    re.compile(
        r"(?:\b[Mm]y name is\b|\b[Ii]\s*'?m\b|\b[Ii] am\b|\b[Cc]all me\b)\s+"
        r"([A-Z][A-Za-z.'-]{1,19})"
    ),
]

# คำที่มักติดมาท้ายชื่อเวลาพูด ต้องตัดทิ้งเพราะภาษาไทยไม่เว้นวรรคระหว่างคำ
_NAME_SUFFIXES = (
    "ครับผม", "ครับ", "คร้าบ", "คับ", "ค่ะ", "คะ", "ค่า", "ขา", "จ้า", "จ๊ะ", "จ้ะ",
    "ฮะ", "นะ", "น่ะ", "เอง", "ก็ได้", "ได้เลย", "แหละ", "ล่ะ", "เลย", "อ่ะ",
)

# คำนำหน้าที่ไม่ใช่ส่วนหนึ่งของชื่อ
_NAME_TITLES = ("นางสาว", "นาย", "นาง", "คุณ", "พี่", "น้อง", "น้า", "ป้า", "ลุง", "อา")

# ถ้าชื่อที่จับได้ "ขึ้นต้นด้วย" คำพวกนี้ แปลว่าไม่ใช่การแนะนำตัว
# (เช่น "ชื่อเสียง" = reputation, "ชื่อไฟล์" = filename)
_NOT_A_NAME_PREFIXES = (
    "เสียง", "เล่น", "จริง", "ไฟล์", "ร้าน", "ผู้", "หนัง", "บริษัท", "เพลง",
    "หนังสือ", "ยา", "โรค", "ถนน", "ซอย", "เมนู", "สินค้า", "แบรนด์", "ทีม",
    "วง", "ตัวละคร", "เรื่อง", "ลูก", "แมว", "หมา", "เพื่อน", "เขา", "เธอ",
    "อะไร", "นี้", "นั้น", "บัญชี", "โดเมน", "จังหวัด", "โรงเรียน", "เดียว",
    "ตาม", "ที่", "ของ",
)

# ถ้าในชื่อมีคำพวกนี้อยู่ แปลว่าจับเกินไปโดนคำอื่นเข้ามาด้วย
_NAME_REJECT_SUBSTRINGS = (
    "อะไร", "ไหม", "มั้ย", "เหรอ", "ยังไง", "คือ", "ว่า", "แล้ว", "ไป", "มา",
    "ให้", "หน่อย", "ด้วย", "จำ", "ไม่",
    # คำเชื่อม — ถ้าติดมาแปลว่าจับเลยขอบชื่อไปโดนประโยคถัดไปแล้ว
    "แต่", "และ", "หรือ", "กับ",
)

_STOPWORD_NAMES = {
    "อะไร", "ไหน", "ใคร", "นี้", "นั้น", "เธอ", "คุณ", "เขา", "มัน", "เรา",
    "ชื่อ", "ตัว", "คน", "งาน", "วันนี้",
}


def _clean_name(raw: str) -> str | None:
    """ตัดคำลงท้าย/คำนำหน้าออกจากชื่อที่จับได้ แล้วตรวจว่าน่าเชื่อถือไหม"""
    name = raw.strip(" .,!?\"'")

    # ตัดคำลงท้ายที่ติดมา (วนซ้ำเผื่อซ้อนกัน เช่น "นะครับ")
    changed = True
    while changed and name:
        changed = False
        for suffix in _NAME_SUFFIXES:
            if len(name) > len(suffix) and name.endswith(suffix):
                name = name[: -len(suffix)].strip()
                changed = True

    # ตัดคำนำหน้า เช่น "พี่เดช" -> "เดช"
    for title in _NAME_TITLES:
        if len(name) > len(title) + 1 and name.startswith(title):
            name = name[len(title) :].strip()
            break

    if len(name) < 2 or len(name) > 15:
        return None
    if name in _STOPWORD_NAMES:
        return None
    if any(name.startswith(prefix) for prefix in _NOT_A_NAME_PREFIXES):
        return None
    if any(bad in name for bad in _NAME_REJECT_SUBSTRINGS):
        return None
    return name


def extract_name_claim(text: str) -> str | None:
    """ดึงชื่อจากประโยคแนะนำตัว เช่น "ผมชื่อสมชายครับ" -> "สมชาย"

    คืน ``None`` เมื่อไม่มั่นใจ ซึ่งเป็นค่าที่ปลอดภัยกว่าเสมอ

    >>> extract_name_claim("ผมชื่อสมชายครับ")
    'สมชาย'
    >>> extract_name_claim("เพื่อนผมชื่อสมชาย")
    >>> extract_name_claim("ชื่อร้านนี้คืออะไร")
    """
    if not text:
        return None
    for pattern in _NAME_PATTERNS:
        for match in pattern.finditer(text):
            name = _clean_name(match.group(1))
            if name:
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
    """ความคล้ายเชิงโคไซน์ (-1 ถึง 1) — คืน 0.0 ถ้าเวกเตอร์ไม่เข้ากันหรือมีค่าเสีย

    ต้องกัน NaN/inf ให้ดี เพราะการเทียบกับ NaN ให้ผลเป็น False ทุกกรณี ถ้าปล่อยให้
    NaN หลุดออกไป การเรียงลำดับผู้พูดจะเพี้ยนทั้งชุด และ "คนที่คล้ายที่สุด" อาจ
    กลายเป็นแถวที่เสียหาย ทำให้ระบบจำใครไม่ได้เลยทั้งที่ลายเสียงตรงเป๊ะ
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        if not (math.isfinite(x) and math.isfinite(y)):
            return 0.0
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    score = dot / (math.sqrt(na) * math.sqrt(nb))
    return score if math.isfinite(score) else 0.0


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
