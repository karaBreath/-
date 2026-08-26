"""เซิร์ฟเวอร์ HTTP + WebSocket สำหรับไคลเอนต์ภายนอก (เว็บ, แอป, LINE bot)

จุดสำคัญของการออกแบบ:

* **สตรีมผ่าน WebSocket** — ส่งทั้งข้อความและเสียงกลับทีละประโยค ไคลเอนต์จึง
  เริ่มเล่นเสียงประโยคแรกได้ก่อนที่คำตอบจะพิมพ์จบ
* **เลือกได้ว่าจะถอดเสียงที่ไหน** — เบราว์เซอร์สมัยใหม่ถอดเสียงไทย (th-TH) ได้เอง
  ด้วย Web Speech API ซึ่งเร็วและไม่ต้องอัปโหลดเสียง แต่ถ้าอยากให้ทุกอย่างอยู่ใน
  เครื่องตัวเอง ก็ส่งไฟล์เสียงมาที่ ``/api/voice`` ให้เซิร์ฟเวอร์ถอดแทนได้
* **แนบเสียงมาด้วยเสมอถ้าทำได้** — เสียงคือสิ่งที่ทำให้ระบบรู้ว่า "ใครพูด"
  ถ้าส่งมาแต่ข้อความ ระบบจะจำผู้พูดจากลายเสียงไม่ได้

**เรื่องความเป็นส่วนตัวที่ต้องเข้าใจ:** ``session_id`` เป็นข้อความที่ไคลเอนต์
กำหนดเอง ใครส่ง id เดียวกันมาก็ถือเป็นบทสนทนาเดียวกัน คำขอที่ไม่ส่ง ``session_id``
จะได้บทสนทนาชั่วคราวที่ *ไม่* จำว่าเมื่อกี้คุยกับใคร เพื่อไม่ให้คนที่สองสวมรอย
เป็นคนแรกและเห็นความจำของคนแรก ถ้านำไปใช้จริงกับผู้ใช้หลายคน ให้ผูก session กับ
ระบบยืนยันตัวตนของคุณเอง อย่าเชื่อค่าที่ไคลเอนต์ส่งมา
"""

# หมายเหตุ: โมดูลนี้ตั้งใจ *ไม่* ใช้ ``from __future__ import annotations``
# เพราะ FastAPI ต้องอ่านชนิดของพารามิเตอร์จริง ๆ เพื่อสร้าง schema ถ้า annotation
# กลายเป็นสตริง FastAPI จะไปหาชื่อนั้นใน globals ของโมดูล แล้วหา UploadFile /
# WebSocket ที่ import ไว้ในฟังก์ชันไม่เจอ ทำให้ทุก endpoint ที่รับไฟล์พังหมด

import asyncio
import base64
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .brain import ThaiBrain
from .config import Settings, get_settings
from .extraction import MemoryExtractor
from .memory import MemoryStore, Speaker
from .session import ConversationSession
from .speaker import SpeakerIdentifier, load_embedder
from .stt import load_stt, wav_to_pcm
from .tts import load_tts

log = logging.getLogger("thaivoice.server")

__all__ = [
    "create_app",
    "run",
    "ServerRuntime",
    "ChatRequest",
    "SpeakRequest",
    "NewSpeakerRequest",
]

# ข้อความยาวเกินนี้ไม่ใช่คำพูดของคนแล้ว และจะถูกส่งซ้ำในทุก prompt ถัดไป
MAX_TEXT_CHARS = 4000
# ไฟล์เสียงหนึ่งประโยคไม่ควรใหญ่กว่านี้ — กันการอัปโหลดที่ทำให้เซิร์ฟเวอร์ค้าง
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# จำนวน session ที่เก็บไว้ในหน่วยความจำพร้อมกัน
MAX_LIVE_SESSIONS = 256


class ChatRequest(BaseModel):
    """คุยด้วยข้อความล้วน"""

    text: str = Field(description="สิ่งที่ผู้ใช้พูดหรือพิมพ์", max_length=MAX_TEXT_CHARS)
    session_id: Optional[str] = Field(default=None, max_length=128)
    speaker_id: Optional[int] = Field(default=None, description="ระบุตัวผู้พูดไว้ล่วงหน้า")
    speak: bool = Field(default=False, description="ให้เซิร์ฟเวอร์สังเคราะห์เสียงคำตอบด้วยไหม")


class SpeakRequest(BaseModel):
    """ขอให้สังเคราะห์เสียงจากข้อความ"""

    text: str = Field(max_length=MAX_TEXT_CHARS)
    voice: Optional[str] = Field(default=None, max_length=100)


class NewSpeakerRequest(BaseModel):
    """สร้างผู้สนทนาใหม่"""

    name: str = Field(min_length=1, max_length=80)
    gender: Optional[str] = Field(default=None, pattern="^(male|female)$")


class ServerRuntime:
    """ของที่แพงและใช้ร่วมกันได้ — สร้างครั้งเดียวตอนเปิดแอป

    ของเดิมเรียก ``create_session()`` ใหม่ทุกครั้งที่เจอ ``session_id`` ที่ไม่เคยเห็น
    ซึ่งโหลดโมเดลจำลายเสียง สร้างไคลเอนต์ และเปิด thread pool ใหม่ทุกครั้ง
    แล้วไม่เคยคืนทรัพยากรเลย ส่ง id สุ่ม ๆ มาไม่กี่พันครั้งก็กินหน่วยความจำหมดเครื่อง
    """

    def __init__(
        self,
        settings: "Settings | None" = None,
        *,
        store: "MemoryStore | None" = None,
        client: Any = None,
        embedder: Any = None,
        tts: Any = None,
        stt: Any = None,
        with_memory_extraction: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or MemoryStore(self.settings.db_path)
        self.brain = ThaiBrain(self.store, client=client, settings=self.settings)
        self.identifier = SpeakerIdentifier(
            self.store,
            embedder if embedder is not None else load_embedder(self.settings.speaker_backend),
            threshold=self.settings.speaker_threshold,
        )
        self.tts = tts if tts is not None else load_tts(self.settings)
        self.stt = stt if stt is not None else load_stt(self.settings)
        self.extractor = (
            MemoryExtractor(self.store, self.brain.client, self.settings)
            if with_memory_extraction
            else None
        )
        self._sessions: "OrderedDict[str, ConversationSession]" = OrderedDict()
        self._lock = threading.Lock()

    def session(self, session_id: "str | None") -> ConversationSession:
        """คืนบทสนทนาสำหรับ session นี้

        คำขอที่ไม่ระบุ ``session_id`` จะได้บทสนทนาชั่วคราวที่ไม่จำผู้พูดข้ามคำขอ
        ไม่งั้นคำขอของคนที่สองจะสวมรอยเป็นคนแรกและได้เห็นความจำของคนแรกทั้งหมด
        """
        if not session_id:
            return self._make(None, sticky=False)

        with self._lock:
            found = self._sessions.get(session_id)
            if found is not None:
                self._sessions.move_to_end(session_id)
                return found
            made = self._make(session_id, sticky=True)
            self._sessions[session_id] = made
            while len(self._sessions) > MAX_LIVE_SESSIONS:
                evicted_id, _evicted = self._sessions.popitem(last=False)
                log.info("ปลด session เก่าออกจากหน่วยความจำ: %s", evicted_id)
            return made

    def _make(self, session_id: "str | None", sticky: bool) -> ConversationSession:
        return ConversationSession(
            store=self.store,
            brain=self.brain,
            identifier=self.identifier,
            extractor=self.extractor,
            tts=self.tts,
            settings=self.settings,
            session_id=session_id,
            sticky_speaker=sticky,
        )

    @property
    def live_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def close(self) -> None:
        if self.extractor is not None:
            self.extractor.shutdown(wait=False)
        with self._lock:
            self._sessions.clear()


def _find_web_root() -> "Path | None":
    """หาโฟลเดอร์หน้าเว็บสาธิตที่ build จาก TypeScript แล้ว"""
    candidates = [
        Path(__file__).resolve().parent / "web",
        Path(__file__).resolve().parents[2] / "typescript" / "web",
        Path.cwd() / "typescript" / "web",
    ]
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return None


def _speaker_json(
    speaker: "Speaker | None", store: "MemoryStore | None" = None
) -> "dict | None":
    if speaker is None:
        return None
    data: dict = {
        "id": speaker.id,
        "name": speaker.call_name,
        "display_name": speaker.display_name,
        "nickname": speaker.nickname,
        "gender": speaker.gender,
        "particle": speaker.particle,
        "last_seen_at": speaker.last_seen_at,
    }
    if store is not None:
        stats = store.stats(speaker.id)
        data["turns"] = stats.turns
        data["facts"] = stats.facts
    return data


def create_app(settings: "Settings | None" = None, runtime: "ServerRuntime | None" = None):
    """สร้างแอป FastAPI

    ส่ง ``runtime`` เข้ามาได้เพื่อฉีดไคลเอนต์/โมเดลของตัวเอง (ใช้ในเทสต์)
    """
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, Response
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ต้องติดตั้งส่วนเซิร์ฟเวอร์ก่อน: pip install 'thaivoice[server]'"
        ) from exc

    settings = settings or get_settings()
    runtime = runtime or ServerRuntime(settings)
    store = runtime.store

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        yield
        # คืนทรัพยากรตอนปิดเซิร์ฟเวอร์ — thread pool ของตัวสกัดความจำต้องถูกปิด
        # ไม่งั้นกระบวนการอาจค้างรอเธรดตอนออก
        runtime.close()

    app = FastAPI(
        title="thaivoice",
        description="ระบบสนทนาด้วยเสียงภาษาไทยที่จดจำผู้สนทนาได้",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runtime = runtime

    # ── ตัวช่วยที่ใช้ร่วมกัน ─────────────────────────────────────────────
    def require_speaker(speaker_id: "int | None") -> "Speaker | None":
        """แปลง speaker_id เป็นผู้สนทนา — ถ้าระบุมาแล้วไม่มีจริงต้องแจ้ง 404

        ของเดิมเงียบ ๆ คืน None ซึ่งทำให้คำขอตกไปใช้ผู้พูดคนก่อนของ session นั้น
        กลายเป็นช่องให้เห็นความจำของคนอื่น
        """
        if speaker_id is None:
            return None
        found = store.get_speaker(speaker_id)
        if found is None:
            raise HTTPException(404, f"ไม่พบผู้สนทนา id={speaker_id}")
        return found

    async def read_wav(upload: "UploadFile") -> "tuple[bytes, int]":
        """อ่านและถอดไฟล์ WAV แบบไม่บล็อก event loop"""
        raw = await upload.read()
        if not raw:
            raise HTTPException(400, "ไม่มีข้อมูลเสียง")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"ไฟล์เสียงใหญ่เกินไป (จำกัด {MAX_UPLOAD_BYTES // 1024 // 1024} MB)"
            )
        try:
            # การถอด WAV เป็นงาน CPU ล้วนและรวมการรวมช่องสัญญาณเป็น mono
            # ถ้าทำบน event loop คำขออื่นทั้งหมดจะค้างรอ
            return await asyncio.to_thread(wav_to_pcm, raw)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, f"อ่านไฟล์เสียงไม่ได้: {exc}") from exc

    async def synthesize_reply(text: str) -> "str | None":
        """สังเคราะห์เสียงคำตอบหนึ่งครั้ง คืน base64"""
        if runtime.tts is None or not text.strip():
            return None
        speech = await asyncio.to_thread(runtime.tts.synthesize, text)
        if speech and speech.audio:
            return base64.b64encode(speech.audio).decode()
        return None

    # ── หน้าเว็บสาธิต ───────────────────────────────────────────────────
    web_root = _find_web_root()
    if web_root is not None:
        js_dir = web_root / "js"
        if js_dir.is_dir():
            from fastapi.staticfiles import StaticFiles

            app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        if web_root is not None and (web_root / "index.html").is_file():
            return HTMLResponse((web_root / "index.html").read_text(encoding="utf-8"))
        return HTMLResponse(
            "<h1>thaivoice</h1>"
            "<p>API พร้อมใช้งานแล้ว ดูเอกสารที่ <a href='/docs'>/docs</a></p>"
            "<p>อยากได้หน้าเว็บสาธิต ให้สร้างไฟล์ก่อนด้วย "
            "<code>cd typescript &amp;&amp; npm install &amp;&amp; npm run build</code></p>"
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "model": settings.model,
            "stt": runtime.stt.name if runtime.stt else None,
            "tts": runtime.tts.name if runtime.tts else None,
            "speaker_recognition": runtime.identifier.enabled,
            "speakers": len(store.list_speakers()),
            "live_sessions": runtime.live_sessions,
        }

    # ── คุยด้วยข้อความ ──────────────────────────────────────────────────
    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> dict:
        """คุยด้วยข้อความล้วน — ใช้เมื่อไคลเอนต์ถอดเสียงมาเองแล้ว

        ไม่มีเสียงแนบมา ระบบจึงระบุตัวผู้พูดจากลายเสียงไม่ได้ ต้องส่ง ``speaker_id``
        มาเอง หรือให้ผู้ใช้บอกชื่อในประโยค
        """
        text = req.text.strip()
        if not text:
            raise HTTPException(422, "ข้อความว่างเปล่า")

        session = runtime.session(req.session_id)
        speaker = require_speaker(req.speaker_id)
        result = await asyncio.to_thread(
            session.exchange, text, None, speaker, False
        )
        return {
            "reply": result.reply,
            "chunks": result.chunks,
            "speaker": _speaker_json(result.speaker, store),
            "session_id": session.session_id,
            "audio": await synthesize_reply(result.reply) if req.speak else None,
        }

    # ── คุยด้วยเสียง (อัปโหลดไฟล์) ──────────────────────────────────────
    @app.post("/api/voice")
    async def voice(
        audio: UploadFile = File(..., description="ไฟล์ WAV 16-bit mono"),
        transcript: str = Form("", description="ข้อความที่ไคลเอนต์ถอดมาแล้ว (ถ้ามี)"),
        session_id: str = Form(""),
        speaker_id: str = Form(""),
        speak: bool = Form(True),
    ) -> dict:
        """หนึ่งเทิร์นเต็ม: รับเสียง -> รู้ว่าใครพูด -> ถอดเสียง -> ตอบ -> พูดกลับ"""
        pcm, rate = await read_wav(audio)

        text = transcript.strip()[:MAX_TEXT_CHARS]
        if not text:
            if runtime.stt is None or runtime.stt.name == "external":
                raise HTTPException(
                    400,
                    "เซิร์ฟเวอร์ไม่ได้เปิดการถอดเสียง — ส่งข้อความที่ถอดแล้วมาในฟิลด์ transcript",
                )
            # ถอดเสียงเป็นงานหนักมาก (Whisper large-v3 ใช้เวลาหลายวินาทีบน CPU)
            # ถ้าเรียกตรง ๆ บน event loop ทุกคำขออื่นจะค้างรอตลอดช่วงนั้น
            result = await asyncio.to_thread(runtime.stt.transcribe, pcm, rate)
            text = result.text
        if not text:
            raise HTTPException(422, "ถอดเสียงไม่ได้ความ ลองพูดใหม่อีกครั้ง")

        chosen: "Speaker | None" = None
        if speaker_id.strip():
            try:
                chosen = require_speaker(int(speaker_id))
            except ValueError as exc:
                raise HTTPException(422, "speaker_id ต้องเป็นตัวเลข") from exc

        session = runtime.session(session_id or None)
        # ส่ง rate จริงของไฟล์ไปด้วย ไม่งั้นลายเสียงจะถูกคำนวณผิดอัตราสุ่มตัวอย่าง
        result = await asyncio.to_thread(
            lambda: session.exchange(
                text, pcm, chosen, speak=False, sample_rate=rate
            )
        )
        return {
            "transcript": text,
            "reply": result.reply,
            "chunks": result.chunks,
            "speaker": _speaker_json(result.speaker, store),
            "identified_by": (
                result.identification.method
                if result.identification and result.identification.speaker is not None
                else None
            ),
            # สังเคราะห์เสียงครั้งเดียวตอนท้าย ของเดิมสังเคราะห์ทุกท่อนระหว่างคุย
            # แล้วทิ้ง จากนั้นสังเคราะห์คำตอบเต็มอีกรอบ รวมเป็น 3 ครั้งต่อหนึ่งเทิร์น
            "audio": await synthesize_reply(result.reply) if speak else None,
            "session_id": session.session_id,
        }

    # ── บริการเดี่ยว ────────────────────────────────────────────────────
    @app.post("/api/stt")
    async def transcribe(audio: UploadFile = File(...)) -> dict:
        if runtime.stt is None or runtime.stt.name == "external":
            raise HTTPException(400, "เซิร์ฟเวอร์ไม่ได้เปิดการถอดเสียง")
        pcm, rate = await read_wav(audio)
        result = await asyncio.to_thread(runtime.stt.transcribe, pcm, rate)
        return {"text": result.text, "confidence": result.confidence}

    @app.post("/api/tts")
    async def synthesize(req: SpeakRequest):
        if runtime.tts is None:
            raise HTTPException(400, "เซิร์ฟเวอร์ไม่ได้เปิดการสังเคราะห์เสียง")
        speech = await asyncio.to_thread(runtime.tts.synthesize, req.text, req.voice)
        if not speech or not speech.audio:
            raise HTTPException(400, "สังเคราะห์เสียงไม่สำเร็จ")
        return Response(content=speech.audio, media_type=speech.mime)

    @app.post("/api/identify")
    async def identify(audio: UploadFile = File(...)) -> dict:
        pcm, rate = await read_wav(audio)
        ident = await asyncio.to_thread(runtime.identifier.identify, pcm, rate)
        return {
            "speaker": _speaker_json(ident.speaker, store),
            "score": ident.score,
            "is_new": ident.is_new,
            "method": ident.method,
        }

    # ── จัดการผู้สนทนาและความจำ ─────────────────────────────────────────
    @app.get("/api/speakers")
    def list_speakers() -> dict:
        return {"speakers": [_speaker_json(s, store) for s in store.list_speakers()]}

    @app.post("/api/speakers")
    def create_speaker(req: NewSpeakerRequest) -> dict:
        from .thai_text import particle_for_gender

        name = " ".join(req.name.split())
        if not name:
            raise HTTPException(422, "ชื่อว่างเปล่า")
        speaker, created = store.get_or_create_speaker(
            name, gender=req.gender, particle=particle_for_gender(req.gender)
        )
        return {"speaker": _speaker_json(speaker, store), "created": created}

    @app.get("/api/speakers/{speaker_id}")
    def speaker_detail(speaker_id: int) -> dict:
        speaker = require_speaker(speaker_id)
        assert speaker is not None
        summary = store.latest_summary(speaker_id)
        return {
            "speaker": _speaker_json(speaker, store),
            "facts": [
                {
                    "key": f.key,
                    "value": f.value,
                    "category": f.category,
                    "confidence": f.confidence,
                }
                for f in store.facts_for(speaker_id)
            ],
            "summary": summary[0] if summary else None,
            "recent_turns": [
                {"role": t.role, "content": t.content, "at": t.created_at}
                for t in store.recent_turns(speaker_id, limit=20)
            ],
        }

    @app.post("/api/speakers/{speaker_id}/enroll")
    async def enroll(speaker_id: int, audio: UploadFile = File(...)) -> dict:
        speaker = require_speaker(speaker_id)
        assert speaker is not None
        if not runtime.identifier.enabled:
            raise HTTPException(400, "ยังไม่ได้เปิดการจำลายเสียงบนเซิร์ฟเวอร์นี้")
        pcm, rate = await read_wav(audio)
        ok = await asyncio.to_thread(runtime.identifier.enroll, speaker, pcm, rate)
        return {"ok": ok, "speaker": _speaker_json(speaker, store)}

    @app.delete("/api/speakers/{speaker_id}")
    def delete_speaker(speaker_id: int) -> dict:
        return {"deleted": store.delete_speaker(speaker_id)}

    @app.delete("/api/speakers/{speaker_id}/facts")
    def forget_facts(speaker_id: int) -> dict:
        """ลบเฉพาะข้อเท็จจริงที่จำไว้ (บทสนทนาและบทสรุปยังอยู่)"""
        require_speaker(speaker_id)
        return {"removed": store.forget_all_facts(speaker_id)}

    @app.delete("/api/speakers/{speaker_id}/memory")
    def forget_memory(speaker_id: int) -> dict:
        """ลบความจำทั้งหมด — ข้อเท็จจริง บทสรุป และบทสนทนา แต่ยังรู้จักตัวคนอยู่"""
        require_speaker(speaker_id)
        return {"removed": store.forget_everything(speaker_id)}

    # ── WebSocket แบบสตรีม ──────────────────────────────────────────────
    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        """สตรีมบทสนทนา

        ไคลเอนต์ส่ง::

            {"text": "สวัสดีครับ", "speaker_id": 3, "audio": "<wav base64>", "speak": true}

        เซิร์ฟเวอร์ส่งกลับเป็นชุดเหตุการณ์ ``speaker`` / ``delta`` / ``chunk`` / ``done``
        โดย ``chunk`` จะมีเสียง base64 มาด้วยถ้าเปิด TTS ไว้

        ทุกความผิดพลาดจะถูกตอบกลับเป็นเหตุการณ์ ``error`` เสมอ ห้ามปิดการเชื่อมต่อ
        เงียบ ๆ เพราะไคลเอนต์จะค้างรอคำตอบที่ไม่มีวันมาตลอดกาล
        """
        await websocket.accept()
        session = runtime.session(websocket.query_params.get("session_id"))

        async def fail(message: str) -> None:
            try:
                await websocket.send_json({"type": "error", "text": message})
            except Exception:
                pass

        try:
            while True:
                try:
                    message = await websocket.receive_json()
                except Exception:
                    # ไม่ใช่ JSON หรือการเชื่อมต่อปิดไปแล้ว
                    break
                if not isinstance(message, dict):
                    await fail("รูปแบบข้อความไม่ถูกต้อง ต้องเป็นอ็อบเจ็กต์ JSON")
                    continue

                raw_text = message.get("text")
                text = raw_text.strip() if isinstance(raw_text, str) else ""
                if not text:
                    await fail("ไม่มีข้อความ")
                    continue
                if len(text) > MAX_TEXT_CHARS:
                    await fail(f"ข้อความยาวเกิน {MAX_TEXT_CHARS} ตัวอักษร")
                    continue

                pcm = None
                rate = None
                if message.get("audio"):
                    try:
                        pcm, rate = await asyncio.to_thread(
                            lambda: wav_to_pcm(base64.b64decode(message["audio"]))
                        )
                    except Exception:
                        log.info("ถอดรหัสเสียงจาก WebSocket ไม่ได้", exc_info=True)
                        pcm, rate = None, None

                speaker = None
                if message.get("speaker_id") is not None:
                    try:
                        speaker = store.get_speaker(int(message["speaker_id"]))
                    except (TypeError, ValueError):
                        await fail("speaker_id ต้องเป็นตัวเลข")
                        continue
                    if speaker is None:
                        await fail("ไม่พบผู้สนทนาตาม speaker_id ที่ส่งมา")
                        continue

                speak = bool(message.get("speak", True))
                try:
                    async for event in _stream_async(
                        lambda: session.stream_exchange(
                            text, pcm=pcm, speaker=speaker, speak=speak, sample_rate=rate
                        )
                    ):
                        payload: dict = {"type": event.type, "text": event.text}
                        if event.speaker is not None:
                            payload["speaker"] = _speaker_json(event.speaker, store)
                        if event.identification is not None:
                            payload["identified_by"] = event.identification.method
                            payload["score"] = event.identification.score
                        if event.speech is not None and event.speech.audio:
                            payload["audio"] = base64.b64encode(event.speech.audio).decode()
                            payload["mime"] = event.speech.mime
                        await websocket.send_json(payload)
                except Exception as exc:
                    log.exception("ประมวลผลเทิร์นไม่สำเร็จ")
                    await fail(f"ประมวลผลไม่สำเร็จ: {exc}")
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    return app


async def _stream_async(make_iterator):
    """รัน generator ที่บล็อกในเธรดแยก แล้วส่งออกมาแบบ async

    งานเรียกโมเดลและสังเคราะห์เสียงเป็นงานบล็อก ถ้ารันบน event loop ตรง ๆ
    จะทำให้ WebSocket ตัวอื่นค้างทั้งหมด

    เมื่อผู้บริโภคเลิกอ่านกลางคัน (ไคลเอนต์หลุด) เธรดต้องหยุดด้วย ไม่งั้นมันจะ
    เดินหน้าเรียกโมเดลและสังเคราะห์เสียงจนจบใส่คิวที่ไม่มีใครอ่าน กินทั้งเธรดใน
    executor และโควตา API ไปเปล่า ๆ
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancelled = threading.Event()
    sentinel = object()

    def push(item: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            # event loop ปิดไปแล้วระหว่างที่เธรดยังทำงานอยู่
            cancelled.set()

    def worker() -> None:
        try:
            for item in make_iterator():
                if cancelled.is_set():
                    return
                push(item)
        except Exception as exc:
            if not cancelled.is_set():
                push(exc)
        finally:
            push(sentinel)

    loop.run_in_executor(None, worker)

    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        cancelled.set()


def run(host: "str | None" = None, port: "int | None" = None) -> None:  # pragma: no cover
    """เปิดเซิร์ฟเวอร์ด้วย uvicorn"""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host=host or settings.host,
        port=port or settings.port,
        log_level="info",
    )
