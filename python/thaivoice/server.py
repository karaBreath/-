"""เซิร์ฟเวอร์ HTTP + WebSocket สำหรับไคลเอนต์ภายนอก (เว็บ, แอป, LINE bot)

จุดสำคัญของการออกแบบ:

* **สตรีมผ่าน WebSocket** — ส่งทั้งข้อความและเสียงกลับทีละประโยค ไคลเอนต์จึง
  เริ่มเล่นเสียงประโยคแรกได้ก่อนที่คำตอบจะพิมพ์จบ
* **เลือกได้ว่าจะถอดเสียงที่ไหน** — เบราว์เซอร์สมัยใหม่ถอดเสียงไทย (th-TH) ได้เอง
  ด้วย Web Speech API ซึ่งเร็วและไม่ต้องอัปโหลดเสียง แต่ถ้าอยากให้ทุกอย่างอยู่ใน
  เครื่องตัวเอง ก็ส่งไฟล์เสียงมาที่ ``/api/voice`` ให้เซิร์ฟเวอร์ถอดแทนได้
* **แนบเสียงมาด้วยเสมอถ้าทำได้** — เสียงคือสิ่งที่ทำให้ระบบรู้ว่า "ใครพูด"
  ถ้าส่งมาแต่ข้อความ ระบบจะจำผู้พูดจากลายเสียงไม่ได้
"""

# หมายเหตุ: โมดูลนี้ตั้งใจ *ไม่* ใช้ ``from __future__ import annotations``
# เพราะ FastAPI ต้องอ่านชนิดของพารามิเตอร์จริง ๆ เพื่อสร้าง schema ถ้า annotation
# กลายเป็นสตริง FastAPI จะไปหาชื่อนั้นใน globals ของโมดูล แล้วหา UploadFile /
# WebSocket ที่ import ไว้ในฟังก์ชันไม่เจอ ทำให้ทุก endpoint ที่รับไฟล์พังหมด

import asyncio
import base64
import logging
from typing import Any

from pydantic import BaseModel, Field

from . import create_session
from .config import Settings, get_settings
from .memory import MemoryStore, Speaker
from .session import ConversationSession
from .stt import load_stt, wav_to_pcm

log = logging.getLogger("thaivoice.server")

__all__ = ["create_app", "run", "ChatRequest", "SpeakRequest", "NewSpeakerRequest"]


class ChatRequest(BaseModel):
    """คุยด้วยข้อความล้วน"""

    text: str = Field(description="สิ่งที่ผู้ใช้พูดหรือพิมพ์")
    session_id: str | None = None
    speaker_id: int | None = Field(default=None, description="ระบุตัวผู้พูดไว้ล่วงหน้า")
    speak: bool = Field(default=True, description="ให้เซิร์ฟเวอร์สังเคราะห์เสียงคำตอบด้วยไหม")


class SpeakRequest(BaseModel):
    """ขอให้สังเคราะห์เสียงจากข้อความ"""

    text: str
    voice: str | None = None


class NewSpeakerRequest(BaseModel):
    """สร้างผู้สนทนาใหม่"""

    name: str
    gender: str | None = None


def _find_web_root():
    """หาโฟลเดอร์หน้าเว็บสาธิตที่ build จาก TypeScript แล้ว"""
    from pathlib import Path

    candidates = [
        Path(__file__).resolve().parent / "web",
        Path(__file__).resolve().parents[2] / "typescript" / "web",
        Path.cwd() / "typescript" / "web",
    ]
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return None


def _speaker_json(speaker: "Speaker | None", store: "MemoryStore | None" = None) -> "dict | None":
    if speaker is None:
        return None
    data: dict[str, Any] = {
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


def create_app(settings: "Settings | None" = None):
    """สร้างแอป FastAPI"""
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, Response
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ต้องติดตั้งส่วนเซิร์ฟเวอร์ก่อน: pip install 'thaivoice[server]'"
        ) from exc

    settings = settings or get_settings()
    store = MemoryStore(settings.db_path)
    stt = load_stt(settings)

    # หนึ่ง session ต่อหนึ่งบทสนทนา — เก็บไว้เพื่อจำว่าใครคุยอยู่ในห้องนั้น
    sessions: dict[str, ConversationSession] = {}

    def get_session(session_id: str | None) -> ConversationSession:
        key = session_id or "default"
        if key not in sessions:
            sessions[key] = create_session(settings, store=store, session_id=key)
        return sessions[key]

    app = FastAPI(
        title="thaivoice",
        description="ระบบสนทนาด้วยเสียงภาษาไทยที่จดจำผู้สนทนาได้",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── หน้าเว็บสาธิต ───────────────────────────────────────────────────
    web_root = _find_web_root()
    if web_root is not None:
        # เสิร์ฟ ES module ที่คอมไพล์จาก TypeScript (typescript/web/js)
        from fastapi.staticfiles import StaticFiles

        js_dir = web_root / "js"
        if js_dir.is_dir():
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
            "stt": stt.name if stt else None,
            "speaker_recognition": get_session(None).identifier.enabled,
            "speakers": len(store.list_speakers()),
        }

    # ── คุยด้วยข้อความ ──────────────────────────────────────────────────
    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict:
        """คุยด้วยข้อความล้วน — ใช้เมื่อไคลเอนต์ถอดเสียงมาเองแล้ว

        หมายเหตุ: ไม่มีเสียงแนบมา ระบบจึงระบุตัวผู้พูดจากลายเสียงไม่ได้
        ต้องส่ง ``speaker_id`` มาเอง หรือให้ผู้ใช้บอกชื่อในประโยค
        """
        session = get_session(req.session_id)
        speaker = store.get_speaker(req.speaker_id) if req.speaker_id else None
        result = session.exchange(req.text, speaker=speaker, speak=req.speak)
        return {
            "reply": result.reply,
            "chunks": result.chunks,
            "speaker": _speaker_json(result.speaker, store),
            "session_id": session.session_id,
        }

    # ── คุยด้วยเสียง (อัปโหลดไฟล์) ──────────────────────────────────────
    @app.post("/api/voice")
    async def voice(
        audio: UploadFile = File(..., description="ไฟล์ WAV 16-bit mono"),
        transcript: str = Form("", description="ข้อความที่ไคลเอนต์ถอดมาแล้ว (ถ้ามี)"),
        session_id: str = Form(""),
        speak: bool = Form(True),
    ) -> dict:
        """หนึ่งเทิร์นเต็ม: รับเสียง -> รู้ว่าใครพูด -> ถอดเสียง -> ตอบ -> พูดกลับ"""
        raw = await audio.read()
        try:
            pcm, rate = wav_to_pcm(raw)
        except Exception as exc:
            raise HTTPException(400, f"อ่านไฟล์เสียงไม่ได้: {exc}") from exc

        text = transcript.strip()
        if not text:
            if stt is None or stt.name == "external":
                raise HTTPException(
                    400,
                    "เซิร์ฟเวอร์ไม่ได้เปิดการถอดเสียง — ส่งข้อความที่ถอดแล้วมาในฟิลด์ transcript",
                )
            text = stt.transcribe(pcm, rate).text
        if not text:
            raise HTTPException(422, "ถอดเสียงไม่ได้ความ ลองพูดใหม่อีกครั้ง")

        session = get_session(session_id or None)
        result = await asyncio.to_thread(
            session.exchange, text, pcm, None, speak
        )
        audio_b64 = None
        if speak and session.tts is not None:
            speech = await asyncio.to_thread(session.tts.synthesize, result.reply)
            if speech and speech.audio:
                audio_b64 = base64.b64encode(speech.audio).decode()
        return {
            "transcript": text,
            "reply": result.reply,
            "chunks": result.chunks,
            "speaker": _speaker_json(result.speaker, store),
            "identified_by": result.identification.method if result.identification else None,
            "audio": audio_b64,
            "session_id": session.session_id,
        }

    # ── บริการเดี่ยว ────────────────────────────────────────────────────
    @app.post("/api/stt")
    async def transcribe(audio: UploadFile = File(...)) -> dict:
        if stt is None or stt.name == "external":
            raise HTTPException(400, "เซิร์ฟเวอร์ไม่ได้เปิดการถอดเสียง")
        pcm, rate = wav_to_pcm(await audio.read())
        result = await asyncio.to_thread(stt.transcribe, pcm, rate)
        return {"text": result.text, "confidence": result.confidence}

    @app.post("/api/tts")
    async def synthesize(req: SpeakRequest):
        session = get_session(None)
        if session.tts is None:
            raise HTTPException(400, "เซิร์ฟเวอร์ไม่ได้เปิดการสังเคราะห์เสียง")
        speech = await asyncio.to_thread(session.tts.synthesize, req.text, req.voice)
        if not speech or not speech.audio:
            raise HTTPException(400, "สังเคราะห์เสียงไม่สำเร็จ")
        return Response(content=speech.audio, media_type=speech.mime)

    @app.post("/api/identify")
    async def identify(audio: UploadFile = File(...)) -> dict:
        session = get_session(None)
        pcm, rate = wav_to_pcm(await audio.read())
        ident = await asyncio.to_thread(session.identifier.identify, pcm, rate)
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

        existing = store.find_speaker_by_name(req.name)
        speaker = existing or store.create_speaker(
            req.name, gender=req.gender, particle=particle_for_gender(req.gender)
        )
        return {"speaker": _speaker_json(speaker, store), "created": existing is None}

    @app.get("/api/speakers/{speaker_id}")
    def speaker_detail(speaker_id: int) -> dict:
        speaker = store.get_speaker(speaker_id)
        if speaker is None:
            raise HTTPException(404, "ไม่พบผู้สนทนาคนนี้")
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
        speaker = store.get_speaker(speaker_id)
        if speaker is None:
            raise HTTPException(404, "ไม่พบผู้สนทนาคนนี้")
        session = get_session(None)
        if not session.identifier.enabled:
            raise HTTPException(400, "ยังไม่ได้เปิดการจำลายเสียงบนเซิร์ฟเวอร์นี้")
        pcm, rate = wav_to_pcm(await audio.read())
        ok = await asyncio.to_thread(session.identifier.enroll, speaker, pcm, rate)
        return {"ok": ok, "speaker": _speaker_json(speaker, store)}

    @app.delete("/api/speakers/{speaker_id}")
    def delete_speaker(speaker_id: int) -> dict:
        return {"deleted": store.delete_speaker(speaker_id)}

    @app.delete("/api/speakers/{speaker_id}/facts")
    def forget_facts(speaker_id: int) -> dict:
        return {"removed": store.forget_all_facts(speaker_id)}

    # ── WebSocket แบบสตรีม ──────────────────────────────────────────────
    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        """สตรีมบทสนทนา

        ไคลเอนต์ส่ง::

            {"text": "สวัสดีครับ", "speaker_id": 3, "audio": "<wav base64>", "speak": true}

        เซิร์ฟเวอร์ส่งกลับเป็นชุดเหตุการณ์ ``speaker`` / ``delta`` / ``chunk`` / ``done``
        โดย ``chunk`` จะมีเสียง base64 มาด้วยถ้าเปิด TTS ไว้
        """
        await websocket.accept()
        session = get_session(websocket.query_params.get("session_id"))
        try:
            while True:
                message = await websocket.receive_json()
                text = (message.get("text") or "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "text": "ไม่มีข้อความ"})
                    continue

                pcm = None
                if message.get("audio"):
                    try:
                        pcm, _rate = wav_to_pcm(base64.b64decode(message["audio"]))
                    except Exception:
                        log.warning("ถอดรหัสเสียงจาก WebSocket ไม่ได้", exc_info=True)

                speaker = (
                    store.get_speaker(int(message["speaker_id"]))
                    if message.get("speaker_id")
                    else None
                )
                speak = bool(message.get("speak", True))

                async for event in _stream_async(
                    lambda: session.stream_exchange(text, pcm=pcm, speaker=speaker, speak=speak)
                ):
                    payload: dict[str, Any] = {"type": event.type, "text": event.text}
                    if event.speaker is not None:
                        payload["speaker"] = _speaker_json(event.speaker, store)
                    if event.identification is not None:
                        payload["identified_by"] = event.identification.method
                        payload["score"] = event.identification.score
                    if event.speech is not None and event.speech.audio:
                        payload["audio"] = base64.b64encode(event.speech.audio).decode()
                        payload["mime"] = event.speech.mime
                    await websocket.send_json(payload)
        except Exception:
            log.info("ปิดการเชื่อมต่อ WebSocket", exc_info=True)

    return app


async def _stream_async(make_iterator):
    """รัน generator ที่บล็อกในเธรดแยก แล้วส่งออกมาแบบ async

    งานเรียกโมเดลและสังเคราะห์เสียงเป็นงานบล็อก ถ้ารันบน event loop ตรง ๆ
    จะทำให้ WebSocket ตัวอื่นค้างทั้งหมด
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def worker() -> None:
        try:
            for item in make_iterator():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # ส่งข้อผิดพลาดกลับไปให้ผู้เรียกจัดการ
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    loop.run_in_executor(None, worker)

    while True:
        item = await queue.get()
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item


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
