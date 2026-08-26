"""เครื่องมือบรรทัดคำสั่ง — คุยด้วยเสียง คุยด้วยข้อความ และจัดการความจำ

คำสั่งที่มี::

    thaivoice doctor            ตรวจว่าติดตั้งอะไรครบแล้วบ้าง
    thaivoice talk              คุยด้วยเสียงผ่านไมโครโฟน
    thaivoice chat              คุยด้วยการพิมพ์ (ใช้ทดสอบตอนไม่มีไมค์)
    thaivoice speakers          ดูรายชื่อคนที่ระบบรู้จัก
    thaivoice memory <id>       ดูความจำของคนคนนั้น
    thaivoice enroll --name X --wav a.wav   สอนให้จำเสียงคนใหม่
    thaivoice forget <id>       ลบคนนั้นและความจำทั้งหมด
    thaivoice serve             เปิดเซิร์ฟเวอร์ HTTP
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import unicodedata
from pathlib import Path

from . import create_session
from .audio import AudioPlayer, Microphone, audio_available
from .brain import MissingCredentialsError
from .config import get_settings, load_dotenv
from .memory import MemoryStore
from .prompts import human_delta_th
from .stt import load_stt, wav_to_pcm

# สีสำหรับเทอร์มินัล — ปิดเองเมื่อไม่ได้ต่อ TTY
_TTY = sys.stdout.isatty()
DIM = "\033[2m" if _TTY else ""
BOLD = "\033[1m" if _TTY else ""
CYAN = "\033[36m" if _TTY else ""
GREEN = "\033[32m" if _TTY else ""
YELLOW = "\033[33m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""


def _p(text: str = "") -> None:
    print(text, flush=True)


def _width(text: str) -> int:
    """ความกว้างที่แสดงจริงบนเทอร์มินัล

    ภาษาไทยมีสระบนสระล่างและวรรณยุกต์ที่เป็นอักขระผสม (combining mark) ซึ่งกิน
    ตำแหน่งใน string แต่ไม่กินความกว้างบนจอ ถ้าใช้ len() จัดคอลัมน์ ตารางจะเบี้ยว
    """
    return sum(0 if unicodedata.combining(ch) else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """เติมช่องว่างให้ครบความกว้างที่แสดงจริง"""
    return text + " " * max(0, width - _width(text))


# ── doctor ──────────────────────────────────────────────────────────────────
def cmd_doctor(args: argparse.Namespace) -> int:
    """บอกว่าอะไรพร้อมใช้แล้ว และอะไรยังขาด พร้อมคำสั่งติดตั้ง"""
    settings = get_settings()
    # การโหลด backend ที่ยังไม่ได้ติดตั้งเป็นเรื่องปกติของ doctor — ไม่ต้องพ่น traceback
    logging.getLogger("thaivoice").setLevel(logging.CRITICAL)
    _p(f"{BOLD}ตรวจสภาพระบบ thaivoice{RESET}\n")

    def line(label: str, ok: bool, detail: str = "", fix: str = "") -> None:
        mark = f"{GREEN}พร้อม{RESET}" if ok else f"{YELLOW}ยังไม่พร้อม{RESET}"
        _p(f"  {_pad(label, 26)} {mark}  {DIM}{detail}{RESET}")
        if not ok and fix:
            _p(f"  {'':<26} {DIM}วิธีแก้: {fix}{RESET}")

    import os

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    line(
        "Claude API",
        has_key,
        "พบคีย์ใน environment" if has_key else "ไม่พบ ANTHROPIC_API_KEY",
        "export ANTHROPIC_API_KEY=... หรือรัน `ant auth login`",
    )

    mic_ok, mic_why = audio_available()
    line("ไมโครโฟน", mic_ok, mic_why or "ใช้ได้", "pip install 'thaivoice[audio]'")

    player = AudioPlayer()
    line(
        "ลำโพง",
        player.available,
        "ใช้โปรแกรมเล่นเสียงของระบบ" if player.available else "ไม่พบ ffplay/mpv/mpg123",
        "ติดตั้ง ffmpeg (มี ffplay มาด้วย)",
    )

    stt = load_stt(settings)
    line(
        f"ถอดเสียง ({settings.stt_backend})",
        stt is not None,
        type(stt).__name__ if stt else "โหลด backend ไม่ได้",
        "pip install 'thaivoice[stt]'",
    )

    from .tts import load_tts

    tts = load_tts(settings)
    line(
        f"สังเคราะห์เสียง ({settings.tts_backend})",
        tts is not None and tts.name not in {"none"},
        type(tts).__name__ if tts else "-",
        "pip install 'thaivoice[tts]'",
    )

    from .speaker import load_embedder

    embedder = load_embedder(settings.speaker_backend)
    line(
        f"จำลายเสียง ({settings.speaker_backend})",
        embedder is not None,
        "จำได้ว่าใครพูด" if embedder else "จะใช้การบอกชื่อแทน",
        "pip install 'thaivoice[speaker]'",
    )

    try:
        import pythainlp  # type: ignore # noqa: F401

        thai_ok = True
    except Exception:
        thai_ok = False
    line(
        "ตัดประโยคไทย",
        thai_ok,
        "ใช้ pythainlp" if thai_ok else "ใช้กฎสำรอง (ยังทำงานได้)",
        "pip install 'thaivoice[thai]'",
    )

    store = MemoryStore(settings.db_path)
    speakers = store.list_speakers()
    _p(f"\n  {DIM}ความจำ: {settings.db_path} — รู้จักอยู่ {len(speakers)} คน{RESET}")
    store.close()
    _p(f"\n  {DIM}โมเดล: {settings.model} (effort={settings.effort}){RESET}")
    return 0


# ── คุยด้วยข้อความ ──────────────────────────────────────────────────────────
def cmd_chat(args: argparse.Namespace) -> int:
    """โหมดพิมพ์คุย — ใช้ทดสอบสมองและความจำโดยไม่ต้องมีไมโครโฟน"""
    session = create_session(with_tts=args.speak)
    speaker = None
    if args.speaker:
        speaker = session.store.find_speaker_by_name(args.speaker) or session.register_speaker(
            args.speaker
        )
        session.current_speaker = speaker

    _p(f"{BOLD}โหมดพิมพ์คุย{RESET} {DIM}(ออกด้วย Ctrl-C หรือพิมพ์ /ออก){RESET}")
    if speaker:
        _p(f"{DIM}คุยในนามของ: {speaker.call_name}{RESET}")
    _p()

    player = AudioPlayer() if args.speak else None
    try:
        while True:
            try:
                text = input(f"{CYAN}คุณ:{RESET} ").strip()
            except EOFError:
                break
            if not text or text in {"/ออก", "/quit", "/exit"}:
                break

            printed_name = False
            for event in session.stream_exchange(text, speak=args.speak):
                if event.type == "speaker" and not printed_name:
                    printed_name = True
                    who = event.speaker.call_name if event.speaker else "ยังไม่รู้จัก"
                    method = event.identification.method if event.identification else "-"
                    _p(f"{DIM}[ผู้พูด: {who} · วิธีระบุ: {method}]{RESET}")
                    print(f"{GREEN}บอท:{RESET} ", end="", flush=True)
                elif event.type == "delta":
                    print(event.text, end="", flush=True)
                elif event.type == "chunk" and player and event.speech and event.speech.audio:
                    player.play(event.speech.audio)
                elif event.type == "done":
                    _p("\n")
    except KeyboardInterrupt:
        _p("\nลาก่อนครับ")
    return 0


# ── คุยด้วยเสียง ────────────────────────────────────────────────────────────
def cmd_talk(args: argparse.Namespace) -> int:
    """โหมดคุยด้วยเสียงเต็มรูปแบบ"""
    settings = get_settings()

    ok, why = audio_available()
    if not ok:
        _p(f"{YELLOW}ใช้ไมโครโฟนไม่ได้: {why}{RESET}")
        _p("ติดตั้งด้วย: pip install 'thaivoice[audio]'")
        _p("หรือใช้โหมดพิมพ์คุยแทน: thaivoice chat")
        return 1

    stt = load_stt(settings)
    if stt is None or stt.name == "external":
        _p(f"{YELLOW}ยังไม่มีตัวถอดเสียงที่ใช้ได้ (backend={settings.stt_backend}){RESET}")
        _p("ติดตั้งด้วย: pip install 'thaivoice[stt]'")
        return 1

    session = create_session()
    mic = Microphone(settings)
    player = AudioPlayer()

    _p(f"{BOLD}เริ่มคุยได้เลยครับ{RESET} {DIM}(กด Ctrl-C เพื่อออก){RESET}")
    if not session.identifier.enabled:
        _p(f"{DIM}หมายเหตุ: ยังไม่ได้เปิดการจำลายเสียง ระบบจะรู้จักคุณจากการบอกชื่อ{RESET}")
    _p()

    try:
        while True:
            _p(f"{DIM}กำลังฟัง...{RESET}")
            pcm = mic.record_utterance()
            if not pcm:
                continue

            started = time.time()
            result = stt.transcribe(pcm, settings.sample_rate)
            if not result:
                _p(f"{DIM}(ไม่ได้ยินเป็นคำพูด){RESET}")
                continue
            _p(f"{CYAN}คุณ:{RESET} {result.text} {DIM}({time.time() - started:.1f} วิ){RESET}")

            printed = False
            for event in session.stream_exchange(result.text, pcm=pcm):
                if event.type == "speaker":
                    who = event.speaker.call_name if event.speaker else "ยังไม่รู้จัก"
                    score = event.identification.score if event.identification else 0.0
                    detail = f" · ความคล้ายเสียง {score:.2f}" if score else ""
                    _p(f"{DIM}[ผู้พูด: {who}{detail}]{RESET}")
                elif event.type == "delta":
                    if not printed:
                        print(f"{GREEN}บอท:{RESET} ", end="", flush=True)
                        printed = True
                    print(event.text, end="", flush=True)
                elif event.type == "chunk" and event.speech and event.speech.audio:
                    player.play(event.speech.audio)
                elif event.type == "done":
                    _p("\n")
    except KeyboardInterrupt:
        _p(f"\n{DIM}ลาก่อนครับ{RESET}")
        mic.stop()
        player.stop()
    return 0


# ── จัดการความจำ ────────────────────────────────────────────────────────────
def cmd_speakers(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = MemoryStore(settings.db_path)
    speakers = store.list_speakers()
    if not speakers:
        _p("ยังไม่รู้จักใครเลยครับ")
        return 0
    header = (
        _pad("id", 5) + _pad("ชื่อ", 20) + _pad("คำลงท้าย", 12)
        + _pad("เทิร์น", 9) + _pad("ความจำ", 9) + "เจอล่าสุด"
    )
    _p(f"{BOLD}{header}{RESET}")
    now = time.time()
    for speaker in speakers:
        stats = store.stats(speaker.id)
        last = human_delta_th(now - speaker.last_seen_at)
        _p(
            _pad(str(speaker.id), 5)
            + _pad(speaker.call_name, 20)
            + _pad(speaker.particle or "-", 12)
            + _pad(str(stats.turns), 9)
            + _pad(str(stats.facts), 9)
            + last
        )
    store.close()
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = MemoryStore(settings.db_path)
    speaker = store.get_speaker(args.speaker_id)
    if speaker is None:
        _p(f"ไม่พบผู้สนทนา id={args.speaker_id}")
        return 1

    stats = store.stats(speaker.id)
    _p(f"{BOLD}{speaker.call_name}{RESET} {DIM}(id={speaker.id}){RESET}")
    _p(f"  ชื่อเต็ม: {speaker.display_name}")
    _p(f"  คำลงท้าย: {speaker.particle or '-'} · เพศที่เดาไว้: {speaker.gender or '-'}")
    _p(f"  คุยกันมาแล้ว {stats.turns} เทิร์น · จำข้อเท็จจริงไว้ {stats.facts} ข้อ")
    _p(f"  ลายเสียง: {'มี' if store.has_voiceprint(speaker.id, 'resemblyzer') else 'ยังไม่มี'}")

    facts = store.facts_for(speaker.id)
    if facts:
        _p(f"\n{BOLD}สิ่งที่จำได้{RESET}")
        for fact in facts:
            _p(f"  [{fact.category}] {fact.key}: {fact.value} {DIM}({fact.confidence:.2f}){RESET}")

    summary = store.latest_summary(speaker.id)
    if summary:
        _p(f"\n{BOLD}สรุปบทสนทนา{RESET}\n  {summary[0]}")

    if args.turns:
        _p(f"\n{BOLD}บทสนทนาล่าสุด{RESET}")
        for turn in store.recent_turns(speaker.id, limit=args.turns):
            who = "คุณ" if turn.role == "user" else "บอท"
            _p(f"  {who}: {turn.content}")
    store.close()
    return 0


def cmd_enroll(args: argparse.Namespace) -> int:
    """สอนให้ระบบจำเสียงคนหนึ่งจากไฟล์ WAV"""
    settings = get_settings()
    session = create_session(with_tts=False, with_memory_extraction=False)
    if not session.identifier.enabled:
        _p(f"{YELLOW}ยังใช้การจำลายเสียงไม่ได้ — ติดตั้งด้วย pip install 'thaivoice[speaker]'{RESET}")
        return 1

    total = 0
    for path in args.wav:
        data = Path(path).read_bytes()
        pcm, rate = wav_to_pcm(data)
        speaker = session.register_speaker(args.name, gender=args.gender)
        if session.identifier.enroll(speaker, pcm, rate):
            total += 1
            _p(f"{GREEN}เพิ่มตัวอย่างเสียงจาก {path} แล้ว{RESET}")
        else:
            _p(f"{YELLOW}ใช้ไฟล์ {path} ไม่ได้{RESET}")
    if total:
        speaker = session.store.find_speaker_by_name(args.name)
        _p(f"\nจำเสียงของ {args.name} แล้วครับ (id={speaker.id if speaker else '?'})")
    return 0 if total else 1


def cmd_forget(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = MemoryStore(settings.db_path)
    speaker = store.get_speaker(args.speaker_id)
    if speaker is None:
        _p(f"ไม่พบผู้สนทนา id={args.speaker_id}")
        return 1
    if not args.yes:
        answer = input(f"ลบ {speaker.call_name} และความจำทั้งหมด? พิมพ์ 'ใช่' เพื่อยืนยัน: ")
        if answer.strip() not in {"ใช่", "yes", "y"}:
            _p("ยกเลิกแล้ว")
            return 1
    store.delete_speaker(speaker.id)
    _p(f"ลบ {speaker.call_name} และความจำทั้งหมดแล้ว")
    store.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import run

    settings = get_settings()
    run(host=args.host or settings.host, port=args.port or settings.port)
    return 0


# ── argparse ────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thaivoice",
        description="ระบบสนทนาด้วยเสียงภาษาไทยที่จดจำผู้สนทนาได้",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="ตรวจว่าติดตั้งอะไรครบแล้วบ้าง").set_defaults(func=cmd_doctor)

    talk = sub.add_parser("talk", help="คุยด้วยเสียงผ่านไมโครโฟน")
    talk.set_defaults(func=cmd_talk)

    chat = sub.add_parser("chat", help="คุยด้วยการพิมพ์ (ไม่ต้องมีไมค์)")
    chat.add_argument("--speaker", help="ระบุว่าคุยในนามของใคร")
    chat.add_argument("--speak", action="store_true", help="อ่านคำตอบออกเสียงด้วย")
    chat.set_defaults(func=cmd_chat)

    sub.add_parser("speakers", help="ดูรายชื่อคนที่ระบบรู้จัก").set_defaults(func=cmd_speakers)

    memory = sub.add_parser("memory", help="ดูความจำของคนคนหนึ่ง")
    memory.add_argument("speaker_id", type=int)
    memory.add_argument("--turns", type=int, default=0, help="แสดงบทสนทนาล่าสุดกี่เทิร์น")
    memory.set_defaults(func=cmd_memory)

    enroll = sub.add_parser("enroll", help="สอนให้จำเสียงคนใหม่จากไฟล์ WAV")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--gender", choices=["male", "female"])
    enroll.add_argument("--wav", nargs="+", required=True, help="ไฟล์ WAV 16-bit หนึ่งไฟล์ขึ้นไป")
    enroll.set_defaults(func=cmd_enroll)

    forget = sub.add_parser("forget", help="ลบคนนั้นและความจำทั้งหมด")
    forget.add_argument("speaker_id", type=int)
    forget.add_argument("--yes", action="store_true", help="ไม่ต้องถามยืนยัน")
    forget.set_defaults(func=cmd_forget)

    serve = sub.add_parser("serve", help="เปิดเซิร์ฟเวอร์ HTTP")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except MissingCredentialsError as exc:
        _p(f"\n{YELLOW}{exc}{RESET}")
        return 1
    except KeyboardInterrupt:
        _p(f"\n{DIM}ยกเลิกแล้ว{RESET}")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
