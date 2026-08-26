"""ชั้นความจำถาวร (SQLite) — "จำได้ว่าคุยกับใคร และเคยคุยอะไรกันไว้"

ความจำแบ่งเป็น 4 ชั้น:

* ``speakers``   — ตัวตนผู้สนทนา (ชื่อ ชื่อเล่น คำลงท้ายที่ใช้ ภาษา)
* ``voiceprints``— ลายเสียง (embedding) ไว้จำว่า "คนที่พูดอยู่ตอนนี้คือใคร"
* ``facts``      — ข้อเท็จจริงระยะยาวรายคน (คีย์-ค่า) เช่น อาชีพ ความชอบ
* ``turns`` + ``summaries`` — บทสนทนาดิบ และบทสรุปสะสมเมื่อบทสนทนายาวขึ้น

ออกแบบให้เรียกจากหลายเธรดได้ (เซิร์ฟเวอร์ + งานเบื้องหลังสกัดความจำ) ด้วย
``check_same_thread=False`` คู่กับ lock และเปิด WAL
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["MemoryStore", "Speaker", "Turn", "Fact", "SpeakerStats", "normalize_name"]

SCHEMA_VERSION = 2

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS speakers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT    NOT NULL,
    -- ชื่อที่ normalize แล้ว ใช้เป็นกุญแจกันสร้างคนซ้ำเมื่อมีหลายคำขอพร้อมกัน
    name_key     TEXT,
    nickname     TEXT,
    gender       TEXT,              -- 'male' | 'female' | NULL (เดาจากคำลงท้าย)
    particle     TEXT,              -- คำลงท้ายที่บอทใช้กับคนนี้: ครับ / ค่ะ
    language     TEXT    NOT NULL DEFAULT 'th',
    notes        TEXT,
    meta_json    TEXT    NOT NULL DEFAULT '{}',
    created_at   REAL    NOT NULL,
    last_seen_at REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS voiceprints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id  INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    backend     TEXT    NOT NULL,   -- resemblyzer | speechbrain | ...
    dim         INTEGER NOT NULL,
    embedding   BLOB    NOT NULL,   -- float32 little-endian
    sample_count INTEGER NOT NULL DEFAULT 1,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL,
    UNIQUE (speaker_id, backend)
);

CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE CASCADE,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL,    -- 'user' | 'assistant'
    content    TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_speaker ON turns(speaker_id, id);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);

CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id    INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    key           TEXT    NOT NULL,
    value         TEXT    NOT NULL,
    category      TEXT    NOT NULL DEFAULT 'อื่น ๆ',
    confidence    REAL    NOT NULL DEFAULT 0.8,
    source_turn_id INTEGER,
    created_at    REAL    NOT NULL,
    updated_at    REAL    NOT NULL,
    UNIQUE (speaker_id, key)
);
CREATE INDEX IF NOT EXISTS idx_facts_speaker ON facts(speaker_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS summaries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id     INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    content        TEXT    NOT NULL,
    through_turn_id INTEGER NOT NULL DEFAULT 0,
    created_at     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_speaker ON summaries(speaker_id, id DESC);
"""


# ── โครงสร้างข้อมูล ─────────────────────────────────────────────────────────
@dataclass
class Speaker:
    id: int
    display_name: str
    nickname: str | None = None
    gender: str | None = None
    particle: str | None = None
    language: str = "th"
    notes: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_seen_at: float = 0.0

    @property
    def call_name(self) -> str:
        """ชื่อที่ควรใช้เรียกคนนี้ในบทสนทนา"""
        return self.nickname or self.display_name


@dataclass
class Turn:
    id: int
    speaker_id: int | None
    session_id: str
    role: str
    content: str
    created_at: float


@dataclass
class Fact:
    id: int
    speaker_id: int
    key: str
    value: str
    category: str
    confidence: float
    updated_at: float


@dataclass
class SpeakerStats:
    turns: int
    facts: int
    first_seen: float | None
    last_seen: float | None


# ตัวคั่นสำหรับกุญแจของคนชื่อซ้ำ — ต้องเป็นอักขระที่ผู้ใช้พิมพ์เข้ามาไม่ได้
# ไม่งั้นคนที่พิมพ์ชื่อ "สมชาย#2" จะได้ตัวตน (และความจำ) ของคนอื่นไปเลย
_DUP_SEP = "\x00"


def normalize_name(name: str) -> str:
    """ทำชื่อให้เป็นรูปมาตรฐานสำหรับใช้เทียบและกันซ้ำ

    ตัดช่องว่างหัวท้าย ยุบช่องว่างซ้อน และลดเป็นตัวพิมพ์เล็ก
    อักขระควบคุมถูกตัดทิ้ง เพื่อไม่ให้ชื่อที่ผู้ใช้พิมพ์ชนกับกุญแจภายใน
    """
    cleaned = "".join(ch for ch in (name or "") if ch >= " " or ch in "\t\n\r")
    return " ".join(cleaned.split()).lower()


def base_name_key(key: str) -> str:
    """ตัดส่วนแยกแยะ (#N ภายใน) ออก เหลือกุญแจชื่อฐาน"""
    return (key or "").split(_DUP_SEP, 1)[0]


def _row_to_speaker(row: sqlite3.Row) -> Speaker:
    try:
        meta = json.loads(row["meta_json"] or "{}")
    except json.JSONDecodeError:
        meta = {}
    return Speaker(
        id=row["id"],
        display_name=row["display_name"],
        nickname=row["nickname"],
        gender=row["gender"],
        particle=row["particle"],
        language=row["language"],
        notes=row["notes"],
        meta=meta,
        created_at=row["created_at"],
        last_seen_at=row["last_seen_at"],
    )


class MemoryStore:
    """ที่เก็บความจำทั้งหมด — ปลอดภัยต่อการเรียกข้ามเธรด"""

    def __init__(self, db_path: str | Path = "data/memory.db") -> None:
        self.path = Path(db_path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            with self._lock:
                self._conn.executescript(_SCHEMA)
                self._migrate()
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                self._conn.commit()
        except BaseException:
            # อย่าปล่อยให้ connection ค้างไว้พร้อม transaction ที่เขียนค้าง
            # ไม่งั้นการเปิดฐานข้อมูลครั้งถัดไปในโปรเซสเดียวกันจะเจอ "database is locked"
            try:
                self._conn.rollback()
            finally:
                self._conn.close()
            raise

    def _migrate(self) -> None:
        """อัปเกรดฐานข้อมูลเก่าให้เข้ากับ schema ปัจจุบัน (เรียกใต้ lock แล้ว)"""
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(speakers)").fetchall()
        }
        if "name_key" not in columns:
            # ฐานข้อมูลจาก schema เวอร์ชัน 1 — เติมคอลัมน์แล้วเติมค่าย้อนหลัง
            self._conn.execute("ALTER TABLE speakers ADD COLUMN name_key TEXT")

        # เติมค่าย้อนหลังให้ทุกแถวที่ยังไม่มีกุญแจ ไม่ใช่เฉพาะตอนเพิ่งเติมคอลัมน์
        # (ถ้ารอบก่อนล้มกลางคัน ALTER อาจ commit ไปแล้วแต่ค่ายังว่าง)
        pending = self._conn.execute(
            "SELECT id, display_name FROM speakers"
            " WHERE name_key IS NULL OR name_key = '' ORDER BY id"
        ).fetchall()
        if pending:
            taken = {
                row["name_key"]
                for row in self._conn.execute(
                    "SELECT name_key FROM speakers"
                    " WHERE name_key IS NOT NULL AND name_key <> ''"
                ).fetchall()
            }
            for row in pending:
                # schema เวอร์ชัน 1 ไม่มี UNIQUE index จึงมีชื่อซ้ำกันได้
                # ต้องแยกกุญแจให้เอง ไม่งั้น index ที่จะสร้างต่อไปจะพัง
                # แล้วฐานข้อมูลทั้งก้อนจะเปิดไม่ได้อีกเลย
                base = normalize_name(row["display_name"])
                candidate = base
                index = 2
                while candidate in taken:
                    candidate = f"{base}{_DUP_SEP}{index}"
                    index += 1
                taken.add(candidate)
                self._conn.execute(
                    "UPDATE speakers SET name_key = ? WHERE id = ?",
                    (candidate, row["id"]),
                )

        # สร้าง index หลัง migration เสมอ เพราะตารางที่มีอยู่แล้วจาก schema เวอร์ชัน 1
        # ยังไม่มีคอลัมน์นี้ตอนที่รัน schema หลัก
        self._conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_speakers_name_key
               ON speakers(name_key) WHERE name_key IS NOT NULL AND name_key <> ''"""
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── ผู้สนทนา ────────────────────────────────────────────────────────
    def create_speaker(
        self,
        display_name: str,
        *,
        nickname: str | None = None,
        gender: str | None = None,
        particle: str | None = None,
        language: str = "th",
        meta: dict[str, Any] | None = None,
        allow_duplicate_name: bool = False,
    ) -> Speaker:
        """สร้างผู้สนทนาใหม่

        ``allow_duplicate_name`` ใช้เมื่อรู้แน่ว่าเป็นคนละคนแม้ชื่อจะซ้ำกัน
        (เช่นลายเสียงไม่ตรงกับคนเดิมที่ชื่อนี้) ระบบจะเก็บกุญแจภายในแยกกัน
        โดยที่ชื่อที่แสดงยังเหมือนเดิม
        """
        clean = " ".join((display_name or "").split())
        if not clean:
            raise ValueError("ชื่อผู้สนทนาว่างเปล่า")
        now = time.time()
        with self._lock:
            key = self._free_name_key(clean) if allow_duplicate_name else normalize_name(clean)
            cur = self._conn.execute(
                """INSERT INTO speakers
                   (display_name, name_key, nickname, gender, particle, language,
                    meta_json, created_at, last_seen_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    clean,
                    key,
                    nickname,
                    gender,
                    particle,
                    language,
                    json.dumps(meta or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            self._conn.commit()
            speaker_id = int(cur.lastrowid or 0)
        got = self.get_speaker(speaker_id)
        assert got is not None
        return got

    def _free_name_key(self, display_name: str) -> str:
        """หากุญแจชื่อที่ยังว่างอยู่ (เรียกใต้ lock แล้ว)"""
        base = normalize_name(display_name)
        candidate = base
        index = 2
        while self._conn.execute(
            "SELECT 1 FROM speakers WHERE name_key = ?", (candidate,)
        ).fetchone():
            candidate = f"{base}{_DUP_SEP}{index}"
            index += 1
        return candidate

    def get_or_create_speaker(self, display_name: str, **fields: Any) -> tuple[Speaker, bool]:
        """หาคนจากชื่อ ถ้าไม่มีก็สร้าง — ปลอดภัยเมื่อมีหลายคำขอพร้อมกัน

        คืน ``(speaker, created)``

        การทำ find-then-create แยกกันสองคำสั่งทำให้เกิด race: คำขอพร้อมกันหลายตัว
        ต่างก็หาไม่เจอแล้วต่างก็สร้าง ได้คนซ้ำหลายแถวสำหรับคนคนเดียว และความจำจะ
        กระจัดกระจายไปคนละแถว ที่นี่จึงพึ่ง UNIQUE index บน name_key เป็นตัวตัดสิน
        """
        clean = " ".join((display_name or "").split())
        if not clean:
            raise ValueError("ชื่อผู้สนทนาว่างเปล่า")
        existing = self.find_speaker_by_name(clean)
        if existing is not None:
            return existing, False
        try:
            return self.create_speaker(clean, **fields), True
        except sqlite3.IntegrityError:
            # มีคำขออื่นสร้างไปก่อนหน้าเสี้ยววินาที — ใช้ของเขา
            found = self.find_speaker_by_name(clean)
            if found is None:  # pragma: no cover - ไม่ควรเกิด
                raise
            return found, False

    def get_speaker(self, speaker_id: int) -> Speaker | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
        return _row_to_speaker(row) if row else None

    def find_speaker_by_name(self, name: str) -> Speaker | None:
        """หาคนจากชื่อหรือชื่อเล่น (ไม่สนตัวพิมพ์ใหญ่เล็ก และตัดช่องว่างหัวท้าย)"""
        needle = normalize_name(name)
        if not needle:
            return None
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM speakers
                   WHERE name_key = ? OR lower(trim(nickname)) = ?
                   ORDER BY last_seen_at DESC LIMIT 1""",
                (needle, needle),
            ).fetchone()
        return _row_to_speaker(row) if row else None

    def find_speakers_by_name(self, name: str) -> list[Speaker]:
        """หาทุกคนที่ใช้ชื่อนี้ — รวมคนที่ชื่อซ้ำกันแต่เป็นคนละตัวตน

        ``find_speaker_by_name`` คืนแค่คนล่าสุดคนเดียว ซึ่งไม่พอสำหรับการ
        ตัดสินว่าเสียงที่ได้ยินตรงกับ "สมชาย" คนไหน
        """
        needle = normalize_name(name)
        if not needle:
            return []
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM speakers
                   WHERE name_key = ? OR name_key LIKE ? ESCAPE '\\'
                      OR lower(trim(nickname)) = ?
                   ORDER BY last_seen_at DESC""",
                (
                    needle,
                    needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                    + _DUP_SEP
                    + "%",
                    needle,
                ),
            ).fetchall()
        return [_row_to_speaker(r) for r in rows]

    def list_speakers(self) -> list[Speaker]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM speakers ORDER BY last_seen_at DESC"
            ).fetchall()
        return [_row_to_speaker(r) for r in rows]

    def update_speaker(self, speaker_id: int, **fields: Any) -> Speaker | None:
        """อัปเดตข้อมูลผู้สนทนา — รับเฉพาะคอลัมน์ที่อนุญาต"""
        allowed = {
            "display_name",
            "nickname",
            "gender",
            "particle",
            "language",
            "notes",
        }
        sets: list[str] = []
        values: list[Any] = []
        rename_to: str | None = None
        for key, value in fields.items():
            if key == "meta":
                sets.append("meta_json = ?")
                values.append(json.dumps(value or {}, ensure_ascii=False))
            elif key in allowed:
                sets.append(f"{key} = ?")
                values.append(value)
                if key == "display_name":
                    sets.append("name_key = ?")
                    rename_to = normalize_name(str(value))
                    values.append(rename_to)
        if not sets:
            return self.get_speaker(speaker_id)
        values.append(speaker_id)
        with self._lock:
            if rename_to is not None:
                # ถ้ากุญแจใหม่ถูกคนอื่นจองไว้แล้ว ให้หากุญแจว่างแทนที่จะโยน
                # IntegrityError ทิ้งไว้ให้ผู้เรียกกลืน แล้วอัปเดตครึ่ง ๆ กลาง ๆ
                clash = self._conn.execute(
                    "SELECT id FROM speakers WHERE name_key = ? AND id <> ?",
                    (rename_to, speaker_id),
                ).fetchone()
                if clash is not None:
                    values[sets.index("name_key = ?")] = self._free_name_key(rename_to)
            self._conn.execute(
                f"UPDATE speakers SET {', '.join(sets)} WHERE id = ?", values
            )
            self._conn.commit()
        return self.get_speaker(speaker_id)

    def touch_speaker(self, speaker_id: int) -> None:
        """บันทึกว่าเพิ่งเจอคนนี้"""
        with self._lock:
            self._conn.execute(
                "UPDATE speakers SET last_seen_at = ? WHERE id = ?",
                (time.time(), speaker_id),
            )
            self._conn.commit()

    def delete_speaker(self, speaker_id: int) -> bool:
        """ลบคนนี้พร้อมความจำทั้งหมด (สิทธิ์ที่จะถูกลืม)"""
        with self._lock:
            cur = self._conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ── ลายเสียง ────────────────────────────────────────────────────────
    def save_voiceprint(
        self, speaker_id: int, embedding: Sequence[float], backend: str
    ) -> None:
        """บันทึก/รวมลายเสียง — ถ้ามีอยู่แล้วจะเฉลี่ยแบบถ่วงน้ำหนักตามจำนวนตัวอย่าง

        การเฉลี่ยสะสมทำให้ยิ่งคุยบ่อย ยิ่งจำเสียงได้แม่นขึ้น
        """
        vec = [float(x) for x in embedding]
        if not vec:
            raise ValueError("embedding ว่างเปล่า")
        # NaN/inf เกิดได้จริงเมื่อส่งเสียงเงียบสนิทหรือไฟล์ 0 เฟรมเข้าตัวสร้าง
        # embedding ถ้าปล่อยให้บันทึก ค่าเฉลี่ยสะสมของคนคนนั้นจะกลายเป็น NaN
        # ถาวร และจะจำเขาไม่ได้อีกเลย
        if not all(math.isfinite(x) for x in vec):
            raise ValueError("embedding มีค่า NaN หรือ inf — ปฏิเสธเพื่อไม่ให้ลายเสียงเสียถาวร")
        if not any(x for x in vec):
            raise ValueError("embedding เป็นศูนย์ทั้งหมด — น่าจะมาจากเสียงเงียบ")
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding, dim, sample_count FROM voiceprints"
                " WHERE speaker_id = ? AND backend = ?",
                (speaker_id, backend),
            ).fetchone()
            if row is None:
                self._conn.execute(
                    """INSERT INTO voiceprints
                       (speaker_id, backend, dim, embedding, sample_count, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 1, ?, ?)""",
                    (speaker_id, backend, len(vec), _pack(vec), now, now),
                )
            else:
                old = _unpack(row["embedding"])
                if len(old) != len(vec):
                    raise ValueError(
                        f"มิติลายเสียงไม่ตรงกัน: เดิม {len(old)} ใหม่ {len(vec)}"
                    )
                n = int(row["sample_count"])
                merged = [(o * n + v) / (n + 1) for o, v in zip(old, vec)]
                self._conn.execute(
                    """UPDATE voiceprints
                       SET embedding = ?, sample_count = ?, updated_at = ?
                       WHERE speaker_id = ? AND backend = ?""",
                    (_pack(merged), n + 1, now, speaker_id, backend),
                )
            self._conn.commit()

    def all_voiceprints(self, backend: str) -> list[tuple[int, list[float]]]:
        """คืน (speaker_id, embedding) ทุกคน สำหรับนำไปเทียบความคล้าย"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT speaker_id, embedding FROM voiceprints WHERE backend = ?",
                (backend,),
            ).fetchall()
        return [(r["speaker_id"], _unpack(r["embedding"])) for r in rows]

    def has_voiceprint(self, speaker_id: int, backend: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM voiceprints WHERE speaker_id = ? AND backend = ?",
                (speaker_id, backend),
            ).fetchone()
        return row is not None

    # ── บทสนทนา ─────────────────────────────────────────────────────────
    def record_turn(
        self, speaker_id: int | None, session_id: str, role: str, content: str
    ) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError(f"role ต้องเป็น user หรือ assistant ไม่ใช่ {role!r}")
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO turns (speaker_id, session_id, role, content, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (speaker_id, session_id, role, content, time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def recent_turns(
        self, speaker_id: int, limit: int = 16, after_turn_id: int = 0
    ) -> list[Turn]:
        """บทสนทนาล่าสุดของคนนี้ เรียงจากเก่าไปใหม่ (พร้อมส่งเข้าโมเดล)"""
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM (
                       SELECT * FROM turns
                       WHERE speaker_id = ? AND id > ?
                       ORDER BY id DESC LIMIT ?
                   ) ORDER BY id ASC""",
                (speaker_id, after_turn_id, limit),
            ).fetchall()
        return [
            Turn(
                id=r["id"],
                speaker_id=r["speaker_id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def turn_count(self, speaker_id: int) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM turns WHERE speaker_id = ?", (speaker_id,)
            ).fetchone()
        return int(row["n"])

    # ── ข้อเท็จจริงระยะยาว ──────────────────────────────────────────────
    def upsert_fact(
        self,
        speaker_id: int,
        key: str,
        value: str,
        *,
        category: str = "อื่น ๆ",
        confidence: float = 0.8,
        source_turn_id: int | None = None,
    ) -> None:
        """บันทึกหรือทับข้อเท็จจริง — คีย์เดิมถูกทับด้วยค่าใหม่เสมอ

        การทับสำคัญมาก: ถ้าผู้ใช้บอกว่า "ย้ายไปเชียงใหม่แล้ว" ค่าเดิมต้องหายไป
        ไม่ใช่เก็บสองค่าที่ขัดกัน
        """
        key = key.strip()
        value = value.strip()
        if not key or not value:
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO facts
                     (speaker_id, key, value, category, confidence, source_turn_id,
                      created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(speaker_id, key) DO UPDATE SET
                     value = excluded.value,
                     category = excluded.category,
                     confidence = excluded.confidence,
                     source_turn_id = excluded.source_turn_id,
                     updated_at = excluded.updated_at""",
                (
                    speaker_id,
                    key,
                    value,
                    category,
                    confidence,
                    source_turn_id,
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def facts_for(self, speaker_id: int, limit: int = 60) -> list[Fact]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM facts WHERE speaker_id = ?
                   ORDER BY confidence DESC, updated_at DESC LIMIT ?""",
                (speaker_id, limit),
            ).fetchall()
        return [
            Fact(
                id=r["id"],
                speaker_id=r["speaker_id"],
                key=r["key"],
                value=r["value"],
                category=r["category"],
                confidence=r["confidence"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def forget_fact(self, speaker_id: int, key: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM facts WHERE speaker_id = ? AND key = ?",
                (speaker_id, key.strip()),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def forget_all_facts(self, speaker_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM facts WHERE speaker_id = ?", (speaker_id,)
            )
            self._conn.commit()
            return cur.rowcount

    def forget_everything(self, speaker_id: int) -> dict[str, int]:
        """ลบความจำทั้งหมดของคนคนนี้ แต่ยังรู้จักตัวเขาอยู่

        ต้องลบทั้ง facts, summaries และ turns — การลบแค่ facts ทำให้บทสรุปและ
        บทสนทนาดิบยังไหลกลับเข้า prompt รอบถัดไป ซึ่งเท่ากับโกหกผู้ใช้ว่าลบแล้ว
        """
        with self._lock:
            removed = {
                "facts": self._conn.execute(
                    "DELETE FROM facts WHERE speaker_id = ?", (speaker_id,)
                ).rowcount,
                "summaries": self._conn.execute(
                    "DELETE FROM summaries WHERE speaker_id = ?", (speaker_id,)
                ).rowcount,
                "turns": self._conn.execute(
                    "DELETE FROM turns WHERE speaker_id = ?", (speaker_id,)
                ).rowcount,
            }
            self._conn.commit()
        return {key: max(0, value) for key, value in removed.items()}

    def voiceprint_for(self, speaker_id: int, backend: str) -> list[float] | None:
        """ลายเสียงของคนคนนี้ หรือ ``None`` ถ้ายังไม่มี"""
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding FROM voiceprints WHERE speaker_id = ? AND backend = ?",
                (speaker_id, backend),
            ).fetchone()
        return _unpack(row["embedding"]) if row else None

    def speaker_exists(self, speaker_id: int) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
        return row is not None

    # ── บทสรุปสะสม ──────────────────────────────────────────────────────
    def save_summary(self, speaker_id: int, content: str, through_turn_id: int) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO summaries (speaker_id, content, through_turn_id, created_at)
                   VALUES (?, ?, ?, ?)""",
                (speaker_id, content.strip(), through_turn_id, time.time()),
            )
            self._conn.commit()

    def latest_summary(self, speaker_id: int) -> tuple[str, int] | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT content, through_turn_id FROM summaries
                   WHERE speaker_id = ? ORDER BY id DESC LIMIT 1""",
                (speaker_id,),
            ).fetchone()
        return (row["content"], row["through_turn_id"]) if row else None

    # ── สถิติ ───────────────────────────────────────────────────────────
    def stats(self, speaker_id: int) -> SpeakerStats:
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) AS n, MIN(created_at) AS first, MAX(created_at) AS last
                   FROM turns WHERE speaker_id = ?""",
                (speaker_id,),
            ).fetchone()
            facts = self._conn.execute(
                "SELECT COUNT(*) AS n FROM facts WHERE speaker_id = ?", (speaker_id,)
            ).fetchone()
        return SpeakerStats(
            turns=int(row["n"]),
            facts=int(facts["n"]),
            first_seen=row["first"],
            last_seen=row["last"],
        )


# ── ช่วยแปลง embedding เป็น/จาก BLOB ────────────────────────────────────────
def _pack(vec: Iterable[float]) -> bytes:
    arr = array("f", vec)
    return arr.tobytes()


def _unpack(blob: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)
