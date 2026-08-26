"""ทดสอบ HTTP API และ WebSocket ด้วย Claude ปลอม"""

import base64

import pytest
from conftest import FakeAnthropic, FakeEmbedder

fastapi = pytest.importorskip("fastapi", reason="ต้องติดตั้ง thaivoice[server] ก่อน")
from fastapi.testclient import TestClient  # noqa: E402

from thaivoice import server as server_module  # noqa: E402
from thaivoice.memory import MemoryStore  # noqa: E402
from thaivoice.stt import pcm_to_wav  # noqa: E402
from thaivoice.tts import Speech  # noqa: E402


class CountingTTS:
    """TTS ปลอมที่นับจำนวนครั้งที่ถูกเรียก"""

    name = "counting"
    mime = "audio/mpeg"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        self.calls.append(text)
        return Speech(audio=b"MP3" + text.encode()[:8], mime=self.mime, text=text)


@pytest.fixture
def tts() -> CountingTTS:
    return CountingTTS()


@pytest.fixture
def runtime(settings, fake_client: FakeAnthropic, tts: CountingTTS):
    made = server_module.ServerRuntime(
        settings,
        store=MemoryStore(":memory:"),
        client=fake_client,
        embedder=FakeEmbedder(),
        tts=tts,
        stt=None,
        with_memory_extraction=False,
    )
    yield made
    made.close()


@pytest.fixture
def client(settings, runtime):
    with TestClient(server_module.create_app(settings, runtime=runtime)) as test_client:
        yield test_client


def _wav(seed: bytes = b"\x11\x22\x33") -> bytes:
    return pcm_to_wav(seed * 2000, 16000)


class TestBasics:
    def test_health(self, client):
        data = client.get("/health").json()
        assert data["ok"] is True
        assert data["model"] == "claude-opus-5"

    def test_หน้าเว็บสาธิต(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "thaivoice" in response.text or "คุยกับใจ" in response.text


class TestChat:
    def test_คุยด้วยข้อความแล้วสร้างผู้สนทนา(self, client):
        data = client.post(
            "/api/chat", json={"text": "สวัสดีครับ ผมชื่อเดชครับ", "speak": False}
        ).json()
        assert data["reply"]
        assert data["speaker"]["name"] == "เดช"

    def test_ระบุ_speaker_id_ได้โดยตรง(self, client):
        created = client.post("/api/speakers", json={"name": "แนน", "gender": "female"}).json()
        speaker_id = created["speaker"]["id"]

        data = client.post(
            "/api/chat",
            json={"text": "จำได้ไหม", "speaker_id": speaker_id, "speak": False},
        ).json()
        assert data["speaker"]["id"] == speaker_id

    def test_speaker_id_ที่ไม่มีจริงต้องได้_404(self, client):
        """ของเดิมเงียบ ๆ ถือว่าไม่ระบุ แล้วตกไปใช้ผู้พูดคนก่อนของ session นั้น"""
        for bogus in (999999, -1):
            response = client.post(
                "/api/chat", json={"text": "สวัสดี", "speaker_id": bogus}
            )
            assert response.status_code == 404, bogus

    def test_ข้อความว่างต้องได้_422(self, client):
        assert client.post("/api/chat", json={"text": "   "}).status_code == 422

    def test_ข้อความยาวเกินต้องถูกปฏิเสธ(self, client):
        response = client.post("/api/chat", json={"text": "ก" * 5000})
        assert response.status_code == 422

    def test_บทสนทนาถูกบันทึกไว้(self, client):
        client.post("/api/chat", json={"text": "ผมชื่อเดชครับ", "speak": False})
        speakers = client.get("/api/speakers").json()["speakers"]
        detail = client.get(f"/api/speakers/{speakers[0]['id']}").json()
        assert len(detail["recent_turns"]) >= 2

    def test_ขอเสียงคำตอบมาด้วยได้(self, client, tts):
        data = client.post("/api/chat", json={"text": "สวัสดี", "speak": True}).json()
        assert data["audio"], "ขอ speak=True แล้วต้องได้เสียงกลับมา"
        assert len(tts.calls) == 1, f"ต้องสังเคราะห์ครั้งเดียว ไม่ใช่ {len(tts.calls)}"


class TestCrossUserIsolation:
    def test_คำขอที่ไม่ระบุ_session_id_ต้องไม่สวมรอยกัน(self, client, runtime):
        """คนที่สองที่ไม่ระบุตัวตนต้องไม่ได้เห็นความจำของคนแรก

        นี่คือเส้นทางหลักที่ README แนะนำ (เบราว์เซอร์ถอดเสียงเองแล้วส่งข้อความ)
        จึงไม่ใช่เคสมุมเล็ก ๆ แต่เป็นค่าเริ่มต้น
        """
        first = client.post(
            "/api/chat", json={"text": "สวัสดีครับ ผมชื่อสมชายครับ"}
        ).json()
        runtime.store.upsert_fact(first["speaker"]["id"], "รหัสตู้เซฟ", "ลับสุดยอด")

        second = client.post("/api/chat", json={"text": "ตอนนี้กี่โมงแล้ว"}).json()

        assert second["speaker"] is None, "ต้องไม่ถือว่าเป็นคนเดิม"

    def test_ระบุ_session_id_เดียวกันถึงจะจำคนเดิม(self, client):
        client.post(
            "/api/chat", json={"text": "ผมชื่อสมชายครับ", "session_id": "ห้อง-ก"}
        )
        second = client.post(
            "/api/chat", json={"text": "ตอนนี้กี่โมงแล้ว", "session_id": "ห้อง-ก"}
        ).json()
        assert second["speaker"]["name"] == "สมชาย"

    def test_คนละ_session_ไม่เห็นกัน(self, client):
        client.post("/api/chat", json={"text": "ผมชื่อสมชายครับ", "session_id": "ห้อง-ก"})
        other = client.post(
            "/api/chat", json={"text": "ตอนนี้กี่โมงแล้ว", "session_id": "ห้อง-ข"}
        ).json()
        assert other["speaker"] is None

    def test_จำนวน_session_ไม่บานปลาย(self, client, runtime):
        for index in range(server_module.MAX_LIVE_SESSIONS + 40):
            client.post("/api/chat", json={"text": "สวัสดี", "session_id": f"s{index}"})
        assert runtime.live_sessions <= server_module.MAX_LIVE_SESSIONS


class TestVoice:
    def test_ส่งเสียงพร้อมข้อความที่ถอดมาแล้ว(self, client):
        response = client.post(
            "/api/voice",
            files={"audio": ("u.wav", _wav(), "audio/wav")},
            data={"transcript": "สวัสดีครับ ผมชื่อเดชครับ", "speak": "false"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "สวัสดีครับ ผมชื่อเดชครับ"
        assert data["speaker"]["name"] == "เดช"
        assert data["identified_by"] in {"name", "voice"}

    def test_จำเสียงได้ในคำขอถัดไป(self, client):
        wav = _wav(b"\x44\x55\x66")
        client.post(
            "/api/voice",
            files={"audio": ("u.wav", wav, "audio/wav")},
            data={"transcript": "ผมชื่อเดชครับ", "speak": "false"},
        )
        second = client.post(
            "/api/voice",
            files={"audio": ("u.wav", wav, "audio/wav")},
            data={"transcript": "วันนี้อากาศดีนะ", "speak": "false"},
        ).json()

        assert second["speaker"]["name"] == "เดช"
        assert second["identified_by"] == "voice"

    def test_สังเคราะห์เสียงครั้งเดียวต่อหนึ่งเทิร์น(self, client, tts, fake_client):
        """ของเดิมสังเคราะห์ทุกท่อนแล้วทิ้ง จากนั้นสังเคราะห์คำตอบเต็มอีกรอบ"""
        fake_client.replies = ["สวัสดีค่ะ ยินดีที่ได้รู้จักนะคะ วันนี้เป็นยังไงบ้างคะ"]
        client.post(
            "/api/voice",
            files={"audio": ("u.wav", _wav(), "audio/wav")},
            data={"transcript": "สวัสดี", "speak": "true"},
        )
        assert len(tts.calls) == 1, tts.calls

    def test_ระบุ_speaker_id_ผ่าน_form_ได้(self, client):
        created = client.post("/api/speakers", json={"name": "มะลิ"}).json()
        data = client.post(
            "/api/voice",
            files={"audio": ("u.wav", _wav(), "audio/wav")},
            data={
                "transcript": "สวัสดี",
                "speaker_id": str(created["speaker"]["id"]),
                "speak": "false",
            },
        ).json()
        assert data["speaker"]["name"] == "มะลิ"

    def test_speaker_id_ที่ไม่ใช่ตัวเลข(self, client):
        response = client.post(
            "/api/voice",
            files={"audio": ("u.wav", _wav(), "audio/wav")},
            data={"transcript": "สวัสดี", "speaker_id": "abc"},
        )
        assert response.status_code == 422

    def test_ไม่มีทั้งเสียงที่ถอดได้และตัวถอดเสียง(self, client):
        response = client.post(
            "/api/voice",
            files={"audio": ("u.wav", _wav(), "audio/wav")},
            data={"transcript": "", "speak": "false"},
        )
        assert response.status_code == 400
        assert "transcript" in response.json()["detail"]

    @pytest.mark.parametrize(
        "payload", ["ไม่ใช่ไฟล์เสียง".encode(), b"", b"RIFF" + b"\x00" * 20]
    )
    def test_ไฟล์เสียงเสียหายต้องได้_4xx_ไม่ใช่_500(self, client, payload):
        response = client.post(
            "/api/voice",
            files={"audio": ("u.wav", payload, "audio/wav")},
            data={"transcript": "สวัสดี"},
        )
        assert 400 <= response.status_code < 500, response.status_code

    def test_ไฟล์ใหญ่เกินถูกปฏิเสธ(self, client):
        big = b"\x00" * (server_module.MAX_UPLOAD_BYTES + 10)
        response = client.post(
            "/api/voice",
            files={"audio": ("u.wav", big, "audio/wav")},
            data={"transcript": "สวัสดี"},
        )
        assert response.status_code == 413


class TestSpeakerManagement:
    def test_สร้าง_อ่าน_ลบ(self, client):
        created = client.post("/api/speakers", json={"name": "มะลิ"}).json()
        assert created["created"] is True
        speaker_id = created["speaker"]["id"]

        again = client.post("/api/speakers", json={"name": "มะลิ"}).json()
        assert again["created"] is False

        assert client.get(f"/api/speakers/{speaker_id}").status_code == 200
        assert client.delete(f"/api/speakers/{speaker_id}").json()["deleted"] is True
        assert client.get(f"/api/speakers/{speaker_id}").status_code == 404

    @pytest.mark.parametrize("name", ["", "   ", "\n\t "])
    def test_ชื่อว่างต้องถูกปฏิเสธ(self, client, name):
        """ของเดิมสร้างแถวใหม่ทุกครั้งที่ส่งชื่อว่างมา"""
        assert client.post("/api/speakers", json={"name": name}).status_code == 422

    def test_ชื่อยาวเกินถูกปฏิเสธ(self, client):
        assert client.post("/api/speakers", json={"name": "ก" * 200}).status_code == 422

    def test_ลบเฉพาะข้อเท็จจริง(self, client, runtime):
        speaker_id = client.post("/api/speakers", json={"name": "มะลิ"}).json()["speaker"]["id"]
        runtime.store.upsert_fact(speaker_id, "ก", "1")
        runtime.store.record_turn(speaker_id, "s", "user", "เก็บไว้")

        assert client.delete(f"/api/speakers/{speaker_id}/facts").json() == {"removed": 1}
        assert runtime.store.recent_turns(speaker_id), "บทสนทนายังต้องอยู่"

    def test_ลบความจำทั้งหมด(self, client, runtime):
        speaker_id = client.post("/api/speakers", json={"name": "มะลิ"}).json()["speaker"]["id"]
        runtime.store.upsert_fact(speaker_id, "ก", "1")
        runtime.store.record_turn(speaker_id, "s", "user", "ความลับ")
        runtime.store.save_summary(speaker_id, "สรุป", 1)

        removed = client.delete(f"/api/speakers/{speaker_id}/memory").json()["removed"]
        assert removed == {"facts": 1, "summaries": 1, "turns": 1}
        assert runtime.store.recent_turns(speaker_id) == []
        assert runtime.store.latest_summary(speaker_id) is None

    def test_สอนลายเสียงผ่าน_api(self, client):
        speaker_id = client.post("/api/speakers", json={"name": "มะลิ"}).json()["speaker"]["id"]
        response = client.post(
            f"/api/speakers/{speaker_id}/enroll",
            files={"audio": ("s.wav", _wav(b"\x77\x88\x99"), "audio/wav")},
        )
        assert response.json()["ok"] is True

    def test_ถามว่าเสียงนี้คือใคร(self, client):
        speaker_id = client.post("/api/speakers", json={"name": "มะลิ"}).json()["speaker"]["id"]
        wav = _wav(b"\xaa\xbb\xcc")
        client.post(
            f"/api/speakers/{speaker_id}/enroll", files={"audio": ("s.wav", wav, "audio/wav")}
        )

        data = client.post("/api/identify", files={"audio": ("s.wav", wav, "audio/wav")}).json()
        assert data["speaker"]["id"] == speaker_id
        assert data["method"] == "voice"


class TestWebSocket:
    def test_สตรีมคำตอบเป็นเหตุการณ์(self, client):
        with client.websocket_connect("/ws/chat?session_id=t1") as socket:
            socket.send_json({"text": "สวัสดีครับ ผมชื่อเดชครับ", "speak": False})

            events = []
            while True:
                event = socket.receive_json()
                events.append(event)
                if event["type"] == "done":
                    break

        kinds = [e["type"] for e in events]
        assert kinds[0] == "speaker"
        assert "delta" in kinds
        assert "chunk" in kinds
        assert kinds[-1] == "done"
        assert events[0]["speaker"]["name"] == "เดช"

    def test_ส่งเสียงมาด้วยเพื่อระบุตัวผู้พูด(self, client):
        audio = base64.b64encode(_wav(b"\x12\x34\x56")).decode()
        with client.websocket_connect("/ws/chat?session_id=t2") as socket:
            socket.send_json({"text": "ผมชื่อเอกครับ", "audio": audio, "speak": False})
            while socket.receive_json()["type"] != "done":
                pass
            socket.send_json({"text": "จำผมได้ไหม", "audio": audio, "speak": False})
            first = socket.receive_json()

        assert first["speaker"]["name"] == "เอก"
        assert first["identified_by"] == "voice"

    @pytest.mark.parametrize(
        "payload",
        [
            {"text": "   "},
            {"text": "hi", "speaker_id": "xyz"},
            {"text": "hi", "speaker_id": 999999},
            {"text": 12345},
            {"text": None},
            {},
            [1, 2, 3],
            "แค่สตริง",
            42,
            None,
        ],
    )
    def test_ข้อมูลผิดรูปต้องได้_error_frame_ไม่ใช่ค้างตลอดกาล(self, client, payload):
        """ของเดิมปิดการเชื่อมต่อเงียบ ๆ ทำให้ไคลเอนต์รอคำตอบที่ไม่มีวันมา"""
        with client.websocket_connect("/ws/chat") as socket:
            socket.send_json(payload)
            event = socket.receive_json()
            assert event["type"] == "error", event
            assert event["text"]

    def test_การเชื่อมต่อยังใช้ต่อได้หลังเจอ_error(self, client):
        with client.websocket_connect("/ws/chat?session_id=t3") as socket:
            socket.send_json({"text": ""})
            assert socket.receive_json()["type"] == "error"

            socket.send_json({"text": "สวัสดี", "speak": False})
            kinds = []
            while True:
                event = socket.receive_json()
                kinds.append(event["type"])
                if event["type"] == "done":
                    break
            assert "done" in kinds

    def test_โมเดลพังต้องได้_error_frame(self, client, fake_client):
        """ถ้าเรียกโมเดลไม่สำเร็จ ต้องบอกไคลเอนต์ ไม่ใช่ปิดการเชื่อมต่อเงียบ ๆ"""
        fake_client.fail_with = RuntimeError("โมเดลล่ม")
        with client.websocket_connect("/ws/chat?session_id=t4") as socket:
            socket.send_json({"text": "สวัสดี", "speak": False})
            events = [socket.receive_json(), socket.receive_json()]

        assert events[0]["type"] == "speaker"
        assert events[1]["type"] == "error"
        assert "โมเดลล่ม" in events[1]["text"]


class TestWavValidation:
    """ตรวจหัวไฟล์เสียงให้ครบก่อนใช้ — ไฟล์แปลก ๆ ผ่านเข้ามาได้ง่ายกว่าที่คิด"""

    def _wav_header(self, channels: int, rate: int, width: int, frames: bytes) -> bytes:
        import io
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(width)
            wf.setframerate(rate)
            wf.writeframes(frames)
        return buffer.getvalue()

    @pytest.mark.parametrize(
        "channels,rate,width,frames,reason",
        [
            (1, 16000, 2, b"", "ไม่มีเฟรมเลย"),
            (64, 16000, 2, b"\x00" * 256, "ช่องสัญญาณเยอะเกิน"),
            (1, 1, 2, b"\x00" * 64, "อัตราสุ่มตัวอย่างต่ำเกิน"),
            (1, 16000, 1, b"\x00" * 64, "ไม่ใช่ 16-bit"),
        ],
    )
    def test_ไฟล์เสียงผิดปกติถูกปฏิเสธ(self, client, channels, rate, width, frames, reason):
        data = self._wav_header(channels, rate, width, frames)
        response = client.post(
            "/api/voice",
            files={"audio": ("u.wav", data, "audio/wav")},
            data={"transcript": "สวัสดี"},
        )
        assert 400 <= response.status_code < 500, f"{reason}: {response.status_code}"

    def test_สเตอริโอถูกรวมเป็นโมโนถูกต้อง(self):
        import array

        from thaivoice.stt import wav_to_pcm

        # ซ้ายเป็น 1000 ขวาเป็น 2000 ตลอด -> โมโนต้องได้ 1500
        interleaved = array.array("h", [1000, 2000] * 100).tobytes()
        data = self._wav_header(2, 16000, 2, interleaved)
        pcm, rate = wav_to_pcm(data)

        mono = array.array("h")
        mono.frombytes(pcm)
        assert rate == 16000
        assert len(mono) == 100
        assert all(value == 1500 for value in mono), mono[:5]


    def test_รวมช่องสัญญาณได้ผลเท่ากันทั้งมีและไม่มี_numpy(self):
        """สองเส้นทางต้องให้ผลตรงกันเป๊ะ

        np.mean ปัดเข้าหาศูนย์ ส่วนการหารจำนวนเต็มของ Python ปัดลง ค่าติดลบจึง
        ต่างกันหนึ่งหน่วยถ้าไม่ระวัง ซึ่งแปลว่าลายเสียงที่คำนวณได้จะไม่เหมือนกัน
        ระหว่างเครื่องที่ติดตั้ง numpy กับที่ไม่ได้ติดตั้ง
        """
        import array
        import builtins

        from thaivoice.stt import wav_to_pcm

        pytest.importorskip("numpy", reason="ต้องมี numpy จึงจะเทียบสองเส้นทางได้")

        interleaved = array.array(
            "h", [-1, 0, -1, -1, 32767, 32767, 32767, 32767, -3, 0, 0, 0] * 20
        ).tobytes()
        data = self._wav_header(4, 16000, 2, interleaved)

        def decode(with_numpy: bool) -> list[int]:
            real_import = builtins.__import__
            if not with_numpy:

                def blocked(name, *args, **kwargs):
                    if name == "numpy":
                        raise ImportError("ปิดไว้เพื่อทดสอบเส้นทางสำรอง")
                    return real_import(name, *args, **kwargs)

                builtins.__import__ = blocked
            try:
                pcm, _rate = wav_to_pcm(data)
            finally:
                builtins.__import__ = real_import
            samples = array.array("h")
            samples.frombytes(pcm)
            return list(samples)

        assert decode(True) == decode(False)


class TestWebSocketFrames:
    """เฟรมผิดรูปต้องได้คำตอบเสมอ ไม่ใช่ปิดการเชื่อมต่อเงียบ ๆ"""

    def test_เฟรมไบนารีต้องได้_error_ไม่ใช่ปิดเงียบ(self, client):
        with client.websocket_connect("/ws/chat") as socket:
            socket.send_bytes(b"\x00\x01\xff")
            event = socket.receive_json()
            assert event["type"] == "error"
            assert "ไบนารี" in event["text"]

    def test_ข้อความที่ไม่ใช่_JSON_ต้องได้_error(self, client):
        with client.websocket_connect("/ws/chat") as socket:
            socket.send_text("ไม่ใช่ JSON เลย {{{")
            event = socket.receive_json()
            assert event["type"] == "error"

    def test_ใช้ต่อได้หลังส่งเฟรมผิดรูป(self, client):
        with client.websocket_connect("/ws/chat?session_id=frames") as socket:
            socket.send_bytes(b"\xff\xfe")
            assert socket.receive_json()["type"] == "error"
            socket.send_text("[1,2,3]")
            assert socket.receive_json()["type"] == "error"

            socket.send_json({"text": "สวัสดี", "speak": False})
            kinds = []
            while True:
                event = socket.receive_json()
                kinds.append(event["type"])
                if event["type"] == "done":
                    break
            assert "done" in kinds


class TestRuntimeLifecycle:
    def test_ปิดระบบระหว่างมีคำขอค้างต้องไม่กลายเป็น_500(self, settings, fake_client):
        """MemoryExtractor เคยโยน RuntimeError เมื่อมีงานเข้ามาหลัง pool ถูกปิด"""
        from thaivoice.extraction import MemoryExtractor
        from thaivoice.memory import MemoryStore

        store = MemoryStore(":memory:")
        extractor = MemoryExtractor(store, fake_client, settings)
        speaker = store.create_speaker("เดช")
        turns = [store.recent_turns(speaker.id)]

        extractor.shutdown(wait=False)
        # ต้องไม่โยน exception
        extractor.schedule(speaker, turns[0])
        extractor.maybe_summarize(speaker)
        store.close()

    def test_ปิด_runtime_แล้วฐานข้อมูลถูกปิดด้วย(self, settings, fake_client):
        from thaivoice.memory import MemoryStore

        store = MemoryStore(":memory:")
        runtime = server_module.ServerRuntime(
            settings,
            store=store,
            client=fake_client,
            embedder=None,
            tts=None,
            stt=None,
            with_memory_extraction=False,
        )
        runtime.close()

        import sqlite3

        with pytest.raises(sqlite3.ProgrammingError):
            store.list_speakers()

    def test_งานสตรีมใช้_executor_แยกจากงาน_HTTP(self, runtime):
        """ไม่งั้น WebSocket หลายสิบตัวจะยึดเธรดจนทุก endpoint รอคิวเป็นสิบวินาที"""
        import asyncio

        assert runtime.stream_executor is not None
        assert runtime.stream_executor is not asyncio.get_event_loop_policy()
