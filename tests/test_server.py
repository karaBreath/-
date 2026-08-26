"""ทดสอบ HTTP API และ WebSocket ด้วย Claude ปลอม"""

import base64

import pytest
from conftest import FakeAnthropic, FakeEmbedder

fastapi = pytest.importorskip("fastapi", reason="ต้องติดตั้ง thaivoice[server] ก่อน")
from fastapi.testclient import TestClient  # noqa: E402

from thaivoice import server as server_module  # noqa: E402
from thaivoice.brain import ThaiBrain  # noqa: E402
from thaivoice.memory import MemoryStore  # noqa: E402
from thaivoice.session import ConversationSession  # noqa: E402
from thaivoice.speaker import SpeakerIdentifier  # noqa: E402
from thaivoice.stt import pcm_to_wav  # noqa: E402


@pytest.fixture
def client(monkeypatch, settings, fake_client: FakeAnthropic):
    """สร้างแอปที่ทุก session ใช้ Claude ปลอมและตัวจำเสียงปลอม"""

    def fake_create_session(cfg=None, *, store=None, session_id=None, **kwargs):
        memory = store or MemoryStore(settings.db_path)
        return ConversationSession(
            store=memory,
            brain=ThaiBrain(memory, client=fake_client, settings=settings),
            identifier=SpeakerIdentifier(
                memory, FakeEmbedder(), threshold=0.9, margin=0.02
            ),
            settings=settings,
            session_id=session_id,
        )

    monkeypatch.setattr(server_module, "create_session", fake_create_session)
    with TestClient(server_module.create_app(settings)) as test_client:
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
        response = client.post(
            "/api/chat", json={"text": "สวัสดีครับ ผมชื่อเดชครับ", "speak": False}
        )
        assert response.status_code == 200
        data = response.json()
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

    def test_บทสนทนาถูกบันทึกไว้(self, client):
        client.post("/api/chat", json={"text": "ผมชื่อเดชครับ", "speak": False})
        speakers = client.get("/api/speakers").json()["speakers"]
        detail = client.get(f"/api/speakers/{speakers[0]['id']}").json()
        assert len(detail["recent_turns"]) >= 2


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

    def test_ไม่มีทั้งเสียงที่ถอดได้และตัวถอดเสียง(self, client):
        response = client.post(
            "/api/voice",
            files={"audio": ("u.wav", _wav(), "audio/wav")},
            data={"transcript": "", "speak": "false"},
        )
        assert response.status_code == 400
        assert "transcript" in response.json()["detail"]

    def test_ไฟล์เสียงเสียหาย(self, client):
        response = client.post(
            "/api/voice",
            files={"audio": ("u.wav", "ไม่ใช่ไฟล์เสียง".encode(), "audio/wav")},
            data={"transcript": "สวัสดี"},
        )
        assert response.status_code == 400


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

    def test_ลบเฉพาะข้อเท็จจริง(self, client):
        speaker_id = client.post("/api/speakers", json={"name": "มะลิ"}).json()["speaker"]["id"]
        assert client.delete(f"/api/speakers/{speaker_id}/facts").json() == {"removed": 0}
        assert client.get(f"/api/speakers/{speaker_id}").status_code == 200

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
        client.post(f"/api/speakers/{speaker_id}/enroll", files={"audio": ("s.wav", wav, "audio/wav")})

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
        with client.websocket_connect("/ws/chat") as socket:
            socket.send_json({"text": "ผมชื่อเอกครับ", "audio": audio, "speak": False})
            while socket.receive_json()["type"] != "done":
                pass
            socket.send_json({"text": "จำผมได้ไหม", "audio": audio, "speak": False})
            first = socket.receive_json()

        assert first["speaker"]["name"] == "เอก"
        assert first["identified_by"] == "voice"

    def test_ข้อความว่างได้error(self, client):
        with client.websocket_connect("/ws/chat") as socket:
            socket.send_json({"text": "   "})
            assert socket.receive_json()["type"] == "error"
