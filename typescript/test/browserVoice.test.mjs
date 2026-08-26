/**
 * ทดสอบตรรกะฝั่งเบราว์เซอร์ด้วยของปลอม
 *
 * ทุกเคสในไฟล์นี้มาจากบัคจริงที่เคยเกิดขึ้น ไม่ใช่เทสต์เพื่อให้มีเทสต์
 */

import assert from "node:assert/strict";
import test, { beforeEach, describe } from "node:test";

import {
  FakeAudio,
  FakeAudioContext,
  FakeMedia,
  FakeSpeechRecognition,
  FakeTrack,
  FakeWebSocket,
  installFakes,
  sleep,
} from "./fakes.mjs";

installFakes();

const { AudioQueue, PcmRecorder, VoiceConversation, encodeWav } = await import(
  "../dist/browserVoice.js"
);
const { ThaiVoiceClient } = await import("../dist/client.js");

beforeEach(() => {
  installFakes();
  FakeMedia.delay = 5;
  FakeMedia.deny = false;
});

describe("encodeWav", () => {
  test("เขียนหัวไฟล์ WAV ถูกต้อง", async () => {
    const blob = encodeWav(new Float32Array([0, 0.5, -0.5]), 16000);
    const view = new DataView(await blob.arrayBuffer());
    const tag = String.fromCharCode(...[0, 1, 2, 3].map((i) => view.getUint8(i)));
    assert.equal(tag, "RIFF");
    assert.equal(view.getUint16(22, true), 1, "ต้องเป็น mono");
    assert.equal(view.getUint32(24, true), 16000);
    assert.equal(view.getUint16(34, true), 16, "ต้องเป็น 16-bit");
  });
});

describe("AudioQueue", () => {
  test("เล่นตามลำดับจนครบ แล้วแจ้งว่าว่างแล้ว", async () => {
    let idle = 0;
    const queue = new AudioQueue(() => (idle += 1));
    queue.accept(1);
    queue.push(1, "AAA");
    queue.push(1, "BBB");

    await sleep(120);
    assert.equal(FakeAudio.played.length, 2);
    assert.ok(FakeAudio.played[0].endsWith("AAA"));
    assert.ok(FakeAudio.played[1].endsWith("BBB"));
    assert.equal(idle, 1, "ต้องแจ้งว่าว่างครั้งเดียว");
    assert.equal(queue.busy, false);
  });

  test("สั่งหยุดแล้วบอทต้องไม่กลับมาพูดต่อ", async () => {
    // เซิร์ฟเวอร์ยังส่งประโยคที่เหลือของเทิร์นเดิมตามมาอีกหลังผู้ใช้พูดแทรก
    // ของเดิมเห็นว่าคิวว่างแล้วก็เริ่มเล่นใหม่ กลายเป็นบอทพูดต่อทั้งที่ถูกขัด
    const queue = new AudioQueue();
    queue.accept(1);
    queue.push(1, "AAA");
    await sleep(5);

    queue.stop();
    queue.push(1, "BBB");
    queue.push(1, "CCC");
    await sleep(80);

    assert.equal(FakeAudio.played.length, 1, `เล่นไปแล้ว: ${FakeAudio.played}`);
    assert.equal(queue.busy, false);
  });

  test("เทิร์นใหม่หลังถูกขัดยังเล่นได้ตามปกติ", async () => {
    const queue = new AudioQueue();
    queue.accept(1);
    queue.push(1, "AAA");
    await sleep(5);
    queue.stop();

    queue.accept(2);
    queue.push(2, "BBB");
    await sleep(80);

    assert.ok(FakeAudio.played.some((url) => url.endsWith("BBB")));
  });

  test("ไม่มีเสียงซ้อนกันแม้ถูกขัดกลางคัน", async () => {
    // pause() ไม่ยิงทั้ง ended และ error ถ้าไม่ดัก onpause ลูปเดิมจะค้าง
    // แล้วมีลูปที่สองเริ่มขึ้นมาพร้อมกัน กลายเป็นสองเสียงพูดทับกัน
    const queue = new AudioQueue();
    let maxLive = 0;
    const watcher = setInterval(() => {
      maxLive = Math.max(maxLive, FakeAudio.live);
    }, 2);

    for (let turn = 1; turn <= 4; turn += 1) {
      queue.accept(turn);
      queue.push(turn, `T${turn}A`);
      queue.push(turn, `T${turn}B`);
      await sleep(8);
      queue.stop();
    }
    await sleep(80);
    clearInterval(watcher);

    assert.ok(maxLive <= 1, `มีเสียงเล่นพร้อมกัน ${maxLive} เสียง`);
  });

  test("เสียงของเทิร์นเก่าที่มาช้าถูกทิ้ง", async () => {
    const queue = new AudioQueue();
    queue.accept(2);
    queue.push(1, "เก่า");
    await sleep(40);
    assert.equal(FakeAudio.played.length, 0);
  });
});

describe("PcmRecorder", () => {
  test("สั่งหยุดระหว่างกำลังขอสิทธิ์ไมโครโฟนต้องไม่ทิ้งไมค์ค้าง", async () => {
    // getUserMedia ใช้เวลาได้หลายวินาที (ครั้งแรก หรือหูฟังบลูทูธ)
    // ของเดิม stop() จะเห็นว่ายังไม่มี context เลยไม่ทำอะไร แล้ว start() ที่ค้างอยู่
    // ก็สร้าง AudioContext ขึ้นมาหลังจากนั้น ไฟไมค์ในเบราว์เซอร์จะไม่ดับอีกเลย
    FakeMedia.delay = 40;
    const recorder = new PcmRecorder();
    const starting = recorder.start();
    await sleep(5);
    const stopping = recorder.stop();
    await Promise.allSettled([starting, stopping]);
    await sleep(30);

    assert.equal(FakeTrack.live, 0, "ไมโครโฟนยังเปิดค้างอยู่");
    assert.equal(FakeAudioContext.live, 0, "AudioContext ยังเปิดค้างอยู่");
    assert.equal(recorder.recording, false);
  });

  test("เรียก start ซ้ำไม่สร้างของซ้อน", async () => {
    const recorder = new PcmRecorder();
    await Promise.all([recorder.start(), recorder.start(), recorder.start()]);
    assert.equal(FakeAudioContext.live, 1);
    assert.equal(FakeTrack.live, 1);
    await recorder.stop();
    assert.equal(FakeTrack.live, 0);
  });

  test("เรียก stop ซ้ำไม่พัง", async () => {
    const recorder = new PcmRecorder();
    await recorder.start();
    await recorder.stop();
    await recorder.stop();
    assert.equal(FakeTrack.live, 0);
  });

  test("take คืนเสียงแล้วอัดต่อได้", async () => {
    const recorder = new PcmRecorder();
    await recorder.start();
    const context = globalThis.__lastContext;
    assert.ok(context, "ต้องมี AudioContext");

    context.emit(new Float32Array(1000).fill(0.1));
    const first = recorder.take();
    assert.ok(first && first.size > 44, "ต้องได้ไฟล์ WAV");

    assert.equal(recorder.take(), null, "ตัดไปแล้วต้องไม่มีของเหลือ");
    context.emit(new Float32Array(1000).fill(0.2));
    assert.ok(recorder.take(), "ต้องยังอัดต่อได้");
    await recorder.stop();
  });

  test("ความเงียบยาว ๆ ไม่สะสมไม่จำกัด", async () => {
    // ของเดิมอัดตั้งแต่กดเริ่มจนพูดประโยคแรก ถ้าเปิดทิ้งไว้หนึ่งนาทีก่อนพูด
    // ไฟล์ที่ส่งไปจะมีแต่ความเงียบ และลายเสียงที่คำนวณได้ก็ไร้ความหมาย
    const recorder = new PcmRecorder(16000, 1); // เก็บแค่ 1 วินาที
    await recorder.start();
    const context = globalThis.__lastContext;
    for (let i = 0; i < 60; i += 1) context.emit(new Float32Array(4096));

    const blob = recorder.take();
    // 1 วินาทีที่ 48 kHz ย่อเหลือ 16 kHz = ประมาณ 16000 ตัวอย่าง (32000 ไบต์)
    assert.ok(blob.size < 80000, `ไฟล์ใหญ่เกินไป: ${blob.size} ไบต์`);
    await recorder.stop();
  });
});

describe("VoiceConversation", () => {
  const client = () => new ThaiVoiceClient({ baseUrl: "http://x", sessionId: "t" });

  test("พูดสองประโยคติดกันตอนเริ่มคุยต้องไม่หายไปประโยคหนึ่ง", async () => {
    // ของเดิมเก็บ socket ไว้ทันทีที่สร้าง ผู้เรียกคนที่สองจึงส่งข้อความทั้งที่
    // socket ยังอยู่สถานะ CONNECTING ได้ InvalidStateError แล้วประโยคนั้นหายไป
    FakeWebSocket.openDelay = 30;
    const errors = [];
    const talk = new VoiceConversation(client(), {
      onError: (e) => errors.push(e.message),
      sendAudioForSpeakerId: false,
      serverTts: true,
    });
    await talk.start();

    const recognition = FakeSpeechRecognition.instances[0];
    recognition.emitFinal("หนึ่ง", "สอง");
    await sleep(120);

    const socket = FakeWebSocket.instances[0];
    const sent = socket.sent.map((m) => m.text);
    assert.deepEqual(sent, ["หนึ่ง", "สอง"], `ส่งไป: ${sent} | errors: ${errors}`);
    assert.equal(FakeWebSocket.instances.length, 1, "ต้องใช้การเชื่อมต่อเดียว");
    talk.stop();
  });

  test("หยุดฟังระหว่างบอทพูด แล้วกลับมาฟังเมื่อพูดจบ", async () => {
    // ไม่งั้นไมโครโฟนจะได้ยินเสียงบอทเอง ระบบถอดเสียงตัวเองแล้วตอบตัวเองวนไม่จบ
    // และเสียงบอทจะถูกส่งไปสะสมเป็นลายเสียงของผู้ใช้
    FakeWebSocket.openDelay = 5;
    const talk = new VoiceConversation(client(), {
      sendAudioForSpeakerId: false,
      serverTts: true,
      bargeIn: false,
    });
    await talk.start();

    const recognition = FakeSpeechRecognition.instances[0];
    const stoppedBefore = recognition.stopped;
    recognition.emitFinal("สวัสดี");
    await sleep(40);

    assert.ok(recognition.stopped > stoppedBefore, "ต้องหยุดฟังตอนเริ่มตอบ");

    const socket = FakeWebSocket.instances[0];
    socket.emit({ type: "chunk", text: "สวัสดีค่ะ", audio: "QUFB" });
    socket.emit({ type: "done", text: "สวัสดีค่ะ" });
    await sleep(150);

    assert.equal(talk.currentState, "listening", "พูดจบแล้วต้องกลับมาฟัง");
    talk.stop();
  });

  test("สถานะยังเป็นกำลังพูดจนกว่าเสียงจะเล่นจบ", async () => {
    FakeWebSocket.openDelay = 5;
    const states = [];
    const talk = new VoiceConversation(client(), {
      sendAudioForSpeakerId: false,
      serverTts: true,
      onStateChange: (s) => states.push(s),
    });
    await talk.start();
    FakeSpeechRecognition.instances[0].emitFinal("สวัสดี");
    await sleep(30);

    const socket = FakeWebSocket.instances[0];
    socket.emit({ type: "chunk", text: "ตอบ", audio: "QUFB" });
    socket.emit({ type: "done", text: "ตอบ" });
    await sleep(5);

    assert.equal(talk.currentState, "speaking", `สถานะ: ${states}`);
    await sleep(120);
    assert.equal(talk.currentState, "listening");
    talk.stop();
  });

  test("ถูกปฏิเสธไมโครโฟนแล้วยังลองใหม่ได้", async () => {
    // ของเดิมตั้ง running = true ก่อนขอสิทธิ์ พอขอไม่ผ่านก็ค้างสถานะไว้
    // การกดเริ่มใหม่จึงเงียบไปเลยตลอดกาล
    FakeMedia.deny = true;
    const talk = new VoiceConversation(client(), { serverTts: false });
    await assert.rejects(() => talk.start());

    FakeMedia.deny = false;
    await talk.start();
    assert.equal(talk.currentState, "listening");
    talk.stop();
  });

  test("เซิร์ฟเวอร์ส่ง error ต้องไม่ค้างที่สถานะกำลังคิด", async () => {
    FakeWebSocket.openDelay = 5;
    const errors = [];
    const talk = new VoiceConversation(client(), {
      sendAudioForSpeakerId: false,
      onError: (e) => errors.push(e.message),
    });
    await talk.start();
    FakeSpeechRecognition.instances[0].emitFinal("สวัสดี");
    await sleep(30);

    FakeWebSocket.instances[0].emit({ type: "error", text: "โมเดลล่ม" });
    await sleep(20);

    assert.deepEqual(errors, ["โมเดลล่ม"]);
    assert.equal(talk.currentState, "listening");
    talk.stop();
  });

  test("พูดแทรกได้เมื่อเปิด bargeIn", async () => {
    FakeWebSocket.openDelay = 5;
    const talk = new VoiceConversation(client(), {
      sendAudioForSpeakerId: false,
      serverTts: true,
      bargeIn: true,
    });
    await talk.start();
    const recognition = FakeSpeechRecognition.instances[0];
    recognition.emitFinal("สวัสดี");
    await sleep(30);

    const socket = FakeWebSocket.instances[0];
    socket.emit({ type: "chunk", text: "ประโยคหนึ่ง", audio: "QUFB" });
    await sleep(8);
    const playedBefore = FakeAudio.played.length;

    recognition.onspeechstart();
    socket.emit({ type: "chunk", text: "ประโยคสอง", audio: "QkJC" });
    await sleep(80);

    assert.equal(FakeAudio.played.length, playedBefore, "ถูกขัดแล้วต้องไม่เล่นต่อ");
    talk.stop();
  });

  test("หยุดบทสนทนาแล้วต้องคืนไมโครโฟน", async () => {
    const talk = new VoiceConversation(client(), { serverTts: false });
    await talk.start();
    assert.equal(FakeTrack.live, 1);

    talk.stop();
    await sleep(30);
    assert.equal(FakeTrack.live, 0, "ไมโครโฟนยังเปิดค้าง");
    assert.equal(FakeAudioContext.live, 0);
  });
});
