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

describe("VoiceConversation — บัคที่เจอจากการตรวจรอบสอง", () => {
  const client = () => new ThaiVoiceClient({ baseUrl: "http://x", sessionId: "t" });

  async function started(options = {}) {
    FakeWebSocket.openDelay = 5;
    const talk = new VoiceConversation(client(), {
      sendAudioForSpeakerId: false,
      serverTts: true,
      ...options,
    });
    await talk.start();
    const recognition = FakeSpeechRecognition.instances[0];
    return { talk, recognition };
  }

  test("คิวเสียงว่างชั่วคราวกลางเทิร์นต้องไม่ถือว่าจบเทิร์น", async () => {
    // เซิร์ฟเวอร์สังเคราะห์เสียงประโยคถัดไป *หลัง* ส่งประโยคก่อนหน้าไปแล้ว
    // คิวจึงว่างชั่วคราวเป็นเรื่องปกติ ถ้าถือว่าจบเทิร์นตอนนั้น ไมโครโฟนจะถูก
    // เปิดกลับมาทั้งที่บอทยังพูดอยู่ แล้วระบบจะได้ยินเสียงตัวเองแล้วตอบตัวเอง
    const { talk, recognition } = await started({ bargeIn: false });
    recognition.emitFinal("สวัสดี");
    await sleep(30);

    const socket = FakeWebSocket.instances[0];
    socket.emit({ type: "chunk", text: "ประโยคหนึ่ง", audio: "QUFB" });
    await sleep(60); // ปล่อยให้เสียงประโยคแรกเล่นจบก่อนประโยคถัดไปมาถึง

    assert.equal(talk.currentState, "speaking", "ยังตอบไม่จบ ต้องไม่กลับไปฟัง");
    assert.equal(recognition.running, false, "ไมโครโฟนต้องยังปิดอยู่");

    socket.emit({ type: "chunk", text: "ประโยคสอง", audio: "QkJC" });
    socket.emit({ type: "done", text: "ประโยคหนึ่งประโยคสอง" });
    await sleep(120);

    assert.equal(talk.currentState, "listening");
    assert.equal(recognition.running, true, "จบแล้วต้องกลับมาฟัง");
    talk.stop();
  });

  test("หางของเทิร์นก่อนต้องไม่ถูกนับเป็นคำตอบของเทิร์นใหม่", async () => {
    const replies = [];
    const { talk, recognition } = await started({ bargeIn: true, onReply: (t) => replies.push(t) });

    recognition.emitFinal("คำถามที่หนึ่ง");
    await sleep(30);
    const socket = FakeWebSocket.instances[0];
    socket.emit({ type: "delta", text: "ตอบหนึ่ง-" });

    // ผู้ใช้ถามใหม่ก่อนคำตอบแรกจะจบ
    recognition.onspeechstart();
    recognition.emitFinal("คำถามที่สอง");
    await sleep(30);

    // หางของคำตอบแรกเพิ่งมาถึง
    socket.emit({ type: "delta", text: "ตอบหนึ่งท้าย" });
    socket.emit({ type: "chunk", text: "ตอบหนึ่งท้าย", audio: "T05F" });
    socket.emit({ type: "done", text: "ตอบหนึ่งเต็ม" });
    await sleep(80);

    assert.equal(FakeAudio.played.length, 0, `เสียงของเทิร์นเก่าไม่ควรถูกเล่น: ${FakeAudio.played}`);
    assert.ok(!replies.includes("ตอบหนึ่งเต็ม"), `คำตอบเก่าถูกรายงานเป็นของเทิร์นใหม่: ${replies}`);
    assert.deepEqual(socket.sent.map((m) => m.text), ["คำถามที่หนึ่ง", "คำถามที่สอง"]);
    talk.stop();
  });

  test("การเชื่อมต่อหลุดกลางเทิร์นต้องแจ้งและปลดสถานะ", async () => {
    const errors = [];
    const { talk, recognition } = await started({
      bargeIn: false,
      onError: (e) => errors.push(e.message),
    });
    recognition.emitFinal("สวัสดี");
    await sleep(30);
    assert.equal(talk.currentState, "thinking");

    FakeWebSocket.instances[0].close();
    await sleep(30);

    assert.ok(errors.length > 0, "ต้องบอกผู้ใช้ว่าหลุด");
    assert.equal(talk.currentState, "listening", "ต้องไม่ค้างที่ กำลังคิด");
    assert.equal(recognition.running, true, "ไมโครโฟนต้องถูกเปิดกลับมา");
    talk.stop();
  });

  test("การเชื่อมต่อที่ตายแล้วต้องไม่ล้างตัวที่เพิ่งเปิดใหม่", async () => {
    const { talk, recognition } = await started({ bargeIn: true });

    recognition.emitFinal("หนึ่ง");
    await sleep(30);
    const dead = FakeWebSocket.instances[0];
    dead.close();
    await sleep(20);

    recognition.emitFinal("สอง");
    await sleep(40);

    const open = FakeWebSocket.instances.filter((s) => s.readyState === 1);
    assert.equal(open.length, 1, `มี socket เปิดค้าง ${open.length} ตัว`);
    talk.stop();
  });

  test("โหมดกดพูดต้องส่งเฉพาะช่วงที่กดค้าง", async () => {
    // ของเดิมส่งทั้งหน้าต่างที่สะสมไว้ (ถึง 30 วินาที) ทำให้ทั้งการถอดเสียงและ
    // ลายเสียงคำนวณจากเสียงคนอื่นในห้องที่ดังก่อนหน้าไปด้วย
    let uploaded = null;
    const spyClient = new ThaiVoiceClient({
      baseUrl: "http://x",
      sessionId: "t",
      fetchImpl: async (_url, init) => {
        uploaded = init.body.get("audio");
        return {
          ok: true,
          json: async () => ({
            transcript: "x", reply: "", chunks: [], speaker: null,
            session_id: "t", audio: null, identified_by: null,
          }),
        };
      },
    });
    const talk = new VoiceConversation(spyClient, {
      useBrowserRecognition: false,
      sendAudioForSpeakerId: true,
    });
    await talk.start();
    const context = globalThis.__lastContext;

    for (let i = 0; i < 20; i += 1) context.emit(new Float32Array(4096).fill(0.01)); // เสียงรบกวนก่อนกด
    talk.pushToTalkStart();
    context.emit(new Float32Array(4096).fill(0.5)); // ช่วงที่กดค้าง

    await talk.pushToTalkStop();

    assert.ok(uploaded, "ต้องอัปโหลดไฟล์เสียง");
    // 4096 ตัวอย่างที่ 48 kHz ย่อเหลือ 16 kHz ประมาณ 1365 ตัวอย่าง = ~2774 ไบต์
    assert.ok(uploaded.size < 10000, `อัปโหลดเสียงเกินช่วงที่กด: ${uploaded.size} ไบต์`);
    talk.stop();
  });

  test("กดเริ่มคุยซ้ำระหว่างรอสิทธิ์ไมโครโฟนต้องไม่ได้ไมค์สองตัว", async () => {
    FakeMedia.delay = 40;
    const talk = new VoiceConversation(client(), { serverTts: false });
    await Promise.all([talk.start(), talk.start(), talk.start()]);

    assert.equal(FakeTrack.live, 1, `เปิดไมโครโฟน ${FakeTrack.live} ตัว`);
    assert.equal(FakeAudioContext.live, 1);
    assert.equal(FakeSpeechRecognition.instances.length, 1);

    talk.stop();
    await sleep(40);
    assert.equal(FakeTrack.live, 0);
  });
});

describe("VoiceConversation — บัคที่เจอจากการตรวจรอบสาม", () => {
  const client = () => new ThaiVoiceClient({ baseUrl: "http://x", sessionId: "t" });

  async function started(options = {}) {
    FakeWebSocket.openDelay = 5;
    const talk = new VoiceConversation(client(), {
      sendAudioForSpeakerId: false,
      serverTts: true,
      ...options,
    });
    await talk.start();
    return { talk, recognition: FakeSpeechRecognition.instances.at(-1) };
  }

  test("หยุดแล้วเริ่มใหม่ต้องไม่เหลือ recognizer ผีที่ฟังเสียงบอทเอง", async () => {
    // abort() ยิง end แบบ asynchronous ตามสเปก ถ้าตัวเก่าไม่เช็คว่าตัวเองยังถูกใช้อยู่
    // มันจะปลุกตัวเองกลับมา กลายเป็นสอง recognizer แล้ว pauseRecognition หยุดได้
    // แค่ตัวใหม่ ตัวเก่าจึงถอดเสียงบอทระหว่างบอทพูด แล้วส่งกลับเป็นเทิร์นใหม่
    const { talk } = await started({ bargeIn: false });
    const เก่า = FakeSpeechRecognition.instances.at(-1);

    talk.stop();
    await talk.start(); // ผู้ใช้กด "หยุด" แล้วกด "เริ่มคุย" ใหม่บนตัวเดิม
    const ใหม่ = FakeSpeechRecognition.instances.at(-1);
    await sleep(30); // end ของตัวเก่าเพิ่งมาถึงตอนนี้

    assert.notEqual(ใหม่, เก่า);
    assert.equal(เก่า.running, false, "recognizer ตัวเก่าต้องไม่ปลุกตัวเองกลับมา");

    // ถ้าตัวเก่ายังฟังอยู่ มันจะถอดเสียงบอทระหว่างบอทพูดแล้วส่งเป็นเทิร์นใหม่
    ใหม่.emitFinal("ถาม");
    await sleep(30);
    เก่า.emitFinal("เสียงบอทที่ไมค์ผีได้ยิน");
    await sleep(30);

    const socket = FakeWebSocket.instances.at(-1);
    assert.deepEqual(
      socket.sent.map((m) => m.text),
      ["ถาม"],
      "ไมค์ผีต้องไม่ส่งเสียงบอทกลับไปเป็นเทิร์นใหม่",
    );
    talk.stop();
  });

  test("พูดสองประโยคติดกันต้องได้ยินเสียงตอบของประโยคแรก", async () => {
    // Chrome ส่ง final result สองอันในเหตุการณ์เดียวเป็นเรื่องปกติ
    // ของเดิมเก็บ "เทิร์นที่รับเสียงได้" ไว้ช่องเดียว พอส่งประโยคที่สอง
    // เสียงของเทิร์นแรกที่กำลังไหลมาจึงถูกทิ้งทั้งหมด ผู้ใช้เห็นข้อความ
    // แต่บอทเงียบสนิท
    const replies = [];
    const { talk, recognition } = await started({
      bargeIn: false,
      onReply: (t) => replies.push(t),
    });

    recognition.emitFinal("หนึ่ง", "สอง");
    await sleep(40);

    const socket = FakeWebSocket.instances[0];
    socket.emit({ type: "chunk", text: "ตอบหนึ่ง", audio: "T05F" });
    socket.emit({ type: "done", text: "ตอบหนึ่ง" });
    await sleep(80);

    assert.deepEqual(socket.sent.map((m) => m.text), ["หนึ่ง", "สอง"]);
    assert.ok(
      FakeAudio.played.some((url) => url.includes("T05F")),
      `เสียงของเทิร์นแรกต้องถูกเล่น ไม่ใช่ถูกทิ้ง: ${FakeAudio.played}`,
    );
    assert.deepEqual(replies, ["ตอบหนึ่ง"]);
    talk.stop();
  });

  test("socket เก่าที่ปิดตามมาทีหลังต้องไม่ล้างเทิร์นที่กำลังคุยอยู่", async () => {
    const replies = [];
    const errors = [];
    const { talk, recognition } = await started({
      bargeIn: false,
      onReply: (t) => replies.push(t),
      onError: (e) => errors.push(e.message),
    });

    recognition.emitFinal("ประโยคแรก");
    await sleep(30);
    const เก่า = FakeWebSocket.instances[0];
    เก่า.emit({ type: "done", text: "ตอบแรก" });
    await sleep(30);

    // เซิร์ฟเวอร์ปิด socket เพราะไม่มีใครใช้ แล้วผู้ใช้พูดใหม่ทันที
    เก่า.readyState = 3;
    recognition.emitFinal("ประโยคที่สอง");
    await sleep(40);
    เก่า._fire("close", {}); // close ของตัวเก่าเพิ่งมาถึง

    const ใหม่ = FakeWebSocket.instances.at(-1);
    ใหม่.emit({ type: "done", text: "ตอบสอง" });
    await sleep(40);

    assert.deepEqual(errors, [], `ไม่ควรมีข้อความ "การเชื่อมต่อหลุด" ปลอม: ${errors}`);
    assert.deepEqual(replies, ["ตอบแรก", "ตอบสอง"]);
    talk.stop();
  });

  test("ชิ้นเสียงที่ยิงทั้ง error และ end ต้องไม่ทำให้ไมค์เปิดกลางประโยค", async () => {
    // เครื่องสังเคราะห์เสียงส่วนใหญ่ยิงทั้งสองเหตุการณ์เมื่อชิ้นถูกขัดจังหวะ
    // ของเดิมลดตัวนับสองครั้งต่อชิ้นเดียว ไปขโมยการนับของชิ้นอื่น
    const { talk, recognition } = await started({ bargeIn: false, serverTts: false });
    recognition.emitFinal("ถาม");
    await sleep(30);

    const socket = FakeWebSocket.instances[0];
    socket.emit({ type: "chunk", text: "ประโยคหนึ่ง" });
    socket.emit({ type: "chunk", text: "ประโยคสอง" });
    socket.emit({ type: "done", text: "ประโยคหนึ่งประโยคสอง" });
    await sleep(20);

    const [หนึ่ง, สอง] = globalThis.speechSynthesis.spoken.slice(-2);
    หนึ่ง.onerror?.();
    หนึ่ง.onend?.();
    await sleep(20);

    assert.equal(talk.currentState, "speaking", "ชิ้นที่สองยังพูดอยู่");
    assert.equal(recognition.running, false, "ไมโครโฟนต้องยังปิดอยู่");

    สอง.onend?.();
    await sleep(20);
    assert.equal(talk.currentState, "listening");
    talk.stop();
  });

  test("error ที่ไม่ผูกกับเทิร์นไหนต้องไม่หายเงียบ", async () => {
    const errors = [];
    const { talk, recognition } = await started({ onError: (e) => errors.push(e.message) });
    recognition.emitFinal("ทักทาย");
    await sleep(30);

    const socket = FakeWebSocket.instances[0];
    socket.emit({ type: "done", text: "สวัสดี" });
    await sleep(30);
    socket.emit({ type: "error", text: "เฟรมไม่ถูกต้อง" });
    await sleep(10);

    assert.deepEqual(errors, ["เฟรมไม่ถูกต้อง"]);
    talk.stop();
  });
});
