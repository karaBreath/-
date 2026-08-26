/**
 * ตัวปลอมของ Web API สำหรับทดสอบ browserVoice.js ใน Node
 *
 * บัคที่หนักที่สุดของฝั่งเบราว์เซอร์อยู่ในลำดับเวลา (คิวเสียง การขัดจังหวะ
 * การเปิด/ปิดไมโครโฟนซ้อนกัน) ซึ่งอ่านโค้ดเฉย ๆ แล้วมองไม่เห็น จึงต้องรันจริง
 */

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** ตัวเล่นเสียงปลอม — บันทึกลำดับการเล่นและการถูกสั่งหยุด */
export class FakeAudio {
  static played = [];
  static live = 0;

  static reset() {
    FakeAudio.played = [];
    FakeAudio.live = 0;
  }

  constructor(url) {
    this.url = url;
    this.onended = null;
    this.onerror = null;
    this.onpause = null;
    this.paused = false;
    this._timer = null;
  }

  play() {
    FakeAudio.played.push(this.url);
    FakeAudio.live += 1;
    this._timer = setTimeout(() => {
      if (this.paused) return;
      FakeAudio.live -= 1;
      this.onended?.();
    }, 20);
    return Promise.resolve();
  }

  pause() {
    if (this.paused) return;
    this.paused = true;
    if (this._timer) {
      clearTimeout(this._timer);
      FakeAudio.live -= 1;
    }
    this.onpause?.();
  }
}

/** MediaStream / AudioContext ปลอม — นับว่ามีอะไรค้างเปิดอยู่บ้าง */
export class FakeAudioContext {
  static live = 0;
  static reset() {
    FakeAudioContext.live = 0;
  }

  constructor() {
    FakeAudioContext.live += 1;
    this.sampleRate = 48000;
    this.state = "running";
    this.destination = { name: "destination" };
  }

  createMediaStreamSource() {
    return { connect() {}, disconnect() {} };
  }

  createScriptProcessor() {
    const node = {
      onaudioprocess: null,
      connect() {},
      disconnect() {},
    };
    this._processor = node;
    return node;
  }

  close() {
    if (this.state !== "closed") {
      this.state = "closed";
      FakeAudioContext.live -= 1;
    }
    return Promise.resolve();
  }

  /** จำลองว่ามีเสียงเข้ามา */
  emit(samples) {
    this._processor?.onaudioprocess?.({
      inputBuffer: { getChannelData: () => samples },
    });
  }
}

export class FakeTrack {
  static live = 0;
  static reset() {
    FakeTrack.live = 0;
  }
  constructor() {
    FakeTrack.live += 1;
    this.stopped = false;
  }
  stop() {
    if (!this.stopped) {
      this.stopped = true;
      FakeTrack.live -= 1;
    }
  }
}

/** WebSocket ปลอมที่เปิดช้า เพื่อเปิดโอกาสให้เกิด race */
export class FakeWebSocket {
  static instances = [];
  static openDelay = 15;

  static reset() {
    FakeWebSocket.instances = [];
  }

  constructor(url) {
    this.url = url;
    this.readyState = 0; // CONNECTING
    this.sent = [];
    this._listeners = {};
    FakeWebSocket.instances.push(this);
    setTimeout(() => {
      this.readyState = 1; // OPEN
      this._fire("open", {});
    }, FakeWebSocket.openDelay);
  }

  addEventListener(type, handler) {
    (this._listeners[type] ??= []).push(handler);
  }

  _fire(type, event) {
    for (const handler of this._listeners[type] ?? []) handler(event);
  }

  send(data) {
    if (this.readyState !== 1) throw new Error("InvalidStateError: Still in CONNECTING state.");
    this.sent.push(JSON.parse(data));
  }

  /** จำลองเหตุการณ์ที่เซิร์ฟเวอร์ส่งกลับมา */
  emit(event) {
    this._fire("message", { data: JSON.stringify(event) });
  }

  close() {
    this.readyState = 3;
    this._fire("close", {});
  }
}

/** ตัวถอดเสียงปลอม — สั่งให้ยิงผลลัพธ์ได้เอง */
export class FakeSpeechRecognition {
  static instances = [];
  static reset() {
    FakeSpeechRecognition.instances = [];
  }

  constructor() {
    this.lang = "";
    this.continuous = false;
    this.interimResults = false;
    this.maxAlternatives = 1;
    this.started = 0;
    this.stopped = 0;
    this.aborted = 0;
    this.running = false;
    this.onresult = null;
    this.onerror = null;
    this.onend = null;
    this.onspeechstart = null;
    FakeSpeechRecognition.instances.push(this);
  }

  start() {
    if (this.running) throw new Error("already started");
    this.running = true;
    this.started += 1;
  }

  stop() {
    this.stopped += 1;
    if (this.running) {
      this.running = false;
      this.onend?.();
    }
  }

  abort() {
    this.aborted += 1;
    this.running = false;
  }

  /** ยิงผลลัพธ์สุดท้ายหลายรายการในเหตุการณ์เดียว (เกิดขึ้นจริงกับ Chrome) */
  emitFinal(...texts) {
    const results = texts.map((text) => ({
      isFinal: true,
      length: 1,
      0: { transcript: text, confidence: 0.9 },
    }));
    results.length = texts.length;
    this.onresult?.({ resultIndex: 0, results: { ...results, length: texts.length } });
  }
}

/** ติดตั้ง global ปลอมทั้งหมด */
export function installFakes() {
  FakeAudio.reset();
  FakeAudioContext.reset();
  FakeTrack.reset();
  FakeWebSocket.reset();
  FakeSpeechRecognition.reset();

  const contexts = [];
  globalThis.Audio = FakeAudio;
  globalThis.AudioContext = class extends FakeAudioContext {
    constructor() {
      super();
      contexts.push(this);
      // ให้เทสต์เข้าถึงตัวล่าสุดได้เพื่อจำลองเสียงเข้า
      globalThis.__lastContext = this;
    }
  };
  globalThis.WebSocket = FakeWebSocket;
  globalThis.SpeechRecognition = FakeSpeechRecognition;
  // Node 22 มี navigator เป็น getter อย่างเดียว ต้องนิยามทับด้วย defineProperty
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    writable: true,
    value: {
    mediaDevices: {
      getUserMedia: async () => {
        await sleep(FakeMedia.delay);
        if (FakeMedia.deny) throw new Error("NotAllowedError");
        const track = new FakeTrack();
        return { getTracks: () => [track] };
      },
    },
    },
  });
  globalThis.FileReader = class {
    readAsDataURL() {
      this.result = "data:audio/wav;base64,ZmFrZQ==";
      queueMicrotask(() => this.onloadend?.());
    }
  };
  globalThis.speechSynthesis = undefined;
  return { contexts };
}

export const FakeMedia = { delay: 5, deny: false };
