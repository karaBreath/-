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
    // unref เพื่อให้ openDelay ที่ตั้งไว้นาน ๆ (จำลองการจับมือที่ค้าง)
    // ไม่กันไม่ให้โปรเซสของเทสต์จบ
    const timer = setTimeout(() => {
      this.readyState = 1; // OPEN
      this._fire("open", {});
    }, FakeWebSocket.openDelay);
    timer.unref?.();
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

  /**
   * ปิดการเชื่อมต่อ — เหตุการณ์ close ต้องมาแบบ asynchronous
   *
   * เบราว์เซอร์จริงไม่เคยยิง close แบบซิงโครนัสใน close() ของเดิมยิงทันที
   * ซึ่งใจดีกว่าความจริง แล้วบัคที่เกิดจาก socket เก่าปิดตามมาทีหลังจึงซ่อนอยู่
   */
  close() {
    if (this.readyState === 3) return;
    this.readyState = 3;
    queueMicrotask(() => this._fire("close", {}));
  }

  /** ปิดแบบซิงโครนัส (ใช้เฉพาะเทสต์ที่ต้องการลำดับแน่นอน) */
  closeNow() {
    if (this.readyState === 3) return;
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
    this.failMode = null;
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
    // ใช้ setTimeout ไม่ใช่ queueMicrotask — ถ้าโค้ดวนเริ่มใหม่ไม่จบ ลูป
    // microtask จะกิน event loop จนเทสต์แขวน แทนที่จะล้มให้เห็น
    // และต้องมีเพดาน ไม่งั้น timer ที่นัดกันเองไม่จบจะทำให้โปรเซสไม่ยอมออก
    if (this.failMode && this.started <= 20) {
      setTimeout(() => this.fail(this.failMode), 0).unref?.();
    }
  }

  stop() {
    this.stopped += 1;
    if (this.running) {
      this.running = false;
      this.onend?.();
    }
  }

  /**
   * ยกเลิกทันที — แต่เหตุการณ์ end มาทีหลังแบบ asynchronous ตามสเปก
   *
   * ของเดิมไม่ยิง end เลย ซึ่งใจดีกว่าเบราว์เซอร์จริง บัค recognizer ผีจึงซ่อนอยู่
   */
  abort() {
    this.aborted += 1;
    this.running = false;
    queueMicrotask(() => this.onend?.());
  }

  /**
   * จำลองข้อผิดพลาดแบบที่เบราว์เซอร์จริงยิง — error แล้ว end ตามมา
   *
   * ของจริงหยุดทำงานก่อนยิง end เสมอ เรียก onend() เปล่า ๆ ในเทสต์จึงใจดี
   * กว่าความจริง เพราะ start() รอบถัดไปจะโยน "already started" ทิ้งไปเอง
   */
  fail(code) {
    this.onerror?.({ error: code });
    this.running = false;
    this.onend?.();
  }

  /**
   * ทำให้ทุก start() ต่อจากนี้ล้มด้วยรหัสเดิม
   *
   * นี่คือพฤติกรรมจริงเมื่อผู้ใช้ถอนสิทธิ์ไมโครโฟนหรือถอดอุปกรณ์ออก —
   * เริ่มใหม่กี่ครั้งก็ล้มแบบเดิมทันที ของปลอมที่ล้มแค่ครั้งเดียวใจดีเกินไป
   * จนบัคการวนเริ่มใหม่ไม่โผล่
   */
  failForever(code) {
    this.failMode = code;
    this.fail(code);
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

/**
 * เครื่องสังเคราะห์เสียงปลอม — ควบคุมได้ว่าชิ้นไหนจบเมื่อไร
 *
 * ของจริงยิงทั้ง error และ end เมื่อชิ้นถูกขัดจังหวะ ซึ่งเป็นที่มาของบัคการนับ
 */
export class FakeSpeechSynthesis {
  static install() {
    const spoken = [];
    globalThis.SpeechSynthesisUtterance = class {
      constructor(text) {
        this.text = text;
        this.onend = null;
        this.onerror = null;
      }
    };
    globalThis.speechSynthesis = {
      spoken,
      cancelled: 0,
      getVoices: () => [],
      listeners: [],
      addEventListener(type, handler) {
        // เบราว์เซอร์จริงกันตัวจัดการซ้ำให้เอง ของปลอมก็ต้องทำเหมือนกัน
        if (!this.listeners.some((l) => l.type === type && l.handler === handler)) {
          this.listeners.push({ type, handler });
        }
      },
      removeEventListener(type, handler) {
        this.listeners = this.listeners.filter(
          (l) => !(l.type === type && l.handler === handler),
        );
      },
      speak(utterance) {
        spoken.push(utterance);
      },
      cancel() {
        this.cancelled += 1;
      },
    };
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
  FakeSpeechSynthesis.install();
  return { contexts };
}

export const FakeMedia = { delay: 5, deny: false };
