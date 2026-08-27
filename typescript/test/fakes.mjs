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
    // เบราว์เซอร์จริงคิว "pause" เป็น task ไม่ยิงแบบซิงโครนัส
    //
    // การยิงทันทีทำให้หน้าต่างที่คิวเสียงยังรายงานว่า busy หลัง cleanup
    // ดูกว้างแค่ 1 microtask ทั้งที่ของจริงกว้างเต็ม 1 task — บั๊กที่อาศัย
    // หน้าต่างนั้นจึงมองไม่เห็นเลยในชุดทดสอบ
    setTimeout(() => this.onpause?.(), 0);
  }
}

/** MediaStream / AudioContext ปลอม — นับว่ามีอะไรค้างเปิดอยู่บ้าง */
export class FakeAudioContext {
  static live = 0;
  /** อินสแตนซ์ล่าสุดใช้ป้อนเสียงเข้าตัวอัดในเทสต์ได้ */
  static instances = [];
  static reset() {
    FakeAudioContext.live = 0;
    FakeAudioContext.instances = [];
  }

  constructor() {
    FakeAudioContext.live += 1;
    FakeAudioContext.instances.push(this);
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
    this._openTimer = setTimeout(() => {
      // ถูกปิดไประหว่างจับมือแล้ว — เบราว์เซอร์จริงไม่เปิดย้อนหลัง
      //
      // ของเดิมตั้ง readyState กลับเป็น 1 แล้วยิง open ทำให้ฉาก "เซิร์ฟเวอร์
      // ตัดตอนจับมือ" กลายเป็น "ต่อสำเร็จ" ซึ่งกลับหัวกลับหางกับความจริง
      if (this.readyState !== 0) return;
      this.readyState = 1; // OPEN
      this._fire("open", {});
    }, FakeWebSocket.openDelay);
    this._openTimer.unref?.();
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

  /**
   * จำลองเหตุการณ์ที่เซิร์ฟเวอร์ส่งกลับมา
   *
   * socket ที่ปิดแล้วไม่ส่งข้อความต่ออีก — เบราว์เซอร์จริงเป็นแบบนั้น
   * ของปลอมที่ยังส่งต่อใจดีเกินไป และซ่อนผลของการไม่ปิด socket ไว้
   */
  emit(event) {
    if (this.readyState === 3) return;
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
    clearTimeout(this._openTimer);
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

  /**
   * Chrome จบการถอดเสียงเองเป็นระยะแม้ตั้ง continuous = true
   *
   * `onend` มาถึงโดยไม่มีใครสั่ง ซึ่งเป็นเหตุผลเดียวที่บล็อกสั่งเริ่มใหม่ใน
   * `onend` มีอยู่ ของปลอมเดิมยิง `onend` เฉพาะตอน stop()/abort() ซึ่งตอนนั้น
   * ตัวจัดการถูกปลดหรือไมค์ถูกพักไปแล้วเสมอ บล็อกนั้นจึงไม่เคยถูกรันเลย
   */
  endSpontaneously() {
    if (!this.running) return false;
    this.running = false;
    this.onend?.();
    return true;
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
      // เบราว์เซอร์จริงคืนรายชื่อเสียงที่โหลดมาแล้ว และมักว่างในตอนแรก
      // ของเดิมคืน [] เสมอ เส้นทางเลือกเสียงไทยจึงไม่เคยถูกทดสอบเลย
      voices: [],
      getVoices() {
        return this.voices;
      },
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
      /** ชิ้นที่ยังพูดไม่จบ — ใช้ยิง error ตอนถูกยกเลิกเหมือนเบราว์เซอร์จริง */
      speaking: [],
      /** ตั้งเป็น false ถ้าเทสต์อยากคุมจังหวะ onend เอง */
      autoEnd: true,
      speak(utterance) {
        spoken.push(utterance);
        this.speaking.push(utterance);
        // เบราว์เซอร์จริงยิง end เมื่อพูดจบ ของเดิมไม่ยิงเลย ตัวนับชิ้นที่ค้าง
        // จึงไม่เคยลดลง และยามที่กันชิ้นเก่าไม่ให้ขโมยการนับของชิ้นใหม่
        // กลายเป็นโค้ดที่ไม่มีทางถูกรัน
        if (!this.autoEnd) return;
        setTimeout(() => {
          const at = this.speaking.indexOf(utterance);
          if (at < 0) return; // ถูกยกเลิกไปก่อนแล้ว
          this.speaking.splice(at, 1);
          utterance.onend?.();
        }, 5);
      },
      cancel() {
        this.cancelled += 1;
        // เบราว์เซอร์จริงยิง error ให้ทุกชิ้นที่ค้าง *เป็น task* ไม่ใช่ microtask
        // ลำดับนี้สำคัญ เพราะโค้ดจริงเรียก speak() ชิ้นใหม่แบบซิงโครนัสต่อจาก
        // cancel() ถ้าใช้ microtask จะแยกชิ้นเก่ากับชิ้นใหม่ไม่ออก
        const ค้างอยู่ = this.speaking.splice(0);
        setTimeout(() => {
          for (const utterance of ค้างอยู่) utterance.onerror?.({ error: "canceled" });
        }, 0);
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
