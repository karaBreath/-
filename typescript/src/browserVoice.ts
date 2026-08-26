/**
 * คุยด้วยเสียงภาษาไทยในเบราว์เซอร์
 *
 * เบราว์เซอร์ตระกูล Chromium ถอดเสียงภาษาไทย (`th-TH`) ได้เองผ่าน Web Speech API
 * และพูดภาษาไทยได้ผ่าน `speechSynthesis` โมดูลนี้จึงทำให้คุยได้โดยไม่ต้องติดตั้ง
 * โมเดลถอดเสียงใด ๆ บนเซิร์ฟเวอร์
 *
 * แต่ Web Speech API ไม่คืน "ไฟล์เสียง" ให้ ซึ่งเป็นสิ่งที่ระบบต้องใช้จดจำว่าใครพูด
 * เราจึงอัดเสียงดิบคู่ขนานไปด้วย (`PcmRecorder`) แล้วส่งไปพร้อมข้อความ
 *
 * ข้อจำกัดที่ควรรู้:
 * - Web Speech API ของ Chrome ส่งเสียงไปประมวลผลที่เซิร์ฟเวอร์ของ Google
 *   ถ้าต้องการให้ทุกอย่างอยู่ในเครื่อง ให้ปิด `useBrowserRecognition` แล้วให้
 *   เซิร์ฟเวอร์ถอดเสียงด้วย Whisper แทน (โหมดกดพูด)
 * - Safari และ Firefox ยังไม่รองรับ ให้ใช้โหมดกดพูดแทน
 */

import type { ThaiVoiceClient, StreamConnection } from "./client.js";
import type { Speaker, StreamEvent } from "./types.js";

// ── ประกาศชนิดของ Web Speech API (ยังไม่อยู่ใน lib.dom มาตรฐาน) ──────────────
interface SpeechRecognitionAlternativeLike {
  transcript: string;
  confidence: number;
}
interface SpeechRecognitionResultLike {
  isFinal: boolean;
  length: number;
  // ใช้ index signature ไม่ใช่พร็อพเพอร์ตี้ตายตัว เพราะผลลัพธ์ที่ไม่มีตัวเลือกเลย
  // เกิดขึ้นได้จริง ถ้าประกาศเป็น 0 แบบบังคับ TypeScript จะไม่เตือนให้เช็ค
  [index: number]: SpeechRecognitionAlternativeLike | undefined;
}
interface SpeechRecognitionEventLike extends Event {
  resultIndex: number;
  results: { length: number; [index: number]: SpeechRecognitionResultLike | undefined };
}
interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: Event) => void) | null;
  onend: (() => void) | null;
  onspeechstart: (() => void) | null;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function getRecognitionCtor(): SpeechRecognitionCtor | null {
  const w = globalThis as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/** เบราว์เซอร์นี้ถอดเสียงไทยเองได้ไหม */
export function browserRecognitionAvailable(): boolean {
  return getRecognitionCtor() !== null;
}

// ── เข้ารหัส WAV ────────────────────────────────────────────────────────────
/** แปลงตัวอย่างเสียง (-1..1) เป็นไฟล์ WAV 16-bit mono */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true); // ขนาดของ fmt chunk
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/** ลดอัตราสุ่มตัวอย่างแบบเชิงเส้น — พอสำหรับเสียงพูด */
function resample(input: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return input;
  const ratio = from / to;
  const length = Math.floor(input.length / ratio);
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const position = i * ratio;
    const low = Math.floor(position);
    const high = Math.min(low + 1, input.length - 1);
    const weight = position - low;
    output[i] = (input[low] ?? 0) * (1 - weight) + (input[high] ?? 0) * weight;
  }
  return output;
}

/**
 * อัดเสียงดิบจากไมโครโฟนเพื่อใช้จดจำลายเสียง
 *
 * ตั้งใจให้เป็นตัวอัดตัวเดียวที่เปิดค้างไว้ตลอดบทสนทนา แล้วใช้ `take()` ตัดเอา
 * ช่วงของแต่ละประโยคออกมา แทนการ start/stop ใหม่ทุกประโยค
 *
 * เหตุผล: `start()` ต้องรอ `getUserMedia` ซึ่งใช้เวลาได้ถึงหลายวินาที ถ้ามีการ
 * `stop()` เข้ามาระหว่างนั้น ตัวอัดจะถูกสร้างเสร็จ *หลัง* ถูกสั่งหยุด กลายเป็น
 * AudioContext และ track ไมโครโฟนที่ค้างเปิดไว้ตลอดกาล (ไฟไมค์ในเบราว์เซอร์ไม่ดับ)
 * และเสียงของประโยคนั้นก็หายไปด้วย
 *
 * เก็บเสียงแบบหน้าต่างเลื่อน (`maxSeconds`) เพื่อไม่ให้ความเงียบยาว ๆ ก่อนผู้ใช้
 * เริ่มพูดสะสมจนไฟล์ใหญ่เกินจำเป็น แต่ยังไม่ตัดพยางค์แรกทิ้ง ซึ่งสำคัญมากกับ
 * ภาษาไทยเพราะเสียงวรรณยุกต์จะเพี้ยน
 */
export class PcmRecorder {
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private stream: MediaStream | null = null;
  private buffers: Float32Array[] = [];
  private bufferedSamples = 0;
  private starting: Promise<void> | null = null;
  private disposed = false;

  constructor(
    readonly targetSampleRate = 16000,
    readonly maxSeconds = 30,
  ) {}

  get recording(): boolean {
    return this.processor !== null;
  }

  /** ขอสิทธิ์ไมโครโฟนและเริ่มอัด — เรียกซ้ำได้ และเริ่มใหม่ได้หลัง stop() */
  async start(): Promise<void> {
    if (this.recording) return;
    // ต้องเช็ค starting *ก่อน* ล้าง disposed
    //
    // ของเดิมล้างก่อน ทำให้ start() ที่เรียกระหว่าง stop() ยังทำงานค้างอยู่
    // ไปปลุก getUserMedia ที่กำลังจะถูกทิ้งกลับมา แล้ว stop() ก็รื้อทุกอย่าง
    // ทิ้งตามหลัง ผู้เรียกจึงได้ recorder ที่บอกว่าสำเร็จแต่ไม่ได้อัดอะไรเลย
    if (this.starting) return this.starting;
    this.disposed = false;

    this.starting = (async () => {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      if (this.disposed) {
        // ถูกสั่งหยุดระหว่างรอสิทธิ์ไมโครโฟน — ต้องคืนอุปกรณ์ทันที
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      const context = new AudioContext();
      const source = context.createMediaStreamSource(stream);
      // ScriptProcessorNode ถูกประกาศเลิกใช้แล้วแต่ยังรองรับทุกเบราว์เซอร์ที่ใช้จริง
      // และงานนี้ไม่ต้องการความแม่นยำระดับ AudioWorklet
      const processor = context.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        this.append(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      // ต่อเข้า destination ด้วย ไม่งั้นบางเบราว์เซอร์จะไม่เรียก onaudioprocess
      processor.connect(context.destination);

      this.stream = stream;
      this.context = context;
      this.source = source;
      this.processor = processor;
    })();

    try {
      await this.starting;
    } finally {
      this.starting = null;
    }
  }

  private append(chunk: Float32Array): void {
    this.buffers.push(chunk);
    this.bufferedSamples += chunk.length;
    const limit = (this.context?.sampleRate ?? 48000) * this.maxSeconds;
    while (this.bufferedSamples > limit && this.buffers.length > 1) {
      const dropped = this.buffers.shift();
      this.bufferedSamples -= dropped?.length ?? 0;
    }
  }

  /** ทิ้งเสียงที่อัดไว้ แต่ยังอัดต่อ */
  reset(): void {
    this.buffers = [];
    this.bufferedSamples = 0;
  }

  /** ตัดเอาเสียงที่อัดไว้ออกมาเป็นไฟล์ WAV แล้วเริ่มสะสมใหม่ (ยังอัดต่อ) */
  take(): Blob | null {
    if (this.bufferedSamples === 0 || !this.context) return null;
    const sourceRate = this.context.sampleRate;
    const merged = new Float32Array(this.bufferedSamples);
    let offset = 0;
    for (const chunk of this.buffers) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    this.reset();
    return encodeWav(resample(merged, sourceRate, this.targetSampleRate), this.targetSampleRate);
  }

  /** ปิดไมโครโฟนและคืนทรัพยากรทั้งหมด — เรียกซ้ำได้ */
  async stop(): Promise<Blob | null> {
    this.disposed = true;
    if (this.starting) {
      // รอให้ start() ที่ค้างอยู่จบก่อน ไม่งั้นมันจะสร้าง AudioContext ขึ้นมา
      // หลังเราปิดไปแล้ว
      try {
        await this.starting;
      } catch {
        // ขอสิทธิ์ไมโครโฟนไม่สำเร็จ — ไม่มีอะไรต้องคืน
      }
    }
    const captured = this.context ? this.take() : null;

    if (this.processor) this.processor.onaudioprocess = null;
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.context && this.context.state !== "closed") void this.context.close();

    this.processor = null;
    this.source = null;
    this.stream = null;
    this.context = null;
    this.reset();
    return captured;
  }
}

/**
 * เล่นเสียงตอบทีละประโยคตามลำดับ และหยุดได้ทันทีเมื่อผู้ใช้พูดแทรก
 *
 * ผูกกับ "เทิร์น" เพราะเซิร์ฟเวอร์ยังส่งประโยคที่เหลือของเทิร์นเดิมตามมาอีก
 * หลังผู้ใช้พูดแทรกไปแล้ว ถ้าไม่กรองตามเทิร์น บอทจะกลับมาพูดต่อทั้งที่ถูกขัด
 */
export class AudioQueue {
  private queue: { url: string; turn: number }[] = [];
  private current: HTMLAudioElement | null = null;
  private currentTurn = -1;
  private playing = false;
  /**
   * เทิร์นที่ยังรับเสียงอยู่ — ต้องเป็น *เซ็ต* ไม่ใช่ตัวเดียว
   *
   * ผู้ใช้พูดสองประโยคติดกันได้ง่ายมาก (Chrome ส่ง final result สองอันในเหตุการณ์
   * เดียวเป็นเรื่องปกติ) เซิร์ฟเวอร์ตอบทีละเทิร์นตามลำดับ ของเดิมเก็บเทิร์นที่
   * รับได้ไว้ช่องเดียว พอส่งประโยคที่สอง ช่องนั้นเลื่อนไปเทิร์น 2 ทันที
   * เสียงของเทิร์น 1 ที่กำลังไหลมาจึงถูกทิ้งทั้งหมด — ผู้ใช้เห็นข้อความตอบ
   * แต่บอทเงียบสนิท
   */
  private accepted = new Set<number>();
  /** เพิ่มขึ้นทุกครั้งที่ถูกสั่งหยุด ใช้ให้ลูปเดิมที่ค้างอยู่รู้ว่าต้องเลิก */
  private generation = 0;

  constructor(private readonly onIdle?: () => void) {}

  /** เริ่มรับเสียงของเทิร์นนี้ */
  accept(turn: number): void {
    this.accepted.add(turn);
  }

  /** เพิ่มเสียง base64 เข้าคิวของเทิร์นนั้น */
  push(turn: number, base64: string, mime = "audio/mpeg"): void {
    if (!this.accepted.has(turn)) return;
    this.queue.push({ url: `data:${mime};base64,${base64}`, turn });
    void this.drain();
  }

  /** เลิกรับและทิ้งเสียงของเทิร์นเดียว (ใช้ตอนผู้ใช้พูดแทรกเทิร์นนั้น) */
  discard(turn: number): void {
    this.accepted.delete(turn);
    this.queue = this.queue.filter((item) => item.turn !== turn);
    if (this.currentTurn === turn) {
      const audio = this.current;
      this.current = null;
      audio?.pause();
    }
  }

  /** ลืมเทิร์นทั้งหมดที่จบไปแล้ว — กันเซ็ตโตไม่หยุดในบทสนทนายาว ๆ */
  reset(): void {
    if (!this.busy) this.accepted.clear();
  }

  /** หยุดทันที ล้างคิว และไม่รับเสียงของเทิร์นไหนอีก */
  stop(): void {
    this.generation += 1;
    this.accepted.clear();
    this.queue = [];
    const audio = this.current;
    this.current = null;
    this.currentTurn = -1;
    audio?.pause();
  }

  get busy(): boolean {
    return this.playing || this.queue.length > 0;
  }

  private async drain(): Promise<void> {
    if (this.playing) return;
    this.playing = true;
    try {
      while (this.queue.length > 0) {
        const item = this.queue.shift();
        if (!item) continue;
        // ไม่กรองตามเซ็ตตรงนี้ — ของที่เข้าคิวมาแล้วคือของที่รับไว้แล้ว
        // ส่วนของที่ต้องทิ้งถูกลบออกจากคิวไปตั้งแต่ discard/stop
        const generation = this.generation;
        this.currentTurn = item.turn;
        await this.playOne(item.url, generation);
        if (generation !== this.generation) break; // ถูกขัดจังหวะระหว่างเล่น
      }
    } finally {
      this.playing = false;
      if (this.queue.length > 0) {
        void this.drain(); // มีของเข้ามาระหว่างที่ลูปเดิมกำลังจบ
      } else {
        this.onIdle?.();
      }
    }
  }

  private playOne(url: string, generation: number): Promise<void> {
    return new Promise<void>((resolve) => {
      const audio = new Audio(url);
      if (generation === this.generation) this.current = audio;
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        if (this.current === audio) this.current = null;
        resolve();
      };
      audio.onended = finish;
      audio.onerror = finish;
      // ต้องปลด promise ตอนถูก pause ด้วย ไม่งั้นการพูดแทรกจะทำให้ลูปค้างตลอดกาล
      // (pause() ไม่ยิงทั้ง ended และ error)
      audio.onpause = finish;
      void audio.play().catch(finish);
    });
  }
}

/** พูดภาษาไทยด้วยเสียงของเบราว์เซอร์เอง (ใช้เมื่อเซิร์ฟเวอร์ไม่ได้สังเคราะห์เสียงให้) */
export class ThaiSpeechSynthesis {
  private voice: SpeechSynthesisVoice | null = null;
  private pending = 0;
  private accepted = new Set<number>();
  /** เพิ่มขึ้นทุกครั้งที่ถูกสั่งหยุด ใช้ให้เหตุการณ์ของชิ้นเก่าที่ยังไหลมารู้ว่าต้องเลิก */
  private generation = 0;

  constructor(
    private readonly onIdle?: () => void,
    readonly rate = 1.05,
    readonly pitch = 1.0,
  ) {
    this.pickVoice();
    // รายชื่อเสียงในบางเบราว์เซอร์โหลดแบบ asynchronous
    globalThis.speechSynthesis?.addEventListener?.("voiceschanged", () => this.pickVoice());
  }

  get available(): boolean {
    return typeof globalThis.speechSynthesis !== "undefined";
  }

  get busy(): boolean {
    return this.pending > 0;
  }

  accept(turn: number): void {
    this.accepted.add(turn);
  }

  discard(turn: number): void {
    this.accepted.delete(turn);
  }

  reset(): void {
    if (!this.busy) this.accepted.clear();
  }

  private pickVoice(): void {
    const voices = globalThis.speechSynthesis?.getVoices?.() ?? [];
    this.voice = voices.find((v) => v.lang?.toLowerCase().startsWith("th")) ?? null;
  }

  speak(turn: number, text: string): void {
    if (!this.available || !text.trim() || !this.accepted.has(turn)) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "th-TH";
    utterance.rate = this.rate;
    utterance.pitch = this.pitch;
    if (this.voice) utterance.voice = this.voice;
    this.pending += 1;
    const generation = this.generation;
    // ชิ้นที่ถูกขัดจังหวะยิงทั้ง error และ end เครื่องสังเคราะห์เสียงส่วนใหญ่
    // ทำแบบนั้น ของเดิมไม่มีตัวกัน ตัวนับจึงลดสองครั้งต่อชิ้นเดียว ไปขโมย
    // การนับของชิ้นอื่น busy กลายเป็น false ทั้งที่บอทยังพูดอยู่ ระบบจึงจบ
    // เทิร์นและเปิดไมโครโฟนกลับมากลางประโยค แล้วได้ยินเสียงบอทเอง
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      // ถ้าถูก stop() ไปแล้ว ตัวนับถูกล้างไปพร้อมกัน อย่าลดซ้ำ
      if (generation !== this.generation) return;
      this.pending = Math.max(0, this.pending - 1);
      if (this.pending === 0) this.onIdle?.();
    };
    utterance.onend = finish;
    utterance.onerror = finish;
    globalThis.speechSynthesis.speak(utterance);
  }

  stop(): void {
    this.generation += 1;
    this.accepted.clear();
    this.pending = 0;
    globalThis.speechSynthesis?.cancel();
  }
}

export interface VoiceConversationHandlers {
  /** ได้ยินผู้ใช้พูด (ระหว่างพูดจะเป็นข้อความชั่วคราว) */
  onTranscript?: (text: string, isFinal: boolean) => void;
  /** ระบบรู้แล้วว่ากำลังคุยกับใคร */
  onSpeaker?: (speaker: Speaker | null, method?: string, score?: number) => void;
  /** คำตอบกำลังไหลออกมา */
  onReplyDelta?: (text: string) => void;
  /** จบหนึ่งเทิร์น */
  onReply?: (text: string) => void;
  onStateChange?: (state: VoiceState) => void;
  onError?: (error: Error) => void;
}

export type VoiceState = "idle" | "listening" | "thinking" | "speaking";

export interface VoiceConversationOptions extends VoiceConversationHandlers {
  /** ใช้ Web Speech API ของเบราว์เซอร์ถอดเสียง (เร็วกว่า) หรือใช้โหมดกดพูด */
  useBrowserRecognition?: boolean;
  /** ส่งเสียงดิบไปด้วยเพื่อให้ระบบจำได้ว่าใครพูด */
  sendAudioForSpeakerId?: boolean;
  /** ให้เซิร์ฟเวอร์สังเคราะห์เสียงตอบ (คุณภาพดีกว่าเสียงในเบราว์เซอร์) */
  serverTts?: boolean;
  /**
   * ฟังต่อระหว่างบอทกำลังพูด เพื่อให้พูดแทรกได้
   *
   * ต้องใส่หูฟัง ไม่งั้นไมโครโฟนจะได้ยินเสียงบอทเองแล้วตัดตัวเองทันทีที่เริ่มพูด
   * ค่าเริ่มต้นคือปิด ซึ่งจะ *หยุดฟัง* ระหว่างบอทพูด เพื่อไม่ให้ระบบถอดเสียงของ
   * ตัวเองแล้วตอบตัวเองวนไม่จบ และไม่ให้เสียงบอทถูกนำไปสะสมเป็นลายเสียงของผู้ใช้
   */
  bargeIn?: boolean;
}

/**
 * บทสนทนาด้วยเสียงแบบครบวงจรในเบราว์เซอร์
 *
 * ```ts
 * const talk = new VoiceConversation(client, {
 *   onTranscript: (t, final) => console.log(t),
 *   onReply: (t) => console.log("บอท:", t),
 * });
 * await talk.start();
 * ```
 */
/** เทิร์นที่ส่งไปแล้วแต่ยังตอบไม่จบ */
interface PendingTurn {
  id: number;
  reply: string;
  /**
   * socket ที่ส่งเทิร์นนี้ออกไป
   *
   * จำเป็นเพราะเมื่อ socket ตาย ต้องรู้ว่าเทิร์นไหนตายไปกับมันบ้าง ของเดิม
   * ล้างเทิร์นทั้งกอง (ฆ่าเทิร์นใหม่ที่ยังไม่ได้ส่งไปด้วย) หรือไม่ล้างเลย
   * (เทิร์นเก่าค้างในคิวตลอดกาล ดูดเหตุการณ์ของเทิร์นใหม่ไปหมด สถานะค้างที่
   * "กำลังคิด" ไมโครโฟนไม่เปิดกลับมา และไม่มีข้อความบอกผู้ใช้เลย)
   */
  stream?: StreamConnection;
  /**
   * ผู้ใช้พูดแทรกตัดเทิร์นนี้ไปแล้ว
   *
   * ต้องคาไว้ในคิวต่อ ไม่ใช่เอาออกทันที เพราะเซิร์ฟเวอร์ยังส่งประโยคที่เหลือ
   * ของเทิร์นนี้ตามมาอีกจนถึง done ถ้าเอาออกก่อน เหตุการณ์ที่เหลือจะถูกจับคู่
   * กับเทิร์นถัดไปแทน ซึ่งเป็นบัคเดิมในรูปแบบใหม่
   */
  abandoned: boolean;
}

/**
 * รหัสข้อผิดพลาดที่เริ่มการถอดเสียงใหม่ไปก็ล้มซ้ำทันที
 *
 * ทั้งสามอย่างนี้ต้องให้ผู้ใช้ไปแก้ที่เบราว์เซอร์หรืออุปกรณ์ก่อน
 */
const FATAL_RECOGNITION_ERRORS = new Set([
  "not-allowed",
  "service-not-allowed",
  "audio-capture",
]);

const FATAL_RECOGNITION_MESSAGES: Record<string, string> = {
  "not-allowed": "เบราว์เซอร์ไม่อนุญาตให้ใช้ไมโครโฟน กรุณาอนุญาตแล้วกดเริ่มคุยใหม่",
  "service-not-allowed": "เบราว์เซอร์ไม่อนุญาตให้ใช้บริการถอดเสียง กรุณาลองใหม่อีกครั้ง",
  "audio-capture": "ไม่พบไมโครโฟน กรุณาเสียบไมโครโฟนแล้วกดเริ่มคุยใหม่",
};

export class VoiceConversation {
  private recognition: SpeechRecognitionLike | null = null;
  private recorder: PcmRecorder | null = null;
  private readonly audio: AudioQueue;
  private readonly synth: ThaiSpeechSynthesis;
  private state: VoiceState = "idle";
  private running = false;
  private starting = false;
  private recognitionPaused = false;
  /** เจอ error ที่เริ่มใหม่ไปก็ล้มซ้ำ — หยุดวนแล้วรอให้ผู้ใช้กดเริ่มใหม่เอง */
  private recognitionFatal = false;
  private stream: StreamConnection | null = null;
  private streamReady: Promise<StreamConnection> | null = null;
  private turnId = 0;
  /**
   * คิวเทิร์นที่ยังตอบไม่จบ เรียงตามลำดับที่ส่งไป
   *
   * เซิร์ฟเวอร์ตอบทีละเทิร์นตามลำดับและไม่ได้แนบเลขเทิร์นกลับมา เหตุการณ์ที่
   * ไหลเข้ามาจึงเป็นของ "เทิร์นที่เก่าที่สุดที่ยังไม่จบ" เสมอ ไม่ใช่เทิร์นล่าสุด
   * ที่เพิ่งส่งไป การอ่านเลขเทิร์นล่าสุด ณ เวลาที่เหตุการณ์มาถึงทำให้หางของ
   * เทิร์นก่อนถูกนับเป็นของเทิร์นใหม่ทั้งเสียงและข้อความ
   */
  private pending: PendingTurn[] = [];
  /** จำนวนเทิร์นโหมดกดพูดที่ยังรอคำตอบทาง HTTP อยู่ (ไม่ผ่านคิว pending) */
  private awaitingDirect = 0;
  /**
   * ต่อคิวการส่งไว้เป็นสายเดียว
   *
   * Chrome ส่งผลถอดเสียงสุดท้ายสองอันในเหตุการณ์เดียวได้ ``submit`` ผลัก
   * เข้าคิว pending แบบซิงโครนัสแต่ส่งจริงแบบ asynchronous ประโยคแรกต้องรอ
   * แปลงเสียงเป็น base64 ส่วนประโยคที่สองไม่มีเสียงให้แปลง (ประโยคแรกดูดไป
   * หมดแล้ว) จึงแซงไปถึงเซิร์ฟเวอร์ก่อน บอทตอบประโยคที่สองก่อน บันทึกบทสนทนา
   * สลับลำดับ และทุกเหตุการณ์ของคำตอบนั้นถูกจับคู่กับเทิร์นผิดตัว
   */
  private sendChain: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly client: ThaiVoiceClient,
    private readonly options: VoiceConversationOptions = {},
  ) {
    this.audio = new AudioQueue(() => this.onPlaybackIdle());
    this.synth = new ThaiSpeechSynthesis(() => this.onPlaybackIdle());
  }

  get currentState(): VoiceState {
    return this.state;
  }

  private setState(state: VoiceState): void {
    if (this.state === state) return;
    this.state = state;
    this.options.onStateChange?.(state);
  }

  /** เริ่มฟัง */
  async start(): Promise<void> {
    this.recognitionFatal = false;
    // ต้องกันตั้งแต่ยังไม่เสร็จ ไม่ใช่กันตอน running กลายเป็น true
    // เพราะช่วงรอสิทธิ์ไมโครโฟนกินเวลาได้หลายวินาที (ครั้งแรก หรือหูฟังบลูทูธ)
    // ถ้าผู้ใช้กดปุ่มซ้ำในช่วงนั้น จะได้ไมโครโฟนสองตัวที่ปิดไม่ได้อีกเลย
    if (this.running || this.starting) return;
    this.starting = true;

    try {
      if (this.options.sendAudioForSpeakerId !== false) {
        this.recorder = new PcmRecorder();
        await this.recorder.start();
      }

      const useBrowser =
        (this.options.useBrowserRecognition ?? true) && browserRecognitionAvailable();
      if (useBrowser) this.startRecognition();

      // ตั้ง running เป็น true หลังทุกอย่างสำเร็จเท่านั้น ของเดิมตั้งไว้ก่อน
      // ทำให้เมื่อผู้ใช้ปฏิเสธสิทธิ์ไมโครโฟนแล้วกดใหม่ ระบบจะเงียบไปเลย
      // เพราะติดเงื่อนไข "กำลังทำงานอยู่แล้ว"
      this.running = true;
      this.setState("listening");
    } catch (error) {
      await this.cleanup();
      throw error;
    } finally {
      this.starting = false;
    }
  }

  private startRecognition(): void {
    const Ctor = getRecognitionCtor();
    if (!Ctor) {
      this.options.onError?.(new Error("เบราว์เซอร์นี้ถอดเสียงเองไม่ได้"));
      return;
    }
    const recognition = new Ctor();
    recognition.lang = "th-TH";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onspeechstart = () => {
      if (this.options.bargeIn) this.interrupt();
    };

    recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const alternative = result?.[0];
        if (!result || !alternative) continue;
        if (result.isFinal) {
          this.options.onTranscript?.(alternative.transcript, true);
          void this.submit(alternative.transcript);
        } else {
          interim += alternative.transcript;
        }
      }
      if (interim) this.options.onTranscript?.(interim, false);
    };

    recognition.onerror = (event) => {
      const code = (event as Event & { error?: string }).error;
      // "no-speech" เกิดตลอดเวลาเมื่อผู้ใช้เงียบ ไม่ใช่ข้อผิดพลาดจริง
      if (!code || code === "no-speech" || code === "aborted") return;
      if (FATAL_RECOGNITION_ERRORS.has(code)) {
        // ยิง error แล้ว end ตามมา ซึ่ง onend จะสั่ง start ใหม่ แล้วก็ล้มแบบเดิม
        // ทันที วนประมาณเก้าร้อยรอบต่อวินาทีตลอดไป หน้าเว็บที่ต่อ onError
        // เข้ากับการเพิ่ม DOM node จะบวมจนค้าง
        this.recognitionFatal = true;
        this.options.onError?.(new Error(FATAL_RECOGNITION_MESSAGES[code] ?? code));
        return;
      }
      this.options.onError?.(new Error(`ถอดเสียงผิดพลาด: ${code}`));
    };

    recognition.onend = () => {
      // ต้องเช็คว่าตัวเองยังเป็นตัวที่ใช้อยู่จริงไหม
      //
      // ตามสเปก abort() ยิง end แบบ asynchronous ถ้าผู้ใช้กดหยุดแล้วกดเริ่มใหม่
      // ก่อนเหตุการณ์นั้นจะมาถึง ตัวเก่าจะปลุกตัวเองกลับมาทำงาน กลายเป็นสอง
      // recognizer พร้อมกัน แล้ว pauseRecognition หยุดได้แค่ตัวใหม่ ตัวเก่าจึง
      // ฟังต่อระหว่างบอทพูด ถอดเสียงบอทเอง แล้วส่งกลับไปเป็นเทิร์นใหม่ —
      // บทสนทนาวนไม่จบที่แก้ไปแล้วรอบก่อน
      if (this.recognition !== recognition) return;
      if (this.recognitionFatal) return; // เริ่มใหม่ไปก็ล้มแบบเดิม
      if (this.running && !this.recognitionPaused) {
        try {
          recognition.start();
        } catch {
          // เบราว์เซอร์บางตัวโยน error ถ้าเรียก start เร็วเกินไป ปล่อยผ่านได้
        }
      }
    };

    this.recognition = recognition;
    recognition.start();
  }

  private pauseRecognition(): void {
    if (!this.recognition || this.recognitionPaused) return;
    this.recognitionPaused = true;
    try {
      this.recognition.stop();
    } catch {
      // ไม่ได้ทำงานอยู่
    }
  }

  private resumeRecognition(): void {
    if (!this.recognition || !this.recognitionPaused) return;
    this.recognitionPaused = false;
    try {
      this.recognition.start();
    } catch {
      // onend จะเรียก start ให้เองอีกที
    }
  }

  /** หยุดเสียงที่บอทกำลังพูดทันที (ใช้ตอนผู้ใช้พูดแทรก) */
  interrupt(): void {
    if (!this.audio.busy && !this.synth.busy && this.pending.length === 0) return;
    this.audio.stop();
    this.synth.stop();
    // ของเดิมยกเลิก pending[0] เสมอ ซึ่งอาจถูกยกเลิกไปแล้ว ส่วนเทิร์นที่กำลัง
    // ให้ผลอยู่จริงคือตัวถัดไป จึงรอดไปโดยไม่ถูกขัด
    const interrupted = this.pending.find((turn) => !turn.abandoned);
    if (interrupted) {
      interrupted.abandoned = true;
      if (interrupted.reply) this.options.onReply?.(interrupted.reply);
    }
    // stop() ล้างเทิร์นที่รับได้ทั้งหมด ต้องรับเทิร์นที่ยังไม่ถูกยกเลิกกลับเข้ามา
    // ไม่งั้นประโยคที่ผู้ใช้พูดต่อจากการพูดแทรกจะเงียบสนิทตลอดไป
    for (const turn of this.pending) {
      if (turn.abandoned) continue;
      this.audio.accept(turn.id);
      this.synth.accept(turn.id);
    }
    this.setState("listening");
  }

  /** หยุดบทสนทนาทั้งหมด */
  stop(): void {
    void this.cleanup();
  }

  private async cleanup(): Promise<void> {
    this.running = false;
    this.recognitionPaused = false;
    this.pending = [];
    const recognition = this.recognition;
    this.recognition = null;
    if (recognition) {
      // ปลดตัวจัดการก่อน abort() ไม่งั้น end ที่ยิงตามมาทีหลังยังวิ่งเข้าโค้ดเดิม
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      recognition.abort();
    }
    const recorder = this.recorder;
    this.recorder = null;
    await recorder?.stop();
    this.audio.stop();
    this.synth.stop();
    this.stream?.close();
    this.stream = null;
    this.streamReady = null;
    this.setState("idle");
  }

  /**
   * โหมดกดพูด — เรียกเมื่อผู้ใช้ปล่อยปุ่ม
   *
   * ใช้เมื่อเบราว์เซอร์ถอดเสียงเองไม่ได้ (Firefox / Safari) หรือเมื่อไม่อยากให้
   * เสียงออกไปที่บริการถอดเสียงของเบราว์เซอร์
   */
  pushToTalkStart(): void {
    // ทิ้งเสียงที่สะสมไว้ก่อนหน้า ไม่งั้นจะอัปโหลดทั้งหน้าต่างที่ค้างอยู่ (ถึง 30
    // วินาที) แทนที่จะเป็นแค่ช่วงที่ผู้ใช้กดปุ่มค้างไว้ ซึ่งทำให้ทั้งการถอดเสียง
    // และลายเสียงคำนวณจากเสียงคนอื่นในห้องไปด้วย
    this.recorder?.reset();
    this.setState("listening");
  }

  async pushToTalkStop(): Promise<void> {
    const wav = this.recorder?.take() ?? null;
    if (!wav) return;

    const turn = ++this.turnId;
    this.audio.accept(turn);
    this.synth.accept(turn);
    this.awaitingDirect += 1;
    this.setState("thinking");
    try {
      const result = await this.client.voice(wav, {
        speak: this.options.serverTts ?? true,
      });
      this.options.onTranscript?.(result.transcript, true);
      this.options.onSpeaker?.(result.speaker, result.identified_by ?? undefined);
      this.options.onReply?.(result.reply);
      if (result.audio) {
        this.setState("speaking");
        this.audio.push(turn, result.audio);
      } else if (result.reply) {
        this.setState("speaking");
        this.synth.speak(turn, result.reply);
      } else {
        this.setState("listening");
      }
    } catch (error) {
      this.options.onError?.(error as Error);
      this.audio.discard(turn);
      this.synth.discard(turn);
      this.setState("listening");
    } finally {
      this.awaitingDirect = Math.max(0, this.awaitingDirect - 1);
    }
  }

  /** ส่งข้อความที่ถอดเสียงได้ พร้อมเสียงดิบสำหรับจดจำผู้พูด */
  private async submit(text: string): Promise<void> {
    if (!text.trim()) return;

    const turn = ++this.turnId;
    this.pending.push({ id: turn, reply: "", abandoned: false });
    this.audio.accept(turn);
    this.synth.accept(turn);

    // จองคิวการส่งตรงนี้ ก่อนจะมี await ตัวไหนทั้งสิ้น
    //
    // submit ถูกเรียกแบบซิงโครนัสทีละประโยคจาก onresult ส่วนหัวที่ไม่มี await
    // จึงรันตามลำดับที่พูดแน่นอน ถ้าไปจองหลังจากแปลงเสียงเป็น base64 เสร็จ
    // ประโยคที่สอง (ซึ่งไม่มีเสียงให้แปลง เพราะประโยคแรกดูดบัฟเฟอร์ไปหมดแล้ว)
    // จะจองได้ก่อน แล้วแซงไปถึงเซิร์ฟเวอร์ก่อน
    const previous = this.sendChain;
    let release: () => void = () => {};
    this.sendChain = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.setState("thinking");

    // ปิดการฟังระหว่างบอทพูด ไม่งั้นไมโครโฟนจะได้ยินเสียงบอทเอง แล้วระบบจะ
    // ถอดเสียงตัวเองเป็นข้อความใหม่ ตอบตัวเองวนไม่จบ และที่แย่กว่านั้นคือเสียง
    // ของบอทจะถูกส่งไปสะสมเป็นลายเสียงของผู้ใช้ ทำให้จำเสียงเพี้ยนถาวร
    if (!this.options.bargeIn) this.pauseRecognition();

    // ตัดไฟล์เสียงของประโยคนี้ออกมาเป็นตัวแปรเฉพาะที่ ไม่เก็บไว้บนอ็อบเจ็กต์
    // เพราะการเรียกซ้อนกันสองครั้งจะทับกันเอง
    let audioBase64: string | undefined;
    const wav = this.recorder?.take() ?? null;
    if (wav) {
      try {
        audioBase64 = await blobToBase64(wav);
      } catch {
        audioBase64 = undefined;
      }
    }

    try {
      try {
        await previous; // ประโยคก่อนหน้าต้องถูกส่งออกไปก่อน
        const stream = await this.ensureStream();
        stream.send(text, { audio: audioBase64, speak: this.options.serverTts ?? true });
        const entry = this.pending.find((item) => item.id === turn);
        if (entry) entry.stream = stream;
      } finally {
        release();
      }
    } catch (error) {
      this.options.onError?.(error as Error);
      // ทิ้งเฉพาะเทิร์นนี้ ของเดิมล้าง pending ทั้งกอง เทิร์นก่อนหน้าที่กำลัง
      // คุยอยู่ดี ๆ จึงถูกลบไปด้วยเพราะประโยคใหม่ส่งไม่สำเร็จ
      this.dropTurn(turn);
    }
  }

  /** ทิ้งเทิร์นเดียวที่ส่งไม่สำเร็จ */
  private dropTurn(turn: number): void {
    const index = this.pending.findIndex((item) => item.id === turn);
    if (index >= 0) this.pending.splice(index, 1);
    this.audio.discard(turn);
    this.synth.discard(turn);
    this.maybeFinish();
  }

  /**
   * เปิดการเชื่อมต่อสตรีม (ใช้ซ้ำได้) — คืน promise เดิมถ้ากำลังเชื่อมต่ออยู่
   *
   * ของเดิมเก็บ socket ไว้ทันทีที่สร้าง แล้วผู้เรียกคนที่สองเห็นว่ามีแล้วจึงส่ง
   * ข้อความทันทีทั้งที่ socket ยังอยู่สถานะ CONNECTING ทำให้เกิด InvalidStateError
   * และประโยคที่สองหายไปเงียบ ๆ (เกิดง่ายมากเวลาพูดสองประโยคติดกันตอนเริ่มคุย)
   */
  private ensureStream(): Promise<StreamConnection> {
    if (this.stream?.open) return Promise.resolve(this.stream);
    // socket ที่กำลังปิดอยู่ (open เป็น false แต่เหตุการณ์ close ยังไม่มาถึง)
    // ต้องไม่ถูกแจกต่อ ของเดิมไม่เคยล้าง streamReady หลังต่อสำเร็จ จึงคืน
    // promise เดิมที่ resolve แล้ว ผู้เรียกส่งข้อความไม่ได้ ได้ error กำกวม
    // แล้ว dropPending ก็ล้างเทิร์นที่ยังคุยอยู่ทิ้งไปด้วย
    if (this.stream && !this.stream.open) {
      const dead = this.stream;
      this.stream = null;
      this.streamReady = null;
      try {
        dead.close();
      } catch {
        // ปิดไปแล้ว
      }
      // เทิร์นที่ส่งไปกับ socket ตัวนี้จะไม่มีคำตอบแล้ว ต้องเก็บกวาดตรงนี้
      // เพราะ onClose ของมันจะเห็นว่า this.stream ไม่ใช่ตัวเองแล้วจึงไม่ทำอะไร
      this.dropPending("การเชื่อมต่อหลุด ลองพูดอีกครั้ง", (turn) => turn.stream === dead);
    }
    if (this.streamReady) return this.streamReady;

    const ready = (async () => {
      let connection: StreamConnection;
      connection = this.client.stream({
        onEvent: (event: StreamEvent) => this.handleEvent(event),
        onClose: () => {
          // ต้องเช็คว่าเป็นตัวที่ใช้อยู่จริงไหม การเชื่อมต่อที่ล้มจะยิง error
          // ก่อนแล้วค่อยยิง close ถ้าล้างสถานะตอน close โดยไม่เช็ค มันจะไปล้าง
          // การเชื่อมต่อตัวใหม่ที่เพิ่งเปิดแทน กลายเป็นสอง socket พร้อมกัน
          if (this.stream !== connection) {
            // socket เก่าที่ปิดตามมาทีหลัง — ห้ามแตะสถานะของตัวที่ใช้อยู่
            // ไม่งั้นเทิร์นที่กำลังคุยอยู่จะถูกล้างทิ้งพร้อมข้อความ "การเชื่อมต่อหลุด"
            // ปลอม ๆ แล้วส่วนท้ายของคำตอบนั้นจะไปโผล่เป็นคำตอบของประโยคถัดไป
            return;
          }
          this.stream = null;
          this.streamReady = null;
          // การเชื่อมต่อหลุดกลางเทิร์นต้องบอกผู้ใช้และปลดสถานะ ไม่งั้น UI จะค้าง
          // ที่ "กำลังคิด" ตลอดกาล และไมโครโฟนที่ถูกพักไว้จะไม่ถูกเปิดกลับมาเลย
          // ทิ้งเฉพาะเทิร์นที่ส่งไปกับ socket ตัวนี้
          //
          // เทิร์นที่ยังไม่ได้ส่งต้องรอด — ผู้ใช้พูดประโยคใหม่ในจังหวะเดียวกับที่
          // socket เก่าปิดเป็นเรื่องปกติมาก ประโยคนั้นจะถูกส่งไปกับ socket ตัวใหม่
          // ถ้าล้างทิ้งตรงนี้ด้วย ผู้ใช้จะไม่ได้คำตอบและเห็นแต่ "การเชื่อมต่อหลุด"
          this.dropPending(
            "การเชื่อมต่อหลุด ลองพูดอีกครั้ง",
            (turn) => turn.stream === connection,
          );
        },
      });
      try {
        await connection.ready;
      } catch (error) {
        if (this.stream === connection) this.stream = null;
        this.streamReady = null;
        throw error;
      }
      this.stream = connection;
      return connection;
    })();
    // ต้องล้างทิ้งเมื่อล้มเหลว ไม่งั้น promise ที่ reject แล้วจะถูกเก็บไว้ตลอด
    // ทางออกที่ ensureStream ใช้ล้าง (this.stream) เป็น null อยู่แล้วในกรณีที่
    // client.stream() โยน error ตั้งแต่ก่อนสร้าง socket ทุกประโยคหลังจากนั้น
    // จะล้มด้วย error เดิมค้างไปทั้งเซสชัน โดยไม่พยายามต่อใหม่เลยสักครั้ง
    ready.catch(() => {
      if (this.streamReady === ready) this.streamReady = null;
    });
    this.streamReady = ready;
    return ready;
  }

  private handleEvent(event: StreamEvent): void {
    // เหตุการณ์เป็นของเทิร์นที่เก่าที่สุดที่ยังไม่จบ ไม่ใช่เทิร์นล่าสุดที่ส่งไป
    const turn = this.pending[0];
    if (!turn) {
      // เซิร์ฟเวอร์ส่ง error ที่ไม่ผูกกับเทิร์นไหนได้ (เฟรมพัง เฟรมไบนารี)
      // ของเดิมกลืนทิ้งหมด ผู้ใช้จึงไม่รู้เลยว่าเกิดอะไรขึ้น
      if (event.type === "error") {
        const message = typeof event.text === "string" ? event.text : "";
        this.options.onError?.(new Error(message || "เกิดข้อผิดพลาด"));
      }
      return;
    }

    const text = typeof event.text === "string" ? event.text : "";

    if (turn.abandoned) {
      // เทิร์นนี้ถูกพูดแทรกตัดไปแล้ว กลืนเหตุการณ์ที่เหลือทิ้งจนกว่าจะถึง done
      if (event.type === "done" || event.type === "error") {
        this.pending.shift();
        this.maybeFinish();
      }
      return;
    }

    switch (event.type) {
      case "speaker":
        this.options.onSpeaker?.(event.speaker ?? null, event.identified_by, event.score);
        break;
      case "delta":
        turn.reply += text;
        this.options.onReplyDelta?.(text);
        break;
      case "chunk":
        this.setState("speaking");
        if (event.audio) this.audio.push(turn.id, event.audio, event.mime ?? "audio/mpeg");
        else this.synth.speak(turn.id, text);
        break;
      case "done": {
        this.pending.shift();
        this.options.onReply?.(text || turn.reply);
        this.maybeFinish();
        break;
      }
      case "error":
        this.pending.shift();
        this.options.onError?.(new Error(text || "เกิดข้อผิดพลาด"));
        this.maybeFinish();
        break;
    }
  }

  private onPlaybackIdle(): void {
    this.maybeFinish();
  }

  /**
   * จบเทิร์นเมื่อ *ทั้ง* ตอบครบแล้ว *และ* พูดจบแล้วเท่านั้น
   *
   * คิวเสียงว่างชั่วคราวกลางเทิร์นเป็นเรื่องปกติ เพราะเซิร์ฟเวอร์สังเคราะห์เสียง
   * ประโยคถัดไปหลังส่งประโยคก่อนหน้าไปแล้ว ถ้าถือว่าคิวว่างเท่ากับจบเทิร์น
   * ระบบจะเปิดไมโครโฟนกลับมาทั้งที่บอทยังพูดอยู่ แล้วก็จะได้ยินเสียงตัวเอง
   * ตอบตัวเองวนไม่จบ ซึ่งเป็นบัคเดิมที่กลับมาทางอ้อม
   */
  private maybeFinish(): void {
    if (this.pending.length > 0) return;
    if (this.audio.busy || this.synth.busy) return;
    // โหมดกดพูดไม่ผ่านคิว pending (มันรอคำตอบทาง HTTP ไม่ใช่ WebSocket)
    // ถ้าลืมหมายเลขเทิร์นตอนนี้ คำตอบที่กำลังเดินทางกลับมาจะถูกทิ้งทั้งเสียง
    // แล้วสถานะจะค้างที่ "กำลังพูด" ตลอดกาล เพราะไม่มีอะไรเข้าคิวให้ onIdle ยิง
    if (this.awaitingDirect > 0) return;
    // ทุกอย่างว่างแล้ว ลืมหมายเลขเทิร์นเก่าได้ กันเซ็ตโตไม่หยุดในบทสนทนายาว ๆ
    this.audio.reset();
    this.synth.reset();
    this.finishTurn();
  }

  /**
   * ยกเลิกเทิร์นที่ค้างอยู่ (ใช้เมื่อการเชื่อมต่อหลุดหรือส่งไม่สำเร็จ)
   *
   * ``match`` เลือกว่าเทิร์นไหนต้องทิ้ง — ค่าเริ่มต้นคือทิ้งทั้งหมด
   *
   * ต้องหยุดเสียงของเทิร์นที่ทิ้งด้วย และต้องผ่าน ``maybeFinish`` ไม่ใช่
   * ``finishTurn`` ตรง ๆ ของเดิมข้ามการเช็คว่าบอทยังพูดอยู่ไหม จึงเปิด
   * ไมโครโฟนกลับมาทั้งที่เสียงที่เข้าคิวไว้ยังเล่นอยู่ ระบบก็ได้ยินเสียงตัวเอง
   * ถอดเป็นข้อความ ตอบตัวเอง แล้วเอาเสียงบอทไปสะสมเป็นลายเสียงของผู้ใช้
   */
  private dropPending(
    message?: string,
    match: (turn: PendingTurn) => boolean = () => true,
  ): void {
    const dropped = this.pending.filter(match);
    if (dropped.length === 0) return;
    this.pending = this.pending.filter((turn) => !match(turn));
    for (const turn of dropped) {
      this.audio.discard(turn.id);
      this.synth.discard(turn.id);
    }
    if (message) this.options.onError?.(new Error(message));
    this.maybeFinish();
  }

  private finishTurn(): void {
    if (!this.running) return;
    this.setState("listening");
    this.resumeRecognition();
  }
}

/** แปลง Blob เป็น base64 (ไม่รวมส่วนหัว data URL) */
export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = String(reader.result);
      resolve(result.slice(result.indexOf(",") + 1));
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}
