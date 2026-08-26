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

  /** ขอสิทธิ์ไมโครโฟนและเริ่มอัด — เรียกซ้ำได้ */
  async start(): Promise<void> {
    if (this.disposed) throw new Error("ตัวอัดเสียงถูกปิดไปแล้ว");
    if (this.recording) return;
    if (this.starting) return this.starting;

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
  private playing = false;
  private acceptingTurn = -1;
  /** เพิ่มขึ้นทุกครั้งที่ถูกสั่งหยุด ใช้ให้ลูปเดิมที่ค้างอยู่รู้ว่าต้องเลิก */
  private generation = 0;

  constructor(private readonly onIdle?: () => void) {}

  /** เริ่มเทิร์นใหม่ — เสียงของเทิร์นก่อนหน้าที่ยังไหลมาจะถูกทิ้ง */
  accept(turn: number): void {
    this.acceptingTurn = turn;
  }

  /** เพิ่มเสียง base64 เข้าคิวของเทิร์นนั้น */
  push(turn: number, base64: string, mime = "audio/mpeg"): void {
    if (turn !== this.acceptingTurn) return;
    this.queue.push({ url: `data:${mime};base64,${base64}`, turn });
    void this.drain();
  }

  /** หยุดทันที ล้างคิว และไม่รับเสียงของเทิร์นปัจจุบันอีก */
  stop(): void {
    this.generation += 1;
    this.acceptingTurn = -1;
    this.queue = [];
    const audio = this.current;
    this.current = null;
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
        if (!item || item.turn !== this.acceptingTurn) continue;
        const generation = this.generation;
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
  private acceptingTurn = -1;

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
    this.acceptingTurn = turn;
  }

  private pickVoice(): void {
    const voices = globalThis.speechSynthesis?.getVoices?.() ?? [];
    this.voice = voices.find((v) => v.lang?.toLowerCase().startsWith("th")) ?? null;
  }

  speak(turn: number, text: string): void {
    if (!this.available || !text.trim() || turn !== this.acceptingTurn) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "th-TH";
    utterance.rate = this.rate;
    utterance.pitch = this.pitch;
    if (this.voice) utterance.voice = this.voice;
    this.pending += 1;
    const finish = () => {
      this.pending = Math.max(0, this.pending - 1);
      if (this.pending === 0) this.onIdle?.();
    };
    utterance.onend = finish;
    utterance.onerror = finish;
    globalThis.speechSynthesis.speak(utterance);
  }

  stop(): void {
    this.acceptingTurn = -1;
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
export class VoiceConversation {
  private recognition: SpeechRecognitionLike | null = null;
  private recorder: PcmRecorder | null = null;
  private readonly audio: AudioQueue;
  private readonly synth: ThaiSpeechSynthesis;
  private state: VoiceState = "idle";
  private running = false;
  private recognitionPaused = false;
  private stream: StreamConnection | null = null;
  private streamReady: Promise<StreamConnection> | null = null;
  private turnId = 0;
  private activeTurn = 0;
  private replyBuffer = "";
  private awaitingAudio = false;

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
    if (this.running) return;

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
      if (code && code !== "no-speech" && code !== "aborted") {
        this.options.onError?.(new Error(`ถอดเสียงผิดพลาด: ${code}`));
      }
    };

    recognition.onend = () => {
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
    if (!this.audio.busy && !this.synth.busy) return;
    this.audio.stop();
    this.synth.stop();
    this.awaitingAudio = false;
    this.setState("listening");
  }

  /** หยุดบทสนทนาทั้งหมด */
  stop(): void {
    void this.cleanup();
  }

  private async cleanup(): Promise<void> {
    this.running = false;
    this.recognitionPaused = false;
    this.recognition?.abort();
    this.recognition = null;
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
  async pushToTalkStop(): Promise<void> {
    const wav = this.recorder?.take() ?? null;
    if (!wav) return;

    const turn = ++this.turnId;
    this.activeTurn = turn;
    this.audio.accept(turn);
    this.synth.accept(turn);
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
        this.awaitingAudio = true;
        this.audio.push(turn, result.audio);
      } else if (result.reply) {
        this.setState("speaking");
        this.awaitingAudio = true;
        this.synth.speak(turn, result.reply);
      } else {
        this.setState("listening");
      }
    } catch (error) {
      this.options.onError?.(error as Error);
      this.setState("listening");
    }
  }

  /** ส่งข้อความที่ถอดเสียงได้ พร้อมเสียงดิบสำหรับจดจำผู้พูด */
  private async submit(text: string): Promise<void> {
    if (!text.trim()) return;

    const turn = ++this.turnId;
    this.activeTurn = turn;
    this.replyBuffer = "";
    this.audio.accept(turn);
    this.synth.accept(turn);
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
      const stream = await this.ensureStream();
      stream.send(text, { audio: audioBase64, speak: this.options.serverTts ?? true });
    } catch (error) {
      this.options.onError?.(error as Error);
      this.finishTurn();
    }
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
    if (this.streamReady) return this.streamReady;

    this.streamReady = (async () => {
      const connection = this.client.stream({
        onEvent: (event: StreamEvent) => this.handleEvent(event),
        onClose: () => {
          this.stream = null;
          this.streamReady = null;
        },
      });
      try {
        await connection.ready;
      } catch (error) {
        this.stream = null;
        this.streamReady = null;
        throw error;
      }
      this.stream = connection;
      return connection;
    })();

    return this.streamReady;
  }

  private handleEvent(event: StreamEvent): void {
    const turn = this.activeTurn;
    switch (event.type) {
      case "speaker":
        this.options.onSpeaker?.(event.speaker ?? null, event.identified_by, event.score);
        break;
      case "delta":
        this.replyBuffer += event.text;
        this.options.onReplyDelta?.(event.text);
        break;
      case "chunk":
        this.setState("speaking");
        this.awaitingAudio = true;
        if (event.audio) this.audio.push(turn, event.audio, event.mime ?? "audio/mpeg");
        else this.synth.speak(turn, event.text);
        break;
      case "done": {
        const reply = event.text || this.replyBuffer;
        this.replyBuffer = "";
        this.options.onReply?.(reply);
        // ยังพูดไม่จบก็ยังไม่ใช่สถานะ "กำลังฟัง" — รอให้คิวเสียงว่างก่อน
        if (!this.audio.busy && !this.synth.busy) this.finishTurn();
        break;
      }
      case "error":
        this.options.onError?.(new Error(event.text));
        this.replyBuffer = "";
        this.finishTurn();
        break;
    }
  }

  private onPlaybackIdle(): void {
    if (!this.awaitingAudio) return;
    this.awaitingAudio = false;
    this.finishTurn();
  }

  private finishTurn(): void {
    this.awaitingAudio = false;
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
