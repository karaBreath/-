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
 *   เซิร์ฟเวอร์ถอดเสียงด้วย Whisper แทน
 * - Safari และ Firefox รองรับไม่ครบ ระบบจะถอยไปใช้การถอดเสียงฝั่งเซิร์ฟเวอร์เอง
 */

import type { ThaiVoiceClient } from "./client.js";
import type { Speaker, StreamEvent } from "./types.js";

// ── ประกาศชนิดของ Web Speech API (ยังไม่อยู่ใน lib.dom มาตรฐาน) ──────────────
interface SpeechRecognitionAlternativeLike {
  transcript: string;
  confidence: number;
}
interface SpeechRecognitionResultLike {
  isFinal: boolean;
  0: SpeechRecognitionAlternativeLike;
  length: number;
}
interface SpeechRecognitionEventLike extends Event {
  resultIndex: number;
  results: { length: number; [index: number]: SpeechRecognitionResultLike };
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
 * ใช้ `ScriptProcessorNode` เพราะรองรับทุกเบราว์เซอร์ที่ใช้งานจริง (แม้จะถูก
 * ประกาศเลิกใช้แล้ว) และงานนี้ไม่ต้องการความแม่นยำระดับ AudioWorklet
 */
export class PcmRecorder {
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private stream: MediaStream | null = null;
  private buffers: Float32Array[] = [];

  constructor(readonly targetSampleRate = 16000) {}

  get recording(): boolean {
    return this.processor !== null;
  }

  /** ขอสิทธิ์ไมโครโฟนและเริ่มอัด */
  async start(): Promise<void> {
    if (this.recording) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this.context = new AudioContext();
    this.source = this.context.createMediaStreamSource(this.stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.buffers = [];
    this.processor.onaudioprocess = (event) => {
      this.buffers.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    this.source.connect(this.processor);
    // ต่อเข้า destination ด้วย ไม่งั้นบางเบราว์เซอร์จะไม่เรียก onaudioprocess
    this.processor.connect(this.context.destination);
  }

  /** ล้างเสียงที่อัดไว้ แต่ยังอัดต่อ (ใช้ตัดช่วงเงียบก่อนผู้ใช้เริ่มพูด) */
  reset(): void {
    this.buffers = [];
  }

  /** หยุดอัดแล้วคืนไฟล์ WAV — คืน `null` ถ้าไม่มีเสียง */
  stop(): Blob | null {
    if (!this.context) return null;
    const sourceRate = this.context.sampleRate;
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    void this.context.close();
    this.processor = null;
    this.source = null;
    this.stream = null;
    this.context = null;

    const total = this.buffers.reduce((sum, chunk) => sum + chunk.length, 0);
    if (total === 0) return null;
    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of this.buffers) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    this.buffers = [];
    return encodeWav(resample(merged, sourceRate, this.targetSampleRate), this.targetSampleRate);
  }
}

/** เล่นเสียงตอบทีละประโยคตามลำดับ และหยุดได้ทันทีเมื่อผู้ใช้พูดแทรก */
export class AudioQueue {
  private queue: string[] = [];
  private current: HTMLAudioElement | null = null;
  private playing = false;

  /** เพิ่มเสียง base64 เข้าคิว */
  push(base64: string, mime = "audio/mpeg"): void {
    this.queue.push(`data:${mime};base64,${base64}`);
    void this.drain();
  }

  /** หยุดทันทีและล้างคิวที่เหลือ */
  stop(): void {
    this.queue = [];
    if (this.current) {
      this.current.pause();
      this.current = null;
    }
    this.playing = false;
  }

  get busy(): boolean {
    return this.playing || this.queue.length > 0;
  }

  private async drain(): Promise<void> {
    if (this.playing) return;
    this.playing = true;
    while (this.queue.length > 0) {
      const url = this.queue.shift();
      if (!url) break;
      await new Promise<void>((resolve) => {
        const audio = new Audio(url);
        this.current = audio;
        audio.onended = () => resolve();
        audio.onerror = () => resolve();
        void audio.play().catch(() => resolve());
      });
      this.current = null;
    }
    this.playing = false;
  }
}

/** พูดภาษาไทยด้วยเสียงของเบราว์เซอร์เอง (ใช้เมื่อเซิร์ฟเวอร์ไม่ได้สังเคราะห์เสียงให้) */
export class ThaiSpeechSynthesis {
  private voice: SpeechSynthesisVoice | null = null;

  constructor(readonly rate = 1.05, readonly pitch = 1.0) {
    this.pickVoice();
    // รายชื่อเสียงในบางเบราว์เซอร์โหลดแบบ asynchronous
    globalThis.speechSynthesis?.addEventListener?.("voiceschanged", () => this.pickVoice());
  }

  get available(): boolean {
    return typeof globalThis.speechSynthesis !== "undefined";
  }

  private pickVoice(): void {
    const voices = globalThis.speechSynthesis?.getVoices?.() ?? [];
    this.voice = voices.find((v) => v.lang?.toLowerCase().startsWith("th")) ?? null;
  }

  speak(text: string): void {
    if (!this.available || !text.trim()) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "th-TH";
    utterance.rate = this.rate;
    utterance.pitch = this.pitch;
    if (this.voice) utterance.voice = this.voice;
    globalThis.speechSynthesis.speak(utterance);
  }

  stop(): void {
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
  /** ใช้ Web Speech API ของเบราว์เซอร์ถอดเสียง (เร็วกว่า) หรือส่งเสียงให้เซิร์ฟเวอร์ถอด */
  useBrowserRecognition?: boolean;
  /** ส่งเสียงดิบไปด้วยเพื่อให้ระบบจำได้ว่าใครพูด */
  sendAudioForSpeakerId?: boolean;
  /** ให้เซิร์ฟเวอร์สังเคราะห์เสียงตอบ (คุณภาพดีกว่าเสียงในเบราว์เซอร์) */
  serverTts?: boolean;
  /** ฟังต่อระหว่างบอทกำลังพูด เพื่อให้พูดแทรกได้ (ควรใส่หูฟัง ไม่งั้นจะได้ยินเสียงตัวเอง) */
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
  private readonly audio = new AudioQueue();
  private readonly synth = new ThaiSpeechSynthesis();
  private state: VoiceState = "idle";
  private running = false;
  private stream: ReturnType<ThaiVoiceClient["stream"]> | null = null;
  private pendingWav: Blob | null = null;

  constructor(
    private readonly client: ThaiVoiceClient,
    private readonly options: VoiceConversationOptions = {},
  ) {}

  get currentState(): VoiceState {
    return this.state;
  }

  private setState(state: VoiceState): void {
    this.state = state;
    this.options.onStateChange?.(state);
  }

  /** เริ่มฟัง */
  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;

    if (this.options.sendAudioForSpeakerId !== false) {
      this.recorder = new PcmRecorder();
      await this.recorder.start();
    }

    const useBrowser =
      (this.options.useBrowserRecognition ?? true) && browserRecognitionAvailable();
    if (!useBrowser) {
      this.setState("listening");
      return; // โหมดกดพูด: ให้เรียก pushToTalkStop() เองเมื่อพูดจบ
    }

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
      // ผู้ใช้เริ่มพูดขณะบอทกำลังพูด -> หยุดเสียงบอททันที
      if (this.options.bargeIn && this.audio.busy) {
        this.audio.stop();
        this.synth.stop();
      }
    };

    recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (!result) continue;
        const text = result[0].transcript;
        if (result.isFinal) {
          this.options.onTranscript?.(text, true);
          void this.submit(text);
        } else {
          interim += text;
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
      if (this.running) {
        try {
          recognition.start();
        } catch {
          // เบราว์เซอร์บางตัวโยน error ถ้าเรียก start เร็วเกินไป ปล่อยผ่านได้
        }
      }
    };

    this.recognition = recognition;
    recognition.start();
    this.setState("listening");
  }

  /** หยุดบทสนทนาทั้งหมด */
  stop(): void {
    this.running = false;
    this.recognition?.abort();
    this.recognition = null;
    this.recorder?.stop();
    this.recorder = null;
    this.audio.stop();
    this.synth.stop();
    this.stream?.close();
    this.stream = null;
    this.setState("idle");
  }

  /** โหมดกดพูด: เรียกเมื่อผู้ใช้ปล่อยปุ่ม (ใช้เมื่อไม่ได้ใช้ Web Speech API) */
  async pushToTalkStop(): Promise<void> {
    const wav = this.recorder?.stop() ?? null;
    this.recorder = null;
    if (!wav) return;
    this.setState("thinking");
    try {
      const result = await this.client.voice(wav, { speak: this.options.serverTts ?? true });
      this.options.onTranscript?.(result.transcript, true);
      this.options.onSpeaker?.(result.speaker, result.identified_by ?? undefined);
      this.options.onReply?.(result.reply);
      if (result.audio) {
        this.setState("speaking");
        this.audio.push(result.audio);
      } else {
        this.synth.speak(result.reply);
      }
    } catch (error) {
      this.options.onError?.(error as Error);
    } finally {
      this.setState("listening");
      if (this.running && this.options.sendAudioForSpeakerId !== false) {
        this.recorder = new PcmRecorder();
        await this.recorder.start();
      }
    }
  }

  /** ส่งข้อความที่ถอดเสียงได้ พร้อมเสียงดิบสำหรับจดจำผู้พูด */
  private async submit(text: string): Promise<void> {
    if (!text.trim()) return;
    this.setState("thinking");

    // ตัดไฟล์เสียงของประโยคนี้ออกมา แล้วเริ่มอัดรอบใหม่ทันที
    let audioBase64: string | undefined;
    if (this.recorder) {
      this.pendingWav = this.recorder.stop();
      this.recorder = new PcmRecorder();
      void this.recorder.start();
      if (this.pendingWav) {
        audioBase64 = await blobToBase64(this.pendingWav);
        this.pendingWav = null;
      }
    }

    try {
      await this.ensureStream();
      let reply = "";
      this.stream?.send(text, { audio: audioBase64, speak: this.options.serverTts ?? true });
      // การตอบกลับมาทาง onEvent ที่ผูกไว้ใน ensureStream
      void reply;
    } catch (error) {
      this.options.onError?.(error as Error);
      this.setState("listening");
    }
  }

  private async ensureStream(): Promise<void> {
    if (this.stream) return;
    let reply = "";
    this.stream = this.client.stream({
      onEvent: (event: StreamEvent) => {
        switch (event.type) {
          case "speaker":
            this.options.onSpeaker?.(event.speaker ?? null, event.identified_by, event.score);
            break;
          case "delta":
            reply += event.text;
            this.options.onReplyDelta?.(event.text);
            break;
          case "chunk":
            this.setState("speaking");
            if (event.audio) this.audio.push(event.audio, event.mime ?? "audio/mpeg");
            else this.synth.speak(event.text);
            break;
          case "done":
            this.options.onReply?.(event.text || reply);
            reply = "";
            this.setState("listening");
            break;
          case "error":
            this.options.onError?.(new Error(event.text));
            this.setState("listening");
            break;
        }
      },
      onClose: () => {
        this.stream = null;
      },
    });
    await this.stream.ready;
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
