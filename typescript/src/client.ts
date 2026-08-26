/**
 * ไคลเอนต์เรียกเซิร์ฟเวอร์ thaivoice — ใช้ได้ทั้งใน Node และเบราว์เซอร์
 *
 * ใช้เฉพาะ fetch/WebSocket ที่เป็นมาตรฐาน จึงไม่ต้องพึ่ง dependency ใด ๆ
 * (Node 22 ขึ้นไปมี WebSocket มาให้ในตัว รุ่นเก่ากว่านั้นส่ง `webSocketImpl` เข้ามาได้)
 */

import type {
  ChatOptions,
  ChatResult,
  IdentifyMethod,
  Speaker,
  SpeakerDetail,
  StreamEvent,
  VoiceResult,
} from "./types.js";

/** ชนิดของตัวสร้าง WebSocket — ให้ส่ง implementation จากภายนอกได้บน Node รุ่นเก่า */
export type WebSocketCtor = new (url: string) => WebSocket;

export interface ClientOptions {
  /** ที่อยู่เซิร์ฟเวอร์ เช่น http://127.0.0.1:8080 */
  baseUrl?: string;
  /** session เริ่มต้นสำหรับทุกคำขอ */
  sessionId?: string;
  fetchImpl?: typeof fetch;
  /** ตัวสร้าง WebSocket — จำเป็นบน Node ที่ต่ำกว่า 22 ซึ่งยังไม่มี WebSocket ในตัว */
  webSocketImpl?: WebSocketCtor;
}

export class ThaiVoiceError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ThaiVoiceError";
  }
}

/**
 * แปลง body ของข้อผิดพลาดให้เป็นข้อความที่อ่านรู้เรื่อง
 *
 * FastAPI คืนข้อผิดพลาดตรวจสอบข้อมูลเป็น `{"detail": [{loc, msg, ...}]}` ถ้าเอา
 * มาต่อสตริงตรง ๆ ผู้ใช้จะเห็นแค่ "[object Object]"
 */
function describeError(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object" && "detail" in detail) {
    const inner = (detail as { detail: unknown }).detail;
    if (typeof inner === "string") return inner;
    if (Array.isArray(inner)) {
      const parts = inner.map((item) => {
        if (item && typeof item === "object") {
          const record = item as { loc?: unknown[]; msg?: unknown };
          const where = Array.isArray(record.loc) ? record.loc.join(".") : "";
          const msg = typeof record.msg === "string" ? record.msg : JSON.stringify(item);
          return where ? `${where}: ${msg}` : msg;
        }
        return String(item);
      });
      if (parts.length) return parts.join("; ");
    }
  }
  return `คำขอล้มเหลว (${status})`;
}

export class ThaiVoiceClient {
  readonly baseUrl: string;
  readonly sessionId: string | undefined;
  private readonly fetchImpl: typeof fetch;
  private readonly webSocketImpl: WebSocketCtor | undefined;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:8080").replace(/\/+$/, "");
    this.sessionId = options.sessionId;
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch?.bind(globalThis);
    this.webSocketImpl =
      options.webSocketImpl ?? (globalThis as { WebSocket?: WebSocketCtor }).WebSocket;
  }

  /**
   * คุยด้วยข้อความล้วน (ไคลเอนต์ถอดเสียงมาเองแล้ว)
   *
   * ถ้าตั้ง `speak: true` เซิร์ฟเวอร์จะแนบเสียงคำตอบแบบ base64 มาใน `audio`
   */
  async chat(text: string, options: ChatOptions = {}): Promise<ChatResult> {
    return this.post<ChatResult>("/api/chat", {
      text,
      session_id: options.sessionId ?? this.sessionId ?? null,
      speaker_id: options.speakerId ?? null,
      speak: options.speak ?? false,
    });
  }

  /**
   * ส่งเสียงหนึ่งประโยคให้เซิร์ฟเวอร์ทั้งระบุตัวผู้พูด ถอดเสียง และตอบกลับ
   *
   * แนบ `transcript` มาด้วยได้ถ้าเบราว์เซอร์ถอดเสียงเองแล้ว เซิร์ฟเวอร์จะข้ามขั้น
   * ถอดเสียง แต่ยังใช้ไฟล์เสียงเพื่อจดจำว่าใครพูดอยู่ดี
   */
  async voice(
    wav: Blob,
    options: ChatOptions & { transcript?: string } = {},
  ): Promise<VoiceResult> {
    const form = new FormData();
    form.append("audio", wav, "utterance.wav");
    form.append("transcript", options.transcript ?? "");
    form.append("session_id", options.sessionId ?? this.sessionId ?? "");
    form.append("speak", String(options.speak ?? true));
    if (options.speakerId != null) form.append("speaker_id", String(options.speakerId));

    const response = await this.fetchImpl(`${this.baseUrl}/api/voice`, {
      method: "POST",
      body: form,
    });
    return this.unwrap<VoiceResult>(response);
  }

  /** ถอดเสียงอย่างเดียว */
  async transcribe(wav: Blob): Promise<{ text: string; confidence: number }> {
    const form = new FormData();
    form.append("audio", wav, "utterance.wav");
    const response = await this.fetchImpl(`${this.baseUrl}/api/stt`, {
      method: "POST",
      body: form,
    });
    return this.unwrap(response);
  }

  /** สังเคราะห์เสียงภาษาไทยจากข้อความ */
  async speak(text: string, voice?: string): Promise<Blob> {
    const response = await this.fetchImpl(`${this.baseUrl}/api/tts`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, voice: voice ?? null }),
    });
    if (!response.ok) {
      let detail: unknown;
      try {
        detail = await response.json();
      } catch {
        detail = await response.text().catch(() => "");
      }
      throw new ThaiVoiceError(describeError(detail, response.status), response.status, detail);
    }
    return response.blob();
  }

  /** ถามว่าเสียงนี้เป็นของใคร โดยไม่ต้องคุย */
  async identify(wav: Blob): Promise<{
    speaker: Speaker | null;
    score: number;
    is_new: boolean;
    method: IdentifyMethod;
  }> {
    const form = new FormData();
    form.append("audio", wav, "sample.wav");
    const response = await this.fetchImpl(`${this.baseUrl}/api/identify`, {
      method: "POST",
      body: form,
    });
    return this.unwrap(response);
  }

  /** รายชื่อคนที่ระบบรู้จัก */
  async speakers(): Promise<Speaker[]> {
    const data = await this.get<{ speakers: Speaker[] }>("/api/speakers");
    return data.speakers;
  }

  /** ความจำทั้งหมดของคนคนหนึ่ง */
  async speaker(id: number): Promise<SpeakerDetail> {
    return this.get<SpeakerDetail>(`/api/speakers/${id}`);
  }

  /** สร้างผู้สนทนาใหม่ (ถ้าชื่อซ้ำจะคืนคนเดิม) */
  async createSpeaker(
    name: string,
    gender?: "male" | "female",
  ): Promise<{ speaker: Speaker; created: boolean }> {
    return this.post("/api/speakers", { name, gender: gender ?? null });
  }

  /** สอนให้ระบบจำเสียงของคนคนนี้เพิ่ม */
  async enroll(id: number, wav: Blob): Promise<{ ok: boolean; speaker: Speaker }> {
    const form = new FormData();
    form.append("audio", wav, "enroll.wav");
    const response = await this.fetchImpl(`${this.baseUrl}/api/speakers/${id}/enroll`, {
      method: "POST",
      body: form,
    });
    return this.unwrap(response);
  }

  /** ลบผู้สนทนาและความจำทั้งหมด */
  async deleteSpeaker(id: number): Promise<boolean> {
    const data = await this.request<{ deleted: boolean }>("DELETE", `/api/speakers/${id}`);
    return data.deleted;
  }

  /** ลบเฉพาะข้อเท็จจริงที่จำไว้ (บทสนทนาและบทสรุปยังอยู่) */
  async forgetFacts(id: number): Promise<number> {
    const data = await this.request<{ removed: number }>(
      "DELETE",
      `/api/speakers/${id}/facts`,
    );
    return data.removed;
  }

  /**
   * ลบความจำทั้งหมด — ข้อเท็จจริง บทสรุป และบทสนทนา แต่ยังรู้จักตัวคนอยู่
   *
   * นี่คือสิ่งที่ผู้ใช้คาดหวังเมื่อกด "ลืมทุกอย่างเกี่ยวกับฉัน" การลบแค่ข้อเท็จจริง
   * ทำให้บทสรุปและบทสนทนาเก่ายังไหลกลับเข้า prompt ได้
   */
  async forgetMemory(
    id: number,
  ): Promise<{ facts: number; summaries: number; turns: number }> {
    const data = await this.request<{
      removed: { facts: number; summaries: number; turns: number };
    }>("DELETE", `/api/speakers/${id}/memory`);
    return data.removed;
  }

  /**
   * เปิดการเชื่อมต่อแบบสตรีม — ได้ยินคำตอบเร็วกว่าเพราะเสียงมาทีละประโยค
   *
   * ```ts
   * const stream = client.stream({
   *   onEvent: (e) => { if (e.type === "chunk") play(e.audio); },
   * });
   * await stream.ready;
   * stream.send("สวัสดีครับ");
   * ```
   */
  stream(
    handlers: {
      onEvent?: (event: StreamEvent) => void;
      onOpen?: () => void;
      onClose?: () => void;
      onError?: (error: Event) => void;
      sessionId?: string;
    } = {},
  ): StreamConnection {
    const Ctor = this.webSocketImpl;
    if (!Ctor) {
      throw new ThaiVoiceError(
        "สภาพแวดล้อมนี้ไม่มี WebSocket — ใช้ Node 22 ขึ้นไป หรือส่ง webSocketImpl เข้ามาตอนสร้าง client",
        0,
      );
    }

    const sessionId = handlers.sessionId ?? this.sessionId;
    const url = new URL(`${this.baseUrl.replace(/^http/, "ws")}/ws/chat`);
    if (sessionId) url.searchParams.set("session_id", sessionId);

    const socket = new Ctor(url.toString());
    let settled = false;
    const ready = new Promise<void>((resolve, reject) => {
      socket.addEventListener("open", () => {
        settled = true;
        handlers.onOpen?.();
        resolve();
      });
      socket.addEventListener("error", (event) => {
        handlers.onError?.(event);
        if (!settled) {
          settled = true;
          reject(new ThaiVoiceError("เชื่อมต่อ WebSocket ไม่สำเร็จ", 0));
        }
      });
      // ปิดก่อนเปิดสำเร็จก็ต้อง reject ไม่งั้นผู้เรียกจะค้างรอตลอดไป
      socket.addEventListener("close", () => {
        if (!settled) {
          settled = true;
          reject(new ThaiVoiceError("การเชื่อมต่อถูกปิดก่อนเริ่มใช้งาน", 0));
        }
      });
    });
    // กัน unhandled rejection เมื่อผู้เรียกไม่ได้ await ready
    ready.catch(() => {});

    socket.addEventListener("message", (event) => {
      try {
        handlers.onEvent?.(JSON.parse(String((event as MessageEvent).data)) as StreamEvent);
      } catch {
        // ข้ามข้อความที่ไม่ใช่ JSON
      }
    });
    socket.addEventListener("close", () => handlers.onClose?.());

    return {
      socket,
      ready,
      get open() {
        return socket.readyState === 1; // WebSocket.OPEN
      },
      send(text, options = {}) {
        if (socket.readyState !== 1) {
          throw new ThaiVoiceError(
            "ยังส่งไม่ได้ การเชื่อมต่อยังไม่พร้อม — รอ ready ให้เสร็จก่อน",
            0,
          );
        }
        socket.send(
          JSON.stringify({
            text,
            audio: options.audio ?? null,
            speaker_id: options.speakerId ?? null,
            speak: options.speak ?? true,
          }),
        );
      },
      close() {
        socket.close();
      },
    };
  }

  // ── ภายใน ──────────────────────────────────────────────────────────
  private async get<T>(path: string): Promise<T> {
    return this.request<T>("GET", path);
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>("POST", path, body);
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method,
      headers: body === undefined ? undefined : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return this.unwrap<T>(response);
  }

  private async unwrap<T>(response: Response): Promise<T> {
    if (!response.ok) {
      let detail: unknown;
      try {
        detail = await response.json();
      } catch {
        detail = await response.text().catch(() => "");
      }
      throw new ThaiVoiceError(describeError(detail, response.status), response.status, detail);
    }
    return (await response.json()) as T;
  }
}

export interface StreamConnection {
  socket: WebSocket;
  /** รอจนเชื่อมต่อสำเร็จ — reject ถ้าเชื่อมต่อไม่ได้หรือถูกปิดก่อน */
  ready: Promise<void>;
  /** การเชื่อมต่อพร้อมส่งข้อมูลแล้วหรือยัง */
  readonly open: boolean;
  send(
    text: string,
    options?: { audio?: string; speakerId?: number; speak?: boolean },
  ): void;
  close(): void;
}
