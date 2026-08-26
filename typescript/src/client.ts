/**
 * ไคลเอนต์เรียกเซิร์ฟเวอร์ thaivoice — ใช้ได้ทั้งใน Node 18+ และเบราว์เซอร์
 *
 * ใช้เฉพาะ fetch/WebSocket ที่เป็นมาตรฐาน จึงไม่ต้องพึ่ง dependency ใด ๆ
 */

import type {
  ChatOptions,
  ChatResult,
  Speaker,
  SpeakerDetail,
  StreamEvent,
  VoiceResult,
} from "./types.js";

export interface ClientOptions {
  /** ที่อยู่เซิร์ฟเวอร์ เช่น http://127.0.0.1:8080 */
  baseUrl?: string;
  /** session เริ่มต้นสำหรับทุกคำขอ */
  sessionId?: string;
  fetchImpl?: typeof fetch;
}

export class ThaiVoiceError extends Error {
  constructor(message: string, readonly status: number, readonly detail?: unknown) {
    super(message);
    this.name = "ThaiVoiceError";
  }
}

export class ThaiVoiceClient {
  readonly baseUrl: string;
  readonly sessionId: string | undefined;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:8080").replace(/\/+$/, "");
    this.sessionId = options.sessionId;
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  /** คุยด้วยข้อความล้วน (ไคลเอนต์ถอดเสียงมาเองแล้ว) */
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
      throw new ThaiVoiceError("สังเคราะห์เสียงไม่สำเร็จ", response.status);
    }
    return response.blob();
  }

  /** ถามว่าเสียงนี้เป็นของใคร โดยไม่ต้องคุย */
  async identify(wav: Blob): Promise<{
    speaker: Speaker | null;
    score: number;
    is_new: boolean;
    method: string;
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

  /** ลบเฉพาะข้อเท็จจริงที่จำไว้ แต่ยังรู้จักคนนี้อยู่ */
  async forgetFacts(id: number): Promise<number> {
    const data = await this.request<{ removed: number }>(
      "DELETE",
      `/api/speakers/${id}/facts`,
    );
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
  stream(handlers: {
    onEvent?: (event: StreamEvent) => void;
    onOpen?: () => void;
    onClose?: () => void;
    onError?: (error: Event) => void;
    sessionId?: string;
  } = {}): StreamConnection {
    const sessionId = handlers.sessionId ?? this.sessionId;
    const url = new URL(`${this.baseUrl.replace(/^http/, "ws")}/ws/chat`);
    if (sessionId) url.searchParams.set("session_id", sessionId);

    const socket = new WebSocket(url.toString());
    const ready = new Promise<void>((resolve, reject) => {
      socket.addEventListener("open", () => {
        handlers.onOpen?.();
        resolve();
      });
      socket.addEventListener("error", (event) => {
        handlers.onError?.(event);
        reject(new Error("เชื่อมต่อ WebSocket ไม่สำเร็จ"));
      });
    });
    socket.addEventListener("message", (event) => {
      try {
        handlers.onEvent?.(JSON.parse(String(event.data)) as StreamEvent);
      } catch {
        // ข้ามข้อความที่ไม่ใช่ JSON
      }
    });
    socket.addEventListener("close", () => handlers.onClose?.());

    return {
      socket,
      ready,
      send(text: string, options: { audio?: string; speakerId?: number; speak?: boolean } = {}) {
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
        detail = await response.text();
      }
      const message =
        typeof detail === "object" && detail !== null && "detail" in detail
          ? String((detail as { detail: unknown }).detail)
          : `คำขอล้มเหลว (${response.status})`;
      throw new ThaiVoiceError(message, response.status, detail);
    }
    return (await response.json()) as T;
  }
}

export interface StreamConnection {
  socket: WebSocket;
  /** รอจนเชื่อมต่อสำเร็จ */
  ready: Promise<void>;
  send(
    text: string,
    options?: { audio?: string; speakerId?: number; speak?: boolean },
  ): void;
  close(): void;
}
