/** ชนิดข้อมูลที่ใช้ร่วมกันระหว่างไคลเอนต์กับเซิร์ฟเวอร์ thaivoice */

/** ผู้สนทนาที่ระบบจดจำไว้ */
export interface Speaker {
  id: number;
  name: string;
  display_name: string;
  nickname: string | null;
  gender: "male" | "female" | null;
  /** คำลงท้ายที่ควรใช้กับคนนี้ เช่น "ครับ" หรือ "ค่ะ" */
  particle: string | null;
  last_seen_at: number;
  /** จำนวนเทิร์นที่เคยคุยกัน (มีเฉพาะบางปลายทาง) */
  turns?: number;
  /** จำนวนข้อเท็จจริงที่จำไว้ */
  facts?: number;
}

/** ข้อเท็จจริงหนึ่งข้อที่ระบบจำเกี่ยวกับผู้สนทนา */
export interface Fact {
  key: string;
  value: string;
  category: string;
  confidence: number;
}

/** วิธีที่ระบบใช้ระบุตัวผู้พูด */
export type IdentifyMethod = "voice" | "name" | "fallback" | "none";

export interface ChatResult {
  reply: string;
  chunks: string[];
  speaker: Speaker | null;
  session_id: string;
}

export interface VoiceResult extends ChatResult {
  transcript: string;
  identified_by: IdentifyMethod | null;
  /** เสียงคำตอบเป็น base64 (มีเมื่อเปิด TTS ฝั่งเซิร์ฟเวอร์) */
  audio: string | null;
}

export interface SpeakerDetail {
  speaker: Speaker;
  facts: Fact[];
  summary: string | null;
  recent_turns: { role: "user" | "assistant"; content: string; at: number }[];
}

/** เหตุการณ์ที่ไหลกลับมาทาง WebSocket ระหว่างที่บอทกำลังตอบ */
export type StreamEvent =
  /** ระบุตัวผู้พูดได้แล้ว */
  | {
      type: "speaker";
      text: string;
      speaker?: Speaker;
      identified_by?: IdentifyMethod;
      score?: number;
    }
  /** ข้อความที่กำลังพิมพ์ออกมาทีละชิ้น */
  | { type: "delta"; text: string }
  /** ประโยคที่พูดได้แล้ว พร้อมเสียง (ถ้าเซิร์ฟเวอร์สังเคราะห์ให้) */
  | { type: "chunk"; text: string; audio?: string; mime?: string }
  /** จบเทิร์น */
  | { type: "done"; text: string; speaker?: Speaker }
  | { type: "error"; text: string };

export interface ChatOptions {
  sessionId?: string;
  speakerId?: number;
  /** ให้เซิร์ฟเวอร์สังเคราะห์เสียงคำตอบมาด้วยหรือไม่ */
  speak?: boolean;
}
