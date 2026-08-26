/**
 * thaivoice-client — ไคลเอนต์ TypeScript สำหรับระบบสนทนาด้วยเสียงภาษาไทย
 *
 * ใช้ใน Node หรือฝั่งเซิร์ฟเวอร์::
 *
 *     import { ThaiVoiceClient } from "thaivoice-client";
 *     const client = new ThaiVoiceClient({ baseUrl: "http://127.0.0.1:8080" });
 *     const result = await client.chat("สวัสดีครับ ผมชื่อเดช");
 *
 * ใช้ในเบราว์เซอร์เพื่อคุยด้วยเสียง::
 *
 *     import { ThaiVoiceClient } from "thaivoice-client";
 *     import { VoiceConversation } from "thaivoice-client/browser";
 *     const talk = new VoiceConversation(client, { onReply: console.log });
 *     await talk.start();
 */

export { ThaiVoiceClient, ThaiVoiceError } from "./client.js";
export type { ClientOptions, StreamConnection } from "./client.js";
export type {
  ChatOptions,
  ChatResult,
  Fact,
  IdentifyMethod,
  Speaker,
  SpeakerDetail,
  StreamEvent,
  VoiceResult,
} from "./types.js";
export {
  AudioQueue,
  PcmRecorder,
  ThaiSpeechSynthesis,
  VoiceConversation,
  blobToBase64,
  browserRecognitionAvailable,
  encodeWav,
} from "./browserVoice.js";
export type {
  VoiceConversationHandlers,
  VoiceConversationOptions,
  VoiceState,
} from "./browserVoice.js";
