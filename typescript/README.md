# thaivoice-client

ไคลเอนต์ TypeScript สำหรับ [thaivoice](../README.md) — ระบบสนทนาด้วยเสียงภาษาไทยที่จดจำผู้สนทนาได้

## ติดตั้งและ build

```bash
npm install
npm run build      # สร้าง dist/ (สำหรับ Node) และ web/js/ (สำหรับเบราว์เซอร์)
```

## ใช้ใน Node

```ts
import { ThaiVoiceClient } from "thaivoice-client";

const client = new ThaiVoiceClient({ baseUrl: "http://127.0.0.1:8080" });

const result = await client.chat("สวัสดีครับ ผมชื่อเดชครับ");
console.log(result.speaker?.name);   // เดช

const speakers = await client.speakers();
const detail = await client.speaker(speakers[0].id);
console.log(detail.facts);           // สิ่งที่ระบบจำเกี่ยวกับเขา
```

## คุยด้วยเสียงในเบราว์เซอร์

`VoiceConversation` รวมสามอย่างเข้าด้วยกัน: ถอดเสียงไทยด้วย Web Speech API (`th-TH`),
อัดเสียงดิบคู่ขนานเพื่อให้เซิร์ฟเวอร์จดจำลายเสียง, และเล่นเสียงตอบทีละประโยค

```ts
import { ThaiVoiceClient } from "thaivoice-client";
import { VoiceConversation } from "thaivoice-client/browser";

const talk = new VoiceConversation(new ThaiVoiceClient({ baseUrl: location.origin }), {
  onTranscript: (text, isFinal) => console.log(isFinal ? "จบ:" : "…", text),
  onSpeaker: (speaker, method) => console.log("คุยกับ", speaker?.name, "ผ่าน", method),
  onReplyDelta: (text) => process.stdout.write(text),
  serverTts: true,
  bargeIn: false,
});

await talk.start();
```

Web Speech API ใช้ได้ดีบนเบราว์เซอร์ตระกูล Chromium — เช็คก่อนด้วย
`browserRecognitionAvailable()` ถ้าไม่รองรับ ให้ตั้ง `useBrowserRecognition: false`
แล้วใช้โหมดกดพูด (`pushToTalkStop()`) ซึ่งส่งไฟล์เสียงให้เซิร์ฟเวอร์ถอดแทน

## เครื่องมือย่อยที่ใช้เดี่ยว ๆ ได้

| ชื่อ | ทำอะไร |
|---|---|
| `PcmRecorder` | อัดเสียงดิบจากไมค์แล้วคืนเป็นไฟล์ WAV 16 kHz |
| `encodeWav()` | แปลง `Float32Array` เป็น Blob ของไฟล์ WAV |
| `AudioQueue` | เล่นเสียงหลายท่อนตามลำดับ และหยุดทันทีเมื่อผู้ใช้พูดแทรก |
| `ThaiSpeechSynthesis` | พูดภาษาไทยด้วยเสียงในเบราว์เซอร์ (ใช้เมื่อไม่มี TTS ฝั่งเซิร์ฟเวอร์) |
