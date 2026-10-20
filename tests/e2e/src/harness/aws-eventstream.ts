// Encoder for the AWS event-stream wire (`application/vnd.amazon.eventstream`)
// that Bedrock's ConverseStream answers with, so a mock upstream can serve
// a real-shaped stream to the gateway's AWS SDK client.
//
// Frame: total length (u32 BE) | headers length (u32 BE) | prelude CRC32 |
// headers | payload | message CRC32. Each header is name length (u8) | name |
// value type (u8, 7 = string) | value length (u16 BE) | value.
// https://docs.aws.amazon.com/transcribe/latest/dg/streaming-setting-up.html#streaming-event-stream

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function stringHeader(name: string, value: string): Buffer {
  const n = Buffer.from(name, "utf8");
  const v = Buffer.from(value, "utf8");
  const out = Buffer.alloc(1 + n.length + 1 + 2 + v.length);
  out.writeUInt8(n.length, 0);
  n.copy(out, 1);
  out.writeUInt8(7, 1 + n.length);
  out.writeUInt16BE(v.length, 2 + n.length);
  v.copy(out, 4 + n.length);
  return out;
}

/** One `:message-type: event` frame carrying a JSON payload. */
export function eventStreamFrame(eventType: string, payload: unknown): Buffer {
  const headers = Buffer.concat([
    stringHeader(":event-type", eventType),
    stringHeader(":content-type", "application/json"),
    stringHeader(":message-type", "event"),
  ]);
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  const total = 12 + headers.length + body.length + 4;
  const prelude = Buffer.alloc(8);
  prelude.writeUInt32BE(total, 0);
  prelude.writeUInt32BE(headers.length, 4);
  const preludeCrc = Buffer.alloc(4);
  preludeCrc.writeUInt32BE(crc32(prelude), 0);
  const withoutCrc = Buffer.concat([prelude, preludeCrc, headers, body]);
  const messageCrc = Buffer.alloc(4);
  messageCrc.writeUInt32BE(crc32(withoutCrc), 0);
  return Buffer.concat([withoutCrc, messageCrc]);
}

/** A whole event stream from `[eventType, payload]` pairs. */
export function eventStream(events: Array<[string, unknown]>): Buffer {
  return Buffer.concat(events.map(([type, payload]) => eventStreamFrame(type, payload)));
}

/** One `:message-type: exception` frame, the way Bedrock raises an error
 * inside a committed event stream (`throttlingException`, …). */
export function eventStreamException(exceptionType: string, payload: unknown): Buffer {
  const headers = Buffer.concat([
    stringHeader(":exception-type", exceptionType),
    stringHeader(":content-type", "application/json"),
    stringHeader(":message-type", "exception"),
  ]);
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  const total = 12 + headers.length + body.length + 4;
  const prelude = Buffer.alloc(8);
  prelude.writeUInt32BE(total, 0);
  prelude.writeUInt32BE(headers.length, 4);
  const preludeCrc = Buffer.alloc(4);
  preludeCrc.writeUInt32BE(crc32(prelude), 0);
  const withoutCrc = Buffer.concat([prelude, preludeCrc, headers, body]);
  const messageCrc = Buffer.alloc(4);
  messageCrc.writeUInt32BE(crc32(withoutCrc), 0);
  return Buffer.concat([withoutCrc, messageCrc]);
}

/** An `InvokeModelWithResponseStream` `chunk` frame carrying one Anthropic
 * Messages stream event. */
export function invokeChunk(event: unknown): Buffer {
  return eventStreamFrame("chunk", {
    bytes: Buffer.from(JSON.stringify(event), "utf8").toString("base64"),
  });
}
