import { once } from 'node:events';
import { type IncomingHttpHeaders, type Server, createServer } from 'node:http';
import type { AddressInfo, Socket } from 'node:net';

export interface RecordedRequest {
  method: string;
  url: string;
  headers: IncomingHttpHeaders;
  bodyText: string;
  bodyJson: unknown;
}

export interface BedrockStreamEvent {
  eventType: string;
  payload: Record<string, unknown>;
}

export interface BedrockMockUpstreamOptions {
  responseDelayMs?: number;
  eventDelayMs?: number;
  nonStreamBody?: Record<string, unknown>;
  streamEvents?: BedrockStreamEvent[];
}

const BEDROCK_CONVERSE_PATH = /^\/model\/([^/]+)\/(converse|converse-stream)$/;

const sleep = async (ms: number) =>
  new Promise((resolve) => setTimeout(resolve, ms));

const CRC32_TABLE = Uint32Array.from({ length: 256 }, (_, index) => {
  let crc = index;
  for (let i = 0; i < 8; i += 1) {
    crc = (crc & 1) === 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  }
  return crc >>> 0;
});

const crc32 = (buffer: Buffer) => {
  let crc = 0xffffffff;

  for (const byte of buffer) {
    crc = CRC32_TABLE[(crc ^ byte) & 0xff]! ^ (crc >>> 8);
  }

  return ~crc >>> 0;
};

const encodeStringHeader = (name: string, value: string) => {
  const nameBytes = Buffer.from(name, 'utf8');
  const valueBytes = Buffer.from(value, 'utf8');
  const valueLength = Buffer.alloc(2);
  valueLength.writeUInt16BE(valueBytes.length, 0);

  return Buffer.concat([
    Buffer.from([nameBytes.length]),
    nameBytes,
    Buffer.from([7]),
    valueLength,
    valueBytes,
  ]);
};

const encodeEventMessage = (
  eventType: string,
  payload: Record<string, unknown>,
) => {
  const headers = Buffer.concat([
    encodeStringHeader(':message-type', 'event'),
    encodeStringHeader(':event-type', eventType),
    encodeStringHeader(':content-type', 'application/json'),
  ]);
  const payloadBytes = Buffer.from(JSON.stringify(payload), 'utf8');
  const totalLength = 16 + headers.length + payloadBytes.length;

  const prelude = Buffer.alloc(8);
  prelude.writeUInt32BE(totalLength, 0);
  prelude.writeUInt32BE(headers.length, 4);

  const preludeCrc = Buffer.alloc(4);
  preludeCrc.writeUInt32BE(crc32(prelude), 0);

  const messageWithoutCrc = Buffer.concat([
    prelude,
    preludeCrc,
    headers,
    payloadBytes,
  ]);
  const messageCrc = Buffer.alloc(4);
  messageCrc.writeUInt32BE(crc32(messageWithoutCrc), 0);

  return Buffer.concat([messageWithoutCrc, messageCrc]);
};

const readBody = async (req: NodeJS.ReadableStream) => {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString('utf8');
};

const parseJsonBody = (bodyText: string) => {
  if (!bodyText) {
    return null;
  }

  try {
    return JSON.parse(bodyText) as unknown;
  } catch {
    return bodyText;
  }
};

const defaultNonStreamBody = () => ({
  output: {
    message: {
      role: 'assistant',
      content: [{ text: 'hello from mock bedrock' }],
    },
  },
  stopReason: 'end_turn',
  usage: {
    inputTokens: 10,
    outputTokens: 8,
    totalTokens: 18,
  },
});

const defaultStreamEvents = (): BedrockStreamEvent[] => [
  {
    eventType: 'messageStart',
    payload: { role: 'assistant' },
  },
  {
    eventType: 'contentBlockDelta',
    payload: {
      contentBlockIndex: 0,
      delta: { text: 'hello from mock ' },
    },
  },
  {
    eventType: 'contentBlockDelta',
    payload: {
      contentBlockIndex: 0,
      delta: { text: 'bedrock stream' },
    },
  },
  {
    eventType: 'messageStop',
    payload: { stopReason: 'end_turn' },
  },
  {
    eventType: 'metadata',
    payload: {
      usage: {
        inputTokens: 7,
        outputTokens: 9,
        totalTokens: 16,
      },
    },
  },
];

export class BedrockMockUpstream {
  constructor(
    private readonly server: Server,
    private readonly sockets: Set<Socket>,
    private readonly requests: RecordedRequest[],
    private readonly state: { options: BedrockMockUpstreamOptions },
    readonly origin: string,
  ) {}

  get baseUrl() {
    return this.origin;
  }

  takeRecordedRequests() {
    const recorded = [...this.requests];
    this.requests.length = 0;
    return recorded;
  }

  configure(options: Partial<BedrockMockUpstreamOptions>) {
    this.state.options = {
      ...this.state.options,
      ...options,
    };
  }

  async close() {
    for (const socket of this.sockets) {
      socket.destroy();
    }

    this.server.close();
    await once(this.server, 'close');
  }
}

export const startBedrockMockUpstream = async (
  options: BedrockMockUpstreamOptions = {},
) => {
  const requests: RecordedRequest[] = [];
  const sockets = new Set<Socket>();
  const state = { options };

  const server = createServer(async (req, res) => {
    if (req.method !== 'POST') {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ message: 'mock upstream route not found' }));
      return;
    }

    const url = req.url ?? '/';
    const match = BEDROCK_CONVERSE_PATH.exec(url);
    if (!match) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ message: 'mock upstream route not found' }));
      return;
    }

    const bodyText = await readBody(req);
    const bodyJson = parseJsonBody(bodyText);
    requests.push({
      method: req.method,
      url,
      headers: req.headers,
      bodyText,
      bodyJson,
    });

    const current = state.options;
    if (current.responseDelayMs) {
      await sleep(current.responseDelayMs);
    }

    const operation = match[2];
    if (operation === 'converse-stream') {
      res.writeHead(200, {
        'Content-Type': 'application/vnd.amazon.eventstream',
      });

      for (const event of current.streamEvents ?? defaultStreamEvents()) {
        res.write(encodeEventMessage(event.eventType, event.payload));
        if (current.eventDelayMs) {
          await sleep(current.eventDelayMs);
        }
      }

      res.end();
      return;
    }

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(current.nonStreamBody ?? defaultNonStreamBody()));
  });

  server.on('connection', (socket) => {
    sockets.add(socket);
    socket.on('close', () => sockets.delete(socket));
  });

  server.listen(0, '127.0.0.1');
  await once(server, 'listening');

  const address = server.address() as AddressInfo;
  const origin = `http://127.0.0.1:${address.port}`;

  return new BedrockMockUpstream(server, sockets, requests, state, origin);
};
