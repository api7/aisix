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

export interface OpenAiMockUpstreamOptions {
  model?: string;
  responseDelayMs?: number;
  eventDelayMs?: number;
  status?: number;
  errorBody?: Record<string, unknown>;
  nonStreamBody?: Record<string, unknown>;
  streamEvents?: Array<Record<string, unknown> | '[DONE]'>;
  embeddings?: {
    model?: string;
    responseDelayMs?: number;
    status?: number;
    errorBody?: Record<string, unknown>;
    body?: Record<string, unknown>;
  };
}

const sleep = async (ms: number) =>
  new Promise((resolve) => setTimeout(resolve, ms));

const OPENAI_CHAT_COMPLETION_PATHS = new Set([
  '/v1/chat/completions',
  '/chat/completions',
]);

const OPENAI_EMBEDDINGS_PATHS = new Set(['/v1/embeddings', '/embeddings']);

const readBody = async (req: NodeJS.ReadableStream) => {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString('utf8');
};

const defaultNonStreamBody = (model: string) => ({
  id: 'chatcmpl-e2e-mock',
  object: 'chat.completion',
  created: 1,
  model,
  choices: [
    {
      index: 0,
      message: {
        role: 'assistant',
        content: 'hello from mock upstream',
      },
      finish_reason: 'stop',
    },
  ],
  usage: {
    prompt_tokens: 10,
    completion_tokens: 8,
    total_tokens: 18,
  },
});

const defaultStreamEvents = (model: string) => [
  {
    id: 'chatcmpl-e2e-mock',
    object: 'chat.completion.chunk',
    created: 1,
    model,
    choices: [
      {
        index: 0,
        delta: { role: 'assistant', content: 'hello ' },
        finish_reason: null,
      },
    ],
  },
  {
    id: 'chatcmpl-e2e-mock',
    object: 'chat.completion.chunk',
    created: 1,
    model,
    choices: [
      {
        index: 0,
        delta: { content: 'from mock upstream' },
        finish_reason: null,
      },
    ],
  },
  {
    id: 'chatcmpl-e2e-mock',
    object: 'chat.completion.chunk',
    created: 1,
    model,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: 'stop',
      },
    ],
  },
  {
    id: 'chatcmpl-e2e-mock',
    object: 'chat.completion.chunk',
    created: 1,
    model,
    choices: [],
    usage: {
      prompt_tokens: 10,
      completion_tokens: 8,
      total_tokens: 18,
    },
  },
  '[DONE]' as const,
];

const defaultEmbeddingsBody = (model: string, inputCount: number) => {
  const dataLength = Math.max(inputCount, 1);

  return {
    object: 'list',
    data: Array.from({ length: dataLength }, (_, index) => ({
      object: 'embedding',
      embedding: [0.01 + index, 0.02 + index, 0.03 + index, 0.04 + index],
      index,
    })),
    model,
    usage: {
      prompt_tokens: dataLength * 3,
      total_tokens: dataLength * 3,
    },
  };
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

const requestModel = (bodyJson: unknown, fallbackModel: string) => {
  if (
    typeof bodyJson === 'object' &&
    bodyJson !== null &&
    'model' in bodyJson &&
    typeof (bodyJson as Record<string, unknown>).model === 'string'
  ) {
    return (bodyJson as Record<string, string>).model;
  }

  return fallbackModel;
};

const requestStream = (bodyJson: unknown) => {
  if (
    typeof bodyJson === 'object' &&
    bodyJson !== null &&
    'stream' in bodyJson
  ) {
    return Boolean((bodyJson as Record<string, unknown>).stream);
  }

  return false;
};

const embeddingInputCount = (bodyJson: unknown) => {
  if (
    typeof bodyJson !== 'object' ||
    bodyJson === null ||
    !('input' in bodyJson)
  ) {
    return 1;
  }

  const input = (bodyJson as Record<string, unknown>).input;
  if (Array.isArray(input)) {
    return input.length || 1;
  }

  return 1;
};

export class OpenAiMockUpstream {
  constructor(
    private readonly server: Server,
    private readonly sockets: Set<Socket>,
    private readonly requests: RecordedRequest[],
    readonly origin: string,
    readonly apiBase: string,
  ) {}

  get baseUrl() {
    return this.origin;
  }

  recordedRequests() {
    return [...this.requests];
  }

  takeRecordedRequests() {
    const recorded = [...this.requests];
    this.requests.length = 0;
    return recorded;
  }

  async close() {
    for (const socket of this.sockets) {
      socket.destroy();
    }

    this.server.close();
    await once(this.server, 'close');
  }
}

export const startOpenAiMockUpstream = async (
  options: OpenAiMockUpstreamOptions = {},
) => {
  const requests: RecordedRequest[] = [];
  const sockets = new Set<Socket>();

  const server = createServer(async (req, res) => {
    if (req.method !== 'POST') {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({ error: { message: 'mock upstream route not found' } }),
      );
      return;
    }

    const url = req.url ?? '/';
    if (
      !OPENAI_CHAT_COMPLETION_PATHS.has(url) &&
      !OPENAI_EMBEDDINGS_PATHS.has(url)
    ) {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({ error: { message: 'mock upstream route not found' } }),
      );
      return;
    }

    const bodyText = await readBody(req);
    const bodyJson = parseJsonBody(bodyText);
    requests.push({
      method: req.method,
      url: req.url ?? '/',
      headers: req.headers,
      bodyText,
      bodyJson,
    });

    if (OPENAI_EMBEDDINGS_PATHS.has(url)) {
      if (options.embeddings?.responseDelayMs) {
        await sleep(options.embeddings.responseDelayMs);
      }

      const model = requestModel(
        bodyJson,
        options.embeddings?.model ?? options.model ?? 'test-embedding-model',
      );
      const status = options.embeddings?.status ?? 200;
      if (status >= 400) {
        res.writeHead(status, { 'Content-Type': 'application/json' });
        res.end(
          JSON.stringify(
            options.embeddings?.errorBody ?? {
              error: {
                message: 'mock embeddings upstream error',
                type: 'mock_embeddings_upstream_error',
              },
            },
          ),
        );
        return;
      }

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify(
          options.embeddings?.body ??
            defaultEmbeddingsBody(model, embeddingInputCount(bodyJson)),
        ),
      );
      return;
    }

    if (options.responseDelayMs) {
      await sleep(options.responseDelayMs);
    }

    const model = requestModel(bodyJson, options.model ?? 'test-model');
    const status = options.status ?? 200;
    if (status >= 400) {
      res.writeHead(status, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify(
          options.errorBody ?? {
            error: {
              message: 'mock upstream error',
              type: 'mock_upstream_error',
            },
          },
        ),
      );
      return;
    }

    if (requestStream(bodyJson)) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      });

      for (const event of options.streamEvents ?? defaultStreamEvents(model)) {
        if (typeof event === 'string') {
          res.write(`data: ${event}\n\n`);
        } else {
          res.write(`data: ${JSON.stringify(event)}\n\n`);
        }

        if (options.eventDelayMs) {
          await sleep(options.eventDelayMs);
        }
      }

      res.end();
      return;
    }

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(
      JSON.stringify(options.nonStreamBody ?? defaultNonStreamBody(model)),
    );
  });

  server.on('connection', (socket) => {
    sockets.add(socket);
    socket.on('close', () => sockets.delete(socket));
  });

  server.listen(0, '127.0.0.1');
  await once(server, 'listening');

  const address = server.address() as AddressInfo;
  const origin = `http://127.0.0.1:${address.port}`;

  return new OpenAiMockUpstream(
    server,
    sockets,
    requests,
    origin,
    `${origin}/v1`,
  );
};
