import { randomUUID } from 'node:crypto';

import OpenAI from 'openai';

import {
  adminPost,
  bearerAuthHeader,
  startIsolatedAdminApp,
} from '../utils/admin.js';
import { client } from '../utils/http.js';
import {
  OpenAiMockUpstream,
  buildOpenAiProviderConfig,
  buildOpenAiProviderModel,
  buildOpenAiToolCallStreamEvents,
  startOpenAiMockUpstream,
} from '../utils/mock-upstream.js';
import { proxyAuthHeader, proxyPost } from '../utils/proxy.js';
import { App } from '../utils/setup.js';
import {
  expectParseableChatCompletionChunks,
  expectSdkErrorStatus,
  expectStreamHasDoneMarker,
  expectStreamHasToolCallDeltas,
  expectStreamHasUsageChunk,
} from '../utils/stream-assert.js';

const ADMIN_KEY = 'test_admin_key_chat_proxy';
const AUTHORIZED_KEY = 'sk-proxy-chat-authorized';
const LIMITED_KEY = 'sk-proxy-chat-limited';
const UPSTREAM_API_KEY = 'upstream-key-chat-proxy';
const UPSTREAM_MODEL = 'test-model';
const PROXY_CHAT_URL = 'http://127.0.0.1:3000/v1/chat/completions';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 500));
};

const sdkClient = (apiKey: string) =>
  new OpenAI({
    apiKey,
    baseURL: 'http://127.0.0.1:3000/v1',
  });

describe('proxy /v1/chat/completions', () => {
  let server: App | undefined;
  let upstream: OpenAiMockUpstream | undefined;
  let mockModelName = '';
  let restrictedModelName = '';

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);
    upstream = await startOpenAiMockUpstream();

    mockModelName = `mock-chat-${randomUUID()}`;
    restrictedModelName = `mock-chat-restricted-${randomUUID()}`;

    const mockModelResp = await adminPost(
      '/models',
      {
        name: mockModelName,
        model: buildOpenAiProviderModel(UPSTREAM_MODEL),
        provider_config: buildOpenAiProviderConfig(
          upstream.apiBase,
          UPSTREAM_API_KEY,
        ),
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(mockModelResp.status).toBe(201);

    const restrictedModelResp = await adminPost(
      '/models',
      {
        name: restrictedModelName,
        model: buildOpenAiProviderModel(UPSTREAM_MODEL),
        provider_config: buildOpenAiProviderConfig(
          upstream.apiBase,
          UPSTREAM_API_KEY,
        ),
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(restrictedModelResp.status).toBe(201);

    const authorizedResp = await adminPost(
      '/apikeys',
      {
        key: AUTHORIZED_KEY,
        allowed_models: [mockModelName, restrictedModelName],
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(authorizedResp.status).toBe(201);

    const limitedResp = await adminPost(
      '/apikeys',
      {
        key: LIMITED_KEY,
        allowed_models: [mockModelName],
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(limitedResp.status).toBe(201);

    await waitConfigPropagation();
  });

  afterEach(async () => {
    await upstream?.close();
    await server?.exit();
  });

  test('authorized upstream-backed model returns normal response', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [
          { role: 'user', content: 'hello from upstream-backed test' },
        ],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('chat.completion');
    expect(Array.isArray(resp.data.choices)).toBe(true);
    expect(resp.data.choices[0].message.role).toBe('assistant');
    expect(typeof resp.data.choices[0].message.content).toBe('string');
    expect(resp.data.choices[0].message.content).toBe(
      'hello from mock upstream',
    );

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(recorded[0]?.headers.authorization).toBe(
      `Bearer ${UPSTREAM_API_KEY}`,
    );
    expect(
      (
        recorded[0]?.bodyJson as {
          model: string;
          messages: Array<{ content: string }>;
        }
      ).model,
    ).toBe(UPSTREAM_MODEL);
    expect(
      (
        recorded[0]?.bodyJson as {
          model: string;
          messages: Array<{ content: string }>;
        }
      ).messages[0]?.content,
    ).toBe('hello from upstream-backed test');
  }, 15_000);

  test('unauthorized model returns forbidden error', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: restrictedModelName,
        messages: [{ role: 'user', content: 'forbidden request' }],
      },
      LIMITED_KEY,
    );

    expect(resp.status).toBe(403);
    expect(resp.data.error.code).toBe('model_access_forbidden');
  });

  test('authorized model with invalid json body returns parse error', async () => {
    const resp = await client.post(PROXY_CHAT_URL, '{"model":', {
      headers: {
        ...proxyAuthHeader(AUTHORIZED_KEY),
        'Content-Type': 'application/json',
      },
    });

    expect(resp.status).toBe(400);
    expect(resp.data.error.code).toBe('invalid_json');
  });

  test('missing auth header returns 401', async () => {
    const resp = await client.post(PROXY_CHAT_URL, {
      model: mockModelName,
      messages: [{ role: 'user', content: 'missing auth' }],
    });

    expect(resp.status).toBe(401);
    expect(resp.data.error.message).toBe('Missing API key in request');
  });

  test('invalid api key returns 401', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [{ role: 'user', content: 'invalid auth' }],
      },
      'sk-invalid-chat',
    );

    expect(resp.status).toBe(401);
    expect(resp.data.error.message).toBe('Invalid API key');
  });

  test('missing model field returns 400 invalid_json', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        messages: [{ role: 'user', content: 'missing model field' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.error.code).toBe('invalid_json');
  });

  test('missing messages field returns 400 invalid_json', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.error.code).toBe('invalid_json');
  });

  test('nonexistent model returns 400 model_not_found', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: `not-exist-${randomUUID()}`,
        messages: [{ role: 'user', content: 'missing model entity' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.error.code).toBe('model_not_found');
  });

  test('non-stream response follows openai shape', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [{ role: 'user', content: 'please echo this sentence' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('chat.completion');
    expect(typeof resp.data.id).toBe('string');
    expect(typeof resp.data.created).toBe('number');
    expect(Array.isArray(resp.data.choices)).toBe(true);
    expect(typeof resp.data.choices[0].index).toBe('number');
    expect(resp.data.choices[0].message.role).toBe('assistant');
  });

  test('stream response includes [DONE] marker', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        stream: true,
        messages: [{ role: 'user', content: 'stream once' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);
    expect(String(resp.headers['content-type'])).toContain('text/event-stream');

    expectStreamHasDoneMarker(String(resp.data));
  });

  test('stream chunks are parseable chat.completion.chunk objects', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        stream: true,
        messages: [{ role: 'user', content: 'stream parse check' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expectParseableChatCompletionChunks(String(resp.data));
  });

  test('stream request forwards include_usage to upstream and emits usage chunk', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        stream: true,
        messages: [{ role: 'user', content: 'stream usage forwarding check' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);

    const usageChunks = expectStreamHasUsageChunk(String(resp.data));
    expect(usageChunks).toHaveLength(1);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);

    const bodyJson = recorded[0]?.bodyJson as {
      model: string;
      stream: boolean;
      stream_options?: { include_usage?: boolean };
    };

    expect(bodyJson.model).toBe(UPSTREAM_MODEL);
    expect(bodyJson.stream).toBe(true);
    expect(bodyJson.stream_options?.include_usage).toBe(true);
  });

  test('stream response preserves tool call deltas from external upstream', async () => {
    upstream?.configure({
      streamEvents: buildOpenAiToolCallStreamEvents(UPSTREAM_MODEL),
    });

    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        stream: true,
        messages: [{ role: 'user', content: 'please emit a tool call delta' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);

    const { toolCallDeltas } = expectStreamHasToolCallDeltas(String(resp.data));
    const combinedArguments = toolCallDeltas
      .map((toolCall) => toolCall.function?.arguments ?? '')
      .join('');

    expect(toolCallDeltas[0]?.id).toBe('call_weather_1');
    expect(
      toolCallDeltas.some(
        (toolCall) => toolCall.function?.name === 'get_weather',
      ),
    ).toBe(true);
    expect(combinedArguments).toBe('{"city":"Shanghai"}');

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);

    const bodyJson = recorded[0]?.bodyJson as {
      stream: boolean;
      stream_options?: { include_usage?: boolean };
    };

    expect(bodyJson.stream).toBe(true);
    expect(bodyJson.stream_options?.include_usage).toBe(true);
  });

  test('accepts common optional parameters', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [{ role: 'user', content: 'optional params test' }],
        max_tokens: 16,
        temperature: 0.2,
        top_p: 0.7,
        n: 1,
        user: 'e2e-test-user',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('chat.completion');
  }, 15_000);

  test('supports unicode content', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [{ role: 'user', content: '你好，测试 emoji 😀 与中文' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.choices[0].message.role).toBe('assistant');
  });

  test('response includes numeric usage fields', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [{ role: 'user', content: 'usage field check' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(typeof resp.data.usage.prompt_tokens).toBe('number');
    expect(typeof resp.data.usage.completion_tokens).toBe('number');
    expect(typeof resp.data.usage.total_tokens).toBe('number');
    expect(resp.data.usage.total_tokens).toBeGreaterThan(0);
  });

  test('OpenAI SDK chat completion request works', async () => {
    const sdk = sdkClient(AUTHORIZED_KEY);

    const response = await sdk.chat.completions.create({
      model: mockModelName,
      messages: [{ role: 'user', content: 'sdk chat completion test' }],
      temperature: 0,
    });

    expect(response.object).toBe('chat.completion');
    expect(typeof response.model).toBe('string');
    expect(response.model.length).toBeGreaterThan(0);
    expect(response.choices[0]?.message.role).toBe('assistant');
    expect(typeof response.usage?.total_tokens).toBe('number');
  });

  test('OpenAI SDK streaming chat request works', async () => {
    const sdk = sdkClient(AUTHORIZED_KEY);

    const stream = await sdk.chat.completions.create({
      model: mockModelName,
      messages: [{ role: 'user', content: 'sdk stream completion test' }],
      stream: true,
    });

    let chunkCount = 0;
    let usageChunkCount = 0;

    for await (const chunk of stream) {
      chunkCount += 1;
      expect(chunk.object).toBe('chat.completion.chunk');
      if (chunk.usage) {
        usageChunkCount += 1;
        expect(typeof chunk.usage.total_tokens).toBe('number');
      }
    }

    expect(chunkCount).toBeGreaterThan(0);
    expect(usageChunkCount).toBeGreaterThan(0);
  });

  test('OpenAI SDK invalid key returns 401', async () => {
    const sdk = sdkClient(`sk-invalid-${randomUUID()}`);

    try {
      await sdk.chat.completions.create({
        model: mockModelName,
        messages: [{ role: 'user', content: 'sdk invalid key test' }],
      });
      throw new Error('expected sdk request to fail');
    } catch (err) {
      expectSdkErrorStatus(err, 401);
    }
  });
});
