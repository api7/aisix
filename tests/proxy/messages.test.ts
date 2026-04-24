import { randomUUID } from 'node:crypto';

import Anthropic from '@anthropic-ai/sdk';

import {
  MODELS_URL,
  PROVIDERS_URL,
  adminPost,
  adminPut,
  bearerAuthHeader,
  startIsolatedAdminApp,
} from '../utils/admin.js';
import { client } from '../utils/http.js';
import {
  type OpenAiMockUpstream,
  buildOpenAiProviderConfig,
  buildOpenAiProviderModel,
  startOpenAiMockUpstream,
} from '../utils/mock-upstream.js';
import { proxyAuthHeader, proxyPost } from '../utils/proxy.js';
import { App } from '../utils/setup.js';
import { expectSdkErrorStatus } from '../utils/stream-assert.js';

const ADMIN_KEY = 'test_admin_key_messages_proxy';
const AUTHORIZED_KEY = 'sk-proxy-messages-authorized';
const LIMITED_KEY = 'sk-proxy-messages-limited';
const UPSTREAM_API_KEY = 'upstream-key-messages-proxy';
const UPSTREAM_MODEL = 'test-model';
const PROXY_MESSAGES_URL = 'http://127.0.0.1:3000/v1/messages';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 1000));
};

const sdkClient = (apiKey: string) =>
  new Anthropic({
    apiKey,
    baseURL: 'http://127.0.0.1:3000',
  });

const parseAnthropicSseEvents = (sseBody: string) => {
  const trimmed = sseBody.trim();
  if (!trimmed) {
    return [] as Array<{ event?: string; data: string }>;
  }

  return trimmed.split(/\r?\n\r?\n/).map((block) => {
    const lines = block
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    return {
      event: lines.find((line) => line.startsWith('event: '))?.slice(7),
      data: lines
        .filter((line) => line.startsWith('data: '))
        .map((line) => line.slice(6))
        .join('\n'),
    };
  });
};

describe('proxy /v1/messages', () => {
  let server: App | undefined;
  let upstream: OpenAiMockUpstream | undefined;
  let mockModelName = '';
  let restrictedModelName = '';

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);
    upstream = await startOpenAiMockUpstream();
    const auth = bearerAuthHeader(ADMIN_KEY);

    mockModelName = `mock-messages-${randomUUID()}`;
    restrictedModelName = `mock-messages-restricted-${randomUUID()}`;
    const mockProviderId = `mock-messages-provider-${randomUUID()}`;
    const restrictedProviderId = `mock-messages-restricted-provider-${randomUUID()}`;

    const mockProviderResp = await adminPut(
      `${PROVIDERS_URL}/${mockProviderId}`,
      {
        name: mockProviderId,
        type: 'openai',
        config: buildOpenAiProviderConfig(upstream.apiBase, UPSTREAM_API_KEY),
      },
      auth,
    );
    expect(mockProviderResp.status).toBe(201);

    const mockModelResp = await adminPost(
      MODELS_URL,
      {
        name: mockModelName,
        model: buildOpenAiProviderModel(UPSTREAM_MODEL),
        provider_id: mockProviderId,
      },
      auth,
    );
    expect(mockModelResp.status).toBe(201);

    const restrictedProviderResp = await adminPut(
      `${PROVIDERS_URL}/${restrictedProviderId}`,
      {
        name: restrictedProviderId,
        type: 'openai',
        config: buildOpenAiProviderConfig(upstream.apiBase, UPSTREAM_API_KEY),
      },
      auth,
    );
    expect(restrictedProviderResp.status).toBe(201);

    const restrictedModelResp = await adminPost(
      MODELS_URL,
      {
        name: restrictedModelName,
        model: buildOpenAiProviderModel(UPSTREAM_MODEL),
        provider_id: restrictedProviderId,
      },
      auth,
    );
    expect(restrictedModelResp.status).toBe(201);

    const authorizedResp = await adminPost(
      '/apikeys',
      {
        key: AUTHORIZED_KEY,
        allowed_models: [mockModelName, restrictedModelName],
      },
      auth,
    );
    expect(authorizedResp.status).toBe(201);

    const limitedResp = await adminPost(
      '/apikeys',
      {
        key: LIMITED_KEY,
        allowed_models: [mockModelName],
      },
      auth,
    );
    expect(limitedResp.status).toBe(201);

    await waitConfigPropagation();
  });

  afterEach(async () => {
    await upstream?.close();
    await server?.exit();
  });

  test('authorized upstream-backed model returns anthropic response', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        messages: [{ role: 'user', content: 'hello from messages route' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.type).toBe('message');
    expect(resp.data.role).toBe('assistant');
    expect(Array.isArray(resp.data.content)).toBe(true);
    expect(resp.data.content[0].type).toBe('text');
    expect(resp.data.content[0].text).toBe('hello from mock upstream');
    expect(resp.data.usage.input_tokens).toBe(10);
    expect(resp.data.usage.output_tokens).toBe(8);

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
          messages: Array<{ content: string }>;
        }
      ).messages[0]?.content,
    ).toBe('hello from messages route');
  });

  test('unauthorized model returns forbidden error', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: restrictedModelName,
        max_tokens: 128,
        messages: [{ role: 'user', content: 'forbidden request' }],
      },
      LIMITED_KEY,
    );

    expect(resp.status).toBe(403);
    expect(resp.data.type).toBe('error');
    expect(resp.data.error.type).toBe('permission_error');
    expect(resp.data.error.message).toContain(restrictedModelName);
    expect(typeof resp.data.request_id).toBe('string');
  });

  test('authorized model with invalid json body returns extractor error', async () => {
    const resp = await client.post(PROXY_MESSAGES_URL, '{"model":', {
      headers: {
        ...proxyAuthHeader(AUTHORIZED_KEY),
        'Content-Type': 'application/json',
      },
    });

    expect(resp.status).toBe(422);
    expect(typeof resp.data).toBe('string');
  });

  test('missing auth header returns 401', async () => {
    const resp = await client.post(PROXY_MESSAGES_URL, {
      model: mockModelName,
      max_tokens: 128,
      messages: [{ role: 'user', content: 'missing auth' }],
    });

    expect(resp.status).toBe(401);
    expect(resp.data.error.message).toBe('Missing API key in request');
  });

  test('non-stream response follows anthropic shape', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        messages: [{ role: 'user', content: 'please echo this sentence' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(typeof resp.data.id).toBe('string');
    expect(resp.data.type).toBe('message');
    expect(resp.data.role).toBe('assistant');
    expect(Array.isArray(resp.data.content)).toBe(true);
    expect(resp.data.content[0].type).toBe('text');
    expect(typeof resp.data.usage.input_tokens).toBe('number');
    expect(typeof resp.data.usage.output_tokens).toBe('number');
  });

  test('Anthropic SDK messages request works', async () => {
    const sdk = sdkClient(AUTHORIZED_KEY);

    const response = await sdk.messages.create({
      model: mockModelName,
      max_tokens: 256,
      messages: [{ role: 'user', content: 'sdk messages test' }],
    });

    expect(response.type).toBe('message');
    expect(response.role).toBe('assistant');
    expect(typeof response.model).toBe('string');
    expect(response.model.length).toBeGreaterThan(0);
    expect(typeof response.usage.input_tokens).toBe('number');
    expect(typeof response.usage.output_tokens).toBe('number');

    const textBlock = response.content[0] as
      | { type: string; text?: string }
      | undefined;
    expect(textBlock?.type).toBe('text');
    expect(textBlock?.text).toBe('hello from mock upstream');
  });

  test('Anthropic SDK streaming messages request works', async () => {
    const sdk = sdkClient(AUTHORIZED_KEY);

    const stream = await sdk.messages.create({
      model: mockModelName,
      max_tokens: 256,
      stream: true,
      messages: [{ role: 'user', content: 'sdk stream test' }],
    });

    const eventTypes: string[] = [];
    let streamedText = '';

    for await (const event of stream) {
      eventTypes.push(event.type);

      if (
        event.type === 'content_block_delta' &&
        event.delta.type === 'text_delta'
      ) {
        streamedText += event.delta.text;
      }
    }

    expect(eventTypes.length).toBeGreaterThan(0);
    expect(eventTypes[0]).toBe('message_start');
    expect(eventTypes).toContain('content_block_start');
    expect(eventTypes).toContain('content_block_delta');
    expect(eventTypes).toContain('message_delta');
    expect(eventTypes.at(-1)).toBe('message_stop');
    expect(streamedText).toBe('hello from mock upstream');
  });

  test('Anthropic SDK invalid key returns 401', async () => {
    const sdk = sdkClient(`sk-invalid-${randomUUID()}`);

    try {
      await sdk.messages.create({
        model: mockModelName,
        max_tokens: 256,
        messages: [{ role: 'user', content: 'sdk invalid key test' }],
      });
      throw new Error('expected sdk request to fail');
    } catch (err) {
      expectSdkErrorStatus(err, 401);
    }
  });

  test('stream response emits anthropic event sequence without done marker', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        stream: true,
        messages: [{ role: 'user', content: 'stream once' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);
    expect(String(resp.headers['content-type'])).toContain('text/event-stream');

    const events = parseAnthropicSseEvents(String(resp.data));
    expect(events.length).toBeGreaterThan(0);
    expect(events.some((event) => event.data === '[DONE]')).toBe(false);
    expect(events[0]?.event).toBe('message_start');
    expect(events.some((event) => event.event === 'content_block_start')).toBe(
      true,
    );
    expect(events.some((event) => event.event === 'content_block_delta')).toBe(
      true,
    );
    expect(events.some((event) => event.event === 'message_delta')).toBe(true);
    expect(events.at(-1)?.event).toBe('message_stop');

    const parsed = events.map((event) => ({
      event: event.event,
      data: JSON.parse(event.data) as { type: string },
    }));

    for (const event of parsed) {
      expect(event.data.type).toBe(event.event);
    }
  });
});
