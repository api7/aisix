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
  buildOpenAiToolCallStreamEvents,
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

  test('stream response emits the exact anthropic text lifecycle and terminal usage semantics', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        stream: true,
        messages: [{ role: 'user', content: 'stream exact lifecycle check' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);

    const events = parseAnthropicSseEvents(String(resp.data));
    expect(events.map((event) => event.event)).toEqual([
      'message_start',
      'content_block_start',
      'content_block_delta',
      'content_block_delta',
      'content_block_stop',
      'message_delta',
      'message_stop',
    ]);

    const messageStart = JSON.parse(events[0]?.data ?? '{}') as {
      message?: {
        usage?: {
          input_tokens?: number;
          output_tokens?: number;
        };
      };
    };
    const messageDelta = JSON.parse(
      events.find((event) => event.event === 'message_delta')?.data ?? '{}',
    ) as {
      delta?: { stop_reason?: string | null };
      usage?: {
        input_tokens?: number;
        output_tokens?: number;
        cache_creation_input_tokens?: number;
        cache_read_input_tokens?: number;
      };
    };

    expect(messageStart.message?.usage?.input_tokens).toBeUndefined();
    expect(messageStart.message?.usage?.output_tokens).toBeUndefined();
    expect(messageDelta.delta?.stop_reason).toBe('end_turn');
    expect(messageDelta.usage?.input_tokens).toBe(10);
    expect(messageDelta.usage?.output_tokens).toBe(8);
    expect(messageDelta.usage?.cache_creation_input_tokens).toBe(0);
    expect(messageDelta.usage?.cache_read_input_tokens).toBe(0);
  });

  test('stream response emits anthropic error events when bridge conversion fails mid-stream', async () => {
    upstream?.configure({
      streamEvents: [
        {
          id: 'chatcmpl-messages-error-e2e-mock',
          object: 'chat.completion.chunk',
          created: 1,
          model: UPSTREAM_MODEL,
          choices: [
            {
              index: 0,
              delta: {
                role: 'assistant',
                tool_calls: [
                  {
                    index: 0,
                    id: 'call_missing_type_1',
                    function: {
                      name: 'get_weather',
                      arguments: '{}',
                    },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        '[DONE]',
      ],
    });

    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        stream: true,
        messages: [{ role: 'user', content: 'emit malformed tool stream' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);

    const events = parseAnthropicSseEvents(String(resp.data));
    const errorEvent = events.find((event) => event.event === 'error');

    expect(errorEvent).toBeDefined();
    expect(events.some((event) => event.event === 'message_stop')).toBe(false);

    const errorPayload = JSON.parse(errorEvent?.data ?? '{}') as {
      type?: string;
      error?: { type?: string; message?: string };
    };

    expect(errorPayload.type).toBe('error');
    expect(errorPayload.error?.type).toBe('api_error');
    expect(errorPayload.error?.message).toContain('tool call types');
  });

  test('non-stream response maps tool calls into anthropic tool_use blocks', async () => {
    upstream?.configure({
      nonStreamBody: {
        id: 'chatcmpl-messages-tool-use-e2e-mock',
        object: 'chat.completion',
        created: 1,
        model: UPSTREAM_MODEL,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Calling tool',
              tool_calls: [
                {
                  id: 'call_weather_1',
                  type: 'function',
                  function: {
                    name: 'get_weather',
                    arguments: '{"city":"SF"}',
                  },
                },
              ],
            },
            finish_reason: 'tool_calls',
          },
        ],
        usage: {
          prompt_tokens: 12,
          completion_tokens: 7,
          total_tokens: 19,
          prompt_tokens_details: {
            cached_tokens: 2,
          },
        },
      },
    });

    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        messages: [{ role: 'user', content: 'call a tool' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.stop_reason).toBe('tool_use');
    expect(resp.data.usage.input_tokens).toBe(10);
    expect(resp.data.usage.cache_read_input_tokens).toBe(2);
    expect(resp.data.content[0]?.type).toBe('text');
    expect(resp.data.content[0]?.text).toBe('Calling tool');
    expect(resp.data.content[1]?.type).toBe('tool_use');
    expect(resp.data.content[1]?.name).toBe('get_weather');
    expect(resp.data.content[1]?.input).toEqual({ city: 'SF' });
  });

  test('request bridge maps anthropic tool_result blocks into upstream tool messages', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        messages: [
          {
            role: 'assistant',
            content: [
              {
                type: 'tool_use',
                id: 'tool_1',
                name: 'get_weather',
                input: { city: 'SF' },
              },
            ],
          },
          {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'tool_1',
                content: 'sunny',
              },
            ],
          },
        ],
        tools: [
          {
            name: 'get_weather',
            description: 'Get weather',
            input_schema: {
              type: 'object',
              properties: { city: { type: 'string' } },
            },
          },
        ],
        tool_choice: { type: 'auto' },
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);

    const bodyJson = recorded[0]?.bodyJson as {
      messages: Array<{
        role: string;
        content?: string;
        tool_calls?: Array<{ id?: string; function?: { name?: string } }>;
        tool_call_id?: string;
      }>;
      tools?: Array<{ function?: { name?: string } }>;
      tool_choice?: string;
    };

    expect(bodyJson.messages[0]?.role).toBe('assistant');
    expect(bodyJson.messages[0]?.tool_calls?.[0]?.id).toBe('tool_1');
    expect(bodyJson.messages[0]?.tool_calls?.[0]?.function?.name).toBe(
      'get_weather',
    );
    expect(bodyJson.messages[1]?.role).toBe('tool');
    expect(bodyJson.messages[1]?.tool_call_id).toBe('tool_1');
    expect(bodyJson.messages[1]?.content).toBe('sunny');
    expect(bodyJson.tools?.[0]?.function?.name).toBe('get_weather');
    expect(bodyJson.tool_choice).toBe('auto');
  });

  test('request bridge maps anthropic image blocks into upstream data url image parts', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'image',
                source: {
                  type: 'base64',
                  media_type: 'image/png',
                  data: 'aGVsbG8=',
                },
              },
              {
                type: 'text',
                text: 'describe this image',
              },
            ],
          },
        ],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);

    const contentParts = (
      recorded[0]?.bodyJson as {
        messages: Array<{
          content: Array<
            | { type: 'text'; text: string }
            | {
                type: 'image_url';
                image_url: { url: string; detail?: string };
              }
          >;
        }>;
      }
    ).messages[0]?.content;

    expect(contentParts).toContainEqual({
      type: 'text',
      text: 'describe this image',
    });
    expect(contentParts).toContainEqual({
      type: 'image_url',
      image_url: {
        url: 'data:image/png;base64,aGVsbG8=',
      },
    });
  });

  test('request bridge maps tool_choice any to upstream required mode', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        messages: [{ role: 'user', content: 'you must choose a tool' }],
        tools: [
          {
            name: 'get_weather',
            input_schema: {
              type: 'object',
              properties: { city: { type: 'string' } },
            },
          },
        ],
        tool_choice: { type: 'any' },
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(
      (recorded[0]?.bodyJson as { tool_choice?: string }).tool_choice,
    ).toBe('required');
  });

  test('request bridge maps named anthropic tool_choice to upstream function selection', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        messages: [{ role: 'user', content: 'call the weather tool only' }],
        tools: [
          {
            name: 'get_weather',
            input_schema: {
              type: 'object',
              properties: { city: { type: 'string' } },
            },
          },
        ],
        tool_choice: { type: 'tool', name: 'get_weather' },
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(
      (
        recorded[0]?.bodyJson as {
          tool_choice?: { type?: string; function?: { name?: string } };
        }
      ).tool_choice,
    ).toEqual({
      type: 'function',
      function: { name: 'get_weather' },
    });
  });

  test('request bridge forwards system blocks metadata top_k and stop_sequences upstream', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        system: [
          {
            type: 'text',
            text: 'You are helpful.',
            cache_control: { type: 'ephemeral' },
          },
        ],
        messages: [{ role: 'user', content: 'hello with anthropic extras' }],
        metadata: { user_id: 'user-123' },
        top_k: 5,
        stop_sequences: ['DONE', 'HALT'],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);

    const bodyJson = recorded[0]?.bodyJson as {
      messages: Array<{ role: string; content: string }>;
      user?: string;
      top_k?: number;
      stop?: string[];
    };

    expect(bodyJson.messages[0]).toEqual({
      role: 'system',
      content: 'You are helpful.',
    });
    expect(bodyJson.user).toBe('user-123');
    expect(bodyJson.top_k).toBe(5);
    expect(bodyJson.stop).toEqual(['DONE', 'HALT']);
  });

  test('request bridge rejects unsupported top-level cache_control', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        cache_control: { type: 'ephemeral' },
        messages: [
          { role: 'user', content: 'this should fail before upstream' },
        ],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.type).toBe('error');
    expect(resp.data.error.type).toBe('invalid_request_error');
    expect(upstream?.takeRecordedRequests() ?? []).toHaveLength(0);
  });

  test('request bridge rejects unsupported user content block cache_control', async () => {
    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: 'cached user block',
                cache_control: { type: 'ephemeral' },
              },
            ],
          },
        ],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.type).toBe('error');
    expect(resp.data.error.type).toBe('invalid_request_error');
    expect(upstream?.takeRecordedRequests() ?? []).toHaveLength(0);
  });

  test('stream bridge preserves text then tool_use lifecycle ordering', async () => {
    upstream?.configure({
      streamEvents: [
        {
          id: 'chatcmpl-messages-mixed-tool-e2e-mock',
          object: 'chat.completion.chunk',
          created: 1,
          model: UPSTREAM_MODEL,
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', content: 'Calling tool' },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chatcmpl-messages-mixed-tool-e2e-mock',
          object: 'chat.completion.chunk',
          created: 1,
          model: UPSTREAM_MODEL,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    id: 'call_weather_1',
                    type: 'function',
                    function: {
                      name: 'get_weather',
                      arguments: '{"city"',
                    },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chatcmpl-messages-mixed-tool-e2e-mock',
          object: 'chat.completion.chunk',
          created: 1,
          model: UPSTREAM_MODEL,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 0,
                    function: {
                      arguments: ':"SF"}',
                    },
                  },
                ],
              },
              finish_reason: 'tool_calls',
            },
          ],
        },
        '[DONE]',
      ],
    });

    const resp = await proxyPost(
      '/v1/messages',
      {
        model: mockModelName,
        max_tokens: 256,
        stream: true,
        messages: [{ role: 'user', content: 'call a tool after text' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);

    const events = parseAnthropicSseEvents(String(resp.data));
    expect(events.map((event) => event.event)).toEqual([
      'message_start',
      'content_block_start',
      'content_block_delta',
      'content_block_stop',
      'content_block_start',
      'content_block_delta',
      'content_block_delta',
      'content_block_stop',
      'message_delta',
      'message_stop',
    ]);

    const firstStart = JSON.parse(events[1]?.data ?? '{}') as {
      content_block?: { type?: string; text?: string };
    };
    const secondStart = JSON.parse(events[4]?.data ?? '{}') as {
      content_block?: { type?: string; name?: string };
    };
    const messageDelta = JSON.parse(events[8]?.data ?? '{}') as {
      delta?: { stop_reason?: string | null };
    };

    expect(firstStart.content_block?.type).toBe('text');
    expect(secondStart.content_block?.type).toBe('tool_use');
    expect(secondStart.content_block?.name).toBe('get_weather');
    expect(messageDelta.delta?.stop_reason).toBe('tool_use');
  });
});
