import { randomUUID } from 'node:crypto';

import OpenAI from 'openai';

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
  OpenAiMockUpstream,
  buildOpenAiProviderConfig,
  buildOpenAiToolCallStreamEvents,
  buildOpenAiTrailingContentAfterFinishReasonStreamEvents,
  startOpenAiMockUpstream,
} from '../utils/mock-upstream.js';
import {
  parseSseDataEvents,
  proxyAuthHeader,
  proxyPost,
} from '../utils/proxy.js';
import { App } from '../utils/setup.js';
import {
  expectParseableChatCompletionChunks,
  expectSdkErrorStatus,
  expectStreamHasDoneMarker,
  expectStreamHasToolCallDeltas,
  expectStreamHasUsageOnlyChunk,
  expectStreamMatchesCompletion,
  expectStreamRetainsTrailingContentAfterFinishReason,
  expectStreamStopsBeforeDone,
  expectStreamToolCallsFormJson,
} from '../utils/stream-assert.js';

const ADMIN_KEY = 'test_admin_key_chat_proxy';
const AUTHORIZED_KEY = 'sk-proxy-chat-authorized';
const LIMITED_KEY = 'sk-proxy-chat-limited';
const UPSTREAM_API_KEY = 'upstream-key-chat-proxy';
const UPSTREAM_MODEL = 'test-model';
const PROXY_CHAT_URL = 'http://127.0.0.1:3000/v1/chat/completions';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 1000));
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
    const auth = bearerAuthHeader(ADMIN_KEY);

    mockModelName = `mock-chat-${randomUUID()}`;
    restrictedModelName = `mock-chat-restricted-${randomUUID()}`;
    const mockProviderId = `mock-chat-provider-${randomUUID()}`;
    const restrictedProviderId = `mock-chat-restricted-provider-${randomUUID()}`;

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
        model: UPSTREAM_MODEL,
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
        model: UPSTREAM_MODEL,
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

  test('authorized model with invalid json body returns extractor error', async () => {
    const resp = await client.post(PROXY_CHAT_URL, '{"model":', {
      headers: {
        ...proxyAuthHeader(AUTHORIZED_KEY),
        'Content-Type': 'application/json',
      },
    });

    expect(resp.status).toBe(422);
    expect(typeof resp.data).toBe('string');
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

  test('missing model field returns extractor rejection', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        messages: [{ role: 'user', content: 'missing model field' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(422);
    expect(typeof resp.data).toBe('string');
  });

  test('missing messages field returns extractor rejection', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(422);
    expect(typeof resp.data).toBe('string');
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

  test('stream response text matches non-stream completion content for the same request', async () => {
    const requestBody = {
      model: mockModelName,
      messages: [{ role: 'user', content: 'compare stream and non-stream' }],
    };

    const nonStreamResp = await proxyPost(
      '/v1/chat/completions',
      requestBody,
      AUTHORIZED_KEY,
    );
    const streamResp = await proxyPost(
      '/v1/chat/completions',
      {
        ...requestBody,
        stream: true,
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(nonStreamResp.status).toBe(200);
    expect(streamResp.status).toBe(200);

    const summary = expectStreamMatchesCompletion(
      String(streamResp.data),
      nonStreamResp.data as {
        choices: Array<{ message?: { content?: string | null } }>;
      },
    );

    expect(summary.finishReasons).toContain('stop');
  });

  test('stream request forwards include_usage to upstream and emits usage-only chunk', async () => {
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

    const usageChunks = expectStreamHasUsageOnlyChunk(String(resp.data));
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

  test('stream keeps trailing content even when upstream emits finish_reason before the last text delta', async () => {
    upstream?.configure({
      streamEvents:
        buildOpenAiTrailingContentAfterFinishReasonStreamEvents(UPSTREAM_MODEL),
    });

    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        stream: true,
        messages: [
          { role: 'user', content: 'emit a trailing delta after stop' },
        ],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);

    const summary = expectStreamRetainsTrailingContentAfterFinishReason(
      String(resp.data),
      'hello from trailing delta',
    );

    expect(summary.finishReasons).toContain('stop');
  });

  test('tool call stream and non-stream responses describe the same logical tool invocation', async () => {
    upstream?.configure({
      nonStreamBody: {
        id: 'chatcmpl-tool-nonstream-e2e-mock',
        object: 'chat.completion',
        created: 1,
        model: UPSTREAM_MODEL,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              tool_calls: [
                {
                  id: 'call_weather_1',
                  type: 'function',
                  function: {
                    name: 'get_weather',
                    arguments: '{"city":"Shanghai"}',
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
        },
      },
      streamEvents: buildOpenAiToolCallStreamEvents(UPSTREAM_MODEL),
    });

    const requestBody = {
      model: mockModelName,
      messages: [{ role: 'user', content: 'what is the weather in Shanghai?' }],
      tools: [
        {
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Fetches weather for a city',
            parameters: {
              type: 'object',
              properties: {
                city: { type: 'string' },
              },
              required: ['city'],
            },
          },
        },
      ],
      tool_choice: 'auto',
    };

    const nonStreamResp = await proxyPost(
      '/v1/chat/completions',
      requestBody,
      AUTHORIZED_KEY,
    );
    const streamResp = await proxyPost(
      '/v1/chat/completions',
      {
        ...requestBody,
        stream: true,
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(nonStreamResp.status).toBe(200);
    expect(streamResp.status).toBe(200);
    expect(nonStreamResp.data.choices[0]?.finish_reason).toBe('tool_calls');

    const responseToolCall = (
      nonStreamResp.data as {
        choices: Array<{
          message: {
            tool_calls: Array<{
              id: string;
              function: { name: string; arguments: string };
            }>;
          };
        }>;
      }
    ).choices[0]?.message.tool_calls[0];
    const { parsedToolCalls } = expectStreamToolCallsFormJson(
      String(streamResp.data),
    );

    expect(parsedToolCalls).toHaveLength(1);
    expect(parsedToolCalls[0]?.id).toBe(responseToolCall?.id);
    expect(parsedToolCalls[0]?.function.name).toBe(
      responseToolCall?.function.name,
    );
    expect(parsedToolCalls[0]?.parsedArguments).toEqual(
      JSON.parse(responseToolCall?.function.arguments ?? '{}'),
    );

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(2);
    expect(
      recorded.every((request) => {
        const bodyJson = request.bodyJson as {
          tools?: Array<{ function?: { name?: string } }>;
          tool_choice?: string;
        };

        return (
          bodyJson.tools?.[0]?.function?.name === 'get_weather' &&
          bodyJson.tool_choice === 'auto'
        );
      }),
    ).toBe(true);
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

  test('streaming response emits no events when upstream returns an empty stream', async () => {
    upstream?.configure({ streamEvents: [] });

    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        stream: true,
        messages: [
          { role: 'user', content: 'empty stream before first chunk' },
        ],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);
    expect(String(resp.headers['content-type'])).toContain('text/event-stream');
    expect(parseSseDataEvents(String(resp.data))).toHaveLength(0);
  });

  test('streaming response ends without [DONE] when upstream disconnects mid-stream', async () => {
    upstream?.configure({ disconnectAfterEvents: 2 });

    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        stream: true,
        messages: [
          { role: 'user', content: 'disconnect in the middle of stream' },
        ],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);
    expect(String(resp.headers['content-type'])).toContain('text/event-stream');

    const events = expectStreamStopsBeforeDone(String(resp.data));
    expect(events).toHaveLength(2);

    const chunks = events.map(
      (item) =>
        JSON.parse(item) as {
          object: string;
          choices: Array<{ delta: { content?: string; role?: string } }>;
        },
    );

    expect(chunks[0]?.object).toBe('chat.completion.chunk');
    expect(chunks[0]?.choices[0]?.delta.role).toBe('assistant');
    expect(chunks[1]?.choices[0]?.delta.content).toBe('from mock upstream');
  });

  test('structured output request forwards response_format json_schema to upstream', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [
          { role: 'user', content: 'respond with structured weather json' },
        ],
        response_format: {
          type: 'json_schema',
          json_schema: {
            name: 'weather_response',
            schema: {
              type: 'object',
              properties: {
                city: { type: 'string' },
                temperature_c: { type: 'number' },
              },
              required: ['city', 'temperature_c'],
            },
            strict: true,
          },
        },
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);

    const bodyJson = recorded[0]?.bodyJson as {
      response_format?: {
        type?: string;
        json_schema?: {
          name?: string;
          schema?: { required?: string[] };
          strict?: boolean;
        };
      };
    };

    expect(bodyJson.response_format?.type).toBe('json_schema');
    expect(bodyJson.response_format?.json_schema?.name).toBe(
      'weather_response',
    );
    expect(bodyJson.response_format?.json_schema?.strict).toBe(true);
    expect(bodyJson.response_format?.json_schema?.schema?.required).toEqual([
      'city',
      'temperature_c',
    ]);
  });

  test('multimodal message parts forward image_url content to upstream', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'describe this image' },
              {
                type: 'image_url',
                image_url: {
                  url: 'https://example.com/cat.png',
                  detail: 'high',
                },
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

    const bodyJson = recorded[0]?.bodyJson as {
      messages: Array<{
        content: Array<
          | { type: 'text'; text: string }
          | {
              type: 'image_url';
              image_url: { url: string; detail?: string };
            }
        >;
      }>;
    };

    expect(bodyJson.messages[0]?.content).toEqual([
      { type: 'text', text: 'describe this image' },
      {
        type: 'image_url',
        image_url: {
          url: 'https://example.com/cat.png',
          detail: 'high',
        },
      },
    ]);
  });

  test('response preserves reasoning token details and forwards reasoning extras upstream', async () => {
    upstream?.configure({
      nonStreamBody: {
        id: 'chatcmpl-reasoning-e2e-mock',
        object: 'chat.completion',
        created: 1,
        model: UPSTREAM_MODEL,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Reasoned answer from mock upstream',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 8,
          total_tokens: 18,
          prompt_tokens_details: {
            cached_tokens: 2,
          },
          completion_tokens_details: {
            reasoning_tokens: 5,
          },
        },
      },
    });

    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [{ role: 'user', content: 'show me your reasoning usage' }],
        reasoning_effort: 'high',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.usage.prompt_tokens_details.cached_tokens).toBe(2);
    expect(resp.data.usage.completion_tokens_details.reasoning_tokens).toBe(5);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(
      (recorded[0]?.bodyJson as { reasoning_effort?: string }).reasoning_effort,
    ).toBe('high');
  });

  test('stream usage-only chunk preserves reasoning token details', async () => {
    upstream?.configure({
      streamEvents: [
        {
          id: 'chatcmpl-reasoning-stream-e2e-mock',
          object: 'chat.completion.chunk',
          created: 1,
          model: UPSTREAM_MODEL,
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', content: 'Reasoned answer' },
              finish_reason: 'stop',
            },
          ],
        },
        {
          id: 'chatcmpl-reasoning-stream-e2e-mock',
          object: 'chat.completion.chunk',
          created: 1,
          model: UPSTREAM_MODEL,
          choices: [],
          usage: {
            prompt_tokens: 11,
            completion_tokens: 9,
            total_tokens: 20,
            prompt_tokens_details: {
              cached_tokens: 2,
            },
            completion_tokens_details: {
              reasoning_tokens: 5,
            },
          },
        },
        '[DONE]',
      ],
    });

    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        stream: true,
        messages: [
          { role: 'user', content: 'show me stream reasoning details' },
        ],
        reasoning_effort: 'high',
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);

    const usageChunks = expectStreamHasUsageOnlyChunk(String(resp.data));
    expect(usageChunks).toHaveLength(1);
    expect(usageChunks[0]?.usage?.prompt_tokens_details?.cached_tokens).toBe(2);
    expect(
      usageChunks[0]?.usage?.completion_tokens_details?.reasoning_tokens,
    ).toBe(5);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(
      (recorded[0]?.bodyJson as { reasoning_effort?: string }).reasoning_effort,
    ).toBe('high');
  });

  test('structured output request forwards response_format json_object to upstream', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [{ role: 'user', content: 'respond with a json object' }],
        response_format: {
          type: 'json_object',
        },
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(
      (recorded[0]?.bodyJson as { response_format?: { type?: string } })
        .response_format?.type,
    ).toBe('json_object');
  });

  test('multimodal data url image input forwards to upstream', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'describe this inline image' },
              {
                type: 'image_url',
                image_url: {
                  url: 'data:image/png;base64,aGVsbG8=',
                  detail: 'low',
                },
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
    expect(
      (
        recorded[0]?.bodyJson as {
          messages: Array<{
            content: Array<{
              type: string;
              image_url?: { url: string; detail?: string };
            }>;
          }>;
        }
      ).messages[0]?.content[1]?.image_url,
    ).toEqual({
      url: 'data:image/png;base64,aGVsbG8=',
      detail: 'low',
    });
  });

  test('tool_choice required forwards upstream and preserves tool-only assistant shape', async () => {
    upstream?.configure({
      nonStreamBody: {
        id: 'chatcmpl-tool-required-e2e-mock',
        object: 'chat.completion',
        created: 1,
        model: UPSTREAM_MODEL,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              tool_calls: [
                {
                  id: 'call_weather_required_1',
                  type: 'function',
                  function: {
                    name: 'get_weather',
                    arguments: '{"city":"Shanghai"}',
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
        },
      },
    });

    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [{ role: 'user', content: 'you must call the weather tool' }],
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              parameters: {
                type: 'object',
                properties: {
                  city: { type: 'string' },
                },
                required: ['city'],
              },
            },
          },
        ],
        tool_choice: 'required',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.choices[0]?.finish_reason).toBe('tool_calls');
    expect(resp.data.choices[0]?.message.tool_calls).toHaveLength(1);
    expect(resp.data.choices[0]?.message.content ?? null).toBeNull();

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(
      (recorded[0]?.bodyJson as { tool_choice?: string }).tool_choice,
    ).toBe('required');
  });

  test('forced function tool_choice forwards object form to upstream', async () => {
    upstream?.configure({
      nonStreamBody: {
        id: 'chatcmpl-tool-forced-e2e-mock',
        object: 'chat.completion',
        created: 1,
        model: UPSTREAM_MODEL,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              tool_calls: [
                {
                  id: 'call_weather_forced_1',
                  type: 'function',
                  function: {
                    name: 'get_weather',
                    arguments: '{"city":"Shanghai"}',
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
        },
      },
    });

    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: mockModelName,
        messages: [{ role: 'user', content: 'call get_weather only' }],
        tools: [
          {
            type: 'function',
            function: {
              name: 'get_weather',
              parameters: {
                type: 'object',
                properties: {
                  city: { type: 'string' },
                },
                required: ['city'],
              },
            },
          },
        ],
        tool_choice: {
          type: 'function',
          function: {
            name: 'get_weather',
          },
        },
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.choices[0]?.message.tool_calls[0]?.function.name).toBe(
      'get_weather',
    );

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
      function: {
        name: 'get_weather',
      },
    });
  });

  test('parallel tool calls stream and non-stream responses preserve two tool invocations', async () => {
    upstream?.configure({
      nonStreamBody: {
        id: 'chatcmpl-tool-parallel-e2e-mock',
        object: 'chat.completion',
        created: 1,
        model: UPSTREAM_MODEL,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              tool_calls: [
                {
                  id: 'call_weather_1',
                  type: 'function',
                  function: {
                    name: 'get_weather',
                    arguments: '{"city":"Shanghai"}',
                  },
                },
                {
                  id: 'call_time_1',
                  type: 'function',
                  function: {
                    name: 'get_time',
                    arguments: '{"timezone":"Asia/Shanghai"}',
                  },
                },
              ],
            },
            finish_reason: 'tool_calls',
          },
        ],
        usage: {
          prompt_tokens: 14,
          completion_tokens: 11,
          total_tokens: 25,
        },
      },
      streamEvents: [
        {
          id: 'chatcmpl-tool-parallel-e2e-mock',
          object: 'chat.completion.chunk',
          created: 1,
          model: UPSTREAM_MODEL,
          choices: [
            {
              index: 0,
              delta: { role: 'assistant' },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chatcmpl-tool-parallel-e2e-mock',
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
                      arguments: '',
                    },
                  },
                  {
                    index: 1,
                    id: 'call_time_1',
                    type: 'function',
                    function: {
                      name: 'get_time',
                      arguments: '',
                    },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chatcmpl-tool-parallel-e2e-mock',
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
                      arguments: '{"city":"Shanghai"}',
                    },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chatcmpl-tool-parallel-e2e-mock',
          object: 'chat.completion.chunk',
          created: 1,
          model: UPSTREAM_MODEL,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: 1,
                    function: {
                      arguments: '{"timezone":"Asia/Shanghai"}',
                    },
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chatcmpl-tool-parallel-e2e-mock',
          object: 'chat.completion.chunk',
          created: 1,
          model: UPSTREAM_MODEL,
          choices: [
            {
              index: 0,
              delta: {},
              finish_reason: 'tool_calls',
            },
          ],
        },
        {
          id: 'chatcmpl-tool-parallel-e2e-mock',
          object: 'chat.completion.chunk',
          created: 1,
          model: UPSTREAM_MODEL,
          choices: [],
          usage: {
            prompt_tokens: 14,
            completion_tokens: 11,
            total_tokens: 25,
          },
        },
        '[DONE]',
      ],
    });

    const requestBody = {
      model: mockModelName,
      messages: [
        { role: 'user', content: 'call weather and time tools in parallel' },
      ],
      tools: [
        {
          type: 'function',
          function: {
            name: 'get_weather',
            parameters: {
              type: 'object',
              properties: {
                city: { type: 'string' },
              },
              required: ['city'],
            },
          },
        },
        {
          type: 'function',
          function: {
            name: 'get_time',
            parameters: {
              type: 'object',
              properties: {
                timezone: { type: 'string' },
              },
              required: ['timezone'],
            },
          },
        },
      ],
      tool_choice: 'required',
      parallel_tool_calls: true,
    };

    const nonStreamResp = await proxyPost(
      '/v1/chat/completions',
      requestBody,
      AUTHORIZED_KEY,
    );
    const streamResp = await proxyPost(
      '/v1/chat/completions',
      {
        ...requestBody,
        stream: true,
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(nonStreamResp.status).toBe(200);
    expect(streamResp.status).toBe(200);

    const responseToolCalls = (
      nonStreamResp.data as {
        choices: Array<{
          message: {
            tool_calls: Array<{
              id: string;
              function: { name: string; arguments: string };
            }>;
          };
        }>;
      }
    ).choices[0]?.message.tool_calls;
    const { parsedToolCalls } = expectStreamToolCallsFormJson(
      String(streamResp.data),
    );

    expect(parsedToolCalls).toHaveLength(2);
    expect(parsedToolCalls.map((toolCall) => toolCall.function.name)).toEqual(
      responseToolCalls?.map((toolCall) => toolCall.function.name),
    );
    expect(parsedToolCalls.map((toolCall) => toolCall.parsedArguments)).toEqual(
      responseToolCalls?.map((toolCall) =>
        JSON.parse(toolCall.function.arguments),
      ),
    );

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(2);
    expect(
      recorded.every((request) => {
        const bodyJson = request.bodyJson as {
          parallel_tool_calls?: boolean;
          tool_choice?: string;
        };

        return (
          bodyJson.parallel_tool_calls === true &&
          bodyJson.tool_choice === 'required'
        );
      }),
    ).toBe(true);
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
