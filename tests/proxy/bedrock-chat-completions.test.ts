import { randomUUID } from 'node:crypto';

import {
  MODELS_URL,
  PROVIDERS_URL,
  adminPost,
  adminPut,
  bearerAuthHeader,
  startIsolatedAdminApp,
} from '../utils/admin.js';
import {
  BedrockMockUpstream,
  buildBedrockProviderConfig,
  buildBedrockProviderModel,
  startBedrockMockUpstream,
} from '../utils/bedrock-mock-upstream.js';
import { parseSseDataEvents, proxyPost } from '../utils/proxy.js';
import { App } from '../utils/setup.js';
import {
  expectParseableChatCompletionChunks,
  expectStreamHasDoneMarker,
  expectStreamHasUsageChunk,
} from '../utils/stream-assert.js';

const ADMIN_KEY = 'test_admin_key_bedrock_chat_proxy';
const AUTHORIZED_KEY = 'sk-proxy-bedrock-authorized';
const BEDROCK_RUNTIME_MODEL =
  'inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0';
const EXPECTED_ENCODED_PATH =
  '/model/inference-profile%2Fus.anthropic.claude-3-7-sonnet-20250219-v1:0';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 1000));
};

describe('proxy /v1/chat/completions with bedrock-backed model', () => {
  let server: App | undefined;
  let upstream: BedrockMockUpstream | undefined;
  let modelName = '';

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);
    upstream = await startBedrockMockUpstream();
    const auth = bearerAuthHeader(ADMIN_KEY);

    modelName = `mock-bedrock-chat-${randomUUID()}`;
    const providerId = `mock-bedrock-provider-${randomUUID()}`;

    const providerResp = await adminPut(
      `${PROVIDERS_URL}/${providerId}`,
      {
        name: providerId,
        type: 'bedrock',
        config: buildBedrockProviderConfig(upstream.baseUrl),
      },
      auth,
    );
    expect(providerResp.status).toBe(201);

    const modelResp = await adminPost(
      MODELS_URL,
      {
        name: modelName,
        model: buildBedrockProviderModel(BEDROCK_RUNTIME_MODEL),
        provider_id: providerId,
      },
      auth,
    );
    expect(modelResp.status).toBe(201);

    const apiKeyResp = await adminPost(
      '/apikeys',
      {
        key: AUTHORIZED_KEY,
        allowed_models: [modelName],
      },
      auth,
    );
    expect(apiKeyResp.status).toBe(201);

    await waitConfigPropagation();
  }, 30_000);

  afterEach(async () => {
    await upstream?.close();
    await server?.exit();
  });

  test('bedrock-backed model returns normal response and signs request', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: modelName,
        messages: [{ role: 'user', content: 'hello from bedrock proxy test' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('chat.completion');
    expect(resp.data.model).toBe(BEDROCK_RUNTIME_MODEL);
    expect(resp.data.choices[0].message.role).toBe('assistant');
    expect(resp.data.choices[0].message.content).toBe(
      'hello from mock bedrock',
    );
    expect(resp.data.usage.total_tokens).toBe(18);

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(recorded[0]?.url).toBe(`${EXPECTED_ENCODED_PATH}/converse`);
    expect(recorded[0]?.headers.authorization).toMatch(/^AWS4-HMAC-SHA256 /);
    expect(recorded[0]?.headers['x-amz-security-token']).toBe('session-token');

    const bodyJson = recorded[0]?.bodyJson as {
      messages: Array<{ content: Array<{ text: string }> }>;
    };
    expect(bodyJson.messages[0]?.content[0]?.text).toBe(
      'hello from bedrock proxy test',
    );
  });

  test('bedrock-backed stream emits chunks usage and done marker', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: modelName,
        stream: true,
        messages: [{ role: 'user', content: 'stream from bedrock proxy test' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);
    expect(String(resp.headers['content-type'])).toContain('text/event-stream');

    expectParseableChatCompletionChunks(String(resp.data));
    const usageChunks = expectStreamHasUsageChunk(String(resp.data));
    expect(usageChunks).toHaveLength(1);
    expectStreamHasDoneMarker(String(resp.data));

    const dataEvents = parseSseDataEvents(String(resp.data)).filter(
      (event) => event !== '[DONE]',
    );
    const textChunks = dataEvents
      .map(
        (event) =>
          JSON.parse(event) as {
            choices?: Array<{ delta?: { content?: string } }>;
          },
      )
      .map((chunk) => chunk.choices?.[0]?.delta?.content ?? '')
      .filter(Boolean);
    expect(textChunks.join('')).toBe('hello from mock bedrock stream');

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(recorded[0]?.url).toBe(`${EXPECTED_ENCODED_PATH}/converse-stream`);
    expect(recorded[0]?.headers.authorization).toMatch(/^AWS4-HMAC-SHA256 /);
    expect(recorded[0]?.headers['x-amz-security-token']).toBe('session-token');

    const bodyJson = recorded[0]?.bodyJson as {
      messages: Array<{ content: Array<{ text: string }> }>;
    };
    expect(bodyJson.messages[0]?.content[0]?.text).toBe(
      'stream from bedrock proxy test',
    );
  });
});
