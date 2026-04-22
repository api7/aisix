import { randomUUID } from 'node:crypto';

import { bearerAuthHeader, startIsolatedAdminApp } from '../../utils/admin.js';
import {
  BedrockMockUpstream,
  type BedrockStreamEvent,
  buildBedrockProviderConfig,
  buildBedrockProviderModel,
  startBedrockMockUpstream,
} from '../../utils/bedrock-mock-upstream.js';
import { client } from '../../utils/http.js';
import { App, randomPort } from '../../utils/setup.js';
import {
  expectStreamHasDoneMarker,
  expectStreamHasUsageChunk,
} from '../../utils/stream-assert.js';

const ADMIN_KEY = 'test_admin_key_proxy_hook_bedrock_rate_limit';
const PROXY_KEY = 'sk-proxy-hook-bedrock-rate-limit';
const BEDROCK_RUNTIME_MODEL =
  'inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0';

const proxyPort = randomPort();
const adminPort = randomPort();

const proxyBaseUrl = () => `http://127.0.0.1:${proxyPort}`;
const adminBaseUrl = () => `http://127.0.0.1:${adminPort}/aisix/admin`;

const adminPostAt = async (
  path: string,
  body: unknown,
  headers: Record<string, string> = {},
) => client.post(`${adminBaseUrl()}${path}`, body, { headers });

const proxyPostAt = async (
  path: string,
  body: unknown,
  apiKey: string,
  config: Record<string, unknown> = {},
) =>
  client.post(`${proxyBaseUrl()}${path}`, body, {
    ...config,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      ...((config.headers as Record<string, string> | undefined) ?? {}),
    },
  });

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 500));
};

// This regression is Bedrock-specific: the generic limiter only sees token
// usage after the Bedrock metadata event has been normalized into a usage chunk.
const usageLimitedStreamEvents: BedrockStreamEvent[] = [
  {
    eventType: 'messageStart',
    payload: { role: 'assistant' },
  },
  {
    eventType: 'contentBlockDelta',
    payload: {
      contentBlockIndex: 0,
      delta: { text: 'token limited stream' },
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
        inputTokens: 8,
        outputTokens: 12,
        totalTokens: 20,
      },
    },
  },
];

describe('proxy hook consumes bedrock stream usage metadata', () => {
  let server: App | undefined;
  let upstream: BedrockMockUpstream | undefined;
  let modelName = '';

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY, {
      proxyPort,
      adminPort,
    });
    upstream = await startBedrockMockUpstream({
      streamEvents: usageLimitedStreamEvents,
    });

    modelName = `rate-limit-bedrock-model-${randomUUID()}`;

    const modelResp = await adminPostAt(
      '/models',
      {
        name: modelName,
        model: buildBedrockProviderModel(BEDROCK_RUNTIME_MODEL),
        provider_config: buildBedrockProviderConfig(upstream.baseUrl),
        rate_limit: {
          tpm: 20,
        },
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(modelResp.status, JSON.stringify(modelResp.data)).toBe(201);

    const apiKeyResp = await adminPostAt(
      '/apikeys',
      {
        key: PROXY_KEY,
        allowed_models: [modelName],
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(apiKeyResp.status, JSON.stringify(apiKeyResp.data)).toBe(201);

    await waitConfigPropagation();
  }, 30_000);

  afterEach(async () => {
    await upstream?.close();
    await server?.exit();
  });

  test('charges tpm after bedrock stream metadata is surfaced as usage', async () => {
    const firstResp = await proxyPostAt(
      '/v1/chat/completions',
      {
        model: modelName,
        stream: true,
        messages: [{ role: 'user', content: 'first token-metered stream' }],
      },
      PROXY_KEY,
      { responseType: 'text' },
    );

    expect(firstResp.status).toBe(200);
    expectStreamHasUsageChunk(String(firstResp.data));
    expectStreamHasDoneMarker(String(firstResp.data));

    await new Promise((resolve) => setTimeout(resolve, 100));

    const secondResp = await proxyPostAt(
      '/v1/chat/completions',
      {
        model: modelName,
        messages: [{ role: 'user', content: 'second request should fail' }],
      },
      PROXY_KEY,
    );

    expect(secondResp.status).toBe(429);
    expect(secondResp.data.error.code).toBe('rate_limit_exceeded');
  }, 15_000);
});
