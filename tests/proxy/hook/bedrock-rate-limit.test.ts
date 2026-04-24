import { randomUUID } from 'node:crypto';

import {
  MODELS_URL,
  PROVIDERS_URL,
  adminPost,
  adminPut,
  bearerAuthHeader,
  startIsolatedAdminApp,
} from '../../utils/admin.js';
import {
  BedrockMockUpstream,
  type BedrockStreamEvent,
  buildBedrockProviderConfig,
  buildBedrockProviderModel,
  startBedrockMockUpstream,
} from '../../utils/bedrock-mock-upstream.js';
import { proxyPost } from '../../utils/proxy.js';
import { App } from '../../utils/setup.js';
import {
  expectStreamHasDoneMarker,
  expectStreamHasUsageChunk,
} from '../../utils/stream-assert.js';

const ADMIN_KEY = 'test_admin_key_proxy_hook_bedrock_rate_limit';
const PROXY_KEY = 'sk-proxy-hook-bedrock-rate-limit';
const BEDROCK_RUNTIME_MODEL =
  'inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 1000));
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
    server = await startIsolatedAdminApp(ADMIN_KEY);
    upstream = await startBedrockMockUpstream({
      streamEvents: usageLimitedStreamEvents,
    });
    const auth = bearerAuthHeader(ADMIN_KEY);

    modelName = `rate-limit-bedrock-model-${randomUUID()}`;
    const providerId = `rate-limit-bedrock-provider-${randomUUID()}`;

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
        rate_limit: {
          tpm: 20,
        },
      },
      auth,
    );
    expect(modelResp.status, JSON.stringify(modelResp.data)).toBe(201);

    const apiKeyResp = await adminPost(
      '/apikeys',
      {
        key: PROXY_KEY,
        allowed_models: [modelName],
      },
      auth,
    );
    expect(apiKeyResp.status, JSON.stringify(apiKeyResp.data)).toBe(201);

    await waitConfigPropagation();
  }, 30_000);

  afterEach(async () => {
    await upstream?.close();
    await server?.exit();
  });

  test('charges tpm after bedrock stream metadata is surfaced as usage', async () => {
    const firstResp = await proxyPost(
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

    const secondResp = await proxyPost(
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
