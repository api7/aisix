import { randomUUID } from 'node:crypto';

import OpenAI from 'openai';

import {
  adminPost,
  bearerAuthHeader,
  startIsolatedAdminApp,
} from '../utils/admin.js';
import { client } from '../utils/http.js';
import { proxyAuthHeader, proxyPost } from '../utils/proxy.js';
import { App } from '../utils/setup.js';
import {
  expectParseableChatCompletionChunks,
  expectSdkErrorStatus,
  expectStreamHasDoneMarker,
} from '../utils/stream-assert.js';

const ADMIN_KEY = 'test_admin_key_chat_sim';
const AUTHORIZED_KEY = 'sk-proxy-sim-authorized';
const LIMITED_KEY = 'sk-proxy-sim-limited';
const PROXY_CHAT_URL = 'http://127.0.0.1:3000/v1/chat/completions';

// llm-d-inference-sim listens on port 18000 (mapped from container port 8000)
const SIM_API_BASE = 'http://127.0.0.1:18000/v1';
const SIM_MODEL = 'sim-model';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 500));
};

const sdkClient = (apiKey: string) =>
  new OpenAI({
    apiKey,
    baseURL: 'http://127.0.0.1:3000/v1',
  });

describe('proxy /v1/chat/completions (llm-d sim)', () => {
  let server: App | undefined;
  let simModelName = '';
  let restrictedModelName = '';

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);

    simModelName = `sim-chat-${randomUUID()}`;
    restrictedModelName = `sim-chat-restricted-${randomUUID()}`;

    const simModelResp = await adminPost(
      '/models',
      {
        name: simModelName,
        model: `openai/${SIM_MODEL}`,
        provider_config: {
          api_key: 'unused',
          api_base: SIM_API_BASE,
        },
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(simModelResp.status).toBe(201);

    const restrictedModelResp = await adminPost(
      '/models',
      {
        name: restrictedModelName,
        model: `openai/${SIM_MODEL}`,
        provider_config: {
          api_key: 'unused',
          api_base: SIM_API_BASE,
        },
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(restrictedModelResp.status).toBe(201);

    const authorizedResp = await adminPost(
      '/apikeys',
      {
        key: AUTHORIZED_KEY,
        allowed_models: [simModelName, restrictedModelName],
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(authorizedResp.status).toBe(201);

    const limitedResp = await adminPost(
      '/apikeys',
      {
        key: LIMITED_KEY,
        allowed_models: [simModelName],
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(limitedResp.status).toBe(201);

    await waitConfigPropagation();
  });

  afterEach(async () => await server?.exit());

  test('authorized sim model returns normal response', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: simModelName,
        messages: [{ role: 'user', content: 'hello from sim' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('chat.completion');
    expect(Array.isArray(resp.data.choices)).toBe(true);
    expect(resp.data.choices[0].message.role).toBe('assistant');
    expect(typeof resp.data.choices[0].message.content).toBe('string');
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
      model: simModelName,
      messages: [{ role: 'user', content: 'missing auth' }],
    });

    expect(resp.status).toBe(401);
    expect(resp.data.error.message).toBe('Missing API key in request');
  });

  test('invalid api key returns 401', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: simModelName,
        messages: [{ role: 'user', content: 'invalid auth' }],
      },
      'sk-invalid-sim',
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
        model: simModelName,
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
        model: simModelName,
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
  }, 15_000);

  test('stream response includes [DONE] marker', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: simModelName,
        stream: true,
        messages: [{ role: 'user', content: 'stream once' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expect(resp.status).toBe(200);
    expect(String(resp.headers['content-type'])).toContain('text/event-stream');

    expectStreamHasDoneMarker(String(resp.data));
  }, 15_000);

  test('stream chunks are parseable chat.completion.chunk objects', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: simModelName,
        stream: true,
        messages: [{ role: 'user', content: 'stream parse check' }],
      },
      AUTHORIZED_KEY,
      { responseType: 'text' },
    );

    expectParseableChatCompletionChunks(String(resp.data));
  }, 15_000);

  test('accepts common optional parameters', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: simModelName,
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
        model: simModelName,
        messages: [{ role: 'user', content: '你好，测试 emoji 😀 与中文' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.choices[0].message.role).toBe('assistant');
  }, 15_000);

  test('response includes numeric usage fields', async () => {
    const resp = await proxyPost(
      '/v1/chat/completions',
      {
        model: simModelName,
        messages: [{ role: 'user', content: 'usage field check' }],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(typeof resp.data.usage.prompt_tokens).toBe('number');
    expect(typeof resp.data.usage.completion_tokens).toBe('number');
    expect(typeof resp.data.usage.total_tokens).toBe('number');
    expect(resp.data.usage.total_tokens).toBeGreaterThan(0);
  }, 15_000);

  test('OpenAI SDK chat completion request works', async () => {
    const sdk = sdkClient(AUTHORIZED_KEY);

    const response = await sdk.chat.completions.create({
      model: simModelName,
      messages: [{ role: 'user', content: 'sdk chat completion test' }],
      temperature: 0,
    });

    expect(response.object).toBe('chat.completion');
    expect(typeof response.model).toBe('string');
    expect(response.model.length).toBeGreaterThan(0);
    expect(response.choices[0]?.message.role).toBe('assistant');
    expect(typeof response.usage?.total_tokens).toBe('number');
  }, 15_000);

  test('OpenAI SDK streaming chat request works', async () => {
    const sdk = sdkClient(AUTHORIZED_KEY);

    const stream = await sdk.chat.completions.create({
      model: simModelName,
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
  }, 15_000);

  test('OpenAI SDK invalid key returns 401', async () => {
    const sdk = sdkClient(`sk-invalid-${randomUUID()}`);

    try {
      await sdk.chat.completions.create({
        model: simModelName,
        messages: [{ role: 'user', content: 'sdk invalid key test' }],
      });
      throw new Error('expected sdk request to fail');
    } catch (err) {
      expectSdkErrorStatus(err, 401);
    }
  });
});
