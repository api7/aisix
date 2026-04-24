import { randomUUID } from 'node:crypto';

import {
  MODELS_URL,
  PROVIDERS_URL,
  adminPost,
  adminPut,
  bearerAuthHeader,
} from '../utils/admin.js';
import { client } from '../utils/http.js';
import {
  OpenAiMockUpstream,
  buildOpenAiProviderModel,
  startOpenAiMockUpstream,
} from '../utils/mock-upstream.js';
import { App, defaultConfig } from '../utils/setup.js';

const ADMIN_KEY = 'test-admin-key-timeout';
const PROXY_KEY = 'sk-proxy-timeout';
const APIKEYS_URL = '/apikeys';
const PROXY_URL = 'http://127.0.0.1:3000';

describe('proxy timeout', () => {
  let server: App | undefined;
  let upstream: OpenAiMockUpstream | undefined;

  beforeEach(async () => {
    upstream = await startOpenAiMockUpstream({ responseDelayMs: 200 });

    server = await (
      await App.spawn(
        defaultConfig({
          deployment: {
            etcd: { prefix: `/${randomUUID()}` },
            admin: { admin_key: [{ key: ADMIN_KEY }] },
          },
        }),
      )
    )
      .waitForReady()
      .then((app) => app.waitForReady(3001));

    const auth = bearerAuthHeader(ADMIN_KEY);
    const providerId = 'timeout-model-provider';

    const providerRes = await adminPut(
      `${PROVIDERS_URL}/${providerId}`,
      {
        name: providerId,
        type: 'openai',
        config: {
          api_key: 'upstream-key-timeout',
          api_base: upstream.apiBase,
        },
      },
      auth,
    );
    expect(providerRes.status).toBe(201);

    const modelRes = await adminPost(
      MODELS_URL,
      {
        name: 'timeout-model',
        model: buildOpenAiProviderModel('timeout-model'),
        provider_id: providerId,
        timeout: 50,
      },
      auth,
    );
    expect(modelRes.status).toBe(201);

    const apikeyRes = await adminPost(
      APIKEYS_URL,
      {
        key: PROXY_KEY,
        allowed_models: ['timeout-model'],
      },
      auth,
    );
    expect(apikeyRes.status).toBe(201);

    // Wait for config to propagate from etcd to the proxy
    await new Promise((resolve) => setTimeout(resolve, 1000));
  });

  afterEach(async () => {
    await upstream?.close();
    await server?.exit();
  });

  test('chat completion returns 504 when upstream exceeds model timeout', async () => {
    const res = await client.post(
      `${PROXY_URL}/v1/chat/completions`,
      {
        model: 'timeout-model',
        messages: [{ role: 'user', content: 'hello' }],
      },
      { headers: { Authorization: `Bearer ${PROXY_KEY}` } },
    );

    expect(res.status).toBe(504);
    expect(res.data.error.code).toBe('request_timeout');
  });

  test('streaming chat completion returns 504 when upstream exceeds model timeout before the stream starts', async () => {
    const res = await client.post(
      `${PROXY_URL}/v1/chat/completions`,
      {
        model: 'timeout-model',
        stream: true,
        messages: [{ role: 'user', content: 'hello stream timeout' }],
      },
      { headers: { Authorization: `Bearer ${PROXY_KEY}` } },
    );

    expect(res.status).toBe(504);
    expect(res.data.error.code).toBe('request_timeout');
  });
});
