import { randomUUID } from 'node:crypto';

import { client } from '../utils/http.js';
import {
  OpenAiMockUpstream,
  buildOpenAiProviderConfig,
  buildOpenAiProviderModel,
  startOpenAiMockUpstream,
} from '../utils/mock-upstream.js';
import { App, defaultConfig } from '../utils/setup.js';

const ADMIN_KEY = 'test-admin-key-timeout';
const PROXY_KEY = 'sk-proxy-timeout';
const ADMIN_URL = 'http://127.0.0.1:3001';
const PROXY_URL = 'http://127.0.0.1:3000';
const ADMIN_PREFIX = '/aisix/admin';

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

    const modelRes = await client.post(
      `${ADMIN_URL}${ADMIN_PREFIX}/models`,
      {
        name: 'timeout-model',
        model: buildOpenAiProviderModel('timeout-model'),
        provider_config: buildOpenAiProviderConfig(
          upstream.apiBase,
          'upstream-key-timeout',
        ),
        timeout: 50,
      },
      { headers: { Authorization: `Bearer ${ADMIN_KEY}` } },
    );
    expect(modelRes.status).toBe(201);

    const apikeyRes = await client.post(
      `${ADMIN_URL}${ADMIN_PREFIX}/apikeys`,
      {
        key: PROXY_KEY,
        allowed_models: ['timeout-model'],
      },
      { headers: { Authorization: `Bearer ${ADMIN_KEY}` } },
    );
    expect(apikeyRes.status).toBe(201);

    // Wait for config to propagate from etcd to the proxy
    await new Promise((resolve) => setTimeout(resolve, 500));
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
});
