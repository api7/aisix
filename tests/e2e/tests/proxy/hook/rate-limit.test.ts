import { randomUUID } from 'node:crypto';

import {
  adminPost,
  bearerAuthHeader,
  startIsolatedAdminApp,
} from '../../../utils/admin.js';
import { proxyPost } from '../../../utils/proxy.js';
import { App } from '../../../utils/setup.js';

const ADMIN_KEY = 'test_admin_key_proxy_hook_rate_limit';
const PROXY_KEY = 'sk-proxy-hook-rate-limit';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 500));
};

describe('proxy hooks rate limit', () => {
  let server: App | undefined;
  let modelName = '';

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);

    modelName = `rate-limit-model-${randomUUID()}`;

    const modelResp = await adminPost(
      '/models',
      {
        name: modelName,
        model: 'mock/mock',
        provider_config: {},
        rate_limit: {
          tpm: 1000,
        },
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(modelResp.status).toBe(201);

    const apiKeyResp = await adminPost(
      '/apikeys',
      {
        key: PROXY_KEY,
        allowed_models: [modelName],
        rate_limit: {
          rpm: 2,
        },
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(apiKeyResp.status).toBe(201);

    await waitConfigPropagation();
  });

  afterEach(async () => await server?.exit());

  test('successful responses include rate limit headers', async () => {
    const firstResp = await proxyPost(
      '/v1/chat/completions',
      {
        model: modelName,
        messages: [{ role: 'user', content: 'first request' }],
      },
      PROXY_KEY,
    );

    expect(firstResp.status).toBe(200);

    const requestLimitHeader = firstResp.headers['x-ratelimit-limit-requests'];
    const requestRemainingHeader =
      firstResp.headers['x-ratelimit-remaining-requests'];
    const tokenLimitHeader = firstResp.headers['x-ratelimit-limit-tokens'];
    const tokenRemainingHeader =
      firstResp.headers['x-ratelimit-remaining-tokens'];

    expect(requestLimitHeader).toBeDefined();
    expect(requestRemainingHeader).toBeDefined();
    expect(tokenLimitHeader).toBeDefined();
    expect(tokenRemainingHeader).toBeDefined();

    const firstRemaining = Number(requestRemainingHeader);
    expect(Number.isFinite(firstRemaining)).toBe(true);

    const secondResp = await proxyPost(
      '/v1/chat/completions',
      {
        model: modelName,
        messages: [{ role: 'user', content: 'second request' }],
      },
      PROXY_KEY,
    );

    expect(secondResp.status).toBe(200);

    const secondRemaining = Number(
      secondResp.headers['x-ratelimit-remaining-requests'],
    );
    expect(Number.isFinite(secondRemaining)).toBe(true);
    expect(secondRemaining).toBeLessThan(firstRemaining);
  }, 15_000);

  test('requests exceeding rpm return 429 with retry-after', async () => {
    const statuses: number[] = [];
    let limitedResp: Awaited<ReturnType<typeof proxyPost>> | undefined;

    for (let i = 0; i < 6; i += 1) {
      const resp = await proxyPost(
        '/v1/chat/completions',
        {
          model: modelName,
          messages: [{ role: 'user', content: `request-${i + 1}` }],
        },
        PROXY_KEY,
      );
      statuses.push(resp.status);

      if (resp.status === 429) {
        limitedResp = resp;
        break;
      }
    }

    expect(limitedResp, `statuses: ${statuses.join(',')}`).toBeDefined();
    expect(limitedResp?.data.error.code).toBe('rate_limit_exceeded');

    const retryAfter = Number(limitedResp?.headers['retry-after']);
    expect(Number.isFinite(retryAfter)).toBe(true);
    expect(retryAfter).toBeGreaterThanOrEqual(0);
  }, 15_000);
});
