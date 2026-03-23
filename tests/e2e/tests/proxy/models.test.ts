import { randomUUID } from 'node:crypto';

import {
  adminPost,
  bearerAuthHeader,
  startIsolatedAdminApp,
} from '../../utils/admin.js';
import { proxyGet } from '../../utils/proxy.js';
import { App } from '../../utils/setup.js';

const ADMIN_KEY = 'test_admin_key_models_proxy';
const PROXY_KEY = 'sk-proxy-models';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 500));
};

const createModel = async (name: string) => {
  const resp = await adminPost(
    '/models',
    {
      name,
      model: 'mock/mock',
      provider_config: {},
    },
    bearerAuthHeader(ADMIN_KEY),
  );

  expect(resp.status).toBe(201);
};

const createApiKey = async (allowedModels: string[]) => {
  const resp = await adminPost(
    '/apikeys',
    {
      key: PROXY_KEY,
      allowed_models: allowedModels,
    },
    bearerAuthHeader(ADMIN_KEY),
  );

  expect(resp.status).toBe(201);
};

describe('proxy /v1/models', () => {
  let server: App | undefined;

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);
  });

  afterEach(async () => await server?.exit());

  test('returns empty model list by default', async () => {
    await createApiKey([]);
    await waitConfigPropagation();

    const resp = await proxyGet('/v1/models', PROXY_KEY);

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('list');
    expect(resp.data.data).toStrictEqual([]);
  });

  test('returns only allowed models when partially authorized', async () => {
    const modelA = `model-a-${randomUUID()}`;
    const modelB = `model-b-${randomUUID()}`;
    const modelC = `model-c-${randomUUID()}`;

    await createModel(modelA);
    await createModel(modelB);
    await createModel(modelC);

    await createApiKey([modelA, modelC]);
    await waitConfigPropagation();

    const resp = await proxyGet('/v1/models', PROXY_KEY);
    const ids = (resp.data.data as { id: string }[])
      .map((item) => item.id)
      .sort();

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('list');
    expect(ids).toStrictEqual([modelA, modelC].sort());
    expect(ids).not.toContain(modelB);

    for (const item of resp.data.data as {
      object: string;
      owned_by: string;
    }[]) {
      expect(item.object).toBe('model');
      expect(item.owned_by).toBe('apisix');
    }
  });

  test('returns all models when all models are authorized', async () => {
    const modelA = `model-all-a-${randomUUID()}`;
    const modelB = `model-all-b-${randomUUID()}`;
    const modelC = `model-all-c-${randomUUID()}`;

    await createModel(modelA);
    await createModel(modelB);
    await createModel(modelC);

    await createApiKey([modelA, modelB, modelC]);
    await waitConfigPropagation();

    const resp = await proxyGet('/v1/models', PROXY_KEY);
    const ids = (resp.data.data as { id: string }[])
      .map((item) => item.id)
      .sort();

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('list');
    expect(ids).toStrictEqual([modelA, modelB, modelC].sort());
    expect(resp.data.data).toHaveLength(3);
  });
});
