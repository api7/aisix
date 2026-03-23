import { randomUUID } from 'node:crypto';

import OpenAI from 'openai';

import {
  adminPost,
  bearerAuthHeader,
  startIsolatedAdminApp,
} from '../../utils/admin.js';
import { client } from '../../utils/http.js';
import { proxyAuthHeader, proxyPost } from '../../utils/proxy.js';
import { App } from '../../utils/setup.js';

const ADMIN_KEY = 'test_admin_key_embeddings_proxy';
const AUTHORIZED_KEY = 'sk-proxy-embeddings-authorized';
const LIMITED_KEY = 'sk-proxy-embeddings-limited';
const PROXY_EMBEDDINGS_URL = 'http://127.0.0.1:3000/v1/embeddings';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 500));
};

const sdkClient = (apiKey: string) =>
  new OpenAI({
    apiKey,
    baseURL: 'http://127.0.0.1:3000/v1',
  });

const expectSdkErrorStatus = (err: unknown, expectedStatus: number) => {
  const status =
    typeof err === 'object' && err !== null && 'status' in err
      ? Number((err as { status: unknown }).status)
      : Number.NaN;

  expect(Number.isFinite(status)).toBe(true);
  expect(status).toBe(expectedStatus);
};

describe('proxy /v1/embeddings', () => {
  let server: App | undefined;

  let embeddingModelName = '';
  let forbiddenModelName = '';
  let failingUpstreamModelName = '';

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);

    embeddingModelName = `embedding-${randomUUID()}`;
    forbiddenModelName = `embedding-forbidden-${randomUUID()}`;
    failingUpstreamModelName = `embedding-failing-${randomUUID()}`;

    const createEmbeddingModelResp = await adminPost(
      '/models',
      {
        name: embeddingModelName,
        model: 'mock/mock',
        provider_config: {},
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(createEmbeddingModelResp.status).toBe(201);

    const createForbiddenModelResp = await adminPost(
      '/models',
      {
        name: forbiddenModelName,
        model: 'mock/mock',
        provider_config: {},
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(createForbiddenModelResp.status).toBe(201);

    const createFailingModelResp = await adminPost(
      '/models',
      {
        name: failingUpstreamModelName,
        model: 'openai/failing-embedding-model',
        provider_config: {
          api_key: 'invalid-key',
          api_base: 'http://127.0.0.1:1/v1',
        },
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(createFailingModelResp.status).toBe(201);

    const authorizedResp = await adminPost(
      '/apikeys',
      {
        key: AUTHORIZED_KEY,
        allowed_models: [embeddingModelName, failingUpstreamModelName],
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(authorizedResp.status).toBe(201);

    const limitedResp = await adminPost(
      '/apikeys',
      {
        key: LIMITED_KEY,
        allowed_models: [embeddingModelName],
      },
      bearerAuthHeader(ADMIN_KEY),
    );
    expect(limitedResp.status).toBe(201);

    await waitConfigPropagation();
  });

  afterEach(async () => await server?.exit());

  test('authorized embeddings request returns success response', async () => {
    const resp = await proxyPost(
      '/v1/embeddings',
      {
        model: embeddingModelName,
        input: ['hello embeddings'],
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(200);
    expect(resp.data.object).toBe('list');
    expect(Array.isArray(resp.data.data)).toBe(true);
    expect(resp.data.data.length).toBe(1);
    expect(resp.data.data[0].object).toBe('embedding');
    expect(Array.isArray(resp.data.data[0].embedding)).toBe(true);
    expect(typeof resp.data.data[0].embedding[0]).toBe('number');
    expect(typeof resp.data.data[0].index).toBe('number');
    expect(typeof resp.data.usage.prompt_tokens).toBe('number');
    expect(typeof resp.data.usage.total_tokens).toBe('number');
    expect(resp.data.usage.total_tokens).toBeGreaterThan(0);
  });

  test('accessing forbidden embeddings model returns 403', async () => {
    const resp = await proxyPost(
      '/v1/embeddings',
      {
        model: forbiddenModelName,
        input: 'forbidden embeddings',
      },
      LIMITED_KEY,
    );

    expect(resp.status).toBe(403);
    expect(resp.data.error.code).toBe('model_access_forbidden');
  });

  test('invalid json for embeddings returns 400 invalid_json', async () => {
    const resp = await client.post(PROXY_EMBEDDINGS_URL, '{"model":', {
      headers: {
        ...proxyAuthHeader(AUTHORIZED_KEY),
        'Content-Type': 'application/json',
      },
    });

    expect(resp.status).toBe(400);
    expect(resp.data.error.code).toBe('invalid_json');
  });

  test('missing model field returns 400 invalid_json', async () => {
    const resp = await proxyPost(
      '/v1/embeddings',
      {
        input: 'missing model',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(400);
    expect(resp.data.error.code).toBe('invalid_json');
  });

  test('upstream failure is mapped to 502 provider_error', async () => {
    const resp = await proxyPost(
      '/v1/embeddings',
      {
        model: failingUpstreamModelName,
        input: 'trigger provider error',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(502);
    expect(resp.data.error.code).toBe('provider_error');
  });

  test('OpenAI SDK embeddings request works', async () => {
    const sdk = sdkClient(AUTHORIZED_KEY);

    const response = await sdk.embeddings.create({
      model: embeddingModelName,
      input: ['sdk embedding test'],
    });

    expect(response.object).toBe('list');
    expect(response.model).toBe(embeddingModelName);
    expect(Array.isArray(response.data)).toBe(true);
    expect(response.data.length).toBe(1);
    expect(response.data[0]?.object).toBe('embedding');
    expect(typeof response.data[0]?.embedding[0]).toBe('number');
    expect(typeof response.usage?.total_tokens).toBe('number');
  });

  test('OpenAI SDK invalid key returns 401 on embeddings', async () => {
    const sdk = sdkClient(`sk-invalid-${randomUUID()}`);

    try {
      await sdk.embeddings.create({
        model: embeddingModelName,
        input: 'sdk invalid key embeddings',
      });
      throw new Error('expected sdk request to fail');
    } catch (err) {
      expectSdkErrorStatus(err, 401);
    }
  });
});
