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
  buildOpenAiProviderModel,
  startOpenAiMockUpstream,
} from '../utils/mock-upstream.js';
import { proxyAuthHeader, proxyPost } from '../utils/proxy.js';
import { App } from '../utils/setup.js';
import { expectSdkErrorStatus } from '../utils/stream-assert.js';

const ADMIN_KEY = 'test_admin_key_embeddings_proxy';
const AUTHORIZED_KEY = 'sk-proxy-embeddings-authorized';
const LIMITED_KEY = 'sk-proxy-embeddings-limited';
const UPSTREAM_API_KEY = 'upstream-key-embeddings';
const FAILING_UPSTREAM_API_KEY = 'upstream-key-embeddings-failing';
const PROXY_EMBEDDINGS_URL = 'http://127.0.0.1:3000/v1/embeddings';

const waitConfigPropagation = async () => {
  await new Promise((resolve) => setTimeout(resolve, 1000));
};

const sdkClient = (apiKey: string) =>
  new OpenAI({
    apiKey,
    baseURL: 'http://127.0.0.1:3000/v1',
  });

describe('proxy /v1/embeddings', () => {
  let server: App | undefined;
  let upstream: OpenAiMockUpstream | undefined;
  let failingUpstream: OpenAiMockUpstream | undefined;

  let embeddingModelName = '';
  let forbiddenModelName = '';
  let failingUpstreamModelName = '';

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);
    upstream = await startOpenAiMockUpstream();
    const auth = bearerAuthHeader(ADMIN_KEY);
    failingUpstream = await startOpenAiMockUpstream({
      embeddings: {
        status: 500,
        errorBody: {
          error: {
            message: 'mock embeddings upstream error',
            type: 'mock_embeddings_upstream_error',
          },
        },
      },
    });

    embeddingModelName = `embedding-${randomUUID()}`;
    forbiddenModelName = `embedding-forbidden-${randomUUID()}`;
    failingUpstreamModelName = `embedding-failing-${randomUUID()}`;
    const embeddingProviderId = `embedding-provider-${randomUUID()}`;
    const forbiddenProviderId = `embedding-forbidden-provider-${randomUUID()}`;
    const failingProviderId = `embedding-failing-provider-${randomUUID()}`;

    const embeddingProviderResp = await adminPut(
      `${PROVIDERS_URL}/${embeddingProviderId}`,
      {
        name: embeddingProviderId,
        type: 'openai',
        config: buildOpenAiProviderConfig(upstream.apiBase, UPSTREAM_API_KEY),
      },
      auth,
    );
    expect(embeddingProviderResp.status).toBe(201);

    const createEmbeddingModelResp = await adminPost(
      MODELS_URL,
      {
        name: embeddingModelName,
        model: buildOpenAiProviderModel(embeddingModelName),
        provider_id: embeddingProviderId,
      },
      auth,
    );
    expect(createEmbeddingModelResp.status).toBe(201);

    const forbiddenProviderResp = await adminPut(
      `${PROVIDERS_URL}/${forbiddenProviderId}`,
      {
        name: forbiddenProviderId,
        type: 'openai',
        config: buildOpenAiProviderConfig(upstream.apiBase, UPSTREAM_API_KEY),
      },
      auth,
    );
    expect(forbiddenProviderResp.status).toBe(201);

    const createForbiddenModelResp = await adminPost(
      MODELS_URL,
      {
        name: forbiddenModelName,
        model: buildOpenAiProviderModel(forbiddenModelName),
        provider_id: forbiddenProviderId,
      },
      auth,
    );
    expect(createForbiddenModelResp.status).toBe(201);

    const failingProviderResp = await adminPut(
      `${PROVIDERS_URL}/${failingProviderId}`,
      {
        name: failingProviderId,
        type: 'openai',
        config: buildOpenAiProviderConfig(
          failingUpstream.apiBase,
          FAILING_UPSTREAM_API_KEY,
        ),
      },
      auth,
    );
    expect(failingProviderResp.status).toBe(201);

    const createFailingModelResp = await adminPost(
      MODELS_URL,
      {
        name: failingUpstreamModelName,
        model: buildOpenAiProviderModel(failingUpstreamModelName),
        provider_id: failingProviderId,
      },
      auth,
    );
    expect(createFailingModelResp.status).toBe(201);

    const authorizedResp = await adminPost(
      '/apikeys',
      {
        key: AUTHORIZED_KEY,
        allowed_models: [embeddingModelName, failingUpstreamModelName],
      },
      auth,
    );
    expect(authorizedResp.status).toBe(201);

    const limitedResp = await adminPost(
      '/apikeys',
      {
        key: LIMITED_KEY,
        allowed_models: [embeddingModelName],
      },
      auth,
    );
    expect(limitedResp.status).toBe(201);

    await waitConfigPropagation();
  });

  afterEach(async () => {
    await failingUpstream?.close();
    await upstream?.close();
    await server?.exit();
  });

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

    const recorded = upstream?.takeRecordedRequests() ?? [];
    expect(recorded).toHaveLength(1);
    expect(recorded[0]?.headers.authorization).toBe(
      `Bearer ${UPSTREAM_API_KEY}`,
    );
    expect((recorded[0]?.bodyJson as { model: string }).model).toBe(
      embeddingModelName,
    );
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

  test('invalid json for embeddings returns extractor error', async () => {
    const resp = await client.post(PROXY_EMBEDDINGS_URL, '{"model":', {
      headers: {
        ...proxyAuthHeader(AUTHORIZED_KEY),
        'Content-Type': 'application/json',
      },
    });

    expect(resp.status).toBe(422);
    expect(typeof resp.data).toBe('string');
  });

  test('missing model field returns extractor rejection', async () => {
    const resp = await proxyPost(
      '/v1/embeddings',
      {
        input: 'missing model',
      },
      AUTHORIZED_KEY,
    );

    expect(resp.status).toBe(422);
    expect(typeof resp.data).toBe('string');
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
