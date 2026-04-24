import {
  MODELS_URL,
  PROVIDERS_URL,
  adminDelete,
  adminGet,
  adminPost,
  adminPut,
  bearerAuthHeader,
  extractIdFromStorageKey,
  startIsolatedAdminApp,
} from '../utils/admin.js';
import { App } from '../utils/setup.js';

const ADMIN_KEY = 'test_admin_key';
const TEST_PROVIDER_MODEL = 'test-admin-model';
const TEST_PROVIDER_CONFIG = { api_key: 'unused-admin-model-key' };

const buildModelBody = (name: string, providerId: string) => ({
  name,
  model: TEST_PROVIDER_MODEL,
  provider_id: providerId,
});

describe('admin models', () => {
  let server: App | undefined;

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);
  });

  afterEach(async () => await server?.exit());

  test('test_crud', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const providerId = 'test-model-provider';

    const listBefore = await adminGet('/models', auth);
    expect(listBefore.status).toBe(200);
    expect(listBefore.data.total).toBe(0);

    const providerResp = await adminPut(
      `${PROVIDERS_URL}/${providerId}`,
      {
        name: providerId,
        type: 'openai',
        config: TEST_PROVIDER_CONFIG,
      },
      auth,
    );
    expect(providerResp.status).toBe(201);

    const createResp = await adminPost(
      MODELS_URL,
      buildModelBody('test_model', providerId),
      auth,
    );
    expect(createResp.status).toBe(201);
    const id = extractIdFromStorageKey(createResp.data.key);

    const listAfterCreate = await adminGet('/models', auth);
    expect(listAfterCreate.status).toBe(200);
    expect(listAfterCreate.data.total).toBe(1);

    const updateResp = await adminPut(
      `${MODELS_URL}/${id}`,
      buildModelBody('updated_model', providerId),
      auth,
    );
    expect(updateResp.status).toBe(200);
    expect(updateResp.data.value.name).toBe('updated_model');

    const getResp = await adminGet(`/models/${id}`, auth);
    expect(getResp.status).toBe(200);
    expect(getResp.data.value.name).toBe('updated_model');

    const deleteResp = await adminDelete(`/models/${id}`, auth);
    expect(deleteResp.status).toBe(200);
    expect(deleteResp.data.deleted).toBe(1);

    const listAfterDelete = await adminGet('/models', auth);
    expect(listAfterDelete.status).toBe(200);
    expect(listAfterDelete.data.total).toBe(0);
  });

  test('test_put_status_codes', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const providerId = 'put-model-provider';
    const providerResp = await adminPut(
      `${PROVIDERS_URL}/${providerId}`,
      {
        name: providerId,
        type: 'openai',
        config: TEST_PROVIDER_CONFIG,
      },
      auth,
    );
    expect(providerResp.status).toBe(201);
    const body = buildModelBody('put_model', providerId);

    const firstPut = await adminPut(
      `${MODELS_URL}/put-test-fixed-id`,
      body,
      auth,
    );
    expect(firstPut.status).toBe(201);

    const secondPut = await adminPut(
      `${MODELS_URL}/put-test-fixed-id`,
      body,
      auth,
    );
    expect(secondPut.status).toBe(200);
  });

  test('test_put_duplicate_name_rejected', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const providerId = 'put-dup-provider';
    const providerResp = await adminPut(
      `${PROVIDERS_URL}/${providerId}`,
      {
        name: providerId,
        type: 'openai',
        config: TEST_PROVIDER_CONFIG,
      },
      auth,
    );
    expect(providerResp.status).toBe(201);

    const firstModel = buildModelBody('put-dup-name-a', providerId);

    const secondModel = buildModelBody('put-dup-name-b', providerId);

    const putA = await adminPut(
      `${MODELS_URL}/put-dup-model-a`,
      firstModel,
      auth,
    );
    expect(putA.status).toBe(201);

    const putB = await adminPut(
      `${MODELS_URL}/put-dup-model-b`,
      secondModel,
      auth,
    );
    expect(putB.status).toBe(201);

    const putDup = await adminPut(
      `${MODELS_URL}/put-dup-model-b`,
      firstModel,
      auth,
    );
    expect(putDup.status).toBe(400);
    expect(putDup.data.error_msg).toBe('Model name already exists');
  });

  test('test_duplicate_name_rejected', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const providerId = 'duplicate-model-provider';
    const providerResp = await adminPut(
      `${PROVIDERS_URL}/${providerId}`,
      {
        name: providerId,
        type: 'openai',
        config: TEST_PROVIDER_CONFIG,
      },
      auth,
    );
    expect(providerResp.status).toBe(201);
    const body = buildModelBody('duplicate_model_name', providerId);

    const createResp = await adminPost(MODELS_URL, body, auth);
    expect(createResp.status).toBe(201);
    expect(typeof createResp.data.key).toBe('string');

    const duplicateResp = await adminPost(MODELS_URL, body, auth);
    expect(duplicateResp.status).toBe(400);
    expect(duplicateResp.data.error_msg).toBe('Model name already exists');
  });
});
