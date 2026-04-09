import {
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
const TEST_PROVIDER_MODEL = 'openai/test-admin-model';
const TEST_PROVIDER_CONFIG = { api_key: 'unused-admin-model-key' };

describe('admin models', () => {
  let server: App | undefined;

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);
  });

  afterEach(async () => await server?.exit());

  test('test_crud', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);

    const listBefore = await adminGet('/models', auth);
    expect(listBefore.status).toBe(200);
    expect(listBefore.data.total).toBe(0);

    const createResp = await adminPost(
      '/models',
      {
        name: 'test_model',
        model: TEST_PROVIDER_MODEL,
        provider_config: TEST_PROVIDER_CONFIG,
      },
      auth,
    );
    expect(createResp.status).toBe(201);
    const id = extractIdFromStorageKey(createResp.data.key);

    const listAfterCreate = await adminGet('/models', auth);
    expect(listAfterCreate.status).toBe(200);
    expect(listAfterCreate.data.total).toBe(1);

    const updateResp = await adminPut(
      `/models/${id}`,
      {
        name: 'updated_model',
        model: TEST_PROVIDER_MODEL,
        provider_config: TEST_PROVIDER_CONFIG,
      },
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
    const body = {
      name: 'put_model',
      model: TEST_PROVIDER_MODEL,
      provider_config: TEST_PROVIDER_CONFIG,
    };

    const firstPut = await adminPut('/models/put-test-fixed-id', body, auth);
    expect(firstPut.status).toBe(201);

    const secondPut = await adminPut('/models/put-test-fixed-id', body, auth);
    expect(secondPut.status).toBe(200);
  });

  test('test_put_duplicate_name_rejected', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);

    const firstModel = {
      name: 'put-dup-name-a',
      model: TEST_PROVIDER_MODEL,
      provider_config: TEST_PROVIDER_CONFIG,
    };

    const secondModel = {
      name: 'put-dup-name-b',
      model: TEST_PROVIDER_MODEL,
      provider_config: TEST_PROVIDER_CONFIG,
    };

    const putA = await adminPut('/models/put-dup-model-a', firstModel, auth);
    expect(putA.status).toBe(201);

    const putB = await adminPut('/models/put-dup-model-b', secondModel, auth);
    expect(putB.status).toBe(201);

    const putDup = await adminPut('/models/put-dup-model-b', firstModel, auth);
    expect(putDup.status).toBe(400);
    expect(putDup.data.error_msg).toBe('Model name already exists');
  });

  test('test_duplicate_name_rejected', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const body = {
      name: 'duplicate_model_name',
      model: TEST_PROVIDER_MODEL,
      provider_config: TEST_PROVIDER_CONFIG,
    };

    const createResp = await adminPost('/models', body, auth);
    expect(createResp.status).toBe(201);
    expect(typeof createResp.data.key).toBe('string');

    const duplicateResp = await adminPost('/models', body, auth);
    expect(duplicateResp.status).toBe(400);
    expect(duplicateResp.data.error_msg).toBe('Model name already exists');
  });
});
