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

describe('admin apikeys', () => {
  let server: App | undefined;

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);
  });

  afterEach(async () => await server?.exit());

  test('test_crud', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);

    const listBefore = await adminGet('/apikeys', auth);
    expect(listBefore.status).toBe(200);
    expect(listBefore.data.total).toBe(0);

    const createResp = await adminPost(
      '/apikeys',
      {
        key: 'sk-test-crud',
        allowed_models: ['test-model-a'],
      },
      auth,
    );
    expect(createResp.status).toBe(201);
    const id = extractIdFromStorageKey(createResp.data.key);

    const listAfterCreate = await adminGet('/apikeys', auth);
    expect(listAfterCreate.status).toBe(200);
    expect(listAfterCreate.data.total).toBe(1);

    const updateResp = await adminPut(
      `/apikeys/${id}`,
      {
        key: 'sk-test-crud',
        allowed_models: ['test-model-a', 'test-model-b'],
      },
      auth,
    );
    expect(updateResp.status).toBe(200);
    expect(updateResp.data.value.allowed_models).toStrictEqual([
      'test-model-a',
      'test-model-b',
    ]);

    const getResp = await adminGet(`/apikeys/${id}`, auth);
    expect(getResp.status).toBe(200);
    expect(getResp.data.value.allowed_models).toStrictEqual([
      'test-model-a',
      'test-model-b',
    ]);

    const deleteResp = await adminDelete(`/apikeys/${id}`, auth);
    expect(deleteResp.status).toBe(200);
    expect(deleteResp.data.deleted).toBe(1);

    const listAfterDelete = await adminGet('/apikeys', auth);
    expect(listAfterDelete.status).toBe(200);
    expect(listAfterDelete.data.total).toBe(0);
  });

  test('test_put_status_codes', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const body = {
      key: 'sk-test-put-status',
      allowed_models: [],
    };

    const firstPut = await adminPut('/apikeys/put-status-fixed-id', body, auth);
    expect(firstPut.status).toBe(201);

    const secondPut = await adminPut(
      '/apikeys/put-status-fixed-id',
      body,
      auth,
    );
    expect(secondPut.status).toBe(200);
  });

  test('test_put_duplicate_key_rejected', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);

    const firstApiKey = {
      key: 'sk-put-dup-a',
      allowed_models: [],
    };

    const secondApiKey = {
      key: 'sk-put-dup-b',
      allowed_models: [],
    };

    const putA = await adminPut('/apikeys/put-dup-apikey-a', firstApiKey, auth);
    expect(putA.status).toBe(201);

    const putB = await adminPut(
      '/apikeys/put-dup-apikey-b',
      secondApiKey,
      auth,
    );
    expect(putB.status).toBe(201);

    const putDup = await adminPut(
      '/apikeys/put-dup-apikey-b',
      firstApiKey,
      auth,
    );
    expect(putDup.status).toBe(400);
    expect(putDup.data.error_msg).toBe('API key already exists');
  });

  test('test_duplicate_key_rejected', async () => {
    const auth = bearerAuthHeader(ADMIN_KEY);
    const body = {
      key: 'sk-duplicate',
      allowed_models: [],
    };

    const createResp = await adminPost('/apikeys', body, auth);
    expect(createResp.status).toBe(201);
    const id = extractIdFromStorageKey(createResp.data.key);

    const duplicateResp = await adminPost('/apikeys', body, auth);
    expect(duplicateResp.status).toBe(400);
    expect(duplicateResp.data.error_msg).toBe('API key already exists');

    const cleanupResp = await adminDelete(`/apikeys/${id}`, auth);
    expect(cleanupResp.status).toBe(200);
  });
});
