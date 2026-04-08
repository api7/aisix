import {
  adminGet,
  bearerAuthHeader,
  startIsolatedAdminApp,
  xApiKeyHeader,
} from '../utils/admin.js';
import { App } from '../utils/setup.js';

const ADMIN_KEY = 'test_admin_key';

describe('admin auth', () => {
  let server: App | undefined;

  beforeEach(async () => {
    server = await startIsolatedAdminApp(ADMIN_KEY);
  });

  afterEach(async () => await server?.exit());

  test('auth_bearer_token_ok', async () => {
    const resp = await adminGet('/models', bearerAuthHeader(ADMIN_KEY));
    expect(resp.status).toBe(200);
  });

  test('auth_x_api_key_ok', async () => {
    const resp = await adminGet('/models', xApiKeyHeader(ADMIN_KEY));
    expect(resp.status).toBe(200);
  });

  test('auth_prefer_bearer_token', async () => {
    const resp = await adminGet('/models', {
      ...bearerAuthHeader(ADMIN_KEY),
      ...xApiKeyHeader('invalid_key'),
    });
    expect(resp.status).toBe(200);
  });

  test('no_auth_header', async () => {
    const resp = await adminGet('/models');
    expect(resp.status).toBe(401);
    expect(resp.data).toStrictEqual({ error_msg: 'Missing API key' });
  });

  test('invalid_auth_header', async () => {
    const resp = await adminGet('/models', bearerAuthHeader('invalid_token'));
    expect(resp.status).toBe(401);
    expect(resp.data).toStrictEqual({ error_msg: 'Invalid API key' });
  });
});
