import { ADMIN_BASE_URL, startIsolatedAdminApp } from '../../utils/admin.js';
import { client } from '../../utils/http.js';
import { App } from '../../utils/setup.js';

describe('admin ui', () => {
  let server: App | undefined;

  beforeEach(async () => {
    server = await startIsolatedAdminApp('test_admin_key');
  });

  afterEach(async () => await server?.exit());

  test('redirect_ui_root', async () => {
    const resp = await client.get(`${ADMIN_BASE_URL}/ui`, { maxRedirects: 0 });
    expect(resp.status).toBe(303);
    expect(resp.headers.location).toBe('/ui/');
  });

  test('serve_spa_index', async () => {
    const resp = await client.get(`${ADMIN_BASE_URL}/ui/`);
    expect(resp.status).toBe(200);
    expect(String(resp.headers['content-type'] ?? '')).toContain('text/html');
  });

  test('serve_explicit_index_html', async () => {
    const resp = await client.get(`${ADMIN_BASE_URL}/ui/index.html`);
    expect(resp.status).toBe(200);
    const body = String(resp.data ?? '');
    expect(
      body.includes('<!doctype html') || body.includes('<!DOCTYPE html'),
    ).toBe(true);
  });

  test('spa_fallback_for_client_routes', async () => {
    for (const path of ['/ui/models', '/ui/apikeys/create', '/ui/settings']) {
      const resp = await client.get(`${ADMIN_BASE_URL}${path}`);
      expect(resp.status).toBe(200);
      expect(String(resp.headers['content-type'] ?? '')).toContain('text/html');
    }
  });

  test('static_asset_not_found_returns_index', async () => {
    const resp = await client.get(
      `${ADMIN_BASE_URL}/ui/assets/definitely-not-real-xxxxxxxx.js`,
    );
    expect(resp.status).toBe(200);
  });

  test('openapi_endpoint_ok', async () => {
    const resp = await client.get(`${ADMIN_BASE_URL}/openapi`);
    expect(resp.status).toBe(200);
  });
});
