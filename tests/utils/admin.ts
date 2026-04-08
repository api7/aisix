import { randomUUID } from 'node:crypto';

import { client } from './http.js';
import { App, defaultConfig } from './setup.js';

export const ADMIN_BASE_URL = 'http://127.0.0.1:3001';
export const ADMIN_PREFIX = '/aisix/admin';

export const adminUrl = (path: string) =>
  `${ADMIN_BASE_URL}${ADMIN_PREFIX}${path}`;

export const bearerAuthHeader = (key: string) => ({
  Authorization: `Bearer ${key}`,
});

export const xApiKeyHeader = (key: string) => ({
  'x-api-key': key,
});

export const extractIdFromStorageKey = (storageKey: string) => {
  const id = storageKey.split('/').pop();
  if (!id) throw new Error(`invalid storage key: ${storageKey}`);
  return id;
};

export const startIsolatedAdminApp = async (adminKey: string) => {
  return (await (
    await App.spawn(
      defaultConfig({
        deployment: {
          etcd: { prefix: `/ai-admin-${randomUUID()}` },
          admin: { admin_key: [{ key: adminKey }] },
        },
      }),
    )
  )
    .waitForReady()
    .then((app) => app.waitForReady(3001))) as App;
};

export const adminGet = async (
  path: string,
  headers: Record<string, string> = {},
) => client.get(adminUrl(path), { headers });

export const adminPost = async (
  path: string,
  body: unknown,
  headers: Record<string, string> = {},
) => client.post(adminUrl(path), body, { headers });

export const adminPut = async (
  path: string,
  body: unknown,
  headers: Record<string, string> = {},
) => client.put(adminUrl(path), body, { headers });

export const adminDelete = async (
  path: string,
  headers: Record<string, string> = {},
) => client.delete(adminUrl(path), { headers });
