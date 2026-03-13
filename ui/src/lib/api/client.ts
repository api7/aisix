import type {
  ApiError,
  ApiKey,
  DeleteResponse,
  ItemResponse,
  ListResponse,
  Model,
} from './types';

/** Admin API base URL — proxied via Vite dev-server to avoid CORS */
const BASE = '/aisix/admin';

export class ApiClientError extends Error {
  readonly status: number;
  readonly body: ApiError;

  constructor(status: number, body: ApiError) {
    super(body.error_msg);
    this.name = 'ApiClientError';
    this.status = status;
    this.body = body;
  }
}

async function request<T>(
  method: string,
  path: string,
  adminKey: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${adminKey}`,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    let errBody: ApiError = { error_msg: res.statusText };
    try {
      errBody = await res.json();
    } catch {
      /* ignore */
    }
    throw new ApiClientError(res.status, errBody);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ── Models ────────────────────────────────────────────────────────────────────
export const modelsApi = {
  list: (adminKey: string) =>
    request<ListResponse<Model>>('GET', '/models', adminKey),

  get: (adminKey: string, id: string) =>
    request<ItemResponse<Model>>('GET', `/models/${id}`, adminKey),

  create: (adminKey: string, data: Model) =>
    request<ItemResponse<Model>>('POST', '/models', adminKey, data),

  update: (adminKey: string, id: string, data: Model) =>
    request<ItemResponse<Model>>('PUT', `/models/${id}`, adminKey, data),

  delete: (adminKey: string, id: string) =>
    request<DeleteResponse>('DELETE', `/models/${id}`, adminKey),
};

// ── ApiKeys ───────────────────────────────────────────────────────────────────
export const apiKeysApi = {
  list: (adminKey: string) =>
    request<ListResponse<ApiKey>>('GET', '/apikeys', adminKey),

  get: (adminKey: string, id: string) =>
    request<ItemResponse<ApiKey>>('GET', `/apikeys/${id}`, adminKey),

  create: (adminKey: string, data: ApiKey) =>
    request<ItemResponse<ApiKey>>('POST', '/apikeys', adminKey, data),

  update: (adminKey: string, id: string, data: ApiKey) =>
    request<ItemResponse<ApiKey>>('PUT', `/apikeys/${id}`, adminKey, data),

  delete: (adminKey: string, id: string) =>
    request<DeleteResponse>('DELETE', `/apikeys/${id}`, adminKey),
};
