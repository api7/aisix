// Tiny fetch wrapper around the admin API. Returns parsed JSON on
// success, throws an `AdminApiError` carrying `error_msg` on failure
// so handlers can render the canonical envelope.

import type {
  AdminError,
  ApiKey,
  Model,
  ResourceEntry,
} from "./types";
import type { Connection } from "./storage";

export class AdminApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "AdminApiError";
    this.status = status;
  }
}

async function request<T>(
  conn: Connection,
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const url = `${conn.endpoint.replace(/\/$/, "")}${path}`;
  const init: RequestInit = {
    method,
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${conn.adminKey}`,
    },
  };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
  }
  const resp = await fetch(url, init);

  if (!resp.ok) {
    let msg = `${method} ${path} failed: ${resp.status} ${resp.statusText}`;
    try {
      const parsed = (await resp.json()) as AdminError;
      if (parsed && typeof parsed.error_msg === "string") {
        msg = parsed.error_msg;
      }
    } catch {
      // body wasn't JSON — keep the default status-line message
    }
    throw new AdminApiError(msg, resp.status);
  }

  // 204 / empty body → return undefined cast to T (the caller knows
  // its endpoint shape).
  if (resp.status === 204 || resp.headers.get("content-length") === "0") {
    return undefined as T;
  }
  return (await resp.json()) as T;
}

export const api = {
  listModels: (c: Connection): Promise<ResourceEntry<Model>[]> =>
    request(c, "GET", "/admin/v1/models"),
  createModel: (c: Connection, m: Model): Promise<ResourceEntry<Model>> =>
    request(c, "POST", "/admin/v1/models", m),
  updateModel: (
    c: Connection,
    id: string,
    m: Model,
  ): Promise<ResourceEntry<Model>> =>
    request(c, "PUT", `/admin/v1/models/${encodeURIComponent(id)}`, m),
  deleteModel: (c: Connection, id: string): Promise<unknown> =>
    request(c, "DELETE", `/admin/v1/models/${encodeURIComponent(id)}`),

  listApiKeys: (c: Connection): Promise<ResourceEntry<ApiKey>[]> =>
    request(c, "GET", "/admin/v1/apikeys"),
  createApiKey: (c: Connection, k: ApiKey): Promise<ResourceEntry<ApiKey>> =>
    request(c, "POST", "/admin/v1/apikeys", k),
  updateApiKey: (
    c: Connection,
    id: string,
    k: ApiKey,
  ): Promise<ResourceEntry<ApiKey>> =>
    request(c, "PUT", `/admin/v1/apikeys/${encodeURIComponent(id)}`, k),
  deleteApiKey: (c: Connection, id: string): Promise<unknown> =>
    request(c, "DELETE", `/admin/v1/apikeys/${encodeURIComponent(id)}`),
};
