import { harnessRequest } from "./http.js";

/**
 * Thin typed wrapper over the Admin API. Keeps the test surface readable
 * — `await admin.createModel({...})` instead of inlined fetch boilerplate.
 */
export class AdminClient {
  constructor(
    private readonly baseUrl: string,
    private readonly adminKey: string,
  ) {}

  async createModel(
    model: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.json("POST", "/admin/v1/models", model);
  }

  async createApiKey(
    key: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.json("POST", "/admin/v1/apikeys", key);
  }

  async listModels(): Promise<Array<Record<string, unknown>>> {
    const res = await this.json<{ items?: Array<{ value: Record<string, unknown> }> }>(
      "GET",
      "/admin/v1/models",
    );
    return (res.items ?? []).map((entry) => entry.value);
  }

  async json<T = Record<string, unknown>>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const res = await harnessRequest(`${this.baseUrl}${path}`, {
      method,
      headers: {
        authorization: `Bearer ${this.adminKey}`,
        "content-type": "application/json",
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await res.body.text();
    if (res.statusCode >= 300) {
      throw new Error(
        `admin ${method} ${path} → ${res.statusCode}: ${text.slice(0, 512)}`,
      );
    }
    return text ? (JSON.parse(text) as T) : ({} as T);
  }
}

/** Convenience: wait the spec-mandated 500ms for snapshot propagation. */
export function waitConfigPropagation(): Promise<void> {
  return new Promise((r) => setTimeout(r, 500));
}
