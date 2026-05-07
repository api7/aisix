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

  async createProviderKey(
    pk: Record<string, unknown>,
  ): Promise<{ id: string; value: Record<string, unknown> }> {
    return this.json("POST", "/admin/v1/provider_keys", pk);
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

/**
 * Wait for the gateway's in-memory snapshot to catch up with admin
 * writes. The spec mandates a ≤500ms propagation budget, but CI
 * runners with slower etcd/disk can occasionally exceed that — when
 * one of those runners only partially propagates a multi-resource
 * write batch, downstream tests see a snapshot with the Model but not
 * its referenced ProviderKey, and dispatch fails with `unknown
 * provider_key_id`.
 *
 * `condition` lets the caller provide a positive readiness probe; if
 * omitted, the helper falls back to the historical fixed-time wait.
 * The poll interval is 50ms and the deadline 5s — plenty of headroom
 * for any healthy runner without masking a genuine bug.
 */
export async function waitConfigPropagation(
  condition?: () => Promise<boolean>,
): Promise<void> {
  if (!condition) {
    await new Promise((r) => setTimeout(r, 500));
    return;
  }
  const deadline = Date.now() + 5_000;
  while (Date.now() < deadline) {
    if (await condition()) return;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error("waitConfigPropagation: condition not met within 5s");
}
