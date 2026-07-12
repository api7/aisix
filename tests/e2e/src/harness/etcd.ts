import { harnessRequest } from "./http.js";

/**
 * Minimal etcd v3 helper that talks to the JSON gRPC-gateway
 * (`/v3/kv/*` endpoints). Avoids pulling a heavy etcd npm dependency.
 */
export class EtcdClient {
  constructor(
    private readonly endpoint: string = process.env.AISIX_E2E_ETCD ??
      "http://127.0.0.1:2379",
  ) {}

  /**
   * Best-effort connectivity probe — returns false if etcd isn't reachable.
   *
   * Beyond a 200 status we also confirm the response is JSON containing the
   * expected `header.cluster_id` field. A stray Docker port-mapping or a
   * dev-tool's "service unavailable" HTML page can return 200 to anything
   * on port 2379 and we don't want to misidentify those as etcd.
   */
  async ping(timeoutMs = 1000): Promise<boolean> {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeoutMs);
      const res = await harnessRequest(`${this.endpoint}/v3/maintenance/status`, {
        method: "POST",
        body: "{}",
        headers: { "content-type": "application/json" },
        signal: ctrl.signal,
      });
      clearTimeout(t);
      if (res.statusCode !== 200) {
        await res.body.dump();
        return false;
      }
      const text = await res.body.text();
      try {
        const parsed = JSON.parse(text) as { header?: { cluster_id?: string } };
        return typeof parsed.header?.cluster_id === "string";
      } catch {
        return false;
      }
    } catch {
      return false;
    }
  }

  /**
   * Put a single key/value (etcd v3 `/v3/kv/put`). Used to seed
   * resources the Admin API doesn't expose (e.g. rate_limit_policies),
   * written under `<prefix>/<kind>/<id>` so the DP loader picks them up.
   */
  async put(key: string, value: string): Promise<void> {
    const res = await harnessRequest(`${this.endpoint}/v3/kv/put`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        key: Buffer.from(key, "utf8").toString("base64"),
        value: Buffer.from(value, "utf8").toString("base64"),
      }),
    });
    if (res.statusCode >= 300) {
      const body = await res.body.text();
      throw new Error(`etcd put failed (${res.statusCode}): ${body}`);
    }
  }

  /** Read one key's value (etcd v3 `/v3/kv/range`); undefined when absent. */
  async get(key: string): Promise<string | undefined> {
    const res = await harnessRequest(`${this.endpoint}/v3/kv/range`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ key: Buffer.from(key, "utf8").toString("base64") }),
    });
    if (res.statusCode >= 300) {
      const body = await res.body.text();
      throw new Error(`etcd range failed (${res.statusCode}): ${body}`);
    }
    const parsed = JSON.parse(await res.body.text()) as {
      kvs?: Array<{ value: string }>;
    };
    const v = parsed.kvs?.[0]?.value;
    return v === undefined ? undefined : Buffer.from(v, "base64").toString("utf8");
  }

  /**
   * Delete exactly one key (etcd v3 `/v3/kv/deleterange`). With
   * `range_end` omitted the DeleteRange request applies to only `key`
   * — the single-key form of the same RPC `deletePrefix` uses.
   */
  async delete(key: string): Promise<void> {
    const res = await harnessRequest(`${this.endpoint}/v3/kv/deleterange`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ key: Buffer.from(key, "utf8").toString("base64") }),
    });
    if (res.statusCode >= 300) {
      const body = await res.body.text();
      throw new Error(`etcd deleterange failed (${res.statusCode}): ${body}`);
    }
  }

  /** Delete every key under `prefix` (range delete in etcd v3 semantics). */
  async deletePrefix(prefix: string): Promise<void> {
    const key = Buffer.from(prefix, "utf8").toString("base64");
    const rangeEnd = prefixRangeEnd(prefix).toString("base64");
    const res = await harnessRequest(`${this.endpoint}/v3/kv/deleterange`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ key, range_end: rangeEnd }),
    });
    if (res.statusCode >= 300) {
      const body = await res.body.text();
      throw new Error(`etcd deleterange failed (${res.statusCode}): ${body}`);
    }
  }
}

/**
 * Calculate the etcd "range end" for a prefix scan: the prefix with its
 * last byte incremented by one. Returned as a Buffer because the
 * incremented byte may not be valid UTF-8.
 */
function prefixRangeEnd(prefix: string): Buffer {
  const bytes = Array.from(Buffer.from(prefix, "utf8"));
  for (let i = bytes.length - 1; i >= 0; i--) {
    if (bytes[i] < 0xff) {
      const head = bytes.slice(0, i);
      return Buffer.from([...head, bytes[i] + 1]);
    }
  }
  return Buffer.from([0]);
}
