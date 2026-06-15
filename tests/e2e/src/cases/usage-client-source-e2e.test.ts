import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E (#492): the proxy logs the downstream client's source IP + client
// type (User-Agent) on every usage event.
//
// Usage telemetry has no cp-api receiver in DP e2e, so we observe the
// emitted field VALUES through the per-env OTLP/HTTP fan-out: register a
// mock OTLP receiver as an `observability_exporter`, drive one chat
// request, and assert the recorded span carries the two new custom
// attributes `aisix.client_source_ip` / `aisix.client_user_agent`.
//
// IP resolution mirrors nginx set_real_ip_from + real_ip_recursive:
//   - peer 127.0.0.1 is a trusted proxy (the e2e client is loopback),
//   - 10.0.0.0/8 is trusted too,
//   - so walking `x-forwarded-for: 203.0.113.7, 10.0.0.1` right-to-left
//     skips 10.0.0.1 (trusted) and yields 203.0.113.7 as the client.
// The negative case (no trusted_proxies) proves XFF is ignored and the
// TCP peer (127.0.0.1) is logged instead.

const CALLER_PLAINTEXT = "sk-client-source-caller-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const PROVIDER_SECRET = "sk-mock-client-source";
const PROBE_UA = "aisix-e2e-probe/1.0";

interface OtlpReceiver {
  url: string;
  /** All span attribute maps recorded across every posted batch. */
  spanAttrs: Array<Record<string, string>>;
  close(): Promise<void>;
}

async function startOtlpReceiver(): Promise<OtlpReceiver> {
  const spanAttrs: Array<Record<string, string>> = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      try {
        const body = JSON.parse(raw);
        for (const rs of body.resourceSpans ?? []) {
          for (const ss of rs.scopeSpans ?? []) {
            for (const span of ss.spans ?? []) {
              const attrs: Record<string, string> = {};
              for (const a of span.attributes ?? []) {
                const v = a.value ?? {};
                attrs[a.key] =
                  v.stringValue ?? String(v.intValue ?? v.boolValue ?? "");
              }
              spanAttrs.push(attrs);
            }
          }
        }
      } catch {
        // ignore malformed bodies — assertions fail on missing spans
      }
      res.statusCode = 200;
      res.end("{}");
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}/v1/traces`,
    spanAttrs,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

async function seedRouting(admin: AdminClient, upstream: OpenAiUpstream) {
  const pk = await admin.createProviderKey({
    display_name: "client-source-pk",
    secret: PROVIDER_SECRET,
    api_base: `${upstream.baseUrl}/v1`,
  });
  await admin.createModel({
    display_name: "client-source-model",
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await admin.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: ["client-source-model"],
  });
}

async function chat(app: SpawnedApp, headers: Record<string, string>) {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
      ...headers,
    },
    body: JSON.stringify({
      model: "client-source-model",
      messages: [{ role: "user", content: "hello" }],
    }),
  });
}

async function waitForSpan(
  recv: OtlpReceiver,
  predicate: (attrs: Record<string, string>) => boolean,
  timeoutMs = 10_000,
): Promise<Record<string, string>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hit = recv.spanAttrs.find(predicate);
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error("no matching OTLP span recorded within timeout");
}

describe("usage client source e2e (#492): source IP + User-Agent on usage events", () => {
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let otlp: OtlpReceiver | undefined;
  const apps: SpawnedApp[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
    otlp = await startOtlpReceiver();
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await upstream?.close();
    await otlp?.close();
  });

  test(
    "trusted-proxy XFF resolves the real client IP and records the User-Agent",
    async (ctx) => {
      if (!etcdReachable || !upstream || !otlp) {
        ctx.skip();
        return;
      }
      const app = await spawnApp({
        realIp: { trusted_proxies: ["127.0.0.1/32", "10.0.0.0/8"], recursive: true },
      });
      apps.push(app);
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      await admin.createObservabilityExporter({
        name: "mock-otlp",
        enabled: true,
        kind: "otlp_http",
        endpoint: otlp.url,
      });
      await seedRouting(admin, upstream);

      await waitConfigPropagation(async () => {
        try {
          const r = await chat(app, {});
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      const res = await chat(app, {
        "x-forwarded-for": "203.0.113.7, 10.0.0.1",
        "user-agent": PROBE_UA,
      });
      expect(res.status).toBe(200);
      await res.text();

      const span = await waitForSpan(otlp, (a) => {
        return (
          a["aisix.client_user_agent"] === PROBE_UA &&
          a["aisix.client_source_ip"] === "203.0.113.7"
        );
      });
      expect(span["aisix.client_source_ip"]).toBe("203.0.113.7");
      expect(span["aisix.client_user_agent"]).toBe(PROBE_UA);
    },
    60_000,
  );

  test(
    "without trusted proxies, XFF is ignored and the TCP peer is logged",
    async (ctx) => {
      if (!etcdReachable || !upstream || !otlp) {
        ctx.skip();
        return;
      }
      const app = await spawnApp(); // no real_ip → trust nothing
      apps.push(app);
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      await admin.createObservabilityExporter({
        name: "mock-otlp-2",
        enabled: true,
        kind: "otlp_http",
        endpoint: otlp.url,
      });
      await seedRouting(admin, upstream);

      const marker = "aisix-e2e-peer-probe/9.9";
      await waitConfigPropagation(async () => {
        try {
          const r = await chat(app, {});
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      const res = await chat(app, {
        "x-forwarded-for": "203.0.113.7",
        "user-agent": marker,
      });
      expect(res.status).toBe(200);
      await res.text();

      const span = await waitForSpan(
        otlp,
        (a) => a["aisix.client_user_agent"] === marker,
      );
      // Peer is loopback; XFF must be ignored because 127.0.0.1 is not
      // a configured trusted proxy.
      expect(span["aisix.client_source_ip"]).toBe("127.0.0.1");
    },
    60_000,
  );
});
