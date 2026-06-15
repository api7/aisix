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

// E2E (#519 B.2): per-exporter `sample_rate` + content capture for
// kind=otlp_http.
//
//   1. Defaults unchanged — an exporter with neither knob ships spans
//      for every request and never carries prompt/response content.
//   2. `content_mode: full` attaches `gen_ai.prompt` / `gen_ai.completion`
//      truncated to `content_max_bytes` (with `aisix.content_truncated`).
//   3. `sample_rate: 0` ships nothing; the same traffic against a
//      rate-1.0 exporter (positive control) ships spans — proving the
//      absence was the sampler, not a broken pipeline.

const CALLER_PLAINTEXT = "sk-otlp-knobs-caller-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const PROVIDER_SECRET = "sk-mock-otlp-knobs";

interface OtlpReceiver {
  url: string;
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
    display_name: "otlp-knobs-pk",
    secret: PROVIDER_SECRET,
    api_base: `${upstream.baseUrl}/v1`,
  });
  await admin.createModel({
    display_name: "otlp-knobs-model",
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await admin.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: ["otlp-knobs-model"],
  });
}

async function chat(app: SpawnedApp, content: string) {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "otlp-knobs-model",
      messages: [{ role: "user", content }],
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
  throw new Error(
    `no matching OTLP span recorded within timeout; ${recv.spanAttrs.length} spans seen; ` +
      `sample keys: ${JSON.stringify(recv.spanAttrs.slice(0, 2).map((a) => Object.keys(a)))}; ` +
      `prompts: ${JSON.stringify(recv.spanAttrs.slice(0, 3).map((a) => a["gen_ai.prompt"]))}`,
  );
}

describe("otlp exporter knobs e2e (#519 B.2): sample_rate + content capture", () => {
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  const apps: SpawnedApp[] = [];
  const receivers: OtlpReceiver[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
  });

  afterAll(async () => {
    await Promise.all(apps.map((a) => a.exit()));
    await Promise.all(receivers.map((r) => r.close()));
    await upstream?.close();
  });

  test(
    "defaults unchanged: knobless exporter ships every span, no content",
    async (ctx) => {
      if (!etcdReachable || !upstream) {
        ctx.skip();
        return;
      }
      const otlp = await startOtlpReceiver();
      receivers.push(otlp);
      const app = await spawnApp();
      apps.push(app);
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      await admin.createObservabilityExporter({
        name: "otlp-defaults",
        enabled: true,
        kind: "otlp_http",
        endpoint: otlp.url,
      });
      await seedRouting(admin, upstream);
      await waitConfigPropagation(async () => {
        try {
          const r = await chat(app, "default-probe");
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      const span = await waitForSpan(otlp, (a) => "aisix.request_id" in a);
      expect(span["gen_ai.prompt"]).toBeUndefined();
      expect(span["gen_ai.completion"]).toBeUndefined();
    },
    60_000,
  );

  test(
    "content_mode=full attaches prompt/completion, truncated to content_max_bytes",
    async (ctx) => {
      if (!etcdReachable || !upstream) {
        ctx.skip();
        return;
      }
      const otlp = await startOtlpReceiver();
      receivers.push(otlp);
      const app = await spawnApp();
      apps.push(app);
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      await admin.createObservabilityExporter({
        name: "otlp-full",
        enabled: true,
        kind: "otlp_http",
        endpoint: otlp.url,
        content_mode: "full",
        // The captured prompt is the serialized request JSON (model +
        // messages wrapper ≈ 60 bytes before the user text), so the cap
        // must leave room for the probe to stay visible untruncated.
        content_max_bytes: 200,
      });
      await seedRouting(admin, upstream);
      await waitConfigPropagation(async () => {
        try {
          const r = await chat(app, "content-probe");
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      // Short prompt: captured whole, no truncation flag.
      const shortSpan = await waitForSpan(
        otlp,
        (a) => (a["gen_ai.prompt"] ?? "").includes("content-probe"),
      );
      expect(shortSpan["gen_ai.completion"]).toBeTruthy();
      expect(shortSpan["aisix.content_truncated"]).toBeUndefined();

      // Oversized prompt: captured but cut at the 200-byte cap.
      const long = "long-content-".repeat(40); // ~520 bytes
      const res = await chat(app, long);
      expect(res.status).toBe(200);
      await res.text();
      const longSpan = await waitForSpan(
        otlp,
        (a) =>
          (a["gen_ai.prompt"] ?? "").includes("long-content-") &&
          a["aisix.content_truncated"] === "true",
      );
      expect(longSpan["gen_ai.prompt"].length).toBeLessThanOrEqual(200);
    },
    60_000,
  );

  test(
    "sample_rate=0 ships nothing while a rate-1.0 exporter ships the same traffic",
    async (ctx) => {
      if (!etcdReachable || !upstream) {
        ctx.skip();
        return;
      }
      const otlpZero = await startOtlpReceiver();
      const otlpAll = await startOtlpReceiver();
      receivers.push(otlpZero, otlpAll);
      const app = await spawnApp();
      apps.push(app);
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      await admin.createObservabilityExporter({
        name: "otlp-sample-zero",
        enabled: true,
        kind: "otlp_http",
        endpoint: otlpZero.url,
        sample_rate: 0,
      });
      await admin.createObservabilityExporter({
        name: "otlp-sample-all",
        enabled: true,
        kind: "otlp_http",
        endpoint: otlpAll.url,
        sample_rate: 1.0,
      });
      await seedRouting(admin, upstream);
      await waitConfigPropagation(async () => {
        try {
          const r = await chat(app, "sampling-probe");
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      for (let i = 0; i < 5; i++) {
        const r = await chat(app, `sampling-probe-${i}`);
        expect(r.status).toBe(200);
        await r.text();
      }

      // Positive control first: the rate-1.0 exporter saw the traffic …
      await waitForSpan(otlpAll, (a) => "aisix.request_id" in a, 15_000);
      // … so the rate-0 silence is the sampler, not a dead pipeline.
      expect(otlpZero.spanAttrs).toHaveLength(0);
    },
    60_000,
  );
});
