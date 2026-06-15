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

// E2E (#655): the DP emits one UsageEvent per upstream attempt — the
// initial try, each retry, and each fallback — all sharing the request's
// `request_id` (the trace/group key). A Model Group whose primary fails
// and secondary succeeds must therefore emit TWO telemetry records: a
// failed `initial` attempt on the primary and a successful `fallback`
// attempt on the secondary.
//
// Usage telemetry has no cp-api receiver in DP e2e, so we observe the
// emitted field VALUES through the per-env OTLP/HTTP fan-out: register a
// mock OTLP receiver as an `observability_exporter`, drive one failover
// request, and assert two spans carrying the per-attempt attributes
// (`aisix.attempt_index` / `aisix.attempt_kind` / `aisix.attempt_model`
// / `aisix.error_class`) share the same `aisix.request_id`.

const CALLER_PLAINTEXT = "sk-per-attempt-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

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

async function waitForAttempts(
  recv: OtlpReceiver,
  requestId: string,
  count: number,
  timeoutMs = 10_000,
): Promise<Array<Record<string, string>>> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const hits = recv.spanAttrs.filter(
      (a) => a["aisix.request_id"] === requestId,
    );
    if (hits.length >= count) return hits;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(
    `expected ${count} attempt spans for request_id=${requestId}, ` +
      `saw ${recv.spanAttrs.filter((a) => a["aisix.request_id"] === requestId).length}`,
  );
}

describe("per-attempt telemetry e2e (#655): one UsageEvent per upstream attempt", () => {
  let etcdReachable = false;
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let otlp: OtlpReceiver | undefined;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    otlp = await startOtlpReceiver();
    await admin.createObservabilityExporter({
      name: "per-attempt-otlp",
      enabled: true,
      kind: "otlp_http",
      endpoint: otlp.url,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await otlp?.close();
  });

  async function createOpenAiModel(
    displayName: string,
    upstream: OpenAiUpstream,
  ): Promise<void> {
    if (!admin) throw new Error("admin client not initialized");
    const providerKey = await admin.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: providerKey.id,
    });
  }

  test("a failover request emits a failed initial + successful fallback attempt sharing request_id", async (ctx) => {
    if (!etcdReachable || !app || !admin || !otlp) {
      ctx.skip();
      return;
    }

    const primary = await startOpenAiUpstream({
      status: 502,
      errorBody: { error: { message: "primary down", type: "server_error" } },
    });
    const secondary = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-per-attempt-fallback",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "fallback worked" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
      },
    });
    upstreams.push(primary, secondary);

    await createOpenAiModel("attempt-primary", primary);
    await createOpenAiModel("attempt-secondary", secondary);
    await admin.createModel({
      display_name: "attempt-virtual",
      routing: {
        strategy: "failover",
        targets: [{ model: "attempt-primary" }, { model: "attempt-secondary" }],
        retries: 0,
        max_fallbacks: 1,
      },
    });

    // Gate on admin-snapshot presence rather than probing the virtual —
    // probing would warm the primary's cooldown (every retryable upstream
    // failure cools the failing direct target) and the measured request
    // would then skip the primary entirely.
    await waitConfigPropagation(async () => {
      try {
        const models = await admin!.listModels();
        return models.some((m) => m.display_name === "attempt-virtual");
      } catch {
        return false;
      }
    });

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "attempt-virtual",
        messages: [{ role: "user", content: "fail over please" }],
      }),
    });
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-call-id");
    expect(requestId).toBeTruthy();
    await res.text();

    // Exactly the primary (initial) then the secondary (fallback).
    expect(primary.receivedRequests.length).toBe(1);
    expect(secondary.receivedRequests.length).toBe(1);

    const attempts = await waitForAttempts(otlp, requestId!, 2);
    attempts.sort(
      (a, b) =>
        Number(a["aisix.attempt_index"]) - Number(b["aisix.attempt_index"]),
    );

    // Attempt 0: failed initial try on the primary.
    expect(attempts[0]["aisix.attempt_index"]).toBe("0");
    expect(attempts[0]["aisix.attempt_kind"]).toBe("initial");
    expect(attempts[0]["aisix.attempt_model"]).toBe("attempt-primary");
    expect(attempts[0]["aisix.error_class"]).toBe("upstream_status");
    expect(attempts[0]["http.response.status_code"]).toBe("502");
    expect(attempts[0]["gen_ai.usage.input_tokens"]).toBe("0");

    // Attempt 1: successful fallback on the secondary with real tokens.
    expect(attempts[1]["aisix.attempt_index"]).toBe("1");
    expect(attempts[1]["aisix.attempt_kind"]).toBe("fallback");
    expect(attempts[1]["aisix.attempt_model"]).toBe("attempt-secondary");
    expect(attempts[1]["aisix.error_class"]).toBeUndefined();
    expect(attempts[1]["http.response.status_code"]).toBe("200");
    expect(attempts[1]["gen_ai.usage.input_tokens"]).toBe("3");
    expect(attempts[1]["gen_ai.usage.output_tokens"]).toBe("4");
  });
});
