import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import { gunzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a passthrough route reports EVERY token dimension the exchange
// carried, not just the canonical prompt/completion pair.
//
// A route that auto-detects a first-class envelope must meter what the
// typed endpoint serving that envelope would: `UsageEvent` has seven token
// fields and cp-api prices the cache ones at their own rates (cache-write
// ~1.25x prompt, cache-read ~0.10x), so dropping them makes a cached
// workload — the dominant shape for IDE agent traffic, where the cache read
// can be 98% of the prompt — unpriceable and invisible.
//
// Three journeys, one per usage shape a route actually meets:
//
//   1. OPAQUE stream, server-labelled usage frame. An IDE agent backend
//      reached through a forward-proxy route has no recognisable envelope
//      and reports its counts as a FLAT token object on its own
//      `event: token_usage` frame — no `usage` wrapper. Read there, and
//      only there.
//   2. OPAQUE stream, unlabelled frame. The same flat shape on a frame the
//      server did not name a usage report mints nothing: an opaque stream
//      offers no envelope to authenticate token-shaped fields against.
//   3. Detected chat envelope, buffered. The nested OpenAI details
//      (`prompt_tokens_details.cached_tokens`,
//      `completion_tokens_details.reasoning_tokens`) reach the event, and
//      the caller's `model` alias attributes the row.
//
// Asserted through a real `datadog` exporter, because that sink serialises
// the WHOLE `UsageEvent` — the OTLP span builder is an explicit allowlist
// that carries only the input/output pair, so it cannot see the dimensions
// under test.

const CALLER_PLAINTEXT = "sk-ptr-usage-dims-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const CREDENTIAL_REF = "ptrdims";
const DD_API_KEY = "dd-ptr-usage-dims-key";
const INTAKE_PATH = "/api/v2/logs";

interface MockDatadog {
  site: string;
  logs: Record<string, unknown>[];
  close(): Promise<void>;
}

/** Mock Datadog Logs intake: collects every delivered log object. */
async function startMockDatadog(): Promise<MockDatadog> {
  const logs: Record<string, unknown>[] = [];
  const server: Server = createServer(
    (req: IncomingMessage, res: import("node:http").ServerResponse) => {
      const chunks: Buffer[] = [];
      req.on("data", (c: Buffer) => chunks.push(c));
      req.on("end", () => {
        const path = (req.url ?? "").split("?")[0];
        if (req.method === "POST" && path === INTAKE_PATH) {
          try {
            const parsed: unknown = JSON.parse(
              gunzipSync(Buffer.concat(chunks)).toString("utf8"),
            );
            if (Array.isArray(parsed)) {
              for (const entry of parsed) {
                if (entry && typeof entry === "object") {
                  logs.push(entry as Record<string, unknown>);
                }
              }
            }
          } catch {
            // A body we cannot decode leaves `logs` short, so the poll
            // below times out loudly instead of passing on nothing.
          }
        }
        res.statusCode = 202;
        res.end();
      });
    },
  );
  const port = await pickFreePort();
  await new Promise<void>((resolve) =>
    server.listen(port, "127.0.0.1", resolve),
  );
  return {
    site: `127.0.0.1:${port}`,
    logs,
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      }),
  };
}

describe("passthrough-route usage event: token dimensions and the caller's wait", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let dd: MockDatadog | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  // The agent-backend shape observed on a real forward-proxied IDE: a flat
  // token object on the server's own `token_usage` event, cache-read
  // dominating the prompt.
  // The upstream holds the first frame this long, then spaces the
  // remaining four out — so the whole stream runs far longer than the wait
  // the caller actually experienced.
  const FIRST_FRAME_MS = 120;
  const TAIL_STEP_MS = 400;

  const AGENT_USAGE = {
    prompt_tokens: 14603,
    completion_tokens: 8,
    cache_creation_input_tokens: 331,
    cache_read_input_tokens: 14272,
    reasoning_tokens: 6,
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    dd = await startMockDatadog();
    app = await spawnApp({
      extraEnv: {
        [`DD_CRED_${CREDENTIAL_REF.toUpperCase()}_API_KEY`]: DD_API_KEY,
      },
    });
    seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.createObservabilityExporter({
      name: "ptr-dims-datadog",
      enabled: true,
      kind: "datadog",
      site: dd.site,
      credential_ref: CREDENTIAL_REF,
      service: "aisix-ptr-dims",
      content_mode: "metadata_only",
    });

    // 1. Opaque stream that labels its usage frame.
    const labelled = await startOpenAiUpstream({
      rawStreamFrames: [
        `event:thought\ndata:${JSON.stringify({ thought: "hm" })}\n\n`,
        `event:output\ndata:${JSON.stringify({ text: "hi" })}\n\n`,
        `event:token_usage\ndata:${JSON.stringify({ name: "", ...AGENT_USAGE })}\n\n`,
      ],
    });
    // 2. The same flat shape on a frame the server never named usage.
    const unlabelled = await startOpenAiUpstream({
      rawStreamFrames: [
        `event:history\ndata:${JSON.stringify({ ...AGENT_USAGE })}\n\n`,
        `event:done\ndata:${JSON.stringify({ ok: true })}\n\n`,
      ],
    });
    // 3. Detected chat envelope, buffered, nested OpenAI usage details.
    const chat = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-dims",
        object: "chat.completion",
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "ok" },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: 900,
          completion_tokens: 25,
          total_tokens: 925,
          prompt_tokens_details: { cached_tokens: 768 },
          completion_tokens_details: { reasoning_tokens: 17 },
        },
      },
    });
    // 4. A stream whose frames are far apart: the caller's wait ends at the
    //    FIRST frame, but the exchange runs for the whole stream.
    const slow = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({ choices: [{ delta: { content: "a" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "b" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "c" } }] }),
        JSON.stringify({ choices: [{ delta: { content: "d" } }] }),
        JSON.stringify({
          choices: [],
          usage: { prompt_tokens: 3, completion_tokens: 4 },
        }),
      ],
      // A real (measurable) wait before the first frame, then a long tail.
      firstEventDelayMs: FIRST_FRAME_MS,
      eventDelayMs: TAIL_STEP_MS,
    });
    upstreams.push(labelled, unlabelled, chat, slow);

    const pk = await seed.createProviderKey({
      display_name: "ptr-dims-pk",
      secret: "sk-mock",
      api_base: "http://unused-on-routes",
    });
    for (const [name, prefix, upstream] of [
      ["ptr-dims-labelled", "/dims-labelled", labelled],
      ["ptr-dims-unlabelled", "/dims-unlabelled", unlabelled],
      ["ptr-dims-chat", "/dims-chat", chat],
      ["ptr-dims-slow", "/dims-slow", slow],
    ] as const) {
      await seed.createPassthroughRoute({
        name,
        path_prefix: prefix,
        target_url: upstream.baseUrl,
        provider_key_id: pk.id,
      });
    }

    // The caller key is seeded LAST, so gating on it authenticating implies
    // every resource above is already in the snapshot.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
      allowed_routes: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    await dd?.close();
  });

  const post = (path: string, body: unknown) =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });

  /** Poll the intake for the usage log of one route. */
  async function logFor(
    route: string,
    timeoutMs = 15_000,
  ): Promise<Record<string, unknown>> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const hit = dd!.logs.find(
        (l) => l["aisix.passthrough_route_name"] === route,
      );
      if (hit) return hit;
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error(`no usage log delivered for route ${route}`);
  }

  test("every token dimension a passthrough exchange reports reaches the usage event", async (ctx) => {
    if (!etcdReachable || !app || !seed || !dd) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await r.text();
      return r.status === 200;
    });

    // 1. Opaque agent stream: private request schema (no `messages` /
    //    `input` / `prompt`), usage on the server's own labelled frame.
    const labelledRes = await post(
      "/dims-labelled/api/agent/v3/create_agent_task",
      {
        config_name: "agent",
        payload: "opaque",
      },
    );
    expect(labelledRes.status).toBe(200);
    await labelledRes.text();

    const labelled = await logFor("ptr-dims-labelled");
    // Pre-fix this row carried 0/0: `usage_of` read only the canonical
    // pair, and only from a `usage`-wrapped object.
    expect(labelled["gen_ai.usage.input_tokens"]).toBe(
      AGENT_USAGE.prompt_tokens,
    );
    expect(labelled["gen_ai.usage.output_tokens"]).toBe(
      AGENT_USAGE.completion_tokens,
    );
    expect(labelled["aisix.cache_creation_tokens"]).toBe(
      AGENT_USAGE.cache_creation_input_tokens,
    );
    expect(labelled["aisix.cache_read_tokens"]).toBe(
      AGENT_USAGE.cache_read_input_tokens,
    );
    expect(labelled["aisix.reasoning_tokens"]).toBe(
      AGENT_USAGE.reasoning_tokens,
    );
    // An opaque body's `config_name` is not a model alias, so the row
    // claims no model identity.
    expect(labelled["aisix.requested_model"] ?? "").toBe("");

    // 2. The identical token object on an unlabelled frame mints nothing —
    //    the no-phantom-tokens guarantee, now covering opaque STREAMS too.
    const unlabelledRes = await post("/dims-unlabelled/api/agent/v3/history", {
      payload: "opaque",
    });
    expect(unlabelledRes.status).toBe(200);
    await unlabelledRes.text();

    const unlabelled = await logFor("ptr-dims-unlabelled");
    expect(unlabelled["gen_ai.usage.input_tokens"]).toBe(0);
    expect(unlabelled["gen_ai.usage.output_tokens"]).toBe(0);
    expect(unlabelled["aisix.cache_read_tokens"] ?? 0).toBe(0);
    expect(unlabelled["aisix.cache_creation_tokens"] ?? 0).toBe(0);
    expect(unlabelled["aisix.reasoning_tokens"] ?? 0).toBe(0);

    // 3. Detected chat envelope: OpenAI's NESTED cache/reasoning details
    //    reach the event, and the caller's alias attributes the row.
    const chatRes = await post("/dims-chat/v1/chat/completions", {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(chatRes.status).toBe(200);
    await chatRes.text();

    const chat = await logFor("ptr-dims-chat");
    expect(chat["gen_ai.usage.input_tokens"]).toBe(900);
    expect(chat["gen_ai.usage.output_tokens"]).toBe(25);
    expect(chat["aisix.cached_prompt_tokens"]).toBe(768);
    expect(chat["aisix.reasoning_tokens"]).toBe(17);
    expect(chat["aisix.requested_model"]).toBe("gpt-4o-mini");
  });

  test("a streamed relay reports the caller's wait to the first frame, not the whole stream", async (ctx) => {
    if (!etcdReachable || !app || !seed || !dd) {
      ctx.skip();
      return;
    }

    // `downstream_latency_ms` means "what the caller waited for before it
    // got something it can use": the complete body when buffered, the first
    // token when streamed. `/a2a` is the ONE documented exception, because
    // an agent's stream of task updates is the call's product — a relay is
    // a delivery mechanism for a response, so it follows the normal rule.
    // Pre-fix a route stamped BOTH `upstream_latency_ms` and
    // `downstream_latency_ms` with the whole-request elapsed at
    // end-of-stream, so a long stream reported the caller as having waited
    // for all of it.
    const started = Date.now();
    const res = await post("/dims-slow/v1/chat/completions", {
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    expect(res.status).toBe(200);
    await res.text();
    const wallMs = Date.now() - started;

    // The exchange really did run for the whole spaced-out stream, so a
    // whole-stream figure and a first-frame figure are far apart here.
    const tailMs = TAIL_STEP_MS * 4;
    expect(wallMs).toBeGreaterThan(tailMs);

    const slow = await logFor("ptr-dims-slow");
    const downstream = Number(slow["aisix.downstream_latency_ms"] ?? 0);
    const upstreamTotal = Number(slow["aisix.upstream_latency_ms"] ?? 0);
    const ttft = Number(slow["aisix.upstream_ttft_ms"] ?? 0);

    // The caller's wait is the first frame's arrival — bounded well below
    // the tail the stream went on to spend.
    expect(downstream).toBeGreaterThanOrEqual(FIRST_FRAME_MS / 2);
    expect(downstream).toBeLessThan(tailMs);
    // ...while the attempt itself ran for the whole stream.
    expect(upstreamTotal).toBeGreaterThan(tailMs);

    // TTFT is attempt-scoped and the caller's wait is request-scoped over
    // the same first frame, so gateway-side work is exactly their
    // difference — never negative.
    expect(ttft).toBeGreaterThanOrEqual(FIRST_FRAME_MS / 2);
    expect(ttft).toBeLessThanOrEqual(downstream);
  });
});
