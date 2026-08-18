import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1317: `aisix_usage_events_emitted_total` carried only
// handler / status_code / inbound_protocol, and
// `aisix_usage_event_drops_total` only `reason`. So an environment with
// several models — or one provider fronted by several ProviderKeys —
// could not tell which of them was still producing usage telemetry, and
// could not tell whose records a drop had lost: `emitted == delivered +
// dropped` only held after summing every dimension away.
//
// The drops half is directly observable here: a standalone gateway wires
// no CP sink, so every emit is also a `sink_disabled` drop. That is the
// scenario the labels exist for — the same request, counted on both
// sides, has to name the same model and key.
//
// The same issue asked for `aisix_proxy_client_cancelled_requests_total`,
// which had `endpoint` alone: its whole purpose is answering "which model
// do callers give up waiting on", and it could not.
const CALLER_PLAINTEXT = "sk-usage-attr-1317";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const PK_NAME = "usage1317-pk";
const EMBED_PK_NAME = "usage1317-embed-pk";
const SLOW_PK_NAME = "usage1317-slow-pk";
const MODEL = "usage1317-chat";
const EMBED_MODEL = "usage1317-embed";
const SLOW_MODEL = "usage1317-slow";

describe("usage-event and cancel attribution #1317 e2e", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let slowUpstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const fast = await startOpenAiUpstream({ nonStreamBody: chatBody() });
    // Headers withheld long enough that the caller can hang up while the
    // gateway is still waiting — the head-phase cancel this counter exists
    // for.
    const slow = await startOpenAiUpstream({
      nonStreamBody: chatBody(),
      responseDelayMs: 10_000,
    });
    // The mock answers every route with one canned body, so the embeddings
    // family needs its own upstream to return an embeddings-shaped one.
    const embeddings = await startOpenAiUpstream({ nonStreamBody: embeddingBody() });
    upstreams.push(fast, slow, embeddings);
    slowUpstream = slow;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const fastPk = await seed.createProviderKey({
      display_name: PK_NAME,
      secret: "sk-mock",
      api_base: `${fast.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: fastPk.id,
    });
    const embedPk = await seed.createProviderKey({
      display_name: EMBED_PK_NAME,
      secret: "sk-mock",
      api_base: `${embeddings.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: EMBED_MODEL,
      provider: "openai",
      model_name: "text-embedding-3-small",
      provider_key_id: embedPk.id,
    });
    const slowPk = await seed.createProviderKey({
      display_name: SLOW_PK_NAME,
      secret: "sk-mock",
      api_base: `${slow.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: SLOW_MODEL,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: slowPk.id,
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL, SLOW_MODEL, EMBED_MODEL],
    });

    // The caller key is seeded last, so `GET /v1/models` answering 200 implies
    // every model and ProviderKey above is already in the snapshot. Gating on
    // a request that exercises the behavior under test would make a handler
    // regression surface as a propagation timeout (tests/e2e/AGENTS.md).
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      return (await proxy.listModels()).status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("emitted and dropped usage events name the same model and ProviderKey", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const res = await proxy.chat({
      model: MODEL,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(200);

    const text = await scrape(app);
    const emitted = sample(text, "aisix_usage_events_emitted_total", {
      handler: "chat",
      model: MODEL,
      provider_key_name: PK_NAME,
    });
    expect(emitted, `no attributed emit sample:\n${text}`).toBeTruthy();

    // The invariant, sliced: this gateway has no sink, so the same request
    // must show up on the drops counter under the same attribution.
    const dropped = sample(text, "aisix_usage_event_drops_total", {
      reason: "sink_disabled",
      model: MODEL,
      provider_key_name: PK_NAME,
    });
    expect(dropped, `no attributed drop sample:\n${text}`).toBeTruthy();

    // Both counters must carry the ProviderKey id too, and it must be the
    // same one the request families report for this model.
    const pkId = labelOf(emitted!, "provider_key_id");
    expect(pkId).not.toBe("unknown");
    expect(labelOf(dropped!, "provider_key_id")).toBe(pkId);
    const request = sample(text, "aisix_proxy_requests_total", { model: MODEL });
    expect(labelOf(request!, "provider_key_id")).toBe(pkId);
  });

  test("the emit labels are wired per handler, not just on chat", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Every handler calls `try_emit` itself, so a family member can land with
    // the attribution argument left at its placeholder.
    const res = await fetch(`${app.proxyUrl}/v1/embeddings`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model: EMBED_MODEL, input: "hi" }),
    });
    expect(res.status, await res.clone().text()).toBe(200);
    await res.text();

    const text = await scrape(app);
    const emitted = sample(text, "aisix_usage_events_emitted_total", {
      handler: "embeddings",
      model: EMBED_MODEL,
      provider_key_name: EMBED_PK_NAME,
    });
    expect(emitted, `no attributed embeddings emit sample:\n${text}`).toBeTruthy();
    const dropped = sample(text, "aisix_usage_event_drops_total", {
      reason: "sink_disabled",
      model: EMBED_MODEL,
      provider_key_name: EMBED_PK_NAME,
    });
    expect(dropped, `no attributed embeddings drop sample:\n${text}`).toBeTruthy();
  });

  test("a request with no resolvable model still carries a bounded placeholder", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Every sample in the family has to carry the same label set, or a
    // PromQL sum over it silently drops rows.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const res = await proxy.chat({
      model: "usage1317-no-such-model",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(404);

    const text = await scrape(app);
    for (const line of text.split("\n")) {
      if (
        line.startsWith("aisix_usage_events_emitted_total{") ||
        line.startsWith("aisix_usage_event_drops_total{")
      ) {
        for (const label of ["model", "provider_key_id", "provider_key_name"]) {
          expect(line, `${label} missing from ${line}`).toContain(`${label}="`);
        }
      }
    }
    const unresolved = sample(text, "aisix_usage_events_emitted_total", {
      handler: "chat",
      model: "unresolved",
      provider_key_id: "unknown",
    });
    expect(unresolved, `no placeholder emit sample:\n${text}`).toBeTruthy();
  });

  test("a cancelled request names the model and key it was waiting on", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const controller = new AbortController();
    const inflight = fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: SLOW_MODEL,
        messages: [{ role: "user", content: "hang up on me" }],
      }),
      signal: controller.signal,
    });
    // Abort once the upstream has actually received the call: by then the
    // gateway has authenticated, resolved the model and picked the target,
    // and the upstream is still 10s away from answering — so the response
    // head is unwritten when the caller goes away. A fixed delay could fire
    // before target selection on a slow machine and silently test nothing.
    for (let i = 0; i < 100 && slowUpstream!.receivedRequests.length === 0; i++) {
      await new Promise((r) => setTimeout(r, 50));
    }
    expect(
      slowUpstream!.receivedRequests.length,
      "the gateway never dispatched to the slow upstream",
    ).toBeGreaterThan(0);
    controller.abort();
    await expect(inflight).rejects.toThrow();

    // The guard fires from Drop, a beat after the socket closes.
    let cancelled: string | undefined;
    for (let i = 0; i < 60; i++) {
      const text = await scrape(app);
      cancelled = sample(text, "aisix_proxy_client_cancelled_requests_total", {
        endpoint: "/v1/chat/completions",
      });
      if (cancelled) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(cancelled, "no client-cancel sample was recorded").toBeTruthy();
    expect(cancelled).toContain(`model="${SLOW_MODEL}"`);
    expect(cancelled).toContain(`provider_key_name="${SLOW_PK_NAME}"`);
    expect(cancelled).not.toContain('provider_key_id="unknown"');
  }, 30_000);
});

async function scrape(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  return res.text();
}

function sample(
  scraped: string,
  metric: string,
  labels: Record<string, string>,
): string | undefined {
  return scraped
    .split("\n")
    .filter((l) => l.startsWith(`${metric}{`))
    .find((l) =>
      Object.entries(labels).every(([k, v]) => l.includes(`${k}="${v}"`)),
    );
}

function labelOf(line: string, label: string): string | undefined {
  return new RegExp(`${label}="([^"]*)"`).exec(line)?.[1];
}

function embeddingBody() {
  return {
    object: "list",
    model: "text-embedding-3-small",
    data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2, 0.3] }],
    usage: { prompt_tokens: 7, total_tokens: 7 },
  };
}

function chatBody() {
  return {
    id: "chatcmpl-usage-1317",
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: "hello" },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
  };
}
