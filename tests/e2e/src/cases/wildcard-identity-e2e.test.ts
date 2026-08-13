import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  awaitWindowHeadroom,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a wildcard alias row (`wid/*`) is ONE configured identity, so its
// gates and telemetry must key on the row, not on whatever concrete
// suffix a caller mints (model-kind audit):
//
//   1. The inline rate_limit bucket is shared across every alias the row
//      serves — previously each distinct suffix opened a fresh
//      full-size bucket, letting any caller multiply the declared cap
//      without bound.
//   2. The Prometheus `model` label for wildcard-served SUCCESS traffic
//      is the row's display_name; caller-minted strings must not mint
//      unbounded series (the #451 cardinality guard, extended to
//      resolvable names).

const CALLER_PLAINTEXT = "sk-wildcard-identity-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function chatBody(content: string) {
  return {
    id: `cmpl-${content}`,
    object: "chat.completion",
    created: 0,
    model: "gpt-4o-mini",
    choices: [
      { index: 0, message: { role: "assistant", content }, finish_reason: "stop" },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

describe("wildcard alias identity e2e", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  async function chat(model: string): Promise<{ status: number; type?: string }> {
    if (!app) throw new Error("app not ready");
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }] }),
    });
    const body = (await res.json()) as { error?: { type?: string } };
    return { status: res.status, type: body.error?.type };
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const upstream = await startOpenAiUpstream({ nonStreamBody: chatBody("served-wid") });
    upstreams.push(upstream);
    const pk = await seed.createProviderKey({
      display_name: "wid-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "wid/*",
      provider: "openai",
      model_name: "*",
      provider_key_id: pk.id,
      rate_limit: { rpm: 1 },
    });
    // A second wildcard row with an SSE fixture (the mock picks SSE vs
    // JSON per FIXTURE): drives the streaming-only TTFT/summary series
    // so the dump-wide assertions cover that family too.
    const sse = await startOpenAiUpstream({
      eventDelayMs: 2,
      streamEvents: [
        JSON.stringify({
          id: "wid-sse",
          object: "chat.completion.chunk",
          model: "gpt-4o-mini",
          choices: [
            { index: 0, delta: { content: "served-wid2" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
        "[DONE]",
      ],
    });
    upstreams.push(sse);
    const pk2 = await seed.createProviderKey({
      display_name: "wid2-pk",
      secret: "sk-mock",
      api_base: `${sse.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "wid2/*",
      provider: "openai",
      model_name: "*",
      provider_key_id: pk2.id,
    });

    // The caller key is seeded LAST: once it authenticates, revision
    // order implies both wildcard rows above are in the snapshot
    // (tests/e2e/AGENTS.md). The gate neither resolves a wildcard alias
    // nor consumes the shared rpm bucket under test.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      await res.arrayBuffer(); // release the socket between polls
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("every caller-minted alias shares the wildcard row's rate-limit bucket, and success metrics label as the row", { timeout: 150_000 }, async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await awaitWindowHeadroom(5);

    // Align on a window that admits `wid/alpha` — that 200 is alias
    // #1's slot in the SHARED bucket (rpm=1, fixed windows keyed on
    // unix time, so the first attempt may land in a spent window).
    const deadline = Date.now() + 90_000;
    let aligned = false;
    while (Date.now() < deadline) {
      const r = await chat("wid/alpha");
      if (r.status === 200) {
        aligned = true;
        break;
      }
      expect(r.status).toBe(429);
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    expect(aligned).toBe(true);

    // A DIFFERENT alias in the same window must land in the same
    // bucket. Pre-fix each suffix opened its own rpm=1 bucket and this
    // passed with 200.
    const second = await chat("wid/beta");
    expect(second.status).toBe(429);
    expect(second.type).toBe("rate_limit_exceeded");

    // Drive a STREAMING request too (fresh window; its own alias): the
    // TTFT/summary families only emit on streamed completions, and the
    // dump-wide assertion below must cover them.
    const streamRes = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "wid2/gamma",
        messages: [{ role: "user", content: "hi" }],
        stream: true,
      }),
    });
    const streamBody = await streamRes.text();
    expect(streamRes.status).toBe(200);
    expect(streamRes.headers.get("content-type")).toContain("text/event-stream");
    expect(streamBody).toContain("served-wid2");

    // Success + throttle series both label as the configured row; the
    // caller-minted suffixes never become label values.
    const metrics = await (await fetch(`${app.metricsUrl}/metrics`)).text();
    expect(metrics).toContain('model="wid/*"');
    expect(metrics).not.toContain('model="wid/alpha"');
    expect(metrics).not.toContain('model="wid/beta"');
    expect(metrics).not.toContain('model="wid2/gamma"');
    expect(metrics).toContain('model="wid2/*"');
    // The upstream_model label is caller-derived on a wildcard hit too
    // (the capture substitutes into the template) — it must collapse to
    // the row's configured template, never the minted suffix.
    expect(metrics).not.toContain('upstream_model="alpha"');
    expect(metrics).not.toContain('upstream_model="beta"');
    expect(metrics).not.toContain('upstream_model="gamma"');
  });
});
