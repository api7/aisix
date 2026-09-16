import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: what a response served from the cache is allowed to claim
// (AISIX-Cloud#1571).
//
// A cache hit contacts no upstream. Two things follow, and neither held
// before this suite existed.
//
// The line must SAY the cache served it. The access log carried no cache
// marker at all, so the only evidence of a hit was the absence of
// target-shaped fields — which is also what a request that failed before
// dispatch looks like. An operator reading by request id could not tell a
// Redis-served answer from one that never reached a provider.
//
// And it must not name a target it did not dispatch to. The pre-flight in
// `resolve_provider_key` runs for every entry that resolves to a single
// candidate, BEFORE the cache is consulted, so a direct model's hit line
// carried that model's mapping and a Model Group's did not — the reported
// asymmetry. The correction is not to fill the group in: which target
// produced the stored entry is recorded nowhere, and the candidate a
// request's strategy ranks first is not it. So the group reports no target,
// the ONE-target group (which does reach the pre-flight) reports none
// either, and the direct model keeps only what is a static property of its
// own row.
//
// Driven against the real binary because both halves are produced by
// different layers than the assertions read: the marker by the handler's
// own exit, the target fields by the request's attribution cell, and the
// usage-event attribution by a third path that reads neither.

const CALLER_PLAINTEXT = "sk-cache-hit-attr-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** Read one `name=value` field off a tracing text line, quoted or bare. */
function field(line: string, name: string): string | undefined {
  const m = line.match(new RegExp(`\\b${name}=(?:"([^"]*)"|([^\\s]+))`));
  if (!m) return undefined;
  return m[1] ?? m[2];
}

describe("cache-hit attribution e2e (AISIX-Cloud#1571)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let soloPkId = "";
  let targetAPkId = "";
  let singleModelId = "";

  /**
   * Poll the DP's captured output for this request's access-log line.
   * Delivery lags the HTTP response (the child's stderr is piped), and the
   * per-attempt `provider call completed` line shares the request id — so
   * the predicate pins the access log's own message too.
   */
  async function accessLine(requestId: string): Promise<string> {
    const deadline = Date.now() + 5_000;
    let last = "";
    while (Date.now() < deadline) {
      last = app!.output();
      const hit = last
        .split("\n")
        .find(
          (l) =>
            l.includes("proxy request completed") &&
            field(l, "request_id") === requestId,
        );
      if (hit) return hit;
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error(
      `timed out waiting for the access-log line of ${requestId}; DP output was:\n${last}`,
    );
  }

  async function chat(
    model: string,
    content: string,
  ): Promise<{ status: number; requestId: string; cache: string }> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model, messages: [{ role: "user", content }] }),
    });
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    const cache = res.headers.get("x-aisix-cache") ?? "";
    await res.text();
    return { status: res.status, requestId, cache };
  }

  /** Sum `aisix_usage_events_emitted_total` over one `provider_key_id`. */
  async function usageEventsFor(providerKeyId: string): Promise<number> {
    const res = await fetch(`${app!.metricsUrl}/metrics`);
    const body = await res.text();
    let total = 0;
    for (const line of body.split("\n")) {
      if (!line.startsWith("aisix_usage_events_emitted_total{")) continue;
      if (!line.includes(`provider_key_id="${providerKeyId}"`)) continue;
      total += Number(line.slice(line.lastIndexOf(" ") + 1)) || 0;
    }
    return total;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    // The access log is a `tracing::info!` event; the harness defaults the
    // gateway to `warn`.
    app = await spawnApp({ extraEnv: { RUST_LOG: "info" } });
    seed = new SeedClient(etcd, app.etcdPrefix);

    const soloPk = await seed.createProviderKey({
      display_name: "chit-solo-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    soloPkId = soloPk.id;
    await seed.createModel({
      display_name: "chit-solo",
      provider: "openai",
      model_name: "up-solo",
      provider_key_id: soloPk.id,
    });

    // Each target gets its OWN provider key and upstream model name, so a
    // line naming one names a target rather than merely some string.
    const pkA = await seed.createProviderKey({
      display_name: "chit-a-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    targetAPkId = pkA.id;
    await seed.createModel({
      display_name: "chit-target-a",
      provider: "openai",
      model_name: "up-model-a",
      provider_key_id: pkA.id,
    });
    const pkB = await seed.createProviderKey({
      display_name: "chit-b-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "chit-target-b",
      provider: "openai",
      model_name: "up-model-b",
      provider_key_id: pkB.id,
    });

    await seed.createModel({
      display_name: "chit-pair",
      routing: {
        strategy: "round_robin",
        targets: [{ model: "chit-target-a" }, { model: "chit-target-b" }],
      },
    });
    // A group with ONE target still reaches the single-candidate pre-flight,
    // which is what made its hit line name a target.
    singleModelId = (
      await seed.createModel({
        display_name: "chit-single",
        routing: {
          strategy: "round_robin",
          targets: [{ model: "chit-target-a" }],
        },
      })
    ).id;

    await seed.createCachePolicy({
      name: "chit-policy",
      enabled: true,
      applies_to: "all",
    });
    // Written LAST: watch events apply in revision order, so a request this
    // key authenticates proves every resource above is in the snapshot too.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(
      async () => (await proxy.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("a direct model's hit says the cache served it and keeps its own mapping", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const prompt = "direct-entry-hit";
    expect((await chat("chit-solo", prompt)).cache).toBe("miss");
    const hit = await chat("chit-solo", prompt);
    expect(hit.cache).toBe("hit");

    const line = await accessLine(hit.requestId);
    expect(field(line, "cache_status"), line).toBe("hit");
    expect(field(line, "cache_hit_layer"), line).toBe("exact");
    expect(field(line, "model"), line).toBe("chit-solo");
    // Kept because they are properties of the row the caller addressed —
    // true whether or not a request ever left the gateway — not because
    // anything was dispatched.
    expect(field(line, "upstream_model"), line).toBe("up-solo");
    expect(field(line, "provider_key_id"), line).toBe(soloPkId);
  });

  test("a Model Group's hit names no target", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const prompt = "group-entry-hit";
    expect((await chat("chit-pair", prompt)).cache).toBe("miss");
    const hit = await chat("chit-pair", prompt);
    expect(hit.cache).toBe("hit");

    const line = await accessLine(hit.requestId);
    expect(field(line, "cache_status"), line).toBe("hit");
    expect(field(line, "model"), line).toBe("chit-pair");
    expect(field(line, "upstream_model"), line).toBeUndefined();
    expect(field(line, "provider_key_id"), line).toBeUndefined();
    expect(field(line, "served_by_model"), line).toBeUndefined();
  });

  test("a one-target group's hit names no target either, and bills the hit to no key", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const prompt = "single-target-group-hit";
    expect((await chat("chit-single", prompt)).cache).toBe("miss");
    // Snapshot AFTER the miss: the miss did dispatch through this key and
    // its event is rightly attributed to it. Only the hit is under test.
    const before = await usageEventsFor(targetAPkId);
    const beforeUnknown = await usageEventsFor("unknown");
    const hit = await chat("chit-single", prompt);
    expect(hit.cache).toBe("hit");

    const line = await accessLine(hit.requestId);
    expect(field(line, "cache_status"), line).toBe("hit");
    // The pre-flight wrote this target before the cache was consulted. It
    // is not the producer, so the line must not report it.
    expect(field(line, "upstream_model"), line).toBeUndefined();
    expect(field(line, "provider_key_id"), line).toBeUndefined();

    // The usage event reads a different path than the line, and had the
    // same fabrication: the hit must not add to the key's event count.
    await new Promise((r) => setTimeout(r, 200));
    expect(await usageEventsFor(targetAPkId)).toBe(before);
    // ...and lands on the no-key series instead, so "did not increment
    // pkA" cannot pass by the event going missing altogether.
    expect(await usageEventsFor("unknown")).toBe(beforeUnknown + 1);
  });

  test("a group request NOT served from cache still names the target it dispatched to", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const miss = await chat("chit-pair", `group-miss-${Date.now()}`);
    expect(miss.cache).toBe("miss");

    const line = await accessLine(miss.requestId);
    expect(field(line, "cache_status"), line).toBe("miss");
    expect(field(line, "cache_hit_layer"), line).toBeUndefined();
    expect(field(line, "upstream_model"), line).toMatch(/^up-model-[ab]$/);
    expect(field(line, "served_by_model"), line).toMatch(/^chit-target-[ab]$/);
    expect(field(line, "provider_key_id"), line).toBeTruthy();
  });

  // LAST: this attaches an output guardrail to `chit-single`, which would
  // block the other cases' replies if it ran before them. Model-scoped so
  // it cannot reach `chit-solo` / `chit-pair` at all.
  test("a cache hit BLOCKED by an output guardrail names no target either", async (ctx) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return;
    }
    const prompt = "blocked-group-hit";
    // Cache the reply while nothing blocks it.
    expect((await chat("chit-single", prompt)).cache).toBe("miss");

    const guardrail = await seed.createGuardrail(
      {
        name: "chit-output-keyword",
        enabled: true,
        hook_point: "output",
        kind: "keyword",
        patterns: [{ kind: "literal", value: "reply" }],
      },
      { attach: false },
    );
    await seed.attachGuardrailToModel(guardrail.id, singleModelId);
    // Gate on a FRESH prompt (a miss, so it dispatches) being refused.
    await waitConfigPropagation(
      async () =>
        (await chat("chit-single", `blocked-probe-${Math.random()}`)).status ===
        422,
    );

    const blocked = await chat("chit-single", prompt);
    expect(blocked.status).toBe(422);

    // The request still reached only the cache, so the pre-flight's target
    // is no more true here than on the success exit — and this line is
    // written by the handler's ERROR branch, which never sees the cache at
    // all and can only read the attribution cell.
    const line = await accessLine(blocked.requestId);
    expect(field(line, "status"), line).toBe("422");
    expect(field(line, "upstream_model"), line).toBeUndefined();
    expect(field(line, "provider_key_id"), line).toBeUndefined();
  });
});
