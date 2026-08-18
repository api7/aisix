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

// AISIX-Cloud#1325: a request that reached a real provider and came back
// 5xx used to emit `provider="unknown"` on `aisix_proxy_requests_total` —
// the handler failure branches all built their labels from
// `Upstream::default()`. That put one ProviderKey's successes and its
// failures on different series, so `by (provider)` reported a 0% failure
// rate for every real provider and 100% for `unknown`.
//
// The success side of each assertion is the control: the SAME key must
// carry the SAME provider / provider_key labels whether the request
// succeeded or failed, or the failure rate is still not computable.
const CALLER_PLAINTEXT = "sk-attr-1325-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const OK_PK = "attr1325-ok-pk";
const FAIL_PK = "attr1325-fail-pk";
const FAIL2_PK = "attr1325-fail2-pk";
const EMBED_PK = "attr1325-embed-pk";
const WILD_PK = "attr1325-wild-pk";

const OK_MODEL = "attr1325-ok";
const FAIL_MODEL = "attr1325-fail";
const FAIL2_MODEL = "attr1325-fail2";
const GROUP_MODEL = "attr1325-group";
const EMBED_MODEL = "attr1325-embed";
const WILD_ROW = "attr1325-wild/*";

describe("failed-request upstream attribution #1325 e2e", () => {
  let app: SpawnedApp | undefined;
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const ok = await startOpenAiUpstream({ nonStreamBody: chatBody() });
    // Upstream 5xx — the DP folds it into a 502 for the caller, which is
    // the exact shape reported on the issue.
    const failing = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "upstream is down" } },
    });
    const failing2 = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "second target is down too" } },
    });
    const embedFailing = await startOpenAiUpstream({
      status: 500,
      errorBody: { error: { message: "embeddings upstream is down" } },
    });
    upstreams.push(ok, failing, failing2, embedFailing);

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const seedPair = async (
      pkName: string,
      modelName: string,
      upstream: OpenAiUpstream,
      upstreamModel = "gpt-4o-mini",
    ) => {
      const pk = await seed.createProviderKey({
        display_name: pkName,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await seed.createModel({
        display_name: modelName,
        provider: "openai",
        model_name: upstreamModel,
        provider_key_id: pk.id,
      });
      return pk;
    };

    await seedPair(OK_PK, OK_MODEL, ok);
    await seedPair(FAIL_PK, FAIL_MODEL, failing);
    await seedPair(FAIL2_PK, FAIL2_MODEL, failing2);
    await seedPair(EMBED_PK, EMBED_MODEL, embedFailing, "text-embedding-3-small");
    // A wildcard row: `resolve_model` hands dispatch a synthetic Model whose
    // upstream id is the caller's own suffix, so the failure path has to
    // collapse it back to the row's template.
    await seedPair(WILD_PK, WILD_ROW, failing, "*");

    // Failover group whose targets both fail: the terminal metric must name
    // the LAST attempt, the one whose error the caller was served.
    await seed.createModel({
      display_name: GROUP_MODEL,
      routing: {
        strategy: "failover",
        targets: [{ model: FAIL_MODEL }, { model: FAIL2_MODEL }],
      },
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        OK_MODEL,
        FAIL_MODEL,
        FAIL2_MODEL,
        GROUP_MODEL,
        EMBED_MODEL,
        WILD_ROW,
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("an upstream 5xx keeps the provider and ProviderKey labels the success path uses", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const probe = await proxy.chat({
        model: FAIL_MODEL,
        messages: [{ role: "user", content: "ready" }],
      });
      // >= 500 distinguishes "reached the failing upstream" from the 404 an
      // un-propagated snapshot would answer with.
      return probe.status >= 500;
    });

    const ok = await proxy.chat({
      model: OK_MODEL,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(ok.status, JSON.stringify(ok.body)).toBe(200);
    const failed = await proxy.chat({
      model: FAIL_MODEL,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(failed.status).toBeGreaterThanOrEqual(500);

    const text = await scrape(app);
    const sample = requestSample(text, { model: FAIL_MODEL });
    expect(sample, `no request sample for ${FAIL_MODEL}:\n${text}`).toBeTruthy();
    expect(sample).toContain('provider="openai"');
    expect(sample).toContain(`provider_key_name="${FAIL_PK}"`);
    expect(sample).toContain('upstream_model="gpt-4o-mini"');
    expect(sample).not.toContain('provider="unknown"');
    expect(sample).not.toContain('provider_key_name="unknown"');

    // The whole point: the same label set on both outcomes, so a failure
    // rate per provider key is computable.
    const success = requestSample(text, { model: OK_MODEL });
    expect(success).toContain('provider="openai"');
    expect(success).toContain(`provider_key_name="${OK_PK}"`);
  });

  test("a streaming request that fails before the first token is attributed too", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // The reported repro is a streamed call: the upstream answers 5xx before
    // any SSE byte, so the handler takes the same failure branch with
    // `stream="true"`.
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: FAIL_MODEL,
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    expect(res.status).toBeGreaterThanOrEqual(500);
    await res.text();

    const text = await scrape(app);
    const sample = requestSample(text, { model: FAIL_MODEL, stream: "true" });
    expect(
      sample,
      `no streaming failure sample for ${FAIL_MODEL}:\n${text}`,
    ).toBeTruthy();
    expect(sample).toContain('provider="openai"');
    expect(sample).toContain(`provider_key_name="${FAIL_PK}"`);
  });

  test("a routing group whose targets all fail names the last attempt", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Both targets have to be loaded before the group can actually fail
    // OVER: a snapshot that carries the group and only its second target
    // would attempt that one first, which is a different scenario.
    await waitConfigPropagation(async () => {
      const first = await proxyChat(app!, FAIL_MODEL);
      const second = await proxyChat(app!, FAIL2_MODEL);
      const group = await proxyChat(app!, GROUP_MODEL);
      return first >= 500 && second >= 500 && group >= 500;
    });
    expect(await proxyChat(app, GROUP_MODEL)).toBeGreaterThanOrEqual(500);

    const text = await scrape(app);
    // Selected on `is_fallback="true"` so the assertion is about the
    // failed-over request specifically: `model` stays the group the caller
    // addressed, and the upstream dimensions name the target the request
    // died on — the second one.
    const sample = requestSample(text, {
      model: GROUP_MODEL,
      is_fallback: "true",
    });
    expect(
      sample,
      `no failed-over request sample for ${GROUP_MODEL}:\n${text}`,
    ).toBeTruthy();
    expect(sample).toContain(`provider_key_name="${FAIL2_PK}"`);
    expect(sample).not.toContain(`provider_key_name="${FAIL_PK}"`);
  });

  test("the fix spans the handler family, not just chat", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      const probe = await embed(app!, EMBED_MODEL);
      return probe >= 500;
    });
    expect(await embed(app, EMBED_MODEL)).toBeGreaterThanOrEqual(500);

    const text = await scrape(app);
    const sample = requestSample(text, { model: EMBED_MODEL });
    expect(sample, `no request sample for ${EMBED_MODEL}:\n${text}`).toBeTruthy();
    expect(sample).toContain('provider="openai"');
    expect(sample).toContain(`provider_key_name="${EMBED_PK}"`);
    expect(sample).toContain('upstream_model="text-embedding-3-small"');
  });

  test("a failed wildcard request is attributed without minting a series", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // The attribution the failure path reads comes off the target the
    // request selected — and for a wildcard row that target's upstream id is
    // whatever suffix the caller typed. It must reach the label as the row's
    // configured template, or a failing wildcard model becomes a cardinality
    // bomb (#451) reachable by anyone who can send a request.
    const minted = ["attr1325-wild/alpha", "attr1325-wild/beta"];
    await waitConfigPropagation(async () => {
      return (await proxyChat(app!, minted[0])) >= 500;
    });
    for (const name of minted) {
      expect(await proxyChat(app, name)).toBeGreaterThanOrEqual(500);
    }

    const text = await scrape(app);
    const sample = requestSample(text, { model: WILD_ROW });
    expect(sample, `no request sample for ${WILD_ROW}:\n${text}`).toBeTruthy();
    expect(sample).toContain(`provider_key_name="${WILD_PK}"`);
    expect(sample).toContain('upstream_model="*"');
    for (const name of minted) {
      expect(text, `caller-minted name leaked into a label`).not.toContain(name);
    }
  });

  test("a request that never reached an upstream still reports unknown", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // The honest half of the contract: nothing was selected, so there is
    // nothing to attribute. A fix that filled these in from stale state
    // would be worse than the bug.
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const res = await proxy.chat({
      model: "attr1325-no-such-model",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(404);

    const text = await scrape(app);
    const sample = requestSample(text, { model: "unresolved" });
    expect(sample, `no unresolved-model sample:\n${text}`).toBeTruthy();
    expect(sample).toContain('provider="unknown"');
    expect(sample).toContain('provider_key_id="unknown"');
    expect(sample).toContain('provider_key_name="unknown"');
  });
});

async function proxyChat(app: SpawnedApp, model: string): Promise<number> {
  const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }] }),
  });
  await res.text();
  return res.status;
}

async function embed(app: SpawnedApp, model: string): Promise<number> {
  const res = await fetch(`${app.proxyUrl}/v1/embeddings`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({ model, input: "hello" }),
  });
  await res.text();
  return res.status;
}

async function scrape(app: SpawnedApp): Promise<string> {
  const res = await fetch(`${app.metricsUrl}/metrics`);
  expect(res.status).toBe(200);
  return res.text();
}

/** The `aisix_proxy_requests_total` line matching every given label. */
function requestSample(
  scraped: string,
  labels: Record<string, string>,
): string | undefined {
  return scraped
    .split("\n")
    .filter((l) => l.startsWith("aisix_proxy_requests_total{"))
    .find((l) =>
      Object.entries(labels).every(([k, v]) => l.includes(`${k}="${v}"`)),
    );
}

function chatBody() {
  return {
    id: "chatcmpl-attr-1325",
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
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
  };
}
