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

// E2E for #911 finding [21]: the non-chat proxy endpoints reserved the
// rate-limit layers but never committed the actual token cost, so their TPM/
// TPD (token-per-minute/day) counters never moved — a caller could bypass
// token-rate limits by routing traffic through them. This exercises the fix
// on /v1/completions: with a TPM cap of 10 and an upstream that reports 16
// tokens, the first call must succeed (and commit its 16 tokens) and the
// second must be rejected 429. Pre-fix the counter stayed 0 and the second
// call also succeeded.

const CALLER_PLAINTEXT = "sk-tpm-commit-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const TPM = 10;
// Upstream-reported usage per call: 8 + 8 = 16 > TPM, so ONE call exhausts it.
const COMPLETION_BODY = {
  id: "cmpl-mock",
  object: "text_completion",
  created: 0,
  model: "gpt-3.5-turbo-instruct",
  choices: [{ text: "hello", index: 0, finish_reason: "stop", logprobs: null }],
  usage: { prompt_tokens: 8, completion_tokens: 8, total_tokens: 16 },
};

// #950 boundary, at the layer a caller sees it: an upstream that reports
// usage landing EXACTLY on the cap must exhaust the budget, not leave one
// more request's worth of it. Post-paid token accounting already lets the
// crossing request through; admitting a further one on top of that is the
// avoidable overshoot.
const EXACT_TPM = 10;
const EXACT_BODY = {
  id: "cmpl-exact",
  object: "text_completion",
  created: 0,
  model: "gpt-3.5-turbo-instruct",
  choices: [{ text: "hello", index: 0, finish_reason: "stop", logprobs: null }],
  usage: { prompt_tokens: 5, completion_tokens: 5, total_tokens: 10 },
};

describe("completions TPM boundary: usage exactly on the cap (#950)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;
  const caller = "sk-tpm-exact-caller";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ nonStreamBody: EXACT_BODY });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: "tpm-exact-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "tpm-exact",
      provider: "openai",
      model_name: "gpt-3.5-turbo-instruct",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: createHash("sha256").update(caller).digest("hex"),
      allowed_models: ["tpm-exact"],
      rate_limit: { tpm: EXACT_TPM },
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("a call that commits exactly the cap leaves no budget for the next one", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const probe = new ProxyClient(app.proxyUrl, caller);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "tpm-exact");
    });

    const post = () =>
      fetch(`${app!.proxyUrl}/v1/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${caller}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: "tpm-exact", prompt: "hi" }),
      });

    const first = await post();
    expect(first.status).toBe(200);
    await first.body?.cancel();

    // 10 committed against a cap of 10 — the whole budget, exactly.
    const second = await post();
    expect(second.status).toBe(429);
    // And the refusal names the window that fired, so this cannot pass on
    // some other limit having tripped.
    expect(second.headers.get("x-ratelimit-scope")).toBe("tpm");
    expect(second.headers.get("x-ratelimit-limit")).toBe(String(EXACT_TPM));
    expect(second.headers.get("x-ratelimit-remaining")).toBe("0");
    await second.body?.cancel();
  });
});

describe("completions TPM commit (#911 [21])", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({ nonStreamBody: COMPLETION_BODY });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "tpm-commit-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "tpm-commit",
      provider: "openai",
      model_name: "gpt-3.5-turbo-instruct",
      provider_key_id: pk.id,
    });
    // TPM=10 on the caller's key. The first /v1/completions call commits 16
    // tokens (> 10), so the second must be rejected on the token counter.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["tpm-commit"],
      rate_limit: { tpm: TPM },
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function postCompletion(): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model: "tpm-commit", prompt: "hi" }),
    });
  }

  test("second /v1/completions call is 429 once TPM is committed", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // listModels doesn't consume the token budget, so it's a safe readiness
    // probe that leaves the TPM quota intact for the test.
    const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "tpm-commit");
    });

    // First call succeeds and commits 16 tokens against the TPM=10 counter.
    const first = await postCompletion();
    expect(first.status).toBe(200);

    // Second call within the same minute window must be rejected: the token
    // counter is now 16 >= 10. Pre-fix (no commit) it stayed 0 and this
    // returned 200.
    const second = await postCompletion();
    expect(second.status).toBe(429);
  });
});
