import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: a DIRECT model — one with no model group — has a retry budget.
//
// The budget used to live only on `routing`, so a model that was not behind
// a group got zero retries and there was no field to change that: a single
// 502 went straight back to the caller. `Model.retries` now carries it,
// defaulting to the deployment-wide `upstream.retries` (2), so the direct
// path behaves like the group path.
//
// Every case drives an always-503 upstream and asserts the ATTEMPT COUNT,
// which is exact and non-flaky (unlike timing). Three properties are pinned:
//   1. the default budget retries a direct model (3 attempts, not 1),
//   2. `retries: 0` on the model is a real opt-out (1 attempt), and
//   3. the budget reaches endpoints that never had a retry loop at all —
//      `/v1/embeddings` stands in for the seven single-model endpoints.

const CALLER_PLAINTEXT = "sk-direct-retries-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** initial + 2 retries, from the built-in `upstream.retries` default. */
const EXPECTED_DEFAULT_ATTEMPTS = 3;

describe("direct model retries e2e: a model outside a group has a retry budget", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Every request returns a retryable 503, so the attempt count is exactly
    // the budget the gateway applied.
    upstream = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "always down", type: "server_error" } },
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "direct-retries-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });

    // Direct model: no `routing` block, no explicit `retries` → inherits the
    // deployment default. This is the configuration that used to be pinned
    // at zero retries with no way to change it.
    await seed.createModel({
      display_name: "direct-default-retries",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // Direct model that opts out explicitly.
    await seed.createModel({
      display_name: "direct-no-retries",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      retries: 0,
    });
    // Direct embeddings model — that endpoint had no retry loop at all.
    await seed.createModel({
      display_name: "direct-embed",
      provider: "openai",
      model_name: "text-embedding-3-small",
      provider_key_id: pk.id,
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "direct-default-retries",
        "direct-no-retries",
        "direct-embed",
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const clientFor = (a: SpawnedApp) =>
    new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${a.proxyUrl}/v1`,
      maxRetries: 0,
    });

  /**
   * Wait until `model` actually reaches the upstream. Before the model +
   * key propagate, the probe 404s at the gateway and never leaves it, so
   * "the upstream saw a new request" is the propagation signal (status
   * agnostic — this upstream always 503s).
   */
  const awaitModelLive = async (
    client: OpenAI,
    model: string,
    call: (c: OpenAI, m: string) => Promise<unknown>,
  ) =>
    waitConfigPropagation(async () => {
      const before = upstream!.receivedRequests.length;
      try {
        await call(client, model);
      } catch {
        // expected: the upstream is always down
      }
      return upstream!.receivedRequests.length > before;
    });

  const chat = (c: OpenAI, model: string) =>
    c.chat.completions.create({
      model,
      messages: [{ role: "user", content: "hi" }],
    });

  const embed = (c: OpenAI, model: string) =>
    c.embeddings.create({ model, input: "hi" });

  test("a direct model spends the default retry budget", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const client = clientFor(app);
    await awaitModelLive(client, "direct-default-retries", chat);

    const hitsBefore = upstream.receivedRequests.length;
    let caught: unknown;
    try {
      await chat(client, "direct-default-retries");
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(APIError);
    // Was 1 before the per-model budget landed: a direct model had no
    // `retries` field and the group knob did not apply to it.
    expect(upstream.receivedRequests.length - hitsBefore).toBe(
      EXPECTED_DEFAULT_ATTEMPTS,
    );
  });

  test("retries: 0 on a direct model makes exactly one attempt", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const client = clientFor(app);
    await awaitModelLive(client, "direct-no-retries", chat);

    const hitsBefore = upstream.receivedRequests.length;
    let caught: unknown;
    try {
      await chat(client, "direct-no-retries");
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(APIError);
    // The opt-out must be real: an explicit 0 cannot fall through to the
    // deployment default, or an operator could never turn retrying off.
    expect(upstream.receivedRequests.length - hitsBefore).toBe(1);
  });

  test("the budget reaches /v1/embeddings, which had no retry loop", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }
    const client = clientFor(app);
    await awaitModelLive(client, "direct-embed", embed);

    const hitsBefore = upstream.receivedRequests.length;
    let caught: unknown;
    try {
      await embed(client, "direct-embed");
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(APIError);
    // Was 1: `/v1/embeddings` dispatched once and returned the failure.
    expect(upstream.receivedRequests.length - hitsBefore).toBe(
      EXPECTED_DEFAULT_ATTEMPTS,
    );
  });
});
