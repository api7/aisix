import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1012: some providers use non-429 4xx codes for transient
// conditions (model overload, queue full, quota). By default the gateway
// treats a non-429 4xx as the caller's error and returns it as-is; a
// routing model can now opt SPECIFIC codes into retry/failover via
// `fallback_on_statuses`. This drives two identical two-target groups —
// first target always answers 422, second target healthy — and pins:
//
//   1. default group: the 422 is relayed to the caller (behavior
//      unchanged; the second target is never consulted)
//   2. `fallback_on_statuses: [422]` group: the request fails over and
//      succeeds on the second target
//   3. codes NOT in the list keep the default (a 400 from the same
//      configured group still relays)

const CALLER_PLAINTEXT = "sk-fos-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

describe("fallback_on_statuses e2e: opt-in 4xx failover (#1012)", () => {
  let app: SpawnedApp | undefined;
  let overloaded: OpenAiUpstream | undefined;
  let badRequest: OpenAiUpstream | undefined;
  let healthy: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    // "Overloaded" provider: answers every call with 422 (the shape some
    // model services use for queue-full/overload conditions).
    overloaded = await startOpenAiUpstream({
      status: 422,
      errorBody: { error: { message: "model overloaded, try later", type: "overloaded" } },
    });
    // Provider that rejects with a plain 400 (a genuine caller error).
    badRequest = await startOpenAiUpstream({
      status: 400,
      errorBody: { error: { message: "bad request", type: "invalid_request_error" } },
    });
    healthy = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-fos",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "served by the backup" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 4, completion_tokens: 4, total_tokens: 8 },
      },
    });

    app = await spawnApp();
    const admin = new AdminClient(app.adminUrl, app.adminKey);

    const overloadedPk = await admin.createProviderKey({
      display_name: "fos-overloaded-pk",
      secret: "sk-mock",
      api_base: `${overloaded.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "fos-overloaded",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: overloadedPk.id,
      cooldown: { enabled: false },
    });
    const badPk = await admin.createProviderKey({
      display_name: "fos-bad-pk",
      secret: "sk-mock",
      api_base: `${badRequest.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "fos-bad",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: badPk.id,
      cooldown: { enabled: false },
    });
    const healthyPk = await admin.createProviderKey({
      display_name: "fos-healthy-pk",
      secret: "sk-mock",
      api_base: `${healthy.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "fos-healthy",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: healthyPk.id,
    });

    // Same two targets, with and without the opt-in.
    await admin.createModel({
      display_name: "fos-default-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "fos-overloaded" }, { model: "fos-healthy" }],
      },
    });
    await admin.createModel({
      display_name: "fos-configured-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "fos-overloaded" }, { model: "fos-healthy" }],
        fallback_on_statuses: [422],
      },
    });
    // Configured group whose FIRST target 400s — 400 is not in the list,
    // so it must still relay.
    await admin.createModel({
      display_name: "fos-unlisted-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "fos-bad" }, { model: "fos-healthy" }],
        fallback_on_statuses: [422],
      },
    });

    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "fos-default-group",
        "fos-configured-group",
        "fos-unlisted-group",
        "fos-healthy",
      ],
    });

    await waitConfigPropagation(async () => {
      const r = await chat("fos-healthy");
      return r.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await overloaded?.close();
    await badRequest?.close();
    await healthy?.close();
  });

  async function chat(model: string): Promise<Response> {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: "route me" }],
      }),
    });
    return res;
  }

  test("default: a non-429 4xx from the first target is relayed, no failover", async (ctx) => {
    if (!etcdReachable || !app || !overloaded || !healthy) {
      ctx.skip();
      return;
    }
    // Gate on the group being resolvable (not 404) before asserting.
    await waitConfigPropagation(async () => {
      const probe = await chat("fos-default-group");
      await probe.text();
      return probe.status !== 404;
    });

    const healthyBefore = healthy.receivedRequests.length;
    const res = await chat("fos-default-group");
    const body = await res.text();
    expect(res.status).toBe(422);
    expect(body).toContain("model overloaded");
    // The healthy backup was never consulted — 4xx stays terminal.
    expect(healthy.receivedRequests.length).toBe(healthyBefore);
  });

  test("fallback_on_statuses [422]: the same 422 fails over and succeeds", async (ctx) => {
    if (!etcdReachable || !app || !overloaded || !healthy) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      const probe = await chat("fos-configured-group");
      await probe.text();
      return probe.status !== 404;
    });

    const overloadedBefore = overloaded.receivedRequests.length;
    const healthyBefore = healthy.receivedRequests.length;
    const res = await chat("fos-configured-group");
    const body = JSON.parse(await res.text()) as {
      choices: Array<{ message: { content: string } }>;
    };
    expect(res.status).toBe(200);
    expect(body.choices[0]!.message.content).toContain("served by the backup");
    // First target was tried (and answered 422), then the backup won.
    expect(overloaded.receivedRequests.length).toBeGreaterThan(overloadedBefore);
    expect(healthy.receivedRequests.length).toBe(healthyBefore + 1);
  });

  test("codes not in the list keep the default: a 400 still relays", async (ctx) => {
    if (!etcdReachable || !app || !badRequest || !healthy) {
      ctx.skip();
      return;
    }
    await waitConfigPropagation(async () => {
      const probe = await chat("fos-unlisted-group");
      await probe.text();
      return probe.status !== 404;
    });

    const healthyBefore = healthy.receivedRequests.length;
    const res = await chat("fos-unlisted-group");
    await res.text();
    expect(res.status).toBe(400);
    expect(healthy.receivedRequests.length).toBe(healthyBefore);
  });
});
