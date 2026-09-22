import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1571: a caller that gives up before the response head is
// written left NO row in the usage log. Every endpoint emits from the tail of
// its own handler, and axum drops that future when the client disconnects —
// so a request that reached a provider and kept it busy was invisible to the
// control plane, which is the case an operator most needs to see (the usual
// reason a caller gives up is a long time to first token).
//
// The scenario is deliberately a ROUTING group: the group is what the caller
// addressed and the target is what the gateway was waiting on, and a single
// `direct` model — which every earlier cancel test used — makes those two
// identities the same value, so it cannot tell a correct row from one that
// reports the group where the target belongs.
const CALLER_PLAINTEXT = "sk-cancel-1571-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");
const PROVIDER_SECRET = "sk-mock-cancel-1571";

const CREDENTIAL_REF = "mock";
const SLS_PROJECT = "aisix-e2e-obs";
const LOGSTORE = "cancel-events";

const GROUP = "c1571-group";
/** A passthrough route onto the same slow upstream: a surface with no model. */
const ROUTE = "c1571-tunnel";
const TARGET = "c1571-target";
const UPSTREAM_MODEL = "gpt-4o-mini";
/** A direct model on a fast upstream, for the success-path line below. */
const FAST_MODEL = "c1571-fast";
const FAST_UPSTREAM_MODEL = "gpt-4o-fast";
/** A direct model on an upstream that trickles its stream, for the
 *  mid-stream abandon below — the head IS written there, which is the
 *  ending the handler tail used to log as a `200`. */
const STREAM_MODEL = "c1571-stream";
const STREAM_UPSTREAM_MODEL = "gpt-4o-stream";

describe("client cancel before the response head (AISIX-Cloud#1571)", () => {
  let etcdReachable = false;
  let slow: OpenAiUpstream | undefined;
  let fast: OpenAiUpstream | undefined;
  let trickle: OpenAiUpstream | undefined;
  let sls: MockSls | undefined;
  let app: SpawnedApp | undefined;
  let targetModelId = "";
  let providerKeyId = "";
  let fastProviderKeyId = "";
  let streamProviderKeyId = "";

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;
    // Long enough that the caller is certainly still waiting when it aborts,
    // and that the gateway never gets a head to forward.
    slow = await startOpenAiUpstream({
      responseDelayMs: 30_000,
      streamEvents: ["[DONE]"],
    });
    fast = await startOpenAiUpstream({
      nonStreamBody: {
        id: "chatcmpl-c1571",
        object: "chat.completion",
        created: 1_700_000_000,
        model: FAST_UPSTREAM_MODEL,
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });
    // Half a second between events, so a caller can read the first one and
    // abort while the rest is still coming — the mid-stream ending.
    trickle = await startOpenAiUpstream({
      eventDelayMs: 500,
      streamEvents: [
        JSON.stringify({
          id: "chatcmpl-c1571-stream",
          object: "chat.completion.chunk",
          created: 1_700_000_000,
          model: STREAM_UPSTREAM_MODEL,
          choices: [{ index: 0, delta: { role: "assistant", content: "one" } }],
        }),
        JSON.stringify({
          id: "chatcmpl-c1571-stream",
          object: "chat.completion.chunk",
          created: 1_700_000_000,
          model: STREAM_UPSTREAM_MODEL,
          choices: [{ index: 0, delta: { content: "two" } }],
        }),
        JSON.stringify({
          id: "chatcmpl-c1571-stream",
          object: "chat.completion.chunk",
          created: 1_700_000_000,
          model: STREAM_UPSTREAM_MODEL,
          choices: [{ index: 0, delta: { content: "three" }, finish_reason: "stop" }],
        }),
        "[DONE]",
      ],
    });
    sls = await startMockSls();

    app = await spawnApp({
      // The access-log line is `tracing::info!`; the harness default of
      // `warn` would leave it unwritten and the assertion below vacuous.
      logLevel: "info",
      extraEnv: {
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    const seed = new SeedClient(new EtcdClient(), app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-cancel",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: SLS_PROJECT,
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });
    const pk = await seed.createProviderKey({
      display_name: "c1571-pk",
      secret: PROVIDER_SECRET,
      api_base: `${slow.baseUrl}/v1`,
    });
    providerKeyId = pk.id;
    await seed.createModel({
      display_name: GROUP,
      routing: { strategy: "failover", targets: [{ model: TARGET }] },
    });
    const target = await seed.createModel({
      display_name: TARGET,
      provider: "openai",
      model_name: UPSTREAM_MODEL,
      provider_key_id: pk.id,
    });
    targetModelId = target.id;
    const fastPk = await seed.createProviderKey({
      display_name: "c1571-fast-pk",
      secret: PROVIDER_SECRET,
      api_base: `${fast.baseUrl}/v1`,
    });
    fastProviderKeyId = fastPk.id;
    await seed.createModel({
      display_name: FAST_MODEL,
      provider: "openai",
      model_name: FAST_UPSTREAM_MODEL,
      provider_key_id: fastPk.id,
    });
    const streamPk = await seed.createProviderKey({
      display_name: "c1571-stream-pk",
      secret: PROVIDER_SECRET,
      api_base: `${trickle.baseUrl}/v1`,
    });
    streamProviderKeyId = streamPk.id;
    await seed.createModel({
      display_name: STREAM_MODEL,
      provider: "openai",
      model_name: STREAM_UPSTREAM_MODEL,
      provider_key_id: streamPk.id,
    });
    await seed.createPassthroughRoute({
      name: ROUTE,
      path_prefix: "/passthrough/c1571",
      target_url: slow.baseUrl,
      provider_key_id: pk.id,
    });
    // Seeded last, so it authenticating implies everything above is in the
    // snapshot (tests/e2e/AGENTS.md).
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
      allowed_routes: ["*"],
    });
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await slow?.close();
    await fast?.close();
    await trickle?.close();
    await sls?.close();
  });

  test(
    "an abandoned request is metered, and names the target it was waiting on",
    async (ctx) => {
      if (!etcdReachable || !app || !slow || !fast || !sls) {
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
          model: GROUP,
          messages: [{ role: "user", content: "hang up on me" }],
          stream: true,
        }),
        signal: controller.signal,
      });

      // Abort once the upstream has actually received the call: by then the
      // gateway has authenticated, resolved the group, picked the target and
      // dispatched, and the upstream is still 30s from answering — so the
      // response head is unwritten when the caller goes away. A fixed sleep
      // could fire before dispatch on a loaded machine and silently assert
      // the pre-dispatch shape instead.
      for (let i = 0; i < 200 && slow.receivedRequests.length === 0; i++) {
        await new Promise((r) => setTimeout(r, 25));
      }
      expect(
        slow.receivedRequests.length,
        "the gateway never dispatched to the slow upstream",
      ).toBeGreaterThan(0);
      controller.abort();
      await expect(inflight).rejects.toThrow();

      const row = await waitForSlsLog(
        sls,
        LOGSTORE,
        (log) => log.get("requested_model") === GROUP,
        `a usage row for the cancelled request to ${GROUP}`,
        20_000,
      );

      expect(row.get("status_code")).toBe("499");
      expect(row.get("error_class")).toBe("client_disconnected");
      expect(row.get("error_message")).toContain("before the response head");
      // The two identities the row has to keep apart: the caller addressed
      // the group, the gateway was waiting on the target. `model_id` is what
      // the control plane prices against, and a group has no pricing row.
      expect(row.get("requested_model")).toBe(GROUP);
      expect(row.get("model_id")).toBe(targetModelId);
      expect(row.get("attempt_model")).toBe(TARGET);
      expect(row.get("operation")).toBe("chat");
      // Abandoned before a single token existed — the row must cost nothing.
      expect(row.get("prompt_tokens") ?? "0").toBe("0");
      expect(row.get("completion_tokens") ?? "0").toBe("0");

      // The access-log line for the SAME request names the target too.
      // Before this change `model=` (the group) was the only identity on the
      // line and the target was unreachable by request id anywhere. The row
      // above carries the id to join on — the aborted fetch never got to
      // read the `x-aisix-request-id` header.
      const requestId = row.get("request_id") ?? "";
      expect(requestId).not.toBe("");
      const line = await waitForLogLine(
        app,
        (l) => l.includes(`request_id="${requestId}"`) && l.includes("status=499"),
        `the 499 access-log line for ${requestId}`,
      );
      expect(line).toContain(`model="${GROUP}"`);
      expect(line).toContain(`upstream_model="${UPSTREAM_MODEL}"`);
      expect(line).toContain(`provider_key_id="${providerKeyId}"`);
    },
    60_000,
  );

  // A surface that names no model at all. The gateway still spent an
  // upstream's time on the caller's behalf, so the row has to exist — and
  // it is attributed by the route, which is all this family has.
  test(
    "an abandoned passthrough request is metered against its route",
    async (ctx) => {
      if (!etcdReachable || !app || !slow || !sls) {
        ctx.skip();
        return;
      }
      const before = slow.receivedRequests.length;
      const controller = new AbortController();
      const inflight = fetch(`${app.proxyUrl}/passthrough/c1571/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: "anything", messages: [] }),
        signal: controller.signal,
      });

      // Abort once the relay has reached the upstream, which is 30s from
      // answering — so the response head is unwritten when the caller goes
      // away. A fixed sleep could fire before the relay dispatched.
      for (let i = 0; i < 200 && slow.receivedRequests.length === before; i++) {
        await new Promise((r) => setTimeout(r, 25));
      }
      expect(
        slow.receivedRequests.length,
        "the relay never reached the upstream",
      ).toBeGreaterThan(before);
      controller.abort();
      await expect(inflight).rejects.toThrow();

      const row = await waitForSlsLog(
        sls,
        LOGSTORE,
        (log) => log.get("operation") === "passthrough",
        `a usage row for the cancelled passthrough request to ${ROUTE}`,
        20_000,
      );

      expect(row.get("status_code")).toBe("499");
      expect(row.get("error_class")).toBe("client_disconnected");
      expect(row.get("error_message")).toContain("before the response head");
      expect(row.get("passthrough_route_name")).toBe(ROUTE);
      // This surface resolves no model, and the row says so rather than
      // borrowing one — `model_id` is what the control plane prices on.
      expect(row.get("requested_model") ?? "").toBe("");
      expect(row.get("model_id") ?? "").toBe("");
      // Attributable all the same: the key is resolved before any of this.
      expect(row.get("api_key_id") ?? "").not.toBe("");
    },
    60_000,
  );

  // The pair is not a 499-only field. Asserting it only on the cancel line
  // would leave a version that fills it from the guard and nowhere else
  // looking entirely correct.
  test("an ordinary completed request names its target on the line too", async (ctx) => {
    if (!etcdReachable || !app || !fast) {
      ctx.skip();
      return;
    }
    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: FAST_MODEL,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    await res.arrayBuffer();
    expect(res.status).toBe(200);
    const requestId = res.headers.get("x-aisix-request-id") ?? "";
    expect(requestId).not.toBe("");

    const line = await waitForLogLine(
      app,
      (l) => l.includes(`request_id="${requestId}"`) && l.includes("status=200"),
      `the 200 access-log line for ${requestId}`,
    );
    expect(line).toContain(`upstream_model="${FAST_UPSTREAM_MODEL}"`);
    expect(line).toContain(`provider_key_id="${fastProviderKeyId}"`);
  });

  // The other half of the same request: a caller that walks away AFTER the
  // response head — the ordinary ending for a long stream. The row was
  // already a 499 before this change; the LINE said 200, because the handler
  // wrote it when it handed the stream over, minutes before the request
  // ended. One request read as two, under two statuses, and nothing in the
  // log said which one was the outcome.
  test(
    "a stream abandoned mid-flight writes exactly one line, and it says what the row says",
    async (ctx) => {
      if (!etcdReachable || !app || !trickle || !fast || !sls) {
        ctx.skip();
        return;
      }
      const controller = new AbortController();
      const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: STREAM_MODEL,
          messages: [{ role: "user", content: "start streaming" }],
          stream: true,
        }),
        signal: controller.signal,
      });
      expect(res.status, "the head must go out — this is the mid-stream ending").toBe(200);
      const requestId = res.headers.get("x-aisix-request-id") ?? "";
      expect(requestId).not.toBe("");

      // Read one event, then hang up with the rest still coming.
      const reader = res.body!.getReader();
      const first = await reader.read();
      expect(first.done, "the stream delivered nothing to abandon").toBe(false);
      controller.abort();
      // `cancel()` races the abort: it resolves when cancellation wins and
      // rejects with the body's `AbortError` when the abort does. Any other
      // rejection is a real failure and must not be swallowed.
      await reader.cancel().catch((err: unknown) => {
        if (!(err instanceof Error) || err.name !== "AbortError") throw err;
      });

      const row = await waitForSlsLog(
        sls,
        LOGSTORE,
        (log) => log.get("request_id") === requestId,
        `a usage row for the abandoned stream ${requestId}`,
        20_000,
      );
      expect(row.get("status_code")).toBe("499");
      expect(row.get("error_class")).toBe("client_disconnected");
      expect(row.get("error_message")).toContain("while the response was streaming");

      // A later request's line is the barrier: the log queue and its
      // writer thread are FIFO, so once this one is visible a SECOND
      // line for the abandoned stream would be visible too. Waiting for
      // the abandoned request's own line would settle on the first of
      // them and leave "exactly one" asserting nothing.
      const barrier = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: FAST_MODEL,
          messages: [{ role: "user", content: "barrier" }],
        }),
      });
      await barrier.arrayBuffer();
      const barrierId = barrier.headers.get("x-aisix-request-id") ?? "";
      await waitForLogLine(
        app,
        (l) =>
          l.includes("proxy request completed") &&
          l.includes(`request_id="${barrierId}"`),
        `the access-log line for the barrier request ${barrierId}`,
      );
      const lines = app
        .output()
        .split("\n")
        .filter(
          (l) => l.includes("proxy request completed") && l.includes(`request_id="${requestId}"`),
        );
      expect(
        lines.length,
        `one request, one line — got ${lines.length} for ${requestId}:\n${lines.join("\n")}`,
      ).toBe(1);
      const line = lines[0];
      expect(line).toContain("status=499");
      expect(line).toContain(`error_kind="client_disconnected"`);
      expect(line).toContain("while the response was streaming");
      // The target is still named — the line an operator reads to find out
      // which upstream the abandoned call was costing them.
      expect(line).toContain(`model="${STREAM_MODEL}"`);
      expect(line).toContain(`upstream_model="${STREAM_UPSTREAM_MODEL}"`);
      expect(line).toContain(`provider_key_id="${streamProviderKeyId}"`);
      // And the two spans the line now separates: what the caller waited for
      // (the first token) inside how long the request ran.
      expect(line).toMatch(/\blatency_ms=\d+/);
      expect(line).toMatch(/\bduration_ms=\d+/);
    },
    60_000,
  );
});
