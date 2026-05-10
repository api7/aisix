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

// E2E: HTTP 413 body-size limit on the proxy surface.
//
// Per `docs/api-proxy.md` §2 (updated in PR #191), the gateway
// rejects requests whose body exceeds axum's built-in `Json<…>`
// extractor limit (2 MiB). The `proxy.request_body_limit_bytes`
// config field is currently unused — see issue #193 — so the
// active limit on every JSON endpoint is the framework default.
//
// Two contracts pinned here:
//
//   1. A request with a JSON body > 2 MiB returns HTTP 413.
//      A regression that removed the limit (or raised it
//      silently) would let an attacker push unbounded JSON at
//      the proxy and exhaust memory.
//
//   2. A request well under 2 MiB does NOT return 413. Catches
//      a regression that lowered the limit too aggressively
//      (e.g. clamped at 64 KiB) and would reject legitimate
//      tool/vision payloads.
//
// Reference:
//   - `docs/api-proxy.md` §2 (413 row)
//   - issue #193 (`request_body_limit_bytes` not yet wired)
//   - axum docs on `DefaultBodyLimit`
//     <https://docs.rs/axum/latest/axum/extract/struct.DefaultBodyLimit.html>

const CALLER_PLAINTEXT = "sk-413-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// 2 MiB. Documented limit per api-proxy.md §2.
const LIMIT_BYTES = 2 * 1024 * 1024;

// A body whose total serialized size sits well above the limit. We
// add a comfortable margin so JSON-stringify overhead (key names,
// quotes, braces) doesn't accidentally bring us back under the line
// on some other harness change.
const OVERSIZED_FILL_BYTES = LIMIT_BYTES + 512 * 1024; // 2.5 MiB

// A body whose total serialized size sits well under the limit. Same
// shape and code path, just a smaller content string.
const UNDERSIZED_FILL_BYTES = 256 * 1024; // 256 KiB

describe("body-limit 413 e2e: oversized JSON rejected, undersized accepted", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "body-limit-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "body-limit-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["body-limit-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("oversized JSON body → 413; undersized passes through", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };

    // Snapshot propagation through the same code path the test
    // uses. A small probe so the wait isn't itself fighting the
    // body-limit check.
    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: "body-limit-model",
            messages: [{ role: "user", content: "ready-probe" }],
          }),
        });
        await r.text();
        return r.status === 200;
      } catch {
        return false;
      }
    });

    // (1) Oversized body — expect 413. Build the body and assert
    // it really is over the 2 MiB documented limit before sending,
    // so a future limit-raise surfaces as a clear test-side
    // assertion (not as the test passing for the wrong reason).
    const oversizedBody = JSON.stringify({
      model: "body-limit-model",
      messages: [
        { role: "user", content: "a".repeat(OVERSIZED_FILL_BYTES) },
      ],
    });
    expect(Buffer.byteLength(oversizedBody, "utf8")).toBeGreaterThan(
      LIMIT_BYTES,
    );

    const oversizedBaseline = upstream.receivedRequests.length;
    const oversizedRes = await fetch(
      `${app.proxyUrl}/v1/chat/completions`,
      {
        method: "POST",
        headers,
        body: oversizedBody,
      },
    );
    expect(oversizedRes.status).toBe(413);
    await oversizedRes.text();

    // The 413 must short-circuit the upstream — a regression that
    // streamed the body to upstream FIRST and only rejected after
    // the upstream responded would still surface 413 to the
    // caller but would (a) waste upstream tokens (b) leak the
    // client's payload to a third party. Catches that.
    expect(upstream.receivedRequests.length).toBe(oversizedBaseline);

    // (2) Undersized body — must pass through (200). Same headers,
    // same model, same shape; only the content size changed. A
    // regression that lowered the limit (or applied it to all
    // requests indiscriminately) would surface here.
    const undersizedBody = JSON.stringify({
      model: "body-limit-model",
      messages: [
        { role: "user", content: "a".repeat(UNDERSIZED_FILL_BYTES) },
      ],
    });
    expect(Buffer.byteLength(undersizedBody, "utf8")).toBeLessThan(
      LIMIT_BYTES,
    );

    const undersizedRes = await fetch(
      `${app.proxyUrl}/v1/chat/completions`,
      {
        method: "POST",
        headers,
        body: undersizedBody,
      },
    );
    expect(undersizedRes.status).toBe(200);
    await undersizedRes.text();
  }, 60_000);
});
