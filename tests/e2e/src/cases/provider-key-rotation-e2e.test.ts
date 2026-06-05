import { createHash } from "node:crypto";
import OpenAI from "openai";
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

// E2E: provider_key rotation with ZERO in-flight disruption
// (#196 L3, ai-gateway #271).
//
// Customer story: an operator rotates a leaked upstream credential by
// PUT-ing a new `secret` onto the existing provider_key (same id, same
// api_base). Live traffic must keep flowing — every request issued
// before, during, and after the rotation succeeds. A DP that rebuilt
// its per-provider_key upstream client non-atomically (briefly losing
// the key, or dropping in-flight requests on the snapshot swap) would
// surface here as ≥1 failed request.
//
// What this pins: under sustained concurrency, an in-place secret
// rotation (PUT /admin/v1/provider_keys/:id) causes zero failed
// requests and bumps the resource revision. The caller's api_key and
// the model alias are never touched, so a real client keeps using the
// same bearer throughout.
//
// Out of scope (separate gap #220): asserting WHICH secret the DP
// dialed upstream — the mock upstream ignores the credential, so the
// rotation's effect on the wire isn't observable here. This test pins
// the disruption-free property, which is L3's distinguishing claim.
//
// Reference: OpenAI Chat Completions shape the caller sees
// (https://platform.openai.com/docs/api-reference/chat); admin
// provider_key update is PUT /admin/v1/provider_keys/:id.

const CALLER_PLAINTEXT = "sk-pkrot-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const TOTAL_REQUESTS = 80;
const CONCURRENCY = 8;
// Fire the rotation ~40% into the batch so a healthy slice of requests
// is in flight across the snapshot swap.
const ROTATE_AT = Math.floor(TOTAL_REQUESTS * 0.4);

function errMsg(e: unknown): string {
  if (e && typeof e === "object" && "status" in e) {
    return `status=${(e as { status?: unknown }).status} ${String((e as { message?: unknown }).message ?? "")}`;
  }
  return String(e);
}

describe("provider_key rotation: zero in-flight disruption (#196 L3 / #271)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let pkId = "";
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-pkrot",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "pkrot-pk",
      secret: "sk-mock-v1",
      api_base: `${upstream.baseUrl}/v1`,
    });
    pkId = pk.id;
    await admin.createModel({
      display_name: "rot-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["rot-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("sustained concurrent chats survive an in-place secret rotation with zero failures", async (ctx) => {
    if (!etcdReachable || !app || !upstream || !admin || !pkId) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0, // surface failures rather than letting the SDK retry over them
    });

    // Readiness: the model + key are live.
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "rot-model",
          messages: [{ role: "user", content: "ready" }],
        });
        return probe.choices[0]?.message.content === "ok";
      } catch {
        return false;
      }
    });

    const revBefore = Number(
      ((await admin.json("GET", `/admin/v1/provider_keys/${pkId}`)) as { revision?: number }).revision ?? 0,
    );

    // Sustained concurrent load. One worker fires the in-place secret
    // rotation when it grabs index ROTATE_AT; the rest keep chatting,
    // so several requests are in flight across the snapshot swap.
    let sent = 0;
    let success = 0;
    let rotated = false;
    const failures: string[] = [];

    const worker = async (): Promise<void> => {
      for (;;) {
        const i = sent++;
        if (i >= TOTAL_REQUESTS) return;
        if (i === ROTATE_AT && !rotated) {
          rotated = true;
          // Rotate the credential in place: same id + api_base, new secret.
          await admin!.json("PUT", `/admin/v1/provider_keys/${pkId}`, {
            provider: "openai",
            adapter: "openai",
            display_name: "pkrot-pk",
            secret: "sk-mock-v2-rotated",
            api_base: `${upstream!.baseUrl}/v1`,
          });
        }
        try {
          const r = await client.chat.completions.create({
            model: "rot-model",
            messages: [{ role: "user", content: `rot-${i}` }],
          });
          if (r.choices[0]?.message.content === "ok") {
            success++;
          } else {
            failures.push(`req ${i}: unexpected content ${JSON.stringify(r.choices[0]?.message)}`);
          }
        } catch (e) {
          failures.push(`req ${i}: ${errMsg(e)}`);
        }
      }
    };

    await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));

    expect(rotated, "rotation was never triggered").toBe(true);
    // Zero in-flight disruption: every request before/during/after the
    // rotation succeeded. The message lists the first few failures so a
    // regression is debuggable.
    expect(
      failures,
      `${failures.length}/${TOTAL_REQUESTS} request(s) failed across the rotation: ${failures.slice(0, 3).join(" | ")}`,
    ).toEqual([]);
    expect(success).toBe(TOTAL_REQUESTS);

    // The rotation actually took effect (revision bumped).
    const revAfter = Number(
      ((await admin.json("GET", `/admin/v1/provider_keys/${pkId}`)) as { revision?: number }).revision ?? 0,
    );
    expect(revAfter).toBeGreaterThan(revBefore);
  }, 90_000);
});
