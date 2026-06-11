import { createHash } from "node:crypto";
import { beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: UsageEvent emission on /v1/images/generations (#407) and
// /v1/audio/transcriptions (#406) — completes the #226 non-chat
// UsageEvent tracker (6/6 endpoints).
//
// Pre-fix both handlers emitted only an AccessLog + metrics; no
// UsageEvent → the requests were invisible to cp-api's budget ledger
// and /logs analytics. This drives real requests through the DP binary
// and asserts the DP's own `aisix_usage_events_emitted_total` counter
// (added in #408) increments for handler="images" / handler="audio".
//
// Token-based cost (gpt-image-1 / gpt-4o-transcribe usage blocks) is
// covered by the Rust unit tests; legacy duration/per-image cost
// (whisper-1 / dall-e-3) is a documented cross-repo follow-up.
//
// References:
// - OpenAI images object (usage): https://platform.openai.com/docs/api-reference/images/object
// - OpenAI audio (usage): https://platform.openai.com/docs/api-reference/audio
// - Issues: api7/AISIX-Cloud#406, #407

const CALLER_PLAINTEXT = "sk-audio-images-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("audio + images UsageEvent emission (#406/#407)", () => {
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
  });

  test("image generation emits a usage event (handler=images, #407)", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const upstream: OpenAiUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        created: 1_700_000_000,
        data: [{ b64_json: "aGVsbG8=" }],
        usage: { input_tokens: 50, output_tokens: 1568, total_tokens: 1618 },
      },
    });
    const app: SpawnedApp = await spawnApp();
    try {
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      const pk = await admin.createProviderKey({
        display_name: "img-usage-pk",
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await admin.createModel({
        display_name: "img-usage",
        provider: "openai",
        model_name: "gpt-image-1",
        provider_key_id: pk.id,
      });
      await admin.createApiKey({
        key_hash: CALLER_KEY_HASH,
        allowed_models: ["img-usage"],
      });

      const call = () =>
        fetch(`${app.proxyUrl}/v1/images/generations`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${CALLER_PLAINTEXT}`,
          },
          body: JSON.stringify({ model: "img-usage", prompt: "a cat", n: 1 }),
        });
      await waitConfigPropagation(async () => {
        // The DP may drop the connection mid-write (EPIPE) while the
        // route is still propagating — treat transport failures as
        // "not ready yet", like every other spec's predicate does.
        try {
          return (await call()).ok;
        } catch {
          return false;
        }
      });
      expect((await call()).status).toBe(200);

      const emitted = await pollUsageEmitted(app.adminUrl, "images");
      expect(emitted, "handler=images emit counter must be > 0 (#407)").toBeGreaterThan(0);
    } finally {
      await app.exit();
      await upstream.close();
    }
  });

  test("audio transcription emits a usage event (handler=audio, #406)", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const upstream: OpenAiUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        text: "hello world",
        usage: { type: "tokens", input_tokens: 14, output_tokens: 4, total_tokens: 18 },
      },
    });
    const app: SpawnedApp = await spawnApp();
    try {
      const admin = new AdminClient(app.adminUrl, app.adminKey);
      const pk = await admin.createProviderKey({
        display_name: "audio-usage-pk",
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}/v1`,
      });
      await admin.createModel({
        display_name: "audio-usage",
        provider: "openai",
        model_name: "gpt-4o-transcribe",
        provider_key_id: pk.id,
      });
      await admin.createApiKey({
        key_hash: CALLER_KEY_HASH,
        allowed_models: ["audio-usage"],
      });

      const call = () => {
        const form = new FormData();
        form.set("model", "audio-usage");
        form.set("file", new Blob([new Uint8Array([0x49, 0x44, 0x33])], { type: "audio/mpeg" }), "a.mp3");
        return fetch(`${app.proxyUrl}/v1/audio/transcriptions`, {
          method: "POST",
          headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
          body: form,
        });
      };
      await waitConfigPropagation(async () => {
        // The DP may drop the connection mid-write (EPIPE) while the
        // route is still propagating — treat transport failures as
        // "not ready yet", like every other spec's predicate does.
        try {
          return (await call()).ok;
        } catch {
          return false;
        }
      });
      expect((await call()).status).toBe(200);

      const emitted = await pollUsageEmitted(app.adminUrl, "audio");
      expect(emitted, "handler=audio emit counter must be > 0 (#406)").toBeGreaterThan(0);
    } finally {
      await app.exit();
      await upstream.close();
    }
  });
});

/**
 * Poll /metrics until the `aisix_usage_events_emitted_total` counter
 * for the given handler appears non-zero (the DP emits the event
 * synchronously on the request path, but the scrape is eventually
 * consistent). Bounded so a regression (no emit) fails rather than
 * hangs. Sums all label-sets for the handler.
 */
async function pollUsageEmitted(adminUrl: string, handler: string): Promise<number> {
  const deadline = Date.now() + 5_000;
  let total = 0;
  while (Date.now() < deadline) {
    const text = await fetch(`${adminUrl}/metrics`).then((r) => r.text());
    total = 0;
    for (const line of text.split("\n")) {
      if (!line.startsWith("aisix_usage_events_emitted_total{")) continue;
      if (!line.includes(`handler="${handler}"`)) continue;
      const v = Number.parseFloat(line.split("}").at(-1)?.trim() ?? "");
      if (!Number.isNaN(v)) total += v;
    }
    if (total > 0) break;
    await new Promise((r) => setTimeout(r, 100));
  }
  return total;
}
