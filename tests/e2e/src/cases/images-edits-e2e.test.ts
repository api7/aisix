import { createHash } from "node:crypto";
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

// E2E: /v1/images/edits end-to-end (AISIX-Cloud#1360).
//
// The image-editing models take the source image(s), optional mask,
// prompt, and every tuning parameter in one multipart/form-data body.
// The gateway drains the form, rewrites the `model` field to the
// upstream model id, rebuilds the form, and forwards it — every other
// field (file bytes included) must arrive at the upstream intact.
//
// User journeys pinned:
//
//   1. Caller POSTs an OpenAI-shape multipart edit request. Gateway
//      forwards to the upstream's /v1/images/edits as multipart with
//      only `model` rewritten; the upstream JSON (b64_json + usage)
//      returns to the caller intact, and a UsageEvent is emitted with
//      the upstream's token block.
//   2. A model on a non-OpenAI provider is rejected 400 at the gateway
//      boundary; the upstream is never contacted.
//
// References:
// - OpenAI Images edit API:
//   <https://platform.openai.com/docs/api-reference/images/createEdit>

const CALLER_PLAINTEXT = "sk-img-edits-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// ASCII markers standing in for PNG bytes — the mock upstream captures
// bodies as utf8 strings, so ASCII markers survive capture while still
// traveling through the gateway as opaque file parts.
const FAKE_IMAGE_BYTES = "PNG-FAKE-IMAGE-BYTES-e2e";
const FAKE_IMAGE_BYTES_2 = "PNG-FAKE-SECOND-IMAGE-e2e";
const FAKE_MASK_BYTES = "PNG-FAKE-MASK-BYTES-e2e";

describe("images edits e2e: /v1/images/edits multipart forward + model translation", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        // OpenAI Images edit response shape per
        // <https://platform.openai.com/docs/api-reference/images/object>:
        // gpt-image models return b64_json plus a usage token block.
        created: 1_700_000_000,
        data: [{ b64_json: "aGVsbG8=" }],
        usage: { input_tokens: 50, output_tokens: 1056, total_tokens: 1106 },
      },
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "img-edits-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "img-edits-e2e",
      provider: "openai",
      model_name: "gpt-image-2",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["img-edits-e2e"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const editCall = (model: string, caller: string = CALLER_PLAINTEXT) => {
    const form = new FormData();
    form.set("model", model);
    form.set("prompt", "add a red hat to the cat");
    form.set("size", "1024x1024");
    form.set("n", "1");
    // append, not set: repeated `image` parts are part of the rebuild
    // contract (multi-image edits) and must all reach the upstream.
    form.append(
      "image",
      new Blob([FAKE_IMAGE_BYTES], { type: "image/png" }),
      "cat.png",
    );
    form.append(
      "image",
      new Blob([FAKE_IMAGE_BYTES_2], { type: "image/png" }),
      "dog.png",
    );
    form.set(
      "mask",
      new Blob([FAKE_MASK_BYTES], { type: "image/png" }),
      "mask.png",
    );
    return fetch(`${app!.proxyUrl}/v1/images/edits`, {
      method: "POST",
      headers: { authorization: `Bearer ${caller}` },
      body: form,
    });
  };

  test("multipart edit: model rewritten, file bytes + fields intact, response and usage relayed", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      try {
        const r = await editCall("img-edits-e2e");
        if (r.status !== 200) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as { data?: unknown };
        return Array.isArray(j.data) && (j.data as unknown[]).length > 0;
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const res = await editCall("img-edits-e2e");

    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      created?: unknown;
      data?: Array<{ b64_json?: unknown }>;
      usage?: { input_tokens?: unknown; output_tokens?: unknown };
    };
    // Caller-side: the upstream's image payload and usage block reach
    // the caller intact — the b64 payload is the product, and the
    // usage block is the caller's own cost signal.
    expect(typeof body.created).toBe("number");
    expect(body.data).toHaveLength(1);
    expect(body.data?.[0]?.b64_json).toBe("aGVsbG8=");
    expect(body.usage?.input_tokens).toBe(50);
    expect(body.usage?.output_tokens).toBe(1056);

    // Dispatch contract: the gateway hit /v1/images/edits (not
    // /v1/images/generations or any other route) as multipart, with
    // the provider credential injected.
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/images/edits");
    expect(testCalls).toHaveLength(1);
    const sent = testCalls[0]!;
    expect(sent.method).toBe("POST");
    expect(sent.headers["authorization"]).toBe("Bearer sk-mock");
    expect(sent.headers["content-type"]).toContain("multipart/form-data");

    // Form contract: `model` rewritten to the upstream model id (the
    // caller alias never reaches the provider); every other field —
    // both repeated image parts, the mask, filenames, prompt, size,
    // n — forwards intact.
    expect(sent.body).toContain("gpt-image-2");
    expect(sent.body).not.toContain("img-edits-e2e");
    expect(sent.body).toContain(FAKE_IMAGE_BYTES);
    expect(sent.body).toContain(FAKE_IMAGE_BYTES_2);
    expect(sent.body).toContain(FAKE_MASK_BYTES);
    expect(sent.body).toContain("cat.png");
    expect(sent.body).toContain("dog.png");
    expect(sent.body).toContain("mask.png");
    expect(sent.body).toContain("add a red hat to the cat");
    expect(sent.body).toContain("1024x1024");

    // The request is metered: the usage-event counter for the images
    // handler family must move (#407 parity for the edits route).
    const emitted = await pollUsageEmitted(app.metricsUrl, "images");
    expect(emitted, "handler=images emit counter must be > 0").toBeGreaterThan(
      0,
    );
  });

  test("non-OpenAI provider rejected 400 at the boundary, upstream untouched", async (ctx) => {
    if (!etcdReachable || !app || !seed || !upstream) {
      ctx.skip();
      return;
    }

    const pk = await seed.createProviderKey({
      display_name: "img-edits-anthropic-pk",
      secret: "sk-anthropic-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "img-edits-anthropic",
      provider: "anthropic",
      model_name: "claude-3-5-haiku-20241022",
      provider_key_id: pk.id,
    });
    // Its own caller key, seeded after every other resource so the
    // propagation gate on it implies the whole seed set (AGENTS.md).
    const nonOaCaller = `${CALLER_PLAINTEXT}-non-oa`;
    await seed.createApiKey({
      key_hash: createHash("sha256").update(nonOaCaller).digest("hex"),
      allowed_models: ["img-edits-anthropic"],
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await editCall("img-edits-anthropic", nonOaCaller);
        if (r.status !== 400) {
          await r.text();
          return false;
        }
        const j = (await r.json()) as { error?: { type?: unknown } };
        return j.error?.type === "invalid_request_error";
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const res = await editCall("img-edits-anthropic", nonOaCaller);
    expect(res.status).toBe(400);
    const body = (await res.json()) as {
      error?: { type?: string; message?: string };
    };
    expect(body.error?.type).toBe("invalid_request_error");
    expect(body.error?.message).toContain("requires OpenAI");
    // Hard contract: the upstream is NEVER hit on a provider-mismatch
    // refusal.
    expect(upstream.receivedRequests.length).toBe(baseline);
  });
});

/**
 * Poll /metrics until `aisix_usage_events_emitted_total` for the given
 * handler is non-zero (the emit is synchronous; the scrape is
 * eventually consistent). Bounded so a regression fails rather than
 * hangs. Sums all label-sets for the handler.
 */
async function pollUsageEmitted(
  metricsUrl: string,
  handler: string,
): Promise<number> {
  const deadline = Date.now() + 5_000;
  let total = 0;
  while (Date.now() < deadline) {
    const text = await fetch(`${metricsUrl}/metrics`).then((r) => r.text());
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
