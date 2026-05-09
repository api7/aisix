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

// E2E: /v1/images/generations end-to-end. Per gateway docs
// `docs/api-proxy.md` §4.9:
//
//   > OpenAI Images API. Forwarded with the `model` field rewritten.
//
// Prior to this file, the gateway had **zero** e2e coverage on
// /v1/images/generations.
//
// One user journey pinned:
//
//   - Caller POSTs OpenAI-shape image-generation request to
//     /v1/images/generations. Gateway forwards to upstream's
//     /v1/images/generations with only the `model` field rewritten.
//     Caller receives upstream's response back unchanged.
//
// References:
// - Gateway's own /v1/images/generations contract:
//   `docs/api-proxy.md` §4.9
// - OpenAI Images API spec:
//   <https://platform.openai.com/docs/api-reference/images/create>

const CALLER_PLAINTEXT = "sk-img-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("images generations e2e: /v1/images/generations verbatim forward + model translation", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        // OpenAI Images API response shape per
        // <https://platform.openai.com/docs/api-reference/images/object>.
        created: Math.floor(Date.now() / 1000),
        data: [
          {
            url: "https://mock.example.com/img-1.png",
            revised_prompt: "A cat sitting in a sunbeam (refined).",
          },
        ],
      },
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "img-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "img-e2e",
      provider: "openai",
      model_name: "gpt-image-1",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["img-e2e"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("OpenAI-shape images.generations: caller body verbatim + model translated, response byte-for-byte", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const headers = {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    };

    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/images/generations`, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: "img-e2e",
            prompt: "ready-probe",
          }),
        });
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
    const requestPayload = {
      model: "img-e2e",
      prompt: "A cat sitting in a sunbeam",
      n: 1,
      size: "1024x1024",
      response_format: "url",
    };
    const res = await fetch(`${app.proxyUrl}/v1/images/generations`, {
      method: "POST",
      headers,
      body: JSON.stringify(requestPayload),
    });

    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      created?: unknown;
      data?: Array<{ url?: unknown; revised_prompt?: unknown }>;
    };
    // Caller-side: response byte-for-byte from upstream. The image
    // url and any revised_prompt the upstream produced must reach
    // the caller intact — they're the only signal the caller has
    // about what was generated.
    expect(typeof body.created).toBe("number");
    expect(body.data).toHaveLength(1);
    expect(body.data?.[0]?.url).toBe("https://mock.example.com/img-1.png");
    expect(body.data?.[0]?.revised_prompt).toBe(
      "A cat sitting in a sunbeam (refined).",
    );

    // Dispatch contract: gateway hit `/v1/images/generations` (not
    // /v1/chat/completions or any other route).
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/images/generations");
    expect(testCalls).toHaveLength(1);
    expect(testCalls[0]?.method).toBe("POST");
    expect(testCalls[0]?.headers["authorization"]).toBe("Bearer sk-mock");

    // Body contract per docs §4.9: forwarded verbatim with the
    // `model` field rewritten. Verify:
    //   - `model` rewritten to upstream model_name
    //   - everything else byte-for-byte (prompt, n, size,
    //     response_format)
    const sentBody = JSON.parse(testCalls[0]!.body) as {
      model?: string;
      prompt?: string;
      n?: number;
      size?: string;
      response_format?: string;
    };
    expect(sentBody.model).toBe("gpt-image-1");
    expect(sentBody.prompt).toBe(requestPayload.prompt);
    expect(sentBody.n).toBe(requestPayload.n);
    expect(sentBody.size).toBe(requestPayload.size);
    expect(sentBody.response_format).toBe(requestPayload.response_format);
  });
});
