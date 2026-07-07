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

// E2E: every proxy response must carry `x-aisix-request-id` so a client
// can correlate its response to the usage event (both key on the id).
// In v0.3.0 the header was set only by some handlers (rerank,
// passthrough) and missing from chat / completions / embeddings /
// responses / messages — QA finding. The `ensure_request_id` middleware
// now stamps it on every response (including short-circuited 4xx), and
// the id is the same one the handler attributes its usage event to.

const VALID_PLAINTEXT = "sk-request-id-e2e-valid";
const VALID_KEY_HASH = createHash("sha256")
  .update(VALID_PLAINTEXT)
  .digest("hex");
const UNKNOWN_PLAINTEXT = "sk-request-id-e2e-unregistered";

const REQUEST_ID_HEADER = "x-aisix-request-id";
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

describe("data plane stamps x-aisix-request-id on every response", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    const admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "request-id-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "request-id-model",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: VALID_KEY_HASH,
      allowed_models: ["request-id-model"],
    });

    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${VALID_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "request-id-model",
          messages: [{ role: "user", content: "ready-probe" }],
        }),
      });
      await res.text();
      return res.status === 200;
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function post(path: string, body: unknown, key = VALID_PLAINTEXT) {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${key}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    await res.text();
    return res;
  }

  test("the whole proxy family + error envelopes carry a UUID request id", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const cases: Array<[string, string, unknown]> = [
      [
        "chat/completions",
        "/v1/chat/completions",
        {
          model: "request-id-model",
          messages: [{ role: "user", content: "hi" }],
        },
      ],
      [
        "completions",
        "/v1/completions",
        { model: "request-id-model", prompt: "hi" },
      ],
      [
        "embeddings",
        "/v1/embeddings",
        { model: "request-id-model", input: "hi" },
      ],
      [
        "rerank",
        "/v1/rerank",
        { model: "request-id-model", query: "q", documents: ["d"] },
      ],
      [
        "responses",
        "/v1/responses",
        { model: "request-id-model", input: "hi" },
      ],
      [
        "messages",
        "/v1/messages",
        {
          model: "request-id-model",
          max_tokens: 16,
          messages: [{ role: "user", content: "hi" }],
        },
      ],
    ];

    for (const [name, path, body] of cases) {
      const res = await post(path, body);
      const id = res.headers.get(REQUEST_ID_HEADER);
      expect(id, `${name} (${res.status}) missing ${REQUEST_ID_HEADER}`).not.toBeNull();
      expect(id, `${name} request id not a UUID`).toMatch(UUID_PATTERN);
    }

    // Short-circuited auth failure (401) still carries an id — the
    // middleware stamps it even when no handler ran.
    const unauth = await post(
      "/v1/chat/completions",
      { model: "request-id-model", messages: [{ role: "user", content: "x" }] },
      UNKNOWN_PLAINTEXT,
    );
    expect(unauth.status).toBe(401);
    const unauthId = unauth.headers.get(REQUEST_ID_HEADER);
    expect(unauthId, "401 envelope missing request id").not.toBeNull();
    expect(unauthId).toMatch(UUID_PATTERN);
  });
});
