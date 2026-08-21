import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: non-UTF-8 `prompt` multipart fields are rejected 400 on every
// multipart surface (api7/aisix#1016).
//
// The input-guardrail scan and mask passes read `prompt` as UTF-8 and
// used to SKIP a field that failed to decode, while the rebuilt form
// forwarded the original bytes verbatim — one invalid prefix byte
// smuggled text past a keyword/DLP guardrail. The gateway now fails
// closed: an invalid `prompt` answers 400 before the guardrail pass
// and before any upstream contact, unconditionally (no guardrail needs
// to be configured), matching the JSON endpoints' own posture (serde
// rejects invalid UTF-8 bodies).
//
// One journey per wired route: /v1/audio/transcriptions,
// /v1/audio/translations, /v1/images/edits.

const CALLER_PLAINTEXT = "sk-non-utf8-prompt-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** Raw multipart body with an invalid-UTF-8 `prompt` part (0xFF prefix
 * ahead of otherwise-normal text) plus the file part each route needs. */
function invalidPromptBody(model: string, fileField: string): Blob {
  const enc = new TextEncoder();
  const head = enc.encode(
    `--b\r\nContent-Disposition: form-data; name="model"\r\n\r\n${model}\r\n` +
      `--b\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n`,
  );
  const invalid = new Uint8Array([0xff]);
  const tail = enc.encode(
    `SMUGGLED\r\n--b\r\nContent-Disposition: form-data; name="${fileField}"; filename="a.bin"\r\n` +
      `Content-Type: application/octet-stream\r\n\r\nFAKEBYTES\r\n--b--\r\n`,
  );
  const body = new Uint8Array(head.length + invalid.length + tail.length);
  body.set(head, 0);
  body.set(invalid, head.length);
  body.set(tail, head.length + invalid.length);
  // Blob, not the bare Uint8Array: fetch's BodyInit typing wants it,
  // and the byte payload passes through unchanged.
  return new Blob([body]);
}

describe("non-UTF-8 prompt rejected on every multipart surface (#1016)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: { text: "should never be reached" },
    });
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "non-utf8-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "non-utf8-audio",
      provider: "openai",
      model_name: "gpt-4o-transcribe",
      provider_key_id: pk.id,
    });
    await seed.createModel({
      display_name: "non-utf8-edit",
      provider: "openai",
      model_name: "gpt-image-2",
      provider_key_id: pk.id,
    });
    // Caller key last, so the readiness gate on it implies the whole
    // seed set (harness AGENTS.md).
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["non-utf8-audio", "non-utf8-edit"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  const cases: Array<[route: string, model: string, fileField: string]> = [
    ["/v1/audio/transcriptions", "non-utf8-audio", "file"],
    ["/v1/audio/translations", "non-utf8-audio", "file"],
    ["/v1/images/edits", "non-utf8-edit", "image"],
  ];

  for (const [route, model, fileField] of cases) {
    test(`${route}: invalid prompt bytes answer 400, upstream untouched`, async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const probe = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
      await waitConfigPropagation(
        async () => (await probe.listModels()).status === 200,
      );

      const baseline = upstream.receivedRequests.length;
      const res = await fetch(`${app.proxyUrl}${route}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "multipart/form-data; boundary=b",
        },
        body: invalidPromptBody(model, fileField),
      });

      expect(res.status).toBe(400);
      const body = (await res.json()) as {
        error?: { type?: string; message?: string };
      };
      expect(body.error?.type).toBe("invalid_request_error");
      expect(body.error?.message).toContain("UTF-8");
      // Hard contract: the smuggled bytes never reach the provider.
      expect(upstream.receivedRequests.length).toBe(baseline);
    });
  }
});
