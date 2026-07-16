import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// Smoke: the recommended standalone configuration path. The gateway
// boots from a declarative resources.yaml (`resources_file`) — no etcd,
// no Admin API writes — and the file-defined resources serve a chat
// round-trip end-to-end. File mode is synchronous at boot, so there is
// no propagation wait anywhere in this suite: the first request must
// already serve.
//
// The Admin API write path stays deliberately covered during its
// deprecation window by the seed-vs-admin characterization case and
// the cases marked "Deliberately seeds via the Admin API".
//
// v3 self-hosted CP wire (§9A.7B.4): the resources file stores SHA-256
// of the plaintext bearer, never the plaintext itself. The gateway
// hashes incoming `Bearer <plaintext>` and looks the key up by hash.
// Keep this helper inline so the test independently re-derives the
// hash the same way `aisix_core::ApiKey::hash_bearer` does on the
// Rust side — divergence between the two is the bug we want to catch.
const CALLER_PLAINTEXT = "sk-smoke-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

function smokeResources(upstreamBase: string): string {
  return `
_format_version: "1"
provider_keys:
  - display_name: smoke-openai
    provider: openai
    api_key: sk-mock
    # The OpenAI bridge appends \`/chat/completions\`, so the api_base
    # already needs the \`/v1\` segment to land on \`/v1/chat/completions\`.
    api_base: ${upstreamBase}/v1
models:
  - display_name: smoke-gpt
    provider: openai
    model_name: gpt-4o-mini
    provider_key: smoke-openai
api_keys:
  - display_name: smoke-caller
    key_hash: ${CALLER_KEY_HASH}
    allowed_models: ["smoke-gpt"]
`;
}

describe("smoke: file-based resources → proxy read", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    upstream = await startOpenAiUpstream();
    app = await spawnApp({ resourcesFile: smokeResources(upstream.baseUrl) });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("a Model + ApiKey defined in resources.yaml are visible to /v1/models", async () => {
    if (!app || !upstream) throw new Error("setup failed");

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const { status, body } = await proxy.listModels();
    expect(status).toBe(200);
    expect(body).toMatchObject({
      object: "list",
      data: expect.arrayContaining([expect.objectContaining({ id: "smoke-gpt" })]),
    });
  });

  test("a chat completion forwards to the mock upstream", async () => {
    if (!app || !upstream) throw new Error("setup failed");

    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const baseline = upstream.receivedRequests.length;
    const { status, body } = await proxy.chat({
      model: "smoke-gpt",
      messages: [{ role: "user", content: "hello" }],
    });

    if (status !== 200) {
      throw new Error(
        `chat returned ${status}: ${JSON.stringify(body)}\n  upstream paths: ${JSON.stringify(upstream.receivedRequests.map((r) => r.path))}`,
      );
    }
    expect(body).toMatchObject({
      object: "chat.completion",
      choices: expect.arrayContaining([
        expect.objectContaining({
          message: expect.objectContaining({ role: "assistant" }),
        }),
      ]),
    });

    // Test call hit the upstream exactly once at the OpenAI Chat
    // Completions path. `some()` would let a regression that double-
    // fires (or short-circuits and leaks through a stray route)
    // silently pass.
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(testCalls).toHaveLength(1);
    expect(testCalls[0]?.method).toBe("POST");
  });
});
