import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E (#952): an upstream error that carries no message must still reach
// the caller as an error envelope that says something.
//
// A provider (or an edge layer in front of it) can answer a non-2xx with
// no body at all, or with an error envelope whose `message` is blank.
// The caller's SDK surfaces `error.message` as the exception text, so an
// empty string leaves them with "something failed" and nothing else. The
// status and its standard reason phrase are the least the gateway knows.
//
// Every client-facing surface renders upstream errors, and they reach the
// renderer through different paths (a provider bridge, a verbatim
// forward, a cross-protocol translation), so each is driven here.

const CALLER_PLAINTEXT = "sk-empty-err-body-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("upstream error with no message → caller sees the upstream status (#952)", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const closers: Array<() => Promise<void>> = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(closers.map((c) => c()));
  });

  // A model whose upstream answers every request with `failure`. The
  // readiness gate waits for the model to be listed, so the asserted error
  // cannot be a snapshot-lag "model not found".
  async function modelFailingWith(
    name: string,
    failure: { status: number; rawErrorBody?: string; errorBody?: unknown },
  ): Promise<void> {
    const upstream = await startOpenAiUpstream(failure);
    closers.push(() => upstream.close());
    const pk = await seed!.createProviderKey({
      display_name: `${name}-pk`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed!.createModel({
      display_name: name,
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === name);
    });
  }

  async function post(path: string, body: unknown): Promise<Response> {
    return fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "x-api-key": CALLER_PLAINTEXT,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });
  }

  async function errorMessage(res: Response): Promise<string> {
    const body = (await res.json()) as { error?: { message?: unknown } };
    expect(typeof body.error?.message).toBe("string");
    return body.error!.message as string;
  }

  const skip = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app || !seed) {
      ctx.skip();
      return true;
    }
    return false;
  };

  test("chat completions: bodyless 404 names the status", async (ctx) => {
    if (skip(ctx)) return;
    await modelFailingWith("empty-err-chat-404", {
      status: 404,
      rawErrorBody: "",
    });
    const res = await post("/v1/chat/completions", {
      model: "empty-err-chat-404",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(404);
    expect(await errorMessage(res)).toBe("upstream returned 404 Not Found");
  });

  test("chat completions: an error envelope with a blank message names the status", async (ctx) => {
    if (skip(ctx)) return;
    await modelFailingWith("empty-err-chat-blank", {
      status: 400,
      errorBody: { error: { message: "", type: "invalid_request_error" } },
    });
    const res = await post("/v1/chat/completions", {
      model: "empty-err-chat-blank",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(400);
    expect(await errorMessage(res)).toBe("upstream returned 400 Bad Request");
  });

  test("chat completions: a bodyless redirect names the status and nothing else", async (ctx) => {
    if (skip(ctx)) return;
    // No Location header: a redirect the HTTP client cannot follow comes
    // back to the gateway as the upstream's answer.
    await modelFailingWith("empty-err-chat-301", {
      status: 301,
      rawErrorBody: "",
    });
    const res = await post("/v1/chat/completions", {
      model: "empty-err-chat-301",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(await errorMessage(res)).toBe(
      "upstream returned 301 Moved Permanently",
    );
  });

  test("responses (verbatim forward): bodyless 403 names the status", async (ctx) => {
    if (skip(ctx)) return;
    await modelFailingWith("empty-err-responses-403", {
      status: 403,
      rawErrorBody: "",
    });
    const res = await post("/v1/responses", {
      model: "empty-err-responses-403",
      input: "hi",
    });
    expect(res.status).toBe(403);
    expect(await errorMessage(res)).toBe("upstream returned 403 Forbidden");
  });

  test("messages (cross-provider to an OpenAI upstream): bodyless 404 names the status", async (ctx) => {
    if (skip(ctx)) return;
    await modelFailingWith("empty-err-messages-404", {
      status: 404,
      rawErrorBody: "",
    });
    const res = await post("/v1/messages", {
      model: "empty-err-messages-404",
      max_tokens: 16,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.status).toBe(404);
    expect(await errorMessage(res)).toBe("upstream returned 404 Not Found");
  });
});
