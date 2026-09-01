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

// E2E: the client-facing `model` field echoes the alias the CALLER
// addressed, on every endpoint that answers with one.
//
// The scenario every case here is built on is the one a gateway alias
// exists for: the operator configures `model_name` (what to ask the
// provider for) and the provider answers with a DIFFERENT id — a dated
// snapshot, or a name it remaps server-side. That is the ordinary case,
// not an edge: ask OpenAI for `gpt-4o-mini` and it answers
// `gpt-4o-mini-2024-07-18`.
//
// Every upstream below therefore reports `UPSTREAM_REPORTED_MODEL`,
// which matches NEITHER the caller's alias NOR the configured
// `model_name`. A gateway that echoes the upstream's document, or that
// only restamps when the upstream happened to repeat the configured
// name back, fails these assertions; one that echoes the alias passes.
//
// Coverage is per endpoint AND per response form, because the two are
// produced by different code on the native passthrough paths: a
// buffered JSON body is restamped once, while a streamed one carries
// the field inside SSE frames that are relayed as bytes.

const CALLER_PLAINTEXT = "sk-model-echo-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

/** What the operator configured as the upstream model id. */
const CONFIGURED_MODEL_NAME = "provider-model-v1";
/** What the provider actually answers with — deliberately neither of the other two. */
const UPSTREAM_REPORTED_MODEL = "provider-model-v1-20260101";

const HEADERS = {
  authorization: `Bearer ${CALLER_PLAINTEXT}`,
  "content-type": "application/json",
};

/**
 * Assert on the whole payload, not just the `model` field: a stream that
 * silently dropped or reordered frames while restamping would otherwise
 * pass. `text` is the concatenated response body.
 */
function expectAliasNotUpstreamId(text: string, alias: string): void {
  expect(text).toContain(`"model":"${alias}"`);
  expect(text).not.toContain(UPSTREAM_REPORTED_MODEL);
  expect(text).not.toContain(`"model":"${CONFIGURED_MODEL_NAME}"`);
}

describe("client-facing model echo e2e: the response names what the caller asked for", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  /**
   * Seed a provider key + model, then the caller key LAST and gate on it
   * authenticating — that single condition implies the whole seed set is
   * in the snapshot (see this directory's AGENTS.md).
   */
  async function seedAlias(
    alias: string,
    upstream: OpenAiUpstream,
    opts: { provider?: string; adapter?: string; modelName?: string } = {},
  ): Promise<void> {
    const pk = await seed!.createProviderKey({
      display_name: `pk-${alias}`,
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
      provider: opts.provider ?? "openai",
      adapter: opts.adapter ?? "openai",
    });
    await seed!.createModel({
      display_name: alias,
      provider: opts.provider ?? "openai",
      model_name: opts.modelName ?? CONFIGURED_MODEL_NAME,
      provider_key_id: pk.id,
    });
    await seed!.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      const ok = r.status === 200;
      const body = ok ? ((await r.json()) as { data?: Array<{ id?: string }> }) : undefined;
      if (!ok) await r.text();
      return !!body?.data?.some((m) => m.id === alias);
    });
  }

  test("/v1/messages native passthrough: buffered response echoes the alias", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_echo_01",
        type: "message",
        role: "assistant",
        model: UPSTREAM_REPORTED_MODEL,
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 4, output_tokens: 2 },
      },
    });
    upstreams.push(upstream);
    await seedAlias("echo-messages", upstream, {
      provider: "anthropic",
      adapter: "anthropic",
    });

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "echo-messages",
        max_tokens: 16,
        messages: [{ role: "user", content: "say ok" }],
      }),
    });
    expect(res.status).toBe(200);
    expectAliasNotUpstreamId(await res.text(), "echo-messages");
  });

  test("/v1/messages native passthrough: streamed message_start echoes the alias", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    // Written as verbatim frames so the assertion is against the real
    // Anthropic wire shape: `model` is nested under `message`, and it
    // appears on `message_start` only.
    const upstream = await startOpenAiUpstream({
      rawStreamFrames: [
        `event: message_start\ndata: {"type":"message_start","message":{"id":"msg_echo_02","type":"message","role":"assistant","model":"${UPSTREAM_REPORTED_MODEL}","content":[],"usage":{"input_tokens":4,"output_tokens":0}}}\n\n`,
        `event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n`,
        `event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n`,
        `event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n`,
        `event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n`,
        `event: message_stop\ndata: {"type":"message_stop"}\n\n`,
      ],
      eventDelayMs: 5,
    });
    upstreams.push(upstream);
    await seedAlias("echo-messages-stream", upstream, {
      provider: "anthropic",
      adapter: "anthropic",
    });

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "echo-messages-stream",
        max_tokens: 16,
        stream: true,
        messages: [{ role: "user", content: "say ok" }],
      }),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    expectAliasNotUpstreamId(text, "echo-messages-stream");
    // The rest of the stream is relayed intact: every frame the upstream
    // wrote is still there, in order, and the deltas are untouched.
    const positions = [
      "message_start",
      "content_block_start",
      "content_block_delta",
      "content_block_stop",
      "message_delta",
      "message_stop",
    ].map((t) => text.indexOf(`"type":"${t}"`));
    expect(positions.every((p) => p >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
    expect(text).toContain('"text":"ok"');
    expect(text).toContain('"output_tokens":2');
  });

  test("/v1/responses native passthrough: buffered response echoes the alias", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "resp_echo_01",
        object: "response",
        created_at: Math.floor(Date.now() / 1000),
        status: "completed",
        model: UPSTREAM_REPORTED_MODEL,
        output: [
          {
            id: "msg_echo_r1",
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "ok" }],
          },
        ],
        usage: { input_tokens: 4, output_tokens: 2, total_tokens: 6 },
      },
    });
    upstreams.push(upstream);
    await seedAlias("echo-responses", upstream);

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({ model: "echo-responses", input: "say ok" }),
    });
    expect(res.status).toBe(200);
    expectAliasNotUpstreamId(await res.text(), "echo-responses");
  });

  test("/v1/responses native passthrough: every streamed snapshot frame echoes the alias", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    // The Responses stream names the model on each snapshot event, not
    // just the first: a restamp that only fixed `response.created` would
    // still hand the upstream id back on `response.completed`, which is
    // the event SDKs build their final Response object from.
    const snapshot = (status: string) =>
      `"id":"resp_echo_02","object":"response","status":"${status}","model":"${UPSTREAM_REPORTED_MODEL}"`;
    const upstream = await startOpenAiUpstream({
      rawStreamFrames: [
        `event: response.created\ndata: {"type":"response.created","response":{${snapshot("in_progress")}}}\n\n`,
        `event: response.in_progress\ndata: {"type":"response.in_progress","response":{${snapshot("in_progress")}}}\n\n`,
        `event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"ok"}\n\n`,
        `event: response.completed\ndata: {"type":"response.completed","response":{${snapshot("completed")},"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}\n\n`,
        `data: [DONE]\n\n`,
      ],
      eventDelayMs: 5,
    });
    upstreams.push(upstream);
    await seedAlias("echo-responses-stream", upstream);

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "echo-responses-stream",
        input: "say ok",
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    expectAliasNotUpstreamId(text, "echo-responses-stream");
    // All THREE snapshot frames, not just the first.
    expect(text.split(`"model":"echo-responses-stream"`).length - 1).toBe(3);
    // Relay integrity: the delta and the terminal sentinel survive.
    expect(text).toContain('"delta":"ok"');
    expect(text).toContain('"total_tokens":6');
    expect(text.trimEnd().endsWith("data: [DONE]")).toBe(true);
  });

  test("/v1/completions echoes the alias", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl_echo_01",
        object: "text_completion",
        created: Math.floor(Date.now() / 1000),
        model: UPSTREAM_REPORTED_MODEL,
        choices: [{ index: 0, text: "ok", finish_reason: "stop" }],
        usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 },
      },
    });
    upstreams.push(upstream);
    await seedAlias("echo-completions", upstream);

    const res = await fetch(`${app.proxyUrl}/v1/completions`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({ model: "echo-completions", prompt: "say ok" }),
    });
    expect(res.status).toBe(200);
    expectAliasNotUpstreamId(await res.text(), "echo-completions");
  });

  test("/v1/videos: the poll echoes the name the caller submitted under, not the wildcard row's", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    // A wildcard row serves many caller-minted names, so the row's own
    // `display_name` (`wan-echo/*`) is NOT what the caller addressed. The
    // submit response already echoed the caller's name; the poll answers
    // about the same job and must not rename it mid-flight.
    // Static (not scripted): submit and poll parse the same DashScope
    // envelope, and the readiness probe would otherwise consume a step.
    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        output: { task_id: "task-echo-01", task_status: "RUNNING" },
        request_id: "req-echo-01",
      },
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "pk-echo-videos",
      secret: "sk-mock-dashscope",
      api_base: upstream.baseUrl,
      provider: "alibaba",
    });
    await seed.createModel({
      display_name: "wan-echo/*",
      provider: "alibaba",
      model_name: "wan-echo-upstream",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    // A wildcard row is deliberately absent from `/v1/models`, so the gate
    // is the caller key authenticating there — seeded last, so that single
    // condition implies the model and provider key landed too (see this
    // directory's AGENTS.md). It must NOT be a submit, which is behavior
    // this test asserts: a broken submit would then surface as a 30s
    // propagation timeout instead of the assertion that names the defect.
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: HEADERS.authorization },
      });
      await r.text();
      return r.status === 200;
    });

    const created = await fetch(`${app.proxyUrl}/v1/videos`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({ model: "wan-echo/turbo", prompt: "a cardboard city" }),
    });
    expect(created.status).toBe(200);
    const video = (await created.json()) as { id?: string; model?: unknown };
    expect(video.model).toBe("wan-echo/turbo");

    const polled = await fetch(`${app.proxyUrl}/v1/videos/${video.id}`, {
      method: "GET",
      headers: { authorization: HEADERS.authorization },
      redirect: "manual",
    });
    expect(polled.status).toBe(200);
    const job = (await polled.json()) as { model?: unknown; id?: unknown };
    expect(job.id).toBe(video.id);
    // The failure this pins: the poll used to answer with the row's
    // `display_name`, renaming the caller's job to `wan-echo/*`.
    expect(job.model).toBe("wan-echo/turbo");

    // The alias is decoded from the id the CLIENT holds, so a forged one
    // must not be echoed back as though the gateway had attested it. Mint an
    // id for the same row carrying a name the row does not serve: the poll
    // falls back to the row's own name instead of parroting the forgery.
    const [entryId] = Buffer.from(video.id!, "base64url")
      .toString("utf8")
      .split(":");
    const forged = Buffer.from(
      `${entryId}:${Buffer.from("not-a-name-this-row-serves").toString("base64url")}:task-echo-01`,
    ).toString("base64url");
    const forgedPoll = await fetch(`${app.proxyUrl}/v1/videos/${forged}`, {
      method: "GET",
      headers: { authorization: HEADERS.authorization },
      redirect: "manual",
    });
    expect(forgedPoll.status).toBe(200);
    const forgedJob = (await forgedPoll.json()) as { model?: unknown };
    expect(forgedJob.model).toBe("wan-echo/*");
  });

  test("/v1/embeddings echoes the alias", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await startOpenAiUpstream({
      nonStreamBody: {
        object: "list",
        model: UPSTREAM_REPORTED_MODEL,
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
        usage: { prompt_tokens: 4, total_tokens: 4 },
      },
    });
    upstreams.push(upstream);
    await seedAlias("echo-embeddings", upstream);

    const res = await fetch(`${app.proxyUrl}/v1/embeddings`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({ model: "echo-embeddings", input: "say ok" }),
    });
    expect(res.status).toBe(200);
    expectAliasNotUpstreamId(await res.text(), "echo-embeddings");
  });
});

// A block-capable output guardrail must see the whole response before any of
// it reaches the caller, so a streamed `/v1/responses` request with one
// attached is BUFFERED and returned as a single body — never touching the
// live relay that splices frame-by-frame. That branch needs the same restamp,
// or attaching a guardrail silently changes which model name the caller is
// told. Its own app: an env-scoped attachment applies to every request, so it
// cannot share the suite above.
describe("client-facing model echo e2e: a buffering output guardrail keeps the alias", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    // BOTH tests here need the hold-back policy in force, so it is seeded
    // once for the app rather than by whichever test happens to run first.
    // `keyword` on the output hook is block-capable, which is what makes the
    // gateway buffer a streamed response instead of forwarding it live. The
    // pattern deliberately never matches: the point is the buffered delivery
    // path, not a block.
    await seed.createGuardrail({
      name: "gr-model-echo-holdback",
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [{ kind: "literal", value: "ABSOLUTELYFORBIDDENWORD" }],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  test("/v1/responses streamed behind a block-capable output guardrail still echoes the alias", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const snapshot = (status: string) =>
      `"id":"resp_guarded","object":"response","status":"${status}","model":"${UPSTREAM_REPORTED_MODEL}"`;
    const upstream = await startOpenAiUpstream({
      rawStreamFrames: [
        `event: response.created\ndata: {"type":"response.created","response":{${snapshot("in_progress")}}}\n\n`,
        `event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"perfectly fine text"}\n\n`,
        `event: response.completed\ndata: {"type":"response.completed","response":{${snapshot("completed")},"usage":{"input_tokens":4,"output_tokens":3,"total_tokens":7}}}\n\n`,
        `data: [DONE]\n\n`,
      ],
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "pk-echo-guarded",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "echo-guarded",
      provider: "openai",
      model_name: CONFIGURED_MODEL_NAME,
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: HEADERS.authorization },
      });
      await r.text();
      return r.status === 200;
    });

    const res = await fetch(`${app.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "echo-guarded",
        input: "say something fine",
        stream: true,
      }),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    expectAliasNotUpstreamId(text, "echo-guarded");
    // Both snapshot frames, and the held stream released intact.
    expect(text.split('"model":"echo-guarded"').length - 1).toBe(2);
    expect(text).toContain('"delta":"perfectly fine text"');
    expect(text.trimEnd().endsWith("data: [DONE]")).toBe(true);
  });

  // The same hold-back policy on `/v1/messages`. Only COMPLETE frames feed
  // the text the output guardrail scans, so an upstream that dies mid-frame
  // leaves a tail nothing ever inspected. Releasing it after the scan would
  // be a way around the very check hold-back exists to apply, so it is
  // dropped — the client could not have parsed an unterminated frame anyway.
  test("/v1/messages hold-back drops an unterminated tail instead of releasing it unscanned", async (ctx) => {
    if (!etcdReachable || !app || !seed) return void ctx.skip();

    const upstream = await startOpenAiUpstream({
      rawStreamFrames: [
        `event: message_start\ndata: {"type":"message_start","message":{"id":"msg_tail","type":"message","role":"assistant","model":"${UPSTREAM_REPORTED_MODEL}","content":[],"usage":{"input_tokens":4,"output_tokens":0}}}\n\n`,
        `event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"delivered text"}}\n\n`,
        // No terminator: the stream ends mid-frame.
        `event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"NEVERSCANNEDTAIL"`,
      ],
    });
    upstreams.push(upstream);

    const pk = await seed.createProviderKey({
      display_name: "pk-echo-tail",
      secret: "sk-mock",
      api_base: upstream.baseUrl,
      provider: "anthropic",
      adapter: "anthropic",
    });
    await seed.createModel({
      display_name: "echo-tail",
      provider: "anthropic",
      model_name: CONFIGURED_MODEL_NAME,
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    await waitConfigPropagation(async () => {
      const r = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: HEADERS.authorization },
      });
      await r.text();
      return r.status === 200;
    });

    const res = await fetch(`${app.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: HEADERS,
      body: JSON.stringify({
        model: "echo-tail",
        max_tokens: 16,
        stream: true,
        messages: [{ role: "user", content: "say ok" }],
      }),
    });
    expect(res.status).toBe(200);
    const text = await res.text();
    // The scanned frames are delivered, with the alias restamped...
    expect(text).toContain('"model":"echo-tail"');
    expect(text).toContain('"text":"delivered text"');
    // ...and the unterminated, never-scanned tail is not.
    expect(text).not.toContain("NEVERSCANNEDTAIL");
  });
});
