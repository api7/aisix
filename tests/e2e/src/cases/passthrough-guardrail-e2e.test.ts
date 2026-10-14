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

// E2E for #911 finding [6]: the raw /passthrough/:provider/*rest tunnel
// forwarded requests verbatim with NO guardrail scanning, so a tenant that
// configured a content/DLP guardrail could bypass it by routing traffic
// through passthrough. Following LiteLLM's passthrough default, the gateway
// now scans the whole request AND response body as text against the resolved
// chain. This drives both directions:
//   - INPUT: a passthrough request whose body carries a forbidden word is
//     blocked 422 before the upstream is ever called.
//   - OUTPUT: a clean request whose upstream reply carries a forbidden word is
//     blocked 422 and the word never reaches the caller.
//   - STREAMING OUTPUT: on an SSE relay the keyword chain folds to the
//     fail-closed BufferFull hold-back — held frames are dropped on a hit
//     and the caller sees only the error frame, never the forbidden text.

const CALLER_PLAINTEXT = "sk-pt-gr-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const FORBIDDEN_INPUT = "forbiddenprompt";
const FORBIDDEN_OUTPUT = "leakedsecret";
const INPUT_GUARDRAIL = "pt-gr-input-keyword";
const OUTPUT_GUARDRAIL = "pt-gr-output-keyword";

describe("passthrough guardrail (#911 [6])", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let sseUpstream: OpenAiUpstream | undefined;
  let hybridUpstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // Upstream reply carries the forbidden OUTPUT word; the caller's request
    // body is innocent, so the forbidden content originates from the model.
    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-leak",
        object: "chat.completion",
        created: 0,
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: `here it is: ${FORBIDDEN_OUTPUT}` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
      },
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "pt-gr-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "pt-gr",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createPassthroughRoute({
      name: "pt-gr-tunnel",
      path_prefix: "/passthrough/openai",
      target_url: `${upstream.baseUrl}/v1`,
      provider_key_id: pk.id,
    });
    await seed.createGuardrail({
      name: INPUT_GUARDRAIL,
      enabled: true,
      hook_point: "input",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_INPUT }],
    });
    await seed.createGuardrail({
      name: OUTPUT_GUARDRAIL,
      enabled: true,
      hook_point: "output",
      kind: "keyword",
      patterns: [{ kind: "literal", value: FORBIDDEN_OUTPUT }],
    });

    sseUpstream = await startOpenAiUpstream({
      streamEvents: [
        JSON.stringify({ choices: [{ index: 0, delta: { content: "prelude " } }] }),
        JSON.stringify({ choices: [{ index: 0, delta: { content: FORBIDDEN_OUTPUT } }] }),
        "[DONE]",
      ],
    });
    await seed.createPassthroughRoute({
      name: "pt-gr-sse-tunnel",
      path_prefix: "/pt-gr-sse",
      target_url: sseUpstream.baseUrl,
      provider_key_id: pk.id,
      protocol: "openai_chat",
    });

    // A reply naming two envelopes at once: clean Chat `choices` beside an
    // Anthropic message body whose `content` carries the forbidden text.
    hybridUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "msg_hybrid",
        object: "chat.completion",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "all clear" },
            finish_reason: "stop",
          },
        ],
        type: "message",
        role: "assistant",
        content: [{ type: "text", text: `here it is: ${FORBIDDEN_OUTPUT}` }],
      },
    });
    await seed.createPassthroughRoute({
      name: "pt-gr-hybrid-tunnel",
      path_prefix: "/pt-gr-hybrid",
      target_url: hybridUpstream.baseUrl,
      provider_key_id: pk.id,
    });

    // Caller key LAST: etcd applies writes in revision order, so a key
    // seeded after every resource it must reach doubles as the
    // propagation barrier for all of them.
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["pt-gr"],
      allowed_routes: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await sseUpstream?.close();
    await hybridUpstream?.close();
  });

  function passthrough(body: unknown): Promise<Response> {
    return fetch(`${app!.proxyUrl}/passthrough/openai/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
  }

  test("upstream-emitted forbidden text is blocked by the output guardrail", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Output guardrails fire after upstream dispatch, so readiness is signaled
    // by the 422-on-blocked-response itself (a 200 means the chain isn't
    // loaded yet and the leaked content was forwarded). The request body here
    // is clean, so only the OUTPUT guardrail can block it.
    await waitConfigPropagation(async () => {
      const res = await passthrough({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: "innocent" }],
      });
      await res.text();
      return res.status === 422;
    });

    const res = await passthrough({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "innocent" }],
    });
    expect(res.status).toBe(422);
    const bodyText = await res.text();
    // The forbidden word MUST NOT reach the caller.
    expect(bodyText).not.toContain(FORBIDDEN_OUTPUT);
    const body = JSON.parse(bodyText) as { error?: { type?: unknown; message?: unknown } };
    expect(body.error?.type).toBe("content_filter");
    expect(String(body.error?.message)).toContain(`guardrail '${OUTPUT_GUARDRAIL}'`);
  });

  test("forbidden request body is blocked by the input guardrail before the upstream is called", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // Self-synchronize on the INPUT guardrail loading (independent of the
    // output-block test above): a forbidden-input request is blocked 422 only
    // once the chain is live. A blocked request never reaches the upstream, so
    // polling here doesn't perturb the hit-count assertion below.
    await waitConfigPropagation(async () => {
      const probe = await passthrough({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content: `probe ${FORBIDDEN_INPUT}` }],
      });
      await probe.text();
      return probe.status === 422;
    });

    const hitsBefore = upstream.receivedRequests.length;

    const res = await passthrough({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: `please ${FORBIDDEN_INPUT} now` }],
    });
    expect(res.status).toBe(422);
    const bodyText = await res.text();
    const body = JSON.parse(bodyText) as { error?: { type?: unknown; message?: unknown } };
    expect(body.error?.type).toBe("content_filter");
    expect(String(body.error?.message)).toContain(`guardrail '${INPUT_GUARDRAIL}'`);

    // Input guardrails run BEFORE the upstream call — a blocked request must
    // not reach the provider.
    expect(upstream.receivedRequests.length - hitsBefore).toBe(0);
  });

  test("streaming output block: held frames are dropped, only the error frame reaches the caller", async (ctx) => {
    if (!etcdReachable || !app || !sseUpstream) {
      ctx.skip();
      return;
    }

    const call = () =>
      fetch(`${app!.proxyUrl}/pt-gr-sse/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ model: "gpt-4o", stream: true, messages: [] }),
      });

    await waitConfigPropagation(async () => {
      try {
        const r = await call();
        const ok = r.status === 200;
        await r.text();
        return ok;
      } catch {
        return false;
      }
    });

    const res = await call();
    expect(res.status).toBe(200);
    const text = await res.text();
    // The keyword chain's default policy is fail-closed BufferFull: the
    // forbidden delta (and everything held with it) must never be
    // relayed; the caller sees the SSE error frame instead.
    expect(text).not.toContain(FORBIDDEN_OUTPUT);
    expect(text).not.toContain("prelude");
    expect(text).toContain("event: error");
    expect(text).toContain("content_filter");
  }, 30_000);

  test("the input guardrail reads an Anthropic Messages body wherever it reads the OpenAI one", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const send = (path: string, body: unknown, headers: Record<string, string> = {}) =>
      fetch(`${app!.proxyUrl}/passthrough/openai/v1/${path}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
          ...headers,
        },
        body: JSON.stringify(body),
      });
    const anthropic = (body: Record<string, unknown>) =>
      send(
        "messages",
        { model: "claude-sonnet-4-5", max_tokens: 64, ...body },
        { "anthropic-version": "2023-06-01" },
      );
    const openai = (messages: unknown[]) =>
      send("chat/completions", { model: "gpt-4o-mini", messages });

    // Gate on the input chain being live (a blocked request never reaches
    // the upstream, so polling here is side-effect free).
    await waitConfigPropagation(async () => {
      const probe = await openai([{ role: "user", content: FORBIDDEN_INPUT }]);
      await probe.text();
      return probe.status === 422;
    });

    // The same forbidden text in each slot a caller can put it, on both
    // envelopes: the system prompt, a tool's result, and a replayed tool
    // call's arguments.
    const pairs: [string, () => Promise<Response>, () => Promise<Response>][] = [
      [
        "system prompt",
        () =>
          anthropic({
            system: [{ type: "text", text: `be ${FORBIDDEN_INPUT}` }],
            messages: [{ role: "user", content: "hi" }],
          }),
        () =>
          openai([
            { role: "system", content: `be ${FORBIDDEN_INPUT}` },
            { role: "user", content: "hi" },
          ]),
      ],
      [
        "tool result",
        () =>
          anthropic({
            messages: [
              {
                role: "assistant",
                content: [{ type: "tool_use", id: "toolu_1", name: "read", input: {} }],
              },
              {
                role: "user",
                content: [
                  {
                    type: "tool_result",
                    tool_use_id: "toolu_1",
                    content: [{ type: "text", text: `file says ${FORBIDDEN_INPUT}` }],
                  },
                ],
              },
            ],
          }),
        () =>
          openai([
            {
              role: "assistant",
              content: null,
              tool_calls: [
                { id: "call_1", type: "function", function: { name: "read", arguments: "{}" } },
              ],
            },
            { role: "tool", tool_call_id: "call_1", content: `file says ${FORBIDDEN_INPUT}` },
          ]),
      ],
      [
        "tool call arguments",
        () =>
          anthropic({
            messages: [
              {
                role: "assistant",
                content: [
                  {
                    type: "tool_use",
                    id: "toolu_2",
                    name: "write",
                    input: { text: FORBIDDEN_INPUT },
                  },
                ],
              },
              {
                role: "user",
                content: [{ type: "tool_result", tool_use_id: "toolu_2", content: "ok" }],
              },
            ],
          }),
        () =>
          openai([
            {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_2",
                  type: "function",
                  function: {
                    name: "write",
                    arguments: JSON.stringify({ text: FORBIDDEN_INPUT }),
                  },
                },
              ],
            },
            { role: "tool", tool_call_id: "call_2", content: "ok" },
          ]),
      ],
    ];

    const hitsBefore = upstream.receivedRequests.length;
    for (const [slot, viaAnthropic, viaOpenai] of pairs) {
      for (const [envelope, call] of [
        ["anthropic", viaAnthropic],
        ["openai", viaOpenai],
      ] as const) {
        const res = await call();
        const body = (await res.json()) as { error?: { message?: unknown } };
        expect(res.status, `${envelope} ${slot}`).toBe(422);
        expect(String(body.error?.message), `${envelope} ${slot}`).toContain(
          `guardrail '${INPUT_GUARDRAIL}'`,
        );
      }
    }
    expect(upstream.receivedRequests.length - hitsBefore).toBe(0);
  });

  test("a reply naming both a Chat and an Anthropic envelope is unscannable, so the fail-closed output guardrail refuses it", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const call = () =>
      fetch(`${app!.proxyUrl}/pt-gr-hybrid/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          messages: [{ role: "user", content: "innocent" }],
        }),
      });

    // Gate on the caller key (seeded last): a 401 is the only transient.
    await waitConfigPropagation(async () => {
      const r = await call();
      await r.text();
      return r.status !== 401;
    });

    const res = await call();
    const bodyText = await res.text();
    // Each selector reads only its own carrier, so scanning the Chat one
    // alone would let the Anthropic `content` through unscanned.
    expect(res.status).toBe(422);
    expect(bodyText).not.toContain(FORBIDDEN_OUTPUT);
  });
});
