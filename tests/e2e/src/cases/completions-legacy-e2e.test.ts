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

// E2E: legacy `POST /v1/completions` endpoint.
//
// Per `docs/api-proxy.md` §4.3, the legacy OpenAI Completions
// endpoint is text-in / text-out (the pre-chat API). It uses
// `prompt: string` instead of `messages: [...]` on the request,
// and `choices[i].text` instead of `choices[i].message.content`
// on the response. Auth and error semantics match chat.
//
// Closes #151 C7 (`/v1/completions` had zero e2e coverage).
// Several real-world callers — long-running `instructor` deployments
// on text-davinci-002, legacy autocomplete tools, retrieval
// pipelines that grep stored prompts — still drive this endpoint.
// A regression that broke the request-body field-name mapping
// (prompt vs messages) or rewrote the response shape would silently
// snap every legacy caller.
//
// One contract pinned:
//
//   - The gateway forwards the caller's `{model, prompt}` body
//     verbatim to the upstream's `/v1/completions` path, with the
//     gateway's Model alias rewritten to the upstream model id and
//     the configured ProviderKey secret as the Authorization bearer.
//   - The upstream's text-completion response shape
//     `{choices: [{text, finish_reason}], usage}` passes back to
//     the caller unmodified.
//
// Reference:
//   - OpenAI legacy Completions API
//     <https://platform.openai.com/docs/api-reference/completions>
//   - `docs/api-proxy.md` §4.3

const CALLER_PLAINTEXT = "sk-completions-legacy-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const PROMPT_TEXT = "Once upon a time in a";
const COMPLETION_TEXT = " distant kingdom, there lived";

const NON_STREAM_BODY = {
  id: "cmpl-completions-legacy-1",
  object: "text_completion",
  created: Math.floor(Date.now() / 1000),
  model: "gpt-3.5-turbo-instruct",
  choices: [
    {
      text: COMPLETION_TEXT,
      index: 0,
      logprobs: null,
      finish_reason: "stop",
    },
  ],
  usage: {
    prompt_tokens: 6,
    completion_tokens: 7,
    total_tokens: 13,
  },
};

describe("legacy /v1/completions e2e: text-in / text-out passthrough", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: NON_STREAM_BODY,
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "completions-legacy-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "completions-legacy-model",
      provider: "openai",
      model_name: "gpt-3.5-turbo-instruct",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["completions-legacy-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test(
    "POST /v1/completions: prompt forwarded, text response passes back",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }

      const reqHeaders = {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      };

      // Readiness — drive the same endpoint with a distinct prompt
      // so the gate verifies the legacy completions dispatcher, not
      // just Model+key+pk visibility on a chat path.
      await waitConfigPropagation(async () => {
        try {
          const r = await fetch(`${app!.proxyUrl}/v1/completions`, {
            method: "POST",
            headers: reqHeaders,
            body: JSON.stringify({
              model: "completions-legacy-model",
              prompt: "ready-probe",
            }),
          });
          await r.text();
          return r.status === 200;
        } catch {
          return false;
        }
      });

      const baseline = upstream.receivedRequests.length;

      // Asserted call.
      const res = await fetch(`${app.proxyUrl}/v1/completions`, {
        method: "POST",
        headers: reqHeaders,
        body: JSON.stringify({
          model: "completions-legacy-model",
          prompt: PROMPT_TEXT,
          max_tokens: 32,
          temperature: 0.7,
        }),
      });

      // (1) Status 200 — endpoint is wired and reachable.
      expect(res.status).toBe(200);
      const body = (await res.json()) as {
        object?: string;
        choices?: Array<{
          text?: string;
          finish_reason?: string;
        }>;
        usage?: {
          total_tokens?: number;
        };
      };

      // (2) Response shape is text-completion (NOT chat-completion).
      //     A regression that re-routed legacy to the chat handler
      //     would surface `object: "chat.completion"` and the
      //     wrong choices shape.
      expect(body.object).toBe("text_completion");
      expect(body.choices?.[0]?.text).toBe(COMPLETION_TEXT);
      expect(body.choices?.[0]?.finish_reason).toBe("stop");
      expect(body.usage?.total_tokens).toBe(13);

      // (3) Upstream wire-shape: exactly one POST to /v1/completions
      //     with our auth + the prompt + the upstream model id.
      //     Closes the wire-shape blind spot CLAUDE.md §8 calls out.
      const sent = upstream.receivedRequests
        .slice(baseline)
        .filter((r) => r.path === "/v1/completions");
      expect(sent).toHaveLength(1);
      const sentReq = sent[0]!;
      expect(sentReq.method).toBe("POST");
      expect(sentReq.headers.authorization).toBe("Bearer sk-mock");
      const sentBody = JSON.parse(sentReq.body);
      // (4) Model alias is rewritten to the upstream model id.
      expect(sentBody.model).toBe("gpt-3.5-turbo-instruct");
      // (5) `prompt` field is preserved as-is. Catches a regression
      //     that translated legacy `prompt` into chat `messages`
      //     somewhere in the pipeline — which would change every
      //     downstream tool that grep'd request logs for the
      //     `prompt` field.
      expect(sentBody.prompt).toBe(PROMPT_TEXT);
      // Optional params survive verbatim too.
      expect(sentBody.max_tokens).toBe(32);
      expect(sentBody.temperature).toBe(0.7);
      // (6) `messages` field MUST NOT appear — legacy callers do
      //     not send it, and the gateway should not synthesize one
      //     from `prompt`.
      expect(sentBody.messages).toBeUndefined();
    },
    60_000,
  );
});
