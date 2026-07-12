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

// E2E regression for #597: Claude Code/cc-switch send a non-spec
// `role: "system"` entry inside `messages[]` on /v1/messages. The
// inbound parser used to reject the whole request with
//   400 "messages[i] role \"system\" is not 'user' or 'assistant'"
// when the model routed to an OpenAI-protocol upstream. Post-fix the
// gateway keeps the turn as a native OpenAI system message.

const CALLER_PLAINTEXT = "sk-issue-597-system-role";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("anthropic /v1/messages with system role in messages[] (#597)", () => {
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
        id: "cmpl-issue597",
        object: "chat.completion",
        created: 1765000000,
        model: "deepseek-chat",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "Bonjour!" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
      },
    });
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "issue597-pk",
      secret: "sk-deepseek-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    // Mirrors the customer setup from #597: Claude CLI talking to a
    // DeepSeek/OpenAI-protocol model through the Anthropic endpoint.
    await seed.createModel({
      display_name: "issue597-model",
      provider: "deepseek",
      model_name: "deepseek-chat",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["issue597-model"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  async function postMessages(): Promise<Response> {
    return fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": CALLER_PLAINTEXT,
        "user-agent": "claude-cli/2.1.118 (external, cli)",
      },
      body: JSON.stringify({
        model: "issue597-model",
        max_tokens: 200,
        messages: [
          { role: "user", content: "hi" },
          { role: "system", content: "respond in French" },
          { role: "user", content: "hello again" },
        ],
      }),
    });
  }

  test("system turn inside messages[] is preserved, not a 400", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    // Wait until the model/key config reached the DP. A pre-fix binary
    // keeps answering 400 here, so distinguish "config not propagated"
    // (404/401) from the regression itself.
    await waitConfigPropagation(async () => {
      try {
        const probe = await postMessages();
        return probe.status !== 404 && probe.status !== 401 && probe.status !== 403;
      } catch {
        return false;
      }
    });
    upstream.receivedRequests.length = 0;

    const resp = await postMessages();
    expect(resp.status).toBe(200);
    const body = (await resp.json()) as {
      type: string;
      content: Array<{ type: string; text: string }>;
    };
    expect(body.type).toBe("message");
    expect(body.content[0]?.text).toBe("Bonjour!");

    // The translated upstream request must keep the system turn in
    // place — dropping it silently (LiteLLM behavior) loses content.
    const sent = upstream.receivedRequests.find((r) =>
      r.path.endsWith("/chat/completions"),
    );
    expect(sent).toBeDefined();
    const sentBody = JSON.parse(sent!.body) as {
      messages: Array<{ role: string; content: string }>;
    };
    expect(sentBody.messages).toEqual([
      { role: "user", content: "hi" },
      { role: "system", content: "respond in French" },
      { role: "user", content: "hello again" },
    ]);
  });
});
