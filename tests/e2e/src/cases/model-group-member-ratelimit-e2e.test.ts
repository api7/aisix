import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  awaitWindowHeadroom,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for AISIX-Cloud#1087: a Model Group dispatch must honor each
// TARGET's own `rate_limit`, not just the group entry's. Pre-fix the
// pre-dispatch reservation covered only the requested entry (the group,
// which carries no limits), so a member's RPM/TPM was silently ignored
// and every request kept landing on the over-limit first target.
//
// Post-fix each routing attempt reserves the target's model-scoped
// layers first (mirroring the ensemble per-sub-call reservation, #620):
// an over-limit member becomes a failed 429 attempt that fails over to
// the next target — LiteLLM's semantics, where rate-limited deployments
// are filtered from the candidate set — and a request served by a
// member commits its token cost to that member's TPM bucket.

const CALLER_PLAINTEXT = "sk-1087-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
// Readiness-probe caller allowed to access NOTHING: a chat call with it
// returns 404 while a name is absent from the snapshot and 403 once it
// propagated. The 403 fires at the ACL gate, before any rate-limit
// reservation, so probing never consumes the member quotas under test
// (and routing models never appear in /v1/models, so listing can't be
// the probe).
const PROBE_PLAINTEXT = "sk-1087-probe";
const PROBE_KEY_HASH = createHash("sha256")
  .update(PROBE_PLAINTEXT)
  .digest("hex");

function chatBody(content: string) {
  return {
    id: "cmpl-1087",
    object: "chat.completion",
    created: 0,
    model: "gpt-4o-mini",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
  };
}

function chunk(json: Record<string, unknown>): string {
  return JSON.stringify({
    id: "chatcmpl-1087",
    object: "chat.completion.chunk",
    created: 0,
    model: "gpt-4o-mini",
    ...json,
  });
}

function streamEvents(content: string, totalTokens: number): string[] {
  return [
    chunk({ choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] }),
    chunk({ choices: [{ index: 0, delta: { content }, finish_reason: null }] }),
    chunk({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }),
    chunk({
      choices: [],
      usage: {
        prompt_tokens: totalTokens - 2,
        completion_tokens: 2,
        total_tokens: totalTokens,
      },
    }),
    "[DONE]",
  ];
}

function anthropicMessageBody(text: string) {
  return {
    id: `msg_${text}`,
    type: "message",
    role: "assistant",
    content: [{ type: "text", text }],
    model: "claude-3-5-haiku-20241022",
    stop_reason: "end_turn",
    usage: { input_tokens: 5, output_tokens: 4 },
  };
}

type ChatResult = {
  status: number;
  body: {
    choices?: Array<{ message?: { content?: string } }>;
    error?: { message?: string };
  };
};

describe("model group member rate limit e2e (AISIX-Cloud#1087)", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let proxy: ProxyClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    await seed.createApiKey({
      key_hash: PROBE_KEY_HASH,
      allowed_models: ["__probe-none__"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  async function newUpstream(opts: Parameters<typeof startOpenAiUpstream>[0]): Promise<OpenAiUpstream> {
    const u = await startOpenAiUpstream(opts);
    upstreams.push(u);
    return u;
  }

  async function createOpenAiModel(
    displayName: string,
    upstream: OpenAiUpstream,
    extra: Record<string, unknown> = {},
  ): Promise<void> {
    if (!seed) throw new Error("seed client not initialized");
    const pk = await seed.createProviderKey({
      display_name: `${displayName}-pk`,
      secret: "sk-openai-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: displayName,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
      ...extra,
    });
  }

  // Readiness: probe every name with the no-access key until each
  // returns 403 (in snapshot, ACL-rejected) instead of 404 (not yet
  // propagated). Deliberately NOT a real chat call with the main caller
  // — that would burn the member's own rate-limit quota under test.
  async function waitModelsListed(names: string[]): Promise<void> {
    if (!app) throw new Error("app not initialized");
    const probe = new ProxyClient(app.proxyUrl, PROBE_PLAINTEXT);
    await waitConfigPropagation(async () => {
      for (const n of names) {
        const res = await probe.chat({
          model: n,
          messages: [{ role: "user", content: "probe" }],
        });
        if (res.status !== 403) return false;
      }
      return true;
    });
  }

  async function callGroup(model: string): Promise<ChatResult> {
    if (!proxy) throw new Error("proxy client not initialized");
    return (await proxy.chat({
      model,
      messages: [{ role: "user", content: "hello" }],
    })) as ChatResult;
  }

  function servedContent(r: ChatResult): string {
    expect(r.status).toBe(200);
    return r.body.choices?.[0]?.message?.content ?? "";
  }


  test("over-limit member fails over to the next target (RPM)", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const limited = await newUpstream({ nonStreamBody: chatBody("served-by-limited") });
    const backup = await newUpstream({ nonStreamBody: chatBody("served-by-backup") });
    await createOpenAiModel("mgrl-limited", limited, { rate_limit: { rpm: 1 } });
    await createOpenAiModel("mgrl-backup", backup);
    await seed!.createModel({
      display_name: "mgrl-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "mgrl-limited" }, { model: "mgrl-backup" }],
      },
    });
    await waitModelsListed(["mgrl-limited", "mgrl-backup", "mgrl-group"]);
    await awaitWindowHeadroom(5);

    // First call through the group lands on the first target.
    expect(servedContent(await callGroup("mgrl-group"))).toBe("served-by-limited");

    // Second call in the same minute: the first target is over its own
    // RPM=1, so dispatch must fail over to the backup. Pre-fix the
    // member's limit was never consulted and this still returned
    // "served-by-limited".
    expect(servedContent(await callGroup("mgrl-group"))).toBe("served-by-backup");
  });

  test("all members over limit surfaces as 429", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const c1 = await newUpstream({ nonStreamBody: chatBody("served-by-c1") });
    const c2 = await newUpstream({ nonStreamBody: chatBody("served-by-c2") });
    await createOpenAiModel("mgrl-c1", c1, { rate_limit: { rpm: 1 } });
    await createOpenAiModel("mgrl-c2", c2, { rate_limit: { rpm: 1 } });
    await seed!.createModel({
      display_name: "mgrl-both-limited",
      routing: {
        strategy: "failover",
        targets: [{ model: "mgrl-c1" }, { model: "mgrl-c2" }],
      },
    });
    await waitModelsListed(["mgrl-c1", "mgrl-c2", "mgrl-both-limited"]);
    await awaitWindowHeadroom(5);

    expect(servedContent(await callGroup("mgrl-both-limited"))).toBe("served-by-c1");
    expect(servedContent(await callGroup("mgrl-both-limited"))).toBe("served-by-c2");

    // Both members exhausted → the request fails with 429, not a 5xx, and
    // carries the limiter's `Retry-After` so SDK back-off still works. The
    // hint is the load-bearing half: a bare 429 makes clients retry
    // immediately against a window that has not reopened.
    const third = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "mgrl-both-limited",
        messages: [{ role: "user", content: "hello" }],
      }),
    });
    expect(third.status).toBe(429);
    const body = (await third.json()) as { error?: { message?: string } };
    expect(body.error?.message ?? "").toContain("rate limit");
    const retryAfter = Number.parseInt(third.headers.get("retry-after") ?? "", 10);
    expect(retryAfter).toBeGreaterThan(0);
    expect(retryAfter).toBeLessThanOrEqual(60);
  });

  test("winning member's TPM bucket is committed, throttling the next call", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    // Upstream reports total_tokens=8 per call (chatBody default); the
    // member's TPM=5 admits the first call (window empty at pre-commit)
    // and must reject the second (8 committed ≥ 5).
    const tpmUp = await newUpstream({ nonStreamBody: chatBody("served-by-tpm") });
    const backup = await newUpstream({ nonStreamBody: chatBody("served-by-tpm-backup") });
    await createOpenAiModel("mgrl-tpm", tpmUp, { rate_limit: { tpm: 5 } });
    await createOpenAiModel("mgrl-tpm-backup", backup);
    await seed!.createModel({
      display_name: "mgrl-tpm-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "mgrl-tpm" }, { model: "mgrl-tpm-backup" }],
      },
    });
    await waitModelsListed(["mgrl-tpm", "mgrl-tpm-backup", "mgrl-tpm-group"]);
    await awaitWindowHeadroom(5);

    // First call is served by the TPM-capped member and commits 8 tokens
    // to ITS bucket (the reservation-merge under test — pre-fix the
    // tokens landed nowhere and the second call stayed on the member).
    expect(servedContent(await callGroup("mgrl-tpm-group"))).toBe("served-by-tpm");
    expect(servedContent(await callGroup("mgrl-tpm-group"))).toBe("served-by-tpm-backup");
  });

  test("streaming: member TPM commits at stream end and the next stream fails over", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const streamLimited = await newUpstream({
      streamEvents: streamEvents("stream-from-limited", 16),
    });
    const streamBackup = await newUpstream({
      streamEvents: streamEvents("stream-from-backup", 16),
    });
    await createOpenAiModel("mgrl-stream-tpm", streamLimited, {
      rate_limit: { tpm: 10 },
    });
    await createOpenAiModel("mgrl-stream-backup", streamBackup);
    await seed!.createModel({
      display_name: "mgrl-stream-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "mgrl-stream-tpm" }, { model: "mgrl-stream-backup" }],
      },
    });
    await waitModelsListed([
      "mgrl-stream-tpm",
      "mgrl-stream-backup",
      "mgrl-stream-group",
    ]);
    // The follow-up poll below runs for up to 5s after the first stream,
    // so this burst needs more headroom than the non-streaming cases.
    await awaitWindowHeadroom(15);

    const streamCall = async (): Promise<{ status: number; text: string }> => {
      const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "mgrl-stream-group",
          messages: [{ role: "user", content: "hello" }],
          stream: true,
        }),
      });
      return { status: res.status, text: await res.text() };
    };

    // First stream is served by the capped member; its terminal usage
    // frame (16 tokens > TPM=10) must be committed to the MEMBER's
    // bucket via the merged post-stream keys.
    const first = await streamCall();
    expect(first.status).toBe(200);
    expect(first.text).toContain("stream-from-limited");

    // The post-stream commit fires when the server drops the response
    // body — a tick after the client finishes reading — so poll until
    // the follow-up stream lands on the backup.
    const deadline = Date.now() + 5_000;
    let servedBy = "";
    while (Date.now() < deadline) {
      const next = await streamCall();
      expect(next.status).toBe(200);
      if (next.text.includes("stream-from-backup")) {
        servedBy = "backup";
        break;
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(servedBy).toBe("backup");
  });

  test("/v1/messages: over-limit member fails over to the next target", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const anLimited = await newUpstream({
      nonStreamBody: anthropicMessageBody("an-served-by-limited"),
    });
    const anBackup = await newUpstream({
      nonStreamBody: anthropicMessageBody("an-served-by-backup"),
    });
    const mkAnthropic = async (name: string, up: OpenAiUpstream, extra: Record<string, unknown> = {}) => {
      const pk = await seed!.createProviderKey({
        display_name: `${name}-pk`,
        provider: "anthropic",
        adapter: "anthropic",
        secret: "sk-ant-mock",
        // Anthropic bridge appends /v1/messages: point at the bare host.
        api_base: up.baseUrl,
      });
      await seed!.createModel({
        display_name: name,
        provider: "anthropic",
        model_name: "claude-3-5-haiku-20241022",
        provider_key_id: pk.id,
        ...extra,
      });
    };
    await mkAnthropic("mgrl-an-limited", anLimited, { rate_limit: { rpm: 1 } });
    await mkAnthropic("mgrl-an-backup", anBackup);
    await seed!.createModel({
      display_name: "mgrl-an-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "mgrl-an-limited" }, { model: "mgrl-an-backup" }],
      },
    });
    await waitModelsListed(["mgrl-an-limited", "mgrl-an-backup", "mgrl-an-group"]);
    await awaitWindowHeadroom(5);

    const callMessages = async (): Promise<{ status: number; text: string }> => {
      const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${CALLER_PLAINTEXT}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          model: "mgrl-an-group",
          max_tokens: 32,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      return { status: res.status, text: await res.text() };
    };

    const first = await callMessages();
    expect(first.status).toBe(200);
    expect(first.text).toContain("an-served-by-limited");

    // Same minute, member over its RPM=1 → the /v1/messages dispatch
    // loop must fail over exactly like /v1/chat/completions.
    const second = await callMessages();
    expect(second.status).toBe(200);
    expect(second.text).toContain("an-served-by-backup");
  });
});
