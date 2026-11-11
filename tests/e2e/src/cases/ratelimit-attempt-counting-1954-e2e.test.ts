import { createHash, randomUUID } from "node:crypto";
import { connect } from "node:net";
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

// E2E for AISIX-Cloud#1954: what a request routed through a Model Group
// counts against each rate-limit layer, across same-target retries,
// fail-overs, and the gateway's own refusals.
//
//   1. A policy pinned to the GROUP (the caller-addressed entry) counts once
//      per downstream request, however many attempts the request makes —
//      chat and the single-shot family (embeddings).
//   2. A MEMBER's own limit counts every upstream attempt, same-target
//      retries included — on count_tokens and embeddings, whose loops used
//      to reserve once per target.
//      A group-pinned policy whose match also depends on the member — a
//      negated provider condition, or a `model ~=` the member half of the
//      {member, group} pair decides — counts every attempt instead.
//   3. A request this gateway's own limiter refuses consumes nothing: a
//      request gate layer that refuses gives back what earlier layers
//      counted, and a routed request whose every member is over its own
//      limit gives back the request gate's counts.
//
// Every scenario runs once on the in-memory store and once on the shared
// Redis store.

const REDIS_URL = process.env.AISIX_E2E_REDIS ?? "redis://127.0.0.1:6379";

/** RESP PING, so the Redis leg skips honestly where no Redis runs. */
async function redisPing(url: string): Promise<boolean> {
  const m = /^redis:\/\/(?:[^@/]*@)?([^:/]+)(?::(\d+))?/.exec(url);
  if (!m) return false;
  const port = m[2] ? Number(m[2]) : 6379;
  return new Promise((resolve) => {
    const sock = connect({ host: m[1], port }, () => sock.write("PING\r\n"));
    const done = (ok: boolean) => {
      sock.destroy();
      resolve(ok);
    };
    sock.once("data", (buf) => done(buf.toString().startsWith("+PONG")));
    sock.once("error", () => done(false));
    sock.setTimeout(1000, () => done(false));
  });
}

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

const CHAT_BODY = {
  id: "cmpl-1954",
  object: "chat.completion",
  created: 0,
  model: "gpt-4o-mini",
  choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
};
const EMBED_BODY = {
  object: "list",
  data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2, 0.3] }],
  model: "text-embedding-3-small",
  usage: { prompt_tokens: 3, total_tokens: 3 },
};
const DOWN = { status: 500, errorBody: { error: { message: "target is down" } } };

type Result = {
  status: number;
  body: { error?: { policy?: { id?: string } } } | null;
};

describe.each(["memory", "redis"] as const)(
  "rate-limit counting across attempts and refusals (AISIX-Cloud#1954), %s store",
  (backend) => {
    let app: SpawnedApp | undefined;
    let ready = false;
    const upstreams: OpenAiUpstream[] = [];
    // Model aliases name the model-limit buckets, and the Redis store is
    // shared across runs: every name is unique to this run.
    const sfx = randomUUID().slice(0, 8);
    const name = (base: string) => `${base}-1954-${sfx}`;
    const keys = {
      groupChat: `sk-1954-gchat-${sfx}`,
      groupEmb: `sk-1954-gemb-${sfx}`,
      memberCt: `sk-1954-ct-${sfx}`,
      memberEmb: `sk-1954-memb-${sfx}`,
      gateCapped: `sk-1954-gate-${sfx}`,
      allRefused: `sk-1954-allref-${sfx}`,
      groupNotProvider: `sk-1954-gnp-${sfx}`,
      groupNotModel: `sk-1954-gnm-${sfx}`,
      burner: `sk-1954-burner-${sfx}`,
    };
    let groupChatPolicyId = "";
    let groupEmbPolicyId = "";
    let groupNotProviderPolicyId = "";
    let groupNotModelPolicyId = "";
    let ctUpstream: OpenAiUpstream | undefined;
    let membEmbUpstream: OpenAiUpstream | undefined;
    let downChat: OpenAiUpstream | undefined;

    async function upstream(opts: Parameters<typeof startOpenAiUpstream>[0]) {
      const u = await startOpenAiUpstream(opts);
      upstreams.push(u);
      return u;
    }

    beforeAll(async () => {
      const etcd = new EtcdClient();
      ready = (await etcd.ping()) && (backend === "memory" || (await redisPing(REDIS_URL)));
      if (!ready) return;

      app = await spawnApp(
        backend === "redis"
          ? { extra: { ratelimit: { backend: "redis", redis: { url: REDIS_URL } } } }
          : {},
      );
      const seed = new SeedClient(etcd, app.etcdPrefix);
      const model = async (
        displayName: string,
        u: OpenAiUpstream,
        extra: Record<string, unknown> = {},
      ) => {
        const anthropic = extra.provider === "anthropic";
        const pk = await seed.createProviderKey({
          display_name: `${displayName}-pk`,
          secret: "sk-mock",
          // The Anthropic bridge appends the path to the bare host.
          api_base: anthropic ? u.baseUrl : `${u.baseUrl}/v1`,
          ...(anthropic ? { provider: "anthropic", adapter: "anthropic" } : {}),
        });
        return (
          await seed.createModel({
            display_name: displayName,
            provider: "openai",
            model_name: "gpt-4o-mini",
            provider_key_id: pk.id,
            ...extra,
          })
        ).id;
      };
      const group = async (displayName: string, members: string[], retries: number) =>
        (
          await seed.createModel({
            display_name: displayName,
            routing: {
              strategy: "failover",
              retries,
              targets: members.map((m) => ({ model: m })),
            },
          })
        ).id;

      downChat = await upstream(DOWN);
      const okChat = await upstream({ nonStreamBody: CHAT_BODY });
      const okEmb = await upstream({ nonStreamBody: EMBED_BODY });
      ctUpstream = await upstream({
        scriptedResponses: [DOWN, { nonStreamBody: { input_tokens: 42 } }, { nonStreamBody: { input_tokens: 42 } }],
      });
      membEmbUpstream = await upstream({
        scriptedResponses: [DOWN, { nonStreamBody: EMBED_BODY }, { nonStreamBody: EMBED_BODY }],
      });

      // 1. group-pinned policies over [always-down, healthy], one retry each.
      await model(name("down"), downChat);
      await model(name("ok"), okChat);
      const gChat = await group(name("g-chat"), [name("down"), name("ok")], 1);
      await model(name("down-emb"), downChat, { model_name: "text-embedding-3-small" });
      await model(name("ok-emb"), okEmb, { model_name: "text-embedding-3-small" });
      const gEmb = await group(name("g-emb"), [name("down-emb"), name("ok-emb")], 1);
      groupChatPolicyId = (
        await seed.createRateLimitPolicy({
          name: name("group-chat-cap"),
          conditions: [
            { dimension: "team", operator: "in", value: [name("team-gchat")] },
            { dimension: "model", operator: "in", value: [gChat] },
          ],
          limits: { rpm: 3 },
        })
      ).id;
      groupEmbPolicyId = (
        await seed.createRateLimitPolicy({
          name: name("group-emb-cap"),
          conditions: [
            { dimension: "team", operator: "in", value: [name("team-gemb")] },
            { dimension: "model", operator: "in", value: [gEmb] },
          ],
          limits: { rpm: 3 },
        })
      ).id;

      groupNotProviderPolicyId = (
        await seed.createRateLimitPolicy({
          name: name("group-not-anthropic-cap"),
          conditions: [
            { dimension: "team", operator: "in", value: [name("team-gnp")] },
            { dimension: "model", operator: "in", value: [gChat] },
            {
              logic: "and",
              negate: true,
              children: [{ dimension: "provider", operator: "==", value: "anthropic" }],
            },
          ],
          limits: { rpm: 3 },
        })
      ).id;
      groupNotModelPolicyId = (
        await seed.createRateLimitPolicy({
          name: name("group-not-model-cap"),
          conditions: [
            { dimension: "team", operator: "in", value: [name("team-gnm")] },
            { dimension: "model", operator: "in", value: [gChat] },
            { dimension: "model", operator: "~=", value: name("elsewhere") },
          ],
          limits: { rpm: 3 },
        })
      ).id;

      // 2. a member with rpm 2 that fails once and is retried.
      await model(name("ct"), ctUpstream, {
        provider: "anthropic",
        model_name: "claude-haiku-4-5-20251001",
        rate_limit: { rpm: 2 },
      });
      await group(name("g-ct"), [name("ct")], 1);
      await model(name("memb"), membEmbUpstream, {
        model_name: "text-embedding-3-small",
        rate_limit: { rpm: 2 },
      });
      await group(name("g-memb"), [name("memb")], 1);

      // 3. refusals.
      await model(name("capped"), okChat, { rate_limit: { rpm: 1 } });
      await model(name("free"), okChat);
      await model(name("full-a"), okChat, { rate_limit: { rpm: 1 } });
      await model(name("full-b"), okChat, { rate_limit: { rpm: 1 } });
      await group(name("g-full"), [name("full-a"), name("full-b")], 0);

      // Caller keys last; the burner key is seeded last of all and is the
      // readiness gate, so its listing implies everything above landed.
      const key = (plaintext: string, extra: Record<string, unknown> = {}) =>
        seed.createApiKey({ key_hash: sha256(plaintext), allowed_models: ["*"], ...extra });
      await key(keys.groupChat, { team_id: name("team-gchat") });
      await key(keys.groupEmb, { team_id: name("team-gemb") });
      await key(keys.groupNotProvider, { team_id: name("team-gnp") });
      await key(keys.groupNotModel, { team_id: name("team-gnm") });
      await key(keys.memberCt);
      await key(keys.memberEmb);
      await key(keys.gateCapped, { rate_limit: { rpm: 2 } });
      await key(keys.allRefused, { rate_limit: { rpm: 1 } });
      await key(keys.burner);
      const probe = new ProxyClient(app.proxyUrl, keys.burner);
      await waitConfigPropagation(async () => {
        const res = await probe.listModels();
        const ids = ((res.body as { data?: Array<{ id?: string }> } | null)?.data ?? []).map(
          (m) => m.id,
        );
        return res.status === 200 && ids.includes(name("g-full"));
      });
    });

    afterAll(async () => {
      await app?.exit();
      await Promise.all(upstreams.map((u) => u.close()));
    });

    async function post(key: string, path: string, body: unknown): Promise<Result> {
      const res = await fetch(`${app!.proxyUrl}${path}`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${key}`,
          "x-api-key": key,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
        },
        body: JSON.stringify(body),
      });
      const text = await res.text();
      let parsed: Result["body"] = null;
      try {
        parsed = JSON.parse(text);
      } catch {
        // non-JSON error body; the status is what the spec reads
      }
      return { status: res.status, body: parsed };
    }
    const chat = (key: string, model: string) =>
      post(key, "/v1/chat/completions", { model, messages: [{ role: "user", content: "hi" }] });
    const embed = (key: string, model: string) =>
      post(key, "/v1/embeddings", { model, input: "hi" });
    const countTokens = (key: string, model: string) =>
      post(key, "/v1/messages/count_tokens", {
        model,
        messages: [{ role: "user", content: "hi" }],
      });

    test.for([
      ["chat", () => keys.groupChat, () => name("g-chat"), () => groupChatPolicyId, chat],
      ["embeddings", () => keys.groupEmb, () => name("g-emb"), () => groupEmbPolicyId, embed],
    ] as const)(
      "%s: a policy pinned to the group counts each request once across its attempts",
      { timeout: 60_000 },
      async ([, key, model, policyId, call], ctx) => {
        if (!ready || !app) {
          ctx.skip();
          return;
        }
        await awaitWindowHeadroom(20);
        const before = downChat!.receivedRequests.length;
        // Each request tries the down member twice (initial + one retry)
        // before the healthy one serves it: three attempts, one request.
        for (let i = 0; i < 3; i++) {
          expect((await call(key(), model())).status, `request ${i + 1}`).toBe(200);
        }
        expect(downChat!.receivedRequests.length - before).toBeGreaterThanOrEqual(2);
        const fourth = await call(key(), model());
        expect(fourth.status).toBe(429);
        expect(fourth.body?.error?.policy?.id).toBe(policyId());
      },
    );

    test.for([
      ["NOT (provider == anthropic)", () => keys.groupNotProvider, () => groupNotProviderPolicyId],
      ["model ~= another model", () => keys.groupNotModel, () => groupNotModelPolicyId],
    ] as const)(
      "chat: a group policy that also tests the member (%s) counts every attempt",
      { timeout: 60_000 },
      async ([, key, policyId], ctx) => {
        if (!ready || !app) {
          ctx.skip();
          return;
        }
        await awaitWindowHeadroom(20);
        // Three attempts (the down member twice, then the healthy one), each
        // matching the policy, fill its rpm 3 in one request.
        expect((await chat(key(), name("g-chat"))).status).toBe(200);
        const second = await chat(key(), name("g-chat"));
        expect(second.status).toBe(429);
        expect(second.body?.error?.policy?.id).toBe(policyId());
      },
    );

    test.for([
      ["count_tokens", () => keys.memberCt, () => name("g-ct"), () => ctUpstream!, countTokens],
      ["embeddings", () => keys.memberEmb, () => name("g-memb"), () => membEmbUpstream!, embed],
    ] as const)(
      "%s: a member's own limit counts the same-target retry",
      { timeout: 60_000 },
      async ([, key, model, upstreamOf, call], ctx) => {
        if (!ready || !app) {
          ctx.skip();
          return;
        }
        await awaitWindowHeadroom(20);
        // The member (rpm 2) fails once and is retried: two attempts.
        expect((await call(key(), model())).status).toBe(200);
        expect(upstreamOf().receivedRequests.length).toBe(2);
        // Both attempts counted, so the member is full and the group's
        // only target is refused without reaching it.
        expect((await call(key(), model())).status).toBe(429);
        expect(upstreamOf().receivedRequests.length).toBe(2);
      },
    );

    test("a request refused at a later gate layer leaves the key's own quota unconsumed", async (ctx) => {
      if (!ready || !app) {
        ctx.skip();
        return;
      }
      await awaitWindowHeadroom(20);
      // Fill the model's own rpm 1 with another key.
      expect((await chat(keys.burner, name("capped"))).status).toBe(200);
      // The key (rpm 2) is refused twice by the model layer, after its own
      // layer admitted it.
      expect((await chat(keys.gateCapped, name("capped"))).status).toBe(429);
      expect((await chat(keys.gateCapped, name("capped"))).status).toBe(429);
      // Neither refusal cost the key anything: both of its slots are free.
      expect((await chat(keys.gateCapped, name("free"))).status).toBe(200);
      expect((await chat(keys.gateCapped, name("free"))).status).toBe(200);
      expect((await chat(keys.gateCapped, name("free"))).status).toBe(429);
    }, 60_000);

    test("a routed request whose every member is over its own limit leaves the gate unconsumed", async (ctx) => {
      if (!ready || !app) {
        ctx.skip();
        return;
      }
      await awaitWindowHeadroom(20);
      expect((await chat(keys.burner, name("full-a"))).status).toBe(200);
      expect((await chat(keys.burner, name("full-b"))).status).toBe(200);
      // Both members refuse; nothing goes upstream.
      expect((await chat(keys.allRefused, name("g-full"))).status).toBe(429);
      // The key's only slot (rpm 1) is still free.
      expect((await chat(keys.allRefused, name("free"))).status).toBe(200);
      expect((await chat(keys.allRefused, name("free"))).status).toBe(429);
    }, 60_000);
  },
);
