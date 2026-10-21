// /v1/responses context affinity through routing model groups.
//
// A follow-up turn carries context only the upstream account that produced
// it can use: a reasoning item's `encrypted_content`, and a stored
// `previous_response_id`. The mock upstream below binds both to the
// credential it was called with (one credential = one account) and rejects
// anything else exactly the way OpenAI does — 400 `invalid_encrypted_content`
// and 404 for an unknown response id. Every target of the groups here serves
// the same model, so the only thing that can make a follow-up fail is which
// account it lands on.
//
// The gateway tags what a group's target produced
// (`aisix_enc_v1:<b64url(target id)>;<original>` and
// `resp_<b64url("aisix:v1;<target id>;<original>")>`) and must route a
// follow-up back to an account that can read it — or strip what no
// available account can read — while never letting a tag reach an upstream.
import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  waitForLogLine,
  type SpawnedApp,
} from "../harness/index.js";

const CALLER_PLAINTEXT = "sk-resp-affinity-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

interface Received {
  account: string;
  path: string;
  body: any;
  raw: string;
}

// One HTTP server standing in for every upstream account; the bearer
// credential names the account.
class AccountsUpstream {
  readonly received: Received[] = [];
  readonly down = new Set<string>();
  private seq = 0;
  private readonly stored = new Map<string, string>();
  private constructor(private readonly server: Server) {}

  static async start(): Promise<AccountsUpstream> {
    let self: AccountsUpstream | undefined;
    const server = createServer((req, res) => {
      void self!.handle(req).then(({ status, body, sse }) => {
        if (sse) {
          res.writeHead(status, { "content-type": "text/event-stream" });
          res.end(sse);
        } else {
          res.writeHead(status, { "content-type": "application/json" });
          res.end(JSON.stringify(body));
        }
      });
    });
    self = new AccountsUpstream(server);
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return self;
  }

  get baseUrl(): string {
    return `http://127.0.0.1:${(this.server.address() as AddressInfo).port}/v1`;
  }

  close(): Promise<void> {
    return new Promise((r) => this.server.close(() => r()));
  }

  forAccount(account: string): Received[] {
    return this.received.filter((r) => r.account === account);
  }

  private async handle(
    req: IncomingMessage,
  ): Promise<{ status: number; body?: unknown; sse?: string }> {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    const account = (req.headers.authorization ?? "").replace(/^Bearer /, "");
    const body = raw ? JSON.parse(raw) : {};
    this.received.push({ account, path: req.url ?? "", body, raw });

    if (this.down.has(account)) {
      return { status: 503, body: { error: { message: "account down", type: "server_error" } } };
    }
    if (req.url?.endsWith("/chat/completions")) {
      if (body.stream) {
        const chunk = (delta: Record<string, unknown>, finish: string | null) =>
          `data: ${JSON.stringify({ id: `chatcmpl-${account}`, object: "chat.completion.chunk", created: 0, model: body.model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
        return {
          status: 200,
          sse: chunk({ role: "assistant", content: "bridged" }, null) + chunk({}, "stop") + "data: [DONE]\n\n",
        };
      }
      return {
        status: 200,
        body: {
          id: `chatcmpl-${account}`,
          object: "chat.completion",
          created: 0,
          model: body.model,
          choices: [{ index: 0, message: { role: "assistant", content: "bridged" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        },
      };
    }
    for (const item of body.input ?? []) {
      if (typeof item?.encrypted_content === "string" && !item.encrypted_content.startsWith(`enc:${account}:`)) {
        return {
          status: 400,
          body: {
            error: {
              message: "The encrypted content could not be verified.",
              type: "invalid_request_error",
              code: "invalid_encrypted_content",
            },
          },
        };
      }
    }
    if (body.previous_response_id !== undefined && this.stored.get(body.previous_response_id) !== account) {
      return {
        status: 404,
        body: { error: { message: `Previous response with id '${body.previous_response_id}' not found.`, type: "invalid_request_error" } },
      };
    }
    const n = ++this.seq;
    const id = `resp_${account}_${n}`;
    this.stored.set(id, account);
    const reasoning = {
      type: "reasoning",
      id: `rs_${account}_${n}`,
      summary: [{ type: "summary_text", text: `thought ${n}` }],
      encrypted_content: `enc:${account}:${n}`,
    };
    const message = {
      type: "message",
      id: `msg_${account}_${n}`,
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: `answer from ${account}`, annotations: [] }],
    };
    const response = {
      id,
      object: "response",
      created_at: 0,
      status: "completed",
      model: body.model,
      previous_response_id: body.previous_response_id ?? null,
      output: [reasoning, message],
      usage: { input_tokens: 3, output_tokens: 2, total_tokens: 5 },
    };
    if (!body.stream) return { status: 200, body: response };
    const frame = (event: Record<string, unknown>) =>
      `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
    const pending = { ...response, status: "in_progress", output: [] };
    return {
      status: 200,
      sse:
        frame({ type: "response.created", response: pending }) +
        frame({ type: "response.output_item.added", output_index: 0, item: { ...reasoning, encrypted_content: null } }) +
        frame({ type: "response.output_item.done", output_index: 0, item: reasoning }) +
        frame({ type: "response.output_item.done", output_index: 1, item: message }) +
        frame({ type: "response.completed", response }),
    };
  }
}

function decodeResponseId(id: string): { target: string; original: string } | undefined {
  if (!id.startsWith("resp_")) return undefined;
  const text = Buffer.from(id.slice(5), "base64url").toString("utf8");
  if (!text.startsWith("aisix:v1;")) return undefined;
  const rest = text.slice("aisix:v1;".length);
  const at = rest.indexOf(";");
  return { target: rest.slice(0, at), original: rest.slice(at + 1) };
}

function decodeEncrypted(value: string): { target: string; original: string } | undefined {
  if (!value.startsWith("aisix_enc_v1:")) return undefined;
  const rest = value.slice("aisix_enc_v1:".length);
  const at = rest.indexOf(";");
  return {
    target: Buffer.from(rest.slice(0, at), "base64url").toString("utf8"),
    original: rest.slice(at + 1),
  };
}

function parseSse(text: string): any[] {
  return text
    .split(/\n\n/)
    .map((f) => f.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n"))
    .filter((d) => d && d !== "[DONE]")
    .map((d) => JSON.parse(d));
}

describe("/v1/responses context affinity through model groups", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let upstream: AccountsUpstream | undefined;
  let etcdReachable = false;
  // Model ids by display name.
  const ids: Record<string, string> = {};

  async function model(
    name: string,
    pkId: string,
    extra: Record<string, unknown> = {},
  ): Promise<void> {
    const m = await seed!.createModel({
      display_name: name,
      provider: "openai",
      model_name: `gpt-${name}`,
      provider_key_id: pkId,
      ...extra,
    });
    ids[name] = m.id;
  }

  async function group(
    name: string,
    strategy: string,
    targets: string[],
  ): Promise<void> {
    const m = await seed!.createModel({
      display_name: name,
      routing: {
        strategy,
        targets: targets.map((t) => ({ model: t })),
        max_fallbacks: targets.length - 1,
      },
    });
    ids[name] = m.id;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await AccountsUpstream.start();
    // `info` for the access log the telemetry assertion reads.
    app = await spawnApp({ extraEnv: { RUST_LOG: "info" } });
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = async (account: string, extra: Record<string, unknown> = {}) =>
      (
        await seed!.createProviderKey({
          display_name: `pk-${account}`,
          secret: account,
          api_base: upstream!.baseUrl,
          ...extra,
        })
      ).id;
    const pkA = await pk("acct-a");
    const pkB = await pk("acct-b");
    const pkC = await pk("acct-c");
    const pkD = await pk("acct-d");
    const pkE = await pk("acct-e");
    // A chat-only key: its models are reached through the Responses bridge.
    const pkBridge = await pk("acct-bridge", { provider: "deepseek", adapter: "openai" });

    await model("aff-a", pkA);
    await model("aff-b", pkB);
    // Same provider key as aff-a: can read aff-a's context.
    await model("aff-a2", pkA);
    await model("aff-c", pkC);
    await model("aff-d", pkD, { cooldown: { enabled: true, default_seconds: 300 } });
    await model("aff-e", pkE);
    await model("aff-bridge", pkBridge, { provider: "deepseek" });

    await group("aff-rr", "round_robin", ["aff-a", "aff-b"]);
    await group("aff-peer", "failover", ["aff-c", "aff-a2"]);
    await group("aff-cool", "failover", ["aff-d", "aff-e"]);
    await group("aff-hop", "failover", ["aff-c", "aff-e"]);
    await group("aff-bridged", "failover", ["aff-bridge"]);

    // Caller key last: once it authenticates, every resource above is in
    // the snapshot.
    await seed.createApiKey({ key_hash: CALLER_KEY_HASH, allowed_models: ["*"] });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      return res.status === 200;
    });
  });

  afterAll(async () => {
    // No tag the gateway minted may ever reach an upstream.
    if (upstream) {
      for (const r of upstream.received) {
        expect(r.raw).not.toContain("aisix_enc_v1:");
        const prev = r.body.previous_response_id;
        if (typeof prev === "string") expect(decodeResponseId(prev)).toBeUndefined();
      }
    }
    await app?.exit();
    await upstream?.close();
  });

  async function respond(
    body: Record<string, unknown>,
  ): Promise<{ status: number; json: any; text: string; requestId: string }> {
    const res = await fetch(`${app!.proxyUrl}/v1/responses`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let json: any;
    try {
      json = JSON.parse(text);
    } catch {
      json = undefined;
    }
    return { status: res.status, json, text, requestId: res.headers.get("x-aisix-request-id") ?? "" };
  }

  const userTurn = (text: string) => ({ type: "message", role: "user", content: text });

  test("a group's non-streaming response carries tagged ids and ciphertext", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const r = await respond({ model: "aff-rr", input: [userTurn("hello")] });
    expect(r.status).toBe(200);
    const served = upstream.received.at(-1)!;
    const target = served.account === "acct-a" ? ids["aff-a"] : ids["aff-b"];

    const decoded = decodeResponseId(r.json.id);
    expect(decoded?.target).toBe(target);
    expect(decoded?.original).toMatch(new RegExp(`^resp_${served.account}_\\d+$`));
    // Telemetry records the upstream's own id, not the tagged one.
    const access = await waitForLogLine(
      app,
      (l) => l.includes("proxy request completed") && l.includes(`request_id="${r.requestId}"`),
      "the access-log line for the tagged response",
    );
    expect(access).toContain(`provider_request_id="${decoded!.original}"`);
    const reasoning = r.json.output.find((i: any) => i.type === "reasoning");
    const enc = decodeEncrypted(reasoning.encrypted_content);
    expect(enc?.target).toBe(target);
    expect(enc?.original).toMatch(new RegExp(`^enc:${served.account}:\\d+$`));
    // Readable fields are untouched.
    expect(reasoning.summary[0].text).toMatch(/^thought \d+$/);
    expect(reasoning.id).toMatch(/^rs_/);
    expect(r.json.model).toBe("aff-rr");
  });

  test("a group's streamed response tags every event that carries the id or ciphertext", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const r = await respond({ model: "aff-rr", input: [userTurn("hello stream")], stream: true });
    expect(r.status).toBe(200);
    const served = upstream.received.at(-1)!;
    const target = served.account === "acct-a" ? ids["aff-a"] : ids["aff-b"];
    const events = parseSse(r.text);

    const snapshots = events.filter((e) => e.type === "response.created" || e.type === "response.completed");
    expect(snapshots).toHaveLength(2);
    for (const e of snapshots) {
      const decoded = decodeResponseId(e.response.id);
      expect(decoded?.target).toBe(target);
      expect(decoded?.original).toMatch(new RegExp(`^resp_${served.account}_\\d+$`));
    }
    const done = events.find((e) => e.type === "response.output_item.done" && e.item.type === "reasoning");
    expect(decodeEncrypted(done.item.encrypted_content)?.target).toBe(target);
    const completed = events.find((e) => e.type === "response.completed");
    const finalReasoning = completed.response.output.find((i: any) => i.type === "reasoning");
    expect(decodeEncrypted(finalReasoning.encrypted_content)?.target).toBe(target);
    // The item announced before its ciphertext exists stays null.
    const added = events.find((e) => e.type === "response.output_item.added");
    expect(added.item.encrypted_content).toBeNull();
  });

  test("encrypted follow-ups stay on the origin target under round_robin", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    let history: any[] = [userTurn("turn 1")];
    const first = await respond({ model: "aff-rr", store: false, input: history });
    expect(first.status).toBe(200);
    const origin = upstream.received.at(-1)!.account;
    history = [...history, ...first.json.output];

    // round_robin alternates between two targets, so without affinity the
    // very next turn lands on the other account and is rejected.
    for (let turn = 2; turn <= 5; turn++) {
      history = [...history, userTurn(`turn ${turn}`)];
      const r = await respond({ model: "aff-rr", store: false, input: history, stream: turn % 2 === 0 });
      expect(r.status, r.text).toBe(200);
      const served = upstream.received.at(-1)!;
      expect(served.account).toBe(origin);
      // The upstream got its own ciphertext back, byte for byte.
      const sent = served.body.input.filter((i: any) => i.type === "reasoning");
      expect(sent.length).toBe(turn - 1);
      for (const item of sent) expect(item.encrypted_content).toMatch(new RegExp(`^enc:${origin}:\\d+$`));
      const output = turn % 2 === 0
        ? parseSse(r.text).find((e) => e.type === "response.completed").response.output
        : r.json.output;
      history = [...history, ...output];
    }
  });

  test("a stored previous_response_id stays on the account that stored it", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const first = await respond({ model: "aff-rr", input: [userTurn("remember this")] });
    expect(first.status).toBe(200);
    const origin = upstream.received.at(-1)!.account;
    let previous = first.json.id;
    for (let turn = 2; turn <= 4; turn++) {
      const r = await respond({ model: "aff-rr", previous_response_id: previous, input: [userTurn(`turn ${turn}`)] });
      expect(r.status, r.text).toBe(200);
      const served = upstream.received.at(-1)!;
      expect(served.account).toBe(origin);
      expect(served.body.previous_response_id).toBe(decodeResponseId(previous)!.original);
      // The Response object echoes the id the client sent.
      expect(r.json.previous_response_id).toBe(previous);
      previous = r.json.id;
    }
  });

  test("an unavailable origin hands the conversation to a target on the same provider key", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    // Produced by aff-a through aff-rr; aff-a is not a member of aff-peer,
    // whose failover order would otherwise pick aff-c first.
    let first;
    do {
      first = await respond({ model: "aff-rr", input: [userTurn("peer")] });
      expect(first.status).toBe(200);
    } while (upstream.received.at(-1)!.account !== "acct-a");

    const r = await respond({ model: "aff-peer", input: [userTurn("peer"), ...first.json.output, userTurn("again")] });
    expect(r.status, r.text).toBe(200);
    const served = upstream.received.at(-1)!;
    expect(served.account).toBe("acct-a");
    expect(served.body.model).toBe("gpt-aff-a2");
    const reasoning = served.body.input.find((i: any) => i.type === "reasoning");
    expect(reasoning.encrypted_content).toMatch(/^enc:acct-a:\d+$/);
  });

  test("an origin in cooldown: the next target gets the reasoning without its ciphertext", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const first = await respond({ model: "aff-cool", input: [userTurn("cool")] });
    expect(first.status).toBe(200);
    expect(upstream.received.at(-1)!.account).toBe("acct-d");

    // One failure puts aff-d into cooldown (this request fails over to aff-e).
    upstream.down.add("acct-d");
    expect((await respond({ model: "aff-cool", input: [userTurn("trip")] })).status).toBe(200);
    upstream.down.delete("acct-d");
    const before = upstream.forAccount("acct-d").length;

    const r = await respond({ model: "aff-cool", input: [userTurn("cool"), ...first.json.output, userTurn("again")] });
    expect(r.status, r.text).toBe(200);
    // The cooling origin is not tried at all.
    expect(upstream.forAccount("acct-d").length).toBe(before);
    const served = upstream.received.at(-1)!;
    expect(served.account).toBe("acct-e");
    const reasoning = served.body.input.find((i: any) => i.type === "reasoning");
    expect(reasoning).toEqual({ type: "reasoning", summary: [{ type: "summary_text", text: expect.stringMatching(/^thought \d+$/) }] });
  });

  test("a failover hop away from the origin strips the ciphertext it cannot read", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const first = await respond({ model: "aff-hop", input: [userTurn("hop")] });
    expect(first.status).toBe(200);
    expect(upstream.received.at(-1)!.account).toBe("acct-c");

    upstream.down.add("acct-c");
    try {
      const r = await respond({ model: "aff-hop", input: [userTurn("hop"), ...first.json.output, userTurn("again")] });
      expect(r.status, r.text).toBe(200);
      const [pinned, hop] = upstream.received.slice(-2);
      // The origin was tried first, with its own ciphertext …
      expect(pinned.account).toBe("acct-c");
      expect(pinned.body.input.find((i: any) => i.type === "reasoning").encrypted_content).toMatch(/^enc:acct-c:/);
      // … and the hop got the readable summary only.
      expect(hop.account).toBe("acct-e");
      const reasoning = hop.body.input.find((i: any) => i.type === "reasoning");
      expect(reasoning.encrypted_content).toBeUndefined();
      expect(reasoning.id).toBeUndefined();
      expect(reasoning.summary).toHaveLength(1);
    } finally {
      upstream.down.delete("acct-c");
    }
  });

  test("a direct model receives untagged context and never a tag", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    let first;
    do {
      first = await respond({ model: "aff-rr", input: [userTurn("direct")] });
      expect(first.status).toBe(200);
    } while (upstream.received.at(-1)!.account !== "acct-b");

    const r = await respond({
      model: "aff-b",
      previous_response_id: first.json.id,
      input: [...first.json.output, userTurn("direct again")],
    });
    expect(r.status, r.text).toBe(200);
    const served = upstream.received.at(-1)!;
    expect(served.account).toBe("acct-b");
    expect(served.body.previous_response_id).toBe(decodeResponseId(first.json.id)!.original);
    expect(served.body.input[0].encrypted_content).toMatch(/^enc:acct-b:\d+$/);
    // A direct model's own output is not tagged.
    expect(decodeResponseId(r.json.id)).toBeUndefined();
    expect(r.json.output[0].encrypted_content).toMatch(/^enc:acct-b:\d+$/);
  });

  test("a forged or undecodable tag is dropped, not forwarded and not an error", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const r = await respond({
      model: "aff-rr",
      input: [
        { type: "reasoning", id: "rs_x", summary: [], encrypted_content: "aisix_enc_v1:!!not-base64;junk" },
        { type: "reasoning", id: "rs_y", summary: [], encrypted_content: `aisix_enc_v1:${Buffer.from("no-such-model").toString("base64url")};enc:acct-zzz:1` },
        userTurn("forged"),
      ],
    });
    expect(r.status, r.text).toBe(200);
    const served = upstream.received.at(-1)!;
    expect(served.body.input.filter((i: any) => i.type === "reasoning")).toHaveLength(0);
  });

  test("a bridged target's response id is tagged too", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const r = await respond({ model: "aff-bridged", input: [userTurn("bridge")] });
    expect(r.status, r.text).toBe(200);
    expect(upstream.received.at(-1)!.path).toMatch(/\/chat\/completions$/);
    expect(decodeResponseId(r.json.id)?.target).toBe(ids["aff-bridge"]);

    const s = await respond({ model: "aff-bridged", input: [userTurn("bridge")], stream: true });
    expect(s.status).toBe(200);
    const created = parseSse(s.text).find((e) => e.type === "response.created");
    expect(decodeResponseId(created.response.id)?.target).toBe(ids["aff-bridge"]);
  });
});
