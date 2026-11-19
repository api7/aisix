// /v1/responses context affinity through routing model groups and direct
// models.
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
// The gateway tags the ciphertext any model produced
// (`aisix_enc_v1:<b64url("<target id>;<provider key id>")>;<original>`) and
// the response id a group's target produced
// (`resp_<b64url("aisix:v1;<target id>;<original>")>`), and must route a
// follow-up back to an account that can read it — or strip what the account
// it lands on cannot read — while never letting a tag reach an upstream.
// Ciphertext follows the provider key that served it, so a model moved to
// another key in place, or deleted, or a conversation moved between a group
// and a direct model, must not change who can read it.
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

function decodeEncrypted(value: string): { target: string; key: string; original: string } | undefined {
  if (!value.startsWith("aisix_enc_v1:")) return undefined;
  const rest = value.slice("aisix_enc_v1:".length);
  const at = rest.indexOf(";");
  const [target, key] = Buffer.from(rest.slice(0, at), "base64url").toString("utf8").split(";");
  return { target, key, original: rest.slice(at + 1) };
}

// A ciphertext tag as gateways wrote it before the provider key was recorded.
const v1Encrypted = (targetId: string, original: string) =>
  `aisix_enc_v1:${Buffer.from(targetId).toString("base64url")};${original}`;

function parseSse(text: string): any[] {
  return text
    .split(/\n\n/)
    .map((f) => f.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n"))
    .filter((d) => d && d !== "[DONE]")
    .map((d) => JSON.parse(d));
}

describe("/v1/responses context affinity", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let upstream: AccountsUpstream | undefined;
  let etcdReachable = false;
  // Model ids by display name, and the documents they were written with.
  const ids: Record<string, string> = {};
  const docs: Record<string, Record<string, unknown>> = {};
  // Provider key ids by account.
  const pks: Record<string, string> = {};

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
    docs[name] = m.value;
  }

  const groupDoc = (name: string, strategy: string, targets: (string | { model: string; priority: number })[]) => ({
    display_name: name,
    routing: {
      strategy,
      targets: targets.map((t) => (typeof t === "string" ? { model: t } : t)),
      max_fallbacks: Math.max(targets.length - 1, 0),
    },
  });

  async function group(
    name: string,
    strategy: string,
    targets: string[],
  ): Promise<void> {
    const m = await seed!.createModel(groupDoc(name, strategy, targets));
    ids[name] = m.id;
  }

  // Wait until a context-free request to `name` is served by `account`.
  async function untilServedBy(name: string, account: string): Promise<void> {
    await waitConfigPropagation(async () => {
      const before = upstream!.received.length;
      const r = await respond({ model: name, input: [userTurn("probe")] });
      return r.status === 200 && upstream!.received.slice(before).at(-1)?.account === account;
    });
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
          api_key: account,
          api_base: upstream!.baseUrl,
          ...extra,
        })
      ).id;
    for (const account of ["acct-f", "acct-g", "acct-h", "acct-i", "acct-j", "acct-k", "acct-l", "acct-m", "acct-n", "acct-p", "acct-q", "acct-r", "acct-s", "acct-t", "acct-u", "acct-v", "acct-w"]) {
      pks[account] = await pk(account);
    }
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

    // Provider-key moves, deletions and group edits, one fixture each.
    await model("kc-o", pks["acct-f"]);
    await model("kc-peer", pks["acct-f"]);
    await group("kc-grp", "failover", ["kc-o", "kc-peer"]);
    await model("kc2-o", pks["acct-h"]);
    await group("kc2-grp", "failover", ["kc2-o"]);
    await model("del-o", pks["acct-j"]);
    await group("del-grp", "failover", ["del-o"]);
    await model("prev-o", pks["acct-k"]);
    await model("prev-x", pks["acct-m"]);
    await group("prev-src", "failover", ["prev-o"]);
    await group("prev-grp", "failover", ["prev-x", "prev-o"]);
    await model("pr-o", pks["acct-n"]);
    await model("pr-hi", pks["acct-p"]);
    await group("pr-grp", "failover", ["pr-o"]);
    await model("rp-o", pks["acct-q"]);
    await model("rp-new", pks["acct-r"]);
    await group("rp-grp", "failover", ["rp-o"]);
    // Direct models.
    await model("dr-o", pks["acct-s"]);
    await model("dm-o", pks["acct-t"]);
    await model("gd-o", pks["acct-v"]);
    await group("gd-grp", "failover", ["gd-o"]);
    await model("gd-peer", pks["acct-v"]);
    await model("gd-other", pks["acct-w"]);

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
    // The provider key that served it is recorded beside the model.
    expect(enc?.key).toBe(served.account === "acct-a" ? docs["aff-a"].provider_key_id : docs["aff-b"].provider_key_id);
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

  test("a group target's context reaches it untagged when addressed directly, and its direct response tags ciphertext only", async (ctx) => {
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
    // A direct model's response id is not tagged; its ciphertext is.
    expect(decodeResponseId(r.json.id)).toBeUndefined();
    const enc = decodeEncrypted(r.json.output[0].encrypted_content);
    expect(enc?.target).toBe(ids["aff-b"]);
    expect(enc?.key).toBe(docs["aff-b"].provider_key_id);
    expect(enc?.original).toMatch(/^enc:acct-b:\d+$/);
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

  // One turn through `name`; returns the history for the next one and the
  // account that served it.
  async function firstTurn(name: string, text: string): Promise<{ history: any[]; account: string }> {
    const r = await respond({ model: name, store: false, input: [userTurn(text)] });
    expect(r.status, r.text).toBe(200);
    return { history: [userTurn(text), ...r.json.output, userTurn(`${text} again`)], account: upstream!.received.at(-1)!.account };
  }

  const sentReasoning = (r: Received) => r.body.input.find((i: any) => i.type === "reasoning");

  test("a model moved to another provider key in place hands its ciphertext to a target still on the old key", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const { history, account } = await firstTurn("kc-grp", "move");
    expect(account).toBe("acct-f");

    await seed!.update("models", ids["kc-o"], { ...docs["kc-o"], provider_key_id: pks["acct-g"] });
    await untilServedBy("kc-o", "acct-g");

    const before = upstream.received.length;
    const r = await respond({ model: "kc-grp", store: false, input: history });
    expect(r.status, r.text).toBe(200);
    const sent = upstream.received.slice(before);
    // The ciphertext never reached the new account …
    for (const s of sent.filter((s) => s.account === "acct-g")) {
      expect(sentReasoning(s)?.encrypted_content).toBeUndefined();
    }
    // … and the peer on the account that issued it got it byte for byte.
    const served = sent.at(-1)!;
    expect(served.account).toBe("acct-f");
    expect(served.body.model).toBe("gpt-kc-peer");
    expect(sentReasoning(served).encrypted_content).toMatch(/^enc:acct-f:\d+$/);
  });

  test("a model moved to another provider key in place gets its old ciphertext stripped", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const { history, account } = await firstTurn("kc2-grp", "move alone");
    expect(account).toBe("acct-h");

    await seed!.update("models", ids["kc2-o"], { ...docs["kc2-o"], provider_key_id: pks["acct-i"] });
    await untilServedBy("kc2-o", "acct-i");

    const r = await respond({ model: "kc2-grp", store: false, input: history });
    expect(r.status, r.text).toBe(200);
    const served = upstream.received.at(-1)!;
    expect(served.account).toBe("acct-i");
    expect(sentReasoning(served)).toEqual({ type: "reasoning", summary: [{ type: "summary_text", text: expect.stringMatching(/^thought \d+$/) }] });
  });

  test("a deleted origin's ciphertext still reaches a new target on the provider key that issued it", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const { history, account } = await firstTurn("del-grp", "delete");
    expect(account).toBe("acct-j");

    await model("del-new", pks["acct-j"]);
    await seed!.update("models", ids["del-grp"], groupDoc("del-grp", "failover", ["del-new"]));
    await seed!.delete("models", ids["del-o"]);
    await waitConfigPropagation(async () => (await respond({ model: "del-o", input: [userTurn("probe")] })).status === 404);
    await untilServedBy("del-grp", "acct-j");

    const r = await respond({ model: "del-grp", store: false, input: history });
    expect(r.status, r.text).toBe(200);
    const served = upstream.received.at(-1)!;
    expect(served.account).toBe("acct-j");
    expect(served.body.model).toBe("gpt-del-new");
    expect(sentReasoning(served).encrypted_content).toMatch(/^enc:acct-j:\d+$/);
  });

  test("ciphertext tagged by an older gateway is still read by the origin's current provider key", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    // aff-a is not a member of aff-peer; its peer aff-a2 shares its key.
    const r = await respond({
      model: "aff-peer",
      store: false,
      input: [
        userTurn("old tag"),
        { type: "reasoning", id: "rs_old", summary: [{ type: "summary_text", text: "old" }], encrypted_content: v1Encrypted(ids["aff-a"], "enc:acct-a:old") },
        userTurn("old tag again"),
      ],
    });
    expect(r.status, r.text).toBe(200);
    const served = upstream.received.at(-1)!;
    expect(served.account).toBe("acct-a");
    expect(served.body.model).toBe("gpt-aff-a2");
    expect(sentReasoning(served).encrypted_content).toBe("enc:acct-a:old");
  });

  test("a stored previous_response_id still follows the origin model after it moves to another key (known behaviour)", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const first = await respond({ model: "prev-src", input: [userTurn("stored")] });
    expect(first.status, first.text).toBe(200);
    expect(upstream.received.at(-1)!.account).toBe("acct-k");

    await seed!.update("models", ids["prev-o"], { ...docs["prev-o"], provider_key_id: pks["acct-l"] });
    await untilServedBy("prev-o", "acct-l");

    // The response id records the model only, so the moved origin is still
    // pinned ahead of prev-x, whose failover order would otherwise go first.
    const before = upstream.received.length;
    await respond({ model: "prev-grp", previous_response_id: first.json.id, input: [userTurn("stored again")] });
    const pinned = upstream.received[before];
    expect(pinned.account).toBe("acct-l");
    expect(pinned.body.model).toBe("gpt-prev-o");
    expect(pinned.body.previous_response_id).toBe(decodeResponseId(first.json.id)!.original);
  });

  test("a higher-priority target added to the group does not take over a tagged conversation", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const { history, account } = await firstTurn("pr-grp", "priority");
    expect(account).toBe("acct-n");

    await seed!.update("models", ids["pr-grp"], groupDoc("pr-grp", "failover", [{ model: "pr-hi", priority: 10 }, { model: "pr-o", priority: 0 }]));
    await untilServedBy("pr-grp", "acct-p");

    const r = await respond({ model: "pr-grp", store: false, input: history });
    expect(r.status, r.text).toBe(200);
    const served = upstream.received.at(-1)!;
    expect(served.account).toBe("acct-n");
    expect(served.body.model).toBe("gpt-pr-o");
    expect(sentReasoning(served).encrypted_content).toMatch(/^enc:acct-n:\d+$/);
  });

  test("a group whose targets were all replaced by models on other keys gets the ciphertext stripped", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const { history, account } = await firstTurn("rp-grp", "replace");
    expect(account).toBe("acct-q");

    await seed!.update("models", ids["rp-grp"], groupDoc("rp-grp", "failover", ["rp-new"]));
    await untilServedBy("rp-grp", "acct-r");

    const r = await respond({ model: "rp-grp", store: false, input: history });
    expect(r.status, r.text).toBe(200);
    const served = upstream.received.at(-1)!;
    expect(served.account).toBe("acct-r");
    expect(sentReasoning(served)).toEqual({ type: "reasoning", summary: [{ type: "summary_text", text: expect.stringMatching(/^thought \d+$/) }] });
  });

  test("a direct model's response tags its ciphertext with the model and provider key, streamed or not", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const r = await respond({ model: "dr-o", input: [userTurn("direct tag")] });
    expect(r.status, r.text).toBe(200);
    expect(upstream.received.at(-1)!.account).toBe("acct-s");
    expect(decodeResponseId(r.json.id)).toBeUndefined();
    expect(r.json.id).toMatch(/^resp_acct-s_\d+$/);
    const enc = decodeEncrypted(r.json.output.find((i: any) => i.type === "reasoning").encrypted_content);
    expect(enc).toEqual({ target: ids["dr-o"], key: pks["acct-s"], original: expect.stringMatching(/^enc:acct-s:\d+$/) });

    const s = await respond({ model: "dr-o", input: [userTurn("direct tag")], stream: true });
    expect(s.status).toBe(200);
    const events = parseSse(s.text);
    const done = events.find((e) => e.type === "response.output_item.done" && e.item.type === "reasoning");
    expect(decodeEncrypted(done.item.encrypted_content)).toEqual({ target: ids["dr-o"], key: pks["acct-s"], original: expect.stringMatching(/^enc:acct-s:\d+$/) });
    const completed = events.find((e) => e.type === "response.completed");
    expect(decodeResponseId(completed.response.id)).toBeUndefined();
    const final = completed.response.output.find((i: any) => i.type === "reasoning");
    expect(decodeEncrypted(final.encrypted_content)?.key).toBe(pks["acct-s"]);
  });

  test("a direct model on an unchanged provider key reads its own ciphertext back verbatim", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const { history, account } = await firstTurn("dr-o", "direct keep");
    expect(account).toBe("acct-s");
    const r = await respond({ model: "dr-o", store: false, input: history });
    expect(r.status, r.text).toBe(200);
    const served = upstream.received.at(-1)!;
    expect(served.account).toBe("acct-s");
    expect(sentReasoning(served).encrypted_content).toBe(decodeEncrypted(history[1].encrypted_content)!.original);
  });

  test("a direct model moved to another provider key in place gets its old ciphertext stripped", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const { history, account } = await firstTurn("dm-o", "direct move");
    expect(account).toBe("acct-t");

    await seed!.update("models", ids["dm-o"], { ...docs["dm-o"], provider_key_id: pks["acct-u"] });
    await untilServedBy("dm-o", "acct-u");

    const r = await respond({ model: "dm-o", store: false, input: history });
    expect(r.status, r.text).toBe(200);
    const served = upstream.received.at(-1)!;
    expect(served.account).toBe("acct-u");
    expect(sentReasoning(served)).toEqual({ type: "reasoning", summary: [{ type: "summary_text", text: expect.stringMatching(/^thought \d+$/) }] });
  });

  test("a group target's ciphertext sent to a direct model is stripped on another key and forwarded on the same key", async (ctx) => {
    if (!app || !upstream) return ctx.skip();
    const { history, account } = await firstTurn("gd-grp", "group to direct");
    expect(account).toBe("acct-v");

    const other = await respond({ model: "gd-other", store: false, input: history });
    expect(other.status, other.text).toBe(200);
    const strippedAt = upstream.received.at(-1)!;
    expect(strippedAt.account).toBe("acct-w");
    expect(sentReasoning(strippedAt)).toEqual({ type: "reasoning", summary: [{ type: "summary_text", text: expect.stringMatching(/^thought \d+$/) }] });

    const peer = await respond({ model: "gd-peer", store: false, input: history });
    expect(peer.status, peer.text).toBe(200);
    const forwardedAt = upstream.received.at(-1)!;
    expect(forwardedAt.account).toBe("acct-v");
    expect(forwardedAt.body.model).toBe("gpt-gd-peer");
    expect(sentReasoning(forwardedAt).encrypted_content).toMatch(/^enc:acct-v:\d+$/);
  });
});
