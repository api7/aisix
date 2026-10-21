// Stored responses addressed by id: GET / DELETE /v1/responses/{id},
// GET /v1/responses/{id}/input_items and POST /v1/responses/{id}/cancel.
//
// A response a Model Group served comes back with an id that names the
// group's target. The caller must be able to use that id exactly as an
// OpenAI client would — retrieve it (also as a replayed stream), list its
// input items, cancel it and delete it — and every call must land on the
// account that stored the response, with the upstream's own id. The mock
// below stores responses per credential (one credential = one account) and
// answers an id it does not hold the way OpenAI does: 404, or 400 for an id
// longer than 64 characters.
//
// The id is not signed, so the caller can write one naming any model. Such
// an id must not reach an account the caller could not otherwise use.
import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMockSls,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type MockSls,
  type SpawnedApp,
} from "../harness/index.js";

const CALLER = "sk-resp-by-id-caller";
const OTHER = "sk-resp-by-id-other";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");
const LOGSTORE = "resp-by-id";
const CREDENTIAL_REF = "respbyid";

interface Received {
  account: string;
  method: string;
  path: string;
}

class AccountsUpstream {
  readonly received: Received[] = [];
  private seq = 0;
  private readonly stored = new Map<string, { account: string; response: any; input: any[] }>();
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
    const url = new URL(req.url ?? "/", "http://x");
    this.received.push({ account, method: req.method ?? "", path: url.pathname + url.search });

    if (req.method === "POST" && url.pathname === "/v1/responses") {
      const body = JSON.parse(raw);
      const n = ++this.seq;
      const id = `resp_${account}_${n}`;
      const response = {
        id,
        object: "response",
        created_at: 0,
        status: body.background ? "queued" : "completed",
        model: body.model,
        output: [
          {
            type: "reasoning",
            id: `rs_${account}_${n}`,
            summary: [{ type: "summary_text", text: "thought" }],
            encrypted_content: `enc:${account}:${n}`,
          },
          {
            type: "message",
            id: `msg_${account}_${n}`,
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: `answer from ${account}`, annotations: [] }],
          },
        ],
        usage: { input_tokens: 3, output_tokens: 2, total_tokens: 5 },
      };
      const input = typeof body.input === "string"
        ? [{ type: "message", id: `msg_in_${n}`, role: "user", content: [{ type: "input_text", text: body.input }] }]
        : body.input;
      this.stored.set(id, { account, response, input });
      return { status: 200, body: response };
    }

    if (req.method === "POST" && url.pathname.endsWith("/chat/completions")) {
      const body = JSON.parse(raw);
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

    const m = url.pathname.match(/^\/v1\/responses\/([^/]+)(\/input_items|\/cancel)?$/);
    if (!m) return { status: 404, body: { error: { message: "no route" } } };
    const id = decodeURIComponent(m[1]);
    if (id.length > 64) {
      return {
        status: 400,
        body: { error: { message: `Invalid 'response_id': string too long. Expected a string with maximum length 64, but got a string with length ${id.length} instead.`, type: "invalid_request_error" } },
      };
    }
    const entry = this.stored.get(id);
    if (!entry || entry.account !== account) {
      return { status: 404, body: { error: { message: `Response with id '${id}' not found.`, type: "invalid_request_error" } } };
    }
    if (m[2] === "/input_items" && req.method === "GET") {
      return {
        status: 200,
        body: { object: "list", data: entry.input, first_id: entry.input[0]?.id, last_id: entry.input.at(-1)?.id, has_more: false },
      };
    }
    if (m[2] === "/cancel" && req.method === "POST") {
      entry.response = { ...entry.response, status: "cancelled" };
      return { status: 200, body: entry.response };
    }
    if (m[2] === undefined && req.method === "DELETE") {
      this.stored.delete(id);
      return { status: 200, body: { id, object: "response", deleted: true } };
    }
    if (m[2] === undefined && req.method === "GET") {
      if (url.searchParams.get("stream") === "true") {
        const frame = (event: Record<string, unknown>) =>
          `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
        return {
          status: 200,
          sse:
            frame({ type: "response.created", response: { ...entry.response, status: "in_progress", output: [] } }) +
            frame({ type: "response.output_item.done", output_index: 0, item: entry.response.output[0] }) +
            frame({ type: "response.completed", response: entry.response }),
        };
      }
      return { status: 200, body: entry.response };
    }
    return { status: 405, body: { error: { message: "method not allowed" } } };
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

const forgeResponseId = (target: string, original: string) =>
  `resp_${Buffer.from(`aisix:v1;${target};${original}`).toString("base64url")}`;

function parseSse(text: string): any[] {
  return text
    .split(/\n\n/)
    .map((f) => f.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n"))
    .filter((d) => d && d !== "[DONE]")
    .map((d) => JSON.parse(d));
}

describe("/v1/responses/{id} retrieve, delete, input_items and cancel", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let upstream: AccountsUpstream | undefined;
  let sls: MockSls | undefined;
  let etcdReachable = false;
  const ids: Record<string, string> = {};

  async function model(name: string, pkId: string, extra: Record<string, unknown> = {}): Promise<void> {
    const m = await seed!.createModel({
      display_name: name,
      provider: "openai",
      model_name: `gpt-${name}`,
      provider_key_id: pkId,
      ...extra,
    });
    ids[name] = m.id;
  }

  async function group(name: string, targets: string[]): Promise<void> {
    const m = await seed!.createModel({
      display_name: name,
      routing: {
        strategy: "round_robin",
        targets: targets.map((t) => ({ model: t })),
        max_fallbacks: 0,
      },
    });
    ids[name] = m.id;
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await AccountsUpstream.start();
    sls = await startMockSls();
    app = await spawnApp({
      extraEnv: {
        RUST_LOG: "info",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
        [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
      },
    });
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seed.createObservabilityExporter({
      name: "sls-resp-by-id",
      enabled: true,
      kind: "aliyun_sls",
      endpoint: sls.url,
      project: "aisix-e2e",
      logstore: LOGSTORE,
      credential_ref: CREDENTIAL_REF,
      content_mode: "metadata_only",
    });

    const pk = async (account: string, extra: Record<string, unknown> = {}) =>
      (
        await seed!.createProviderKey({
          display_name: `pk-${account}`,
          secret: account,
          api_base: upstream!.baseUrl,
          ...extra,
        })
      ).id;
    await model("byid-a", await pk("acct-a"));
    await model("byid-b", await pk("acct-b"));
    await model("byid-solo", await pk("acct-solo"));
    await model("byid-hidden", await pk("acct-hidden"));
    const pkBridge = await pk("acct-bridge", { provider: "deepseek", adapter: "openai" });
    await model("byid-bridge", pkBridge, { provider: "deepseek" });
    await group("byid-group", ["byid-a", "byid-b"]);
    await group("byid-bridged", ["byid-bridge"]);
    await model("byid-fenced-target", await pk("acct-fenced"));
    // The group admits no loopback client; its target has no fence of its own.
    const fenced = await seed.createModel({
      display_name: "byid-fenced",
      allowed_cidrs: ["10.0.0.0/8"],
      routing: { strategy: "round_robin", targets: [{ model: "byid-fenced-target" }], max_fallbacks: 0 },
    });
    ids["byid-fenced"] = fenced.id;

    // The caller reaches the group's targets only through the group.
    await seed.createApiKey({
      key_hash: hash(OTHER),
      allowed_models: ["byid-hidden", "byid-fenced-target"],
    });
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["byid-group", "byid-solo", "byid-bridged", "byid-fenced"],
    });
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER}` },
      });
      return res.status === 200;
    });
  });

  afterAll(async () => {
    // No tagged id may ever reach an upstream.
    for (const r of upstream?.received ?? []) {
      expect(r.path).not.toMatch(/resp_YWlzaXg6djE7/);
    }
    await app?.exit();
    await upstream?.close();
    await sls?.close();
  });

  async function call(
    method: string,
    path: string,
    opts: { key?: string; body?: unknown; headers?: Record<string, string> } = {},
  ): Promise<{ status: number; json: any; text: string; requestId: string }> {
    const res = await fetch(`${app!.proxyUrl}${path}`, {
      method,
      headers: {
        authorization: `Bearer ${opts.key ?? CALLER}`,
        ...(opts.body ? { "content-type": "application/json" } : {}),
        ...opts.headers,
      },
      body: opts.body ? JSON.stringify(opts.body) : undefined,
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

  const create = async (model: string, key = CALLER, extra: Record<string, unknown> = {}) => {
    const r = await call("POST", "/v1/responses", { key, body: { model, input: "hello", store: true, ...extra } });
    expect(r.status, r.text).toBe(200);
    return r.json;
  };

  test("a group's response id retrieves, lists input, cancels and deletes on the account that stored it", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    // Round robin: two creates land on both targets.
    const created = [await create("byid-group"), await create("byid-group")];
    const origins = created.map((c) => decodeResponseId(c.id));
    expect(new Set(origins.map((o) => o?.target))).toEqual(new Set([ids["byid-a"], ids["byid-b"]]));

    for (const [i, c] of created.entries()) {
      const origin = origins[i]!;
      const account = origin.target === ids["byid-a"] ? "acct-a" : "acct-b";
      const before = upstream!.forAccount(account).length;

      const got = await call("GET", `/v1/responses/${c.id}`);
      expect(got.status, got.text).toBe(200);
      expect(got.json.id).toBe(c.id);
      expect(got.json.output[1].content[0].text).toBe(`answer from ${account}`);
      // Ciphertext comes back tagged exactly as the create returned it.
      expect(got.json.output[0].encrypted_content).toBe(c.output[0].encrypted_content);

      const streamed = await call("GET", `/v1/responses/${c.id}?stream=true`);
      expect(streamed.status, streamed.text).toBe(200);
      const events = parseSse(streamed.text);
      expect(events.map((e) => e.type)).toEqual([
        "response.created",
        "response.output_item.done",
        "response.completed",
      ]);
      expect(events[0].response.id).toBe(c.id);
      expect(events.at(-1).response.id).toBe(c.id);
      expect(events[1].item.encrypted_content).toBe(c.output[0].encrypted_content);

      const items = await call("GET", `/v1/responses/${c.id}/input_items?limit=5`);
      expect(items.status, items.text).toBe(200);
      expect(items.json.object).toBe("list");
      expect(items.json.data[0].content[0].text).toBe("hello");

      const cancelled = await call("POST", `/v1/responses/${c.id}/cancel`);
      expect(cancelled.status, cancelled.text).toBe(200);
      expect(cancelled.json.id).toBe(c.id);
      expect(cancelled.json.status).toBe("cancelled");

      const deleted = await call("DELETE", `/v1/responses/${c.id}`);
      expect(deleted.status, deleted.text).toBe(200);
      expect(deleted.json).toEqual({ id: c.id, object: "response", deleted: true });

      const gone = await call("GET", `/v1/responses/${c.id}`);
      expect(gone.status).toBe(404);

      // Every call reached the storing account, with the upstream's own id
      // and the client's query intact.
      const seen = upstream!.forAccount(account).slice(before);
      expect(seen.map((r) => `${r.method} ${r.path}`)).toEqual([
        `GET /v1/responses/${origin.original}`,
        `GET /v1/responses/${origin.original}?stream=true`,
        `GET /v1/responses/${origin.original}/input_items?limit=5`,
        `POST /v1/responses/${origin.original}/cancel`,
        `DELETE /v1/responses/${origin.original}`,
        `GET /v1/responses/${origin.original}`,
      ]);
    }
  });

  test("an untagged id needs the model named, and then uses it", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const c = await create("byid-solo");
    expect(decodeResponseId(c.id)).toBeUndefined();
    const before = upstream!.forAccount("acct-solo").length;

    const bare = await call("GET", `/v1/responses/${c.id}`);
    expect(bare.status, bare.text).toBe(400);
    expect(bare.json.error.message).toMatch(/model/);
    expect(upstream!.forAccount("acct-solo").length).toBe(before);
    // A group served nothing itself; the model that did must be named.
    const byGroup = await call("GET", `/v1/responses/${c.id}?model=byid-group`);
    expect(byGroup.status, byGroup.text).toBe(400);
    expect(upstream!.forAccount("acct-solo").length).toBe(before);

    const byQuery = await call("GET", `/v1/responses/${c.id}?model=byid-solo`);
    expect(byQuery.status, byQuery.text).toBe(200);
    expect(byQuery.json.id).toBe(c.id);
    const byHeader = await call("DELETE", `/v1/responses/${c.id}`, { headers: { "x-aisix-model": "byid-solo" } });
    expect(byHeader.status, byHeader.text).toBe(200);
    // The routing hint stays at the gateway.
    expect(upstream!.forAccount("acct-solo").slice(before).map((r) => `${r.method} ${r.path}`)).toEqual([
      `GET /v1/responses/${c.id}`,
      `DELETE /v1/responses/${c.id}`,
    ]);

    // A named model the key may not use is refused before any upstream.
    const other = await create("byid-hidden", OTHER);
    const refused = await call("GET", `/v1/responses/${other.id}?model=byid-hidden`);
    expect(refused.status, refused.text).toBe(403);
  });

  test("a forged id naming a target the caller cannot reach goes nowhere", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    // A response another key stored on an account this caller has no model for.
    const stored = await create("byid-hidden", OTHER);
    const before = upstream!.forAccount("acct-hidden").length;
    const forged = forgeResponseId(ids["byid-hidden"], stored.id);

    for (const [method, suffix] of [
      ["GET", ""],
      ["DELETE", ""],
      ["GET", "/input_items"],
      ["POST", "/cancel"],
    ] as const) {
      const r = await call(method, `/v1/responses/${forged}${suffix}`);
      expect(r.status, `${method} ${suffix}: ${r.text}`).toBe(404);
    }
    // A tag naming no model at all answers the same.
    const unknown = await call("GET", `/v1/responses/${forgeResponseId("no-such-model", stored.id)}`);
    expect(unknown.status).toBe(404);
    expect(upstream!.forAccount("acct-hidden").length).toBe(before);

    // The response is still there for the key that can reach it.
    const owner = await call("GET", `/v1/responses/${stored.id}?model=byid-hidden`, { key: OTHER });
    expect(owner.status, owner.text).toBe(200);
  });

  test("a group's client-IP fence also fences the ids its targets produced", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    // The caller's only way to the target is the fenced group, which does not
    // admit this client; a request naming the group is refused.
    const viaGroup = await call("POST", "/v1/responses", { body: { model: "byid-fenced", input: "hi" } });
    expect(viaGroup.status, viaGroup.text).toBe(403);
    const stored = await create("byid-fenced-target", OTHER);
    const before = upstream!.forAccount("acct-fenced").length;
    const r = await call("GET", `/v1/responses/${forgeResponseId(ids["byid-fenced-target"], stored.id)}`);
    expect(r.status, r.text).toBe(404);
    expect(upstream!.forAccount("acct-fenced").length).toBe(before);
  });

  test("listed input items carry ciphertext the caller can send back through the group", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const first = await create("byid-group");
    const reasoning = first.output.find((i: any) => i.type === "reasoning");
    const followUp = await create("byid-group", CALLER, {
      input: [reasoning, { type: "message", role: "user", content: "and then?" }],
    });
    expect(decodeResponseId(followUp.id)!.target).toBe(decodeResponseId(first.id)!.target);
    const items = await call("GET", `/v1/responses/${followUp.id}/input_items`);
    expect(items.status, items.text).toBe(200);
    const listed = items.json.data.find((i: any) => i.type === "reasoning");
    expect(listed.encrypted_content).toBe(reasoning.encrypted_content);
  });

  test("a bridged target's id has nothing stored upstream: 404", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const c = await create("byid-bridged");
    expect(decodeResponseId(c.id)?.target).toBe(ids["byid-bridge"]);
    const before = upstream!.forAccount("acct-bridge").length;
    const r = await call("GET", `/v1/responses/${c.id}`);
    expect(r.status, r.text).toBe(404);
    expect(upstream!.forAccount("acct-bridge").length).toBe(before);
  });

  test("a by-id call writes an access-log line and no usage row", async (ctx) => {
    if (!etcdReachable) return ctx.skip();
    const c = await create("byid-group");
    const got = await call("GET", `/v1/responses/${c.id}`);
    expect(got.status, got.text).toBe(200);
    expect(got.requestId).not.toBe("");
    const access = await waitForLogLine(
      app!,
      (l) => l.includes("proxy request completed") && l.includes(`request_id="${got.requestId}"`),
      "the retrieve's access-log line",
    );
    expect(access).toContain("status=200");
    expect(access).toContain(`/v1/responses/${c.id}`);

    // A metered request after it: once its row is exported, a row for the
    // retrieve would have been too.
    const created = await call("POST", "/v1/responses", { body: { model: "byid-group", input: "barrier" } });
    expect(created.status, created.text).toBe(200);
    await waitForSlsLog(sls!, LOGSTORE, (log) => log.get("request_id") === created.requestId, "the barrier's row");
    await expect(
      waitForSlsLog(sls!, LOGSTORE, (log) => log.get("request_id") === got.requestId, "a retrieve row", 1_000),
    ).rejects.toThrow();
  });
});
