import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startMcpUpstream,
  startMockSls,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForSlsLog,
  type McpUpstream,
  type MockSls,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// A request a guardrail blocks exports its body masked under `content_mode:
// full`, yet what its row says the guardrails did must not depend on that: a
// row saying `masked` here would land in the console's Rewritten view only on
// deployments with a full-content exporter. So the same blocked request goes
// through two gateways, differing only in the one exporter's mode.

const CALLER = "sk-block-mask-parity";
const CREDENTIAL_REF = "mock";
const SECRET = "parity-secret-4711";
const BLOCK_WORD = "parityblockword";
const TEXT = `say ${BLOCK_WORD} about ${SECRET}`;

type Mode = "full" | "metadata_only";
type Row = Map<string, string>;

const json = (path: string, body: unknown): Request =>
  new Request(`http://gateway${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "anthropic-version": "2023-06-01",
      accept: "application/json, text/event-stream",
    },
    body: JSON.stringify(body),
  });
const multipart = (path: string, fields: Record<string, string>, fileField: string): Request => {
  const form = new FormData();
  for (const [k, v] of Object.entries(fields)) form.set(k, v);
  form.set(fileField, new Blob(["sentinel-bytes"]), "in.bin");
  return new Request(`http://gateway${path}`, { method: "POST", body: form });
};

const REQUESTS: Array<[string, () => Request]> = [
  ["images", () => json("/v1/images/generations", { model: "parity-openai", prompt: TEXT })],
  ["embeddings", () => json("/v1/embeddings", { model: "parity-openai", input: TEXT })],
  ["rerank", () => json("/v1/rerank", { model: "parity-rerank", query: TEXT, documents: ["a"] })],
  ["completions", () => json("/v1/completions", { model: "parity-openai", prompt: TEXT })],
  ["speech", () => json("/v1/audio/speech", { model: "parity-openai", input: TEXT, voice: "alloy" })],
  [
    "count_tokens",
    () =>
      json("/v1/messages/count_tokens", {
        model: "parity-anthropic",
        messages: [{ role: "user", content: TEXT }],
      }),
  ],
  [
    "image edits",
    () => multipart("/v1/images/edits", { model: "parity-openai", prompt: TEXT }, "image"),
  ],
  [
    "transcription",
    () => multipart("/v1/audio/transcriptions", { model: "parity-openai", prompt: TEXT }, "file"),
  ],
  [
    "mcp",
    () =>
      json("/mcp", {
        jsonrpc: "2.0",
        id: 1,
        method: "tools/call",
        params: { name: "paritymcp__echo", arguments: { text: TEXT } },
      }),
  ],
];

describe("what a blocked row says the guardrails did ignores the exporter's mode", () => {
  let etcdReachable = false;
  let sls: MockSls | undefined;
  let upstream: OpenAiUpstream | undefined;
  let mcp: McpUpstream | undefined;
  const apps = new Map<Mode, SpawnedApp>();

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    sls = await startMockSls();
    upstream = await startOpenAiUpstream({ status: 400, errorBody: { error: { message: "unused" } } });
    mcp = await startMcpUpstream("paritymcp");

    for (const mode of ["full", "metadata_only"] as const) {
      const app = await spawnApp({
        extraEnv: {
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
        },
      });
      apps.set(mode, app);
      const seed = new SeedClient(etcd, app.etcdPrefix);
      await seed.createObservabilityExporter({
        name: `parity-${mode}`,
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: "aisix-e2e-obs",
        logstore: mode,
        credential_ref: CREDENTIAL_REF,
        content_mode: mode,
      });
      for (const [name, provider, model_name, base, adapter] of [
        ["parity-openai", "openai", "gpt-4o-mini", `${upstream.baseUrl}/v1`, undefined],
        ["parity-rerank", "cohere", "rerank-v3.5", `${upstream.baseUrl}/v1`, "openai"],
        ["parity-anthropic", "anthropic", "claude-sonnet-4-5", upstream.baseUrl, undefined],
      ] as const) {
        const pk = await seed.createProviderKey({
          display_name: `${name}-pk`,
          secret: "sk-mock",
          provider,
          ...(adapter ? { adapter } : {}),
          api_base: base,
        });
        await seed.createModel({
          display_name: name,
          provider,
          model_name,
          provider_key_id: pk.id,
        });
      }
      const mcpId = randomUUID();
      await seed.update("mcp_servers", mcpId, {
        display_name: "paritymcp",
        url: mcp.url,
        enabled: true,
      });
      for (const guardrail of [
        {
          name: "parity-mask",
          enabled: true,
          hook_point: "input",
          kind: "pii",
          custom_patterns: [
            { name: "parity_secret", regex: "parity-secret-\\d+", action: "mask", replacement: "***" },
          ],
        },
        {
          name: "parity-block",
          enabled: true,
          hook_point: "input",
          kind: "keyword",
          patterns: [{ kind: "literal", value: BLOCK_WORD }],
        },
      ]) {
        const created = await seed.createGuardrail(guardrail);
        await seed.update("guardrail_attachments", randomUUID(), {
          guardrail_id: created.id,
          scope_type: "mcp_server",
          scope_id: mcpId,
          priority: 100,
        });
      }
      await seed.createApiKey({
        key_hash: createHash("sha256").update(CALLER).digest("hex"),
        allowed_models: ["*"],
        allowed_mcp_servers: ["*"],
      });
      await waitConfigPropagation(async () => {
        const res = await fetch(`${app.proxyUrl}/v1/models`, {
          headers: { authorization: `Bearer ${CALLER}` },
        });
        await res.arrayBuffer();
        return res.status === 200;
      });
    }
  }, 120_000);

  afterAll(async () => {
    for (const app of apps.values()) await app.exit();
    await sls?.close();
    await upstream?.close();
    await mcp?.close();
  });

  async function send(mode: Mode, make: () => Request): Promise<Row> {
    const req = make();
    const headers = new Headers(req.headers);
    headers.set("authorization", `Bearer ${CALLER}`);
    const res = await fetch(`${apps.get(mode)!.proxyUrl}${new URL(req.url).pathname}`, {
      method: "POST",
      headers,
      body: await req.arrayBuffer(),
    });
    await res.text();
    const requestId = res.headers.get("x-aisix-request-id");
    expect(requestId).toBeTruthy();
    return waitForSlsLog(sls!, mode, (r) => r.get("request_id") === requestId, `the ${mode} row`);
  }

  /** The record of what the guardrails did, without its timings. */
  const verdicts = (row: Row) => ({
    blocked: row.get("guardrail_blocked"),
    // Members of equal priority run in row-id order, which each seed draws anew.
    applied: (JSON.parse(row.get("applied_guardrails") ?? "[]") as unknown[])
      .map((a) => JSON.stringify(a))
      .sort(),
    counts: row.get("redacted_entity_counts"),
    hits: (JSON.parse(row.get("guardrail_enforced_hits") ?? "[]") as Array<Record<string, unknown>>)
      .map(({ guardrail_name, hook, action, error_type, entity_counts }) => ({
        guardrail_name,
        hook,
        action,
        error_type,
        entity_counts,
      }))
      .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))),
  });

  test.for(REQUESTS)("%s: both modes record the same verdicts", { timeout: 30_000 }, async ([label, make], ctx) => {
    if (!etcdReachable) return ctx.skip();
    const full = await send("full", make);
    const meta = await send("metadata_only", make);

    expect(full.get("guardrail_blocked"), label).toBe("true");
    const recorded = verdicts(full);
    expect(recorded.hits.map((h) => h.action), label).toContain("blocked");
    expect(recorded).toEqual(verdicts(meta));

    // The export itself is still the masked one.
    const exported = [...full.values()].join("\n");
    expect(exported, label).toContain(BLOCK_WORD);
    expect(exported, label).not.toContain(SECRET);
    expect([...meta.values()].join("\n"), label).not.toContain(BLOCK_WORD);
  });
});
