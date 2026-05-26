import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createServer, request as httpRequest, type Server } from "node:http";
import { request as httpsRequest } from "node:https";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  pickFreePort,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E client-compat: drive the gateway's Anthropic-native /v1/messages
// endpoint through the REAL Claude Code CLI (`claude -p`), pointed at
// a real upstream LLM through ANTHROPIC_BASE_URL.
//
// Sibling to openai-sdk-compat.test.ts. That file proves the gateway
// speaks OpenAI Chat Completions to OpenAI-shape clients. This one
// proves it speaks Anthropic Messages to the wire shape an
// Anthropic-ecosystem client (Claude Code CLI, Claude Desktop, and the
// official anthropic-sdk-{python,typescript}) actually emits — which
// the hand-rolled harness does not generate.
//
// A `claude -p` invocation posts ~17KB to /v1/messages: a structured
// `system` array, a tool catalogue (~30 Anthropic input_schema blocks),
// `cache_control: {type: "ephemeral"}` markers, an `anthropic-beta`
// header, and a `?beta=true` query suffix.
//
// To pin that the gateway both *received* this and *translated it
// correctly* to the upstream wire shape, the test inserts a recording
// reverse-proxy between the gateway and the real upstream:
//
//   Claude Code CLI ──► gateway ──► recording proxy ──► real upstream
//                                         │
//                                         ▼
//                                   captured request
//
// The proxy verbatim-forwards the gateway's outbound request to the
// real upstream and returns the upstream's response. Captured requests
// expose the gateway's cross-provider translation product, so a
// regression that mangled the translation surfaces in the assertions
// (not just "the CLI exited 0 because the upstream tolerated garbage").
//
// Gating: real upstream costs money, so the suite is opt-in.
//   - RUN_CLAUDE_CODE_REAL_CHAIN=1 to acknowledge spend
//   - OPENAI_API_KEY or DEEPSEEK_API_KEY for the upstream credential
//   - `claude` CLI on PATH with `--bare`, `--model`, `--max-budget-usd`,
//     `--disable-slash-commands` available (probed at beforeAll)
//
// When RUN_CLAUDE_CODE_REAL_CHAIN=1 is set, any missing prerequisite
// (no upstream key, no CLI, no required flag, no etcd) throws hard at
// beforeAll — a "skip" in opt-in CI mode would silently turn the
// workflow green without exercising the contract.
//
// References:
// - Anthropic Messages API:
//   https://docs.anthropic.com/en/api/messages
// - OpenAI Chat Completions API (the cross-provider target shape):
//   https://platform.openai.com/docs/api-reference/chat/create
// - Anthropic Node SDK wire shape (the SDK Claude Code uses):
//   https://github.com/anthropics/anthropic-sdk-typescript
// - Claude Code CLI flags: `claude --help` (--bare strips OAuth and
//   keychain so auth is strictly $ANTHROPIC_API_KEY).
// - Claude Code CLI install:
//   https://code.claude.com/docs/en/setup

const CALLER_PLAINTEXT = "sk-claude-code-cli-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const REQUIRED_CLAUDE_FLAGS = [
  "--bare",
  "--model",
  "--max-budget-usd",
  "--disable-slash-commands",
] as const;

const SECRET_PATTERNS: ReadonlyArray<RegExp> = [
  /sk-[A-Za-z0-9_\-]{8,}/g,
  /Bearer\s+[A-Za-z0-9_\-.]+/gi,
];

function redactSecrets(s: string): string {
  let out = s;
  for (const pat of SECRET_PATTERNS) {
    out = out.replace(pat, "[REDACTED]");
  }
  return out;
}

interface UpstreamProfile {
  readonly envVar: "OPENAI_API_KEY" | "DEEPSEEK_API_KEY";
  readonly provider: "openai" | "deepseek";
  readonly upstreamModel: string;
  readonly apiBase: string;
}

const UPSTREAMS: ReadonlyArray<UpstreamProfile> = [
  {
    envVar: "OPENAI_API_KEY",
    provider: "openai",
    upstreamModel: "gpt-4o-mini",
    apiBase: "https://api.openai.com/v1",
  },
  {
    envVar: "DEEPSEEK_API_KEY",
    provider: "deepseek",
    upstreamModel: "deepseek-chat",
    apiBase: "https://api.deepseek.com/v1",
  },
];

// Honor an explicit override so CI matrices can pin a single upstream
// deterministically even when more than one credential is in scope.
function pickUpstream(): UpstreamProfile | undefined {
  const pin = process.env.REAL_CHAIN_UPSTREAM?.toLowerCase();
  if (pin) {
    const explicit = UPSTREAMS.find((u) => u.provider === pin);
    if (explicit && process.env[explicit.envVar]) return explicit;
  }
  return UPSTREAMS.find((u) => !!process.env[u.envVar]);
}

interface ClaudeProbe {
  readonly available: boolean;
  readonly missingFlags: ReadonlyArray<string>;
}

function probeClaude(): ClaudeProbe {
  const help = spawnSync("claude", ["--help"], { encoding: "utf-8" });
  if (help.status !== 0) {
    return { available: false, missingFlags: [...REQUIRED_CLAUDE_FLAGS] };
  }
  const text = `${help.stdout ?? ""}\n${help.stderr ?? ""}`;
  const missing = REQUIRED_CLAUDE_FLAGS.filter((f) => !text.includes(f));
  return { available: true, missingFlags: missing };
}

interface CapturedRequest {
  method: string;
  path: string;
  search: string;
  headers: Record<string, string>;
  body: string;
}

interface RecordingProxy {
  baseUrl: string;
  received: CapturedRequest[];
  close(): Promise<void>;
}

// Forward-proxy that records every inbound request and verbatim-relays
// it to `upstreamBase` (https). The gateway treats this as its real
// upstream — its outbound translated body and headers land in
// `received[…].body / headers` for later assertion.
async function startRecordingProxy(
  upstreamBase: string,
): Promise<RecordingProxy> {
  const url = new URL(upstreamBase);
  const isHttps = url.protocol === "https:";
  const upstreamHost = url.hostname;
  const upstreamPort = url.port ? Number(url.port) : isHttps ? 443 : 80;
  const upstreamPath = url.pathname.replace(/\/$/, "");

  const received: CapturedRequest[] = [];

  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (chunk: Buffer) => {
      raw += chunk.toString("utf8");
    });
    req.on("end", () => {
      const incomingUrl = new URL(req.url ?? "/", "http://placeholder");
      received.push({
        method: req.method ?? "GET",
        path: incomingUrl.pathname,
        search: incomingUrl.search,
        headers: Object.fromEntries(
          Object.entries(req.headers).map(([k, v]) => [
            k,
            Array.isArray(v) ? v.join(",") : (v ?? ""),
          ]),
        ),
        body: raw,
      });

      // Strip hop-by-hop headers and rewrite Host before relaying.
      const fwdHeaders: Record<string, string | string[]> = {};
      for (const [k, v] of Object.entries(req.headers)) {
        if (v === undefined) continue;
        const lk = k.toLowerCase();
        if (
          lk === "host" ||
          lk === "connection" ||
          lk === "content-length" ||
          lk === "transfer-encoding"
        ) {
          continue;
        }
        fwdHeaders[k] = v;
      }
      fwdHeaders.host = upstreamHost;

      const relayPath = `${upstreamPath}${incomingUrl.pathname.replace(/^\/v1/, "")}${incomingUrl.search}`;
      const relay = (isHttps ? httpsRequest : httpRequest)(
        {
          host: upstreamHost,
          port: upstreamPort,
          path: relayPath,
          method: req.method ?? "GET",
          headers: fwdHeaders,
        },
        (upRes) => {
          res.statusCode = upRes.statusCode ?? 502;
          for (const [k, v] of Object.entries(upRes.headers)) {
            if (v !== undefined) res.setHeader(k, v);
          }
          upRes.pipe(res);
        },
      );
      relay.on("error", (err) => {
        res.statusCode = 502;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify({ error: { message: `proxy: ${err.message}` } }));
      });
      if (raw) relay.write(raw);
      relay.end();
    });
  });

  const port = await pickFreePort();
  await new Promise<void>((resolve) =>
    server.listen(port, "127.0.0.1", resolve),
  );

  return {
    baseUrl: `http://127.0.0.1:${port}/v1`,
    received,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

const RUN_REAL_CHAIN = process.env.RUN_CLAUDE_CODE_REAL_CHAIN === "1";
const MODEL_ALIAS = "claude-code-cli-e2e";

describe("claude code CLI compat: drive gateway through `claude -p`", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let proxy: RecordingProxy | undefined;
  let etcdReachable = false;
  let upstream: UpstreamProfile | undefined;
  let claudeProbe: ClaudeProbe = { available: false, missingFlags: [] };

  beforeAll(async () => {
    if (!RUN_REAL_CHAIN) return;

    // In real-chain mode every prerequisite is required. A `ctx.skip()`
    // here would let the workflow exit 0 with the contract un-pinned.
    upstream = pickUpstream();
    if (!upstream) {
      throw new Error(
        "RUN_CLAUDE_CODE_REAL_CHAIN=1 requires OPENAI_API_KEY or DEEPSEEK_API_KEY",
      );
    }

    claudeProbe = probeClaude();
    if (!claudeProbe.available) {
      throw new Error(
        "RUN_CLAUDE_CODE_REAL_CHAIN=1 requires `claude` on PATH; install via https://code.claude.com/docs/en/setup",
      );
    }
    if (claudeProbe.missingFlags.length > 0) {
      throw new Error(
        `installed \`claude\` is missing required flags: ${claudeProbe.missingFlags.join(", ")}; upgrade or pin a known-good version`,
      );
    }

    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) {
      throw new Error(
        "RUN_CLAUDE_CODE_REAL_CHAIN=1 requires etcd reachable at AISIX_E2E_ETCD",
      );
    }

    proxy = await startRecordingProxy(upstream.apiBase);

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "claude-code-cli-e2e-pk",
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      secret: process.env[upstream.envVar]!,
      // Point the gateway at the recording proxy. The proxy verbatim-
      // forwards to the real upstream and captures the outbound body
      // for the wire-shape assertions below.
      api_base: proxy.baseUrl,
    });
    await admin.createModel({
      display_name: MODEL_ALIAS,
      provider: upstream.provider,
      model_name: upstream.upstreamModel,
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [MODEL_ALIAS],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await proxy?.close();
  });

  test("`claude -p` against gateway → real upstream — round-trip + cross-provider translation", async (ctx) => {
    if (!RUN_REAL_CHAIN) {
      ctx.skip();
      return;
    }
    // beforeAll throws (not skips) if anything is missing in CI mode,
    // so we either have everything or the suite already aborted. The
    // narrowing assertions below let strict mode reason about it.
    if (!app || !proxy || !upstream) {
      throw new Error("beforeAll did not initialize the fixture");
    }

    await waitConfigPropagation(async () => {
      try {
        const r = await fetch(`${app!.proxyUrl}/v1/models`, {
          headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
        });
        if (!r.ok) return false;
        const body = (await r.json()) as { data?: Array<{ id?: string }> };
        return (body.data ?? []).some((m) => m.id === MODEL_ALIAS);
      } catch {
        return false;
      }
    });

    // Baseline-isolate the readiness probe so wire-shape assertions
    // match the test call's request, not the probe's.
    const baseline = proxy.received.length;

    // `claude -p` runs print mode (non-interactive, single completion).
    //
    // `--bare` strips OAuth / keychain / hooks / skills / CLAUDE.md so
    // auth is strictly ANTHROPIC_API_KEY — otherwise the developer's
    // local OAuth session wins and our caller key is silently ignored.
    //
    // `--max-budget-usd 0.5` is a belt-and-suspenders cap; the prompt
    // is small so a single completion costs fractions of a cent.
    //
    // `--disable-slash-commands` keeps the CLI from loading user
    // skills at startup (additional latency, no value to the test).
    //
    // Minimal env passthrough: only the variables Claude Code needs
    // to function. Avoids leaking unrelated CI secrets into the
    // subprocess's diagnostic surfaces.
    const minimalEnv: Record<string, string> = {
      PATH: process.env.PATH ?? "",
      HOME: process.env.HOME ?? "",
      ANTHROPIC_BASE_URL: app.proxyUrl,
      ANTHROPIC_API_KEY: CALLER_PLAINTEXT,
      DISABLE_AUTOUPDATER: "1",
      CLAUDE_CODE_SIMPLE: "1",
    };
    const result = spawnSync(
      "claude",
      [
        "-p",
        "Reply with exactly the single word: ready",
        "--bare",
        "--model",
        MODEL_ALIAS,
        "--max-budget-usd",
        "0.5",
        "--disable-slash-commands",
      ],
      { env: minimalEnv, encoding: "utf-8", timeout: 90_000 },
    );

    // Branch on signal first — `spawnSync` returns `status: null`,
    // `signal: "SIGTERM"` on timeout. Without this, the assertion
    // below fails as "expected null to be 0" and hides the real
    // failure mode (timeout) under what looks like a clean exit
    // mismatch.
    if (result.signal) {
      throw new Error(
        `claude killed by ${result.signal} after 90s (likely gateway or upstream hang); stderr=${redactSecrets((result.stderr ?? "").slice(0, 1024))}`,
      );
    }

    const safeStderr = redactSecrets((result.stderr ?? "").slice(0, 2048));
    expect(
      result.status,
      `claude exited ${result.status}; stderr=${safeStderr}`,
    ).toBe(0);
    const stdout = (result.stdout ?? "").trim();
    // A useful response is non-empty and free of recognisable error
    // shapes — strong enough to fail when the gateway short-circuits
    // with an error envelope that the CLI surfaces as a benign-
    // looking string instead of a non-zero exit code.
    expect(stdout, "expected non-empty assistant output").not.toBe("");
    expect(stdout.length, "expected substantive response").toBeGreaterThan(2);
    expect(
      stdout,
      `assistant output looks like an error envelope: ${stdout.slice(0, 200)}`,
    ).not.toMatch(/\b(error|denied|forbidden|unauthorized|not found|invalid)\b/i);
    expect(safeStderr).not.toMatch(/API Error:/);

    // Wire-shape pinning at the proxy: the gateway must have emitted
    // OpenAI-compat chat completions for the OpenAI/DeepSeek upstream
    // (both providers share the OpenAI wire shape).
    //
    // Without these assertions the test only proves the round-trip
    // completed — a regression that dropped tool definitions on
    // translation or sent an Anthropic-shape body to the OpenAI
    // upstream could still pass (the upstream might 200 on a
    // partially-mangled body and the CLI render whatever text
    // came back).
    const upstreamRequests = proxy.received.slice(baseline);
    expect(
      upstreamRequests.length,
      "gateway should have forwarded at least one upstream call",
    ).toBeGreaterThan(0);

    const chat = upstreamRequests.find((r) =>
      r.path.includes("/chat/completions"),
    );
    expect(
      chat,
      `no /chat/completions in upstream calls; saw ${upstreamRequests.map((r) => r.path).join(", ")}`,
    ).toBeDefined();
    expect(chat!.method).toBe("POST");
    // Gateway → real upstream auth uses Bearer (the provider key's
    // secret), not the Anthropic-shape x-api-key the CLI sent inbound.
    expect(
      chat!.headers.authorization ?? "",
      "gateway must auth to upstream with Bearer + provider-key secret",
    ).toMatch(/^Bearer\s+\S+/);

    const sentBody = JSON.parse(chat!.body) as {
      model?: string;
      messages?: Array<{ role?: string; content?: unknown }>;
      tools?: Array<{ type?: string; function?: { name?: string } }>;
    };
    // Upstream sees the provider-side model id (gpt-4o-mini /
    // deepseek-chat), not the gateway's display alias.
    expect(sentBody.model).toBe(upstream.upstreamModel);
    // The user prompt must round-trip through the gateway's
    // Anthropic-→-OpenAI body translation.
    const userMsgs = (sentBody.messages ?? []).filter((m) => m.role === "user");
    expect(userMsgs.length).toBeGreaterThan(0);
    const promptText = JSON.stringify(userMsgs).toLowerCase();
    expect(promptText).toContain("ready");
    // The CLI sends ~30 Anthropic-shape tools. The gateway must
    // translate them to the OpenAI `{type: "function", function: {…}}`
    // shape; a regression that dropped or mistranslated tool blocks
    // surfaces here.
    expect(
      (sentBody.tools ?? []).length,
      "gateway must translate Claude Code tool catalogue to OpenAI tools[]",
    ).toBeGreaterThan(0);
    const firstTool = sentBody.tools![0];
    expect(firstTool.type).toBe("function");
    expect(typeof firstTool.function?.name).toBe("string");
  }, 180_000);
});
