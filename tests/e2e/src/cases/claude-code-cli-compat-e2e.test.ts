import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
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
// the hand-rolled harness does not.
//
// A `claude -p` invocation posts ~17KB to /v1/messages: a structured
// `system` array, a tool catalogue (~30 Anthropic input_schema blocks),
// `cache_control: {type: "ephemeral"}` markers, an `anthropic-beta`
// header, and a `?beta=true` query suffix. A regression in any of:
//
//   - /v1/messages query-string tolerance (`?beta=true`)
//   - x-api-key auth fallback (Authorization-Bearer is NOT sent)
//   - anthropic-beta / anthropic-version header passthrough vs. strip
//   - system: [{type:"text", text, cache_control?}] block parsing
//   - tools[].input_schema → upstream tools[].function.parameters
//     translation for non-Anthropic upstreams
//
// would break Claude Code customers but pass every existing test
// (none of which generate this request shape).
//
// Gating: real upstream costs money, so the suite is opt-in.
//   - RUN_CLAUDE_CODE_REAL_CHAIN=1 to acknowledge spend
//   - OPENAI_API_KEY or DEEPSEEK_API_KEY for the upstream credential
//   - `claude` CLI on PATH (skipped silently if not installed)
//
// References:
// - Anthropic Messages API:
//   https://docs.anthropic.com/en/api/messages
// - Anthropic Node SDK wire shape (the SDK Claude Code uses):
//   https://github.com/anthropics/anthropic-sdk-typescript
// - Claude Code CLI flags: `claude --help` (--bare strips OAuth and
//   keychain so auth is strictly $ANTHROPIC_API_KEY).

const CALLER_PLAINTEXT = "sk-claude-code-cli-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

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

function pickUpstream(): UpstreamProfile | undefined {
  return UPSTREAMS.find((u) => !!process.env[u.envVar]);
}

function claudeOnPath(): boolean {
  // `claude -h` is the cheapest probe; if the CLI isn't installed
  // spawnSync sets `error` and returns status null.
  const probe = spawnSync("claude", ["-h"], { encoding: "utf-8" });
  return probe.status === 0;
}

const RUN_REAL_CHAIN = process.env.RUN_CLAUDE_CODE_REAL_CHAIN === "1";
const MODEL_ALIAS = "claude-code-cli-e2e";

describe("claude code CLI compat: drive gateway through `claude -p`", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  let upstream: UpstreamProfile | undefined;
  let claudeAvailable = false;

  beforeAll(async () => {
    if (!RUN_REAL_CHAIN) return;
    upstream = pickUpstream();
    if (!upstream) return;
    claudeAvailable = claudeOnPath();
    if (!claudeAvailable) return;

    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "claude-code-cli-e2e-pk",
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      secret: process.env[upstream.envVar]!,
      api_base: upstream.apiBase,
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
  });

  test("`claude -p` against gateway returns a non-empty assistant reply", async (ctx) => {
    if (!RUN_REAL_CHAIN) {
      ctx.skip();
      return;
    }
    if (!upstream) {
      ctx.skip();
      return;
    }
    if (!claudeAvailable) {
      ctx.skip();
      return;
    }
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    // Snapshot propagation — poll via /v1/models so the model lookup
    // path the dispatcher uses is the same one the test waits on.
    // Mirrors the pattern in openai-sdk-compat.test.ts.
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
    const proxyUrl = app.proxyUrl;
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
      {
        env: {
          ...process.env,
          ANTHROPIC_BASE_URL: proxyUrl,
          ANTHROPIC_API_KEY: CALLER_PLAINTEXT,
          DISABLE_AUTOUPDATER: "1",
          CLAUDE_CODE_SIMPLE: "1",
        },
        encoding: "utf-8",
        timeout: 90_000,
      },
    );

    expect(
      result.status,
      `claude exited ${result.status}; stderr=${(result.stderr ?? "").slice(0, 2048)}`,
    ).toBe(0);
    expect(
      (result.stdout ?? "").trim(),
      "expected non-empty assistant output on stdout",
    ).not.toBe("");
    // The CLI surfaces gateway-side failures as `API Error: <status>`
    // on stderr. Status === 0 already implies no fatal error, but
    // pin stderr too so a CLI version that swallowed errors and
    // exited 0 anyway would still fail the test.
    expect(result.stderr ?? "").not.toMatch(/API Error:/);
  }, 120_000);
});
