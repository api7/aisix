import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for #554: timeout-triggered fallback.
//
// A "virtual" routing Model fails over to its next target not only on an
// upstream error / 5xx (covered by fallback-e2e) but also when the primary
// is too SLOW:
//   - non-streaming `timeout` — the whole upstream call must finish in time;
//   - streaming `stream_timeout` — each chunk (the first one and every
//     inter-chunk gap) must arrive in time. A first-chunk timeout fails
//     over before any bytes reach the client; a mid-stream stall terminates
//     the stream like any other upstream error (no fallback once committed).
//
// Exercises the `timeout` + `stream_timeout` knobs. Reference: OpenAI
// Chat Completions spec for the request/response shape.

const CALLER_PLAINTEXT = "sk-timeout-fb-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

const SLOW_MS = 3000;
const TIMEOUT_MS = 300;
const GENEROUS_MS = 10_000;

function reply(content: string): unknown {
  return {
    id: `cmpl-${content}`,
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
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

function chunk(content: string): string {
  return JSON.stringify({
    id: "evt",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  });
}

function finish(): string {
  return JSON.stringify({
    id: "evt",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  });
}

function streamFor(content: string): string[] {
  return [chunk(content), finish(), "[DONE]"];
}

describe("timeout fallback e2e (#554)", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  // Non-streaming upstreams.
  let nsSlow: OpenAiUpstream | undefined;
  let nsFast: OpenAiUpstream | undefined;
  let nsBackup: OpenAiUpstream | undefined;
  // Streaming upstreams.
  let stSlow: OpenAiUpstream | undefined;
  let stFast: OpenAiUpstream | undefined;
  let stStall: OpenAiUpstream | undefined;
  let stBackup: OpenAiUpstream | undefined;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    nsSlow = await startOpenAiUpstream({
      responseDelayMs: SLOW_MS,
      nonStreamBody: reply("ns-slow"),
    });
    nsFast = await startOpenAiUpstream({ nonStreamBody: reply("ns-fast") });
    nsBackup = await startOpenAiUpstream({ nonStreamBody: reply("ns-backup") });

    stSlow = await startOpenAiUpstream({
      // Headers fast, first token slow → TTFT timeout.
      firstEventDelayMs: SLOW_MS,
      streamEvents: streamFor("st-slow"),
    });
    stFast = await startOpenAiUpstream({ streamEvents: streamFor("st-fast") });
    stStall = await startOpenAiUpstream({
      // First chunk fast, then a long inter-chunk gap → mid-stream read
      // timeout AFTER the 200 is committed (no fallback possible).
      eventDelayMs: SLOW_MS,
      streamEvents: [chunk("stall-1 "), chunk("stall-2 "), finish(), "[DONE]"],
    });
    stBackup = await startOpenAiUpstream({
      streamEvents: streamFor("st-backup"),
    });

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = async (name: string, u: OpenAiUpstream) =>
      (
        await admin!.createProviderKey({
          display_name: name,
          secret: "sk-mock",
          api_base: `${u.baseUrl}/v1`,
        })
      ).id;

    const nsSlowPk = await pk("t-ns-slow-pk", nsSlow);
    const nsFastPk = await pk("t-ns-fast-pk", nsFast);
    const nsBackupPk = await pk("t-ns-backup-pk", nsBackup);
    const stSlowPk = await pk("t-st-slow-pk", stSlow);
    const stFastPk = await pk("t-st-fast-pk", stFast);
    const stStallPk = await pk("t-st-stall-pk", stStall);
    const stBackupPk = await pk("t-st-backup-pk", stBackup);

    // Direct targets. Cooldown is disabled on the slow/stall primaries so a
    // timeout doesn't take them out of rotation between probe and test call
    // (mirrors fallback-e2e).
    await admin.createModel({
      display_name: "t-m-ns-slow",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: nsSlowPk,
      timeout: TIMEOUT_MS,
      cooldown: { enabled: false },
    });
    await admin.createModel({
      display_name: "t-m-ns-fast",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: nsFastPk,
      timeout: GENEROUS_MS,
    });
    await admin.createModel({
      display_name: "t-m-ns-backup",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: nsBackupPk,
    });
    await admin.createModel({
      display_name: "t-m-st-slow",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stSlowPk,
      stream_timeout: TIMEOUT_MS,
      cooldown: { enabled: false },
    });
    await admin.createModel({
      display_name: "t-m-st-fast",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stFastPk,
      stream_timeout: GENEROUS_MS,
    });
    await admin.createModel({
      display_name: "t-m-st-stall",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stStallPk,
      stream_timeout: TIMEOUT_MS,
      cooldown: { enabled: false },
    });
    await admin.createModel({
      display_name: "t-m-st-backup",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: stBackupPk,
    });

    const router = async (name: string, targets: string[]) =>
      admin!.createModel({
        display_name: name,
        routing: { strategy: "failover", targets: targets.map((model) => ({ model })) },
      });
    await router("t-r-ns-timeout", ["t-m-ns-slow", "t-m-ns-backup"]);
    await router("t-r-ns-fast", ["t-m-ns-fast", "t-m-ns-backup"]);
    await router("t-r-st-ttft", ["t-m-st-slow", "t-m-st-backup"]);
    await router("t-r-st-fast", ["t-m-st-fast", "t-m-st-backup"]);
    await router("t-r-st-stall", ["t-m-st-stall", "t-m-st-backup"]);

    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [
        "t-r-ns-timeout",
        "t-r-ns-fast",
        "t-r-st-ttft",
        "t-r-st-fast",
        "t-r-st-stall",
        "t-m-ns-backup",
      ],
    });

    // Gate on a routing model so the test calls can't fire before the
    // dispatcher has loaded the virtual models (and, transitively, their
    // direct targets — all written before this poll).
    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });
    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "t-r-ns-fast",
          messages: [{ role: "user", content: "ready" }],
        });
        return probe.choices[0]?.message.content === "ns-fast";
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(
      [nsSlow, nsFast, nsBackup, stSlow, stFast, stStall, stBackup].map((u) =>
        u?.close(),
      ),
    );
  });

  function caller(): OpenAI {
    return new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });
  }

  // AC1: non-streaming primary exceeds `timeout` → fall over to backup;
  // caller sees a 200 from the backup.
  test("non-streaming timeout falls over to the backup", async (ctx) => {
    if (!etcdReachable || !app || !nsSlow || !nsBackup) {
      ctx.skip();
      return;
    }
    const slowBase = nsSlow.receivedRequests.length;
    const backupBase = nsBackup.receivedRequests.length;

    const completion = await caller().chat.completions.create({
      model: "t-r-ns-timeout",
      messages: [{ role: "user", content: "hello" }],
    });

    expect(completion.choices[0]?.message.content).toBe("ns-backup");
    expect(nsSlow.receivedRequests.length - slowBase).toBe(1);
    expect(nsBackup.receivedRequests.length - backupBase).toBe(1);
  }, 30_000);

  // AC2: streaming first token exceeds `stream_timeout` → fall over to the
  // backup; the backup's tokens are piped to the caller with no error.
  test("streaming TTFT timeout falls over; backup tokens pipe cleanly", async (ctx) => {
    if (!etcdReachable || !app || !stSlow || !stBackup) {
      ctx.skip();
      return;
    }
    const slowBase = stSlow.receivedRequests.length;
    const backupBase = stBackup.receivedRequests.length;

    const collected: string[] = [];
    let surfacedError = false;
    const stream = await caller().chat.completions.create({
      model: "t-r-st-ttft",
      messages: [{ role: "user", content: "hello" }],
      stream: true,
    });
    try {
      for await (const ev of stream) {
        const c = ev.choices[0]?.delta?.content;
        if (c) collected.push(c);
      }
    } catch {
      surfacedError = true;
    }

    expect(surfacedError).toBe(false);
    expect(collected.join("")).toContain("st-backup");
    expect(stSlow.receivedRequests.length - slowBase).toBe(1);
    expect(stBackup.receivedRequests.length - backupBase).toBe(1);
  }, 30_000);

  // AC3 (non-streaming): a fast primary is served and the backup is never
  // touched — the timeout machinery doesn't interfere with healthy calls.
  test("fast non-streaming primary is served; backup untouched", async (ctx) => {
    if (!etcdReachable || !app || !nsFast || !nsBackup) {
      ctx.skip();
      return;
    }
    const fastBase = nsFast.receivedRequests.length;
    const backupBase = nsBackup.receivedRequests.length;

    const completion = await caller().chat.completions.create({
      model: "t-r-ns-fast",
      messages: [{ role: "user", content: "hello" }],
    });

    expect(completion.choices[0]?.message.content).toBe("ns-fast");
    expect(nsFast.receivedRequests.length - fastBase).toBe(1);
    expect(nsBackup.receivedRequests.length - backupBase).toBe(0);
  }, 30_000);

  // AC3 (streaming): a fast streaming primary is served and the backup is
  // never touched.
  test("fast streaming primary is served; backup untouched", async (ctx) => {
    if (!etcdReachable || !app || !stFast || !stBackup) {
      ctx.skip();
      return;
    }
    const fastBase = stFast.receivedRequests.length;
    const backupBase = stBackup.receivedRequests.length;

    const collected: string[] = [];
    const stream = await caller().chat.completions.create({
      model: "t-r-st-fast",
      messages: [{ role: "user", content: "hello" }],
      stream: true,
    });
    for await (const ev of stream) {
      const c = ev.choices[0]?.delta?.content;
      if (c) collected.push(c);
    }

    expect(collected.join("")).toContain("st-fast");
    expect(stFast.receivedRequests.length - fastBase).toBe(1);
    expect(stBackup.receivedRequests.length - backupBase).toBe(0);
  }, 30_000);

  // Read-timeout semantics: the first chunk arrives in time (200 committed),
  // but a later inter-chunk gap exceeds `stream_timeout`. The stream is
  // terminated with an error — NOT failed over (can't fall back once bytes
  // are on the wire). The first chunk still reached the caller; the backup
  // is never touched.
  test("mid-stream stall terminates the stream without fallback", async (ctx) => {
    if (!etcdReachable || !app || !stStall || !stBackup) {
      ctx.skip();
      return;
    }
    const stallBase = stStall.receivedRequests.length;
    const backupBase = stBackup.receivedRequests.length;

    const collected: string[] = [];
    let surfacedError = false;
    const stream = await caller().chat.completions.create({
      model: "t-r-st-stall",
      messages: [{ role: "user", content: "hello" }],
      stream: true,
    });
    try {
      for await (const ev of stream) {
        const c = ev.choices[0]?.delta?.content;
        if (c) collected.push(c);
      }
    } catch (e) {
      surfacedError = e instanceof APIError || e instanceof Error;
    }

    // First chunk made it to the caller before the stall.
    expect(collected.join("")).toContain("stall-1");
    // The stalled stream errored rather than completing or falling over.
    expect(surfacedError).toBe(true);
    expect(collected.join("")).not.toContain("st-backup");
    expect(stStall.receivedRequests.length - stallBase).toBe(1);
    expect(stBackup.receivedRequests.length - backupBase).toBe(0);
  }, 30_000);
});
