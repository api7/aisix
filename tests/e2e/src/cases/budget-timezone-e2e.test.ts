import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  scrapeMetrics,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `budget_policy.timezone` shifts calendar budget boundaries and the
// `period_resets_at` / `Retry-After` / `aisix_budget_reset_seconds` contract
// from UTC to the configured IANA timezone, and rejects invalid timezone names
// through `/status/config` while keeping the last-known-good policy active.

type FixedZone = { name: string; offsetMinutes: number; offsetSuffix: string };

const SHANGHAI: FixedZone = { name: "Asia/Shanghai", offsetMinutes: 8 * 60, offsetSuffix: "+08:00" };

const hash = (plaintext: string) => createHash("sha256").update(plaintext).digest("hex");
const rfc3339Utc = (ms: number) => new Date(ms).toISOString().replace(".000Z", "Z");
const pad2 = (n: number) => String(n).padStart(2, "0");

// Pick a fixed-offset (no-DST) IANA timezone whose local calendar date differs
// from UTC right now AND whose next local midnight is at least 1 hour away:
// - UTC 00:00..09:59 -> Pacific/Pago_Pago (-11:00): UTC day - 1, >=1h to midnight
// - UTC 10:00..15:59 -> Pacific/Kiritimati (+14:00): UTC day + 1, >=18h to midnight
// - UTC 16:00..23:59 -> Asia/Shanghai (+08:00): UTC day + 1, >=16h to midnight
function pickDifferentDayZone(now: Date): FixedZone {
  const h = now.getUTCHours();
  if (h >= 16) return SHANGHAI;
  if (h >= 10) return { name: "Pacific/Kiritimati", offsetMinutes: 14 * 60, offsetSuffix: "+14:00" };
  return { name: "Pacific/Pago_Pago", offsetMinutes: -11 * 60, offsetSuffix: "-11:00" };
}

function fixedZoneDayWindow(nowMs: number, zone: FixedZone) {
  const offsetMs = zone.offsetMinutes * 60_000;
  const shifted = new Date(nowMs + offsetMs);
  const y = shifted.getUTCFullYear();
  const m = shifted.getUTCMonth();
  const d = shifted.getUTCDate();
  const startUtcMs = Date.UTC(y, m, d) - offsetMs;
  const nextUtcMs = Date.UTC(y, m, d + 1) - offsetMs;
  const nextShifted = new Date(nextUtcMs + offsetMs);
  const nextLocalRfc3339 =
    `${nextShifted.getUTCFullYear()}-${pad2(nextShifted.getUTCMonth() + 1)}-${pad2(nextShifted.getUTCDate())}` +
    `T00:00:00${zone.offsetSuffix}`;
  return { startUtcMs, nextUtcMs, nextLocalRfc3339 };
}

function fixedZoneNextMonthWindow(nowMs: number, zone: FixedZone) {
  const offsetMs = zone.offsetMinutes * 60_000;
  const shifted = new Date(nowMs + offsetMs);
  const nextUtcMs = Date.UTC(shifted.getUTCFullYear(), shifted.getUTCMonth() + 1, 1) - offsetMs;
  const nextShifted = new Date(nextUtcMs + offsetMs);
  const nextLocalRfc3339 =
    `${nextShifted.getUTCFullYear()}-${pad2(nextShifted.getUTCMonth() + 1)}-01T00:00:00${zone.offsetSuffix}`;
  return { nextUtcMs, nextLocalRfc3339 };
}

function utcNextMonthWindow(nowMs: number) {
  const d = new Date(nowMs);
  const nextUtcMs = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1);
  return { nextUtcMs, nextRfc3339: rfc3339Utc(nextUtcMs) };
}

const KEYS = {
  daily: "sk-btz-e2e-daily",
  sentinelMonthly: "sk-btz-e2e-sentinel-monthly",
  ready: "sk-btz-e2e-ready",
};

const IDS = {
  policy: randomUUID(),
  dailyKey: randomUUID(),
  dailyBudget: randomUUID(),
  sentinelKey: randomUUID(),
  sentinelBudget: randomUUID(),
};

describe("budget_policy.timezone governs budget periods and reset headers", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let zone: FixedZone = SHANGHAI;
  let dayWindow = fixedZoneDayWindow(Date.now(), SHANGHAI);

  const rawChat = async (key: string) => {
    const res = await fetch(`${app!.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${key}`, "content-type": "application/json" },
      body: JSON.stringify({ model: "btz-direct", messages: [{ role: "user", content: "hello" }] }),
    });
    return { status: res.status, body: (await res.json()) as Record<string, unknown>, headers: res.headers };
  };

  const rawMessages = async (key: string) => {
    const res = await fetch(`${app!.proxyUrl}/v1/messages`, {
      method: "POST",
      headers: {
        "x-api-key": key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "btz-direct",
        max_tokens: 8,
        messages: [{ role: "user", content: "hi" }],
      }),
    });
    return { status: res.status, body: (await res.json()) as Record<string, unknown>, headers: res.headers };
  };

  const fetchConfigStatus = async () => {
    const res = await fetch(`${app!.metricsUrl}/status/config`);
    return (await res.json()) as {
      rejected: Array<{
        resource_kind: string;
        resource_id: string;
        last_error_kind: string;
        last_error: string;
      }>;
    };
  };

  // Watch events apply in etcd revision order, so seeding a fresh sentinel key
  // after a mutation and waiting for it to authenticate guarantees every prior
  // write has been processed — keeping assertions out of wait loops.
  const syncThroughSentinel = async () => {
    const plaintext = `sk-btz-sync-${randomUUID()}`;
    await seed!.createApiKey({ key_hash: hash(plaintext), allowed_models: ["*"] });
    await waitConfigPropagation(
      async () => (await new ProxyClient(app!.proxyUrl, plaintext).listModels()).status === 200,
    );
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    const now = new Date();
    zone = pickDifferentDayZone(now);
    dayWindow = fixedZoneDayWindow(now.getTime(), zone);

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "btz-pk",
      api_key: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "btz-direct",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.update("budget_policy", IDS.policy, {
      fail_mode: "sticky",
      closed_grace_seconds: 300,
      timezone: zone.name,
    });
    await seed.update("budgets", IDS.dailyBudget, {
      scope: "api_key",
      scope_ref: IDS.dailyKey,
      period: "day",
      exhausted: {
        period_start: rfc3339Utc(dayWindow.startUtcMs),
        reason: {
          message: "daily budget exceeded",
          scope: "api_key",
          scope_ref: IDS.dailyKey,
          limit: "10.00",
          spent: "10.50",
          remaining: "-0.50",
          period: "day",
        },
      },
    });
    await seed.update("budgets", IDS.sentinelBudget, {
      scope: "api_key",
      scope_ref: IDS.sentinelKey,
      period: "month",
      exhausted: {
        reason: {
          message: "monthly sentinel budget exceeded",
          scope: "api_key",
          scope_ref: IDS.sentinelKey,
          limit: "0.00",
          spent: "0.00",
          remaining: "0.00",
          period: "month",
        },
      },
    });

    // Seed caller keys last so authenticating `KEYS.ready` implies all earlier writes landed.
    await seed.update("api_keys", IDS.dailyKey, { key_hash: hash(KEYS.daily), allowed_models: ["*"] });
    await seed.update("api_keys", IDS.sentinelKey, { key_hash: hash(KEYS.sentinelMonthly), allowed_models: ["*"] });
    await seed.createApiKey({ key_hash: hash(KEYS.ready), allowed_models: ["*"] });

    await waitConfigPropagation(
      async () => (await new ProxyClient(app!.proxyUrl, KEYS.ready).listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("enforces a configured non-UTC timezone on /v1/chat/completions and /v1/messages, and lets a mismatched UTC period_start pass", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();

    const beforeUpstream = upstream!.receivedRequests.length;
    const chatRes = await rawChat(KEYS.daily);
    expect(chatRes.status).toBe(429);
    const err = chatRes.body.error as Record<string, unknown>;
    expect(err.type).toBe("billing_error");
    expect(err.code).toBe("budget_exceeded");
    expect(err.period_resets_at).toBe(dayWindow.nextLocalRfc3339);

    const retryAfter = Number(chatRes.headers.get("retry-after"));
    expect(retryAfter).toBe(err.retry_after_seconds);
    const expectedSeconds = (dayWindow.nextUtcMs - Date.now()) / 1000;
    expect(Math.abs(retryAfter - expectedSeconds)).toBeLessThan(60);

    const samples = await scrapeMetrics(app.metricsUrl);
    const resetGauge = samples.find(
      (s) => s.name === "aisix_budget_reset_seconds" && s.labels.api_key_id === IDS.dailyKey,
    )?.value;
    expect(resetGauge).toBe(retryAfter);

    const msgRes = await rawMessages(KEYS.daily);
    expect(msgRes.status).toBe(429);
    expect(msgRes.body.type).toBe("error");
    expect((msgRes.body.error as Record<string, unknown>).type).toBe("rate_limit_error");
    const msgRetryAfter = Number(msgRes.headers.get("retry-after"));
    expect(Math.abs(msgRetryAfter - expectedSeconds)).toBeLessThan(60);
    expect(upstream!.receivedRequests.length).toBe(beforeUpstream);

    // Replace period_start with UTC's current day start, which differs from the
    // configured timezone's day start: both endpoints must now allow the request.
    const now = new Date();
    const utcDayStart = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
    expect(utcDayStart).not.toBe(dayWindow.startUtcMs);
    await seed.update("budgets", IDS.dailyBudget, {
      scope: "api_key",
      scope_ref: IDS.dailyKey,
      period: "day",
      exhausted: {
        period_start: rfc3339Utc(utcDayStart),
        reason: {
          message: "utc period_start should not match non-UTC timezone",
          scope: "api_key",
          scope_ref: IDS.dailyKey,
          period: "day",
        },
      },
    });
    await syncThroughSentinel();
    expect((await rawChat(KEYS.daily)).status).toBe(200);
    expect((await rawMessages(KEYS.daily)).status).toBe(200);
  });

  test("Asia/Shanghai formats period_resets_at with +08:00 and invalid timezone updates are rejected while keeping last-known-good", async (ctx) => {
    if (!etcdReachable || !app || !seed) return ctx.skip();

    await seed.update("budget_policy", IDS.policy, {
      fail_mode: "sticky",
      closed_grace_seconds: 300,
      timezone: SHANGHAI.name,
    });
    await syncThroughSentinel();

    const beforeMs = Date.now();
    const shanghaiRes = await rawChat(KEYS.sentinelMonthly);
    const afterMs = Date.now();
    expect(shanghaiRes.status).toBe(429);
    const shanghaiErr = shanghaiRes.body.error as Record<string, unknown>;
    const validResets = new Set([
      fixedZoneNextMonthWindow(beforeMs, SHANGHAI).nextLocalRfc3339,
      fixedZoneNextMonthWindow(afterMs, SHANGHAI).nextLocalRfc3339,
    ]);
    expect(validResets.has(String(shanghaiErr.period_resets_at))).toBe(true);
    const shanghaiRetryAfter = Number(shanghaiRes.headers.get("retry-after"));
    expect(shanghaiRetryAfter).toBe(shanghaiErr.retry_after_seconds);

    // Push an invalid timezone: /status/config rejected[] and the rejected
    // metric record it, while last-known-good Asia/Shanghai stays in force.
    await seed.update("budget_policy", IDS.policy, {
      fail_mode: "sticky",
      closed_grace_seconds: 300,
      timezone: "Mars/Olympus",
    });
    await syncThroughSentinel();

    const rejected = (await fetchConfigStatus()).rejected;
    const entry = rejected.find(
      (r) => r.resource_kind === "budget_policy" && r.resource_id === IDS.policy,
    );
    expect(entry).toBeDefined();
    expect(entry!.last_error_kind).toBe("schema_failed");
    expect(entry!.last_error).toContain("IANA timezone");

    const metricsText = await (await fetch(`${app.metricsUrl}/metrics`)).text();
    expect(metricsText).toMatch(/aisix_config_rejected_resources\{kind="budget_policy"\} 1\b/);

    // Last-known-good Asia/Shanghai policy still serves.
    const stillShanghai = await rawChat(KEYS.sentinelMonthly);
    expect(stillShanghai.status).toBe(429);
    expect(String((stillShanghai.body.error as Record<string, unknown>).period_resets_at)).toMatch(
      /\+08:00$/,
    );

    // Clearing timezone restores UTC `Z` format and clears the rejection.
    await seed.update("budget_policy", IDS.policy, {
      fail_mode: "sticky",
      closed_grace_seconds: 300,
    });
    await syncThroughSentinel();

    const clearedStatus = await fetchConfigStatus();
    expect(clearedStatus.rejected.some((r) => r.resource_kind === "budget_policy")).toBe(false);
    const utcBefore = Date.now();
    const utcRes = await rawChat(KEYS.sentinelMonthly);
    const utcAfter = Date.now();
    expect(utcRes.status).toBe(429);
    const validUtcResets = new Set([
      utcNextMonthWindow(utcBefore).nextRfc3339,
      utcNextMonthWindow(utcAfter).nextRfc3339,
    ]);
    expect(validUtcResets.has(String((utcRes.body.error as Record<string, unknown>).period_resets_at))).toBe(true);
  });
});
