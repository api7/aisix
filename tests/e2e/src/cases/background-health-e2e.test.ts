import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

const CALLER_PLAINTEXT = "sk-background-health-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("background health e2e", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;
  let unhealthyUpstream: OpenAiUpstream | undefined;
  let ignoredUpstream: OpenAiUpstream | undefined;
  let stableUpstream: OpenAiUpstream | undefined;
  let unhealthyModelID = "";
  let ignoredModelID = "";
  let stableModelID = "";

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    unhealthyUpstream = await startOpenAiUpstream({
      status: 503,
      errorBody: { error: { message: "background unhealthy", type: "server_error" } },
    });
    ignoredUpstream = await startOpenAiUpstream({
      status: 429,
      errorBody: { error: { message: "background ignore", type: "rate_limit_error" } },
    });
    stableUpstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-stable-bg",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "healthy target" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
    });

    // The admin listener is off; `admin` here is used only for
    // listModelStatuses, which reads GET /status/models on the metrics
    // listener. Resources are seeded straight to etcd via `seed`.
    app = await spawnApp({ admin: false });
    admin = new AdminClient(app.adminUrl, app.adminKey, app.metricsUrl);
    seed = new SeedClient(etcd, app.etcdPrefix);

    const unhealthyPk = await seed.createProviderKey({
      display_name: "bg-unhealthy-pk",
      secret: "sk-mock",
      api_base: `${unhealthyUpstream.baseUrl}/v1`,
    });
    const ignoredPk = await seed.createProviderKey({
      display_name: "bg-ignored-pk",
      secret: "sk-mock",
      api_base: `${ignoredUpstream.baseUrl}/v1`,
    });
    const stablePk = await seed.createProviderKey({
      display_name: "bg-stable-pk",
      secret: "sk-mock",
      api_base: `${stableUpstream.baseUrl}/v1`,
    });

    unhealthyModelID = (
      await seed.createModel({
        display_name: "bg-unhealthy",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: unhealthyPk.id,
        background_model_check: {
          enabled: true,
          interval_seconds: 5,
          timeout_seconds: 10,
          prompt: "Respond with OK",
          max_tokens: 8,
          ignore_statuses: [408, 429],
          stale_after_seconds: 90,
        },
      })
    ).id;

    ignoredModelID = (
      await seed.createModel({
        display_name: "bg-ignored",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: ignoredPk.id,
        background_model_check: {
          enabled: true,
          interval_seconds: 5,
          timeout_seconds: 10,
          prompt: "Respond with OK",
          max_tokens: 8,
          ignore_statuses: [408, 429],
          stale_after_seconds: 90,
        },
      })
    ).id;

    stableModelID = (
      await seed.createModel({
        display_name: "bg-stable",
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: stablePk.id,
      })
    ).id;

    await seed.createModel({
      display_name: "bg-router",
      routing: {
        strategy: "failover",
        targets: [
          { model: "bg-unhealthy" },
          { model: "bg-stable" },
        ],
        max_fallbacks: 1,
      },
    });

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["bg-router", "bg-stable"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await unhealthyUpstream?.close();
    await ignoredUpstream?.close();
    await stableUpstream?.close();
  });

  test("background unhealthy is surfaced and ignored 429 stays visible but healthy", async (ctx) => {
    if (!etcdReachable || !app || !admin || !unhealthyUpstream || !ignoredUpstream || !stableUpstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    let lastStatuses: Array<Record<string, unknown>> = [];
    try {
      await waitConfigPropagation(async () => {
        lastStatuses = await admin!.listModelStatuses();
        const unhealthy = lastStatuses.find((row) => row.id === unhealthyModelID);
        const ignored = lastStatuses.find((row) => row.id === ignoredModelID);
        return unhealthy?.status === "unhealthy" && ignored?.last_check_status === 429;
      });
    } catch (err) {
      throw new Error(
        `${(err as Error).message}\nlast model statuses: ${JSON.stringify(lastStatuses, null, 2)}`,
      );
    }

    const statuses = await admin.listModelStatuses();
    const unhealthy = statuses.find((row) => row.id === unhealthyModelID)!;
    const ignored = statuses.find((row) => row.id === ignoredModelID)!;
    const stable = statuses.find((row) => row.id === stableModelID)!;

    expect(unhealthy.status).toBe("unhealthy");
    expect(unhealthy.last_check_status).toBe(503);
    expect(unhealthy.status_reason).toBe("background_check_failed");

    expect(ignored.status).toBe("healthy");
    expect(ignored.last_check_status).toBe(429);
    expect(ignored.status_reason).toBe("ignored_transient_error");

    expect(stable.status).toBe("healthy");

    await waitConfigPropagation(async () => {
      try {
        const probe = await client.chat.completions.create({
          model: "bg-router",
          messages: [{ role: "user", content: "ready-bg-router" }],
        });
        return probe.choices[0]?.message.content === "healthy target";
      } catch {
        return false;
      }
    });

    const unhealthyBaseline = unhealthyUpstream.receivedRequests.length;
    const stableBaseline = stableUpstream.receivedRequests.length;

    const completion = await client.chat.completions.create({
      model: "bg-router",
      messages: [{ role: "user", content: "skip background unhealthy" }],
    });
    expect(completion.choices[0]?.message.content).toBe("healthy target");
    expect(unhealthyUpstream.receivedRequests.length - unhealthyBaseline).toBe(0);
    expect(stableUpstream.receivedRequests.length - stableBaseline).toBe(1);
  });

  test("active background checks keep unhealthy state fresh even with a short stale window", async (ctx) => {
    if (!etcdReachable || !app || !admin || !seed || !unhealthyUpstream) {
      ctx.skip();
      return;
    }

    const shortStalePk = await seed.createProviderKey({
      display_name: "bg-stale-short-pk",
      secret: "sk-mock",
      api_base: `${unhealthyUpstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "bg-stale-short",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: shortStalePk.id,
      background_model_check: {
        enabled: true,
        interval_seconds: 5,
        timeout_seconds: 10,
        prompt: "Respond with OK",
        max_tokens: 8,
        ignore_statuses: [],
        // Stale window must be > probe interval so a successful
        // probe at t≈interval refreshes last_checked_at into the
        // future — keeping the unhealthy entry "fresh" across the
        // wait below. If stale_after < interval, the entry could
        // expire between probes and the test would race.
        stale_after_seconds: 8,
      },
    });

    // Config propagation is asynchronous: the model seeded above reaches
    // the status listing only after the snapshot refresh. Poll for the
    // row instead of asserting eagerly — on a loaded CI runner the
    // immediate read raced the watch and failed with `undefined`.
    await waitConfigPropagation(async () => {
      const rows = await admin!.listModelStatuses();
      return rows.some((row) => row.display_name === "bg-stale-short");
    });

    await waitConfigPropagation(async () => {
      const rows = await admin!.listModelStatuses();
      const row = rows.find((item) => item.display_name === "bg-stale-short");
      return row?.status === "unhealthy";
    });

    // Wait longer than one probe interval but well within the
    // stale window — an active probe must fire and re-mark the
    // entry unhealthy, keeping the state fresh.
    await new Promise((r) => setTimeout(r, 6500));

    const rows = await admin.listModelStatuses();
    const row = rows.find((item) => item.display_name === "bg-stale-short")!;
    expect(row.status).toBe("unhealthy");
  });
});
