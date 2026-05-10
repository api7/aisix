import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: max_budget_usd round-trips through the admin API. Closes the
// gap that PR #182 surfaced — the JSON schema previously rejected
// max_budget_usd, making the field unreachable from a standalone
// admin POST. This test pins the documented §4.2 contract: a body
// carrying max_budget_usd is accepted, persisted, and returned by
// GET unchanged. A regression that re-tightens the schema (or
// re-adds additionalProperties: false without re-listing the field)
// fails here loudly.
//
// Reference: docs/api-admin.md §4.2 (the example body now includes
// "max_budget_usd": 500.0).

const PLAINTEXT = "sk-budget-e2e";
const KEY_HASH = createHash("sha256").update(PLAINTEXT).digest("hex");

describe("apikey max_budget_usd e2e: admin POST + GET round-trip", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
  });

  afterAll(async () => {
    await app?.exit();
  });

  test("POST persists max_budget_usd; GET returns the same value", async (ctx) => {
    if (!etcdReachable || !admin) {
      ctx.skip();
      return;
    }

    const created = await admin.createApiKey({
      key_hash: KEY_HASH,
      allowed_models: ["*"],
      max_budget_usd: 500.0,
    });
    expect(created.id).toBeTruthy();

    const got = await admin.json<{
      id: string;
      value: { max_budget_usd?: number };
    }>("GET", `/admin/v1/apikeys/${created.id}`);
    expect(got.value.max_budget_usd).toBe(500);
  });

  test("POST with negative max_budget_usd is rejected with 400", async (ctx) => {
    if (!etcdReachable || !admin) {
      ctx.skip();
      return;
    }

    let caught: unknown;
    try {
      await admin.createApiKey({
        key_hash: createHash("sha256").update("sk-neg-budget").digest("hex"),
        allowed_models: ["*"],
        max_budget_usd: -1,
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toContain("400");
  });
});
