import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: standalone admin API must reject max_budget_usd writes.
// Budget policy belongs to the managed control-plane path, so the
// standalone admin public contract must not accept local budget authoring.

const PLAINTEXT = "sk-budget-e2e";
const KEY_HASH = createHash("sha256").update(PLAINTEXT).digest("hex");

describe("apikey max_budget_usd e2e: standalone admin rejects budget field", () => {
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

  test("POST rejects max_budget_usd with 400", async (ctx) => {
    if (!etcdReachable || !admin) {
      ctx.skip();
      return;
    }

    let caught: unknown;
    try {
      await admin.createApiKey({
        key_hash: KEY_HASH,
        allowed_models: ["*"],
        max_budget_usd: 500.0,
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toContain("400");
  });
});
