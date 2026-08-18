import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `GET /v1/models` is the caller's model-discovery surface, so it must
// list every name the caller is allowed to send as `model` — a Model Group
// included. The scenario is the deployment where a group is the only public
// entry point: callers get the stable group name and its targets are an
// internal detail they are not authorized for.
//
// Two keys exercise the two halves of the contract:
//   - group-only key: discovery returns exactly the group, and that name is
//     callable. Nothing about the group's targets leaks into the listing.
//   - unrestricted key: discovery returns the group *and* the direct models,
//     because the key may send any of those names.
//
// Reference: OpenAI Models API spec
// (https://platform.openai.com/docs/api-reference/models/list) — `data[].id`
// is "the model identifier, which can be referenced in the API endpoints".
//
// Note the group name is asserted to be callable, not just present: a listing
// entry a client cannot actually use would be worse than omitting it.

const GROUP_ONLY_PLAINTEXT = "sk-ml-e2e-group-only";
const GROUP_ONLY_KEY_HASH = createHash("sha256")
  .update(GROUP_ONLY_PLAINTEXT)
  .digest("hex");

const ALL_MODELS_PLAINTEXT = "sk-ml-e2e-all-models";
const ALL_MODELS_KEY_HASH = createHash("sha256")
  .update(ALL_MODELS_PLAINTEXT)
  .digest("hex");

interface ModelListBody {
  object?: unknown;
  data?: { id?: unknown; object?: unknown; created?: unknown; owned_by?: unknown }[];
}

function idsOf(body: unknown): string[] {
  const data = (body as ModelListBody | null)?.data ?? [];
  return data.map((m) => String(m.id));
}

describe("models listing e2e: a Model Group is a discoverable entry point", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "ml-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "ml-primary",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await seed.createModel({
      display_name: "ml-secondary",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // The public entry point. Its targets are the two direct models above.
    await seed.createModel({
      display_name: "ml-group",
      routing: {
        strategy: "failover",
        targets: [{ model: "ml-primary" }, { model: "ml-secondary" }],
      },
    });

    await seed.createApiKey({
      key_hash: GROUP_ONLY_KEY_HASH,
      allowed_models: ["ml-group"],
    });
    await seed.createApiKey({
      key_hash: ALL_MODELS_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("group-only key discovers the group, can call it, and never sees its targets", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const probe = new ProxyClient(app.proxyUrl, GROUP_ONLY_PLAINTEXT);
    const client = new OpenAI({
      apiKey: GROUP_ONLY_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // Both caller keys are seeded after every model and watch events apply in
    // revision order, so this key authenticating at all means the provider
    // key, both targets and the group are already in the snapshot. Gating on
    // the key rather than on a model name keeps the gate independent of what
    // the assertions check, and keeps a regression an assertion diff rather
    // than a propagation timeout.
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      return res.status === 200;
    });

    const res = await probe.listModels();
    expect(res.status).toBe(200);
    expect((res.body as ModelListBody).object).toBe("list");
    // Exactly the group: the targets exist in the snapshot but this key is
    // not authorized for them, so discovery must not disclose them.
    expect(idsOf(res.body)).toEqual(["ml-group"]);

    // OpenAI Models spec shape for the group entry.
    const entry = (res.body as ModelListBody).data?.[0];
    expect(entry?.object).toBe("model");
    expect(typeof entry?.created).toBe("number");
    // A group has no provider of its own, so it is owned by the gateway.
    // A target's provider surfacing here would be exactly the target
    // detail this listing must not disclose.
    expect(entry?.owned_by).toBe("aisix");

    // The discovered name is usable as-is: a client that picked `ml-group`
    // off the listing sends it straight back as `model`.
    const completion = await client.chat.completions.create({
      model: idsOf(res.body)[0],
      messages: [{ role: "user", content: "hello from the listing" }],
    });
    expect(completion.choices[0]?.message.role).toBe("assistant");
  });

  test("unrestricted key discovers the group alongside the direct models", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    const probe = new ProxyClient(app.proxyUrl, ALL_MODELS_PLAINTEXT);

    // Same gate as above, and this key is seeded last of all: authenticating
    // proves every model landed. Waiting on one target's name would not —
    // the other target and the group are written after it.
    await waitConfigPropagation(async () => {
      const res = await probe.listModels();
      return res.status === 200;
    });

    const res = await probe.listModels();
    expect(res.status).toBe(200);
    // Every name this key may send, group and targets alike.
    expect(idsOf(res.body).sort()).toEqual([
      "ml-group",
      "ml-primary",
      "ml-secondary",
    ]);
  });
});
