import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E for what invalidates the runtime guardrail index (AISIX-Cloud#1542).
//
// The index is rebuilt lazily, on the first request that resolves a
// guardrail after the configuration changed, on the worker thread serving
// that request — and building it constructs one runtime instance per
// enabled attachment, each with its own HTTP client. So "what counts as a
// change" is a latency question, not a bookkeeping one: while the index
// keyed on the single global snapshot version, one API-key edit made every
// worker reconstruct every attachment before it could answer anything,
// liveness probes included.
//
// The three things a spec has to hold apart here:
//   - a write to a resource the index does not read must not rebuild it;
//   - a write to an attachment must, and exactly once;
//   - and after either, the guardrails in force must still be the
//     configured ones — a cache that never invalidates would also pass a
//     test that only counted rebuilds.
//
// Rebuilds are observed through the gateway's own `guardrail index
// rebuilt` log line rather than a metric, which is why the app runs at
// `info`.

const CALLER = "sk-guardrail-index-invalidation";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const BLOCK_MARKER = "guardrailindexblockmarker";
const SCREENED_MODEL = "gi-screened";
const UNSCREENED_MODEL = "gi-unscreened";
const LATE_MODEL = "gi-late";
// Enough attachments that a rebuild is a real cost rather than a rounding
// error, without making the seed slow.
const EXTRA_SCOPES = 6;

const REBUILT = /guardrail index rebuilt/;

interface GuardMock {
  baseUrl: string;
  calls: number;
  close(): Promise<void>;
}

// A minimal Lakera Guard `/v2/guard` mock: flags BLOCK_MARKER as a
// prompt attack, passes everything else. Its only job here is to be a
// network-backed guardrail whose enforcement is unambiguous.
async function startGuardMock(): Promise<GuardMock> {
  const state = { calls: 0 };
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      state.calls += 1;
      let flagged = false;
      try {
        const body = JSON.parse(raw);
        const messages: Array<{ content?: string }> = Array.isArray(body.messages)
          ? body.messages
          : [];
        flagged = messages.some((m) => (m.content ?? "").includes(BLOCK_MARKER));
      } catch {
        // leave defaults
      }
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          flagged,
          payload: [],
          breakdown: flagged
            ? [{ detector_type: "prompt_attack", detected: true }]
            : [],
        }),
      );
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", resolve);
  });
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    get calls() {
      return state.calls;
    },
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

describe("guardrail index invalidation", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let guard: GuardMock | undefined;
  let seed: SeedClient | undefined;
  let proxy: ProxyClient | undefined;
  let etcdReachable = false;
  let guardrailID = "";
  let lateModelID = "";

  function rebuilds(): number {
    return app!
      .output()
      .split("\n")
      .filter((l) => REBUILT.test(l)).length;
  }

  async function chat(model: string, content: string): Promise<number> {
    return (await proxy!.chat({ model, messages: [{ role: "user", content }] }))
      .status;
  }

  // Gate on a caller key seeded AFTER the write under test: etcd delivers
  // in revision order, so that key authenticating implies the write ahead
  // of it is in the snapshot. Gating on the enforcement itself would turn
  // a broken invalidation into a 30s timeout instead of an assertion, and
  // `listModels` resolves no guardrail, so the gate cannot consume the
  // rebuild the next assertion counts.
  async function propagated(tag: string): Promise<void> {
    const plaintext = `sk-guardrail-index-gate-${tag}`;
    await seed!.createApiKey({
      key_hash: hash(plaintext),
      allowed_models: [SCREENED_MODEL],
    });
    const gate = new ProxyClient(app!.proxyUrl, plaintext);
    await waitConfigPropagation(
      async () => (await gate.listModels()).status === 200,
    );
  }

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    guard = await startGuardMock();
    // `info`: the rebuild line this spec counts is emitted there.
    app = await spawnApp({ logLevel: "info" });
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "gi-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const model = (display_name: string) => ({
      display_name,
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    const screened = await seed.createModel(model(SCREENED_MODEL));
    await seed.createModel(model(UNSCREENED_MODEL));
    const late = await seed.createModel(model(LATE_MODEL));
    lateModelID = late.id;

    const guardrail = await seed.createGuardrail(
      {
        name: "gi-lakera",
        kind: "lakera",
        hook_point: "input",
        api_key: "test-key",
        endpoint: guard.baseUrl,
      },
      { attach: false },
    );
    guardrailID = guardrail.id;
    await seed.attachGuardrailToModel(guardrailID, screened.id);
    // Several more attachments onto scopes this spec never calls, so the
    // index carries a realistic number of entries for one guardrail.
    for (let i = 0; i < EXTRA_SCOPES; i += 1) {
      const extra = await seed.createModel(model(`gi-extra-${i}`));
      await seed.attachGuardrailToModel(guardrailID, extra.id);
    }

    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: [
        SCREENED_MODEL,
        UNSCREENED_MODEL,
        LATE_MODEL,
        ...Array.from({ length: EXTRA_SCOPES }, (_, i) => `gi-extra-${i}`),
      ],
    });
    proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(
      async () => (await proxy!.listModels()).status === 200,
    );
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await guard?.close();
  });

  test("an unrelated write does not rebuild the index; an attachment write does, once", async (ctx) => {
    if (!etcdReachable) ctx.skip();

    // Warm: the first request that resolves a guardrail builds the index,
    // and the guardrail must actually be in force before anything below
    // means anything.
    expect(await chat(SCREENED_MODEL, BLOCK_MARKER)).toBe(422);
    const callsAfterWarm = guard!.calls;
    expect(callsAfterWarm).toBeGreaterThan(0);
    const afterWarm = rebuilds();
    expect(afterWarm).toBeGreaterThan(0);

    // Thirty writes to a resource kind the index does not read — the
    // shape of the bulk API-key edit in the report. Each one publishes a
    // new snapshot and moves the global version.
    const unrelated = await seed!.createApiKey({
      key_hash: hash("sk-guardrail-index-unrelated"),
      allowed_models: [SCREENED_MODEL],
    });
    for (let i = 0; i < 30; i += 1) {
      await seed!.update("api_keys", unrelated.id, {
        key_hash: hash("sk-guardrail-index-unrelated"),
        allowed_models: [SCREENED_MODEL, `filler-${i}`],
      });
    }
    // A positive gate that the writes landed: the last one is visible to
    // the gateway's own authentication.
    const unrelatedCaller = new ProxyClient(
      app!.proxyUrl,
      "sk-guardrail-index-unrelated",
    );
    await waitConfigPropagation(
      async () => (await unrelatedCaller.listModels()).status === 200,
    );

    // Requests after those writes still screen, and still on the index
    // built before them.
    expect(await chat(SCREENED_MODEL, BLOCK_MARKER)).toBe(422);
    expect(await chat(SCREENED_MODEL, "hello")).toBe(200);
    expect(rebuilds()).toBe(afterWarm);
    expect(guard!.calls).toBeGreaterThan(callsAfterWarm);

    // A model with no attachment is not screened at all, before or after.
    const beforeUnscreened = guard!.calls;
    expect(await chat(UNSCREENED_MODEL, BLOCK_MARKER)).toBe(200);
    expect(guard!.calls).toBe(beforeUnscreened);

    // Now attach the guardrail to a model that had none. This one MUST
    // rebuild — and the new scope must actually be enforced, which is
    // what separates a correct cache key from one that never invalidates.
    const attachment = await seed!.attachGuardrailToModel(guardrailID, lateModelID);
    await propagated("attached");
    expect(await chat(LATE_MODEL, BLOCK_MARKER)).toBe(422);
    expect(rebuilds()).toBe(afterWarm + 1);

    // And removing it stops the enforcement.
    await seed!.delete("guardrail_attachments", attachment.id);
    await propagated("detached");
    expect(await chat(LATE_MODEL, BLOCK_MARKER)).toBe(200);
    expect(rebuilds()).toBe(afterWarm + 2);
    // The scope that was configured all along is untouched by either.
    expect(await chat(SCREENED_MODEL, BLOCK_MARKER)).toBe(422);
  });
});
