import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  spawnApp,
  startEtcdRelay,
  startOpenAiUpstream,
  waitConfigPropagation,
  waitForLogLine,
  type EtcdRelay,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// A configuration store that compacts can discard revisions a gateway's
// watch has not been sent yet. etcd then cancels that watch with the
// compaction revision, and the gateway must re-read everything and watch
// again from the new head — otherwise it keeps serving what it had,
// missing whatever was written or deleted in the discarded range.
//
// A plain reconnect cannot reach that path: every watch cycle starts with
// a fresh read, and a watch that is keeping up is never cancelled by a
// compaction. So the spec makes the watch fall behind for real. The
// gateway dials etcd through a relay that stops delivering etcd's replies
// while still forwarding the gateway's requests; etcd keeps producing
// events for the watch until the stream's flow-control window and etcd's
// per-stream buffer are full, and then parks the watcher as slow (visible
// in etcd's own `slow_watcher_total`). Only then is the store compacted
// past everything that watcher still has to deliver.

const CALLER_PLAINTEXT = "sk-etcd-compaction-resync-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const DELETED = "compaction-deleted-model";
const WRITTEN_AFTER = "compaction-written-after-model";
const WRITTEN_LATER = "compaction-written-later-model";

const RESYNC_LINE = "etcd compaction detected";
const BACKOFF_LINE = "etcd watch failed; backing off";

// Enough filler to exceed any flow-control window plus etcd's 128-response
// per-stream buffer many times over; the loop stops as soon as etcd
// reports the watcher slow.
const FILLER_VALUE_BYTES = 16 * 1024;
const MAX_FILLER_PUTS = 4096;

describe("etcd compaction behind a lagging watch", () => {
  let etcd: EtcdClient | undefined;
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let relay: EtcdRelay | undefined;
  let app: SpawnedApp | undefined;

  beforeAll(async () => {
    etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
    relay = await startEtcdRelay();
    await relay.release();
  });

  afterAll(async () => {
    relay?.resumeReplies();
    await app?.exit();
    await relay?.stop();
    await upstream?.close();
  });

  test("the gateway resyncs and serves what was written and deleted across the compaction", async (ctx) => {
    if (!etcdReachable || !etcd || !relay || !upstream) {
      ctx.skip();
      return;
    }
    const prefix = `/aisix-e2e-compaction-${randomUUID()}`;
    const seed = new SeedClient(etcd, prefix);
    const pk = await seed.createProviderKey({
      display_name: "compaction-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    const model = (display_name: string) =>
      seed.createModel({
        display_name,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: [DELETED, WRITTEN_AFTER, WRITTEN_LATER],
    });

    app = await spawnApp({
      etcdPrefix: prefix,
      logLevel: "warn",
      extra: { etcd: { endpoints: [relay.endpoint], prefix } },
    });
    const proxy = new ProxyClient(app.proxyUrl, CALLER_PLAINTEXT);
    const served = async (): Promise<string[]> => {
      const res = await proxy.listModels();
      if (res.status !== 200) return [];
      return ((res.body as { data?: Array<{ id: string }> }).data ?? []).map((m) => m.id);
    };
    // Written after the boot read, so seeing it served proves the watch is
    // established and delivering before its replies are stalled.
    const doomed = await model(DELETED);
    await waitConfigPropagation(async () => (await served()).includes(DELETED));

    // Make the gateway's watch fall behind. The gauge is cluster-wide, and
    // locally several forks share one etcd, so wait for it to rise above
    // what it read before the stall rather than for it to be non-zero.
    const slowBefore = await etcd.slowWatchers();
    relay.stallReplies();
    const filler = "x".repeat(FILLER_VALUE_BYTES);
    const fillerKey = `${prefix}/compaction_filler/${randomUUID()}`;
    let puts = 0;
    do {
      if (puts >= MAX_FILLER_PUTS) {
        throw new Error(`etcd never reported the stalled watch as slow after ${puts} puts`);
      }
      for (let i = 0; i < 64; i++, puts++) await etcd.put(fillerKey, filler);
      // etcd refreshes the gauge on its ~100ms sync tick.
      await new Promise((r) => setTimeout(r, 150));
    } while ((await etcd.slowWatchers()) <= slowBefore);
    await etcd.delete(fillerKey);

    // A deletion the lagging watch has not delivered, then a compaction
    // that discards it from history, then a write after the compaction.
    await seed.delete("models", doomed.id);
    await etcd.compact(await etcd.currentRevision());
    await model(WRITTEN_AFTER);

    relay.resumeReplies();

    // The compaction path, not an ordinary reconnect or a failure.
    await waitForLogLine(
      app,
      (l) => l.includes("WARN") && l.includes(RESYNC_LINE),
      "the compaction resync warning",
      30_000,
    );
    // Retrying assertions rather than a propagation gate: this IS the
    // behaviour under test, and a failure should name the models served.
    await expect.poll(served, { timeout: 30_000, interval: 200 }).toContain(WRITTEN_AFTER);
    await expect.poll(served, { timeout: 30_000, interval: 200 }).not.toContain(DELETED);
    const chat = await proxy.chat({
      model: WRITTEN_AFTER,
      messages: [{ role: "user", content: "after compaction" }],
    });
    expect(chat.status, JSON.stringify(chat.body)).toBe(200);
    const gone = await proxy.chat({
      model: DELETED,
      messages: [{ role: "user", content: "deleted before compaction" }],
    });
    expect(gone.status, JSON.stringify(gone.body)).toBe(404);

    // The watch re-established after the resync keeps applying writes.
    await model(WRITTEN_LATER);
    await expect.poll(served, { timeout: 30_000, interval: 200 }).toContain(WRITTEN_LATER);

    // Compaction is not a failure: the supervisor re-reads immediately
    // instead of backing off.
    expect(app.output()).not.toContain(BACKOFF_LINE);
  }, 120_000);
});
