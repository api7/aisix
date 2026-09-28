import { createHash } from "node:crypto";
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { connect } from "node:net";
import { networkInterfaces, tmpdir } from "node:os";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  ProxyClient,
  SeedClient,
  scrapeMetrics,
  spawnApp,
  startOpenAiUpstream,
  sumMetric,
  waitConfigPropagation,
  type MetricSample,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// Memory observability: what an operator reads to find where a gateway's
// memory went — allocator, process and runtime gauges, per-store gauges,
// heap profiles on demand from a loopback-only listener, and heap profiles
// written automatically as resident memory nears the limit.

const CALLER = "sk-memory-observability-caller";
const CALLER_HASH = createHash("sha256").update(CALLER).digest("hex");

const sample = (samples: MetricSample[], name: string, labels: Record<string, string> = {}) =>
  samples.find(
    (s) => s.name === name && Object.entries(labels).every(([k, v]) => s.labels[k] === v),
  );

async function pollMetrics(
  app: SpawnedApp,
  pred: (samples: MetricSample[]) => boolean,
  what: string,
  timeoutMs = 20_000,
): Promise<MetricSample[]> {
  const until = Date.now() + timeoutMs;
  let last: MetricSample[] = [];
  while (Date.now() < until) {
    last = await scrapeMetrics(app.metricsUrl);
    if (pred(last)) return last;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(`timed out waiting for ${what}`);
}

/** Minimal protobuf reader: enough of `perftools.profiles.Profile` to list
 * its sample types and function names. */
function readProfile(pprof: Buffer): { sampleTypes: string[]; functions: string[] } {
  type Field = { field: number; wire: number; value: number | Buffer };
  const fields = (buf: Buffer): Field[] => {
    const out: Field[] = [];
    let i = 0;
    const varint = () => {
      let result = 0;
      let shift = 1;
      for (;;) {
        const b = buf[i++];
        result += (b & 0x7f) * shift;
        if ((b & 0x80) === 0) return result;
        shift *= 128;
      }
    };
    while (i < buf.length) {
      const tag = varint();
      const field = Math.floor(tag / 8);
      const wire = tag % 8;
      if (wire === 0) out.push({ field, wire, value: varint() });
      else if (wire === 2) {
        const len = varint();
        out.push({ field, wire, value: buf.subarray(i, i + len) });
        i += len;
      } else if (wire === 1) i += 8;
      else if (wire === 5) i += 4;
      else throw new Error(`unexpected wire type ${wire}`);
    }
    return out;
  };
  const top = fields(pprof);
  const strings = top.filter((f) => f.field === 6).map((f) => (f.value as Buffer).toString("utf8"));
  const num = (msg: Buffer, n: number) =>
    (fields(msg).find((f) => f.field === n)?.value as number | undefined) ?? 0;
  const sampleTypes = top
    .filter((f) => f.field === 1)
    .map((f) => strings[num(f.value as Buffer, 1)]);
  const functions = top
    .filter((f) => f.field === 5)
    .map((f) => strings[num(f.value as Buffer, 2)]);
  return { sampleTypes, functions };
}

/** What the gateway's memory limit is, read the way the kernel exposes it
 * — the oracle the `aisix_memory_limit_bytes` series is checked against. */
async function cgroupLimit(pid: number): Promise<number | undefined> {
  const lines = (await readFile(`/proc/${pid}/cgroup`, "utf8")).trim().split("\n");
  const tryRead = async (paths: string[]) => {
    for (const p of paths) {
      try {
        return (await readFile(p, "utf8")).trim();
      } catch {
        // next candidate
      }
    }
    return undefined;
  };
  for (const line of lines) {
    const [id, controllers, path] = line.split(":", 3);
    if (id === "0" && controllers === "") {
      const raw = await tryRead([`/sys/fs/cgroup${path}/memory.max`, "/sys/fs/cgroup/memory.max"]);
      if (raw === undefined) continue;
      return raw === "max" ? undefined : Number(raw);
    }
    if (controllers.split(",").includes("memory")) {
      const raw = await tryRead([
        `/sys/fs/cgroup/memory${path}/memory.limit_in_bytes`,
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
      ]);
      if (raw === undefined) continue;
      const v = Number(raw);
      return v >= 2 ** 62 ? undefined : v;
    }
  }
  return undefined;
}

const chatChunk = (delta: Record<string, unknown>, finish: string | null = null) =>
  JSON.stringify({
    id: "chatcmpl-mem",
    object: "chat.completion.chunk",
    model: "gpt-4o-mini",
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
const TEXT = "x".repeat(400);
// Thirty frames 100 ms apart: a three-second stream the gateway holds back
// whole, because a masking guardrail scans it before release.
const CHAT_SLOW = [
  chatChunk({ role: "assistant" }),
  ...Array.from({ length: 30 }, () => chatChunk({ content: TEXT })),
  chatChunk({}, "stop"),
  "[DONE]",
];
const ANTHROPIC_SLOW = [
  JSON.stringify({
    type: "message_start",
    message: {
      id: "msg_mem",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-3-5-haiku-20241022",
      stop_reason: null,
      usage: { input_tokens: 5, output_tokens: 1 },
    },
  }),
  JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ...Array.from({ length: 30 }, () =>
    JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: TEXT } }),
  ),
  JSON.stringify({ type: "content_block_stop", index: 0 }),
  JSON.stringify({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 40 } }),
  JSON.stringify({ type: "message_stop" }),
];
const RESPONSES_SLOW = [
  JSON.stringify({
    type: "response.created",
    response: { id: "resp_mem", object: "response", status: "in_progress", model: "gpt-4o-mini", output: [] },
  }),
  ...Array.from({ length: 30 }, () =>
    JSON.stringify({ type: "response.output_text.delta", item_id: "msg_mem", output_index: 0, content_index: 0, delta: TEXT }),
  ),
  JSON.stringify({
    type: "response.completed",
    response: {
      id: "resp_mem",
      object: "response",
      status: "completed",
      model: "gpt-4o-mini",
      output: [],
      usage: { input_tokens: 5, output_tokens: 40, total_tokens: 45 },
    },
  }),
];

describe("memory observability", () => {
  let app: SpawnedApp | undefined;
  let dumpDir = "";
  const upstreams: OpenAiUpstream[] = [];
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    dumpDir = await mkdtemp(join(tmpdir(), "aisix-heap-"));
    app = await spawnApp({
      threadPerCore: true,
      heapProfiling: {
        // Thresholds any running process is past: all three fire on the
        // first check, and `keep` leaves two of the three files.
        auto_dump: { enabled: true, thresholds: [0.000001, 0.000002, 0.000003], dir: dumpDir, keep: 2 },
      },
    });
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const hold = await seed.createGuardrail(
      {
        name: "mem-hold",
        enabled: true,
        hook_point: "output",
        kind: "pii",
        detectors: [{ type: "email", action: "mask" }],
      },
      { attach: false },
    );
    const model = async (
      display: string,
      provider: "openai" | "anthropic",
      opts: Parameters<typeof startOpenAiUpstream>[0],
      guarded: boolean,
      pkExtra: Record<string, unknown> = {},
    ) => {
      const upstream = await startOpenAiUpstream(opts);
      upstreams.push(upstream);
      const pk = await seed.createProviderKey({
        display_name: `${display}-pk`,
        secret: "sk-mock",
        api_base: provider === "openai" ? `${upstream.baseUrl}/v1` : upstream.baseUrl,
        ...pkExtra,
      });
      const m = await seed.createModel({
        display_name: display,
        provider,
        model_name: provider === "openai" ? "gpt-4o-mini" : "claude-3-5-haiku-20241022",
        provider_key_id: pk.id,
      });
      if (guarded) await seed.attachGuardrailToModel(hold.id, m.id);
      return upstream;
    };
    await model("mem-chat", "openai", { streamEvents: CHAT_SLOW, eventDelayMs: 100 }, true);
    await model("mem-msg-native", "anthropic", { streamEvents: ANTHROPIC_SLOW, eventDelayMs: 100 }, true);
    await model("mem-msg-bridge", "openai", { streamEvents: CHAT_SLOW, eventDelayMs: 100 }, true);
    await model("mem-resp-native", "openai", { streamEvents: RESPONSES_SLOW, eventDelayMs: 100 }, true);
    await model("mem-resp-bridge", "openai", { streamEvents: CHAT_SLOW, eventDelayMs: 100 }, true, { apis: {} });
    await model(
      "mem-slow",
      "openai",
      {
        responseDelayMs: 3_000,
        nonStreamBody: {
          id: "chatcmpl-slow",
          object: "chat.completion",
          model: "gpt-4o-mini",
          choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        },
      },
      false,
    );
    const ptUpstream = await startOpenAiUpstream({ streamEvents: CHAT_SLOW, eventDelayMs: 100 });
    upstreams.push(ptUpstream);
    const ptKey = await seed.createProviderKey({
      display_name: "mem-pt-pk",
      secret: "sk-mock",
      api_base: ptUpstream.baseUrl,
    });
    await seed.createPassthroughRoute({
      name: "mem-pt",
      path_prefix: "/mem-pt",
      target_url: ptUpstream.baseUrl,
      provider_key_id: ptKey.id,
    });
    // Env-wide, so the passthrough route holds its stream back too.
    await seed.createGuardrail({
      name: "mem-pt-hold",
      enabled: true,
      hook_point: "output",
      kind: "pii",
      detectors: [{ type: "email", action: "block" }],
    });

    await seed.createApiKey({
      key_hash: CALLER_HASH,
      allowed_models: [
        "mem-chat",
        "mem-msg-native",
        "mem-msg-bridge",
        "mem-resp-native",
        "mem-resp-bridge",
        "mem-slow",
      ],
      allowed_routes: ["*"],
    });
    const proxy = new ProxyClient(app.proxyUrl, CALLER);
    await waitConfigPropagation(async () => (await proxy.listModels()).status === 200);
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
    if (dumpDir) await rm(dumpDir, { recursive: true, force: true });
  });

  const ready = (ctx: { skip: () => void }) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return false;
    }
    return true;
  };

  const post = (path: string, body: unknown) =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER}`,
        "x-api-key": CALLER,
      },
      body: JSON.stringify(body),
    });

  test("automatic dumps fire once per threshold and keep only the newest files", async (ctx) => {
    if (!ready(ctx)) return;
    const samples = await pollMetrics(
      app!,
      (s) => sumMetric(s, "aisix_heap_profile_dumps_total", { trigger: "auto", result: "ok" }) >= 3,
      "three automatic heap dumps",
      60_000,
    );
    expect(sumMetric(samples, "aisix_heap_profile_dumps_total", { trigger: "auto", result: "ok" })).toBe(3);
    expect(sumMetric(samples, "aisix_heap_profile_dumps_total", { trigger: "auto", result: "error" })).toBe(0);
    const files = (await readdir(dumpDir)).filter((f) => f.endsWith(".pb.gz"));
    expect(files, "keep: 2 leaves the two newest of three").toHaveLength(2);
    for (const f of files) {
      expect(f).toMatch(/^\d{8}T\d{6}\.\d{3}Z-auto-0\.000[23]\.pb\.gz$/);
      const profile = readProfile(gunzipSync(await readFile(join(dumpDir, f))));
      expect(profile.sampleTypes).toContain("inuse_space");
    }
  });

  test("GET /debug/pprof/heap returns a symbolized gzipped pprof", async (ctx) => {
    if (!ready(ctx)) return;
    const before = await scrapeMetrics(app!.metricsUrl);
    const res = await fetch(`${app!.debugUrl}/debug/pprof/heap`);
    expect(res.status).toBe(200);
    const body = Buffer.from(await res.arrayBuffer());
    expect(body.subarray(0, 2)).toEqual(Buffer.from([0x1f, 0x8b]));
    const profile = readProfile(gunzipSync(body));
    expect(profile.sampleTypes).toContain("inuse_space");
    expect(profile.functions.length).toBeGreaterThan(0);
    expect(
      profile.functions.some((f) => /^aisix_[a-z_]+::/.test(f)),
      `no gateway function among ${profile.functions.length} names`,
    ).toBe(true);
    const after = await scrapeMetrics(app!.metricsUrl);
    expect(
      sumMetric(after, "aisix_heap_profile_dumps_total", { trigger: "manual", result: "ok" }) -
        sumMetric(before, "aisix_heap_profile_dumps_total", { trigger: "manual", result: "ok" }),
    ).toBe(1);
  });

  test("a heap profile requested while one is being taken is refused with 429", async (ctx) => {
    if (!ready(ctx)) return;
    const statuses = await Promise.all(
      Array.from({ length: 6 }, async () => {
        const res = await fetch(`${app!.debugUrl}/debug/pprof/heap`);
        await res.arrayBuffer();
        return res.status;
      }),
    );
    expect(statuses.every((s) => s === 200 || s === 429), statuses.join(",")).toBe(true);
    expect(statuses).toContain(200);
    expect(statuses).toContain(429);
  });

  test("allocator, process, limit and runtime families report this process", async (ctx) => {
    if (!ready(ctx)) return;
    const s = await scrapeMetrics(app!.metricsUrl);
    const alloc = (stat: string) => sample(s, "aisix_allocator_bytes", { stat })?.value ?? NaN;
    for (const stat of ["allocated", "active", "resident", "mapped", "retained", "metadata"]) {
      expect(sample(s, "aisix_allocator_bytes", { stat }), stat).toBeDefined();
    }
    expect(alloc("allocated")).toBeGreaterThan(0);
    expect(alloc("active")).toBeGreaterThanOrEqual(alloc("allocated"));
    expect(alloc("mapped")).toBeGreaterThanOrEqual(alloc("active"));
    expect(alloc("metadata")).toBeGreaterThan(0);

    const rss = sample(s, "process_resident_memory_bytes")!.value;
    expect(rss).toBeGreaterThan(1 << 20);
    expect(sample(s, "process_virtual_memory_bytes")!.value).toBeGreaterThanOrEqual(rss);
    expect(sample(s, "process_threads")!.value).toBeGreaterThanOrEqual(2);
    const fds = sample(s, "process_open_fds")!.value;
    expect(fds).toBeGreaterThan(0);
    expect(sample(s, "process_max_fds")!.value).toBeGreaterThanOrEqual(fds);
    expect(sample(s, "process_cpu_seconds_total")!.value).toBeGreaterThan(0);
    const started = sample(s, "process_start_time_seconds")!.value;
    expect(started).toBeGreaterThan(Date.now() / 1000 - 600);
    expect(started).toBeLessThanOrEqual(Date.now() / 1000);

    const limit = await cgroupLimit(app!.pid);
    expect(sample(s, "aisix_memory_limit_bytes")?.value).toBe(limit);

    expect(sample(s, "aisix_runtime_alive_tasks", { runtime: "control" })!.value).toBeGreaterThan(0);
    expect(sample(s, "aisix_runtime_global_queue_depth", { runtime: "control" })).toBeDefined();
    expect(sample(s, "aisix_runtime_alive_tasks", { runtime: "tpc-0" })!.value).toBeGreaterThan(0);
  });

  test("every component store is reported", async (ctx) => {
    if (!ready(ctx)) return;
    const s = await scrapeMetrics(app!.metricsUrl);
    for (const component of [
      "usage_event_queue",
      "log_queue",
      "response_cache",
      "semantic_cache",
      "budget_cache",
      "route_embedding_cache",
      "ratelimit_local_keys",
      "snapshot_pending_reclaim",
      "metric_series",
      "upstream_clients",
      "in_flight_request_bodies",
    ]) {
      expect(sample(s, "aisix_component_entries", { component }), component).toBeDefined();
    }
    for (const component of ["log_queue", "guardrail_holdback", "in_flight_request_bodies"]) {
      expect(sample(s, "aisix_component_bytes", { component }), component).toBeDefined();
    }
    // The scrape itself is a few hundred series.
    expect(sample(s, "aisix_component_entries", { component: "metric_series" })!.value).toBeGreaterThan(50);
  });

  const holdbackBytes = (s: MetricSample[]) =>
    sample(s, "aisix_component_bytes", { component: "guardrail_holdback" })?.value ?? NaN;

  test.for([
    ["chat", "/v1/chat/completions", { model: "mem-chat", stream: true, messages: [{ role: "user", content: "go" }] }],
    ["messages native", "/v1/messages", { model: "mem-msg-native", stream: true, max_tokens: 64, messages: [{ role: "user", content: "go" }] }],
    ["messages bridged", "/v1/messages", { model: "mem-msg-bridge", stream: true, max_tokens: 64, messages: [{ role: "user", content: "go" }] }],
    ["responses native", "/v1/responses", { model: "mem-resp-native", stream: true, input: "go" }],
    ["responses bridged", "/v1/responses", { model: "mem-resp-bridge", stream: true, input: "go" }],
    ["passthrough", "/mem-pt/v1/chat/completions", { model: "gpt-4o-mini", stream: true, messages: [{ role: "user", content: "go" }] }],
  ] as const)("%s: held-back stream bytes are counted while held and returned after", async ([, path, body], ctx) => {
    if (!ready(ctx)) return;
    await pollMetrics(app!, (s) => holdbackBytes(s) === 0, "an idle hold-back gauge");
    const inFlight = post(path, body).then(async (res) => ({ status: res.status, text: await res.text() }));
    // Nothing is released until the whole stream has been scanned, so the
    // held bytes grow past a content frame's text while it is still arriving.
    await pollMetrics(app!, (s) => holdbackBytes(s) > TEXT.length, "held-back content bytes");
    const done = await inFlight;
    expect(done.status).toBe(200);
    expect(done.text).toContain(TEXT);
    await pollMetrics(app!, (s) => holdbackBytes(s) === 0, "the hold-back gauge back at zero");
  });

  const bodyGauge = (s: MetricSample[]) => ({
    entries: sample(s, "aisix_component_entries", { component: "in_flight_request_bodies" })?.value ?? NaN,
    bytes: sample(s, "aisix_component_bytes", { component: "in_flight_request_bodies" })?.value ?? NaN,
  });
  // Two MiB of inline "image" data waiting three seconds on its upstream.
  const BIG = "A".repeat(2 * 1024 * 1024);

  test.for([
    ["chat", "/v1/chat/completions", { model: "mem-slow", messages: [{ role: "user", content: BIG }] }],
    ["messages", "/v1/messages", { model: "mem-slow", max_tokens: 16, messages: [{ role: "user", content: BIG }] }],
    ["responses", "/v1/responses", { model: "mem-slow", input: BIG }],
    ["embeddings", "/v1/embeddings", { model: "mem-slow", input: BIG }],
  ] as const)("%s: a large request body is counted while its request is in flight", async ([, path, body], ctx) => {
    if (!ready(ctx)) return;
    await pollMetrics(app!, (s) => bodyGauge(s).entries === 0, "no bodies in flight");
    const inFlight = post(path, body).then(async (res) => {
      await res.arrayBuffer();
      return res.status;
    });
    const during = await pollMetrics(
      app!,
      (s) => bodyGauge(s).bytes >= BIG.length,
      "the request body counted in flight",
    );
    expect(bodyGauge(during).entries).toBe(1);
    await inFlight;
    const after = await pollMetrics(app!, (s) => bodyGauge(s).entries === 0, "the body released");
    expect(bodyGauge(after).bytes).toBe(0);
  });
});

describe("memory observability: serving modes", () => {
  test("without thread-per-core only the control runtime is reported", async (ctx) => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return ctx.skip();
    const app = await spawnApp({ threadPerCore: false });
    try {
      const s = await scrapeMetrics(app.metricsUrl);
      const runtimes = s.filter((x) => x.name === "aisix_runtime_alive_tasks").map((x) => x.labels.runtime);
      expect(runtimes).toEqual(["control"]);
    } finally {
      await app.exit();
    }
  });

  test("the debug listener binds loopback only by default", async (ctx) => {
    const etcd = new EtcdClient();
    if (!(await etcd.ping())) return ctx.skip();
    const external = Object.values(networkInterfaces())
      .flat()
      .find((i) => i && i.family === "IPv4" && !i.internal)?.address;
    expect(external, "the host needs a non-loopback address to probe").toBeDefined();
    const app = await spawnApp({ debug: "binary-default" });
    try {
      expect(app.debugUrl).toBe("http://127.0.0.1:9091");
      const res = await fetch(`${app.debugUrl}/debug/pprof/heap`);
      expect(res.status).toBe(200);
      await res.arrayBuffer();
      const refused = await new Promise<string>((resolve) => {
        const sock = connect({ host: external!, port: 9091 });
        sock.once("connect", () => {
          sock.destroy();
          resolve("connected");
        });
        sock.once("error", (e: NodeJS.ErrnoException) => resolve(e.code ?? "error"));
      });
      expect(refused).toBe("ECONNREFUSED");
    } finally {
      await app.exit();
    }
  });
});
