import { createHash } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  spawnApp,
  startOpenAiUpstream,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// AISIX-Cloud#1226: the two request-latency histograms no longer share one
// bucket set, and every set is operator-overridable.
//
// Both halves are observable contracts, so both are pinned here against a
// real scrape rather than against the Rust constants: a dashboard reads
// `le` values off this exposition, and the override half is the only way a
// deployment can move them. The override is delivered through an `AISIX_*`
// environment variable because that is the channel the Kubernetes chart
// uses — config there is injected as env vars on top of an image-baked
// config file, so a knob that only works from YAML is unreachable in the
// deployment that needs it most.

const CALLER_PLAINTEXT = "sk-histogram-buckets-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const E2E_SERIES = "aisix_request_e2e_latency_seconds";
const TTFT_SERIES = "aisix_request_ttft_seconds";

function resources(upstreamBase: string): string {
  return `
_format_version: "1"
provider_keys:
  - display_name: buckets-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstreamBase}/v1
models:
  - display_name: buckets-model
    provider: openai
    model_name: gpt-4o-mini
    provider_key: buckets-pk
api_keys:
  - display_name: buckets-caller
    key_hash: ${CALLER_KEY_HASH}
    allowed_models: ["buckets-model"]
`;
}

/** Every `le` value the scrape exposes for `series`, in exposition order. */
function edgesOf(body: string, series: string): string[] {
  return body
    .split("\n")
    .filter((l) => l.startsWith(`${series}_bucket{`))
    .map((l) => l.match(/le="([^"]+)"/)?.[1])
    .filter((le): le is string => le !== undefined);
}

async function streamOnce(app: SpawnedApp): Promise<void> {
  const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "buckets-model",
      messages: [{ role: "user", content: "hello" }],
      stream: true,
    }),
  });
  await res.text();
  expect(res.status).toBe(200);
}

/**
 * Scrape until both histograms have appeared. A streaming request records
 * its TTFT at stream completion, so the first scrape after the response
 * body is drained can still race ahead of the observation.
 */
async function scrapeBothSeries(app: SpawnedApp): Promise<string> {
  const deadline = Date.now() + 10_000;
  for (;;) {
    const res = await fetch(`${app.metricsUrl}/metrics`);
    expect(res.status).toBe(200);
    const body = await res.text();
    const ready = edgesOf(body, E2E_SERIES).length > 0 && edgesOf(body, TTFT_SERIES).length > 0;
    if (ready || Date.now() > deadline) return body;
    await new Promise((r) => setTimeout(r, 100));
  }
}

function startStreamingUpstream(): Promise<OpenAiUpstream> {
  return startOpenAiUpstream({
    streamEvents: [
      '{"id":"buckets","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
      '{"id":"buckets","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}',
      '{"id":"buckets","object":"chat.completion.chunk","model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
      "[DONE]",
    ],
    eventDelayMs: 20,
  });
}

describe("histogram buckets e2e: TTFT and e2e latency expose distinct default edges", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    upstream = await startStreamingUpstream();
    app = await spawnApp({ resourcesFile: resources(upstream.baseUrl) });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("the two series carry their own edges, and TTFT is not a copy of e2e", async () => {
    await streamOnce(app!);
    const body = await scrapeBothSeries(app!);

    // The published default sets. Pinned literally: moving an edge breaks
    // every dashboard that selects on it, so it must be a deliberate edit.
    expect(edgesOf(body, E2E_SERIES)).toEqual([
      "0.005",
      "0.01",
      "0.025",
      "0.05",
      "0.1",
      "0.25",
      "0.5",
      "1",
      "2",
      "5",
      "10",
      "30",
      "60",
      "120",
      "300",
      "420",
      "600",
      "+Inf",
    ]);
    expect(edgesOf(body, TTFT_SERIES)).toEqual([
      "0.05",
      "0.1",
      "0.25",
      "0.5",
      "1",
      "2",
      "5",
      "10",
      "30",
      "60",
      "120",
      "300",
      "+Inf",
    ]);

    // The property behind those two lists, stated directly: TTFT drops the
    // millisecond edges it can never reach through a hosted provider, and
    // e2e keeps the headroom above 300s that a long generation needs.
    for (const le of ["0.005", "0.01", "0.025"]) {
      expect(edgesOf(body, TTFT_SERIES), `TTFT must not carry le=${le}`).not.toContain(le);
      expect(edgesOf(body, E2E_SERIES), `e2e must keep le=${le}`).toContain(le);
    }
    expect(edgesOf(body, E2E_SERIES)).toContain("600");
    expect(edgesOf(body, TTFT_SERIES)).not.toContain("600");

    // Still a well-formed histogram: +Inf last, and the sum/count pair
    // histogram_quantile() needs.
    for (const series of [E2E_SERIES, TTFT_SERIES]) {
      expect(edgesOf(body, series).at(-1)).toBe("+Inf");
      expect(body).toContain(`${series}_sum`);
      expect(body).toContain(`${series}_count`);
    }
  });
});

describe("histogram buckets e2e: an operator override replaces only the named metric", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    upstream = await startStreamingUpstream();
    app = await spawnApp({
      resourcesFile: resources(upstream.baseUrl),
      extraEnv: {
        AISIX_OBSERVABILITY__METRICS__BUCKETS__REQUEST_TTFT: "0.5,3",
      },
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("the overridden series uses the supplied edges; the others keep defaults", async () => {
    await streamOnce(app!);
    const body = await scrapeBothSeries(app!);

    expect(edgesOf(body, TTFT_SERIES)).toEqual(["0.5", "3", "+Inf"]);
    // Overriding one metric must not re-cut the others — the reason the
    // knob is per-metric rather than one list for every histogram.
    expect(edgesOf(body, E2E_SERIES).at(0)).toBe("0.005");
    expect(edgesOf(body, E2E_SERIES).at(-2)).toBe("600");
  });
});
