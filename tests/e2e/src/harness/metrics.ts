/**
 * Reading the DP's Prometheus scrape from a test.
 *
 * Assertions on these counters must be DELTAS across one action, never
 * absolute values or "no such series exists": the suite drives readiness
 * probes and several tests through one app, so any series the action does
 * not touch still carries whatever earlier traffic left in it.
 */

/** One counter sample from a `GET /metrics` scrape. */
export interface MetricSample {
  name: string;
  labels: Record<string, string>;
  value: number;
}

/** Parse the counter samples out of a Prometheus text scrape. */
export async function scrapeMetrics(
  metricsUrl: string,
  path = "/metrics",
): Promise<MetricSample[]> {
  const res = await fetch(`${metricsUrl}${path}`);
  if (!res.ok) {
    throw new Error(`scrape ${metricsUrl}${path} returned ${res.status}`);
  }
  const out: MetricSample[] = [];
  for (const line of (await res.text()).split("\n")) {
    const m = /^([a-z_]+)(\{(.*)\})? ([0-9.e+-]+)$/.exec(line.trim());
    if (!m) continue;
    const labels: Record<string, string> = {};
    for (const pair of m[3]?.match(/[a-z_]+="[^"]*"/g) ?? []) {
      const eq = pair.indexOf("=");
      labels[pair.slice(0, eq)] = pair.slice(eq + 2, -1);
    }
    out.push({ name: m[1], labels, value: Number(m[4]) });
  }
  return out;
}

/**
 * Total of every sample of `name` whose labels include `want`, or that
 * satisfies `want` when it is a predicate.
 *
 * Summing rather than requiring a single matching series is deliberate:
 * these families carry label dimensions an individual test says nothing
 * about, and pinning the whole tuple would break the test the next time one
 * is added.
 */
export function sumMetric(
  samples: MetricSample[],
  name: string,
  want: Record<string, string> | ((labels: Record<string, string>) => boolean) = {},
): number {
  const matches =
    typeof want === "function"
      ? want
      : (labels: Record<string, string>) =>
          Object.entries(want).every(([k, v]) => labels[k] === v);
  return samples
    .filter((s) => s.name === name && matches(s.labels))
    .reduce((acc, s) => acc + s.value, 0);
}

/**
 * `sumMetric(after) - sumMetric(before)` — the shape every assertion on
 * these counters should take.
 */
export function metricDelta(
  before: MetricSample[],
  after: MetricSample[],
  name: string,
  want: Record<string, string> | ((labels: Record<string, string>) => boolean) = {},
): number {
  return sumMetric(after, name, want) - sumMetric(before, name, want);
}
