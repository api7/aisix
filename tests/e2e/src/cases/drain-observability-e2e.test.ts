import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  spawnApp,
  startOpenAiUpstream,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the connection-level identity a rolling update has to be traced by.
//
// Reconstructing what happened to a request during a rollout means joining
// three records that no shared field used to connect: the fronting proxy's
// log of the connection it dispatched on, the gateway's log of the request,
// and the fact that a request was ever received at all. The fields below
// close that gap, and each is asserted through the binary's real stderr —
// the only place an operator would look for them.
//
// The proxy's own `request_id` is deliberately NOT the join key: it is
// minted here, so nothing upstream of the gateway knows it.

const CALLER_PLAINTEXT = "sk-drain-obs-caller";
const CALLER_KEY_ENV = "DRAIN_OBS_CALLER_KEY";

/** Long enough that the drain heartbeat (5s) fires at least once. */
const DRAIN_WINDOW_SECS = 2;
/** Outlives the window several times over, so requests span the drain. */
const SLOW_UPSTREAM_MS = 14_000;

function resources(upstreamBase: string): string {
  return `
_format_version: "1"
provider_keys:
  - display_name: drain-obs-pk
    provider: openai
    api_key: sk-mock
    api_base: ${upstreamBase}/v1
models:
  - display_name: drain-obs
    provider: openai
    model_name: gpt-4o-mini
    provider_key: drain-obs-pk
api_keys:
  - display_name: drain-obs-caller
    key_env: ${CALLER_KEY_ENV}
    allowed_models: ["drain-obs"]
`;
}

function chat(proxyUrl: string, headers: Record<string, string> = {}): Promise<Response> {
  return fetch(`${proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
      ...headers,
    },
    body: JSON.stringify({
      model: "drain-obs",
      messages: [{ role: "user", content: "hi" }],
    }),
  });
}

/** The access-log line for `requestId`, or undefined while it is absent. */
function accessLogFor(output: string, requestId: string): string | undefined {
  return output
    .split("\n")
    .find(
      (line) =>
        line.includes("proxy request completed") &&
        line.includes(`request_id="${requestId}"`),
    );
}

/** Poll until `read` yields a value, or fail with `what`. */
async function waitFor<T>(
  read: () => T | undefined,
  what: string,
  timeoutMs = 20_000,
): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = read();
    if (value !== undefined) return value;
    if (Date.now() >= deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 50));
  }
}

describe("access log carries the downstream connection identity", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    upstream = await startOpenAiUpstream();
    app = await spawnApp({
      // The fields under test are reported at INFO; the suite default is WARN.
      logLevel: "info",
      resourcesFile: resources(upstream.baseUrl),
      extraEnv: { [CALLER_KEY_ENV]: CALLER_PLAINTEXT },
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("logs the peer socket address, with its port", async () => {
    if (!app) throw new Error("setup failed");

    const res = await chat(app.proxyUrl);
    expect(res.status).toBe(200);
    await res.text();
    const requestId = res.headers.get("x-aisix-request-id")!;
    expect(requestId).toBeTruthy();

    const line = await waitFor(
      () => accessLogFor(app!.output(), requestId),
      "the access-log line for the request",
    );
    // The port is the point: an address without one names a host, not a
    // connection, and a host-network deployment behind a layer-4 balancer
    // has every request arriving from the same handful of hosts.
    const peer = /\bpeer=(\d{1,3}(?:\.\d{1,3}){3}):(\d+)\b/.exec(line);
    expect(peer, `no peer=<ip>:<port> on: ${line}`).not.toBeNull();
    expect(Number(peer![2])).toBeGreaterThan(0);
  });

  test("records the fronting proxy's request id as its own field", async () => {
    if (!app) throw new Error("setup failed");

    const downstreamId = `ingress-${Date.now()}`;
    const res = await chat(app.proxyUrl, { "x-request-id": downstreamId });
    expect(res.status).toBe(200);
    await res.text();
    const requestId = res.headers.get("x-aisix-request-id")!;

    // Two ids, two fields. The gateway must not adopt the ingress's id by
    // default, and must not report its own under the ingress's name.
    expect(requestId).not.toBe(downstreamId);

    const line = await waitFor(
      () => accessLogFor(app!.output(), requestId),
      "the access-log line for the request",
    );
    expect(line).toContain(`downstream_request_id=${downstreamId}`);
  });

  test("omits the field for a request that carried no downstream id", async () => {
    if (!app) throw new Error("setup failed");

    const res = await chat(app.proxyUrl);
    expect(res.status).toBe(200);
    await res.text();
    const requestId = res.headers.get("x-aisix-request-id")!;

    const line = await waitFor(
      () => accessLogFor(app!.output(), requestId),
      "the access-log line for the request",
    );
    // An always-present empty field would defeat filtering on it — the
    // same rule the failure fields on this line already follow. Pinned
    // against `peer`, which the same request DOES carry, so the assertion
    // cannot pass merely because neither field is reported at all.
    expect(line).toMatch(/\bpeer=\d{1,3}(?:\.\d{1,3}){3}:\d+\b/);
    expect(line).not.toContain("downstream_request_id");
  });

  test("ignores a downstream id that is not safe to log verbatim", async () => {
    if (!app) throw new Error("setup failed");

    // A value with a space cannot be a correlation id and must not be
    // written into a whitespace-delimited log line as one.
    const res = await chat(app.proxyUrl, { "x-request-id": "not an id" });
    expect(res.status).toBe(200);
    await res.text();
    const requestId = res.headers.get("x-aisix-request-id")!;

    const line = await waitFor(
      () => accessLogFor(app!.output(), requestId),
      "the access-log line for the request",
    );
    expect(line).toMatch(/\bpeer=\d{1,3}(?:\.\d{1,3}){3}:\d+\b/);
    expect(line).not.toContain("downstream_request_id");
  });
});

describe("the drain reports what is still arriving and what is still open", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;

  beforeAll(async () => {
    upstream = await startOpenAiUpstream({ responseDelayMs: SLOW_UPSTREAM_MS });
    app = await spawnApp({
      logLevel: "info",
      resourcesFile: resources(upstream.baseUrl),
      extraEnv: { [CALLER_KEY_ENV]: CALLER_PLAINTEXT },
      extra: { shutdown: { min_drain_secs: DRAIN_WINDOW_SECS } },
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test(
    "logs arrival, the accept, and the open-connection count",
    async () => {
      if (!app || !upstream) throw new Error("setup failed");
      const proxyUrl = app.proxyUrl;

      // A request that arrives BEFORE the signal, and is still running
      // when it lands. Its completion line is the control: arrival is a
      // drain-only line, so a request received in steady state must not
      // produce one.
      const beforeSignal = chat(proxyUrl, { "x-request-id": "ingress-before" });
      await waitFor(
        () => (upstream!.receivedRequests.length > 0 ? true : undefined),
        "the first request to reach the upstream",
      );
      expect(app.output()).not.toContain("request arrived while draining");

      app.signal("SIGTERM");
      await waitFor(
        () => (app!.output().includes("draining — /readyz") ? true : undefined),
        "the drain to start",
      );
      // The signal-time line already reports both counts, so an operator
      // knows what the drain inherited.
      const startLine = app
        .output()
        .split("\n")
        .find((l) => l.includes("draining — /readyz"))!;
      expect(startLine).toMatch(/\bin_flight=\d+\b/);
      expect(startLine).toMatch(/\bopen_connections=[1-9]\d*\b/);

      // A second request, on a new connection, inside the window — the
      // shape that says the balancer has not withdrawn this instance yet.
      const duringDrain = chat(proxyUrl, { "x-request-id": "ingress-during" });

      const acceptLine = await waitFor(
        () =>
          app!
            .output()
            .split("\n")
            .find((l) => l.includes("accepted a new connection while draining")),
        "the accept-during-drain line",
      );
      expect(acceptLine).toMatch(/\bpeer=\d{1,3}(?:\.\d{1,3}){3}:\d+\b/);
      expect(acceptLine).toMatch(/\bopen_connections=[1-9]\d*\b/);

      const arrivalLine = await waitFor(
        () =>
          app!
            .output()
            .split("\n")
            .find((l) => l.includes("request arrived while draining")),
        "the arrival-during-drain line",
      );
      expect(arrivalLine).toContain("method=POST");
      expect(arrivalLine).toContain("path=/v1/chat/completions");
      // Arrival is written before anything about the request is known, so
      // its only identity is the connection's — which is exactly what a
      // request killed before it could complete leaves behind.
      expect(arrivalLine).toMatch(/\bpeer=\d{1,3}(?:\.\d{1,3}){3}:\d+\b/);
      expect(arrivalLine).toContain("downstream_request_id=ingress-during");

      // Past the window, with both requests still running: the heartbeat
      // separates work already taken on from connections still held.
      const heartbeat = await waitFor(
        () =>
          app!
            .output()
            .split("\n")
            .find((l) => l.includes("still draining in-flight requests")),
        "a drain heartbeat",
      );
      expect(heartbeat).toMatch(/\bin_flight=[1-9]\d*\b/);
      expect(heartbeat).toMatch(/\bopen_connections=[1-9]\d*\b/);

      for (const pending of [beforeSignal, duringDrain]) {
        const res = await pending;
        expect(res.status).toBe(200);
        await res.text();
      }
    },
    90_000,
  );
});
