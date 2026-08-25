import { createServer, type Server } from "node:http";
import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  ProxyClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: the `custom` guardrail runs an operator-supplied script inside the
// gateway. The point of the kind is that the adapter lives here rather than
// in a service the operator has to deploy, so this exercises the whole path
// against a real screening service with its own protocol: a real aisix
// binary loads the script from etcd, runs it in the embedded sandbox, and
// the script calls the service and maps its answer onto a verdict.
//
// Covered: block from the script's own decision, allow, the script reading
// a configured secret and the service seeing it on the wire, and
// fail-closed when the screening service is down.

const CALLER = "sk-custom-script-e2e-caller";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

const RISKY_MARKER = "customscriptriskymarker";
const OUTAGE_MARKER = "customscriptoutagemarker";
const SCAN_KEY = "sk-screening-live-42";

interface ScreeningMock {
  baseUrl: string;
  requests: Array<{ apiKey: string | undefined; text: string }>;
  close(): Promise<void>;
}

// A screening service that speaks a shape no built-in kind understands:
// the verdict is nested under `outcome`, and it answers 503 for the
// outage marker.
async function startScreeningMock(): Promise<ScreeningMock> {
  const requests: ScreeningMock["requests"] = [];
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      let text = "";
      try {
        const body = JSON.parse(raw);
        text = typeof body.text === "string" ? body.text : "";
      } catch {
        // leave default
      }
      requests.push({
        apiKey: req.headers["x-api-key"] as string | undefined,
        text,
      });

      if (text.includes(OUTAGE_MARKER)) {
        res.statusCode = 503;
        res.end("screening service unavailable");
        return;
      }
      if (req.headers["x-api-key"] !== SCAN_KEY) {
        res.statusCode = 401;
        res.end("bad key");
        return;
      }

      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          outcome: {
            deny: text.includes(RISKY_MARKER),
            rule: text.includes(RISKY_MARKER) ? "R-17" : null,
          },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("screening mock did not bind a port");
  }
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    requests,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

// What an operator actually writes: read the prompt, call their own
// service with their own credential, map their own response shape.
const SCRIPT = (baseUrl: string) => `
export async function checkInput(ctx) {
  const resp = await fetch("${baseUrl}/scan", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": ctx.secrets.SCAN_KEY,
    },
    body: JSON.stringify({ text: ctx.text }),
  });
  if (!resp.ok) {
    throw new Error("screening returned " + resp.status);
  }
  const body = await resp.json();
  if (body.outcome.deny) {
    return { action: "block", reason_code: body.outcome.rule };
  }
  return { action: "none" };
}
`;

describe("custom script guardrail e2e: operator script screens against its own service", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let screening: ScreeningMock | undefined;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    screening = await startScreeningMock();

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-clean",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "a safe and clean reply" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 8, total_tokens: 13 },
      },
    });

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "custom-script-e2e-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "custom-script-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    // fail_open=false so an unreachable screening service blocks rather
    // than releasing unscreened traffic — the default a security control
    // should have.
    await seed.createGuardrail({
      name: "custom-script-e2e-guard",
      enabled: true,
      hook_point: "input",
      fail_open: false,
      kind: "custom",
      script: SCRIPT(screening.baseUrl),
      secrets: { SCAN_KEY },
      timeout_ms: 5000,
    });

    // Seeded last so that this key authenticating implies every resource
    // above it is already in the gateway's snapshot.
    await seed.createApiKey({
      key_hash: hash(CALLER),
      allowed_models: ["custom-script-e2e"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
    await screening?.close();
  });

  const client = () =>
    new OpenAI({
      apiKey: CALLER,
      baseURL: `${app!.proxyUrl}/v1`,
      maxRetries: 0,
    });

  const send = (content: string) =>
    client().chat.completions.create({
      model: "custom-script-e2e",
      messages: [{ role: "user", content }],
    });

  const expectBlocked = async (content: string) => {
    let caught: unknown;
    try {
      await send(content);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(APIError);
    if (!(caught instanceof APIError)) throw new Error("unreachable");
    expect(caught.status).toBe(422);
    expect((caught.error as { type?: unknown })?.type).toBe("content_filter");
    return caught;
  };

  const ensureSeedLive = () =>
    waitConfigPropagation(async () => {
      const res = await new ProxyClient(app!.proxyUrl, CALLER).listModels();
      if (res.status !== 200) return false;
      const data = (res.body as { data?: Array<{ id?: string }> }).data ?? [];
      return data.some((m) => m.id === "custom-script-e2e");
    });

  test("the script's block decision reaches the caller as 422, upstream never called", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    await ensureSeedLive();

    const before = upstream!.receivedRequests.length;
    await expectBlocked(`please help with ${RISKY_MARKER}`);
    expect(upstream!.receivedRequests.length).toBe(before);
  });

  test("the script sees the configured secret and the service receives it", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    await ensureSeedLive();

    const before = screening!.requests.length;
    await expectBlocked(`please help with ${RISKY_MARKER}`);
    const seen = screening!.requests.slice(before);
    expect(seen.length).toBeGreaterThan(0);
    // The 401 branch of the mock would have made the script throw, so a
    // clean verdict already implies the key arrived; assert it directly
    // so a regression names the cause.
    expect(seen.every((r) => r.apiKey === SCAN_KEY)).toBe(true);
    expect(seen.some((r) => r.text.includes(RISKY_MARKER))).toBe(true);
  });

  test("clean content passes through to the upstream", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    await ensureSeedLive();

    const before = upstream!.receivedRequests.length;
    const reply = await send("what is the capital of France");
    expect(reply.choices[0]?.message?.content).toBe("a safe and clean reply");
    expect(upstream!.receivedRequests.length).toBe(before + 1);
  });

  test("a screening outage blocks rather than releasing unscreened traffic", async (ctx) => {
    if (!etcdReachable) ctx.skip();
    await ensureSeedLive();

    // The script throws on a non-2xx, and fail_open=false turns that into
    // a block: the row is a security control, so its failure must not be
    // an open door.
    const before = upstream!.receivedRequests.length;
    await expectBlocked(`please help with ${OUTAGE_MARKER}`);
    expect(upstream!.receivedRequests.length).toBe(before);
  });
});
