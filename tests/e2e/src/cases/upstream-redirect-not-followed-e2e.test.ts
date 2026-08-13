import { createHash } from "node:crypto";
import { createServer, type Server as HttpServer } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an upstream that answers a dispatched request with a redirect must
// not be followed.
//
// The configured endpoint is the only host an operator has authorised the
// gateway to send a caller's prompt and the Provider Key's credential to.
// A redirect names a different one, chosen by the upstream at request
// time, and nothing on the caller's side — status, body, or access log —
// would say the answer came from somewhere else.
//
// The scenario: the configured endpoint answers `301` with a `Location`
// pointing at a second, perfectly healthy OpenAI-shaped upstream. If the
// gateway follows, the caller gets that second host's completion and the
// call looks entirely successful. It must instead fail the request as a
// bad gateway, and the second host must never be dialed.

const CALLER_PLAINTEXT = "sk-redirect-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

describe("upstream redirects are not followed", () => {
  let app: SpawnedApp | undefined;
  let elsewhere: OpenAiUpstream | undefined;
  let redirector: HttpServer | undefined;
  let redirectorHits = 0;
  let seed: SeedClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    // The host the redirect points at: a healthy upstream that would
    // happily answer, so following the redirect produces a *success*
    // rather than an error — the failure mode that hides itself.
    elsewhere = await startOpenAiUpstream();

    const location = `${elsewhere.baseUrl}/v1/chat/completions`;
    redirector = createServer((req, res) => {
      redirectorHits += 1;
      req.resume();
      req.on("end", () => {
        res.writeHead(301, { location });
        res.end();
      });
    });
    await new Promise<void>((resolve) =>
      redirector!.listen(0, "127.0.0.1", resolve),
    );
    const address = redirector.address();
    if (address === null || typeof address === "string") {
      throw new Error("redirecting upstream: no listen address");
    }

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "redirect-pk",
      secret: "sk-mock-redirect",
      api_base: `http://127.0.0.1:${address.port}/v1`,
    });
    await seed.createModel({
      display_name: "redirect-gpt",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["redirect-gpt"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await elsewhere?.close();
    await new Promise<void>((resolve) =>
      redirector ? redirector.close(() => resolve()) : resolve(),
    );
  });

  test("a 301 from the configured endpoint fails the call instead of hopping", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }

    await waitConfigPropagation(async () => {
      try {
        const models = await fetch(`${app!.proxyUrl}/v1/models`, {
          headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
        });
        if (models.status !== 200) return false;
        const ids =
          ((await models.json()) as { data?: Array<{ id?: string }> }).data?.map(
            (m) => m.id,
          ) ?? [];
        return ids.includes("redirect-gpt");
      } catch {
        return false;
      }
    });

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
      },
      body: JSON.stringify({
        model: "redirect-gpt",
        messages: [{ role: "user", content: "hello" }],
      }),
    });

    const body = (await res.json()) as {
      error?: { message?: string; type?: string };
      choices?: unknown[];
    };

    // The upstream gave the gateway nothing it can answer with, which is
    // what a bad gateway is.
    expect(res.status, JSON.stringify(body)).toBe(502);
    expect(body.error, JSON.stringify(body)).toBeDefined();
    expect(body.choices).toBeUndefined();

    // The point of the case: the prompt and the key's credential never
    // left for the host the operator did not configure.
    expect(redirectorHits).toBeGreaterThan(0);
    expect(
      elsewhere!.receivedRequests.map((r) => `${r.method} ${r.path}`),
    ).toEqual([]);
  });
});
