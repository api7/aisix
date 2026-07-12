import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  pickFreePort,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E (#519 UV.3, e2e-able half): Bedrock guardrail credential
// isolation. Every ApplyGuardrail request is SigV4-signed, and the
// `Authorization: AWS4-HMAC-SHA256 Credential=<ACCESS_KEY_ID>/…` header
// is the wire-level equivalent of what AWS CloudTrail would report as
// the calling principal. A mock Bedrock endpoint records
// (guardrail id → signing key) per request, pinning:
//
//   1. Two guardrails with different static credentials sign with
//      EXACTLY their own key — no shared client/credential pool.
//   2. Rotating a guardrail's credentials takes effect (the client is
//      keyed by config, not cached forever).
//   3. Ambient AWS_* env credentials on the DP process NEVER appear on
//      the wire — the SDK's default credential chain must not bleed
//      into per-guardrail calls.
//
// What stays manual (the residual of UV.3): real-AWS semantics —
// assume-role/STS paths and actual IAM evaluation — which no mock can
// stand in for.

const CALLER_PLAINTEXT = "sk-bedrock-iso-caller-PLAINTEXT";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");
const PROVIDER_SECRET = "sk-mock-bedrock-iso";

const KEY_A = "AKIDGUARDAAAAAAAAAA1";
const KEY_A2 = "AKIDGUARDAROTATED002";
const KEY_B = "AKIDGUARDBBBBBBBBBB1";
const AMBIENT_KEY = "AKIDAMBIENTLEAK00000";

interface BedrockCall {
  guardrailId: string;
  accessKey: string;
  region: string;
}

interface MockBedrock {
  url: string;
  calls: BedrockCall[];
  close(): Promise<void>;
}

// Minimal ApplyGuardrail receiver: extracts the guardrail id from the
// request path and the signing ACCESS_KEY_ID + region from the SigV4
// Authorization header, then answers `action: NONE` so the chain
// allows the request (we're testing who signed, not blocking).
async function startMockBedrock(): Promise<MockBedrock> {
  const calls: BedrockCall[] = [];
  const credRe =
    /^AWS4-HMAC-SHA256 Credential=([^/]+)\/\d{8}\/([^/]+)\/([^/]+)\/aws4_request/;
  const pathRe = /^\/guardrail\/([^/]+)\/version\/[^/]+\/apply$/;
  const server: Server = createServer((req, res) => {
    let raw = "";
    req.on("data", (c: Buffer) => (raw += c.toString("utf8")));
    req.on("end", () => {
      const pm = pathRe.exec(req.url ?? "");
      const am = credRe.exec(req.headers.authorization ?? "");
      if (pm && am) {
        calls.push({
          guardrailId: decodeURIComponent(pm[1]),
          accessKey: am[1],
          region: am[2],
        });
      }
      res.statusCode = 200;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ action: "NONE", outputs: [] }));
    });
  });
  const port = await pickFreePort();
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}`,
    calls,
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

function bedrockGuardrail(
  name: string,
  guardrailId: string,
  accessKeyId: string,
) {
  return {
    name,
    enabled: true,
    hook_point: "input",
    fail_open: false,
    kind: "bedrock",
    guardrail_id: guardrailId,
    guardrail_version: "DRAFT",
    region: "us-east-1",
    aws_credentials: {
      kind: "static",
      access_key_id: accessKeyId,
      secret_access_key: `secret-for-${accessKeyId}`,
    },
    latency_mode: { kind: "serial" },
  };
}

async function seedRouting(seed: SeedClient, upstream: OpenAiUpstream) {
  const pk = await seed.createProviderKey({
    display_name: "bedrock-iso-pk",
    secret: PROVIDER_SECRET,
    api_base: `${upstream.baseUrl}/v1`,
  });
  await seed.createModel({
    display_name: "bedrock-iso-model",
    provider: "openai",
    model_name: "gpt-4o-mini",
    provider_key_id: pk.id,
  });
  await seed.createApiKey({
    key_hash: CALLER_KEY_HASH,
    allowed_models: ["bedrock-iso-model"],
  });
}

async function chat(app: SpawnedApp, content: string) {
  return fetch(`${app.proxyUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${CALLER_PLAINTEXT}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "bedrock-iso-model",
      messages: [{ role: "user", content }],
    }),
  });
}

const GUARD_A = "guardaaaa001";
const GUARD_B = "guardbbbb001";

describe("bedrock guardrail credential isolation (#519 UV.3)", () => {
  let etcdReachable = false;
  let upstream: OpenAiUpstream | undefined;
  let bedrock: MockBedrock | undefined;
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let guardrailAId: string | undefined;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    upstream = await startOpenAiUpstream();
    bedrock = await startMockBedrock();
    app = await spawnApp({
      extra: { bedrock_endpoint_url: bedrock.url },
      // Ambient credentials that must NEVER sign an ApplyGuardrail
      // call: the SDK's default chain would pick these up if the
      // per-guardrail static provider ever fell through.
      extraEnv: {
        AWS_ACCESS_KEY_ID: AMBIENT_KEY,
        AWS_SECRET_ACCESS_KEY: "ambient-secret-must-not-sign",
        AWS_REGION: "us-west-2",
      },
    });
    seed = new SeedClient(etcd, app.etcdPrefix);
    await seedRouting(seed, upstream);
    const created = (await seed.createGuardrail({
      ...bedrockGuardrail("gr-bedrock-a", GUARD_A, KEY_A),
    })) as { id: string };
    guardrailAId = created.id;
    await seed.createGuardrail({
      ...bedrockGuardrail("gr-bedrock-b", GUARD_B, KEY_B),
    });

    // Ready once a chat passes AND both guardrails have fired (both
    // run on the same input; mock answers NONE → allow).
    await waitConfigPropagation(async () => {
      try {
        const r = await chat(app!, "warmup");
        await r.text();
        if (r.status !== 200) return false;
        const ids = new Set(bedrock!.calls.map((c) => c.guardrailId));
        return ids.has(GUARD_A) && ids.has(GUARD_B);
      } catch {
        return false;
      }
    });
  });

  afterAll(async () => {
    await app?.exit();
    await bedrock?.close();
    await upstream?.close();
  });

  test(
    "each guardrail signs with exactly its own configured key",
    async (ctx) => {
      if (!etcdReachable || !bedrock) {
        ctx.skip();
        return;
      }
      // A few more requests so the assertion isn't over a single call.
      for (let i = 0; i < 3; i++) {
        const r = await chat(app!, `probe-${i}`);
        expect(r.status).toBe(200);
        await r.text();
      }
      const aCalls = bedrock.calls.filter((c) => c.guardrailId === GUARD_A);
      const bCalls = bedrock.calls.filter((c) => c.guardrailId === GUARD_B);
      expect(aCalls.length).toBeGreaterThan(0);
      expect(bCalls.length).toBeGreaterThan(0);
      for (const c of aCalls) {
        expect(c.accessKey).toBe(KEY_A);
        expect(c.region).toBe("us-east-1");
      }
      for (const c of bCalls) {
        expect(c.accessKey).toBe(KEY_B);
      }
    },
    60_000,
  );

  test(
    "rotating a guardrail's credentials re-signs with the new key",
    async (ctx) => {
      if (!etcdReachable || !bedrock || !guardrailAId) {
        ctx.skip();
        return;
      }
      await seed!.update(
        "guardrails",
        guardrailAId,
        bedrockGuardrail("gr-bedrock-a", GUARD_A, KEY_A2),
      );

      // Propagation is async — poll until guardrail A's LATEST call is
      // signed with the rotated key. A cached-forever client would
      // keep signing with KEY_A and time this out.
      await waitConfigPropagation(async () => {
        try {
          const r = await chat(app!, "post-rotate");
          await r.text();
          if (r.status !== 200) return false;
          const aCalls = bedrock!.calls.filter(
            (c) => c.guardrailId === GUARD_A,
          );
          return aCalls.length > 0 && aCalls[aCalls.length - 1].accessKey === KEY_A2;
        } catch {
          return false;
        }
      });

      // And once rotated, it stays rotated.
      const before = bedrock.calls.length;
      const r = await chat(app!, "post-rotate-confirm");
      expect(r.status).toBe(200);
      await r.text();
      const fresh = bedrock.calls
        .slice(before)
        .filter((c) => c.guardrailId === GUARD_A);
      expect(fresh.length).toBeGreaterThan(0);
      for (const c of fresh) {
        expect(c.accessKey).toBe(KEY_A2);
      }
    },
    60_000,
  );

  test(
    "ambient AWS_* process credentials never sign a guardrail call",
    async (ctx) => {
      if (!etcdReachable || !bedrock) {
        ctx.skip();
        return;
      }
      // Across EVERYTHING recorded in this suite — warmup, probes,
      // rotation traffic — the DP process's ambient key must be absent,
      // and so must its ambient region.
      expect(bedrock.calls.length).toBeGreaterThan(0);
      for (const c of bedrock.calls) {
        expect(c.accessKey).not.toBe(AMBIENT_KEY);
        expect(c.region).not.toBe("us-west-2");
      }
    },
    60_000,
  );
});
