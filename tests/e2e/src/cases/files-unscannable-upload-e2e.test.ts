import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: #1022 — `/v1/files` used to scan a LOSSY decode of the uploaded
// blob and then forward the ORIGINAL bytes. Every invalid sequence became
// U+FFFD before the guardrail ever saw it, so a term written in a
// non-UTF-8 encoding was invisible to the scan while the real bytes left
// the boundary anyway. With a chain attached, an undecodable upload is now
// refused with the same `unscannable_body` posture the LLM routes take.
//
// The upstream recorder is the load-bearing assertion in all three legs:
// the whole bug was that a "scanned" upload still reached the provider.
//
// The scope boundary is pinned here too, deliberately, on three axes. The
// refusal is conditional on a guardrail that READS the request being
// attached — this is NOT structural validation of the file; an operator
// running no guardrails, or running them on the output hook alone, sees no
// behaviour change at all. And it is conditional on the declared `purpose`
// naming a payload that is contractually UTF-8 text (`batch`, `fine-tune`,
// `evals`); an `assistants`/`vision`/`user_data` upload, or one that
// declares no purpose at all, legitimately carries binary and is forwarded
// as it always was. The later legs are what fail if someone widens this
// back into an unconditional UTF-8 check on the files API.

const GUARDED_KEY = "sk-files-unscannable-guarded";
const UNGUARDED_KEY = "sk-files-unscannable-open";
const OUTPUT_ONLY_KEY = "sk-files-unscannable-output";
const BOTH_HOOKS_KEY = "sk-files-unscannable-both";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

/** GBK bytes for 你好: `0xC4` opens a two-byte sequence and `0xE3` is not
 *  a continuation byte, so the blob is not valid UTF-8. */
const NON_UTF8_LINE = Buffer.concat([
  Buffer.from('{"custom_id":"r1","note":"', "utf8"),
  Buffer.from([0xc4, 0xe3, 0xba, 0xc3]),
  Buffer.from('"}\n', "utf8"),
]);

/** The same term, correctly encoded. */
const CLEAN_LINE = Buffer.from('{"custom_id":"r1","note":"你好"}\n', "utf8");

/** The input chain these tests attach. Its first pattern cannot match any
 *  fixture, so a refusal must come from the blob being unscannable rather
 *  than from a hit — a fix that simply started blocking every upload fails
 *  the forwarding legs. The second pattern appears in exactly one fixture,
 *  the one that proves a binary-purpose upload is still scanned. */
const guardrail = (hook: string) => ({
  name: `files-unscannable-e2e-${hook}`,
  enabled: true,
  hook_point: hook,
  fail_open: false,
  kind: "keyword",
  patterns: [
    { kind: "literal", value: "zzz-no-such-term-zzz" },
    // The one term these fixtures can hit, used by the final leg to prove
    // a binary-purpose upload is still scanned rather than skipped. Kept
    // out of every other fixture above.
    { kind: "literal", value: "zzz-blocked-term-zzz" },
  ],
});

/** The one term {@link GUARDRAIL} can actually hit. */
const BLOCKED_TERM = "zzz-blocked-term-zzz";

interface FilesUpstream {
  baseUrl: string;
  uploads: Buffer[];
  close(): Promise<void>;
}

/** Minimal `POST /v1/files` mock that keeps the RAW request bytes, so a
 *  test can assert byte-for-byte forwarding rather than a lossy re-read. */
async function startFilesUpstream(): Promise<FilesUpstream> {
  const uploads: Buffer[] = [];
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = (req.url ?? "/").split("?")[0];
      if (req.method === "POST" && path === "/v1/files") {
        uploads.push(Buffer.concat(chunks));
        res.statusCode = 200;
        res.setHeader("content-type", "application/json");
        return res.end(
          JSON.stringify({
            id: "file-e2e-in",
            object: "file",
            purpose: "batch",
            filename: "input.jsonl",
          }),
        );
      }
      res.statusCode = 404;
      res.end("{}");
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = (server.address() as { port: number }).port;
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    uploads,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

/** Hand-built multipart so the `file` part's bytes survive verbatim.
 *  `purpose` is omitted entirely when null. */
function multipartUpload(
  boundary: string,
  model: string,
  purpose: string | null,
  file: Buffer,
): Buffer {
  const purposePart =
    purpose === null
      ? ""
      : `--${boundary}\r\ncontent-disposition: form-data; name="purpose"\r\n\r\n${purpose}\r\n`;
  return Buffer.concat([
    Buffer.from(
      `--${boundary}\r\ncontent-disposition: form-data; name="model"\r\n\r\n${model}\r\n` +
        purposePart +
        `--${boundary}\r\ncontent-disposition: form-data; name="file"; filename="input.jsonl"\r\n` +
        `content-type: application/jsonl\r\n\r\n`,
      "utf8",
    ),
    file,
    Buffer.from(`\r\n--${boundary}--\r\n`, "utf8"),
  ]);
}

interface Env {
  app: SpawnedApp;
  upstream: FilesUpstream;
  key: string;
  model: string;
}

/** `hooks` names the `hook_point` of every guardrail row this environment
 *  attaches — `[]` for a deployment running none. The refusal keys on
 *  whether one of them READS the request, not on whether any is attached,
 *  so `["output"]` is a non-empty chain that must still forward. */
const startEnv = async (
  etcd: EtcdClient,
  key: string,
  model: string,
  hooks: string[],
): Promise<Env> => {
  const upstream = await startFilesUpstream();
  const app = await spawnApp();
  const seed = new SeedClient(etcd, app.etcdPrefix);

  const pk = await seed.createProviderKey({
    display_name: `${model}-pk`,
    secret: "sk-upstream-files",
    api_base: `${upstream.baseUrl}/v1`,
  });
  await seed.createModel({
    display_name: model,
    provider: "openai",
    model_name: "gpt-4o",
    provider_key_id: pk.id,
  });
  for (const hook of hooks) {
    await seed.createGuardrail(guardrail(hook));
  }
  // Written LAST: the key authenticating implies every row above it
  // landed (etcd applies in revision order).
  await seed.createApiKey({ key_hash: sha256(key), allowed_models: ["*"] });

  await waitConfigPropagation(async () => {
    const res = await fetch(`${app.proxyUrl}/v1/models`, {
      headers: { authorization: `Bearer ${key}` },
    });
    if (res.status !== 200) {
      // Drain it: an unread body leaves the socket held, and a failure
      // while reading should surface here rather than as a gate timeout.
      await res.text();
      return false;
    }
    const body = (await res.json()) as { data?: Array<{ id?: string }> };
    return (body.data ?? []).some((m) => m.id === model);
  });

  return { app, upstream, key, model };
};

const upload = (env: Env, file: Buffer, purpose: string | null = "batch") => {
  const boundary = "XFILESUNSCANNABLEX";
  return fetch(`${env.app.proxyUrl}/v1/files`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.key}`,
      "content-type": `multipart/form-data; boundary=${boundary}`,
    },
    body: multipartUpload(boundary, env.model, purpose, file),
  });
};

describe("files: an unscannable upload is refused, not forwarded", () => {
  let guarded: Env | undefined;
  let open: Env | undefined;
  let outputOnly: Env | undefined;
  let bothHooks: Env | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    guarded = await startEnv(etcd, GUARDED_KEY, "files-guarded", ["input"]);
    open = await startEnv(etcd, UNGUARDED_KEY, "files-open", []);
    outputOnly = await startEnv(etcd, OUTPUT_ONLY_KEY, "files-output", [
      "output",
    ]);
    bothHooks = await startEnv(etcd, BOTH_HOOKS_KEY, "files-both", [
      "input",
      "output",
    ]);
  });

  afterAll(async () => {
    for (const env of [guarded, open, outputOnly, bothHooks]) {
      await env?.app.exit();
      await env?.upstream.close();
    }
  });

  test("non-UTF-8 blob with a guardrail attached: refused, provider never contacted", async (ctx) => {
    if (!etcdReachable || !guarded) {
      ctx.skip();
      return;
    }
    const res = await upload(guarded, NON_UTF8_LINE);
    expect(res.status).toBe(422);
    const body = (await res.json()) as {
      error?: { type?: string; code?: string; message?: string };
    };
    expect(body.error?.type).toBe("content_filter");
    expect(body.error?.code).toBe("guardrail_unavailable");
    // The caller must be able to tell an unscannable body from a policy
    // hit — they are the same status and type.
    expect(body.error?.message).toContain("unscannable_body");
    expect(guarded.upstream.uploads).toHaveLength(0);
  });

  // The gate is "a guardrail would READ this upload", not "a guardrail is
  // attached". A row on the output hook alone resolves into the chain —
  // `GuardrailIndex::resolve` matches on scope, not hook — but it is never
  // offered the request, so it cannot be the reason the request is refused.
  test("a guardrail on the OUTPUT hook alone does not refuse the upload", async (ctx) => {
    if (!etcdReachable || !outputOnly) {
      ctx.skip();
      return;
    }
    const res = await upload(outputOnly, NON_UTF8_LINE);
    expect(res.status).toBe(200);
    expect(outputOnly.upstream.uploads).toHaveLength(1);
    expect(outputOnly.upstream.uploads[0].includes(NON_UTF8_LINE)).toBe(true);
  });

  // ...and one input-hook row is enough: adding the output-only row must
  // not soften the refusal into a forward.
  test("input + output hooks attached: still refused", async (ctx) => {
    if (!etcdReachable || !bothHooks) {
      ctx.skip();
      return;
    }
    const res = await upload(bothHooks, NON_UTF8_LINE);
    expect(res.status).toBe(422);
    const body = (await res.json()) as {
      error?: { type?: string; code?: string; message?: string };
    };
    expect(body.error?.type).toBe("content_filter");
    expect(body.error?.code).toBe("guardrail_unavailable");
    expect(body.error?.message).toContain("unscannable_body");
    expect(bothHooks.upstream.uploads).toHaveLength(0);
  });

  test("the same blob with NO guardrail attached is unaffected", async (ctx) => {
    if (!etcdReachable || !open) {
      ctx.skip();
      return;
    }
    const res = await upload(open, NON_UTF8_LINE);
    expect(res.status).toBe(200);
    expect(open.upstream.uploads).toHaveLength(1);
    expect(open.upstream.uploads[0].includes(NON_UTF8_LINE)).toBe(true);
  });

  test("a decodable blob with a guardrail attached is still forwarded", async (ctx) => {
    if (!etcdReachable || !guarded) {
      ctx.skip();
      return;
    }
    // Counted rather than asserted at length 1: this env is shared with
    // the first leg, whose whole point is that it forwarded nothing.
    const before = guarded.upstream.uploads.length;
    const res = await upload(guarded, CLEAN_LINE);
    expect(res.status).toBe(200);
    expect(guarded.upstream.uploads).toHaveLength(before + 1);
    expect(guarded.upstream.uploads[before].includes(CLEAN_LINE)).toBe(true);
  });

  // A binary upload under a purpose whose payload is not contractually
  // text is the case the first version of this refusal broke: a PDF or an
  // image started returning 422 for every deployment running any input
  // guardrail.
  for (const purpose of ["assistants", "vision", "user_data"]) {
    test(`a non-UTF-8 blob under purpose=${purpose} is forwarded, not refused`, async (ctx) => {
      if (!etcdReachable || !guarded) {
        ctx.skip();
        return;
      }
      const before = guarded.upstream.uploads.length;
      const res = await upload(guarded, NON_UTF8_LINE, purpose);
      expect(res.status).toBe(200);
      expect(guarded.upstream.uploads).toHaveLength(before + 1);
      // Byte-for-byte: the refusal was narrowed, the forwarding was not.
      expect(guarded.upstream.uploads[before].includes(NON_UTF8_LINE)).toBe(
        true,
      );
    });
  }

  test("a non-UTF-8 blob with no declared purpose is forwarded", async (ctx) => {
    if (!etcdReachable || !guarded) {
      ctx.skip();
      return;
    }
    // An upload we cannot classify is not one we can claim should have
    // been text.
    const before = guarded.upstream.uploads.length;
    const res = await upload(guarded, NON_UTF8_LINE, null);
    expect(res.status).toBe(200);
    expect(guarded.upstream.uploads).toHaveLength(before + 1);
  });

  test("purpose=fine-tune is refused on the same terms as batch", async (ctx) => {
    if (!etcdReachable || !guarded) {
      ctx.skip();
      return;
    }
    const before = guarded.upstream.uploads.length;
    const res = await upload(guarded, NON_UTF8_LINE, "fine-tune");
    expect(res.status).toBe(422);
    const body = (await res.json()) as { error?: { message?: string } };
    expect(body.error?.message).toContain("unscannable_body");
    expect(guarded.upstream.uploads).toHaveLength(before);
  });

  test("a binary purpose is still SCANNED, just not refused", async (ctx) => {
    if (!etcdReachable || !guarded) {
      ctx.skip();
      return;
    }
    // The chain still runs on a lossy decode, so a term that survives it
    // still blocks. This is the leg that fails if someone "fixes" the
    // regression by skipping the chain for non-text purposes instead of
    // narrowing the refusal.
    //
    // The term sits AFTER the invalid bytes on purpose: scanning only the
    // valid UTF-8 prefix would miss it, and that is the shape a real
    // evasion takes — a few bad bytes up front, the payload behind them.
    const before = guarded.upstream.uploads.length;
    const res = await upload(
      guarded,
      Buffer.concat([
        Buffer.from('{"x":"', "utf8"),
        Buffer.from([0xc4, 0xe3, 0xba, 0xc3]),
        Buffer.from(`","note":"${BLOCKED_TERM}"}\n`, "utf8"),
      ]),
      "assistants",
    );
    expect(res.status).toBe(422);
    const body = (await res.json()) as {
      error?: { type?: string; message?: string };
    };
    expect(body.error?.type).toBe("content_filter");
    // A policy hit, not the unscannable-body refusal.
    expect(body.error?.message).not.toContain("unscannable_body");
    expect(guarded.upstream.uploads).toHaveLength(before);
  });
});
