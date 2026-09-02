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
// The scope boundary is pinned here too, deliberately. The refusal is
// conditional on a guardrail being attached — this is NOT structural
// validation of the file, and an operator running no guardrails sees no
// behaviour change at all. The second leg is the one that fails if someone
// later widens this into an unconditional UTF-8 check on the files API.

const GUARDED_KEY = "sk-files-unscannable-guarded";
const UNGUARDED_KEY = "sk-files-unscannable-open";
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

/** A keyword row that cannot match these fixtures: the refusal must come
 *  from the blob being unscannable, not from a hit. A fix that simply
 *  started blocking every upload fails the third leg. */
const NEVER_MATCHING_GUARDRAIL = {
  name: "files-unscannable-e2e",
  enabled: true,
  hook_point: "input",
  fail_open: false,
  kind: "keyword",
  patterns: [{ kind: "literal", value: "zzz-no-such-term-zzz" }],
};

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

/** Hand-built multipart so the `file` part's bytes survive verbatim. */
function multipartUpload(boundary: string, model: string, file: Buffer): Buffer {
  return Buffer.concat([
    Buffer.from(
      `--${boundary}\r\ncontent-disposition: form-data; name="model"\r\n\r\n${model}\r\n` +
        `--${boundary}\r\ncontent-disposition: form-data; name="purpose"\r\n\r\nbatch\r\n` +
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

const startEnv = async (
  etcd: EtcdClient,
  key: string,
  guarded: boolean,
): Promise<Env> => {
  const upstream = await startFilesUpstream();
  const app = await spawnApp();
  const seed = new SeedClient(etcd, app.etcdPrefix);
  const model = guarded ? "files-guarded" : "files-open";

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
  if (guarded) {
    await seed.createGuardrail(NEVER_MATCHING_GUARDRAIL);
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

const upload = (env: Env, file: Buffer) => {
  const boundary = "XFILESUNSCANNABLEX";
  return fetch(`${env.app.proxyUrl}/v1/files`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.key}`,
      "content-type": `multipart/form-data; boundary=${boundary}`,
    },
    body: multipartUpload(boundary, env.model, file),
  });
};

describe("files: an unscannable upload is refused, not forwarded", () => {
  let guarded: Env | undefined;
  let open: Env | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;
    guarded = await startEnv(etcd, GUARDED_KEY, true);
    open = await startEnv(etcd, UNGUARDED_KEY, false);
  });

  afterAll(async () => {
    await guarded?.app.exit();
    await guarded?.upstream.close();
    await open?.app.exit();
    await open?.upstream.close();
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

  test("a clean UTF-8 blob still scans and forwards", async (ctx) => {
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
});
