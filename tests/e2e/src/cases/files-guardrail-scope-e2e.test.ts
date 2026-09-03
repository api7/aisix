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

// E2E: guardrails do not apply to the files surface (#1120). An uploaded
// file is a batch of independent requests, not one message, so scanning
// the whole blob as a single synthetic turn answered with one verdict over
// decoded JSON syntax and could rewrite nothing. Screening it per record,
// through the ordinary request chain, is #1120's job.
//
// Every leg here attaches a guardrail whose pattern IS present in the
// fixture, on both hooks and fail-closed, so a scan anywhere on the path
// would refuse. The upstream recorder is the load-bearing half of the
// upload legs: the bytes must reach the provider unchanged.
//
// The last leg is the scope boundary. `/v1/batches` sends a serialised
// JSON request body rather than a caller's blob, and is still screened —
// it fails if the removal was widened past the files surface.

const KEY = "sk-files-guardrail-scope";
const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

/** The term the attached row blocks on. It is in every fixture below. */
const BLOCKED_TERM = "zzz-blocked-term-zzz";

/** A blob that is both a policy hit and undecodable: `0xC4` opens a
 *  two-byte sequence and `0xE3` is not a continuation byte, so it is not
 *  valid UTF-8. Both are ways a whole-blob scan could stop the upload. */
const BLOB = Buffer.concat([
  Buffer.from(`{"custom_id":"r1","note":"${BLOCKED_TERM}","raw":"`, "utf8"),
  Buffer.from([0xc4, 0xe3, 0xba, 0xc3]),
  Buffer.from('"}\n', "utf8"),
]);

/** What a download returns, so an output-hook row has a match in it. */
const STORED = Buffer.from(
  `{"custom_id":"r1","response":"${BLOCKED_TERM}"}\n`,
  "utf8",
);

const guardrail = {
  name: "files-guardrail-scope-e2e",
  enabled: true,
  hook_point: "both",
  fail_open: false,
  kind: "keyword",
  patterns: [
    { kind: "literal", value: BLOCKED_TERM },
    // Present in the outbound `/v1/batches` body, which is what the last
    // leg refuses on.
    { kind: "literal", value: "completion_window" },
  ],
};

interface JobsUpstream {
  baseUrl: string;
  uploads: Buffer[];
  batches: Buffer[];
  close(): Promise<void>;
}

/** Keeps the RAW upload bytes so a test can assert byte-for-byte
 *  forwarding, and echoes the uploaded `filename` so an output-hook row
 *  has something under the caller's control to match on. */
async function startJobsUpstream(): Promise<JobsUpstream> {
  const uploads: Buffer[] = [];
  const batches: Buffer[] = [];
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const path = (req.url ?? "/").split("?")[0];
      const raw = Buffer.concat(chunks);
      if (req.method === "POST" && path === "/v1/files") {
        uploads.push(raw);
        const echoed =
          /filename="([^"]*)"/.exec(raw.toString("latin1"))?.[1] ??
          "input.jsonl";
        res.statusCode = 200;
        res.setHeader("content-type", "application/json");
        return res.end(
          JSON.stringify({
            id: "file-e2e-in",
            object: "file",
            purpose: "batch",
            filename: echoed,
          }),
        );
      }
      if (req.method === "GET" && path === "/v1/files/file-e2e-in/content") {
        res.statusCode = 200;
        res.setHeader("content-type", "application/octet-stream");
        return res.end(STORED);
      }
      if (req.method === "POST" && path === "/v1/batches") {
        batches.push(raw);
        res.statusCode = 200;
        res.setHeader("content-type", "application/json");
        return res.end(
          JSON.stringify({
            id: "batch-e2e",
            object: "batch",
            status: "validating",
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
    batches,
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
  filename: string,
): Buffer {
  const purposePart =
    purpose === null
      ? ""
      : `--${boundary}\r\ncontent-disposition: form-data; name="purpose"\r\n\r\n${purpose}\r\n`;
  return Buffer.concat([
    Buffer.from(
      `--${boundary}\r\ncontent-disposition: form-data; name="model"\r\n\r\n${model}\r\n` +
        purposePart +
        `--${boundary}\r\ncontent-disposition: form-data; name="file"; filename="${filename}"\r\n` +
        `content-type: application/jsonl\r\n\r\n`,
      "utf8",
    ),
    file,
    Buffer.from(`\r\n--${boundary}--\r\n`, "utf8"),
  ]);
}

const MODEL = "files-guardrail-scope";

let app: SpawnedApp | undefined;
let upstream: JobsUpstream | undefined;
let etcdReachable = false;

const upload = (
  purpose: string | null,
  file: Buffer = BLOB,
  filename = "input.jsonl",
) =>
  fetch(`${app!.proxyUrl}/v1/files`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${KEY}`,
      "content-type": "multipart/form-data; boundary=XFILESSCOPEX",
    },
    body: multipartUpload("XFILESSCOPEX", MODEL, purpose, file, filename),
  });

describe("files: the surface is not guardrail-screened", () => {
  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startJobsUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    const pk = await seed.createProviderKey({
      display_name: `${MODEL}-pk`,
      secret: "sk-upstream-files",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: MODEL,
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });
    await seed.createGuardrail(guardrail);
    // Written LAST: the key authenticating implies every row above it
    // landed (etcd applies in revision order).
    await seed.createApiKey({ key_hash: sha256(KEY), allowed_models: ["*"] });

    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${KEY}` },
      });
      if (res.status !== 200) {
        await res.text();
        return false;
      }
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === MODEL);
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  // Every class of declared `purpose`: contractually JSONL, legitimately
  // binary, and none at all.
  for (const purpose of ["batch", "assistants", null]) {
    test(`an upload with purpose=${purpose ?? "(none)"} is forwarded`, async (ctx) => {
      if (!etcdReachable) {
        ctx.skip();
        return;
      }
      const before = upstream!.uploads.length;
      const res = await upload(purpose);
      expect(res.status).toBe(200);
      await res.json();
      expect(upstream!.uploads).toHaveLength(before + 1);
      expect(upstream!.uploads[before].includes(BLOB)).toBe(true);
    });
  }

  test("the upload response is relayed even when it matches", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    // The upstream echoes the filename, so the response body carries the
    // blocked term on the way back.
    const res = await upload("batch", BLOB, `${BLOCKED_TERM}.jsonl`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { filename?: string };
    expect(body.filename).toBe(`${BLOCKED_TERM}.jsonl`);
  });

  test("a download relays the stored bytes verbatim", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const uploaded = (await (await upload("batch")).json()) as { id: string };
    const res = await fetch(
      `${app!.proxyUrl}/v1/files/${encodeURIComponent(uploaded.id)}/content`,
      { headers: { authorization: `Bearer ${KEY}` } },
    );
    expect(res.status).toBe(200);
    expect(Buffer.from(await res.arrayBuffer()).equals(STORED)).toBe(true);
  });

  // The scope boundary: `/v1/batches` carries a serialised JSON request
  // body, not a caller's blob, and the input chain still runs over it.
  test("a batch create is still screened", async (ctx) => {
    if (!etcdReachable) {
      ctx.skip();
      return;
    }
    const uploaded = (await (await upload("batch")).json()) as { id: string };
    const before = upstream!.batches.length;
    const res = await fetch(`${app!.proxyUrl}/v1/batches`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${KEY}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        input_file_id: uploaded.id,
        endpoint: "/v1/chat/completions",
        completion_window: "24h",
      }),
    });
    expect(res.status).toBe(422);
    const body = (await res.json()) as { error?: { type?: string } };
    expect(body.error?.type).toBe("content_filter");
    expect(upstream!.batches).toHaveLength(before);
  });
});
