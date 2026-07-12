import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import OpenAI, { toFile } from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: first-class /v1/files + /v1/batches + /v1/fine_tuning/jobs (#720,
// AISIX-Cloud#873 §⑤). Pins the LiteLLM-baseline routing mechanism end to
// end against a real `aisix` binary:
//
//   1. A file uploaded with a routing hint (x-aisix-model header — the
//      OpenAI SDK path; multipart `model` field is the raw-HTTP path)
//      returns a GATEWAY id (`aisix-…`) embedding the Model.
//   2. Batch create referencing that id decodes it, lands on the right
//      provider with the RAW file id, and re-encodes every id in the
//      response.
//   3. Batch retrieve of a completed batch triggers the detached usage
//      attribution: the gateway itself downloads the output JSONL from
//      the provider (observable as a mock-side hit on
//      `/v1/files/<output>/content`).
//   4. Fine-tuning jobs route via the encoded `training_file` while the
//      body's `model` (the provider BASE model) forwards verbatim.
//
// All calls go through the official OpenAI Node SDK (files/batches/
// fineTuning namespaces) — the point of the surface is drop-in SDK
// compatibility.

const CALLER_PLAINTEXT = "sk-jobs-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

interface JobsUpstream {
  baseUrl: string;
  received: { method: string; path: string; body: string; headers: Record<string, string> }[];
  close(): Promise<void>;
}

/** Minimal OpenAI jobs-surface mock: files upload/retrieve/content,
 * batches create/retrieve/cancel, fine_tuning create/cancel. */
async function startJobsUpstream(): Promise<JobsUpstream> {
  const received: JobsUpstream["received"] = [];
  const server: Server = createServer((req, res) => {
    res.on("error", () => {});
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      const body = Buffer.concat(chunks).toString("utf8");
      const path = (req.url ?? "/").split("?")[0];
      received.push({
        method: req.method ?? "GET",
        path,
        body,
        headers: Object.fromEntries(
          Object.entries(req.headers).map(([k, v]) => [
            k,
            Array.isArray(v) ? v.join(",") : (v ?? ""),
          ]),
        ),
      });

      const json = (status: number, payload: unknown) => {
        res.statusCode = status;
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify(payload));
      };

      if (req.method === "POST" && path === "/v1/files") {
        return json(200, {
          id: "file-e2e-in",
          object: "file",
          purpose: "batch",
          filename: "input.jsonl",
          bytes: body.length,
        });
      }
      if (req.method === "GET" && path === "/v1/files/file-e2e-out/content") {
        res.statusCode = 200;
        res.setHeader("content-type", "application/jsonl");
        return res.end(
          [
            JSON.stringify({
              id: "batch_req_1",
              custom_id: "r1",
              response: {
                status_code: 200,
                body: {
                  model: "gpt-4o-2024-08-06",
                  usage: { prompt_tokens: 11, completion_tokens: 4 },
                },
              },
            }),
            JSON.stringify({
              id: "batch_req_2",
              custom_id: "r2",
              response: {
                status_code: 200,
                body: {
                  model: "gpt-4o-2024-08-06",
                  usage: { prompt_tokens: 6, completion_tokens: 2 },
                },
              },
            }),
          ].join("\n") + "\n",
        );
      }
      if (req.method === "GET" && path.startsWith("/v1/files/")) {
        const id = path.split("/")[3];
        return json(200, { id, object: "file", purpose: "batch", filename: "input.jsonl" });
      }
      if (req.method === "POST" && path === "/v1/batches") {
        return json(200, {
          id: "batch_e2e_1",
          object: "batch",
          status: "validating",
          endpoint: "/v1/chat/completions",
          input_file_id: JSON.parse(body).input_file_id,
        });
      }
      if (req.method === "GET" && path === "/v1/batches/batch_e2e_1") {
        return json(200, {
          id: "batch_e2e_1",
          object: "batch",
          status: "completed",
          endpoint: "/v1/chat/completions",
          input_file_id: "file-e2e-in",
          output_file_id: "file-e2e-out",
        });
      }
      if (req.method === "POST" && path === "/v1/batches/batch_e2e_1/cancel") {
        return json(200, { id: "batch_e2e_1", object: "batch", status: "cancelling" });
      }
      if (req.method === "POST" && path === "/v1/fine_tuning/jobs") {
        const parsed = JSON.parse(body);
        return json(200, {
          id: "ftjob-e2e-1",
          object: "fine_tuning.job",
          model: parsed.model,
          training_file: parsed.training_file,
          status: "validating_files",
        });
      }
      if (req.method === "POST" && path === "/v1/fine_tuning/jobs/ftjob-e2e-1/cancel") {
        return json(200, {
          id: "ftjob-e2e-1",
          object: "fine_tuning.job",
          status: "cancelled",
        });
      }
      return json(404, { error: { message: `no mock for ${req.method} ${path}` } });
    });
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    baseUrl: `http://127.0.0.1:${addr.port}`,
    received,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((e) => (e ? reject(e) : resolve())),
      ),
  };
}

describe("jobs e2e: /v1/files + /v1/batches + /v1/fine_tuning/jobs (#720)", () => {
  let app: SpawnedApp | undefined;
  let seed: SeedClient | undefined;
  let upstream: JobsUpstream | undefined;
  let etcdReachable = false;
  let client: OpenAI;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    seed = new SeedClient(etcd, app.etcdPrefix);
    upstream = await startJobsUpstream();

    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
    const pk = await seed.createProviderKey({
      display_name: "jobs-e2e-pk",
      secret: "sk-upstream-jobs",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await seed.createModel({
      display_name: "jobs-e2e-model",
      provider: "openai",
      model_name: "gpt-4o",
      provider_key_id: pk.id,
    });

    // Gate on the DP snapshot, not the store: /v1/models only lists the
    // model once the snapshot has it, and only authenticates once the
    // caller key has propagated too. Touches no upstream.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) return false;
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === "jobs-e2e-model");
    });

    client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("full batch lifecycle through the OpenAI SDK: upload → create → retrieve → content → cancel", async (ctx) => {
    if (!etcdReachable || !app || !seed || !upstream) {
      ctx.skip();
      return;
    }

    // 1. Upload with the routing hint on a header (SDK-clean path).
    const file = await client.files.create(
      {
        file: await toFile(
          Buffer.from('{"custom_id":"r1","method":"POST","url":"/v1/chat/completions"}\n'),
          "input.jsonl",
        ),
        purpose: "batch",
      },
      { headers: { "x-aisix-model": "jobs-e2e-model" } },
    );
    expect(file.id.startsWith("aisix-")).toBe(true);
    const uploadReq = upstream.received.find(
      (r) => r.method === "POST" && r.path === "/v1/files",
    );
    expect(uploadReq).toBeDefined();
    // Upstream auth is the provider secret, not the caller key.
    expect(uploadReq?.headers.authorization).toBe("Bearer sk-upstream-jobs");
    expect(uploadReq?.body).toContain('name="purpose"');
    expect(uploadReq?.body).not.toContain('name="model"');

    // 2. Batch create referencing the ENCODED id — no model hint needed.
    const batch = await client.batches.create({
      input_file_id: file.id,
      endpoint: "/v1/chat/completions",
      completion_window: "24h",
    });
    expect(batch.id.startsWith("aisix-")).toBe(true);
    const createReq = upstream.received.find(
      (r) => r.method === "POST" && r.path === "/v1/batches",
    );
    expect(createReq).toBeDefined();
    const createdBody = JSON.parse(createReq?.body ?? "{}");
    expect(createdBody.input_file_id).toBe("file-e2e-in");
    expect(createdBody.model).toBeUndefined();

    // 3. Retrieve → completed. The gateway must (a) re-encode the output
    // file id for the caller and (b) fire the detached usage-attribution
    // download of the output JSONL.
    const got = await client.batches.retrieve(batch.id);
    expect(got.status).toBe("completed");
    expect(got.output_file_id?.startsWith("aisix-")).toBe(true);

    await expect
      .poll(
        () =>
          upstream!.received.some(
            (r) => r.method === "GET" && r.path === "/v1/files/file-e2e-out/content",
          ),
        { timeout: 5000 },
      )
      .toBe(true);

    // 4. Output file content downloads through the gateway via the
    // encoded id from the retrieve response.
    const content = await client.files.content(got.output_file_id!);
    const text = await content.text();
    expect(text).toContain('"custom_id":"r1"');

    // 5. Cancel round-trips on the encoded id.
    const cancelled = await client.batches.cancel(batch.id);
    expect(cancelled.status).toBe("cancelling");
    expect(
      upstream.received.some(
        (r) => r.method === "POST" && r.path === "/v1/batches/batch_e2e_1/cancel",
      ),
    ).toBe(true);
  });

  test("fine-tuning: encoded training_file routes the job, provider base model forwards verbatim", async (ctx) => {
    if (!etcdReachable || !app || !seed || !upstream) {
      ctx.skip();
      return;
    }

    const file = await client.files.create(
      {
        file: await toFile(
          Buffer.from('{"messages":[{"role":"user","content":"hi"}]}\n'),
          "train.jsonl",
        ),
        purpose: "fine-tune",
      },
      { headers: { "x-aisix-model": "jobs-e2e-model" } },
    );

    const job = await client.fineTuning.jobs.create({
      model: "gpt-4o-mini-2024-07-18",
      training_file: file.id,
    });
    expect(job.id.startsWith("aisix-")).toBe(true);

    const createReq = upstream.received.find(
      (r) => r.method === "POST" && r.path === "/v1/fine_tuning/jobs",
    );
    expect(createReq).toBeDefined();
    const sent = JSON.parse(createReq?.body ?? "{}");
    expect(sent.training_file).toBe("file-e2e-in");
    expect(sent.model).toBe("gpt-4o-mini-2024-07-18");

    const cancelled = await client.fineTuning.jobs.cancel(job.id);
    expect(cancelled.status).toBe("cancelled");
  });

  test("raw multipart `model` form field routes the upload (non-SDK path)", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const form = new FormData();
    form.set("purpose", "batch");
    form.set("model", "jobs-e2e-model");
    form.set(
      "file",
      new Blob(['{"custom_id":"raw"}\n'], { type: "application/jsonl" }),
      "raw.jsonl",
    );
    const resp = await fetch(`${app.proxyUrl}/v1/files`, {
      method: "POST",
      headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      body: form,
    });
    expect(resp.status).toBe(200);
    const body = (await resp.json()) as { id: string };
    expect(body.id.startsWith("aisix-")).toBe(true);
  });

  test("auth is mandatory on the jobs surface", async (ctx) => {
    if (!etcdReachable || !app) {
      ctx.skip();
      return;
    }
    const resp = await fetch(`${app.proxyUrl}/v1/batches`, {
      method: "GET",
    });
    expect(resp.status).toBe(401);
  });
});
