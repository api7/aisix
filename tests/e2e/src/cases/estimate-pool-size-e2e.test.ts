import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { availableParallelism } from "node:os";
import { readdir, readFile } from "node:fs/promises";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import { keyHash, fileId, spawnApp, type SpawnedApp } from "../harness/index.js";

// E2E: `proxy.workers` sizes the gateway's token-estimate pool. Each pool
// thread holds its own tokenizer copies, so the worker count is what lets an
// operator bound that memory; the pool must not grow to the CPU count.
//
// The workers are set one above the CPUs this process may run on, which is
// an upper bound on what the gateway can see, so the two counts differ.

const CALLER_PLAINTEXT = "sk-estimate-pool-size";
const MODEL = "estimate-pool-size";
const WORKERS = availableParallelism() + 1;

let upstream: Server;
let app: SpawnedApp;

async function poolThreads(pid: number): Promise<number> {
  let count = 0;
  for (const task of await readdir(`/proc/${pid}/task`)) {
    const comm = await readFile(`/proc/${pid}/task/${task}/comm`, "utf8").catch(() => "");
    if (comm.startsWith("token-estimate")) count++;
  }
  return count;
}

beforeAll(async () => {
  // No usage in the response and more text than is counted inline, so the
  // gateway estimates it on the pool.
  upstream = createServer(async (req, res) => {
    for await (const _ of req) {
      // drain
    }
    res.setHeader("content-type", "application/json");
    res.end(
      JSON.stringify({
        id: "chatcmpl-estimate-pool-size",
        object: "chat.completion",
        created: 1,
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "word ".repeat(20_000) },
            finish_reason: "stop",
          },
        ],
      }),
    );
  });
  await new Promise<void>((r) => upstream.listen(0, "127.0.0.1", r));
  const port = (upstream.address() as AddressInfo).port;
  app = await spawnApp({
    resourcesFile: `
_format_version: "2"
provider_keys:
  - id: ${fileId(`provider_keys/estimate-pool-size-pk`)}
    display_name: estimate-pool-size-pk
    provider: openai
    api_key: sk-mock
    api_base: http://127.0.0.1:${port}/v1
models:
  - id: ${fileId(`models/${MODEL}`)}
    display_name: ${MODEL}
    provider: openai
    model_name: gpt-4o
    provider_key_id: ${fileId(`provider_keys/estimate-pool-size-pk`)}
api_keys:
  - id: ${fileId(`api_keys/estimate-pool-size-caller`)}
    display_name: estimate-pool-size-caller
    key_hash: ${keyHash(CALLER_PLAINTEXT)}
    allowed_models: ["${MODEL}"]
`,
    extraEnv: { AISIX_PROXY__WORKERS: String(WORKERS) },
  });
});

afterAll(async () => {
  await app?.exit();
  await new Promise<void>((r) => upstream?.close(() => r()));
});

describe.runIf(process.platform === "linux")("token-estimate pool size", () => {
  test("has one thread per configured proxy worker", async () => {
    expect(await poolThreads(app.pid)).toBe(0);

    const res = await fetch(`${app.proxyUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${CALLER_PLAINTEXT}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: "hi" }] }),
    });
    expect(res.status, await res.text()).toBe(200);

    // A pool thread names itself once it runs, so wait for every name.
    const deadline = Date.now() + 10_000;
    let threads = await poolThreads(app.pid);
    while (threads < WORKERS && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 50));
      threads = await poolThreads(app.pid);
    }
    expect(threads).toBe(WORKERS);
  });
});
