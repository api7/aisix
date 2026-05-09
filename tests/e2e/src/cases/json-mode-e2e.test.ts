import { createHash } from "node:crypto";
import OpenAI from "openai";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  AdminClient,
  EtcdClient,
  spawnApp,
  startOpenAiUpstream,
  waitConfigPropagation,
  type OpenAiUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: structured outputs / JSON mode pass-through. OpenAI's
// `response_format` request field controls structured-output
// behavior on the upstream side. Two documented forms:
//
//   1. `{ type: "json_object" }` — model emits any valid JSON.
//   2. `{ type: "json_schema", json_schema: {...} }` — model emits
//      JSON conforming to the supplied JSON Schema.
//
// Per <https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format>
// and <https://platform.openai.com/docs/guides/structured-outputs>.
//
// User journey: caller sends OpenAI chat completion with
// `response_format` set; gateway must forward it verbatim to the
// upstream so the model honors the structured-output constraint.
// A regression that dropped `response_format` would leave the
// model in default unconstrained mode, breaking every caller that
// depends on parseable JSON output.
//
// Prior to this file the gateway had **zero** e2e coverage on
// `response_format` propagation.
//
// Two cases pinned:
//
//   1. `json_object` form — gateway forwards the simple
//      `{type: "json_object"}` field; caller asserts upstream's
//      JSON-shape body reaches the SDK as a parseable string.
//
//   2. `json_schema` form — gateway forwards the schema descriptor
//      (name, schema, strict) verbatim; caller asserts the same
//      JSON-shape body round-trips.
//
// References:
// - OpenAI structured-outputs guide
//   <https://platform.openai.com/docs/guides/structured-outputs>
// - OpenAI Chat Completions API spec, response_format param
//   <https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format>

const CALLER_PLAINTEXT = "sk-jsonmode-e2e-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

// JSON the upstream emits — pinned exactly so a regression that
// re-serialised or normalised the body would surface here.
const UPSTREAM_JSON_REPLY = JSON.stringify({
  city: "San Francisco",
  population: 815201,
  region: "California",
});

describe("json mode e2e: response_format passthrough on /v1/chat/completions", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream({
      nonStreamBody: {
        id: "cmpl-json-mode",
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: "gpt-4o-mini",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: UPSTREAM_JSON_REPLY,
            },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 8, completion_tokens: 16, total_tokens: 24 },
      },
    });
    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);

    const pk = await admin.createProviderKey({
      display_name: "jsonmode-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });
    await admin.createModel({
      display_name: "jsonmode-e2e",
      provider: "openai",
      model_name: "gpt-4o-mini",
      provider_key_id: pk.id,
    });
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["jsonmode-e2e"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test("response_format: { type: 'json_object' } forwarded verbatim", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    await waitConfigPropagation(async () => {
      try {
        const r = await client.chat.completions.create({
          model: "jsonmode-e2e",
          messages: [{ role: "user", content: "ready-probe" }],
        });
        return r.choices[0]?.message.role === "assistant";
      } catch {
        return false;
      }
    });

    const baseline = upstream.receivedRequests.length;
    const completion = await client.chat.completions.create({
      model: "jsonmode-e2e",
      messages: [
        { role: "user", content: "Return facts about SF as JSON." },
      ],
      response_format: { type: "json_object" },
    });

    // Caller-side: the JSON content reaches the SDK byte-for-byte
    // (it's just an assistant message whose content is a JSON
    // string). A regression that JSON.parse → JSON.stringify
    // round-tripped the content (re-keying or re-formatting
    // whitespace) would silently break callers that key off
    // exact-byte content (rare) or that compare hashes.
    expect(completion.choices[0]?.message.content).toBe(UPSTREAM_JSON_REPLY);
    // Sanity: the content actually parses as JSON (catches a
    // regression that prefixed/suffixed text to the body).
    const parsed = JSON.parse(
      completion.choices[0]!.message.content as string,
    ) as { city?: string };
    expect(parsed.city).toBe("San Francisco");

    // Upstream-side: response_format reached upstream verbatim. A
    // regression that dropped the field would leave the model in
    // default mode and produce non-JSON text instead.
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(testCalls).toHaveLength(1);
    const sentBody = JSON.parse(testCalls[0]!.body) as {
      response_format?: { type?: unknown };
    };
    expect(sentBody.response_format).toEqual({ type: "json_object" });
  });

  test("response_format: { type: 'json_schema', json_schema: {...} } forwarded verbatim", async (ctx) => {
    if (!etcdReachable || !app || !upstream) {
      ctx.skip();
      return;
    }

    const client = new OpenAI({
      apiKey: CALLER_PLAINTEXT,
      baseURL: `${app.proxyUrl}/v1`,
      maxRetries: 0,
    });

    // OpenAI's structured-outputs json_schema descriptor per
    // <https://platform.openai.com/docs/guides/structured-outputs>.
    const schema = {
      type: "object" as const,
      properties: {
        city: { type: "string" as const },
        population: { type: "integer" as const },
        region: { type: "string" as const },
      },
      required: ["city", "population", "region"] as const,
      additionalProperties: false as const,
    };
    const responseFormat = {
      type: "json_schema" as const,
      json_schema: {
        name: "city_facts",
        schema,
        strict: true,
      },
    };

    const baseline = upstream.receivedRequests.length;
    await client.chat.completions.create({
      model: "jsonmode-e2e",
      messages: [
        { role: "user", content: "Return facts about SF as JSON." },
      ],
      response_format: responseFormat,
    });

    // Upstream-side: the json_schema descriptor reaches upstream
    // verbatim — name, schema, strict all preserved. A regression
    // that dropped any field (especially `strict`) would relax the
    // upstream's constraints and break callers that rely on
    // schema-conforming output.
    const testCalls = upstream.receivedRequests
      .slice(baseline)
      .filter((r) => r.path === "/v1/chat/completions");
    expect(testCalls).toHaveLength(1);
    const sentBody = JSON.parse(testCalls[0]!.body) as {
      response_format?: {
        type?: unknown;
        json_schema?: { name?: unknown; schema?: unknown; strict?: unknown };
      };
    };
    expect(sentBody.response_format?.type).toBe("json_schema");
    expect(sentBody.response_format?.json_schema?.name).toBe("city_facts");
    expect(sentBody.response_format?.json_schema?.strict).toBe(true);
    // The full schema reaches upstream — a regression that
    // simplified or stripped sub-fields would break strict-mode
    // validation upstream.
    expect(sentBody.response_format?.json_schema?.schema).toEqual(schema);
  });
});
