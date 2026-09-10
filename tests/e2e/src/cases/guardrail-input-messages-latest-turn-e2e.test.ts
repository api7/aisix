import { createHash } from "node:crypto";
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

// E2E: `input_messages: latest_turn` narrows an input guardrail to the part
// of the conversation the model has not answered yet (AISIX-Cloud#1558).
//
// The reported problem: IDE and agent clients replay the whole conversation
// on every call, so a rule that matched one message keeps refusing every
// later request in the session even though the new prompt is clean. The
// window is "everything after the last assistant message, system excluded" —
// this turn's user message plus the tool results answering it.
//
// Two boundaries are pinned per protocol, deliberately, because the gateway
// answers the question twice and in two representations:
//
//   * the CHECK pass reads the parsed ChatFormat, so a `keyword` row
//     exercises `aisix_guardrails::latest_turn_view`;
//   * the SEGMENT pass rewrites raw wire slots and never sees a parsed
//     view, so a `custom` row — which moderates via the segment hooks —
//     exercises the per-shape walkers in `aisix-proxy::redact`.
//
// A `custom` row runs its script in-process, so the segment half needs no
// external service. The two must agree; a drift between them is exactly
// what these cases exist to catch.
//
// Reference:
//   - <https://platform.openai.com/docs/api-reference/chat/create>
//   - <https://docs.anthropic.com/en/api/messages>
//   - <https://platform.openai.com/docs/api-reference/responses/create>

const CALLER = "sk-latest-turn-e2e-caller";
const HASH = createHash("sha256").update(CALLER).digest("hex");

/** The pattern every guardrail below refuses. */
const MARKER = "latestturnforbiddenmarker";
/** Masked rather than blocked, by the mask row only. */
const SECRET = "latestturnsecretvalue";
const MASKED = "[REDACTED]";

/** Blocks whenever the text it is given contains the marker. */
const BLOCK_SCRIPT = `
export async function checkInput(ctx) {
  if (ctx.text.indexOf("${MARKER}") !== -1) {
    return { action: "block", reason_code: "LT-1" };
  }
  return { action: "none" };
}
`;

/** Rewrites the secret out of every slot it is offered. */
const MASK_SCRIPT = `
export async function checkInput(ctx) {
  return {
    action: "mask",
    segments: ctx.segments.map(function (s) {
      return s.split("${SECRET}").join("${MASKED}");
    }),
  };
}
`;

interface Lane {
  /** Model display name, also the model a request addresses. */
  model: string;
}

describe("guardrail input_messages: latest_turn (AISIX-Cloud#1558)", () => {
  let app: SpawnedApp | undefined;
  let upstream: OpenAiUpstream | undefined;
  let etcdReachable = false;

  // One model per (guardrail kind × window) so the rows never share scope:
  // a guardrail's scope is its attachments and nothing else, so attaching
  // each to its own model keeps the lanes independent on one gateway.
  const kwLatest: Lane = { model: "lt-kw-latest" };
  const kwAll: Lane = { model: "lt-kw-all" };
  const scriptLatest: Lane = { model: "lt-script-latest" };
  const scriptAll: Lane = { model: "lt-script-all" };
  const maskLatest: Lane = { model: "lt-mask-latest" };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    upstream = await startOpenAiUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    const pk = await seed.createProviderKey({
      display_name: "lt-pk",
      secret: "sk-mock",
      api_base: `${upstream.baseUrl}/v1`,
    });

    const lane = async (
      lane: Lane,
      guardrail: Record<string, unknown>,
    ): Promise<void> => {
      const model = await seed.createModel({
        display_name: lane.model,
        provider: "openai",
        model_name: "gpt-4o-mini",
        provider_key_id: pk.id,
      });
      const gr = await seed.createGuardrail(guardrail, { attach: false });
      await seed.attachGuardrailToModel(gr.id, model.id);
    };

    await lane(kwLatest, {
      name: "lt-kw-latest-row",
      enabled: true,
      hook_point: "input",
      input_messages: "latest_turn",
      kind: "keyword",
      patterns: [{ kind: "literal", value: MARKER }],
    });
    await lane(kwAll, {
      name: "lt-kw-all-row",
      enabled: true,
      hook_point: "input",
      // `input_messages` deliberately unset: the default must stay `all`,
      // so this lane also pins that no existing configuration changed.
      kind: "keyword",
      patterns: [{ kind: "literal", value: MARKER }],
    });
    await lane(scriptLatest, {
      name: "lt-script-latest-row",
      enabled: true,
      hook_point: "input",
      input_messages: "latest_turn",
      fail_open: false,
      kind: "custom",
      script: BLOCK_SCRIPT,
      timeout_ms: 5000,
    });
    await lane(scriptAll, {
      name: "lt-script-all-row",
      enabled: true,
      hook_point: "input",
      fail_open: false,
      kind: "custom",
      script: BLOCK_SCRIPT,
      timeout_ms: 5000,
    });
    await lane(maskLatest, {
      name: "lt-mask-latest-row",
      enabled: true,
      hook_point: "input",
      input_messages: "latest_turn",
      fail_open: false,
      kind: "custom",
      script: MASK_SCRIPT,
      timeout_ms: 5000,
    });

    // Seeded last: this key authenticating implies every resource above it
    // is already in the gateway's snapshot.
    await seed.createApiKey({
      key_hash: HASH,
      allowed_models: [
        kwLatest.model,
        kwAll.model,
        scriptLatest.model,
        scriptAll.model,
        maskLatest.model,
      ],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  // ── request builders, one per protocol ────────────────────────────────

  const post = (path: string, body: unknown): Promise<Response> =>
    fetch(`${app!.proxyUrl}${path}`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${CALLER}`,
        "x-api-key": CALLER,
      },
      body: JSON.stringify(body),
    });

  const chat = (model: string, messages: unknown[]) =>
    post("/v1/chat/completions", { model, messages });

  const anthropic = (
    model: string,
    messages: unknown[],
    system?: string,
  ) =>
    post("/v1/messages", {
      model,
      max_tokens: 64,
      ...(system === undefined ? {} : { system }),
      messages,
    });

  const responses = (model: string, input: unknown[], instructions?: string) =>
    post("/v1/responses", {
      model,
      ...(instructions === undefined ? {} : { instructions }),
      input,
    });

  /** 422 is the guardrail refusal; anything else is "not blocked". */
  const blocked = async (r: Promise<Response>): Promise<boolean> =>
    (await r).status === 422;

  /**
   * Gate on the lane's guardrail being loaded AND in force, using the one
   * shape every window agrees on: the marker in the CURRENT user message.
   * A positive condition, so a transient 401 while the caller key is still
   * propagating cannot be mistaken for a guardrail verdict.
   */
  async function waitForLane(model: string): Promise<void> {
    await waitConfigPropagation(async () => {
      if (await blocked(chat(model, [{ role: "user", content: "probe" }]))) {
        return false;
      }
      return blocked(
        chat(model, [{ role: "user", content: `probe ${MARKER}` }]),
      );
    });
  }

  // ── the four boundary cases, per protocol × per pass ──────────────────
  //
  // Each builder returns a body whose marker sits in exactly one place.

  const chatBodies = {
    // Replayed history the model has already answered.
    history: () => [
      { role: "system", content: "be helpful" },
      { role: "user", content: `earlier ${MARKER}` },
      { role: "assistant", content: "understood" },
      { role: "user", content: "a clean new question" },
    ],
    // This turn's user message.
    latest: () => [
      { role: "user", content: "earlier" },
      { role: "assistant", content: "understood" },
      { role: "user", content: `now do ${MARKER}` },
    ],
    // This turn's tool result: it follows the assistant turn that asked
    // for it, so it is inside the window.
    toolResult: () => [
      { role: "user", content: "look it up" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "lookup", arguments: "{}" },
          },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: `found ${MARKER}` },
    ],
    system: () => [
      { role: "system", content: `policy mentions ${MARKER}` },
      { role: "user", content: "earlier" },
      { role: "assistant", content: "understood" },
      { role: "user", content: "a clean new question" },
    ],
  };

  const anthropicBodies = {
    history: (): [unknown[], string | undefined] => [
      [
        { role: "user", content: `earlier ${MARKER}` },
        { role: "assistant", content: "understood" },
        { role: "user", content: "a clean new question" },
      ],
      "be helpful",
    ],
    latest: (): [unknown[], string | undefined] => [
      [
        { role: "user", content: "earlier" },
        { role: "assistant", content: "understood" },
        { role: "user", content: `now do ${MARKER}` },
      ],
      undefined,
    ],
    // A tool_use turn is an assistant message, so it is the boundary; the
    // tool_result answering it rides the FINAL user message.
    toolResult: (): [unknown[], string | undefined] => [
      [
        { role: "user", content: "look it up" },
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "tu_1", name: "lookup", input: {} },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "tu_1",
              content: `found ${MARKER}`,
            },
          ],
        },
      ],
      undefined,
    ],
    system: (): [unknown[], string | undefined] => [
      [
        { role: "user", content: "earlier" },
        { role: "assistant", content: "understood" },
        { role: "user", content: "a clean new question" },
      ],
      `policy mentions ${MARKER}`,
    ],
  };

  const assistantMessageItem = (text: string) => ({
    type: "message",
    role: "assistant",
    content: [{ type: "output_text", text }],
  });

  const responsesBodies = {
    history: (): [unknown[], string | undefined] => [
      [
        { role: "user", content: `earlier ${MARKER}` },
        assistantMessageItem("understood"),
        { role: "user", content: "a clean new question" },
      ],
      "be helpful",
    ],
    latest: (): [unknown[], string | undefined] => [
      [
        { role: "user", content: "earlier" },
        assistantMessageItem("understood"),
        { role: "user", content: `now do ${MARKER}` },
      ],
      undefined,
    ],
    // `function_call` is the model's turn (no `role` on this wire) and so
    // the boundary; the `function_call_output` answering it is inside.
    toolResult: (): [unknown[], string | undefined] => [
      [
        { role: "user", content: "look it up" },
        {
          type: "function_call",
          call_id: "call_1",
          name: "lookup",
          arguments: "{}",
        },
        {
          type: "function_call_output",
          call_id: "call_1",
          output: `found ${MARKER}`,
        },
      ],
      undefined,
    ],
    system: (): [unknown[], string | undefined] => [
      [
        { role: "user", content: "earlier" },
        assistantMessageItem("understood"),
        { role: "user", content: "a clean new question" },
      ],
      `policy mentions ${MARKER}`,
    ],
  };

  const send = {
    chat: (model: string, which: keyof typeof chatBodies) =>
      chat(model, chatBodies[which]()),
    messages: (model: string, which: keyof typeof anthropicBodies) => {
      const [msgs, system] = anthropicBodies[which]();
      return anthropic(model, msgs, system);
    },
    responses: (model: string, which: keyof typeof responsesBodies) => {
      const [input, instructions] = responsesBodies[which]();
      return responses(model, input, instructions);
    },
  };

  const protocols = ["chat", "messages", "responses"] as const;
  const passes = [
    { pass: "check pass (keyword)", latest: kwLatest, all: kwAll },
    { pass: "segment pass (custom script)", latest: scriptLatest, all: scriptAll },
  ];

  for (const { pass, latest, all } of passes) {
    for (const protocol of protocols) {
      const drive = send[protocol];

      test(
        `${pass} / ${protocol}: a marker in replayed history blocks under \`all\` and passes under \`latest_turn\``,
        async (ctx) => {
          if (!etcdReachable || !app) {
            ctx.skip();
            return;
          }
          await waitForLane(all.model);
          await waitForLane(latest.model);

          expect(
            await blocked(drive(all.model, "history")),
            "`all` must still read the whole conversation",
          ).toBe(true);
          expect(
            await blocked(drive(latest.model, "history")),
            "`latest_turn` must not read a message the model already answered",
          ).toBe(false);
        },
        60_000,
      );

      test(
        `${pass} / ${protocol}: a marker in the current user message blocks under \`latest_turn\``,
        async (ctx) => {
          if (!etcdReachable || !app) {
            ctx.skip();
            return;
          }
          await waitForLane(latest.model);
          expect(await blocked(drive(latest.model, "latest"))).toBe(true);
        },
        60_000,
      );

      test(
        `${pass} / ${protocol}: a marker in this turn's tool result blocks under \`latest_turn\``,
        async (ctx) => {
          if (!etcdReachable || !app) {
            ctx.skip();
            return;
          }
          await waitForLane(latest.model);
          expect(
            await blocked(drive(latest.model, "toolResult")),
            "the tool result answering this turn is inside the window",
          ).toBe(true);
        },
        60_000,
      );

      test(
        `${pass} / ${protocol}: a marker in the system prompt blocks under \`all\` and passes under \`latest_turn\``,
        async (ctx) => {
          if (!etcdReachable || !app) {
            ctx.skip();
            return;
          }
          await waitForLane(all.model);
          await waitForLane(latest.model);

          expect(await blocked(drive(all.model, "system"))).toBe(true);
          expect(
            await blocked(drive(latest.model, "system")),
            "system messages are never inside the window",
          ).toBe(false);
        },
        60_000,
      );
    }
  }

  // ── /v1/responses: the model's turn is a typed item, not a role ───────

  test(
    "/v1/responses: a replayed function_call's arguments are scanned under `all` and excluded once answered",
    async (ctx) => {
      if (!etcdReachable || !app) {
        ctx.skip();
        return;
      }
      await waitForLane(kwAll.model);
      await waitForLane(kwLatest.model);

      const input = [
        { role: "user", content: "look it up" },
        {
          type: "function_call",
          call_id: "call_1",
          name: "lookup",
          arguments: JSON.stringify({ q: MARKER }),
        },
        {
          type: "function_call_output",
          call_id: "call_1",
          output: "nothing found",
        },
        assistantMessageItem("all done"),
        { role: "user", content: "a clean new question" },
      ];

      expect(
        await blocked(responses(kwAll.model, input)),
        "a replayed tool call's arguments are caller-supplied text entering the model",
      ).toBe(true);
      expect(
        await blocked(responses(kwLatest.model, input)),
        "the whole loop sits before the last assistant message",
      ).toBe(false);
    },
    60_000,
  );

  test(
    "/v1/responses: a function_call_output with no later assistant item blocks under `latest_turn`",
    async (ctx) => {
      if (!etcdReachable || !app) {
        ctx.skip();
        return;
      }
      await waitForLane(kwLatest.model);

      const input = [
        { role: "user", content: "look it up" },
        {
          type: "function_call",
          call_id: "call_1",
          name: "lookup",
          arguments: "{}",
        },
        {
          type: "function_call_output",
          call_id: "call_1",
          output: `found ${MARKER}`,
        },
      ];
      expect(await blocked(responses(kwLatest.model, input))).toBe(true);
    },
    60_000,
  );

  test(
    "/v1/responses: a model turn carrying no readable text still opens the window",
    async (ctx) => {
      if (!etcdReachable || !app) {
        ctx.skip();
        return;
      }
      await waitForLane(kwLatest.model);
      await waitForLane(scriptLatest.model);

      // What an agent client replays when reasoning summaries are off:
      // the item's only payload is provider ciphertext, so it carries no
      // text to scan — but it is still the model's turn, and both halves
      // of the window rule have to agree that the marker before it is
      // history.
      const input = [
        { role: "user", content: `earlier ${MARKER}` },
        { type: "reasoning", encrypted_content: "opaque-provider-blob" },
        { role: "user", content: "a clean new question" },
      ];
      expect(
        await blocked(responses(kwLatest.model, input)),
        "check pass: the marker sits before the model's turn",
      ).toBe(false);
      expect(
        await blocked(responses(scriptLatest.model, input)),
        "segment pass must agree with the check pass",
      ).toBe(false);
    },
    60_000,
  );

  // ── masking follows the same window ───────────────────────────────────

  test(
    "a masking row on `latest_turn` rewrites the current turn and forwards history byte-identical",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) {
        ctx.skip();
        return;
      }
      // This lane never blocks, so gate it on the request reaching the
      // upstream rather than on a refusal.
      await waitConfigPropagation(async () => {
        const before = upstream!.receivedRequests.length;
        await chat(maskLatest.model, [
          { role: "user", content: `probe ${SECRET}` },
        ]);
        const forwarded = upstream!.receivedRequests
          .slice(before)
          .some((r) => r.body.includes(MASKED));
        return forwarded;
      });

      const before = upstream.receivedRequests.length;
      await chat(maskLatest.model, [
        { role: "system", content: `system holds ${SECRET}` },
        { role: "user", content: `earlier turn holds ${SECRET}` },
        { role: "assistant", content: "understood" },
        { role: "user", content: `this turn holds ${SECRET}` },
      ]);
      const sent = upstream.receivedRequests.slice(before);
      expect(sent.length, "the request must reach the upstream").toBeGreaterThan(0);
      const body = JSON.parse(sent[sent.length - 1].body) as {
        messages: Array<{ role: string; content: string | null }>;
      };
      const byRole = (role: string, index: number) =>
        body.messages.filter((m) => m.role === role)[index]?.content ?? "";

      expect(
        byRole("system", 0),
        "a system message is outside the window",
      ).toContain(SECRET);
      expect(
        byRole("user", 0),
        "history the model already answered is forwarded as the caller sent it",
      ).toContain(SECRET);
      expect(
        byRole("user", 1),
        "the current turn is rewritten",
      ).toContain(MASKED);
      expect(byRole("user", 1)).not.toContain(SECRET);
    },
    60_000,
  );
});
