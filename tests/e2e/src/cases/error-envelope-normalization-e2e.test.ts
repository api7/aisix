import { createHash } from "node:crypto";
import OpenAI, { APIError } from "openai";
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

// E2E: cross-provider error envelope normalization. Each upstream
// provider has its own native error wire shape; the OpenAI-SDK
// caller must always see an OpenAI-shape error envelope no matter
// which upstream the gateway dispatched to. Without normalization,
// SDK error-handling code that branches on `error.type` would be
// broken across half the gateway's provider matrix.
//
// User journey: caller speaks OpenAI Chat Completions to the
// gateway → gateway dispatches to {anthropic, gemini, deepseek}
// upstream → upstream returns 4xx in its native shape → gateway
// translates body to OpenAI shape AND preserves the upstream's
// status code → SDK caller sees an APIError with parseable
// envelope.
//
// Native error shapes (each derived from the upstream provider's
// official docs, not from gateway source):
//
//   - Anthropic: `{ type: "error", error: { type, message } }`
//     <https://docs.anthropic.com/en/api/errors>
//   - Gemini via OpenAI-compat endpoint (the wire the gateway's
//     gemini bridge actually talks to per Google's published
//     compatibility doc) returns OpenAI-shape errors:
//     `{ error: { message, type, code } }`
//     <https://ai.google.dev/gemini-api/docs/openai>
//   - DeepSeek (OpenAI-compat): `{ error: { message, type, code } }`
//     <https://api-docs.deepseek.com>
//
// Target shape (OpenAI Chat Completions error envelope):
//   `{ error: { message: string, type: string, code?: string|null } }`
//   <https://platform.openai.com/docs/guides/error-codes/api-errors>
//
// Two of the three providers (gemini, deepseek) speak OpenAI-shape
// upstream → the gateway's job is faithful pass-through; a regression
// that mangled the envelope would fail those cases. Anthropic is the
// one true translation case (Anthropic-shape in, OpenAI-shape out).

const CALLER_PLAINTEXT = "sk-err-norm-caller";
const CALLER_KEY_HASH = createHash("sha256")
  .update(CALLER_PLAINTEXT)
  .digest("hex");

interface ProviderCase {
  readonly provider: "anthropic" | "google" | "deepseek";
  readonly upstreamModelId: string;
  readonly displayName: string;
  // The wire shape the upstream sends back on a 400. Each provider
  // has its OWN native shape; the gateway must translate to OpenAI.
  readonly nativeErrorBody: unknown;
  // A distinctive substring of the upstream's error message that
  // should reach the caller (the gateway can rewrap the envelope
  // shape, but the underlying reason must be preserved so the
  // caller knows what actually went wrong).
  readonly upstreamMessageSubstr: string;
  // Anthropic's documented endpoint is
  // `https://api.anthropic.com/v1/messages` so api_base is the bare
  // host (the bridge composes `/v1/messages` on its own).
  // OpenAI-compat providers expose at `<host>/v1/chat/completions`
  // — `/v1` belongs to api_base; cf. `docs/api-admin.md` example
  // `"api_base": "https://api.openai.com/v1"`.
  readonly apiBaseSuffix: "" | "/v1";
}

const CASES: ReadonlyArray<ProviderCase> = [
  {
    provider: "anthropic",
    upstreamModelId: "claude-3-5-haiku-20241022",
    displayName: "err-norm-anthropic",
    // Anthropic native 400 per
    // <https://docs.anthropic.com/en/api/errors#http-errors>:
    //   {"type":"error","error":{"type":"invalid_request_error","message":"..."}}
    nativeErrorBody: {
      type: "error",
      error: {
        type: "invalid_request_error",
        message:
          "Anthropic upstream rejected the request: bogus_param missing",
      },
    },
    upstreamMessageSubstr: "bogus_param missing",
    apiBaseSuffix: "",
  },
  {
    provider: "google",
    upstreamModelId: "gemini-2.0-flash",
    displayName: "err-norm-google",
    // The google bridge talks to Google's OpenAI-compatibility
    // endpoint per <https://ai.google.dev/gemini-api/docs/openai>,
    // which returns errors in OpenAI envelope shape. The
    // "normalization" the gateway performs for gemini is therefore
    // trivial pass-through — but pinning it catches a regression
    // that mangles or drops the envelope on the way through.
    nativeErrorBody: {
      error: {
        message: "Gemini upstream rejected: invalid temperature",
        type: "invalid_request_error",
        code: "invalid_temperature",
      },
    },
    upstreamMessageSubstr: "invalid temperature",
    apiBaseSuffix: "/v1",
  },
  {
    provider: "deepseek",
    upstreamModelId: "deepseek-chat",
    displayName: "err-norm-deepseek",
    // DeepSeek is OpenAI-compatible per
    // <https://api-docs.deepseek.com> — so its native error wire
    // is already OpenAI-shape. The gateway should pass-through
    // without mangling the envelope.
    nativeErrorBody: {
      error: {
        message: "DeepSeek upstream rejected: malformed messages array",
        type: "invalid_request_error",
        code: "invalid_request",
      },
    },
    upstreamMessageSubstr: "malformed messages array",
    apiBaseSuffix: "/v1",
  },
];

describe("error envelope normalization e2e: provider-native 4xx → OpenAI-shape envelope", () => {
  let app: SpawnedApp | undefined;
  let admin: AdminClient | undefined;
  let etcdReachable = false;
  const upstreams: OpenAiUpstream[] = [];

  beforeAll(async () => {
    etcdReachable = await new EtcdClient().ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    admin = new AdminClient(app.adminUrl, app.adminKey);
    await admin.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });
  });

  afterAll(async () => {
    await app?.exit();
    await Promise.all(upstreams.map((u) => u.close()));
  });

  for (const tc of CASES) {
    test(`${tc.provider} upstream 400 → caller sees OpenAI-shape error envelope with preserved status + reason`, async (ctx) => {
      if (!etcdReachable || !app || !admin) {
        ctx.skip();
        return;
      }

      // Mock upstream returns the provider's NATIVE 400 shape. A
      // gateway with broken normalization would either drop the
      // body entirely (caller sees empty error), echo the native
      // shape verbatim (SDK can't parse it), or convert to a
      // generic 500 (status preservation broken).
      const upstream = await startOpenAiUpstream({
        status: 400,
        errorBody: tc.nativeErrorBody,
      });
      upstreams.push(upstream);

      const pk = await admin.createProviderKey({
        display_name: `${tc.displayName}-pk`,
        secret: "sk-mock",
        api_base: `${upstream.baseUrl}${tc.apiBaseSuffix}`,
      });
      await admin.createModel({
        display_name: tc.displayName,
        provider: tc.provider,
        model_name: tc.upstreamModelId,
        provider_key_id: pk.id,
      });

      const client = new OpenAI({
        apiKey: CALLER_PLAINTEXT,
        baseURL: `${app.proxyUrl}/v1`,
        maxRetries: 0,
      });

      // Readiness gate: poll until the gateway returns the
      // *upstream's* 400, not a snapshot-lag "model not found"
      // 400. The latter would prove the gateway returns 400 but
      // not that the upstream was actually reached — a regression
      // that broke envelope normalization could pass on the wrong
      // 400 path. Same disambiguation pattern fallback-edges-e2e
      // uses: pin the upstream's distinctive substring.
      await waitConfigPropagation(async () => {
        try {
          await client.chat.completions.create({
            model: tc.displayName,
            messages: [{ role: "user", content: "ready-probe" }],
          });
          return false; // unexpected 200 — keep polling
        } catch (e) {
          if (!(e instanceof APIError) || e.status !== 400) return false;
          const msg = (e.error as { message?: unknown })?.message;
          return (
            typeof msg === "string" && msg.includes(tc.upstreamMessageSubstr)
          );
        }
      });

      let caught: unknown;
      try {
        await client.chat.completions.create({
          model: tc.displayName,
          messages: [{ role: "user", content: "trigger 400" }],
        });
      } catch (e) {
        caught = e;
      }

      // Status preservation: a regression that wrapped 4xx as 5xx
      // (e.g. "all upstream errors are 500") would mislead callers
      // about whether the request was their fault or the gateway's.
      expect(caught).toBeInstanceOf(APIError);
      if (!(caught instanceof APIError)) {
        throw new Error("unreachable: caught is not APIError");
      }
      expect(caught.status).toBe(400);

      // Envelope shape: OpenAI Chat Completions error spec requires
      // an `error` object with at minimum `message` (string) and
      // `type` (string). `code` is optional but conventionally
      // populated; assert it's either absent or a string when
      // present.
      expect(typeof caught.error).toBe("object");
      const err = caught.error as {
        message?: unknown;
        type?: unknown;
        code?: unknown;
      };
      expect(typeof err.message).toBe("string");
      expect((err.message as string).length).toBeGreaterThan(0);
      expect(typeof err.type).toBe("string");
      expect((err.type as string).length).toBeGreaterThan(0);
      if (err.code !== undefined && err.code !== null) {
        expect(typeof err.code).toBe("string");
      }

      // Reason preservation: the underlying upstream message must
      // be reachable from the caller's error envelope, otherwise
      // the caller has no signal about WHY the request failed
      // (just "something was 400"). The gateway is allowed to
      // re-wrap or annotate, but the upstream's distinctive
      // substring must survive somewhere in the message field.
      // Without this, a regression that replaced upstream messages
      // with a generic "request failed" string would silently lose
      // critical debugging signal.
      expect(err.message).toContain(tc.upstreamMessageSubstr);
    });
  }
});
