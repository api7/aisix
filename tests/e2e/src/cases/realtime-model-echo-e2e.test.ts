import { createHash } from "node:crypto";
import { WebSocket as WsClient, WebSocketServer, type WebSocket as WsSocket } from "ws";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  waitConfigPropagation,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: `/v1/realtime` names the model the CALLER addressed (#1088).
//
// `session.created` and `session.updated` are the only Realtime server
// events that carry a model, under `session.model`, and the relay used to
// hand the provider's own value straight through — so a caller who
// connected with a gateway alias was told a different model name than the
// one they asked for. Same divergence #1086 removed from the HTTP native
// passthrough paths.
//
// The scenario is the one an alias exists for: the operator configures
// `model_name` (what to ask the provider for) and the provider answers with
// a DIFFERENT id. Every frame below therefore reports
// `UPSTREAM_REPORTED_MODEL`, which matches neither the caller's alias nor
// the configured `model_name`.
//
// The reverse direction is covered too, because this fix creates the need
// for it: a client that echoes the session object back as `session.update`
// now sends the alias where it used to send the provider's id.

const CALLER_PLAINTEXT = "sk-realtime-model-echo-caller";
const CALLER_KEY_HASH = createHash("sha256").update(CALLER_PLAINTEXT).digest("hex");

const ALIAS = "echo-realtime";
/** What the operator configured as the upstream model id. */
const CONFIGURED_MODEL_NAME = "gpt-realtime-mock";
/** What the provider actually answers with — deliberately neither of the other two. */
const UPSTREAM_REPORTED_MODEL = "gpt-realtime-2026-01-01";
/**
 * A SECOND model the client configures itself, nested inside the session's
 * audio config. The gateway never aliased it, so restamping it would rename
 * a model the caller is entitled to see under its real name.
 */
const CLIENT_TRANSCRIPTION_MODEL = "gpt-4o-transcribe";

interface RealtimeUpstream {
  port: number;
  /** Every text frame the gateway relayed upstream. */
  frames: string[];
  close(): Promise<void>;
}

/**
 * Mock OpenAI Realtime upstream: greets with `session.created`, answers a
 * `session.update` with `session.updated`, and anything else with a
 * `response.done`. Every session frame names `UPSTREAM_REPORTED_MODEL` at
 * `session.model` and `CLIENT_TRANSCRIPTION_MODEL` deeper down.
 */
async function startRealtimeUpstream(): Promise<RealtimeUpstream> {
  const frames: string[] = [];
  const sessionObject = {
    id: "sess_echo_01",
    object: "realtime.session",
    model: UPSTREAM_REPORTED_MODEL,
    output_modalities: ["audio"],
    audio: {
      input: { transcription: { model: CLIENT_TRANSCRIPTION_MODEL } },
    },
  };
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  wss.on("connection", (socket: WsSocket) => {
    socket.send(
      JSON.stringify({
        type: "session.created",
        event_id: "ev_created",
        session: sessionObject,
      }),
    );
    socket.on("message", (data) => {
      const text = data.toString();
      frames.push(text);
      const type = (JSON.parse(text) as { type?: string }).type;
      if (type === "session.update") {
        socket.send(
          JSON.stringify({
            type: "session.updated",
            event_id: "ev_updated",
            session: sessionObject,
          }),
        );
        return;
      }
      socket.send(
        JSON.stringify({
          type: "response.done",
          response: {
            id: "resp_echo_01",
            usage: { input_tokens: 9, output_tokens: 4 },
          },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => wss.on("listening", resolve));
  const addr = wss.address();
  if (addr === null || typeof addr === "string") throw new Error("no port");
  return {
    port: addr.port,
    frames,
    close: () =>
      new Promise<void>((resolve, reject) => wss.close((e) => (e ? reject(e) : resolve()))),
  };
}

/** Open a relay session: send frames, and await the next one the gateway delivers. */
function session(app: SpawnedApp): {
  send(frame: unknown): void;
  next(): Promise<string>;
  close(): void;
  opened: Promise<void>;
} {
  const url = `${app.proxyUrl.replace("http://", "ws://")}/v1/realtime?model=${ALIAS}`;
  const ws = new WsClient(url, {
    headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
  });
  const inbox: string[] = [];
  const waiters: Array<(v: string) => void> = [];
  ws.on("message", (d) => {
    const text = d.toString();
    const waiter = waiters.shift();
    if (waiter) waiter(text);
    else inbox.push(text);
  });
  return {
    opened: new Promise<void>((resolve, reject) => {
      ws.on("open", () => resolve());
      ws.on("unexpected-response", (_q, res) =>
        reject(new Error(`upgrade refused: ${res.statusCode}`)),
      );
      ws.on("error", (e) => reject(e));
    }),
    send: (frame) => ws.send(JSON.stringify(frame)),
    next: () =>
      new Promise<string>((resolve) => {
        const buffered = inbox.shift();
        if (buffered !== undefined) resolve(buffered);
        else waiters.push(resolve);
      }),
    close: () => ws.terminate(),
  };
}

describe("realtime model echo e2e: session frames name what the caller asked for", () => {
  let app: SpawnedApp | undefined;
  let upstream: RealtimeUpstream | undefined;
  let etcdReachable = false;

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);
    upstream = await startRealtimeUpstream();

    const pk = await seed.createProviderKey({
      display_name: "realtime-echo-pk",
      secret: "sk-upstream-realtime",
      api_base: `http://127.0.0.1:${upstream.port}/v1`,
    });
    await seed.createModel({
      display_name: ALIAS,
      provider: "openai",
      model_name: CONFIGURED_MODEL_NAME,
      provider_key_id: pk.id,
    });
    await seed.createApiKey({
      key_hash: CALLER_KEY_HASH,
      allowed_models: ["*"],
    });

    // Gate on the DP snapshot: the WS upgrade authenticates against the same
    // snapshot, and a handshake fired before the caller key propagates is
    // rejected outright.
    await waitConfigPropagation(async () => {
      const res = await fetch(`${app!.proxyUrl}/v1/models`, {
        headers: { authorization: `Bearer ${CALLER_PLAINTEXT}` },
      });
      if (res.status !== 200) {
        await res.text();
        return false;
      }
      const body = (await res.json()) as { data?: Array<{ id?: string }> };
      return (body.data ?? []).some((m) => m.id === ALIAS);
    });
  }, 60_000);

  afterAll(async () => {
    await app?.exit();
    await upstream?.close();
  });

  test(
    "session.created and session.updated name the alias, not the upstream id",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) return void ctx.skip();

      const s = session(app);
      await s.opened;

      const created = JSON.parse(await s.next()) as {
        type?: string;
        session?: {
          model?: string;
          id?: string;
          audio?: { input?: { transcription?: { model?: string } } };
        };
      };
      expect(created.type).toBe("session.created");
      expect(created.session?.model).toBe(ALIAS);
      // The alias is the whole point: neither the provider's own id nor the
      // configured upstream name may reach the caller in that slot.
      expect(created.session?.model).not.toBe(UPSTREAM_REPORTED_MODEL);
      expect(created.session?.model).not.toBe(CONFIGURED_MODEL_NAME);
      // Everything else in the session object relays as written — including
      // the transcription model the CLIENT configured, which is a different
      // model and was never aliased.
      expect(created.session?.id).toBe("sess_echo_01");
      expect(created.session?.audio?.input?.transcription?.model).toBe(
        CLIENT_TRANSCRIPTION_MODEL,
      );

      s.send({ type: "session.update", session: { instructions: "hi" } });
      const updated = JSON.parse(await s.next()) as {
        type?: string;
        session?: { model?: string };
      };
      expect(updated.type).toBe("session.updated");
      expect(updated.session?.model).toBe(ALIAS);

      s.close();
    },
    60_000,
  );

  test(
    "a client that echoes the session back reaches the provider with the provider's own id",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) return void ctx.skip();

      // The mirror of the fix above, and the reason it is needed: the
      // gateway just told this client its model is `echo-realtime`, so the
      // ordinary read-modify-write of the session object sends that name
      // back up. Before the restamp the client would have echoed the
      // provider's own id and the provider would have recognised it.
      const s = session(app);
      await s.opened;
      const created = JSON.parse(await s.next()) as { session?: { model?: string } };
      expect(created.session?.model).toBe(ALIAS);

      const baseline = upstream.frames.length;
      s.send({
        type: "session.update",
        session: { model: created.session?.model, instructions: "carry on" },
      });
      await s.next();

      const relayed = JSON.parse(upstream.frames[baseline]!) as {
        session?: { model?: string; instructions?: string };
      };
      expect(relayed.session?.model).toBe(CONFIGURED_MODEL_NAME);
      expect(relayed.session?.instructions).toBe("carry on");

      // Only the alias is translated. A client naming something else keeps
      // its own words, so the provider answers about the model the client
      // actually named rather than one the gateway substituted.
      const otherBaseline = upstream.frames.length;
      s.send({
        type: "session.update",
        session: { model: "some-other-model" },
      });
      await s.next();
      const other = JSON.parse(upstream.frames[otherBaseline]!) as {
        session?: { model?: string };
      };
      expect(other.session?.model).toBe("some-other-model");

      s.close();
    },
    60_000,
  );

  test(
    "frames that name no model relay byte-for-byte",
    async (ctx) => {
      if (!etcdReachable || !app || !upstream) return void ctx.skip();

      // The Response object a `response.done` frame carries has no `model`
      // field at all, so the relay must not acquire one — and the usage the
      // session accounting reads off it must survive untouched.
      const s = session(app);
      await s.opened;
      await s.next(); // session.created

      s.send({ type: "response.create" });
      const done = JSON.parse(await s.next()) as {
        type?: string;
        model?: unknown;
        response?: { id?: string; model?: unknown; usage?: { input_tokens?: number } };
      };
      expect(done.type).toBe("response.done");
      expect(done.response?.id).toBe("resp_echo_01");
      expect(done.response?.usage?.input_tokens).toBe(9);
      expect(done.model).toBeUndefined();
      expect(done.response?.model).toBeUndefined();

      s.close();
    },
    60_000,
  );
});
