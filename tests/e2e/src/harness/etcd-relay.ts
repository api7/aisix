import { once } from "node:events";
import { connect, createServer, type Socket } from "node:net";

import { etcdEndpoint } from "./etcd.js";
import { pickFreePort } from "./ports.js";

/**
 * A TCP relay the gateway dials instead of etcd, so a spec can decide
 * exactly when the first configuration read is allowed to complete.
 *
 * Three states, and the difference between the first two matters:
 *
 * - `hold()` — listening, connections accepted, nothing forwarded. The
 *   gateway's range read stays in flight indefinitely (etcd-client's
 *   `Client::connect` is lazy, so the connection is only established
 *   here). No error is produced; the read is simply slow.
 * - `refuse()` — not listening. Every dial gets `ECONNREFUSED`, which is
 *   the reachability failure the watch supervisor retries on.
 * - `release()` — listening and forwarding, including every connection
 *   held since `hold()`.
 *
 * The target is always `etcdEndpoint()`, so the relay stays on this
 * fork's own cluster.
 */
export interface EtcdRelay {
  /** Endpoint to put in the gateway's `etcd.endpoints`. */
  endpoint: string;
  /** Accept connections but forward nothing. */
  hold(): Promise<void>;
  /** Stop listening: dials are refused and held connections are dropped. */
  refuse(): Promise<void>;
  /** Forward held and future connections to the real etcd. */
  release(): Promise<void>;
  /** Tear the relay down. */
  stop(): Promise<void>;
}

export async function startEtcdRelay(): Promise<EtcdRelay> {
  const target = new URL(etcdEndpoint());
  if (!target.port) {
    throw new Error(`etcd endpoint ${target.href} has no port; the relay cannot target it`);
  }
  const targetHost = target.hostname;
  const targetPort = Number(target.port);
  const port = await pickFreePort();

  let forwarding = false;
  const held: Socket[] = [];
  const open = new Set<Socket>();

  const track = (s: Socket) => {
    open.add(s);
    s.on("close", () => open.delete(s));
    // A peer vanishing (gateway teardown, `refuse()`) is expected here and
    // must not surface as an unhandled 'error' event.
    s.on("error", () => s.destroy());
  };

  const forward = (client: Socket) => {
    const upstream = connect(targetPort, targetHost);
    track(upstream);
    upstream.on("error", () => client.destroy());
    client.on("error", () => upstream.destroy());
    client.pipe(upstream);
    upstream.pipe(client);
  };

  const server = createServer((client) => {
    track(client);
    if (forwarding) forward(client);
    else held.push(client);
  });
  // Keeps a post-listen accept error from throwing on an EventEmitter with
  // no 'error' listener. `listen()` adds its own, which still sees the event.
  server.on("error", () => {});

  const listen = async () => {
    if (server.listening) return;
    // Race the two outcomes: awaiting `listening` alone turns a bind
    // failure into a hang, and the spec would time out with no cause.
    const up = new Promise<void>((resolve, reject) => {
      const onListening = () => {
        server.off("error", onError);
        resolve();
      };
      const onError = (err: Error) => {
        server.off("listening", onListening);
        reject(err);
      };
      server.once("listening", onListening);
      server.once("error", onError);
    });
    server.listen(port, "127.0.0.1");
    await up;
  };

  const unlisten = async () => {
    if (!server.listening) return;
    const closed = once(server, "close");
    server.close();
    for (const s of open) s.destroy();
    open.clear();
    held.length = 0;
    await closed;
  };

  return {
    endpoint: `http://127.0.0.1:${port}`,
    async hold() {
      forwarding = false;
      await listen();
    },
    async refuse() {
      forwarding = false;
      await unlisten();
    },
    async release() {
      forwarding = true;
      await listen();
      for (const client of held.splice(0)) forward(client);
    },
    async stop() {
      await unlisten();
    },
  };
}
