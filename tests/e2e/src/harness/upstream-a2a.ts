import {
  createServer,
  type IncomingMessage,
  type Server as HttpServer,
  type ServerResponse,
} from "node:http";

/** One request the stub agent received, as the gateway sent it. */
export interface A2aReceivedRequest {
  httpMethod: string;
  path: string;
  /** The `A2A-Version` header, or `null` when the gateway sent none. */
  version: string | null;
  authorization: string | null;
  apiKey: string | null;
  body?: Record<string, unknown>;
}

/**
 * Where the agent publishes its card.
 *
 * - `origin` — at the RFC 8615 origin URI, the shape of an agent that owns its
 *   whole domain. This is the only shape the gateway used to be able to reach.
 * - `path` — under the service endpoint's own path prefix, with every other
 *   path answering `405` from a catch-all. This is the shape of any platform
 *   that multiplexes tenants under a prefix, and of a self-hosted agent behind
 *   an ingress path (api7/aisix#913).
 */
export type A2aCardMount = "origin" | "path";

export interface A2aUpstream {
  /** Register this as the agent's `url`: the A2A service endpoint. */
  url: string;
  /** Every request the agent received, in arrival order. */
  requests: A2aReceivedRequest[];
  close(): Promise<void>;
}

export interface A2aUpstreamOptions {
  cardMount?: A2aCardMount;
  /** When set, the agent is registered with `auth_type: bearer` and this token. */
  token?: string;
}

const PATH_PREFIX = "/v3/agents/serve/tenant-42";

/**
 * A stub upstream A2A agent: serves an agent card and answers JSON-RPC, while
 * recording what the gateway actually sent it — the wire version it announced
 * and the credential it presented.
 *
 * The card it serves deliberately advertises an unreachable `https://` address
 * both at the top level and inside `supportedInterfaces`, so a test can prove
 * the gateway rewrote every one of them before handing the card to a caller.
 */
export async function startA2aUpstream(
  options: A2aUpstreamOptions = {},
): Promise<A2aUpstream> {
  const mount: A2aCardMount = options.cardMount ?? "origin";
  const requests: A2aReceivedRequest[] = [];
  const servicePath = mount === "path" ? PATH_PREFIX : "/a2a";
  const cardPath =
    mount === "path"
      ? `${PATH_PREFIX}/.well-known/agent-card.json`
      : "/.well-known/agent-card.json";

  const httpServer: HttpServer = createServer((req, res) => {
    handle(req, res, {
      requests,
      servicePath,
      cardPath,
      token: options.token,
    }).catch((err: unknown) => {
      // `handle` rejects on a malformed body, and on a write to an already
      // closed socket. Unhandled, that terminates the test process; worse, the
      // request never gets a response, so a stub fault reads as a gateway
      // timeout instead of what it is. Answer on the wire either way.
      if (!res.headersSent) {
        res.writeHead(500, { "content-type": "application/json" });
      }
      res.end(JSON.stringify({ error: `a2a stub failed: ${String(err)}` }));
    });
  });
  await new Promise<void>((resolve) =>
    httpServer.listen(0, "127.0.0.1", resolve),
  );
  const address = httpServer.address();
  if (address === null || typeof address === "string") {
    throw new Error("a2a upstream: no listen address");
  }

  return {
    url: `http://127.0.0.1:${address.port}${servicePath}`,
    requests,
    close: () => new Promise((resolve) => httpServer.close(() => resolve())),
  };
}

async function handle(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: {
    requests: A2aReceivedRequest[];
    servicePath: string;
    cardPath: string;
    token?: string;
  },
): Promise<void> {
  const path = new URL(req.url ?? "/", "http://127.0.0.1").pathname;
  const header = (name: string): string | null => {
    const value = req.headers[name];
    return typeof value === "string" ? value : null;
  };
  const send = (status: number, body: unknown): void => {
    res.writeHead(status, { "content-type": "application/json" });
    res.end(JSON.stringify(body));
  };

  let body: Record<string, unknown> | undefined;
  if (req.method === "POST") {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    const raw = Buffer.concat(chunks).toString("utf8");
    body = raw ? (JSON.parse(raw) as Record<string, unknown>) : undefined;
  }

  ctx.requests.push({
    httpMethod: req.method ?? "GET",
    path,
    version: header("a2a-version"),
    authorization: header("authorization"),
    apiKey: header("x-api-key"),
    body,
  });

  if (ctx.token !== undefined && header("authorization") !== `Bearer ${ctx.token}`) {
    send(401, { error: "unauthorized" });
    return;
  }

  if (req.method === "GET" && path === ctx.cardPath) {
    send(200, {
      name: "Invoice Processor",
      description: "Stub agent for the A2A gateway e2e.",
      protocolVersion: "1.0",
      version: "2.1.0",
      url: "https://internal.upstream.invalid/a2a",
      supportedInterfaces: [
        {
          url: "https://internal.upstream.invalid/a2a",
          protocolBinding: "JSONRPC",
          protocolVersion: "1.0",
        },
      ],
      skills: [{ id: "invoice", name: "Process invoice", tags: ["billing"] }],
    });
    return;
  }

  if (req.method === "POST" && path === ctx.servicePath) {
    send(200, {
      jsonrpc: "2.0",
      id: body?.id ?? null,
      result: {
        kind: "task",
        id: "task-e2e-1",
        status: { state: "completed" },
        // Echoed so the gateway's forwarding can be asserted from the caller
        // side as well as from `requests`.
        sawVersion: header("a2a-version"),
        sawMethod: body?.method ?? null,
      },
    });
    return;
  }

  // The catch-all a real path-hosting agent platform answers with: not a 404,
  // which is what made the original report look like a missing card rather
  // than a mis-built URL.
  send(405, { error: "method not allowed" });
}
