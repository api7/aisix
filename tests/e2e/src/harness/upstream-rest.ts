import {
  createServer,
  type IncomingMessage,
  type Server as HttpServer,
  type ServerResponse,
} from "node:http";

export interface RestUpstream {
  /** Base URL of the fake REST API (`http://127.0.0.1:<port>/v1`). */
  baseUrl: string;
  /** The bearer token the API requires on its authenticated routes. */
  token: string;
  close(): Promise<void>;
}

/**
 * A fake "ERP" REST API for OpenAPI-backed MCP server tests — a plain HTTP
 * server, no MCP involved. Routes (all under `/v1`, bearer-authenticated
 * except `/fail`):
 *   - `GET  /v1/items/<id>`  → echoes `{ id, query }` (path + query params)
 *   - `POST /v1/orders`      → echoes `{ created: <json body> }`
 *   - `GET  /v1/fail`        → always `500 boom`
 *
 * The matching OpenAPI document lives with the test case; this server is the
 * live interop partner its generated tools are called against.
 */
export async function startRestUpstream(): Promise<RestUpstream> {
  const token = "tok-erp-e2e";

  const httpServer: HttpServer = createServer((req, res) => {
    void handle(token, req, res);
  });
  await new Promise<void>((resolve) =>
    httpServer.listen(0, "127.0.0.1", resolve),
  );
  const address = httpServer.address();
  if (address === null || typeof address === "string") {
    throw new Error("rest upstream: no listen address");
  }
  return {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    token,
    close: () => new Promise((resolve) => httpServer.close(() => resolve())),
  };
}

async function handle(
  token: string,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const url = new URL(req.url ?? "/", "http://127.0.0.1");
  const send = (status: number, body: unknown): void => {
    res.writeHead(status, { "content-type": "application/json" });
    res.end(JSON.stringify(body));
  };

  if (req.method === "GET" && url.pathname === "/v1/fail") {
    res.writeHead(500, { "content-type": "text/plain" });
    res.end("boom");
    return;
  }

  if (req.headers.authorization !== `Bearer ${token}`) {
    send(401, { error: "no auth" });
    return;
  }

  const itemMatch = /^\/v1\/items\/([^/]+)$/.exec(url.pathname);
  if (req.method === "GET" && itemMatch) {
    send(200, {
      id: decodeURIComponent(itemMatch[1]),
      query: Object.fromEntries(url.searchParams),
    });
    return;
  }

  if (req.method === "POST" && url.pathname === "/v1/orders") {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    let body: unknown;
    try {
      body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    } catch {
      send(400, { error: "invalid json" });
      return;
    }
    send(200, { created: body });
    return;
  }

  send(404, { error: "not found" });
}
