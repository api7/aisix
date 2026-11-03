import { createServer, request, type Server } from "node:http";

import { pickFreePort } from "./ports.js";

/** Forwards `${prefix}/...` to `target` as `/...`; anything else is a 404. */
export async function startPrefixStrippingProxy(prefix: string, target: URL) {
  const port = await pickFreePort();
  const seen: string[] = [];
  const server: Server = createServer((req, res) => {
    const path = req.url ?? "";
    if (!path.startsWith(`${prefix}/`)) {
      res.writeHead(404).end();
      return;
    }
    seen.push(path.split("?")[0]);
    const upstream = request(
      {
        hostname: target.hostname,
        port: target.port,
        method: req.method,
        path: path.slice(prefix.length),
        headers: req.headers,
      },
      (upstreamRes) => {
        res.writeHead(upstreamRes.statusCode ?? 502, upstreamRes.headers);
        upstreamRes.pipe(res);
      },
    );
    upstream.on("error", () => {
      if (res.headersSent) res.destroy();
      else res.writeHead(502).end();
    });
    req.pipe(upstream);
  });
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${port}`,
    seen,
    close: () => {
      server.closeAllConnections();
      return new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}
