import {
  createServer,
  type IncomingMessage,
  type Server as HttpServer,
  type ServerResponse,
} from "node:http";

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

export interface McpUpstream {
  /** Streamable HTTP endpoint of this upstream (`http://127.0.0.1:<port>/mcp`). */
  url: string;
  close(): Promise<void>;
}

export interface McpUpstreamOptions {
  /**
   * Also expose a `lookup` tool whose result carries `structuredContent`
   * alongside a fixed, always-clean text block — the shape a tool uses to
   * return machine-readable output. Off by default so the tool inventory
   * every other suite asserts on stays `echo` + `reverse`.
   */
  structuredTool?: boolean;
}

/**
 * A real MCP upstream server built on the official TypeScript SDK, speaking
 * the stateless Streamable HTTP transport with JSON responses — the exact
 * interop partner of the gateway's per-operation ephemeral MCP client.
 *
 * Every upstream exposes the same two tools, labelled so tests can both
 * observe routing and grant one tool while denying the other on one server:
 *   - `echo`    → returns `<label>:<text>`
 *   - `reverse` → returns <text> reversed
 *
 * A fresh SDK `Server` + transport is built per request (the SDK's stateless
 * pattern); the gateway reconnects per operation, so nothing is shared.
 */
export async function startMcpUpstream(
  label: string,
  options: McpUpstreamOptions = {},
): Promise<McpUpstream> {
  const httpServer: HttpServer = createServer((req, res) => {
    void handle(label, req, res, options);
  });
  await new Promise<void>((resolve) =>
    httpServer.listen(0, "127.0.0.1", resolve),
  );
  const address = httpServer.address();
  if (address === null || typeof address === "string") {
    throw new Error("mcp upstream: no listen address");
  }
  return {
    url: `http://127.0.0.1:${address.port}/mcp`,
    close: () =>
      new Promise((resolve) => httpServer.close(() => resolve())),
  };
}

async function handle(
  label: string,
  req: IncomingMessage,
  res: ServerResponse,
  options: McpUpstreamOptions,
): Promise<void> {
  try {
    if (req.method !== "POST") {
      res.writeHead(405).end();
      return;
    }
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    const body: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));

    const server = new Server(
      { name: `${label}-upstream`, version: "0.1.0" },
      { capabilities: { tools: {} } },
    );
    server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: "echo",
          description: `echo the text back prefixed with "${label}:"`,
          inputSchema: {
            type: "object",
            properties: { text: { type: "string" } },
          },
        },
        {
          name: "reverse",
          description: "return the text reversed",
          inputSchema: {
            type: "object",
            properties: { text: { type: "string" } },
          },
        },
        ...(options.structuredTool
          ? [
              {
                name: "lookup",
                description: "return the text as structured output",
                inputSchema: {
                  type: "object" as const,
                  properties: { text: { type: "string" } },
                },
              },
            ]
          : []),
      ],
    }));
    server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const text = String(request.params.arguments?.text ?? "");
      if (request.params.name === "lookup") {
        // The text block is deliberately constant and clean: only
        // `structuredContent` carries the caller's value, which is exactly
        // the case a content-blocks-only scan would miss.
        return {
          content: [{ type: "text", text: "lookup ok" }],
          structuredContent: { record: { note: text } },
        };
      }
      const out =
        request.params.name === "reverse"
          ? [...text].reverse().join("")
          : `${label}:${text}`;
      return { content: [{ type: "text", text: out }] };
    });

    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true,
    });
    res.on("close", () => {
      void transport.close();
      void server.close();
    });
    await server.connect(transport);
    await transport.handleRequest(req, res, body);
  } catch {
    if (!res.headersSent) res.writeHead(500).end();
  }
}
