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
  /**
   * Raw request bodies this upstream received, in arrival order — exactly
   * the bytes the gateway's MCP client sent, so a masking suite can
   * byte-diff what reached the upstream.
   */
  received: string[];
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
  /**
   * Also expose a `report` tool returning fixed rich content regardless of
   * its arguments: a text summary block, an embedded resource carrying
   * `log` as `resource.text` (the natural shape for a compile/sim log),
   * and a `structuredContent` object with a `log` string leaf plus a
   * numeric field. The mask write-back suite owns the strings.
   *
   * `structuredFields` are merged into `structuredContent` as further
   * members — the shape where the sensitive value is a real JSON field
   * (`{"version": "12.1"}`) rather than a number inside prose, so the
   * field NAME is the only thing telling a rule what the value is.
   */
  reportContent?: {
    summary: string;
    log: string;
    structuredLog: string;
    structuredFields?: Record<string, string | number>;
  };
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
  const received: string[] = [];
  const httpServer: HttpServer = createServer((req, res) => {
    void handle(label, req, res, options, received);
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
    received,
    close: () =>
      new Promise((resolve) => httpServer.close(() => resolve())),
  };
}

async function handle(
  label: string,
  req: IncomingMessage,
  res: ServerResponse,
  options: McpUpstreamOptions,
  received: string[],
): Promise<void> {
  try {
    if (req.method !== "POST") {
      res.writeHead(405).end();
      return;
    }
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    const raw = Buffer.concat(chunks).toString("utf8");
    received.push(raw);
    const body: unknown = JSON.parse(raw);

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
        ...(options.reportContent
          ? [
              {
                name: "report",
                description: "return a fixed rich report (text + resource + structured)",
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
      if (request.params.name === "report" && options.reportContent) {
        const { summary, log, structuredLog, structuredFields } =
          options.reportContent;
        return {
          content: [
            { type: "text", text: summary },
            {
              type: "resource",
              resource: {
                uri: "file:///run.log",
                mimeType: "text/plain",
                text: log,
              },
            },
          ],
          structuredContent: {
            log: structuredLog,
            cells: 42,
            ...(structuredFields ?? {}),
          },
        };
      }
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
