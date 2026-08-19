import { createHash, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  spawnApp,
  startRestUpstream,
  waitConfigPropagation,
  type RestUpstream,
  type SpawnedApp,
} from "../harness/index.js";

// E2E: an OpenAPI-backed MCP server (`type: openapi`) against a real gateway
// + etcd + a real (fake-ERP) REST upstream. No MCP upstream exists — the
// gateway generates the tools from the registered OpenAPI document and
// executes tool calls as plain HTTP requests.
//
// Pinned contract (the issue's acceptance criteria at the binary level):
//   - `tools/list` exposes one namespaced tool per spec operation, with the
//     input schema generated from the spec (params + `body` property);
//   - `tools/call` performs the REST request: path substitution, query
//     serialization, JSON body, and the gateway-held bearer credential the
//     agent never sees;
//   - a non-2xx REST response surfaces as a tool-level `isError` result with
//     the status and body;
//   - the per-tool ACL governs generated tools exactly like real MCP tools:
//     a key scoped to one tool neither lists nor calls the others.

const KEY_FULL = "sk-mcp-openapi-full";
const KEY_SCOPED = "sk-mcp-openapi-scoped";

const sha256 = (s: string) => createHash("sha256").update(s).digest("hex");

interface ToolDef {
  name: string;
  description?: string;
  inputSchema?: {
    type?: string;
    properties?: Record<string, { type?: string; description?: string }>;
    required?: string[];
  };
}

interface RpcReply {
  status: number;
  json?: {
    result?: {
      tools?: ToolDef[];
      content?: Array<{ type: string; text?: string }>;
      isError?: boolean;
    };
    error?: { code: number; message: string };
  };
}

describe("mcp openapi e2e: REST API exposed as MCP tools", () => {
  let app: SpawnedApp | undefined;
  let erp: RestUpstream | undefined;
  let etcdReachable = false;

  const post = async (token: string, body: unknown): Promise<RpcReply> => {
    const res = await fetch(`${app!.proxyUrl}/mcp`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${token}`,
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
      },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let json: RpcReply["json"];
    try {
      json = text ? JSON.parse(text) : undefined;
    } catch {
      json = undefined;
    }
    return { status: res.status, json };
  };

  /** Spec-faithful per-operation handshake (the endpoint is stateless). */
  const initialize = async (token: string): Promise<number> => {
    const init = await post(token, {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: {
        protocolVersion: "2025-11-25",
        capabilities: {},
        clientInfo: { name: "mcp-openapi-e2e", version: "0.1" },
      },
    });
    await post(token, { jsonrpc: "2.0", method: "notifications/initialized" });
    return init.status;
  };

  const listTools = async (token: string): Promise<ToolDef[]> => {
    await initialize(token);
    const r = await post(token, {
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
      params: {},
    });
    expect(r.status).toBe(200);
    return r.json?.result?.tools ?? [];
  };

  const callTool = async (
    token: string,
    name: string,
    args: unknown,
  ): Promise<{
    status: number;
    isError?: boolean;
    text?: string;
    rpcError?: string;
  }> => {
    const r = await post(token, {
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name, arguments: args },
    });
    return {
      status: r.status,
      isError: r.json?.result?.isError,
      text: r.json?.result?.content?.[0]?.text,
      rpcError: r.json?.error?.message,
    };
  };

  beforeAll(async () => {
    const etcd = new EtcdClient();
    etcdReachable = await etcd.ping();
    if (!etcdReachable) return;

    erp = await startRestUpstream();
    app = await spawnApp();
    const seed = new SeedClient(etcd, app.etcdPrefix);

    await seed.update("mcp_servers", randomUUID(), {
      name: "erp",
      type: "openapi",
      url: erp.baseUrl,
      auth_type: "bearer",
      secret: erp.token,
      enabled: true,
      spec: {
        openapi: "3.0.0",
        info: { title: "ERP", version: "1.0.0" },
        paths: {
          "/items/{id}": {
            get: {
              operationId: "getItem",
              summary: "Fetch one item",
              parameters: [
                {
                  name: "id",
                  in: "path",
                  required: true,
                  schema: { type: "integer" },
                },
                {
                  name: "verbose",
                  in: "query",
                  description: "Include details",
                  schema: { type: "boolean" },
                },
              ],
            },
          },
          "/orders": {
            post: {
              operationId: "createOrder",
              requestBody: {
                required: true,
                content: {
                  "application/json": {
                    schema: {
                      type: "object",
                      properties: { note: { type: "string" } },
                      required: ["note"],
                    },
                  },
                },
              },
            },
          },
          "/fail": { get: { operationId: "failOp" } },
        },
      },
    });

    await seed.createApiKey({
      key_hash: sha256(KEY_FULL),
      allowed_models: [],
      mcp_access: { allow: ["*"] },
    });
    await seed.createApiKey({
      key_hash: sha256(KEY_SCOPED),
      allowed_models: [],
      mcp_access: { allow: ["erp__getitem"] },
    });

    // Tolerant probe (no assertions): both the key and the server must have
    // propagated before the pinned tests run.
    await waitConfigPropagation(async () => {
      if ((await initialize(KEY_FULL)) !== 200) return false;
      const r = await post(KEY_FULL, {
        jsonrpc: "2.0",
        id: 2,
        method: "tools/list",
        params: {},
      });
      return (r.json?.result?.tools ?? []).length === 3;
    });
  }, 120_000);

  afterAll(async () => {
    await app?.exit();
    await erp?.close();
  });

  test("tools/list exposes generated tools with spec-derived schemas", async () => {
    if (!etcdReachable) return;
    const tools = await listTools(KEY_FULL);
    const names = tools.map((t) => t.name).sort();
    expect(names).toEqual(["erp__createorder", "erp__failop", "erp__getitem"]);

    const getItem = tools.find((t) => t.name === "erp__getitem")!;
    expect(getItem.description).toBe("Fetch one item");
    expect(getItem.inputSchema?.properties?.id?.type).toBe("integer");
    expect(getItem.inputSchema?.properties?.verbose?.type).toBe("boolean");
    expect(getItem.inputSchema?.properties?.verbose?.description).toBe(
      "Include details",
    );
    expect(getItem.inputSchema?.required).toEqual(["id"]);

    const createOrder = tools.find((t) => t.name === "erp__createorder")!;
    expect(createOrder.inputSchema?.properties?.body?.type).toBe("object");
    expect(createOrder.inputSchema?.required).toEqual(["body"]);
  });

  test("tools/call executes GET with path + query + gateway-held auth", async () => {
    if (!etcdReachable) return;
    await initialize(KEY_FULL);
    const r = await callTool(KEY_FULL, "erp__getitem", {
      id: 42,
      verbose: true,
    });
    expect(r.status).toBe(200);
    expect(r.isError).not.toBe(true);
    const echoed = JSON.parse(r.text ?? "{}");
    // The REST server 401s without the gateway-held bearer, so a successful
    // echo proves the credential was injected gateway-side.
    expect(echoed.id).toBe("42");
    expect(echoed.query.verbose).toBe("true");
  });

  test("tools/call executes POST with the body argument as JSON body", async () => {
    if (!etcdReachable) return;
    await initialize(KEY_FULL);
    const r = await callTool(KEY_FULL, "erp__createorder", {
      body: { note: "from-e2e" },
    });
    expect(r.status).toBe(200);
    expect(r.isError).not.toBe(true);
    const echoed = JSON.parse(r.text ?? "{}");
    expect(echoed.created.note).toBe("from-e2e");
  });

  test("non-2xx REST response surfaces as a tool-level error result", async () => {
    if (!etcdReachable) return;
    await initialize(KEY_FULL);
    const r = await callTool(KEY_FULL, "erp__failop", {});
    expect(r.status).toBe(200);
    expect(r.isError).toBe(true);
    expect(r.text).toContain("HTTP 500");
    expect(r.text).toContain("boom");
  });

  test("missing required path parameter is a readable tool-level error", async () => {
    if (!etcdReachable) return;
    await initialize(KEY_FULL);
    const r = await callTool(KEY_FULL, "erp__getitem", { verbose: true });
    expect(r.status).toBe(200);
    expect(r.isError).toBe(true);
    expect(r.text).toContain("missing required path parameter 'id'");
  });

  test("per-tool ACL applies to generated tools", async () => {
    if (!etcdReachable) return;
    const scoped = await listTools(KEY_SCOPED);
    expect(scoped.map((t) => t.name)).toEqual(["erp__getitem"]);

    await initialize(KEY_SCOPED);
    const denied = await callTool(KEY_SCOPED, "erp__createorder", {
      body: { note: "nope" },
    });
    expect(denied.rpcError ?? "").toContain("not available");

    const allowed = await callTool(KEY_SCOPED, "erp__getitem", { id: 7 });
    expect(allowed.isError).not.toBe(true);
    expect(JSON.parse(allowed.text ?? "{}").id).toBe("7");
  });
});
