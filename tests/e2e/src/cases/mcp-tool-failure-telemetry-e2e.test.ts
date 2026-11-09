import { createHash, randomUUID } from "node:crypto";
import { createServer as createTcpServer } from "node:net";
import { afterAll, beforeAll, describe, expect, test } from "vitest";
import {
  EtcdClient,
  SeedClient,
  scrapeMetrics,
  slsLogsFor,
  spawnApp,
  startMcpUpstream,
  startMockSls,
  startRestUpstream,
  waitConfigPropagation,
  waitForLogLine,
  waitForSlsLog,
  type McpUpstream,
  type MockSls,
  type RestUpstream,
  type SpawnedApp,
} from "../harness/index.js";
import { startMockOtlp, type CapturedSpan, type MockOtlp } from "../harness/otlp-mock.js";

// A tool call can fail behind a `200`: the tool says so with `isError`, or the
// gateway turns an upstream failure into a JSON-RPC error. The caller's answer
// stays as it was. The telemetry names the failure with fixed text, marks the
// tool span as failed, and carries what the tool said only where the exporter's
// content mode and the output guardrails allow it.

const CALLER = "sk-mcp-tool-failure";
const RESTRICTED = "sk-mcp-tool-failure-restricted";
const CREDENTIAL_REF = "mock";
const LOGSTORE = "tool-failure";
const hash = (s: string) => createHash("sha256").update(s).digest("hex");

/** What a failing tool or upstream said. It must never reach `error_message`. */
const SENTINEL = "sentinel-tool-output-7f3a";
const EMAIL = "alice@example.com";
const TOOL_TEXT = `${SENTINEL} for ${EMAIL}`;

interface RpcAnswer {
  status: number;
  requestId: string;
  body: {
    result?: { isError?: boolean; content?: Array<{ text?: string }>; tools?: unknown[] };
    error?: { code: number; message: string };
  };
}

interface Failure {
  errorClass: string;
  errorMessage: string;
}

describe.each(["full", "metadata_only"] as const)(
  "tool call failures under content_mode=%s",
  (mode) => {
    let etcdReachable = false;
    let app: SpawnedApp | undefined;
    let sls: MockSls | undefined;
    let otlp: MockOtlp | undefined;
    let faulty: McpUpstream | undefined;
    let rest: RestUpstream | undefined;

    beforeAll(async () => {
      const etcd = new EtcdClient();
      etcdReachable = await etcd.ping();
      if (!etcdReachable) return;
      sls = await startMockSls();
      otlp = await startMockOtlp();
      rest = await startRestUpstream();
      faulty = await startMcpUpstream("faulty", {
        mediaResult: { content: [{ type: "text", text: TOOL_TEXT }] },
        toolFaults: {
          http500: { kind: "http", status: 500, body: TOOL_TEXT },
          rpcerror: { kind: "jsonrpc", code: -32000, message: TOOL_TEXT },
          hangs: { kind: "hang" },
        },
      });
      app = await spawnApp({
        extraEnv: {
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_ID`]: "mock-akid",
          [`SLS_CRED_${CREDENTIAL_REF.toUpperCase()}_AK_SECRET`]: "mock-secret",
        },
      });
      const seed = new SeedClient(etcd, app.etcdPrefix);
      await seed.createObservabilityExporter({
        name: "tool-failure-sls",
        enabled: true,
        kind: "aliyun_sls",
        endpoint: sls.url,
        project: "tool-failure",
        logstore: LOGSTORE,
        credential_ref: CREDENTIAL_REF,
        content_mode: mode,
      });
      await seed.createObservabilityExporter({
        name: "tool-failure-otlp",
        enabled: true,
        kind: "otlp_http",
        endpoint: otlp.url,
        content_mode: mode,
      });
      await seed.createGuardrail({
        name: "tool-failure-mask",
        enabled: true,
        hook_point: "output",
        kind: "pii",
        detectors: [{ type: "email", action: "mask" }],
      });
      await seed.update("mcp_servers", randomUUID(), {
        display_name: "faulty",
        url: faulty.url,
        enabled: true,
      });
      await seed.update("mcp_servers", randomUUID(), {
        display_name: "slow",
        url: faulty.url,
        enabled: true,
        timeout_ms: 500,
      });
      const closedPort = await unusedPort();
      await seed.update("mcp_servers", randomUUID(), {
        display_name: "gone",
        url: `http://127.0.0.1:${closedPort}/mcp`,
        enabled: true,
      });
      await seed.update("mcp_servers", randomUUID(), {
        name: "erp",
        type: "openapi",
        url: rest.baseUrl,
        enabled: true,
        spec: {
          openapi: "3.0.0",
          info: { title: "ERP", version: "1.0.0" },
          paths: { "/fail": { get: { operationId: "failAlways" } } },
        },
      });
      await seed.createApiKey({
        key_hash: hash(RESTRICTED),
        allowed_models: ["*"],
        mcp_access: { allow: ["faulty__echo"] },
      });
      await seed.createApiKey({
        key_hash: hash(CALLER),
        allowed_models: ["*"],
        mcp_access: { allow: ["*"] },
      });
      await waitConfigPropagation(async () => {
        const res = await fetch(`${app!.proxyUrl}/v1/models`, {
          headers: { authorization: `Bearer ${CALLER}` },
        });
        await res.arrayBuffer();
        return res.status === 200;
      });
    }, 120_000);

    afterAll(async () => {
      await app?.exit();
      await sls?.close();
      await otlp?.close();
      await faulty?.close();
      await rest?.close();
    });

    async function rpc(
      method: string,
      params: Record<string, unknown>,
      token = CALLER,
    ): Promise<RpcAnswer> {
      const res = await fetch(`${app!.proxyUrl}/mcp`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${token}`,
          "content-type": "application/json",
          accept: "application/json, text/event-stream",
        },
        body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
      });
      return {
        status: res.status,
        requestId: res.headers.get("x-aisix-request-id") ?? "",
        body: (await res.json()) as RpcAnswer["body"],
      };
    }

    const callTool = (name: string, args: Record<string, unknown> = {}, token = CALLER) =>
      rpc("tools/call", { name, arguments: args }, token);

    const rowOf = (requestId: string) =>
      waitForSlsLog(sls!, LOGSTORE, (l) => l.get("request_id") === requestId, `row ${requestId}`);

    async function spansOf(row: Map<string, string>): Promise<CapturedSpan[]> {
      const traceId = row.get("trace_id");
      expect(traceId, "the row names its trace").toBeTruthy();
      const deadline = Date.now() + 10_000;
      while (Date.now() < deadline) {
        const spans = otlp!.spans.filter((s) => s.traceId === traceId);
        if (spans.some((s) => s.attributes["gen_ai.operation.name"] === "execute_tool")) {
          return spans;
        }
        await new Promise((r) => setTimeout(r, 100));
      }
      throw new Error(`no execute_tool span in trace ${traceId}`);
    }

    /** The row and the tool span say the same thing about how the call ended. */
    async function expectTelemetry(answer: RpcAnswer, failure: Failure | null) {
      const row = await rowOf(answer.requestId);
      expect(row.get("status_code")).toBe("200");
      expect(row.get("error_class")).toBe(failure?.errorClass);
      expect(row.get("error_message")).toBe(failure?.errorMessage);

      const spans = await spansOf(row);
      const tool = spans.find((s) => s.attributes["gen_ai.operation.name"] === "execute_tool")!;
      expect(tool.kind, "the tool span is a CLIENT span").toBe(3);
      expect(tool.statusCode).toBe(failure ? 2 : 0);
      expect(tool.attributes["error.type"]).toBe(failure?.errorClass);
      expect(tool.attributes["aisix.error_class"]).toBe(failure?.errorClass);
      expect(tool.attributes["aisix.error_message"]).toBe(failure?.errorMessage);
      expect(tool.attributes["http.response.status_code"]).toBe(200);
      const server = spans.find((s) => s.kind === 2);
      expect(server, "the request has a SERVER span").toBeDefined();
      expect(server!.statusCode, "a 200 leaves the SERVER span unset").toBe(0);
      expect(server!.attributes["error.type"]).toBeUndefined();
      return { row, tool };
    }

    function expectJsonRpcFailure(answer: RpcAnswer, server: string) {
      expect(answer.status).toBe(200);
      expect(answer.body.result).toBeUndefined();
      expect(answer.body.error?.code).toBe(-32603);
      expect(answer.body.error?.message).toBe(
        `upstream MCP server '${server}' failed to call tool`,
      );
    }

    test("isError: the result reaches the caller and the row says the tool failed", async (ctx) => {
      if (!etcdReachable) return ctx.skip();
      const answer = await callTool("faulty__media", { failed: true });
      expect(answer.status).toBe(200);
      expect(answer.body.error).toBeUndefined();
      expect(answer.body.result?.isError).toBe(true);
      expect(answer.body.result?.content?.[0]?.text).toBe(
        `${SENTINEL} for [EMAIL_REDACTED]`,
      );

      const { row, tool } = await expectTelemetry(answer, {
        errorClass: "tool_error",
        errorMessage: "MCP tool returned isError",
      });

      const exported = JSON.stringify(Object.fromEntries(row)) + JSON.stringify(tool.attributes);
      expect(exported, "the output mask holds on a failure row").not.toContain(EMAIL);
      if (mode === "full") {
        expect(row.get("response")).toContain(SENTINEL);
        expect(row.get("response")).toContain("[EMAIL_REDACTED]");
        expect(String(tool.attributes["gen_ai.tool.call.result"])).toContain(SENTINEL);
      } else {
        expect(exported, "metadata_only exports nothing the tool said").not.toContain(SENTINEL);
        expect(row.get("response")).toBeUndefined();
        expect(tool.attributes["gen_ai.tool.call.result"]).toBeUndefined();
      }
    });

    test.for([
      [
        "upstream 500",
        "faulty__http500",
        "faulty",
        "upstream_status",
        "upstream MCP server returned HTTP 500",
      ],
      [
        "upstream timeout",
        "slow__hangs",
        "slow",
        "timeout",
        "upstream MCP server timed out",
      ],
      [
        "connection failure",
        "gone__echo",
        "gone",
        "transport",
        "upstream MCP server connection failed",
      ],
      [
        "JSON-RPC error inside a 200",
        "faulty__rpcerror",
        "faulty",
        "upstream_in_band",
        "upstream MCP server returned JSON-RPC error -32000",
      ],
    ] as const)("%s", async ([, tool, server, errorClass, errorMessage], ctx) => {
      if (!etcdReachable) return ctx.skip();
      const answer = await callTool(tool, { text: "hi" });
      expectJsonRpcFailure(answer, server);

      const { row, tool: span } = await expectTelemetry(answer, { errorClass, errorMessage });
      const exported = JSON.stringify(Object.fromEntries(row)) + JSON.stringify(span.attributes);
      expect(exported, "what the upstream said stays out").not.toContain(SENTINEL);
    });

    test("an openapi tool whose API answers 500 reports the status", async (ctx) => {
      if (!etcdReachable) return ctx.skip();
      const answer = await callTool("erp__failalways");
      expect(answer.status).toBe(200);
      expect(answer.body.result?.isError).toBe(true);
      await expectTelemetry(answer, {
        errorClass: "upstream_status",
        errorMessage: "upstream MCP server returned HTTP 500",
      });
    });

    test("a successful tools/call is not marked failed", async (ctx) => {
      if (!etcdReachable) return ctx.skip();
      const answer = await callTool("faulty__echo", { text: "hi" });
      expect(answer.status).toBe(200);
      expect(answer.body.result?.isError).toBe(false);
      expect(answer.body.result?.content?.[0]?.text).toBe("faulty:hi");
      await expectTelemetry(answer, null);
    });

    test("tools/list files no row and no failure", async (ctx) => {
      if (!etcdReachable) return ctx.skip();
      const answer = await rpc("tools/list", {});
      expect(answer.status).toBe(200);
      expect(answer.body.result?.tools?.length).toBeGreaterThan(0);
      // A later call's row proves the exporter has passed this request.
      await rowOf((await callTool("faulty__echo", { text: "barrier" })).requestId);
      expect(slsLogsFor(sls!, LOGSTORE).filter((l) => l.get("request_id") === answer.requestId)).toEqual([]);
    });

    test("an ACL refusal is the gateway's own error and carries no failure mark", async (ctx) => {
      if (!etcdReachable) return ctx.skip();
      const answer = await callTool("faulty__reverse", { text: "hi" }, RESTRICTED);
      expect(answer.status).toBe(200);
      expect(answer.body.error?.code).toBe(-32602);
      await expectTelemetry(answer, null);
    });

    test("the access log and the request counters name the failure as the upstream's", async (ctx) => {
      if (!etcdReachable) return ctx.skip();
      const answer = await callTool("faulty__media", { failed: true });
      await rowOf(answer.requestId);
      const line = await waitForLogLine(
        app!,
        (l) => l.includes(answer.requestId) && l.includes("status_code"),
        "the tool call's access-log line",
      );
      expect(line).toContain("tool_error");
      expect(line).toContain("MCP tool returned isError");
      expect(line).not.toContain(SENTINEL);

      const samples = await scrapeMetrics(app!.metricsUrl);
      const upstreamFailures = samples.filter(
        (s) =>
          s.labels.provider === "mcp" &&
          s.labels.status === "200" &&
          s.labels.error_source === "upstream",
      );
      expect(upstreamFailures.length, "a counter names the upstream").toBeGreaterThan(0);
    });
  },
);

async function unusedPort(): Promise<number> {
  const server = createTcpServer();
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const port = (server.address() as { port: number }).port;
  await new Promise((r) => server.close(r));
  return port;
}
