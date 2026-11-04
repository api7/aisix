import { createHash, randomBytes } from "node:crypto";
import { createServer, type IncomingMessage, type Server as HttpServer, type ServerResponse } from "node:http";

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

export interface OAuthMcpUpstreamOptions {
  /** Publish RFC 9728 / RFC 8414 metadata. */
  discovery?: boolean;
  /** Accept RFC 7591 registration of public clients. */
  registration?: boolean;
  /** `expires_in` of issued access tokens, `undefined` to omit it. */
  expiresIn?: number;
  /** A configured confidential client this server accepts. */
  staticClient?: { id: string; secret: string };
  /** Issue no refresh tokens at all. */
  noRefreshTokens?: boolean;
  /**
   * A refresh ends the access token it replaces as soon as it arrives, and an MCP
   * request with a token this server does not honour is served as anonymous
   * rather than refused.
   */
  anonymousOnDeadToken?: boolean;
  /** Serve `/authorize` and `/token` under this path, as Jira Data Center does. */
  oauthPathPrefix?: string;
  /** What the metadata documents name, instead of this server's own address. */
  advertise?: { authorizationServer?: string; tokenEndpoint?: string; registrationEndpoint?: string };
  /** Answer token requests with a redirect to this address. */
  tokenRedirect?: string;
}

interface IssuedCode {
  clientId: string;
  redirectUri: string;
  challenge: string;
  user: string;
}

/**
 * An upstream authorization server and MCP server in one process, for the
 * per-user OAuth flow. It enforces what a real one would: PKCE, single-use
 * codes, a refresh token that rotates so the old one stops working, and a
 * `401` for any token it did not issue or has revoked. `tools/call whoami`
 * answers with the signed-in user, so a spec can see whose token arrived.
 */
export class OAuthMcpUpstream {
  url = "";
  /** The user the next sign-in authenticates as. */
  user = "alice";
  options: OAuthMcpUpstreamOptions;
  /** `Authorization` header of every MCP request, in arrival order. */
  mcpAuthorizations: (string | undefined)[] = [];
  /** Decoded form of every token request, in arrival order. */
  tokenRequests: Record<string, string>[] = [];
  /** Query of every authorization request, in arrival order. */
  authorizeRequests: Record<string, string>[] = [];
  /** Body of every registration request, in arrival order. */
  registrations: Record<string, unknown>[] = [];
  /** Answer for a refresh grant: a status + `error`, or normal operation. */
  refreshAnswer: { status: number; error: string } | "ok" = "ok";
  /** Reject every access token, whatever it is. */
  rejectEveryToken = false;
  /** Method and path of every request, in arrival order. */
  requestsSeen: string[] = [];
  /** Query string of every token request: a credential belongs in the body. */
  tokenRequestQueries: string[] = [];
  /** Delay before answering a refresh grant, for overlapping requests. */
  refreshDelayMs = 0;

  private http: HttpServer | undefined;
  private codes = new Map<string, IssuedCode>();
  private accessTokens = new Map<string, string>();
  private refreshTokens = new Map<string, string>();
  private clients = new Set<string>();
  private counter = 0;

  constructor(options: OAuthMcpUpstreamOptions = {}) {
    this.options = { discovery: true, registration: true, expiresIn: 3600, ...options };
  }

  get mcpUrl(): string {
    return `${this.url}/mcp`;
  }

  /** The upstream revokes `token`, or every access token of `user`. */
  revokeAccessTokens(user: string): void {
    for (const [token, owner] of this.accessTokens) {
      if (owner === user) this.accessTokens.delete(token);
    }
  }

  /** Requests that carried a token this server honours as `user`'s. */
  tokensSeenFor(user: string): string[] {
    return this.mcpAuthorizations.filter((a): a is string => a?.startsWith(`Bearer at-${user}-`) ?? false);
  }

  refreshGrants(): Record<string, string>[] {
    return this.tokenRequests.filter((f) => f.grant_type === "refresh_token");
  }

  async start(): Promise<this> {
    this.http = createServer((req, res) => {
      void this.route(req, res).catch(() => {
        if (!res.headersSent) res.writeHead(500);
        res.end();
      });
    });
    await new Promise<void>((resolve) => this.http!.listen(0, "127.0.0.1", resolve));
    const address = this.http.address();
    if (address === null || typeof address === "string") throw new Error("no address");
    this.url = `http://127.0.0.1:${address.port}`;
    return this;
  }

  async close(): Promise<void> {
    this.http?.closeAllConnections();
    await new Promise<void>((resolve) => this.http?.close(() => resolve()));
  }

  private async route(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? "/", this.url);
    const json = (status: number, body: unknown) =>
      res.writeHead(status, { "content-type": "application/json" }).end(JSON.stringify(body));
    this.requestsSeen.push(`${req.method} ${url.pathname}`);
    const oauthPrefix = this.options.oauthPathPrefix ?? "";
    switch (`${req.method} ${url.pathname.startsWith(oauthPrefix) ? url.pathname.slice(oauthPrefix.length) : url.pathname}`) {
      case "GET /.well-known/oauth-protected-resource/mcp":
        if (!this.options.discovery) return void json(404, {});
        return void json(200, {
          resource: this.mcpUrl,
          authorization_servers: [this.options.advertise?.authorizationServer ?? this.url],
          bearer_methods_supported: ["header"],
        });
      case "GET /.well-known/oauth-authorization-server":
        if (!this.options.discovery) return void json(404, {});
        return void json(200, {
          issuer: this.url,
          authorization_endpoint: `${this.url}/authorize`,
          token_endpoint: this.options.advertise?.tokenEndpoint ?? `${this.url}/token`,
          ...(this.options.registration
            ? { registration_endpoint: this.options.advertise?.registrationEndpoint ?? `${this.url}/register` }
            : {}),
          response_types_supported: ["code"],
          code_challenge_methods_supported: ["S256"],
        });
      case "POST /register":
        return this.register(req, json);
      case "GET /authorize":
        return this.authorize(url, res, json);
      case "POST /token":
        this.tokenRequestQueries.push(url.search);
        if (this.options.tokenRedirect !== undefined) {
          return void res.writeHead(307, { location: this.options.tokenRedirect }).end();
        }
        return this.token(req, json);
      case "POST /mcp":
        return this.mcp(req, res);
      case "GET /mcp":
        return void res
          .writeHead(401, {
            "www-authenticate": `Bearer resource_metadata="${this.url}/.well-known/oauth-protected-resource/mcp"`,
          })
          .end();
      default:
        return void json(404, {});
    }
  }

  private async readBody(req: IncomingMessage): Promise<string> {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    return Buffer.concat(chunks).toString("utf8");
  }

  private async register(req: IncomingMessage, json: (s: number, b: unknown) => void): Promise<void> {
    const body = JSON.parse(await this.readBody(req)) as Record<string, unknown>;
    this.registrations.push(body);
    if (!this.options.registration || body.token_endpoint_auth_method !== "none") {
      return json(400, { error: "invalid_client_metadata" });
    }
    const clientId = `registered-${++this.counter}`;
    this.clients.add(clientId);
    json(201, { client_id: clientId, redirect_uris: body.redirect_uris });
  }

  private authorize(url: URL, res: ServerResponse, json: (s: number, b: unknown) => void): void {
    const q = Object.fromEntries(url.searchParams);
    this.authorizeRequests.push(q);
    const known = this.clients.has(q.client_id) || q.client_id === this.options.staticClient?.id;
    if (!known || q.response_type !== "code" || q.code_challenge_method !== "S256" || !q.code_challenge) {
      return json(400, { error: "invalid_request" });
    }
    const code = `upstream-code-${randomBytes(8).toString("hex")}`;
    this.codes.set(code, {
      clientId: q.client_id,
      redirectUri: q.redirect_uri,
      challenge: q.code_challenge,
      user: this.user,
    });
    const back = new URL(q.redirect_uri);
    back.searchParams.set("code", code);
    back.searchParams.set("state", q.state);
    res.writeHead(302, { location: back.toString() }).end();
  }

  private issue(user: string, json: (s: number, b: unknown) => void): void {
    const n = ++this.counter;
    const access = `at-${user}-${n}`;
    const refresh = `rt-${user}-${n}`;
    this.accessTokens.set(access, user);
    this.refreshTokens.set(refresh, user);
    json(200, {
      access_token: access,
      ...(this.options.noRefreshTokens ? {} : { refresh_token: refresh }),
      token_type: "Bearer",
      ...(this.options.expiresIn === undefined ? {} : { expires_in: this.options.expiresIn }),
    });
  }

  private async token(req: IncomingMessage, json: (s: number, b: unknown) => void): Promise<void> {
    const form = Object.fromEntries(new URLSearchParams(await this.readBody(req)));
    this.tokenRequests.push(form);
    const staticClient = this.options.staticClient;
    if (staticClient && form.client_id === staticClient.id && form.client_secret !== staticClient.secret) {
      return json(401, { error: "invalid_client" });
    }
    if (form.grant_type === "authorization_code") {
      const code = this.codes.get(form.code);
      this.codes.delete(form.code);
      const verified =
        code !== undefined &&
        code.clientId === form.client_id &&
        code.redirectUri === form.redirect_uri &&
        createHash("sha256").update(form.code_verifier ?? "").digest("base64url") === code.challenge;
      return verified ? this.issue(code.user, json) : json(400, { error: "invalid_grant" });
    }
    if (form.grant_type === "refresh_token") {
      if (this.options.anonymousOnDeadToken) this.revokeAccessTokens(this.refreshTokens.get(form.refresh_token) ?? "");
      if (this.refreshDelayMs > 0) await new Promise((r) => setTimeout(r, this.refreshDelayMs));
      if (this.refreshAnswer !== "ok") {
        return json(this.refreshAnswer.status, { error: this.refreshAnswer.error });
      }
      const user = this.refreshTokens.get(form.refresh_token);
      // Rotation: a refresh token works exactly once.
      this.refreshTokens.delete(form.refresh_token);
      return user === undefined ? json(400, { error: "invalid_grant" }) : this.issue(user, json);
    }
    json(400, { error: "unsupported_grant_type" });
  }

  private async mcp(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const raw = await this.readBody(req);
    const authorization = req.headers.authorization;
    this.mcpAuthorizations.push(authorization);
    const token = authorization?.replace(/^Bearer /, "");
    const honoured = token === undefined ? undefined : this.accessTokens.get(token);
    const user = honoured ?? (token !== undefined && this.options.anonymousOnDeadToken ? "anonymous" : undefined);
    if (this.rejectEveryToken || user === undefined) {
      res
        .writeHead(401, {
          "content-type": "application/json",
          "www-authenticate": this.options.discovery
            ? `Bearer resource_metadata="${this.url}/.well-known/oauth-protected-resource/mcp"`
            : "Bearer",
        })
        .end(JSON.stringify({ error: "invalid_token" }));
      return;
    }
    const server = new Server({ name: "oauth-upstream", version: "0.1.0" }, { capabilities: { tools: {} } });
    server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [{ name: "whoami", description: "the signed-in user", inputSchema: { type: "object", properties: {} } }],
    }));
    server.setRequestHandler(CallToolRequestSchema, async () => ({
      content: [{ type: "text", text: user }],
    }));
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
    await server.connect(transport);
    await transport.handleRequest(req, res, JSON.parse(raw));
  }
}
