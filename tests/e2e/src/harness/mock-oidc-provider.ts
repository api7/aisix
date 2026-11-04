import { createHash, createSign, generateKeyPairSync, randomBytes, type KeyObject } from "node:crypto";
import { createServer, type Server } from "node:http";

import { pickFreePort } from "./ports.js";

export interface MockOidcProvider {
  /** Also the issuer. */
  url: string;
  clientId: string;
  clientSecret: string | undefined;
  /** The claims of whoever signs in next; `sub` at the least. */
  claims: Record<string, unknown>;
  /** `aud` of the next ID token, when it should not be the client's id. */
  audience: string | undefined;
  /** `nonce` of the next ID token, when it should not be the request's. */
  nonce: string | undefined;
  /** `scope` of the next token response; the requested one by default, `null` to omit it. */
  grantedScope: string | null | undefined;
  /** Sign ID tokens with a key the JWKS does not publish. */
  signWithForeignKey: boolean;
  /** Query of every authorization request, in order. */
  authorizeRequests: Record<string, string>[];
  /** Decoded form of every token request, in order. */
  tokenRequests: Record<string, string>[];
  close(): Promise<void>;
}

const b64u = (data: Buffer | string) => Buffer.from(data).toString("base64url");

function sign(privateKey: KeyObject, claims: Record<string, unknown>): string {
  const head = b64u(JSON.stringify({ alg: "RS256", kid: "login-key", typ: "JWT" }));
  const payload = b64u(JSON.stringify(claims));
  const signature = createSign("RSA-SHA256").update(`${head}.${payload}`).sign(privateKey);
  return `${head}.${payload}.${b64u(signature)}`;
}

/**
 * An OIDC provider in one process, enforcing what a real one would: its
 * registered client, PKCE, a single-use code bound to the redirect URI, and the
 * client secret in the POST body. The user is whoever `claims` says: there is
 * no login form, as there is none after a real provider's own session.
 */
export async function startMockOidcProvider(
  options: { clientId?: string; clientSecret?: string } = {},
): Promise<MockOidcProvider> {
  const { privateKey, publicKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
  const foreign = generateKeyPairSync("rsa", { modulusLength: 2048 });
  const jwk = { ...publicKey.export({ format: "jwk" }), kid: "login-key", use: "sig", alg: "RS256" };
  const port = await pickFreePort();
  const url = `http://127.0.0.1:${port}`;
  const codes = new Map<string, { query: Record<string, string>; claims: Record<string, unknown> }>();

  const provider: MockOidcProvider = {
    url,
    clientId: options.clientId ?? "gateway-login",
    clientSecret: options.clientSecret,
    claims: { sub: "alice" },
    audience: undefined,
    nonce: undefined,
    grantedScope: undefined,
    signWithForeignKey: false,
    authorizeRequests: [],
    tokenRequests: [],
    close: async () => {},
  };

  const server: Server = createServer((req, res) => {
    const parsed = new URL(req.url ?? "/", url);
    const json = (status: number, body: unknown) =>
      res.writeHead(status, { "content-type": "application/json" }).end(JSON.stringify(body));
    if (parsed.pathname === "/.well-known/openid-configuration") {
      return void json(200, {
        issuer: url,
        authorization_endpoint: `${url}/authorize`,
        token_endpoint: `${url}/token`,
        jwks_uri: `${url}/jwks`,
      });
    }
    if (parsed.pathname === "/jwks") return void json(200, { keys: [jwk] });
    if (parsed.pathname === "/authorize") {
      const query = Object.fromEntries(parsed.searchParams);
      provider.authorizeRequests.push(query);
      if (
        query.client_id !== provider.clientId ||
        query.response_type !== "code" ||
        query.code_challenge_method !== "S256" ||
        !query.code_challenge ||
        !query.nonce ||
        !(query.scope ?? "").split(" ").includes("openid")
      ) {
        return void json(400, { error: "invalid_request" });
      }
      const code = `login-code-${randomBytes(8).toString("hex")}`;
      codes.set(code, { query, claims: { ...provider.claims } });
      const back = new URL(query.redirect_uri);
      back.searchParams.set("code", code);
      back.searchParams.set("state", query.state);
      return void res.writeHead(302, { location: back.toString() }).end();
    }
    if (parsed.pathname === "/token" && req.method === "POST") {
      const chunks: Buffer[] = [];
      req.on("data", (chunk: Buffer) => chunks.push(chunk));
      req.on("end", () => {
        const form = Object.fromEntries(new URLSearchParams(Buffer.concat(chunks).toString("utf8")));
        provider.tokenRequests.push(form);
        if (form.client_id !== provider.clientId || form.client_secret !== provider.clientSecret) {
          return void json(401, { error: "invalid_client" });
        }
        const issued = codes.get(form.code);
        codes.delete(form.code);
        const verified =
          issued !== undefined &&
          issued.query.redirect_uri === form.redirect_uri &&
          createHash("sha256").update(form.code_verifier ?? "").digest("base64url") === issued.query.code_challenge;
        if (!verified) return void json(400, { error: "invalid_grant" });
        const now = Math.floor(Date.now() / 1000);
        const idToken = sign(provider.signWithForeignKey ? foreign.privateKey : privateKey, {
          iss: url,
          aud: provider.audience ?? provider.clientId,
          exp: now + 300,
          iat: now,
          nonce: provider.nonce ?? issued.query.nonce,
          ...issued.claims,
        });
        const scope = provider.grantedScope === undefined ? issued.query.scope : provider.grantedScope;
        json(200, {
          access_token: `provider-at-${randomBytes(8).toString("hex")}`,
          token_type: "Bearer",
          expires_in: 300,
          id_token: idToken,
          ...(scope === null ? {} : { scope }),
        });
      });
      return;
    }
    res.writeHead(404).end();
  });
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  provider.close = () => {
    server.closeAllConnections();
    return new Promise<void>((resolve) => server.close(() => resolve()));
  };
  return provider;
}
