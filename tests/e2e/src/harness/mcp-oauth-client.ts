import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import type { OAuthClientProvider } from "@modelcontextprotocol/sdk/client/auth.js";
import { UnauthorizedError } from "@modelcontextprotocol/sdk/client/auth.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type {
  OAuthClientInformationMixed,
  OAuthClientMetadata,
  OAuthTokens,
} from "@modelcontextprotocol/sdk/shared/auth.js";

export const OAUTH_CLIENT_REDIRECT = "http://127.0.0.1:53682/callback";

/** What an MCP client persists: here, for the length of a test. */
export class ClientAuthState implements OAuthClientProvider {
  redirectUrl = OAUTH_CLIENT_REDIRECT;
  clientMetadata: OAuthClientMetadata = {
    redirect_uris: [OAUTH_CLIENT_REDIRECT],
    client_name: "e2e",
    token_endpoint_auth_method: "none",
    grant_types: ["authorization_code"],
    response_types: ["code"],
  };
  info?: OAuthClientInformationMixed;
  saved?: OAuthTokens;
  verifier = "";
  /** Where the client would have opened a browser, in order. */
  browserOpens: URL[] = [];

  clientInformation() {
    return this.info;
  }
  saveClientInformation(info: OAuthClientInformationMixed) {
    this.info = info;
  }
  tokens() {
    return this.saved;
  }
  saveTokens(tokens: OAuthTokens) {
    this.saved = tokens;
  }
  redirectToAuthorization(url: URL) {
    this.browserOpens.push(url);
  }
  saveCodeVerifier(verifier: string) {
    this.verifier = verifier;
  }
  codeVerifier() {
    return this.verifier;
  }
}

/** Follow redirects as a browser would, until one lands on the client. */
export async function followBrowser(start: URL | string, landing = OAUTH_CLIENT_REDIRECT): Promise<URL> {
  let next = start.toString();
  for (let hop = 0; hop < 6; hop++) {
    if (next.startsWith(landing)) return new URL(next);
    const res = await fetch(next, { redirect: "manual" });
    const location = res.headers.get("location");
    if (location === null) {
      throw new Error(`browser stopped at ${next}: HTTP ${res.status} ${await res.text()}`);
    }
    next = new URL(location, next).toString();
  }
  throw new Error("too many redirects");
}

export async function connectMcpClient(
  url: string,
  headers: Record<string, string>,
  authProvider?: OAuthClientProvider,
): Promise<{ client: Client; transport: StreamableHTTPClientTransport }> {
  const transport = new StreamableHTTPClientTransport(new URL(url), {
    authProvider,
    requestInit: { headers },
  });
  const client = new Client({ name: "mcp-user-oauth-e2e", version: "0.1" }, { capabilities: {} });
  await client.connect(transport);
  return { client, transport };
}

/**
 * The whole sign-in as a client runs it: the gateway challenges, the client
 * opens a browser, and the code comes back to it.
 */
export async function signInMcpClient(
  url: string,
  headers: Record<string, string>,
  state = new ClientAuthState(),
): Promise<{ state: ClientAuthState; client: Client }> {
  const transport = new StreamableHTTPClientTransport(new URL(url), {
    authProvider: state,
    requestInit: { headers },
  });
  const client = new Client({ name: "mcp-user-oauth-e2e", version: "0.1" }, { capabilities: {} });
  const refused = await client.connect(transport).then(
    () => undefined,
    (error: unknown) => error,
  );
  if (!(refused instanceof UnauthorizedError)) {
    throw new Error(`expected a sign-in challenge, got: ${String(refused)}`);
  }
  const landed = await followBrowser(state.browserOpens.at(-1)!);
  const code = landed.searchParams.get("code");
  if (code === null) throw new Error(`sign-in failed: ${landed}`);
  await transport.finishAuth(code);
  await client.close();
  return { state, client: (await connectMcpClient(url, headers, state)).client };
}

export const whoami = async (client: Client): Promise<string> => {
  const result = await client.callTool({ name: "whoami", arguments: {} });
  return (result.content as { text: string }[])[0].text;
};
