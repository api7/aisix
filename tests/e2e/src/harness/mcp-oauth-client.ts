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
  invalidateCredentials(scope: "all" | "client" | "tokens" | "verifier" | "discovery") {
    if (scope === "all" || scope === "tokens") this.saved = undefined;
    if (scope === "all" || scope === "client") this.info = undefined;
  }
}

/** Where a walk through the browser pass stopped. */
export type BrowserStop =
  | { kind: "landed"; url: URL }
  | { kind: "page"; url: string; status: number; text: string };

/**
 * A browser with a cookie jar, honouring what a browser honours of a cookie:
 * its path and its lifetime. A consent page is clicked through unless the walk
 * is told to stop at it.
 */
export class Browser {
  cookies = new Map<string, { value: string; path: string }>();
  /** Every `Set-Cookie` received, verbatim. */
  setCookies: string[] = [];
  /** Every consent page shown, verbatim. */
  consentPages: string[] = [];
  /** Every URL requested, in order. */
  visited: string[] = [];

  private cookieHeader(url: URL): Record<string, string> {
    const matching = [...this.cookies]
      .filter(([, cookie]) => url.pathname === cookie.path || url.pathname.startsWith(`${cookie.path.replace(/\/$/, "")}/`))
      .map(([name, cookie]) => `${name}=${cookie.value}`);
    return matching.length === 0 ? {} : { cookie: matching.join("; ") };
  }

  private remember(res: Response): void {
    for (const raw of res.headers.getSetCookie()) {
      this.setCookies.push(raw);
      const [pair, ...attributes] = raw.split(";").map((part) => part.trim());
      const separator = pair.indexOf("=");
      const name = pair.slice(0, separator);
      const attribute = (key: string) =>
        attributes.find((a) => a.toLowerCase().startsWith(`${key}=`))?.slice(key.length + 1);
      if (attribute("max-age") === "0") this.cookies.delete(name);
      else this.cookies.set(name, { value: pair.slice(separator + 1), path: attribute("path") ?? "/" });
    }
  }

  async request(url: string, init: RequestInit = {}): Promise<Response> {
    this.visited.push(url);
    const res = await fetch(url, {
      redirect: "manual",
      ...init,
      headers: { ...(init.headers as Record<string, string>), ...this.cookieHeader(new URL(url)) },
    });
    this.remember(res);
    return res;
  }

  /** Follow redirects until the browser lands on `landing` or a page. */
  async walk(start: URL | string, options: { landing?: string; clickAllow?: boolean } = {}): Promise<BrowserStop> {
    const landing = options.landing ?? OAUTH_CLIENT_REDIRECT;
    const clickAllow = options.clickAllow ?? true;
    let next = start.toString();
    let res: Response | undefined;
    for (let hop = 0; hop < 12; hop++) {
      if (next.startsWith(landing)) return { kind: "landed", url: new URL(next) };
      res = await this.request(next);
      let location = res.headers.get("location");
      if (location === null) {
        const text = await res.text();
        if (!isConsentPage(text)) return { kind: "page", url: next, status: res.status, text };
        this.consentPages.push(text);
        if (!clickAllow) return { kind: "page", url: next, status: res.status, text };
        res = await this.allow(text);
        location = res.headers.get("location");
        if (location === null) {
          return { kind: "page", url: next, status: res.status, text: await res.text() };
        }
      }
      next = new URL(location, next).toString();
    }
    throw new Error("too many redirects");
  }

  /** Submit a consent page's form, as its `Allow` button does. */
  allow(page: string): Promise<Response> {
    return this.submitConsent(page, "allow");
  }

  /** Submit a consent page's form, as the button carrying `decision` does. */
  submitConsent(page: string, decision: "allow" | "cancel"): Promise<Response> {
    const form = consentForm(page);
    if (!page.includes(`<button type="submit" name="decision" value="${decision}">`)) {
      throw new Error(`no ${decision} button in: ${page}`);
    }
    return this.request(form.action, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ ...form.fields, decision }).toString(),
    });
  }
}

const isConsentPage = (text: string) => /<form[^>]*>/.test(text) && /<button[^>]*>\s*Allow\s*<\/button>/.test(text);

/** The action and hidden fields of a consent page's form. */
export function consentForm(page: string): { action: string; fields: Record<string, string> } {
  const action = /<form[^>]*action="([^"]+)"/.exec(page)?.[1];
  if (action === undefined) throw new Error(`no form in: ${page}`);
  const fields: Record<string, string> = {};
  for (const match of page.matchAll(/<input type="hidden" name="([^"]+)" value="([^"]*)"/g)) {
    fields[match[1]] = match[2];
  }
  return { action, fields };
}

/** Follow redirects as a browser would, until one lands on the client. */
export async function followBrowser(
  start: URL | string,
  landing = OAUTH_CLIENT_REDIRECT,
  browser = new Browser(),
): Promise<URL> {
  const stop = await browser.walk(start, { landing });
  if (stop.kind === "page") {
    throw new Error(`browser stopped at ${stop.url}: HTTP ${stop.status} ${stop.text}`);
  }
  return stop.url;
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
  browser = new Browser(),
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
  const landed = await followBrowser(state.browserOpens.at(-1)!, OAUTH_CLIENT_REDIRECT, browser);
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
