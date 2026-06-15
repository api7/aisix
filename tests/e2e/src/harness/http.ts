import { Agent, request as undiciRequest, type Dispatcher } from "undici";

/**
 * Build a fresh undici Agent for each request. We deliberately do NOT
 * cache a module-level agent: a stuck/aborted ECONNREFUSED on the
 * shared agent has been observed to poison subsequent polls during
 * waitForReady (the agent caches the failure even after the listener
 * comes up). One agent per request is cheap for the volumes involved
 * and avoids the failure mode.
 *
 * The custom agent also bypasses any `HTTP_PROXY`/`ALL_PROXY` env vars
 * that local dev tools (ClashX etc.) may set.
 */
function freshAgent(): Agent {
  return new Agent({ connectTimeout: 2000, keepAliveTimeout: 1000 });
}

export type HttpMethod = Dispatcher.HttpMethod;

const METHODS = new Set<HttpMethod>([
  "GET",
  "POST",
  "PUT",
  "DELETE",
  "PATCH",
  "HEAD",
  "OPTIONS",
  "TRACE",
  "CONNECT",
]);

export function asMethod(m: string): HttpMethod {
  const upper = m.toUpperCase() as HttpMethod;
  if (!METHODS.has(upper)) throw new Error(`unsupported HTTP method: ${m}`);
  return upper;
}

export interface HarnessRequestOptions {
  method?: HttpMethod | string;
  headers?: Record<string, string>;
  body?: string | Buffer;
  signal?: AbortSignal;
}

export async function harnessRequest(
  url: string,
  opts: HarnessRequestOptions = {},
): Promise<Dispatcher.ResponseData> {
  return undiciRequest(url, {
    method: asMethod(opts.method ?? "GET"),
    headers: opts.headers,
    body: opts.body,
    signal: opts.signal,
    dispatcher: freshAgent(),
  });
}
