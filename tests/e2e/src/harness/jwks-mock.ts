import { createServer, type Server } from "node:http";
import {
  createPrivateKey,
  createPublicKey,
  createSign,
  generateKeyPairSync,
  type KeyObject,
} from "node:crypto";

import { pickFreePort } from "./ports.js";

/**
 * Mock OIDC identity provider for JWT-auth e2e: serves a JWKS document
 * and an OIDC discovery document from a local HTTP server, and signs
 * RS256 tokens with the matching private keys.
 *
 * `rotate()` replaces the published key set with a freshly generated
 * key under a new `kid` — the mock-side half of the "key rotation
 * without a gateway restart" scenario. Old keys are dropped from the
 * JWKS (the strictest rotation), but their signer handles stay usable
 * so a test can still mint a token with a retired key.
 */
export interface MockIdp {
  /** Base URL, also the issuer unless a test overrides `iss`. */
  url: string;
  /** JWKS endpoint URL (`<url>/jwks`). */
  jwksUrl: string;
  /** `kid` of the currently published signing key. */
  currentKid: string;
  /** Number of JWKS fetches served, for cache-behavior assertions. */
  jwksFetches: number;
  /** Sign an RS256 JWT with the current key. */
  sign(claims: Record<string, unknown>, opts?: SignOpts): string;
  /** Publish a brand-new signing key under a new kid, dropping the old. */
  rotate(): void;
  close(): Promise<void>;
}

export interface SignOpts {
  /** Override the `kid` placed in the JOSE header (default: current key). */
  kid?: string;
  /** Sign with a specific retired key instead of the current one. */
  signWithKid?: string;
  /** Omit the `kid` header entirely. */
  omitKid?: boolean;
  /** Extra JOSE header fields (e.g. a different `alg` label). */
  header?: Record<string, unknown>;
}

interface KeyEntry {
  kid: string;
  privateKey: KeyObject;
  jwk: Record<string, unknown>;
}

function b64u(data: Buffer | string): string {
  return Buffer.from(data).toString("base64url");
}

function newKey(kid: string): KeyEntry {
  const { privateKey, publicKey } = generateKeyPairSync("rsa", {
    modulusLength: 2048,
  });
  const jwk = publicKey.export({ format: "jwk" }) as Record<string, unknown>;
  return {
    kid,
    privateKey,
    jwk: { ...jwk, kid, use: "sig", alg: "RS256" },
  };
}

export async function startMockIdp(): Promise<MockIdp> {
  const keys: KeyEntry[] = [newKey("kid-1")];
  const retired: KeyEntry[] = [];
  let generation = 1;

  const state = {
    jwksFetches: 0,
  };

  const port = await pickFreePort();
  const url = `http://127.0.0.1:${port}`;

  const server: Server = createServer((req, res) => {
    const path = (req.url ?? "").split("?")[0];
    if (path === "/jwks") {
      state.jwksFetches += 1;
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ keys: keys.map((k) => k.jwk) }));
      return;
    }
    if (path === "/.well-known/openid-configuration") {
      res.setHeader("content-type", "application/json");
      res.end(
        JSON.stringify({
          issuer: url,
          jwks_uri: `${url}/jwks`,
        }),
      );
      return;
    }
    res.statusCode = 404;
    res.end();
  });
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));

  function findSigner(opts?: SignOpts): KeyEntry {
    if (opts?.signWithKid) {
      const k = [...keys, ...retired].find((k) => k.kid === opts.signWithKid);
      if (!k) throw new Error(`no key ${opts.signWithKid}`);
      return k;
    }
    return keys[0];
  }

  return {
    url,
    jwksUrl: `${url}/jwks`,
    get currentKid() {
      return keys[0].kid;
    },
    get jwksFetches() {
      return state.jwksFetches;
    },
    sign(claims, opts) {
      const signer = findSigner(opts);
      const header: Record<string, unknown> = {
        alg: "RS256",
        typ: "JWT",
        ...(opts?.omitKid ? {} : { kid: opts?.kid ?? signer.kid }),
        ...opts?.header,
      };
      const signingInput = `${b64u(JSON.stringify(header))}.${b64u(
        JSON.stringify(claims),
      )}`;
      const sig = createSign("RSA-SHA256")
        .update(signingInput)
        .sign(createPrivateKey(signer.privateKey.export({ format: "pem", type: "pkcs8" })));
      return `${signingInput}.${sig.toString("base64url")}`;
    },
    rotate() {
      generation += 1;
      retired.push(...keys.splice(0));
      keys.push(newKey(`kid-${generation}`));
    },
    async close() {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    },
  };
}

/** Standard claim set for a mock agent token, expiring in one hour. */
export function agentClaims(
  issuer: string,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    iss: issuer,
    aud: "aisix-gateway",
    sub: "agent-1",
    exp: Math.floor(Date.now() / 1000) + 3600,
    ...overrides,
  };
}

// Re-exported so tests can build unrelated public keys (e.g. a second
// IdP whose keys must NOT verify against the first one's JWKS).
export { createPublicKey };
