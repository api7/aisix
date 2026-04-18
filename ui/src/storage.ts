// Minimal localStorage wrapper for the connection settings the UI uses
// to talk to the admin API. Kept tiny on purpose — there's no shared
// state library because there's nothing else to share yet.

const STORAGE_KEY = "aisix.connection";

export interface Connection {
  endpoint: string; // e.g. "http://127.0.0.1:3001"
  adminKey: string;
}

export const DEFAULT_CONNECTION: Connection = {
  endpoint: typeof window !== "undefined" ? window.location.origin : "",
  adminKey: "",
};

export function loadConnection(): Connection {
  if (typeof window === "undefined") {
    return DEFAULT_CONNECTION;
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return DEFAULT_CONNECTION;
    }
    const parsed = JSON.parse(raw) as Partial<Connection>;
    return {
      endpoint: parsed.endpoint ?? DEFAULT_CONNECTION.endpoint,
      adminKey: parsed.adminKey ?? "",
    };
  } catch {
    return DEFAULT_CONNECTION;
  }
}

export function saveConnection(c: Connection): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(c));
}

export function clearConnection(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(STORAGE_KEY);
}

export function isConfigured(c: Connection): boolean {
  return c.endpoint.trim().length > 0 && c.adminKey.trim().length > 0;
}
