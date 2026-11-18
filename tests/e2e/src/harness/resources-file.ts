import { createHash } from "node:crypto";

/**
 * A stable UUID for a resources-file entry. Every entry carries the id its
 * etcd key would, and references name it by that id, so a spec derives the
 * same id on both sides from one seed (`fileId("pk")` in the entry and in
 * every `provider_key_id` that points at it).
 */
export function fileId(seed: string): string {
  const h = createHash("sha256").update(seed).digest("hex");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-4${h.slice(13, 16)}-8${h.slice(17, 20)}-${h.slice(20, 32)}`;
}

/** The `key_hash` an api key document stores for `plaintext`. */
export function keyHash(plaintext: string): string {
  return createHash("sha256").update(plaintext).digest("hex");
}

/**
 * What `/status/config` says the gateway dropped: a row the loader
 * rejected, or one of a kind it does not know. A file-mode gateway boots
 * past both, so a spec loading a file it expects to load whole asserts
 * this is empty.
 */
export async function droppedRows(app: { metricsUrl: string }): Promise<{
  rejected: unknown[];
  unknown_kinds: unknown[];
}> {
  const res = await fetch(`${app.metricsUrl}/status/config`);
  const body = (await res.json()) as { rejected?: unknown[]; unknown_kinds?: unknown[] };
  return { rejected: body.rejected ?? [], unknown_kinds: body.unknown_kinds ?? [] };
}
