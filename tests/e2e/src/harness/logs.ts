import type { SpawnedApp } from "./app.js";

/**
 * The COMPLETE lines of the gateway's captured output.
 *
 * `output()` concatenates the raw chunks the child's pipe delivered, with
 * no line framing, so its tail is routinely half a line — and a predicate
 * anchored on a field the formatter writes early will match that prefix
 * and hand the spec a line whose later fields are simply missing. Every
 * log line ends in a newline, so anything after the last one is a
 * fragment and is dropped until the rest of it arrives.
 */
function completeLines(app: SpawnedApp): string[] {
  const lines = app.output().split("\n");
  lines.pop();
  return lines;
}

/**
 * Poll the gateway's captured output for a line satisfying `pred`.
 *
 * **A log line is never readable at the moment the request that produced
 * it returns.** The gateway hands every event to a bounded queue drained
 * by a dedicated writer thread, so the request path never blocks on the
 * log descriptor; the line is then written to a pipe that the harness
 * drains from its own event loop. Both hops are asynchronous, and neither
 * is ordered against the HTTP response — a bare `app.output()` read after
 * an assertion-worthy request therefore passes only because the machine
 * happened to be idle.
 *
 * Use this for every line the gateway writes as a consequence of
 * something the spec just did. A line written before the gateway was
 * ready (a boot warning, a rejected configuration at startup) is already
 * in `output()` by the time `spawnApp` resolves and needs no wait.
 *
 * Throws with the whole captured output, which is the only diagnostic a
 * spec has when the line never arrives.
 */
export async function waitForLogLine(
  app: SpawnedApp,
  pred: (line: string) => boolean,
  what: string,
  timeoutMs = 5_000,
): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const hit = completeLines(app).find(pred);
    if (hit) return hit;
    if (Date.now() >= deadline) break;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(
    `timed out waiting for ${what}; gateway output was:\n${app.output()}`,
  );
}

/**
 * As [`waitForLogLine`], for a spec that needs every matching line rather
 * than the first: waits until at least `count` of them have arrived.
 */
export async function waitForLogLines(
  app: SpawnedApp,
  pred: (line: string) => boolean,
  count: number,
  what: string,
  timeoutMs = 5_000,
): Promise<string[]> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const hits = completeLines(app).filter(pred);
    if (hits.length >= count) return hits;
    if (Date.now() >= deadline) break;
    await new Promise((r) => setTimeout(r, 50));
  }
  throw new Error(
    `timed out waiting for ${count} × ${what}; gateway output was:\n${app.output()}`,
  );
}
