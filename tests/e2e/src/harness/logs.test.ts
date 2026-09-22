import { expect, test } from "vitest";
import type { SpawnedApp } from "./app.js";
import { waitForLogLine, waitForLogLines } from "./logs.js";

/**
 * A gateway whose output gains `line` only after `afterMs` — the shape a
 * real one always has, because the log queue, the writer thread and the
 * harness's own pipe drain all run after the HTTP response.
 */
function lateOutput(before: string, line: string, afterMs: number): SpawnedApp {
  const at = Date.now() + afterMs;
  return {
    output: () => (Date.now() >= at ? `${before}\n${line}` : before),
  } as unknown as SpawnedApp;
}

test("a line that has not been written yet is waited for, not missed", async () => {
  const app = lateOutput("boot line", 'proxy request completed status=200', 300);
  // The read a spec would do straight after its request sees nothing.
  expect(app.output().includes("status=200")).toBe(false);
  const hit = await waitForLogLine(
    app,
    (l) => l.includes("status=200"),
    "the completion line",
  );
  expect(hit).toContain("status=200");
});

test("a line already in the output is returned without waiting", async () => {
  const app = lateOutput("proxy request completed status=200", "later", 60_000);
  const started = Date.now();
  await waitForLogLine(app, (l) => l.includes("status=200"), "the completion line");
  expect(Date.now() - started).toBeLessThan(50);
});

test("the timeout message carries the output, which is the only diagnostic", async () => {
  const app = lateOutput("boot line", "never", 60_000);
  await expect(
    waitForLogLine(app, (l) => l.includes("status=200"), "the completion line", 150),
  ).rejects.toThrow(/timed out waiting for the completion line[\s\S]*boot line/);
});

test("waiting for several lines does not settle for the first one", async () => {
  const app = lateOutput("hit one", "hit two", 300);
  await expect(
    waitForLogLines(app, (l) => l.startsWith("hit"), 2, "a hit", 100),
  ).rejects.toThrow(/timed out waiting for 2/);
  expect(await waitForLogLines(app, (l) => l.startsWith("hit"), 2, "a hit")).toEqual([
    "hit one",
    "hit two",
  ]);
});
