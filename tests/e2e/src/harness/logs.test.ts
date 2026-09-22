import { expect, test } from "vitest";
import type { SpawnedApp } from "./app.js";
import { waitForLogLine, waitForLogLines } from "./logs.js";

/** A gateway whose captured output is whatever `read` says it is. */
function fakeApp(read: () => string): SpawnedApp & { reads: number } {
  const app = {
    reads: 0,
    output() {
      app.reads += 1;
      return read();
    },
  };
  return app as unknown as SpawnedApp & { reads: number };
}

/**
 * A gateway that gains `line` only after `afterMs` — the shape a real one
 * always has, because the log queue, the writer thread and the harness's
 * own pipe drain all run after the HTTP response.
 */
function lateOutput(before: string, line: string, afterMs: number) {
  const at = Date.now() + afterMs;
  return fakeApp(() => (Date.now() >= at ? `${before}\n${line}\n` : `${before}\n`));
}

test("a line that has not been written yet is waited for, not missed", async () => {
  const app = lateOutput("boot line", "proxy request completed status=200", 300);
  // The read a spec would do straight after its request sees nothing.
  expect(app.output().includes("status=200")).toBe(false);
  const hit = await waitForLogLine(
    app,
    (l) => l.includes("status=200"),
    "the completion line",
  );
  expect(hit).toContain("status=200");
});

test("a line already in the output is returned on the first read", async () => {
  const app = fakeApp(() => "proxy request completed status=200\n");
  await waitForLogLine(app, (l) => l.includes("status=200"), "the completion line");
  expect(app.reads).toBe(1);
});

/**
 * `output()` is a raw concatenation of pipe chunks, so its tail is
 * routinely half a line. Matching that prefix hands the spec a line whose
 * later fields are missing — which reads as "the field is absent", the
 * exact claim several specs make.
 */
test("a half-delivered line is not matched until the rest of it arrives", async () => {
  let tail = "";
  const app = fakeApp(() => `boot line\nproxy request completed request_id="r1"${tail}`);
  await expect(
    waitForLogLine(app, (l) => l.includes('request_id="r1"'), "the completion line", 150),
  ).rejects.toThrow(/timed out/);

  tail = ' upstream_model="gpt-4o-mini"\n';
  const hit = await waitForLogLine(
    app,
    (l) => l.includes('request_id="r1"'),
    "the completion line",
  );
  expect(hit).toContain('upstream_model="gpt-4o-mini"');
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
