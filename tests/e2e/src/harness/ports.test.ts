import { createServer, type Server } from "node:net";
import { describe, expect, test } from "vitest";
import { isAddrInUseStartupFailure } from "./app.js";
import { pickFreePort, pickFreePorts } from "./ports.js";

function listen(port: number): Promise<Server> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.once("error", reject);
    srv.listen(port, "127.0.0.1", () => resolve(srv));
  });
}

function close(srv: Server): Promise<void> {
  return new Promise((resolve, reject) =>
    srv.close((err) => (err ? reject(err) : resolve())),
  );
}

describe("harness port allocator", () => {
  test("hands out distinct, immediately bindable ports from the fork's range", async () => {
    const ports = await pickFreePorts(10);
    expect(new Set(ports).size).toBe(ports.length);
    for (const p of ports) {
      expect(p).toBeGreaterThanOrEqual(21000);
      expect(p).toBeLessThan(32200);
      const srv = await listen(p); // must not be handed out already bound
      await close(srv);
    }
  });

  test("skips a port that is currently bound", async () => {
    // The cursor is monotonic, so the next pick is deterministic within
    // this process: occupy it and assert the allocator steps over it.
    const probe = await pickFreePort();
    const next = probe + 1;
    const squatter = await listen(next);
    try {
      const picked = await pickFreePort();
      expect(picked).not.toBe(next);
    } finally {
      await close(squatter);
    }
  });
});

describe("isAddrInUseStartupFailure", () => {
  const tail = "  binary state: aisix exited early with code=1 signal=null\n  stderr:\n";

  test("matches the OS error text and Rust's ErrorKind rendering", () => {
    expect(
      isAddrInUseStartupFailure(
        new Error(`${tail}Error: failed to bind proxy listener: Address already in use (os error 98)`),
      ),
    ).toBe(true);
    expect(
      isAddrInUseStartupFailure(new Error(`${tail}Error: AddrInUse binding 127.0.0.1:21001`)),
    ).toBe(true);
  });

  test("does not match other startup failures or a still-running binary", () => {
    expect(
      isAddrInUseStartupFailure(
        new Error(`${tail}Error: etcd initial sync failed: deadline exceeded`),
      ),
    ).toBe(false);
    expect(
      isAddrInUseStartupFailure(
        new Error(
          "timed out waiting for /livez\n  binary state: still running\n  stderr:\nAddress already in use",
        ),
      ),
    ).toBe(false);
  });
});
