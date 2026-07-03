import { createServer } from "node:net";

// Port allocation for everything the harness binds (spawned `aisix`
// listeners and mock upstreams).
//
// The old implementation asked the OS for an ephemeral port (bind :0,
// read, close) and handed it to the caller to re-bind. That is a
// check-then-use race: with vitest running two forks, a port picked in
// fork A could be re-issued by the kernel to fork B (or grabbed by a
// mock server) before `aisix` — whose bind happens a spawn later —
// got to it. Observed as startup flakes where `aisix` exits 1 with
// AddrInUse while the readiness probe sees a *different* instance
// answer 404 on the stolen port, then ECONNREFUSED.
//
// Instead, carve a disjoint port range per vitest fork
// (`VITEST_POOL_ID`; pid fallback outside vitest) and hand out ports
// from a monotonic cursor within that range, verifying each is
// actually bindable. No two in-run processes can be issued the same
// port, and within a process a port is never re-issued while a
// previous owner may still be starting up. The only remaining
// collision source is an unrelated external process, which the
// bind-probe skips when static and `spawnApp`'s AddrInUse retry
// absorbs when racing. The range sits below Linux's default ephemeral
// range (32768+) so the kernel never assigns from it.

const RANGE_BASE = 21000;
const RANGE_SIZE = 800;
const RANGE_SLOTS = 14; // 21000..32199, below the ephemeral range

function rangeSlot(): number {
  const poolId = Number(process.env.VITEST_POOL_ID);
  if (Number.isFinite(poolId) && poolId >= 0) return poolId % RANGE_SLOTS;
  return process.pid % RANGE_SLOTS;
}

let cursor = 0;

/** True when the port can be bound on 127.0.0.1 right now. */
function canBind(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const srv = createServer();
    srv.once("error", () => resolve(false));
    srv.listen(port, "127.0.0.1", () => {
      srv.close(() => resolve(true));
    });
  });
}

/**
 * Hand out the next free port from this process's dedicated range.
 * See the module comment for why this is not "ask the OS for :0".
 */
export async function pickFreePort(): Promise<number> {
  const base = RANGE_BASE + rangeSlot() * RANGE_SIZE;
  for (let i = 0; i < RANGE_SIZE; i++) {
    const port = base + (cursor++ % RANGE_SIZE);
    if (await canBind(port)) return port;
  }
  throw new Error(
    `no free port in range ${base}-${base + RANGE_SIZE - 1} after ${RANGE_SIZE} attempts`,
  );
}

/** Pick N free ports in sequence. */
export async function pickFreePorts(n: number): Promise<number[]> {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(await pickFreePort());
  return out;
}
