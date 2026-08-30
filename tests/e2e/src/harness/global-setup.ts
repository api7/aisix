import { EtcdClient, etcdEndpoint, onCI } from "./etcd.js";

/**
 * Fail the run when a configured etcd cluster is not answering.
 *
 * Nearly every case file opens with `if (!(await etcd.ping())) return;`
 * or `ctx.skip()`, which is right for a developer without etcd running
 * — but it means an unreachable cluster produces a GREEN run with those
 * files contributing nothing. With one shared cluster that was at least
 * all-or-nothing and impossible to miss. Splitting the suite across four
 * clusters (see etcdEndpoint) turns it into a partial, silent loss: one
 * dead endpoint takes out only the quarter of the files whose fork was
 * assigned to it, and `e2e (vitest) + coverage` still reports success.
 *
 * So the reachability check moves here, once, before any fork starts:
 *
 *   - on CI, every configured endpoint must answer. A skipped file on CI
 *     is a coverage hole, not a convenience.
 *   - locally, the same applies as soon as MORE THAN ONE endpoint is
 *     configured, because that is the partial-loss shape. A developer
 *     with no etcd at all, or one, keeps the quiet skip.
 */
export async function setup(): Promise<void> {
  const configured = (process.env.AISIX_E2E_ETCD_ENDPOINTS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const endpoints = configured.length > 0 ? configured : [etcdEndpoint()];

  if (!onCI() && endpoints.length < 2) return;

  const probes = await Promise.all(
    endpoints.map(async (endpoint) => ({
      endpoint,
      // `reachable`, not `ping`: ping throws on CI, and this needs to
      // name EVERY dead endpoint rather than stop at the first.
      //
      // 5s, not the client's 1s default: this runs before any fork, so
      // a cold container that is still opening its listener must not be
      // mistaken for a dead one.
      up: await new EtcdClient(endpoint).reachable(5000),
    })),
  );
  const dead = probes.filter((p) => !p.up).map((p) => p.endpoint);
  if (dead.length === 0) return;

  throw new Error(
    `etcd not reachable at ${dead.join(", ")} (of ${endpoints.length} configured). ` +
      "Every case file that landed on one of these would SKIP silently and the run " +
      "would still pass, so it fails here instead. Check the etcd services in " +
      ".github/workflows/ci.yml against AISIX_E2E_ETCD_ENDPOINTS — the two lists " +
      "must match entry for entry.",
  );
}
