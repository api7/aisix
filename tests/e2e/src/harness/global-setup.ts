import { EtcdClient, etcdEndpoint, onCI } from "./etcd.js";
import { configuredEtcdEndpoints, forkBudget } from "./forks.js";

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
/**
 * Probe one endpoint, retrying before calling it dead.
 *
 * A generous single timeout buys nothing against the failure this
 * actually guards: a container still opening its listener REFUSES the
 * connection in a few milliseconds rather than hanging, so a 5s budget
 * returns false just as fast as a 50ms one. Only a retry spans a cold
 * start — and it has to, because a false negative here fails the whole
 * merge-blocking e2e leg before a single test runs. ping() escalates the
 * same way on CI for the same reason.
 */
async function reachableWithRetry(endpoint: string): Promise<boolean> {
  const client = new EtcdClient(endpoint);
  for (let attempt = 1; attempt <= 5; attempt++) {
    if (await client.reachable(2000)) return true;
    if (attempt < 5) await new Promise((r) => setTimeout(r, 500 * attempt));
  }
  return false;
}

export async function setup(): Promise<void> {
  // Only the endpoints this run will actually USE. forkBudget is capped
  // by the core count too, so a 4-entry list on a 2-core runner drives
  // two forks — and failing there because clusters 3 and 4 are missing
  // would be a demand the run never makes. etcdEndpoint assigns
  // `poolId % list.length`, so with fewer forks than entries the ones in
  // use are the leading slice.
  const configured = configuredEtcdEndpoints().slice(0, forkBudget());
  const endpoints = configured.length > 0 ? configured : [etcdEndpoint()];

  if (!onCI() && endpoints.length < 2) return;

  const probes = await Promise.all(
    endpoints.map(async (endpoint) => ({
      endpoint,
      // `reachable`, not `ping`: ping throws on CI, and this needs to
      // name EVERY dead endpoint rather than stop at the first.
      up: await reachableWithRetry(endpoint),
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
