import { availableParallelism } from "node:os";

import { RANGE_SLOTS } from "./ports.js";

/** Endpoints from AISIX_E2E_ETCD_ENDPOINTS, trimmed, in order. */
export function configuredEtcdEndpoints(): string[] {
  return (process.env.AISIX_E2E_ETCD_ENDPOINTS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

/**
 * How many vitest forks this run may use.
 *
 * One per etcd cluster, because that is the constraint that forced
 * maxForks=2 in the first place (#157: every fork's gateway watching one
 * shared cluster blew past waitConfigPropagation). Bounded three ways:
 *
 *   - floor 2, the value the suite ran at before the split, and what a
 *     developer with a single local etcd gets;
 *   - RANGE_SLOTS, because ports.ts hands fork `poolId % RANGE_SLOTS`
 *     its own port range and beyond that two forks share one;
 *   - the core count, because each fork drives a real `aisix` process
 *     and several cases assert wall-clock bounds derived on a loaded
 *     runner (audio-timeout, ttft-first-frame, per-attempt-telemetry).
 *
 * The core cap is why adding a fifth etcd service to a 4-core runner
 * would buy nothing: the endpoint list is a ceiling on concurrency, not
 * a target.
 */
export function forkBudget(): number {
  const ceiling = Math.min(RANGE_SLOTS, Math.max(2, availableParallelism()));
  const override = Number(process.env.AISIX_E2E_MAX_FORKS);
  if (Number.isInteger(override) && override >= 1) return Math.min(override, ceiling);
  return Math.min(Math.max(2, configuredEtcdEndpoints().length), ceiling);
}
