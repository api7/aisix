import { afterEach, describe, expect, test, vi } from "vitest";
import { awaitWindowHeadroom } from "./admin.js";

// The rate limiter buckets on fixed wall-clock windows — `roll_if_stale`
// in `crates/aisix-ratelimit/src/window.rs` computes
// `bucket_start = (now / window_secs) * window_secs`. A burst that
// straddles a boundary lands in two buckets, the later call gets a fresh
// allowance, and any "the next call must be 429" assertion flaps
// depending on when in the minute CI happened to run it.
//
// These pin the two branches with fake timers, because the real behaviour
// is only observable in the last few seconds of a wall-clock minute and
// would otherwise go untested on almost every run.

describe("awaitWindowHeadroom", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  test("returns immediately when the window has enough left", async () => {
    vi.useFakeTimers();
    // 10s into the minute → 50s of headroom.
    vi.setSystemTime(new Date("2026-01-01T00:00:10.000Z"));

    let settled = false;
    const pending = awaitWindowHeadroom(10).then(() => {
      settled = true;
    });
    // No timer advance at all: the helper must not have scheduled a wait.
    await pending;
    expect(settled).toBe(true);
  });

  test("waits past the boundary when the window is nearly over", async () => {
    vi.useFakeTimers();
    // 55s into the minute → only 5s left, less than the 10s asked for.
    vi.setSystemTime(new Date("2026-01-01T00:00:55.000Z"));

    let settled = false;
    const pending = awaitWindowHeadroom(10).then(() => {
      settled = true;
    });

    await vi.advanceTimersByTimeAsync(1_000);
    expect(settled).toBe(false);

    // 5s remaining + the helper's 100ms cushion puts us in the next window.
    await vi.advanceTimersByTimeAsync(4_200);
    await pending;
    expect(settled).toBe(true);
  });

  test("treats the requested headroom as the threshold, not a sleep", async () => {
    vi.useFakeTimers();
    // Exactly at the threshold (30s left, 30s asked) → no wait.
    vi.setSystemTime(new Date("2026-01-01T00:00:30.000Z"));

    let settled = false;
    const pending = awaitWindowHeadroom(30).then(() => {
      settled = true;
    });
    await pending;
    expect(settled).toBe(true);
  });
});
