# onthebench-style rig harness

Reproduces the public onthebench AI-gateway benchmark methodology on our own
`m7g.4xlarge` (Graviton3, 16 cores, single NUMA node), so performance work can
be measured on the same rig shape the public board uses — and so any published
figure can be checked against a box we control.

## One command

```sh
bench/onthebench/bench.sh ubuntu@<rig-ip> /path/to/results
BENCH_RIG=ubuntu@<rig-ip> bench/onthebench/bench.sh   # results under ./bench-results/
```

The rig address is deliberately never committed: this is a public repository,
and a committed default would publish a live passwordless-sudo box.

The tree you run it from is the tree that gets measured. The script rsyncs the
source to the rig, provisions it (idempotently), builds a native release
binary, runs every load point, and rsyncs the results back. The rig is
disposable; a rebuilt box only needs passwordless ssh + sudo and this harness.

## Methodology

Replicates the published onthebench setup (https://onthebench.ai/gateways/performance):

- **Core split** — three disjoint pinned groups via `taskset`: gateway `0-3`
  (4 cores), load generator `4-9` (6), mock upstream `10-15` (6). Neither
  instrument can starve the gateway or bottleneck it.
- **Instruments** — the prebuilt, pinned `otb` load generator and `mock`
  upstream from the public benchmark rig release (engine pin
  `f3adbb1315b26129f5e317af5279decefb1cea8f`, tag `engine-v1`, from
  https://github.com/GetBusbar/benchmarking). `rig-setup.sh` verifies their
  sha256 so a re-provisioned rig either runs the byte-identical instrument or
  fails loudly. These are the same binaries behind the public board and the
  same `otb loadgen` used for the api7/aisix#891 / #902 tables.
- **Default shipped config** — a setting may appear only if the process cannot
  run the benchmark without it. The full claim set (in `run-baseline.sh`):
  `resources_file` (boot: standalone source, else AISIX demands etcd),
  `proxy.addr` (bind: the port the harness drives), `admin.admin_keys` (boot:
  config refuses to load without one), plus a resources file wiring the mock
  as the only upstream and minting the single client key (boot: AISIX has no
  anonymous mode). Everything else — thread-per-core, worker count, telemetry —
  is whatever the defaults do.
- **Load grid** — fixed concurrency, the api7/aisix#891 grid by default:
  c=16/32/128 against the 0-delay mock, c=768 against the 10ms-TTFT mock
  (`MOCK_TTFT_MS=10`). 25s windows, 5s warmup per point, ≥4 repetitions;
  a window with any failed request (`fail`, `rigrefused`, `budgetexceeded`,
  `spawnfailed`) is recorded but marked invalid, and the point retries until
  4 valid windows or 8 attempts. A run in which any point ends incomplete
  exits nonzero so it can never be mistaken for a baseline.
- **Rig floor** — the load generator driving the mock directly, so every
  gateway number can be checked against the instrument's own ceiling; two
  valid windows per floor point, under the same record-mark-retry validity
  policy as the gateway points. The 0-delay tier gets one floor at its
  largest concurrency (the ceiling barely moves with c); each delayed tier
  gets a floor per concurrency, because there the ceiling is ~conc/delay.
- **Recorded per window** — rps, fail/ok counts, p50/p99 (µs, from otb),
  gateway CPU% (`/proc/<pid>/stat` utime+stime across the window), gateway
  peak RSS (VmRSS sampled at 5 Hz), the raw otb stats line. Idle RSS and
  VmHWM are in the metadata, alongside commit, binary sha256, instrument
  sha256s, core split, kernel, and the full method parameters.
- **Flamegraph** — one on-CPU flamegraph at the c=128 saturation point per run
  (`perf record -F 499 --call-graph dwarf` → inferno), the api7/aisix#847
  workflow. `rig-setup.sh` sets `kernel.perf_event_paranoid=1` (session-scoped,
  reverts on reboot) so an unprivileged run can sample its own process, and
  `run-baseline.sh` refuses a stripped binary before wasting a run.

## Method overrides

Every method knob is a `BENCH_*` environment variable (defaults in `lib.sh`
match the #891 grid exactly): `BENCH_GRID` (`"ttft_ms:conc ..."`, e.g.
`"20:1536 20:2048"` for an added delay tier, `"0:128"` for a spot-check),
`BENCH_REPS`, `BENCH_WINDOW`, `BENCH_WARMUP`, `BENCH_MAX_TRIES`,
`BENCH_FLOOR_REPS`, `BENCH_FLAMEGRAPH`. `bench.sh` forwards them to the rig.
A changed knob is recorded in `meta.json`, so a non-default run can never
pass silently as the standard grid.

Values are validated at startup and nonsense refuses to run: grid entries
must be `ttft_ms:conc` with a positive concurrency; window, reps, max tries
and floor reps must be positive integers (warmup may be `0`);
`BENCH_FLAMEGRAPH` is `0` or `1`.

## Measuring another target ("entrant")

`run-entrant.sh <entrant-dir> <out-dir>` measures any gateway-shaped process
with the same instruments, floors, windows and validity policy — both runners
source `lib.sh`, so cross-target numbers are comparable by construction. The
entrant dir provides an `entrant.sh` implementing the contract documented at
the top of `run-entrant.sh` (start the target pinned to the gateway cores on
the harness port, upstream at the mock; optionally a prepare/teardown step,
identity metadata, and a request-shape override for targets whose ingress
speaks a dialect other than OpenAI chat). Entrant dirs are deliberately not
part of this repository; flamegraphs default off for entrants because shipped
release binaries are usually stripped (a stripped target skips the flamegraph
with a warning rather than failing the run).

## Output

One directory per run: `results.jsonl` (one JSON object per measured window,
`kind` gateway/floor, `entrant` naming the measured target), `meta.json`, the
generated config files, and the gateway/mock logs. `flamegraph-c128.svg` is
present when Inferno rendering succeeded; on a rendering failure the run
keeps `perf.data` instead, so the SVG can be produced off-rig.

## Deviations from stock defaults, and why

- `ulimit -n 65536` in the run shell: c=768 plus per-worker upstream pools
  exceeds the 1024 default soft limit; without it the run measures fd
  exhaustion, not the gateway.
- `kernel.perf_event_paranoid=1`: flamegraph sampling only; not active during
  measurement windows other than the dedicated flamegraph window.

## Trust posture

Provisioning executes third-party bytes as a passwordless-sudo user: rustup
via its official `curl | sh` installer, and the prebuilt instrument binaries
from the public benchmark release. The sha256 pins freeze reproducibility,
not trustworthiness — the accepted trade is that the rig is a disposable,
single-purpose box holding no secrets beyond its own ssh host key, rebuilt at
will. Do not point this harness at a machine that matters.
