# Contributing to AISIX

Thanks for your interest in AISIX! Contributions of every kind are welcome —
bug reports, feature requests, docs fixes, and code.

## Where to start

- **Bug reports & feature requests** — open a
  [GitHub issue](https://github.com/api7/aisix/issues). For bugs, include the
  gateway version (`aisix --version`), your config shape (redact secrets), and
  a minimal reproduction.
- **Questions & ideas** — ask on
  [Discord](https://discord.gg/dUmRZ7Rvf); rough ideas are welcome there
  before they harden into an issue.
- **Roadmap** — see [ROADMAP.md](ROADMAP.md) for where the project is headed.

## Development setup

Prerequisites: the Rust toolchain pinned in `rust-toolchain.toml` (rustup picks
it up automatically), plus Docker (for etcd).

```bash
cargo check --workspace
cargo fmt --check
cargo clippy --workspace -- -D warnings
cargo test --workspace

# Run locally (needs a reachable etcd + a config.yaml — see the docs quickstart)
cargo run -p aisix-server --bin aisix -- --config config.yaml
```

CI enforces `fmt`, `clippy -D warnings`, unit tests with a coverage gate, the
E2E suite, and two generated-artifact checks: the resource JSON Schemas in
`schemas/` must be regenerated (`cargo run -p aisix-core --bin dump-schema`)
and committed when the resource structs change — CI fails on drift — and the
Admin API OpenAPI document (generated at build time, not committed) must
still build and pass its structural checks
(`cargo run -p aisix-admin --bin dump-openapi`).

## Making changes

1. Fork and create a topic branch from `main`.
2. Keep PRs small and focused — one logical change per PR.
3. Add or update tests for what you change. E2E tests live in `tests/` and
   assert observable gateway behavior (wire-level requests and responses), not
   implementation details.
4. Make sure the checks above pass locally before pushing.

### Changing resource models: unknown fields vs. new enum values

The gateway reads its resources leniently from etcd and strictly on write
(issue #871). That split makes two kinds of schema change behave very
differently on a data plane that has not been upgraded yet, and each needs a
different discipline:

- **Adding a field** is the safe, expected change. An older gateway loads the
  document with the new field ignored and reports it as partially compatible
  (`GET /status/config` `partially_compatible[]`, the heartbeat, and the
  `aisix_config_partially_compatible_resources` metric). Never assume a new
  field is enforced fleet-wide until every data plane runs a version that
  knows it — this matters most for restriction-type fields (an old gateway
  keeps allowing what the new field would forbid).
- **Adding an enum value** (a routing strategy, an adapter, a guardrail
  `kind`, …) is NOT forward compatible, by design: a value the gateway cannot
  interpret has no old behavior to fall back to, so the whole document stays
  rejected on older versions. Do not "fix" that by opening the enum. Every
  new enum value needs an explicit rollout decision, made in the PR that adds
  it:
  1. **Version-gate at the control plane** (preferred for values that change
     serving behavior): the control plane only offers the value once the
     environment's data planes are on a version that knows it, using the
     version the heartbeat already reports.
  2. **Ship a degradable fallback** via `#[serde(other)]` — only when a
     fallback is semantically safe (e.g. an advisory label where "unknown"
     is a reasonable interpretation). Never for values that select serving
     behavior: silently running a different routing strategy than configured
     is worse than rejecting the document.
  3. **Accept the rejection** for values that are new capabilities: an old
     gateway that cannot serve the capability rejecting the row loudly (the
     rejection reaches `rejected[]` and the heartbeat) can be the correct
     outcome — state which of the three you chose and why in the PR.

### Commit and PR style

Commit subjects follow Conventional Commits, matching the existing history:

```
<type>(<scope>): <imperative summary>
```

with types like `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`, and
scopes like `routing`, `guardrails`, `mcp`, `obs`. Mark breaking changes with a
`!` after the scope (e.g. `refactor(routing)!: ...`). PRs are squash-merged, so
the PR title should follow the same convention — it becomes the commit subject
on `main`.

## License

AISIX is licensed under [Apache 2.0](LICENSE). By contributing, you agree that
your contributions are licensed under the same terms.
