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
differently on a gateway older than the one you are changing, and each needs
a different discipline:

- **Adding a field** is the safe, expected change, at any depth of any
  resource — as long as the Rust field is optional or carries a serde
  default, since a field the current build requires kills every stored row
  written before it (`CLAUDE.md`). An older gateway loads the document with
  the new field ignored and reports it as partially compatible
  (`GET /status/config` `partially_compatible[]`, the heartbeat, and the
  `aisix_config_partially_compatible_resources` metric). Never assume a new
  field is enforced fleet-wide until every data plane runs a version that
  knows it — this matters most for restriction-type fields (an old gateway
  keeps allowing what the new field would forbid).
- **Adding an enum value** (a routing strategy, an adapter, a guardrail
  `kind`, …) is NOT forward compatible, by design: a value the gateway cannot
  interpret has no old behavior to fall back to, so the whole document stays
  rejected on older versions. Do not "fix" that by opening the enum, and do
  not add a `#[serde(other)]` fallback for a value that selects serving
  behavior — silently running a different routing strategy than configured is
  worse than rejecting the document. Behind a control plane the rollout
  decision is not made here — it refuses to save a projection the gateways
  registered against it cannot load, using the versions the heartbeat already
  reports. Nothing checks it on the paths that bypass a control plane (a
  resources file, a direct etcd put) or in an environment with no gateway
  registered yet: there an older gateway simply rejects the row into
  `rejected[]`, which for a genuinely new capability is the correct outcome.

Which releases a change has to stay loadable on is set by the support floor —
see the projection rule in `CLAUDE.md`.

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
