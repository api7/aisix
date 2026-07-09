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
E2E suite, and two generated-artifact checks: resource JSON schemas
(`schemas/`) and the Admin API OpenAPI document must be regenerated when the
resource structs change — CI fails on drift, and the failing job's log shows
the regeneration command.

## Making changes

1. Fork and create a topic branch from `main`.
2. Keep PRs small and focused — one logical change per PR.
3. Add or update tests for what you change. E2E tests live in `tests/` and
   assert observable gateway behavior (wire-level requests and responses), not
   implementation details.
4. Make sure the checks above pass locally before pushing.

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
