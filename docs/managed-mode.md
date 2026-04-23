# Managed mode (aisix.cloud tenant)

When `managed.enabled = true`, aisix runs as a tenant of an
[aisix.cloud](https://github.com/api7/AISIX-Cloud) control plane:

- The admin API listener is **not** bound.
- The admin UI is **not** served.
- The Playground endpoint is **not** exposed.
- All entity configuration (models / API keys / routing / guardrails)
  is read from etcd over an mTLS channel.

This document covers running the official Docker image as a managed
DP. Source-of-truth: `crates/aisix-server/src/{register,heartbeat}.rs`
and `crates/aisix-core/src/config.rs`.

## First boot — register against the control plane

The DP performs a one-shot `POST /dp/register` with the deployment
token and persists the returned mTLS bundle + `dp_id` to disk.
Subsequent boots reuse the bundle and skip the round-trip.

```bash
docker run --rm \
  -e AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml \
  -e AISIX_MANAGED__REGISTRATION_TOKEN=aisix_dp_us_east_1_AbCd... \
  -e AISIX_MANAGED__CP_BASE_URL=https://api.us.aisix.cloud \
  -v aisix-mtls:/var/lib/aisix \
  -p 3000:3000 \
  ghcr.io/moonming/ai-gateway:main
```

What each flag does:

- `AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml` — point the
  binary at the bootstrap template baked into the image. The default
  `/etc/aisix/config.yaml` doesn't exist in the image; standalone
  users mount their own there.
- `AISIX_MANAGED__REGISTRATION_TOKEN` — single-use deployment token
  issued by the control plane (see prd-09 §9.3.1). Burned on first
  successful register; subsequent boots ignore it.
- `AISIX_MANAGED__CP_BASE_URL` — the CP HTTPS origin. Used for
  `/dp/register` and `/dp/heartbeat`.
- `-v aisix-mtls:/var/lib/aisix` — persist the cert bundle + `dp_id`
  across container restarts. Without this every restart triggers a
  fresh registration which the CP will reject (token already used).

## Env-var overrides

Every config field is reachable via `AISIX_<UPPER>__<UPPER>` (the
`config` crate maps `__` to nested-path separators). Common knobs:

| Env var | Maps to | Default |
|:--------|:--------|:--------|
| `AISIX_PROXY__ADDR` | `proxy.addr` | `0.0.0.0:3000` |
| `AISIX_OBSERVABILITY__LOG_LEVEL` | `observability.log_level` | `info` |
| `AISIX_CACHE__BACKEND` | `cache.backend` | `memory` |
| `AISIX_MANAGED__MTLS_DIR` | `managed.mtls_dir` | `/var/lib/aisix/mtls` |
| `AISIX_MANAGED__DP_ID_FILE` | `managed.dp_id_file` | `/var/lib/aisix/dp_id` |
| `AISIX_MANAGED__SNAPSHOT_CACHE_PATH` | `managed.snapshot_cache_path` | `/var/lib/aisix/config_cache.json` (set `""` to disable) |

## Restart semantics

1. **First boot, token + URL set, no bundle on disk** → register;
   write `ca.crt`, `client.crt`, `client.key`, `dp_id`; connect etcd
   via mTLS; spawn heartbeat worker.
2. **Restart, bundle on disk** → skip register; reload bundle; load
   `dp_id` for heartbeat payloads; restore the snapshot from the
   on-disk cache so the proxy is ready before etcd is reached;
   connect etcd; spawn heartbeat.
3. **Bundle missing AND token missing** → log a warning; the etcd
   connect will fail since the placeholder endpoint is unreachable.
   This is the user-error path.

## Offline resilience (prd-09 §9.7.2)

The supervisor mirrors every applied resync / put / delete to
`managed.snapshot_cache_path` (default `/var/lib/aisix/config_cache.json`).
On boot, `restore_from_cache` runs **before** the etcd cycle starts,
so the proxy can serve traffic from the last-known config the moment
it accepts connections — even if the CP is briefly unreachable.

What this gets you:

- **CP outage**: the DP keeps proxying with whatever models / API
  keys / routing it last saw. The first successful etcd `load_all`
  on reconnect overwrites the cache with fresh state.
- **Container restart with CP down**: same as above; the proxy comes
  up with the previous config.
- **Operator opt-out**: set `AISIX_MANAGED__SNAPSHOT_CACHE_PATH=""`.
  Useful in ephemeral test rigs where a stale cache would mask a
  real failure.

Caveats:

- The cache file holds raw etcd values; if the CP rotates them with
  shapes the local DP code can't parse, the rejected entries are
  silently dropped on rebuild. The DP keeps the previous good values
  for those keys.
- The cache is best-effort: writes happen on a Tokio task that's
  fire-and-forget, so a crash mid-write is recovered by the atomic
  `write-tmp + fsync + rename` in [`SnapshotCache::store`].

## Verifying

After `docker run`, you should see:

```
INFO managed mode: registering with aisix.cloud CP
INFO registered with control plane dp_id=dp_xxx gateway_id=aigg_xxx etcd=dp-manager:7943
INFO heartbeat started url=https://api.us.aisix.cloud/dp/heartbeat dp_id=dp_xxx interval_secs=15
```

The CP dashboard's gateway view will show the new DP within one
heartbeat interval.

## Troubleshooting

- **`DP registration failed`** — token may be already-consumed or
  expired; revoke + re-issue from the CP dashboard.
- **`build reqwest client failed`** — usually a TLS bundle path
  mismatch. The default `/var/lib/aisix` must be writable by uid
  `10001` (the `aisix` user); make sure your bind-mounted volume
  has the right ownership.
- **Heartbeat 404** — the CP doesn't recognize `dp_id`. Most often
  the DP got the bundle from one CP and is now pointing at another;
  delete the volume and re-register.
