# Managed mode (aisix.cloud tenant)

When `managed.enabled = true`, aisix runs as a tenant of an
[aisix.cloud](https://github.com/api7/AISIX-Cloud) control plane:

- The admin API listener is **not** bound.
- The Playground endpoint is **not** exposed.
- All entity configuration (models / API keys / routing / guardrails)
  is read from etcd over an mTLS channel.

This document covers running the official Docker image as a managed
DP. Source-of-truth: `crates/aisix-server/src/{register,cert_bundle,heartbeat}.rs`
and `crates/aisix-core/src/config.rs`.

Two bootstrap paths are supported. Pick whichever fits the target
deployment surface:

1. **Pre-provisioned cert bundle (recommended)** — operator mints
   the mTLS leaf via the CP dashboard's `CertIssueCard` and pastes
   the resulting PEMs into the DP's environment. No `/dp/register`
   round-trip; `env_id` and `dp_id` are parsed out of the leaf cert's
   URI SAN.
2. **Self-register with a deployment token** — DP performs a one-shot
   `POST /dp/register` on first boot to obtain its bundle. Token is
   single-use; subsequent boots reuse the persisted bundle.

## First boot — pre-provisioned cert bundle (recommended)

Three PEMs are required: leaf certificate, the SEC1 EC private key
paired with the leaf, and the CA cert that the DP installs as the
trust anchor for dp-manager mTLS. Each is supplied either inline
(`AISIX_MANAGED__CP_<NAME>_PEM`) or by file path
(`AISIX_MANAGED__CP_<NAME>_FILE`). Inline and file variants are
mutually exclusive per slot — mixing them is rejected at boot.

```bash
docker run --rm \
  -e AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml \
  -e AISIX_MANAGED__CP_CERT_PEM="$(cat leaf.crt)" \
  -e AISIX_MANAGED__CP_KEY_PEM="$(cat leaf.key)" \
  -e AISIX_MANAGED__CP_CA_PEM="$(cat ca.crt)" \
  -e AISIX_MANAGED__CP_BASE_URL=https://api.us.aisix.cloud \
  -e AISIX_MANAGED__CP_ETCD_ENDPOINT=dp-manager.aisix.cloud:7943 \
  -v aisix-mtls:/var/lib/aisix \
  -p 3000:3000 \
  ghcr.io/api7/ai-gateway:dev
```

For systemd or k8s Secret mounts where multi-line PEMs in env vars
are awkward, use the file-path variants:

```bash
  -e AISIX_MANAGED__CP_CERT_FILE=/run/secrets/aisix-leaf.crt \
  -e AISIX_MANAGED__CP_KEY_FILE=/run/secrets/aisix-leaf.key \
  -e AISIX_MANAGED__CP_CA_FILE=/run/secrets/aisix-ca.crt \
```

What each flag does:

- `AISIX_MANAGED__CP_CERT_{PEM,FILE}` — the operator-minted leaf
  certificate. Its URI SAN encodes `x-aisix://env/<env_id>` and
  `x-aisix://dp/<dp_id>`; the DP parses these out and uses `env_id`
  to scope every etcd Range/Watch to `/aisix/<env_id>/`.
- `AISIX_MANAGED__CP_KEY_{PEM,FILE}` — the SEC1 EC private key
  paired with the leaf.
- `AISIX_MANAGED__CP_CA_{PEM,FILE}` — the CA the DP installs as the
  trust anchor for outbound dp-manager mTLS.
- `AISIX_MANAGED__CP_BASE_URL` — the CP HTTPS origin used by the
  heartbeat worker (`<base>/dp/heartbeat`). The cert-bundle path
  skips `/dp/register` but still needs this for periodic heartbeats.
- `AISIX_MANAGED__CP_ETCD_ENDPOINT` — bare `host:port` of the
  dp-manager mTLS-fronted etcd endpoint (no scheme — the DP attaches
  `https://` itself). Optional in the cert-bundle path: when unset
  the DP falls back to `cp_base_url`'s host:port.
- `-v aisix-mtls:/var/lib/aisix` — persist the materialised bundle.
  Re-running with the same inputs over an existing volume is safe;
  the atomic-write helper truncates+rewrites the targets.

## Alternative: register against the control plane

The DP performs a one-shot `POST /dp/register` with the deployment
token and persists the returned mTLS bundle + `dp_id` + `env_id` to
disk. Subsequent boots reuse the bundle and skip the round-trip.

```bash
docker run --rm \
  -e AISIX_CONFIG_PATH=/etc/aisix/config.managed.yaml \
  -e AISIX_MANAGED__REGISTRATION_TOKEN=aisix_dp_us_east_1_AbCd... \
  -e AISIX_MANAGED__CP_BASE_URL=https://api.us.aisix.cloud \
  -v aisix-mtls:/var/lib/aisix \
  -p 3000:3000 \
  ghcr.io/api7/ai-gateway:dev
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
| `AISIX_MANAGED__CP_ETCD_ENDPOINT` | `managed.cp_etcd_endpoint` | bare `host:port`, no scheme. **Required** for the register path; in the cert-bundle path falls back to `cp_base_url`'s host:port if unset |
| `AISIX_MANAGED__CP_CA_CERT_FILE` | `managed.cp_ca_cert_file` | unset; optional override for the dp-manager trust anchor |
| `AISIX_MANAGED__CP_CERT_PEM` / `_FILE` | `managed.cp_cert_pem` / `cp_cert_file` | unset (cert-bundle path; one of the two must be set) |
| `AISIX_MANAGED__CP_KEY_PEM` / `_FILE` | `managed.cp_key_pem` / `cp_key_file` | unset (cert-bundle path; one of the two must be set) |
| `AISIX_MANAGED__CP_CA_PEM` / `_FILE` | `managed.cp_ca_pem` / `cp_ca_file` | unset (cert-bundle path; one of the two must be set) |

## Restart semantics

The bootstrap branch is selected in this order (see
`crates/aisix-server/src/main.rs`):

1. **No bundle on disk yet, cert-bundle env vars set** → materialise
   the bundle into `mtls_dir` (atomic `write-tmp + fsync + rename`);
   parse `env_id` + `dp_id` from the leaf cert's URI SAN; connect
   dp-manager etcd via mTLS; spawn heartbeat worker against
   `cp_base_url`.
2. **No bundle on disk, registration token + `CP_BASE_URL` set** →
   `POST /dp/register`; write `ca.crt`, `client.crt`, `client.key`,
   `dp_id`, `env_id`; connect etcd via mTLS; spawn heartbeat worker.
3. **Restart, bundle on disk** → skip register / cert-bundle
   provisioning; reload bundle; load `dp_id` and `env_id` for
   heartbeat payloads + snapshot scoping; restore the snapshot from
   the on-disk cache so the proxy is ready before etcd is reached;
   connect etcd; spawn heartbeat.
4. **Bundle missing AND no cert-bundle env vars AND no registration
   token** → boot fails with an explicit error listing the env-var
   combinations that satisfy each path. This is the user-error path.

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

After `docker run`, you should see one of two boot branches.

Cert-bundle path:

```
INFO managed mode: provisioning from supplied cert bundle (api7ee parity)
INFO provisioned with dashboard-issued cert bundle dp_id=dp_xxx env_id=11111111-... etcd=dp-manager.aisix.cloud:7943
INFO heartbeat started url=https://api.us.aisix.cloud/dp/heartbeat dp_id=dp_xxx interval_secs=15
```

Register path:

```
INFO managed mode: registering with aisix.cloud CP
INFO registered with control plane dp_id=dp_xxx env_id=11111111-... etcd=dp-manager.aisix.cloud:7943
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
