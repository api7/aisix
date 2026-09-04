//! etcd sub-clients configured with this gateway's gRPC decode limit.
//!
//! `Client::get` / `Client::watch` delegate to sub-clients the `Client`
//! keeps private, and `Client::kv_client()` / `watch_client()` hand back
//! clones — so the decode limit can only be raised on the sub-client that
//! actually issues the call. Every etcd read in this repository is built
//! here so no call site can drift back onto tonic's default.

use etcd_client::{Client, KvClient, WatchClient};

/// Maximum size, in bytes, of a gRPC message decoded from etcd.
///
/// The gateway reads its whole configuration set in a single range
/// response and keeps it resident in memory, so tonic's 4 MiB default is
/// really a cap on how much configuration a deployment may hold — and
/// crossing it is unrecoverable on its own: the supervisor backs off and
/// re-issues the identical oversized range forever. Any finite ceiling
/// would only move that failure to a larger config set, so the limit goes
/// to `i32::MAX`, the conventional ceiling for a gRPC message size.
///
/// The trade-off that buys: an oversized length prefix is no longer
/// rejected before the buffer for it is reserved. The peer is the etcd
/// the operator configured, whose entire contents this process already
/// trusts and holds resident, so the ceiling was never what stood between
/// it and this gateway's memory.
pub const MAX_DECODING_MESSAGE_SIZE: usize = i32::MAX as usize;

/// KV client for reads under [`MAX_DECODING_MESSAGE_SIZE`].
pub fn kv_client(client: &Client) -> KvClient {
    client
        .kv_client()
        .max_decoding_message_size(MAX_DECODING_MESSAGE_SIZE)
}

/// Watch client for streams under [`MAX_DECODING_MESSAGE_SIZE`].
pub fn watch_client(client: &Client) -> WatchClient {
    client
        .watch_client()
        .max_decoding_message_size(MAX_DECODING_MESSAGE_SIZE)
}
