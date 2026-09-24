//! Referring to an MCP server by its resource id instead of by its name.
//!
//! Every MCP grant a key or a policy carries used to name its server
//! literally, as the `<server>` half of the namespaced `<server>__<tool>`
//! form the gateway exposes: an `mcp_access` / `mcp_policies` allow or deny
//! pattern, and a per-key MCP rate-limit entry. Renaming a registered server
//! therefore invalidated every document that named it, and the control plane
//! had to rewrite each of them.
//!
//! Each of those sites now also accepts the server's resource id, with the
//! tool still named. The id survives a rename, so the referencing document
//! never has to be rewritten — and because the client-facing namespace is
//! still the server's CURRENT name, a rename changes what callers type and
//! nothing else.

use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwapOption;
use serde::{Deserialize, Serialize};

use super::mcp_server::McpServer;
use super::AisixSnapshot;
use crate::snapshot::ResourceTable;

/// One tool on one MCP server, the server named by its resource id and the
/// tool by name.
///
/// The tool half is matched with the same single-`*` glob rule the name form
/// uses, so `tool: "*"` is the id spelling of `<server>__*`. The server half
/// is an exact id: it is compared against the id of the server the addressed
/// tool actually belongs to, never glob-matched, so a server whose *name*
/// happens to contain a `*` cannot widen a grant.
///
/// Both halves are required on the WRITE path and defaulted by the runtime
/// loader — the strict schemas add them to this definition's `required`,
/// the types do not. An entry these were required of at the type level
/// would fail to deserialize, and the loader skips a row it cannot
/// deserialize whole: one malformed entry in one `allow_ids` array would
/// stop the entire `api_key` from authenticating any traffic at all, not
/// merely lose it MCP access. Defaulted, the malformed entry matches
/// nothing instead — an empty `server_id` names no registered server, and
/// an empty `tool` glob covers no tool name the gateway exposes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpToolRef {
    /// Resource id of the registered MCP server (`mcp_servers/<id>`) this
    /// entry refers to. An id matching no registered server refers to
    /// nothing: the entry never matches, and the other entries are
    /// unaffected.
    #[serde(default)]
    pub server_id: String,

    /// Tool on that server, matched as a single-`*` glob against the bare
    /// tool name — the part after the `<server>__` namespace prefix. `"*"`
    /// covers every tool the server exposes.
    #[serde(default)]
    pub tool: String,
}

/// The registered MCP servers' names, mapped to the resource ids they are
/// stored under — the direction an ACL check needs, because the tool name a
/// client sends carries the server's NAME.
///
/// Derived from one table, so it is rebuilt exactly when that table changes
/// (see [`LiveMcpServerIndex`]).
#[derive(Debug, Default)]
pub struct McpServerIndex {
    by_name: HashMap<String, String>,
}

/// The namespace separator between a server name and a tool name in the
/// client-facing `<server>__<tool>` form. Duplicated from `aisix-mcp`, which
/// depends on this crate rather than the other way round; a server name may
/// not contain it (`McpServer::name`'s pattern), so the split below is exact.
const TOOL_NAMESPACE_SEPARATOR: &str = "__";

impl McpServerIndex {
    /// Build from the `mcp_servers` table. A name is mapped to whichever id
    /// the table's own name index says owns it, so a window in which two
    /// rows carry one name resolves here exactly as `get_by_name` does.
    pub fn build(servers: &ResourceTable<McpServer>) -> Self {
        let mut by_name = HashMap::new();
        for entry in servers.entries() {
            let name = entry.value.name.clone();
            if let Some(owner) = servers.get_by_name(&name) {
                by_name.insert(name, owner.id.clone());
            }
        }
        Self { by_name }
    }

    /// Split a client-facing `<server>__<tool>` name into the id of the
    /// server it addresses and the bare tool name.
    ///
    /// `None` when the name carries no namespace prefix or names a server
    /// that is not registered — in either case no id-form entry can match
    /// it, which is the fail-closed answer.
    pub fn address<'a>(&self, namespaced_tool: &'a str) -> Option<(&str, &'a str)> {
        let (server, tool) = namespaced_tool.split_once(TOOL_NAMESPACE_SEPARATOR)?;
        Some((self.by_name.get(server)?.as_str(), tool))
    }

    /// The resource id the registered server `name` is stored under.
    pub fn id_of(&self, name: &str) -> Option<&str> {
        self.by_name.get(name).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

#[derive(Debug)]
struct Cached {
    generation: u64,
    index: Arc<McpServerIndex>,
}

/// Lazily-rebuilt [`McpServerIndex`] shared by every reader that has to turn
/// an addressed tool into the id its server is stored under. A hit is one
/// atomic load.
///
/// Keyed on the `mcp_servers` table's own generation, never the snapshot
/// version: registering an API key must not make the next MCP request
/// rebuild this (AISIX-Cloud#1542). A racing rebuild produces two equal
/// indexes and one is discarded, the same benign duplication the pricing
/// index accepts.
#[derive(Debug, Default)]
pub struct LiveMcpServerIndex {
    cached: ArcSwapOption<Cached>,
}

impl LiveMcpServerIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// The index for `snap`, rebuilt only if the `mcp_servers` table changed
    /// since the cached copy was built. Takes the caller's snapshot rather
    /// than loading its own, so one request resolves its ACL and its
    /// per-server limits against the same published configuration.
    pub fn for_snapshot(&self, snap: &AisixSnapshot) -> Arc<McpServerIndex> {
        let generation = snap.mcp_servers.generation();
        if let Some(cached) = self.cached.load_full() {
            if cached.generation == generation {
                return Arc::clone(&cached.index);
            }
        }
        let index = Arc::new(McpServerIndex::build(&snap.mcp_servers));
        self.cached.store(Some(Arc::new(Cached {
            generation,
            index: Arc::clone(&index),
        })));
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::ResourceEntry;

    fn snapshot_with(servers: &[(&str, &str)]) -> AisixSnapshot {
        let snap = AisixSnapshot::default();
        for (id, name) in servers {
            let server: McpServer = serde_json::from_str(&format!(
                r#"{{"name":"{name}","url":"https://example.test/mcp"}}"#
            ))
            .unwrap();
            snap.mcp_servers.insert(ResourceEntry::new(*id, server, 1));
        }
        snap
    }

    #[test]
    fn addresses_a_namespaced_tool_to_its_server_id() {
        let snap = snapshot_with(&[("s-1", "github"), ("s-2", "slack")]);
        let index = McpServerIndex::build(&snap.mcp_servers);
        assert_eq!(
            index.address("github__create_issue"),
            Some(("s-1", "create_issue"))
        );
        assert_eq!(
            index.address("slack__post_message"),
            Some(("s-2", "post_message"))
        );
    }

    #[test]
    fn a_tool_name_may_itself_contain_the_separator() {
        // The split takes the FIRST separator: server names may not contain
        // one, so everything after it is the tool.
        let snap = snapshot_with(&[("s-1", "github")]);
        let index = McpServerIndex::build(&snap.mcp_servers);
        assert_eq!(index.address("github__a__b"), Some(("s-1", "a__b")));
    }

    #[test]
    fn an_unregistered_or_bare_name_addresses_nothing() {
        let snap = snapshot_with(&[("s-1", "github")]);
        let index = McpServerIndex::build(&snap.mcp_servers);
        assert!(index.address("gitlab__create_issue").is_none());
        assert!(index.address("create_issue").is_none());
    }

    #[test]
    fn the_index_follows_a_rename() {
        let before = snapshot_with(&[("s-1", "github")]);
        assert_eq!(
            McpServerIndex::build(&before.mcp_servers).id_of("github"),
            Some("s-1")
        );

        let after = snapshot_with(&[("s-1", "github-v2")]);
        let index = McpServerIndex::build(&after.mcp_servers);
        assert_eq!(index.id_of("github-v2"), Some("s-1"));
        assert!(index.id_of("github").is_none());
    }

    #[test]
    fn the_live_index_rebuilds_only_when_the_server_table_changes() {
        let live = LiveMcpServerIndex::new();
        let snap = snapshot_with(&[("s-1", "github")]);

        let first = live.for_snapshot(&snap);
        assert!(Arc::ptr_eq(&first, &live.for_snapshot(&snap)));

        // An unrelated table moving does not invalidate it.
        let key: crate::models::ApiKey =
            serde_json::from_str(r#"{"key_hash":"h","allowed_models":[]}"#).unwrap();
        snap.apikeys.insert(ResourceEntry::new("k-1", key, 1));
        assert!(Arc::ptr_eq(&first, &live.for_snapshot(&snap)));

        // Registering a server does.
        let server: McpServer =
            serde_json::from_str(r#"{"name":"slack","url":"https://example.test/mcp"}"#).unwrap();
        snap.mcp_servers
            .insert(ResourceEntry::new("s-2", server, 1));
        let second = live.for_snapshot(&snap);
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(second.id_of("slack"), Some("s-2"));
    }
}
