//! Reading the protocol-level facts of an A2A call out of its JSON-RPC
//! envelopes: which operation was invoked, and which task / context / task
//! state the call touched.
//!
//! The gateway forwards A2A bodies verbatim (see [`crate::bridge`]), so an
//! operator running a mix of agents sees BOTH wire vocabularies on the same
//! endpoint: an agent pinned to 0.3 is called with `message/send`, one pinned
//! to 1.0 with `SendMessage`, and those are the same operation. Everything
//! here exists to record one fact once — a canonical operation name, a task
//! id, a task state — regardless of which version produced it.
//!
//! Wire references:
//! - Methods: the 1.0 RPC names (`SendMessage`, `GetTask`, …) are section 9.4
//!   of the specification; the `message/send`-style names are their 0.3
//!   spelling. <https://a2a-protocol.org/latest/specification/>
//! - `TaskState`: the 1.0 enum is `TASK_STATE_<NAME>`; its 0.3 wire string is
//!   the name lowercased with `_` → `-`, which is why [`normalize_task_state`]
//!   needs no per-value table.
//! - A `Task` carries `id`, `contextId` and `status.state`; the streaming
//!   update events carry `taskId` / `contextId` instead. Both shapes are read
//!   by [`A2aCallFacts::observe_result`].

use serde_json::Value;

/// Bounded label for an operation this gateway does not recognise. A caller
/// picks the JSON-RPC method, so the raw value can be anything at all; it is
/// kept in `a2a_method` for forensics and collapsed to this in every
/// aggregated position.
pub const UNKNOWN_OPERATION: &str = "unknown";

/// Bounded label for a task state that is absent, unspecified, or not one the
/// specification defines.
pub const UNKNOWN_TASK_STATE: &str = "unknown";

/// The canonical operation names, in their 0.3 spelling — the vocabulary the
/// dashboard, metrics and docs use. The 1.0 PascalCase RPC name for the same
/// operation maps onto the same entry.
const OPERATIONS: &[(&str, &str)] = &[
    // (wire method, canonical operation)
    ("message/send", "message/send"),
    ("SendMessage", "message/send"),
    ("message/stream", "message/stream"),
    ("SendStreamingMessage", "message/stream"),
    ("tasks/get", "tasks/get"),
    ("GetTask", "tasks/get"),
    ("tasks/list", "tasks/list"),
    ("ListTasks", "tasks/list"),
    ("tasks/cancel", "tasks/cancel"),
    ("CancelTask", "tasks/cancel"),
    ("tasks/resubscribe", "tasks/resubscribe"),
    ("SubscribeToTask", "tasks/resubscribe"),
    (
        "tasks/pushNotificationConfig/set",
        "tasks/pushNotificationConfig/set",
    ),
    (
        "CreateTaskPushNotificationConfig",
        "tasks/pushNotificationConfig/set",
    ),
    (
        "tasks/pushNotificationConfig/get",
        "tasks/pushNotificationConfig/get",
    ),
    (
        "GetTaskPushNotificationConfig",
        "tasks/pushNotificationConfig/get",
    ),
    (
        "tasks/pushNotificationConfig/list",
        "tasks/pushNotificationConfig/list",
    ),
    (
        "ListTaskPushNotificationConfigs",
        "tasks/pushNotificationConfig/list",
    ),
    (
        "tasks/pushNotificationConfig/delete",
        "tasks/pushNotificationConfig/delete",
    ),
    (
        "DeleteTaskPushNotificationConfig",
        "tasks/pushNotificationConfig/delete",
    ),
    (
        "agent/getAuthenticatedExtendedCard",
        "agent/getAuthenticatedExtendedCard",
    ),
    ("GetExtendedAgentCard", "agent/getAuthenticatedExtendedCard"),
];

/// The task states the specification defines, in their 0.3 wire spelling.
const TASK_STATES: &[&str] = &[
    "submitted",
    "working",
    "input-required",
    "completed",
    "canceled",
    "failed",
    "rejected",
    "auth-required",
];

/// Map a wire method to its canonical operation, collapsing an unrecognised
/// one to [`UNKNOWN_OPERATION`] so it is safe to use as a metric label.
///
/// Both wire vocabularies map onto the 0.3 spelling: `SendStreamingMessage`
/// and `message/stream` are one operation, so a deployment fronting agents on
/// both versions still aggregates as one.
pub fn canonical_operation(method: &str) -> &'static str {
    OPERATIONS
        .iter()
        .find(|(wire, _)| *wire == method)
        .map(|(_, canonical)| *canonical)
        .unwrap_or(UNKNOWN_OPERATION)
}

/// Whether an operation's response is an SSE event stream rather than a single
/// JSON-RPC envelope.
///
/// Takes the CANONICAL operation, so a 1.0 caller's `SendStreamingMessage`
/// cannot be routed down the buffering path just because the match arm listed
/// only the 0.3 spelling.
pub fn is_streaming_operation(operation: &str) -> bool {
    matches!(operation, "message/stream" | "tasks/resubscribe")
}

/// Normalize a wire task state to its 0.3 spelling, or [`UNKNOWN_TASK_STATE`]
/// when it is absent, `TASK_STATE_UNSPECIFIED`, or not a state the
/// specification defines.
///
/// The 1.0 protobuf enum name (`TASK_STATE_INPUT_REQUIRED`) becomes the 0.3
/// wire string (`input-required`) by dropping the prefix, lowercasing and
/// swapping `_` for `-`; the result is validated against the defined set, so a
/// state invented by an upstream lands on `unknown` rather than becoming an
/// unbounded label.
pub fn normalize_task_state(state: &str) -> &'static str {
    let stripped = state.strip_prefix("TASK_STATE_").unwrap_or(state);
    let candidate = stripped.to_ascii_lowercase().replace('_', "-");
    TASK_STATES
        .iter()
        .find(|known| **known == candidate)
        .copied()
        .unwrap_or(UNKNOWN_TASK_STATE)
}

/// What one A2A call touched, accumulated as its request and response(s) are
/// seen.
///
/// A streaming call feeds every event through [`Self::observe_result`], so the
/// recorded state is the LAST one the upstream reported — the state the task
/// was actually left in when the caller stopped watching, which is what an
/// operator auditing a task needs. A caller that walks away mid-task leaves
/// the last state it did see, not a fabricated terminal one.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct A2aCallFacts {
    /// Task the call created or acted on. Empty when the call names no task
    /// (a first `message/send` whose agent answers with a bare message).
    pub task_id: String,
    /// Context (conversation) the call belongs to; empty when none was seen.
    pub context_id: String,
    /// Last task state reported, in its 0.3 spelling. Empty when no response
    /// carried one — a state is never invented for a call that failed before
    /// the upstream answered.
    pub task_state: String,
}

impl A2aCallFacts {
    /// Read what the REQUEST already tells us: `message/send` and
    /// `message/stream` carry the task and context they continue on
    /// `params.message`, while the task operations name the task directly.
    ///
    /// Called before the upstream is contacted, so a call that fails outright
    /// still records which task the caller was asking about.
    pub fn observe_request(&mut self, request: &Value) {
        let Some(params) = request.get("params") else {
            return;
        };
        // A send / stream continues a task from its message; the
        // push-notification operations name the task on params itself; and
        // `tasks/get` / `tasks/cancel` / `tasks/resubscribe` carry it as
        // `params.id`. The three are mutually exclusive per operation, and the
        // field names are identical in both wire versions (1.0's protobuf JSON
        // renders `task_id` / `context_id` in camelCase).
        for source in [params.get("message"), Some(params)].into_iter().flatten() {
            self.set_task_id(str_field(source, "taskId"));
            self.set_context_id(str_field(source, "contextId"));
        }
        self.set_task_id(str_field(params, "id"));
    }

    /// Read a JSON-RPC response envelope — or one streamed event — for the
    /// task it concerns and the state it reports.
    ///
    /// Handles every `result` shape the protocol defines: a `Task` (ids on
    /// `id` / `contextId`, state under `status`), a `Message` (no state), and
    /// the streaming status / artifact update events (ids on `taskId`).
    pub fn observe_result(&mut self, response: &Value) {
        let Some(result) = response.get("result") else {
            return;
        };
        self.set_context_id(str_field(result, "contextId"));
        self.set_task_id(str_field(result, "taskId"));
        // A `Task` names itself in `id`, but so does a `Message` — and a
        // message id is not a task id. `status` is the discriminator: every
        // Task has one and no Message does, in either wire version.
        if let Some(status) = result.get("status") {
            self.set_task_id(str_field(result, "id"));
            if let Some(state) = str_field(status, "state") {
                self.task_state = normalize_task_state(state).to_string();
            }
        }
    }

    fn set_task_id(&mut self, value: Option<&str>) {
        if let Some(value) = value {
            self.task_id = value.to_string();
        }
    }

    fn set_context_id(&mut self, value: Option<&str>) {
        if let Some(value) = value {
            self.context_id = value.to_string();
        }
    }
}

/// A non-empty string field, or `None` — an absent field and an empty one are
/// the same "nothing was said" to a caller accumulating facts, and an empty
/// string must never overwrite an id read earlier in the call.
fn str_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn both_wire_vocabularies_map_to_one_operation() {
        // The whole point: a gateway fronting a 0.3 agent and a 1.0 agent must
        // aggregate their identical operations as one, not as two.
        for (v03, v10) in [
            ("message/send", "SendMessage"),
            ("message/stream", "SendStreamingMessage"),
            ("tasks/get", "GetTask"),
            ("tasks/list", "ListTasks"),
            ("tasks/cancel", "CancelTask"),
            ("tasks/resubscribe", "SubscribeToTask"),
            (
                "tasks/pushNotificationConfig/set",
                "CreateTaskPushNotificationConfig",
            ),
            (
                "tasks/pushNotificationConfig/get",
                "GetTaskPushNotificationConfig",
            ),
            (
                "tasks/pushNotificationConfig/list",
                "ListTaskPushNotificationConfigs",
            ),
            (
                "tasks/pushNotificationConfig/delete",
                "DeleteTaskPushNotificationConfig",
            ),
            ("agent/getAuthenticatedExtendedCard", "GetExtendedAgentCard"),
        ] {
            assert_eq!(canonical_operation(v03), v03, "0.3 name is the canonical");
            assert_eq!(
                canonical_operation(v10),
                v03,
                "{v10} must canonicalise to {v03}"
            );
        }
    }

    #[test]
    fn an_unrecognised_method_is_bounded() {
        // A caller picks the method, so this is the cardinality gate.
        for method in ["", "message/sendx", "../../etc/passwd", "SendMessage "] {
            assert_eq!(canonical_operation(method), UNKNOWN_OPERATION);
        }
    }

    #[test]
    fn streaming_is_decided_on_the_canonical_operation() {
        for method in [
            "message/stream",
            "SendStreamingMessage",
            "tasks/resubscribe",
            "SubscribeToTask",
        ] {
            assert!(is_streaming_operation(canonical_operation(method)));
        }
        for method in ["message/send", "SendMessage", "tasks/get", "GetTask", ""] {
            assert!(!is_streaming_operation(canonical_operation(method)));
        }
    }

    #[test]
    fn task_states_normalise_across_versions() {
        for (v10, v03) in [
            ("TASK_STATE_SUBMITTED", "submitted"),
            ("TASK_STATE_WORKING", "working"),
            ("TASK_STATE_INPUT_REQUIRED", "input-required"),
            ("TASK_STATE_COMPLETED", "completed"),
            ("TASK_STATE_CANCELED", "canceled"),
            ("TASK_STATE_FAILED", "failed"),
            ("TASK_STATE_REJECTED", "rejected"),
            ("TASK_STATE_AUTH_REQUIRED", "auth-required"),
        ] {
            assert_eq!(normalize_task_state(v10), v03);
            assert_eq!(normalize_task_state(v03), v03);
        }
        // Unspecified and anything an upstream invents are bounded, so a
        // task-state metric cannot be blown up from the far side.
        for state in ["TASK_STATE_UNSPECIFIED", "", "wat", "unknown"] {
            assert_eq!(normalize_task_state(state), UNKNOWN_TASK_STATE);
        }
    }

    #[test]
    fn a_send_that_continues_a_task_is_read_from_the_request() {
        // Recorded before the upstream is contacted, so a call that fails
        // outright still says which task the caller was asking about.
        let mut facts = A2aCallFacts::default();
        facts.observe_request(&json!({
            "jsonrpc": "2.0", "id": 1, "method": "message/send",
            "params": {"message": {"taskId": "t-1", "contextId": "c-1", "role": "user"}}
        }));
        assert_eq!(facts.task_id, "t-1");
        assert_eq!(facts.context_id, "c-1");
        assert_eq!(facts.task_state, "");
    }

    #[test]
    fn task_operations_name_their_subject_in_either_version() {
        // `GetTaskRequest` / `CancelTaskRequest` / `SubscribeToTaskRequest`
        // all carry the task as `id` in both wire versions.
        for method in ["tasks/get", "GetTask", "tasks/cancel", "tasks/resubscribe"] {
            let mut facts = A2aCallFacts::default();
            facts.observe_request(&json!({"method": method, "params": {"id": "t-7"}}));
            assert_eq!(facts.task_id, "t-7", "{method} names its task");
        }

        // The push-notification operations name the parent task instead.
        let mut cfg = A2aCallFacts::default();
        cfg.observe_request(&json!({
            "method": "GetTaskPushNotificationConfig",
            "params": {"taskId": "t-8", "configId": "c-9"}
        }));
        assert_eq!(cfg.task_id, "t-8");
    }

    #[test]
    fn a_task_result_yields_the_task_and_its_state() {
        let mut facts = A2aCallFacts::default();
        facts.observe_result(&json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {
                "kind": "task", "id": "t-2", "contextId": "c-2",
                "status": {"state": "working"}
            }
        }));
        assert_eq!(facts.task_id, "t-2");
        assert_eq!(facts.context_id, "c-2");
        assert_eq!(facts.task_state, "working");
    }

    #[test]
    fn a_message_result_is_not_mistaken_for_a_task() {
        // An agent may answer `message/send` with a bare Message, whose `id`
        // is a MESSAGE id. Recording it as the task id would attach the call
        // to a task that does not exist.
        let mut facts = A2aCallFacts::default();
        facts.observe_result(&json!({
            "result": {"kind": "message", "id": "m-1", "role": "agent", "contextId": "c-3"}
        }));
        assert_eq!(facts.task_id, "");
        assert_eq!(facts.context_id, "c-3");
        assert_eq!(facts.task_state, "");
    }

    #[test]
    fn a_stream_records_the_last_state_the_caller_saw() {
        // Streamed status updates carry `taskId`, not `id`, and the state
        // advances event by event. The recorded state is the last one the
        // upstream reported — including when the caller walks away mid-task,
        // where inventing a terminal state would be a lie.
        let mut facts = A2aCallFacts::default();
        for state in ["submitted", "working", "input-required"] {
            facts.observe_result(&json!({
                "result": {
                    "kind": "status-update", "taskId": "t-4", "contextId": "c-4",
                    "status": {"state": state}, "final": false
                }
            }));
        }
        assert_eq!(facts.task_id, "t-4");
        assert_eq!(facts.context_id, "c-4");
        assert_eq!(facts.task_state, "input-required");
    }

    #[test]
    fn an_error_envelope_leaves_the_request_facts_standing() {
        // A JSON-RPC error carries no `result`; the task the caller named in
        // the request must survive, so a failed `tasks/get` is still
        // attributable to its task.
        let mut facts = A2aCallFacts::default();
        facts.observe_request(&json!({"method": "tasks/get", "params": {"id": "t-5"}}));
        facts.observe_result(&json!({
            "jsonrpc": "2.0", "id": 1,
            "error": {"code": -32001, "message": "task not found"}
        }));
        assert_eq!(facts.task_id, "t-5");
        assert_eq!(facts.task_state, "");
    }

    #[test]
    fn an_empty_id_never_erases_one_already_seen() {
        let mut facts = A2aCallFacts::default();
        facts.observe_request(&json!({"method": "message/send", "params": {
            "message": {"taskId": "t-6", "contextId": "c-6"}
        }}));
        facts.observe_result(&json!({"result": {"taskId": "", "contextId": null}}));
        assert_eq!(facts.task_id, "t-6");
        assert_eq!(facts.context_id, "c-6");
    }
}
