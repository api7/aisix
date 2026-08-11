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

/// The canonical operation names — the vocabulary the dashboard, metrics and
/// docs use. Each is a method's 0.3 spelling where 0.3 defines the operation,
/// and the 1.0 PascalCase RPC name for the same operation maps onto it.
///
/// `ListTasks` is the exception: 1.0 added it, so there is no 0.3 spelling to
/// borrow and only the 1.0 method appears on the left. Its canonical name
/// follows the same shape as the rest so the vocabulary stays uniform.
const OPERATIONS: &[(&str, &str)] = &[
    // (wire method, canonical operation)
    ("message/send", "message/send"),
    ("SendMessage", "message/send"),
    ("message/stream", "message/stream"),
    ("SendStreamingMessage", "message/stream"),
    ("tasks/get", "tasks/get"),
    ("GetTask", "tasks/get"),
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
///
/// 0.3's own enum also carries a literal `unknown` (1.0 replaced it with
/// `TASK_STATE_UNSPECIFIED`). It is deliberately absent here so it falls
/// through to [`UNKNOWN_TASK_STATE`] — the recorded string is the same either
/// way, and listing it would only imply the two are distinguishable.
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

/// Task states that mean the agent has said everything it is going to say on
/// this stream: the task is finished, or it is waiting on the caller.
const STREAM_ENDING_STATES: &[&str] = &[
    "completed",
    "canceled",
    "failed",
    "rejected",
    "input-required",
    "auth-required",
];

/// Whether a streamed event is the last one its stream will carry.
///
/// A caller stops reading here, which makes this — not the upstream's
/// eventual EOF — the moment the response was fully delivered. Anything that
/// waits for the connection to close instead reports a completed task as
/// abandoned whenever the agent leaves the stream open afterwards.
///
/// 0.3 says so with `final`. 1.0's `TaskStatusUpdateEvent` has no such field
/// at all, so there the task's own state is the only signal: a task that is
/// neither `submitted` nor `working` has nothing further to stream until the
/// caller acts.
pub fn is_stream_end(event: &Value) -> bool {
    let Some(result) = event.get("result") else {
        return false;
    };
    let payload = unwrap_payload(result);
    if payload.get("final").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    let Some(state) = payload
        .get("status")
        .and_then(|status| str_field(status, "state"))
    else {
        return false;
    };
    STREAM_ENDING_STATES.contains(&normalize_task_state(state))
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
    ///
    /// `&'static str` rather than a `String`: the value is always one of the
    /// specification's states or `unknown`, and typing it that way is what
    /// lets it be used as a metric label without a caller having to promise
    /// it is bounded.
    pub task_state: &'static str,
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
        // push-notification operations name it as `params.taskId`. Both spell
        // the field the same way in either wire version (1.0's protobuf JSON
        // renders `task_id` / `context_id` in camelCase).
        for source in [params.get("message"), Some(params)].into_iter().flatten() {
            self.set_task_id(str_field(source, "taskId"));
            self.set_context_id(str_field(source, "contextId"));
        }
        // `params.id` is a FALLBACK, never an override: it is the task for
        // `tasks/get` / `tasks/cancel` / `tasks/resubscribe` (both versions)
        // and for the 0.3 push-notification operations, but on their 1.0
        // counterparts it is the push-notification CONFIG's id sitting
        // alongside the task's own `taskId`. Applying it unconditionally would
        // file those calls under a task that does not exist.
        if self.task_id.is_empty() {
            self.set_task_id(str_field(params, "id"));
        }
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
        let payload = unwrap_payload(result);
        self.set_context_id(str_field(payload, "contextId"));
        self.set_task_id(str_field(payload, "taskId"));
        // `status` is the discriminator between a Task and a Message: every
        // Task has one and no Message does, in either wire version. Neither
        // spells a message id as `id` (both use `messageId`), so this guards
        // only against a nonstandard shape — it costs nothing and keeps a
        // message from ever being filed as a task.
        if let Some(status) = payload.get("status") {
            self.set_task_id(str_field(payload, "id"));
            if let Some(state) = str_field(status, "state") {
                self.task_state = normalize_task_state(state);
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

/// The 1.0 payload `oneof` field names, in `SendMessageResponse` /
/// `StreamResponse` order.
const V1_PAYLOAD_KEYS: [&str; 4] = ["task", "message", "statusUpdate", "artifactUpdate"];

/// Reach the Task / Message / update event a `result` carries.
///
/// 0.3 puts the object directly at `result`, tagged with `kind`. 1.0's
/// `SendMessage` and streaming responses instead wrap it in a `oneof payload`,
/// which protobuf JSON renders as the set field's own name — so the same task
/// arrives as `{"task": {…}}` there. Reading only the 0.3 shape found nothing
/// at all on 1.0, which is the DEFAULT version for a registered agent.
///
/// A `kind` settles it as 0.3 outright; no 1.0 payload has one. Anything else
/// is returned untouched, which is also right for the 1.0 operations that
/// answer with a bare `Task` (`GetTask`, `CancelTask` define no wrapper).
fn unwrap_payload(result: &Value) -> &Value {
    if result.get("kind").is_some() {
        return result;
    }
    V1_PAYLOAD_KEYS
        .iter()
        .find_map(|key| result.get(*key).filter(|value| value.is_object()))
        .unwrap_or(result)
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
    fn a_1_0_only_operation_has_no_0_3_spelling_to_accept() {
        // `ListTasks` was added in 1.0, so `tasks/list` is a name this
        // gateway's own vocabulary uses — not a method any version defines.
        // Accepting it on the wire would invent a 0.3 method.
        assert_eq!(canonical_operation("ListTasks"), "tasks/list");
        assert_eq!(canonical_operation("tasks/list"), UNKNOWN_OPERATION);
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

        // The push-notification operations name the parent task instead — and
        // in 1.0 they carry the CONFIG's id in `params.id` right next to it.
        // Reading `id` last would file the call under a task that does not
        // exist, so an explicit `taskId` has to win.
        let mut v10_cfg = A2aCallFacts::default();
        v10_cfg.observe_request(&json!({
            "method": "GetTaskPushNotificationConfig",
            "params": {"taskId": "t-8", "id": "cfg-9"}
        }));
        assert_eq!(v10_cfg.task_id, "t-8");

        // 0.3 spells the same request with the TASK in `id`, so the fallback
        // still has to fire when no `taskId` is present.
        let mut v03_cfg = A2aCallFacts::default();
        v03_cfg.observe_request(&json!({
            "method": "tasks/pushNotificationConfig/get",
            "params": {"id": "t-8", "pushNotificationConfigId": "cfg-9"}
        }));
        assert_eq!(v03_cfg.task_id, "t-8");
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
        // An agent may answer `message/send` with a bare Message rather than a
        // Task. It carries a context but no task and no state, and none of
        // those may be invented for it.
        let mut facts = A2aCallFacts::default();
        facts.observe_result(&json!({
            "result": {"kind": "message", "messageId": "m-1", "role": "agent", "contextId": "c-3"}
        }));
        assert_eq!(facts.task_id, "");
        assert_eq!(facts.context_id, "c-3");
        assert_eq!(facts.task_state, "");
    }

    #[test]
    fn a_1_0_result_is_read_through_its_payload_wrapper() {
        // 1.0 — the DEFAULT version for a registered agent — wraps the object
        // in the response's `oneof payload`, which protobuf JSON renders as
        // the set field's name. Reading only 0.3's flat shape found nothing at
        // all here, so every field this module exists to record came back
        // empty on the default version.
        let mut send = A2aCallFacts::default();
        send.observe_result(&json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {"task": {
                "id": "t-10", "contextId": "c-10",
                "status": {"state": "TASK_STATE_COMPLETED"}
            }}
        }));
        assert_eq!(send.task_id, "t-10");
        assert_eq!(send.context_id, "c-10");
        assert_eq!(send.task_state, "completed");

        // The streaming wrapper has two more variants.
        let mut status = A2aCallFacts::default();
        status.observe_result(&json!({
            "result": {"statusUpdate": {
                "taskId": "t-11", "contextId": "c-11",
                "status": {"state": "TASK_STATE_WORKING"}
            }}
        }));
        assert_eq!(status.task_id, "t-11");
        assert_eq!(status.task_state, "working");

        let mut artifact = A2aCallFacts::default();
        artifact.observe_result(&json!({
            "result": {"artifactUpdate": {"taskId": "t-12", "contextId": "c-12"}}
        }));
        assert_eq!(artifact.task_id, "t-12");
        assert_eq!(artifact.context_id, "c-12");

        // A 1.0 Message payload still yields its context and no task.
        let mut message = A2aCallFacts::default();
        message.observe_result(&json!({
            "result": {"message": {"messageId": "m-2", "contextId": "c-13", "role": "agent"}}
        }));
        assert_eq!(message.task_id, "");
        assert_eq!(message.context_id, "c-13");
    }

    #[test]
    fn a_1_0_bare_task_result_is_read_without_a_wrapper() {
        // `GetTask` and `CancelTask` define no response wrapper — they answer
        // with the Task itself, so the unwrapping must not require one.
        let mut facts = A2aCallFacts::default();
        facts.observe_result(&json!({
            "result": {"id": "t-14", "contextId": "c-14", "status": {"state": "TASK_STATE_CANCELED"}}
        }));
        assert_eq!(facts.task_id, "t-14");
        assert_eq!(facts.context_id, "c-14");
        assert_eq!(facts.task_state, "canceled");
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
    fn the_terminal_event_is_recognised_in_both_versions() {
        // The moment a caller stops reading. 0.3 marks it with `final`; 1.0's
        // status-update has no such field, so only the task's own state says
        // so — and both spellings of that state have to work.
        for terminal in [
            json!({"result": {"kind": "status-update", "taskId": "t", "final": true,
                              "status": {"state": "input-required"}}}),
            json!({"result": {"kind": "task", "id": "t", "status": {"state": "completed"}}}),
            json!({"result": {"statusUpdate": {"taskId": "t",
                              "status": {"state": "TASK_STATE_FAILED"}}}}),
            json!({"result": {"task": {"id": "t", "status": {"state": "TASK_STATE_CANCELED"}}}}),
        ] {
            assert!(is_stream_end(&terminal), "{terminal} ends the stream");
        }
        // A task still running does not, nor does an event with no state at
        // all (an artifact chunk), nor an error envelope.
        for ongoing in [
            json!({"result": {"kind": "status-update", "taskId": "t", "final": false,
                              "status": {"state": "working"}}}),
            json!({"result": {"statusUpdate": {"taskId": "t",
                              "status": {"state": "TASK_STATE_SUBMITTED"}}}}),
            json!({"result": {"artifactUpdate": {"taskId": "t"}}}),
            json!({"error": {"code": -32000, "message": "boom"}}),
        ] {
            assert!(
                !is_stream_end(&ongoing),
                "{ongoing} does not end the stream"
            );
        }
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
