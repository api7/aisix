//! Conversion between the two decision wire formats `/v1/systemone` and
//! `/v1/decisions` speak, for a request whose upstream speaks the other one.
//!
//! - **System One** (TypeSafe's format, also served by Microsoft Foundry and
//!   OpenRouter): `{model, state, questions: {<name>: noul | choice | score}}`
//!   answered by `{model, answers: {<name>: …}, usage}`.
//! - **Decisions** (OpenAI's format): `{model, input, questions: [predicate |
//!   choice | score]}` answered by `{model, answers: [...in question order],
//!   usage}`.
//!
//! Both directions go through one intermediate shape ([`Request`] and
//! [`Answer`]), so each format has exactly one reader and one writer. The
//! mapping follows the mainstream gateway convention for these two formats:
//!
//! - message input becomes a `state` string joined with blank lines, and an
//!   image part cannot reach a System One upstream;
//! - System One keys are the question names when every question has a
//!   distinct one, else `"0".."n-1"` for all of them, mapped back by position;
//! - a noul question's `true` / `false` criteria fold into the predicate's
//!   instructions, and a predicate becomes a noul with instructions only;
//! - boolean choice values travel as the keys `"true"` / `"false"` and are
//!   restored on the way back;
//! - score levels travel as `"label: description"` criteria, and answers come
//!   back labelled from the request, with a level the upstream gave no
//!   probability reported as 0;
//! - an answer that is missing, malformed, or of the wrong type is a refusal
//!   on the Decisions side and an absent key on the System One side;
//! - anything one format has no place for is dropped.
//!
//! Object member order is load-bearing here (question order IS answer order on
//! the Decisions side, and choice order is the option order a model is shown),
//! and `serde_json::Map` sorts its keys, so the request side reads into
//! [`Json`], which keeps document order.

use std::collections::HashMap;

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use serde_json::Value;

/// A JSON value whose objects keep their members in document order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            // The last occurrence wins, as it does for `serde_json::Value`.
            Json::Object(members) => members.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// A member that is present and not `null`.
    fn field(&self, key: &str) -> Option<&Json> {
        self.get(key).filter(|v| !matches!(v, Json::Null))
    }

    /// The top-level `model` string of a request.
    pub(crate) fn model(&self) -> Option<&str> {
        self.get("model").and_then(Json::as_str)
    }

    /// This document with its top-level `model` replaced.
    pub(crate) fn with_model(&self, model: &str) -> Json {
        match self {
            Json::Object(members) => Json::Object(
                members
                    .iter()
                    .filter(|(k, _)| k != "model")
                    .cloned()
                    .chain([("model".to_owned(), Json::str(model))])
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// The value as model-facing text: a string as-is, anything else as
    /// compact JSON.
    fn text(&self) -> String {
        match self {
            Json::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        }
    }

    fn number(n: f64) -> Json {
        serde_json::Number::from_f64(n).map_or(Json::Null, Json::Number)
    }

    fn str(s: impl Into<String>) -> Json {
        Json::String(s.into())
    }
}

impl From<&Value> for Json {
    fn from(v: &Value) -> Self {
        match v {
            Value::Null => Json::Null,
            Value::Bool(b) => Json::Bool(*b),
            Value::Number(n) => Json::Number(n.clone()),
            Value::String(s) => Json::String(s.clone()),
            Value::Array(a) => Json::Array(a.iter().map(Json::from).collect()),
            Value::Object(o) => {
                Json::Object(o.iter().map(|(k, v)| (k.clone(), v.into())).collect())
            }
        }
    }
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Json::Null => s.serialize_unit(),
            Json::Bool(b) => s.serialize_bool(*b),
            Json::Number(n) => n.serialize(s),
            Json::String(v) => s.serialize_str(v),
            Json::Array(items) => {
                let mut seq = s.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Json::Object(members) => {
                let mut map = s.serialize_map(Some(members.len()))?;
                for (k, v) in members {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct JsonVisitor;
        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = Json;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("any JSON value")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_none<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
                Ok(Json::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
                Ok(Json::Number(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
                Ok(Json::Number(v.into()))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Json, E> {
                Ok(Json::number(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<Json, E> {
                Ok(Json::String(v.to_owned()))
            }
            fn visit_string<E>(self, v: String) -> Result<Json, E> {
                Ok(Json::String(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Json::Array(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut members = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, Json>()? {
                    members.push((k, v));
                }
                Ok(Json::Object(members))
            }
        }
        d.deserialize_any(JsonVisitor)
    }
}

// ---------------------------------------------------------------------------
// The intermediate request
// ---------------------------------------------------------------------------

pub(crate) struct Request {
    input: Input,
    questions: Vec<Question>,
    safety_identifier: Option<String>,
    /// System One's embedded media clips; there is no Decisions equivalent.
    audio: Option<Json>,
    videos: Option<Json>,
}

enum Input {
    State(Json),
    Messages(Vec<Vec<Part>>),
}

enum Part {
    Text(String),
    Image,
}

struct Question {
    name: Option<String>,
    instructions: Option<Json>,
    kind: Kind,
}

enum Kind {
    /// A noul (System One) / predicate (Decisions) question.
    Predicate {
        criteria: Option<Vec<(String, Json)>>,
    },
    Choice(Vec<ChoiceOption>),
    Score(Vec<Level>),
}

struct ChoiceOption {
    value: ChoiceValue,
    description: Option<Json>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ChoiceValue {
    Text(String),
    Bool(bool),
}

impl ChoiceValue {
    /// The System One key a value travels as.
    fn key(&self) -> String {
        match self {
            ChoiceValue::Text(s) => s.clone(),
            ChoiceValue::Bool(b) => b.to_string(),
        }
    }

    fn to_json(&self) -> Json {
        match self {
            ChoiceValue::Text(s) => Json::str(s.clone()),
            ChoiceValue::Bool(b) => Json::Bool(*b),
        }
    }

    fn from_value(v: &Value) -> Option<Self> {
        match v {
            Value::String(s) => Some(ChoiceValue::Text(s.clone())),
            Value::Bool(b) => Some(ChoiceValue::Bool(*b)),
            _ => None,
        }
    }
}

struct Level {
    label: Json,
    description: Option<String>,
}

const NOT_AN_OBJECT: &str = "request body must be a JSON object";

/// Read a `/v1/systemone` request body.
pub(crate) fn parse_systemone_request(doc: &Json) -> Result<Request, String> {
    let Json::Object(_) = doc else {
        return Err(NOT_AN_OBJECT.into());
    };
    let state = doc.field("state").ok_or("`state` is required")?.clone();
    let Some(Json::Object(entries)) = doc.get("questions") else {
        return Err("`questions` must be an object of named questions".into());
    };
    if entries.is_empty() {
        return Err("`questions` must not be empty".into());
    }
    // A repeated name is one question, the last spelling winning, as it is
    // for every other JSON reader of the same body.
    let entries: Vec<&(String, Json)> = entries
        .iter()
        .enumerate()
        .filter(|(i, (name, _))| !entries[i + 1..].iter().any(|(k, _)| k == name))
        .map(|(_, e)| e)
        .collect();
    let questions = entries
        .iter()
        .map(|(name, q)| systemone_question(name, q))
        .collect::<Result<_, _>>()?;
    Ok(Request {
        input: Input::State(state),
        questions,
        safety_identifier: None,
        audio: doc.field("audio").cloned(),
        videos: doc.field("videos").cloned(),
    })
}

fn systemone_question(name: &str, q: &Json) -> Result<Question, String> {
    let Json::Object(_) = q else {
        return Err(format!("question `{name}` must be an object"));
    };
    let instructions = q.field("instructions").cloned();
    let kind = match q.get("type").and_then(Json::as_str) {
        Some("noul") => {
            let criteria = match q.field("criteria") {
                None => None,
                Some(Json::Object(rules)) => {
                    if rules.iter().any(|(k, _)| k != "true" && k != "false") {
                        return Err(format!(
                            "question `{name}`: noul criteria keys must be `true` or `false`"
                        ));
                    }
                    Some(rules.clone())
                }
                Some(_) => {
                    return Err(format!(
                        "question `{name}`: noul criteria must be an object"
                    ))
                }
            };
            if instructions.is_none() && criteria.is_none() {
                return Err(format!(
                    "question `{name}`: a noul question needs instructions or criteria"
                ));
            }
            Kind::Predicate { criteria }
        }
        Some("choice") => match q.get("criteria") {
            Some(Json::Object(options)) if !options.is_empty() => Kind::Choice(
                options
                    .iter()
                    .map(|(value, description)| ChoiceOption {
                        value: ChoiceValue::Text(value.clone()),
                        description: (!matches!(description, Json::Null))
                            .then(|| description.clone()),
                    })
                    .collect(),
            ),
            _ => {
                return Err(format!(
                    "question `{name}`: choice criteria must be a non-empty object of options"
                ))
            }
        },
        Some("score") => match q.get("criteria") {
            Some(Json::Array(levels)) if !levels.is_empty() => Kind::Score(
                levels
                    .iter()
                    .map(|label| Level {
                        label: label.clone(),
                        description: None,
                    })
                    .collect(),
            ),
            _ => {
                return Err(format!(
                    "question `{name}`: score criteria must be a non-empty array of levels"
                ))
            }
        },
        _ => {
            return Err(format!(
                "question `{name}` has an unsupported type; expected noul, choice or score"
            ))
        }
    };
    Ok(Question {
        name: Some(name.to_owned()),
        instructions,
        kind,
    })
}

/// Read a `/v1/decisions` request body.
pub(crate) fn parse_decisions_request(doc: &Json) -> Result<Request, String> {
    let Json::Object(_) = doc else {
        return Err(NOT_AN_OBJECT.into());
    };
    let input = match doc.get("input") {
        Some(Json::String(s)) => Input::State(Json::str(s.clone())),
        Some(Json::Array(messages)) => Input::Messages(
            messages
                .iter()
                .enumerate()
                .map(|(i, m)| input_message(i, m))
                .collect::<Result<_, _>>()?,
        ),
        _ => return Err("`input` must be a string or an array of input messages".into()),
    };
    let Some(Json::Array(items)) = doc.get("questions") else {
        return Err("`questions` must be an array".into());
    };
    if items.is_empty() {
        return Err("`questions` must not be empty".into());
    }
    let questions = items
        .iter()
        .enumerate()
        .map(|(i, q)| decisions_question(i, q))
        .collect::<Result<_, _>>()?;
    let safety_identifier = match doc.field("safety_identifier") {
        None => None,
        Some(Json::String(s)) => Some(s.clone()),
        Some(_) => return Err("`safety_identifier` must be a string".into()),
    };
    Ok(Request {
        input,
        questions,
        safety_identifier,
        audio: doc.field("audio").cloned(),
        videos: doc.field("videos").cloned(),
    })
}

fn input_message(i: usize, m: &Json) -> Result<Vec<Part>, String> {
    match m.field("content") {
        Some(Json::String(s)) => Ok(vec![Part::Text(s.clone())]),
        Some(Json::Array(parts)) => parts
            .iter()
            .map(|p| match p.get("type").and_then(Json::as_str) {
                Some("input_text") => p
                    .get("text")
                    .and_then(Json::as_str)
                    .map(|t| Part::Text(t.to_owned()))
                    .ok_or_else(|| format!("`input[{i}]`: an input_text part needs `text`")),
                Some("input_image") => Ok(Part::Image),
                _ => Err(format!(
                    "`input[{i}]`: content parts must be input_text or input_image"
                )),
            })
            .collect(),
        _ => Err(format!(
            "`input[{i}].content` must be a string or an array of content parts"
        )),
    }
}

fn decisions_question(i: usize, q: &Json) -> Result<Question, String> {
    let Json::Object(_) = q else {
        return Err(format!("`questions[{i}]` must be an object"));
    };
    let name = match q.field("name") {
        None => None,
        Some(Json::String(s)) => Some(s.clone()),
        Some(_) => return Err(format!("`questions[{i}].name` must be a string")),
    };
    let instructions = match q.get("instructions") {
        Some(Json::String(s)) => Json::str(s.clone()),
        _ => return Err(format!("`questions[{i}].instructions` must be a string")),
    };
    let kind = match q.get("type").and_then(Json::as_str) {
        Some("predicate") => Kind::Predicate { criteria: None },
        Some("choice") => {
            let Some(Json::Array(choices)) = q.get("choices") else {
                return Err(format!("`questions[{i}].choices` must be an array"));
            };
            let options = choices
                .iter()
                .map(|c| {
                    let value = match c.get("value") {
                        Some(Json::String(s)) => ChoiceValue::Text(s.clone()),
                        Some(Json::Bool(b)) => ChoiceValue::Bool(*b),
                        _ => {
                            return Err(format!(
                                "`questions[{i}].choices[].value` must be a string or a boolean"
                            ))
                        }
                    };
                    Ok(ChoiceOption {
                        value,
                        description: optional_string(c, "description", i)?.map(Json::String),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let mut keys = std::collections::HashSet::new();
            if !options.iter().all(|o| keys.insert(o.value.key())) {
                return Err(format!(
                    "`questions[{i}]`: choice values must be unique, and a boolean cannot share \
                     its text with a string choice"
                ));
            }
            Kind::Choice(options)
        }
        Some("score") => {
            let Some(Json::Array(levels)) = q.get("levels") else {
                return Err(format!("`questions[{i}].levels` must be an array"));
            };
            Kind::Score(
                levels
                    .iter()
                    .map(|l| {
                        let label = l.get("label").and_then(Json::as_str).ok_or_else(|| {
                            format!("`questions[{i}].levels[].label` must be a string")
                        })?;
                        Ok(Level {
                            label: Json::str(label),
                            description: optional_string(l, "description", i)?,
                        })
                    })
                    .collect::<Result<_, String>>()?,
            )
        }
        _ => {
            return Err(format!(
                "`questions[{i}].type` must be predicate, choice or score"
            ))
        }
    };
    Ok(Question {
        name,
        instructions: Some(instructions),
        kind,
    })
}

fn optional_string(obj: &Json, key: &str, i: usize) -> Result<Option<String>, String> {
    match obj.field(key) {
        None => Ok(None),
        Some(Json::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(format!("`questions[{i}]`: `{key}` must be a string")),
    }
}

/// The System One keys of `questions`: their names when every question has
/// a distinct one, else positions for all of them.
fn systemone_keys(questions: &[Question]) -> Vec<String> {
    let names: Vec<&str> = questions.iter().filter_map(|q| q.name.as_deref()).collect();
    let distinct: std::collections::HashSet<&str> = names.iter().copied().collect();
    if names.len() == questions.len() && distinct.len() == names.len() {
        names.into_iter().map(str::to_owned).collect()
    } else {
        (0..questions.len()).map(|i| i.to_string()).collect()
    }
}

/// Write the request a System One upstream receives.
pub(crate) fn to_systemone_request(model: &str, req: &Request) -> Result<Json, String> {
    if req.safety_identifier.is_some() {
        return Err(
            "`safety_identifier` is supported only when the model is served by OpenAI".into(),
        );
    }
    let state = match &req.input {
        Input::State(state) => state.clone(),
        Input::Messages(messages) => {
            if messages.iter().flatten().any(|p| matches!(p, Part::Image)) {
                return Err(
                    "input_image content parts are not supported because this model's \
                            upstream accepts text input only"
                        .into(),
                );
            }
            Json::str(
                messages
                    .iter()
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|p| match p {
                                Part::Text(t) => Some(t.as_str()),
                                Part::Image => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            )
        }
    };
    let mut out = vec![
        ("model".to_owned(), Json::str(model)),
        ("state".to_owned(), state),
    ];
    for (key, media) in [("audio", &req.audio), ("videos", &req.videos)] {
        if let Some(media) = media {
            out.push((key.to_owned(), media.clone()));
        }
    }
    let questions = systemone_keys(&req.questions)
        .into_iter()
        .zip(&req.questions)
        .map(|(key, q)| (key, systemone_question_json(q)))
        .collect();
    out.push(("questions".to_owned(), Json::Object(questions)));
    Ok(Json::Object(out))
}

fn systemone_question_json(q: &Question) -> Json {
    let mut out = Vec::new();
    let kind = match &q.kind {
        Kind::Predicate { .. } => "noul",
        Kind::Choice(_) => "choice",
        Kind::Score(_) => "score",
    };
    out.push(("type".to_owned(), Json::str(kind)));
    if let Some(instructions) = &q.instructions {
        out.push(("instructions".to_owned(), instructions.clone()));
    }
    match &q.kind {
        Kind::Predicate { criteria } => {
            if let Some(criteria) = criteria {
                out.push(("criteria".to_owned(), Json::Object(criteria.clone())));
            }
        }
        Kind::Choice(options) => out.push((
            "criteria".to_owned(),
            Json::Object(
                options
                    .iter()
                    .map(|o| (o.value.key(), o.description.clone().unwrap_or(Json::Null)))
                    .collect(),
            ),
        )),
        Kind::Score(levels) => out.push((
            "criteria".to_owned(),
            Json::Array(
                levels
                    .iter()
                    .map(|l| match &l.description {
                        None => l.label.clone(),
                        Some(d) => Json::str(format!("{}: {d}", l.label.text())),
                    })
                    .collect(),
            ),
        )),
    }
    Json::Object(out)
}

/// Write the request an OpenAI Decisions upstream receives.
pub(crate) fn to_decisions_request(model: &str, req: &Request) -> Result<Json, String> {
    if req.audio.is_some() || req.videos.is_some() {
        return Err(
            "audio and video input are not supported when the model is served by OpenAI".into(),
        );
    }
    if req.questions.iter().any(|q| match &q.kind {
        Kind::Choice(options) => options.len() < 2,
        Kind::Score(levels) => levels.len() < 2,
        Kind::Predicate { .. } => false,
    }) {
        return Err(
            "OpenAI needs at least 2 choices or levels on every choice or score question".into(),
        );
    }
    let input = match &req.input {
        Input::State(state) => Json::str(state.text()),
        // A Decisions request never reaches here: it is forwarded verbatim
        // to a Decisions upstream.
        Input::Messages(_) => {
            return Err("message input cannot be converted for this model's upstream".into())
        }
    };
    let questions = req.questions.iter().map(decisions_question_json).collect();
    let mut out = vec![
        ("model".to_owned(), Json::str(model)),
        ("input".to_owned(), input),
        ("questions".to_owned(), Json::Array(questions)),
    ];
    if let Some(id) = &req.safety_identifier {
        out.push(("safety_identifier".to_owned(), Json::str(id.clone())));
    }
    Ok(Json::Object(out))
}

fn decisions_question_json(q: &Question) -> Json {
    let mut out = Vec::new();
    let instructions_or = |default: &str| {
        q.instructions
            .as_ref()
            .map_or_else(|| default.to_owned(), Json::text)
    };
    let (kind, instructions) = match &q.kind {
        Kind::Predicate { criteria } => {
            let mut parts: Vec<String> = q.instructions.iter().map(Json::text).collect();
            parts.extend(
                criteria
                    .iter()
                    .flatten()
                    .filter(|(_, rule)| !matches!(rule, Json::Null))
                    .map(|(answer, rule)| format!("Answer {answer} when: {}", rule.text())),
            );
            let joined = parts.join("\n\n");
            let instructions = if joined.is_empty() {
                "Is this true of the input?".to_owned()
            } else {
                joined
            };
            ("predicate", instructions)
        }
        Kind::Choice(_) => (
            "choice",
            instructions_or("Which choice best fits the input?"),
        ),
        Kind::Score(_) => ("score", instructions_or("Which level best fits the input?")),
    };
    out.push(("type".to_owned(), Json::str(kind)));
    if let Some(name) = &q.name {
        out.push(("name".to_owned(), Json::str(name.clone())));
    }
    out.push(("instructions".to_owned(), Json::str(instructions)));
    match &q.kind {
        Kind::Predicate { .. } => {}
        Kind::Choice(options) => out.push((
            "choices".to_owned(),
            Json::Array(
                options
                    .iter()
                    .map(|o| {
                        let mut c = vec![("value".to_owned(), o.value.to_json())];
                        if let Some(d) = &o.description {
                            c.push(("description".to_owned(), Json::str(d.text())));
                        }
                        Json::Object(c)
                    })
                    .collect(),
            ),
        )),
        Kind::Score(levels) => out.push((
            "levels".to_owned(),
            Json::Array(
                levels
                    .iter()
                    .map(|l| {
                        let mut level = vec![("label".to_owned(), Json::str(l.label.text()))];
                        if let Some(d) = &l.description {
                            level.push(("description".to_owned(), Json::str(d.clone())));
                        }
                        Json::Object(level)
                    })
                    .collect(),
            ),
        )),
    }
    Json::Object(out)
}

// ---------------------------------------------------------------------------
// Answers
// ---------------------------------------------------------------------------

/// One answer, in the position of the question it answers.
pub(crate) enum Answer {
    Predicate(f64),
    Choice {
        choice: ChoiceValue,
        confidence: f64,
        probabilities: Vec<(ChoiceValue, f64)>,
    },
    Score {
        score: f64,
        confidence: f64,
        /// `(level index, label, probability)`.
        probabilities: Vec<(i64, Json, f64)>,
    },
    Refusal,
}

/// Token counts a decision upstream reported, in either format.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecisionUsage {
    pub input: u32,
    pub output: u32,
    /// Prompt-cache hits, a subset of `input` (Decisions format only).
    pub cached: u32,
    /// Raw cache-write count, when the upstream reported one.
    pub cache_write: Option<u32>,
    /// Reasoning tokens, a subset of `output` (Decisions format only).
    pub reasoning: u32,
    /// The upstream's own total; 0 when it reported none.
    pub upstream_total: u32,
}

fn count(v: &Value, pointer: &str) -> Option<u32> {
    v.pointer(pointer)
        .and_then(Value::as_u64)
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
}

/// Usage from a System One response (`usage.input_tokens` /
/// `usage.output_tokens`; OpenRouter adds a `cost` beside them).
pub(crate) fn systemone_usage(body: &Value) -> Option<DecisionUsage> {
    body.get("usage")?;
    Some(DecisionUsage {
        input: count(body, "/usage/input_tokens").unwrap_or(0),
        output: count(body, "/usage/output_tokens").unwrap_or(0),
        ..Default::default()
    })
}

/// Usage from a Decisions response.
pub(crate) fn decisions_usage(body: &Value) -> Option<DecisionUsage> {
    body.get("usage")?;
    Some(DecisionUsage {
        input: count(body, "/usage/input_tokens").unwrap_or(0),
        output: count(body, "/usage/output_tokens").unwrap_or(0),
        cached: count(body, "/usage/input_tokens_details/cached_tokens").unwrap_or(0),
        cache_write: count(body, "/usage/input_tokens_details/cache_write_tokens"),
        reasoning: count(body, "/usage/output_tokens_details/reasoning_tokens").unwrap_or(0),
        upstream_total: count(body, "/usage/total_tokens").unwrap_or(0),
    })
}

/// Read a System One response's answers in the request's question order.
pub(crate) fn answers_from_systemone(body: &Value, req: &Request) -> Vec<Answer> {
    let answers = body.get("answers").and_then(Value::as_object);
    systemone_keys(&req.questions)
        .iter()
        .zip(&req.questions)
        .map(|(key, q)| {
            answers
                .and_then(|a| a.get(key))
                .and_then(|a| systemone_answer(q, a))
                .unwrap_or(Answer::Refusal)
        })
        .collect()
}

fn systemone_answer(q: &Question, a: &Value) -> Option<Answer> {
    let f = |key: &str| a.get(key).and_then(Value::as_f64);
    match (&q.kind, a.get("type")?.as_str()?) {
        (Kind::Predicate { .. }, "noul") => Some(Answer::Predicate(f("noul")?)),
        (Kind::Choice(options), "choice") => {
            let typed: HashMap<String, ChoiceValue> = options
                .iter()
                .map(|o| (o.value.key(), o.value.clone()))
                .collect();
            let restore = |key: &str| {
                typed
                    .get(key)
                    .cloned()
                    .unwrap_or_else(|| ChoiceValue::Text(key.to_owned()))
            };
            Some(Answer::Choice {
                choice: restore(a.get("choice")?.as_str()?),
                confidence: f("confidence")?,
                probabilities: a
                    .get("probabilities")?
                    .as_object()?
                    .iter()
                    .filter_map(|(k, p)| Some((restore(k), p.as_f64()?)))
                    .collect(),
            })
        }
        (Kind::Score(levels), "score") => {
            let legend = a.get("legend").and_then(Value::as_object);
            let probabilities = a.get("probabilities").and_then(Value::as_object);
            Some(Answer::Score {
                score: f("score")?,
                confidence: f("confidence")?,
                probabilities: levels
                    .iter()
                    .enumerate()
                    .map(|(i, level)| {
                        let key = i.to_string();
                        (
                            i as i64,
                            legend
                                .and_then(|l| l.get(&key))
                                .map(Json::from)
                                .unwrap_or_else(|| level.label.clone()),
                            probabilities
                                .and_then(|p| p.get(&key))
                                .and_then(Value::as_f64)
                                .unwrap_or(0.0),
                        )
                    })
                    .collect(),
            })
        }
        _ => None,
    }
}

/// Read a Decisions response's answers, matched to questions by position.
pub(crate) fn answers_from_decisions(body: &Value, req: &Request) -> Vec<Answer> {
    let answers = body.get("answers").and_then(Value::as_array);
    req.questions
        .iter()
        .enumerate()
        .map(|(i, q)| {
            answers
                .and_then(|a| a.get(i))
                .and_then(|a| decisions_answer(q, a))
                .unwrap_or(Answer::Refusal)
        })
        .collect()
}

fn decisions_answer(q: &Question, a: &Value) -> Option<Answer> {
    let f = |v: &Value, key: &str| v.get(key).and_then(Value::as_f64);
    match (&q.kind, a.get("type")?.as_str()?) {
        (Kind::Predicate { .. }, "predicate") => Some(Answer::Predicate(f(a, "probability")?)),
        (Kind::Choice(_), "choice") => Some(Answer::Choice {
            choice: ChoiceValue::from_value(a.get("choice")?)?,
            confidence: f(a, "confidence")?,
            probabilities: a
                .get("probabilities")?
                .as_array()?
                .iter()
                .filter_map(|p| {
                    Some((
                        ChoiceValue::from_value(p.get("value")?)?,
                        f(p, "probability")?,
                    ))
                })
                .collect(),
        }),
        (Kind::Score(levels), "score") => Some(Answer::Score {
            score: f(a, "score")?,
            confidence: f(a, "confidence")?,
            probabilities: a
                .get("probabilities")?
                .as_array()?
                .iter()
                .filter_map(|p| {
                    let value = p.get("value")?.as_i64()?;
                    let label = usize::try_from(value)
                        .ok()
                        .and_then(|i| levels.get(i))
                        .map(|l| l.label.clone())
                        .or_else(|| p.get("label").map(Json::from))?;
                    Some((value, label, f(p, "probability")?))
                })
                .collect(),
        }),
        _ => None,
    }
}

/// Write the response a `/v1/decisions` caller receives.
pub(crate) fn to_decisions_response(
    model: &str,
    req: &Request,
    answers: &[Answer],
    usage: &DecisionUsage,
) -> Json {
    let answers = req
        .questions
        .iter()
        .zip(answers)
        .map(|(q, a)| {
            let name = q.name.clone().map_or(Json::Null, Json::String);
            let entry = |kind: &str, rest: Vec<(&str, Json)>| {
                let mut out = vec![
                    ("type".to_owned(), Json::str(kind)),
                    ("name".to_owned(), name.clone()),
                ];
                out.extend(rest.into_iter().map(|(k, v)| (k.to_owned(), v)));
                Json::Object(out)
            };
            match (&q.kind, a) {
                (Kind::Predicate { .. }, Answer::Predicate(p)) => {
                    entry("predicate", vec![("probability", Json::number(*p))])
                }
                (
                    Kind::Choice(options),
                    Answer::Choice {
                        choice,
                        confidence,
                        probabilities,
                    },
                ) => {
                    let by_key: HashMap<String, f64> =
                        probabilities.iter().map(|(v, p)| (v.key(), *p)).collect();
                    entry(
                        "choice",
                        vec![
                            ("choice", choice.to_json()),
                            (
                                "probabilities",
                                Json::Array(
                                    options
                                        .iter()
                                        .map(|o| {
                                            Json::Object(vec![
                                                ("value".to_owned(), o.value.to_json()),
                                                (
                                                    "probability".to_owned(),
                                                    Json::number(
                                                        by_key
                                                            .get(&o.value.key())
                                                            .copied()
                                                            .unwrap_or(0.0),
                                                    ),
                                                ),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                            ("confidence", Json::number(*confidence)),
                        ],
                    )
                }
                (
                    Kind::Score(levels),
                    Answer::Score {
                        score,
                        confidence,
                        probabilities,
                    },
                ) => {
                    let by_index: HashMap<i64, f64> =
                        probabilities.iter().map(|(i, _, p)| (*i, *p)).collect();
                    entry(
                        "score",
                        vec![
                            ("score", Json::number(*score)),
                            (
                                "probabilities",
                                Json::Array(
                                    levels
                                        .iter()
                                        .enumerate()
                                        .map(|(i, l)| {
                                            Json::Object(vec![
                                                (
                                                    "value".to_owned(),
                                                    Json::Number((i as u64).into()),
                                                ),
                                                ("label".to_owned(), Json::str(l.label.text())),
                                                (
                                                    "probability".to_owned(),
                                                    Json::number(
                                                        by_index
                                                            .get(&(i as i64))
                                                            .copied()
                                                            .unwrap_or(0.0),
                                                    ),
                                                ),
                                            ])
                                        })
                                        .collect(),
                                ),
                            ),
                            ("confidence", Json::number(*confidence)),
                        ],
                    )
                }
                _ => entry("refusal", vec![]),
            }
        })
        .collect();
    let n = |v: u32| Json::Number(u64::from(v).into());
    Json::Object(vec![
        ("model".to_owned(), Json::str(model)),
        ("answers".to_owned(), Json::Array(answers)),
        (
            "usage".to_owned(),
            Json::Object(vec![
                ("input_tokens".to_owned(), n(usage.input)),
                (
                    "input_tokens_details".to_owned(),
                    Json::Object(vec![
                        ("cached_tokens".to_owned(), n(usage.cached)),
                        (
                            "cache_write_tokens".to_owned(),
                            n(usage.cache_write.unwrap_or(0)),
                        ),
                    ]),
                ),
                ("output_tokens".to_owned(), n(usage.output)),
                (
                    "output_tokens_details".to_owned(),
                    Json::Object(vec![("reasoning_tokens".to_owned(), n(usage.reasoning))]),
                ),
                (
                    "total_tokens".to_owned(),
                    n(usage.input.saturating_add(usage.output)),
                ),
            ]),
        ),
    ])
}

/// Write the response a `/v1/systemone` caller receives.
pub(crate) fn to_systemone_response(
    model: &str,
    req: &Request,
    answers: &[Answer],
    usage: &DecisionUsage,
) -> Json {
    let answers = systemone_keys(&req.questions)
        .into_iter()
        .zip(answers)
        .filter_map(|(key, a)| {
            let answer = match a {
                Answer::Predicate(p) => vec![
                    ("type".to_owned(), Json::str("noul")),
                    ("noul".to_owned(), Json::number(*p)),
                ],
                Answer::Choice {
                    choice,
                    confidence,
                    probabilities,
                } => vec![
                    ("type".to_owned(), Json::str("choice")),
                    ("choice".to_owned(), Json::str(choice.key())),
                    ("confidence".to_owned(), Json::number(*confidence)),
                    (
                        "probabilities".to_owned(),
                        Json::Object(
                            probabilities
                                .iter()
                                .map(|(v, p)| (v.key(), Json::number(*p)))
                                .collect(),
                        ),
                    ),
                ],
                Answer::Score {
                    score,
                    confidence,
                    probabilities,
                } => vec![
                    ("type".to_owned(), Json::str("score")),
                    ("score".to_owned(), Json::number(*score)),
                    ("confidence".to_owned(), Json::number(*confidence)),
                    (
                        "legend".to_owned(),
                        Json::Object(
                            probabilities
                                .iter()
                                .map(|(i, label, _)| (i.to_string(), label.clone()))
                                .collect(),
                        ),
                    ),
                    (
                        "probabilities".to_owned(),
                        Json::Object(
                            probabilities
                                .iter()
                                .map(|(i, _, p)| (i.to_string(), Json::number(*p)))
                                .collect(),
                        ),
                    ),
                ],
                Answer::Refusal => return None,
            };
            Some((key, Json::Object(answer)))
        })
        .collect();
    let n = |v: u32| Json::Number(u64::from(v).into());
    Json::Object(vec![
        ("model".to_owned(), Json::str(model)),
        ("answers".to_owned(), Json::Object(answers)),
        (
            "usage".to_owned(),
            Json::Object(vec![
                ("input_tokens".to_owned(), n(usage.input)),
                ("output_tokens".to_owned(), n(usage.output)),
            ]),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(s: &str) -> Json {
        serde_json::from_str(s).unwrap()
    }

    fn to_value(j: &Json) -> Value {
        serde_json::to_value(j).unwrap()
    }

    #[test]
    fn ordered_json_round_trips_member_order() {
        let src = r#"{"z":1,"a":[true,null,"x"],"m":{"y":2.5,"b":"c"}}"#;
        assert_eq!(serde_json::to_string(&parse(src)).unwrap(), src);
    }

    #[test]
    fn question_order_survives_both_directions() {
        // Keys deliberately out of alphabetical order: a sorted map would
        // reorder the Decisions questions and their answers.
        let req = parse_systemone_request(&parse(
            r#"{"state":"s","questions":{
                "zeta":{"type":"noul","instructions":"Z?"},
                "alpha":{"type":"choice","criteria":{"b":"B","a":null}}}}"#,
        ))
        .unwrap();
        let out = to_value(&to_decisions_request("gpt-x", &req).unwrap());
        assert_eq!(out["questions"][0]["name"], "zeta");
        assert_eq!(out["questions"][1]["name"], "alpha");
        assert_eq!(
            out["questions"][1]["choices"],
            json!([{"value":"b","description":"B"},{"value":"a"}])
        );
    }

    #[test]
    fn noul_criteria_fold_into_predicate_instructions() {
        let req = parse_systemone_request(&parse(
            r#"{"state":{"ticket":1},"questions":{
                "q":{"type":"noul","instructions":"Is it urgent?",
                     "criteria":{"true":"outage","false":null}},
                "bare":{"type":"noul","criteria":{"true":null}}}}"#,
        ))
        .unwrap();
        let out = to_value(&to_decisions_request("m", &req).unwrap());
        assert_eq!(out["input"], r#"{"ticket":1}"#);
        assert_eq!(
            out["questions"][0]["instructions"],
            "Is it urgent?\n\nAnswer true when: outage"
        );
        assert_eq!(
            out["questions"][1]["instructions"],
            "Is this true of the input?"
        );
    }

    #[test]
    fn duplicate_or_missing_names_fall_back_to_positions() {
        let req = parse_decisions_request(&parse(
            r#"{"input":[{"role":"user","content":[{"type":"input_text","text":"a"},
                         {"type":"input_text","text":"b"}]},{"role":"user","content":"c"}],
               "questions":[{"type":"predicate","name":"x","instructions":"P?"},
                            {"type":"predicate","instructions":"Q?"}]}"#,
        ))
        .unwrap();
        let out = to_value(&to_systemone_request("jev", &req).unwrap());
        assert_eq!(out["state"], "a\n\nb\n\nc");
        assert_eq!(
            out["questions"],
            json!({"0":{"type":"noul","instructions":"P?"},"1":{"type":"noul","instructions":"Q?"}})
        );
        let answers =
            answers_from_systemone(&json!({"answers":{"1":{"type":"noul","noul":0.25}}}), &req);
        let resp = to_value(&to_decisions_response(
            "alias",
            &req,
            &answers,
            &DecisionUsage::default(),
        ));
        assert_eq!(resp["answers"][0], json!({"type":"refusal","name":"x"}));
        assert_eq!(
            resp["answers"][1],
            json!({"type":"predicate","name":null,"probability":0.25})
        );
    }

    #[test]
    fn boolean_choices_travel_as_keys_and_come_back_typed() {
        let req = parse_decisions_request(&parse(
            r#"{"input":"x","questions":[{"type":"choice","name":"ok","instructions":"OK?",
                 "choices":[{"value":true,"description":"yes"},{"value":false}]}]}"#,
        ))
        .unwrap();
        let out = to_value(&to_systemone_request("jev", &req).unwrap());
        assert_eq!(
            out["questions"]["ok"]["criteria"],
            json!({"true":"yes","false":null})
        );
        let answers = answers_from_systemone(
            &json!({"answers":{"ok":{"type":"choice","choice":"false","confidence":0.9,
                    "probabilities":{"true":0.1,"false":0.9}}}}),
            &req,
        );
        let resp = to_value(&to_decisions_response(
            "alias",
            &req,
            &answers,
            &DecisionUsage::default(),
        ));
        assert_eq!(resp["answers"][0]["choice"], json!(false));
        assert_eq!(
            resp["answers"][0]["probabilities"],
            json!([{"value":true,"probability":0.1},{"value":false,"probability":0.9}])
        );
    }

    #[test]
    fn a_boolean_colliding_with_a_string_choice_is_refused() {
        let err = parse_decisions_request(&parse(
            r#"{"input":"x","questions":[{"type":"choice","instructions":"?",
                 "choices":[{"value":true},{"value":"true"}]}]}"#,
        ))
        .err()
        .unwrap();
        assert!(err.contains("unique"), "{err}");
    }

    #[test]
    fn score_levels_round_trip_through_criteria_and_legend() {
        let req = parse_decisions_request(&parse(
            r#"{"input":"x","questions":[{"type":"score","name":"sev","instructions":"How bad?",
                 "levels":[{"label":"low","description":"cosmetic"},{"label":"mid"},{"label":"high"}]}]}"#,
        ))
        .unwrap();
        let out = to_value(&to_systemone_request("jev", &req).unwrap());
        assert_eq!(
            out["questions"]["sev"]["criteria"],
            json!(["low: cosmetic", "mid", "high"])
        );
        let answers = answers_from_systemone(
            &json!({"answers":{"sev":{"type":"score","score":1.9,"confidence":0.8,
                    "legend":{"0":"low: cosmetic","1":"mid","2":"high"},
                    "probabilities":{"1":0.1,"2":0.9}}}}),
            &req,
        );
        let resp = to_value(&to_decisions_response(
            "alias",
            &req,
            &answers,
            &DecisionUsage::default(),
        ));
        assert_eq!(
            resp["answers"][0]["probabilities"],
            json!([{"value":0,"label":"low","probability":0.0},
                   {"value":1,"label":"mid","probability":0.1},
                   {"value":2,"label":"high","probability":0.9}])
        );
    }

    #[test]
    fn decisions_answers_map_back_to_systemone_keys_and_drop_refusals() {
        let req = parse_systemone_request(&parse(
            r#"{"state":"s","questions":{
                "bug":{"type":"noul","instructions":"Bug?"},
                "team":{"type":"choice","criteria":{"billing":null,"eng":null}},
                "sev":{"type":"score","criteria":[{"rank":"low"},"high"]}}}"#,
        ))
        .unwrap();
        let body = json!({"model":"gpt-x","answers":[
            {"type":"refusal","name":"bug"},
            {"type":"choice","name":"team","choice":"eng","confidence":1.0,
             "probabilities":[{"value":"billing","probability":0.0},{"value":"eng","probability":1.0}]},
            {"type":"score","name":"sev","score":0.2,"confidence":0.7,
             "probabilities":[{"value":0,"label":"{\"rank\":\"low\"}","probability":0.8},
                              {"value":1,"label":"high","probability":0.2}]}],
            "usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}});
        let answers = answers_from_decisions(&body, &req);
        let usage = decisions_usage(&body).unwrap();
        let resp = to_value(&to_systemone_response("alias", &req, &answers, &usage));
        assert_eq!(
            resp,
            json!({"model":"alias","answers":{
                "team":{"type":"choice","choice":"eng","confidence":1.0,
                        "probabilities":{"billing":0.0,"eng":1.0}},
                "sev":{"type":"score","score":0.2,"confidence":0.7,
                       "legend":{"0":{"rank":"low"},"1":"high"},
                       "probabilities":{"0":0.8,"1":0.2}}},
                "usage":{"input_tokens":10,"output_tokens":2}})
        );
    }

    #[test]
    fn a_repeated_question_name_is_one_question_and_keeps_its_name() {
        let req = parse_systemone_request(&parse(
            r#"{"state":"s","questions":{"q":{"type":"noul","instructions":"old"},
                "r":{"type":"noul","instructions":"R?"},"q":{"type":"noul","instructions":"new"}}}"#,
        ))
        .unwrap();
        let out = to_value(&to_decisions_request("gpt", &req).unwrap());
        assert_eq!(
            out["questions"],
            json!([{"type":"predicate","name":"r","instructions":"R?"},
                   {"type":"predicate","name":"q","instructions":"new"}])
        );
    }

    #[test]
    fn with_model_replaces_every_model_member() {
        let doc = parse(r#"{"model":"a","x":1,"model":"b"}"#);
        assert_eq!(
            serde_json::to_string(&doc.with_model("up")).unwrap(),
            r#"{"x":1,"model":"up"}"#
        );
    }

    #[test]
    fn lossy_requests_are_refused_rather_than_degraded() {
        let image = parse_decisions_request(&parse(
            r#"{"input":[{"role":"user","content":[{"type":"input_image","image_url":"data:x"}]}],
               "questions":[{"type":"predicate","instructions":"?"}]}"#,
        ))
        .unwrap();
        assert!(to_systemone_request("jev", &image)
            .unwrap_err()
            .contains("input_image"));

        let sid = parse_decisions_request(&parse(
            r#"{"input":"x","safety_identifier":"u1","questions":[{"type":"predicate","instructions":"?"}]}"#,
        ))
        .unwrap();
        assert!(to_systemone_request("jev", &sid)
            .unwrap_err()
            .contains("safety_identifier"));

        let single = parse_systemone_request(&parse(
            r#"{"state":"x","questions":{"c":{"type":"choice","criteria":{"only":null}}}}"#,
        ))
        .unwrap();
        assert!(to_decisions_request("gpt", &single)
            .unwrap_err()
            .contains("at least 2"));
    }
}
