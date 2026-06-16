//! Ensemble config attached to a [`Model`](super::Model).
//!
//! When a Model carries an `ensemble` block, the proxy fans the request
//! out to every panel member concurrently, then calls the judge model to
//! synthesize a single answer from the panel responses. Unlike a
//! [`Routing`](super::Routing) model — which picks ONE target per request
//! — an ensemble model calls ALL panel members and combines their output.
//!
//! Panel members and the judge reference other (direct) Models by
//! `display_name`, the same way routing targets do. Mutual exclusivity
//! with the direct-upstream fields and `routing` is enforced by the
//! runtime schema (`super::schema`), not by this type.

use serde::{Deserialize, Serialize};

/// One member of an ensemble panel. `model` references a direct model alias.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PanelMember {
    /// Model alias for a direct model that receives one panel request.
    pub model: String,
    /// Sampling temperature for this panel member. Omit it to keep the request's temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Sampling seed for this panel member.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Reserved for the future voting/quorum strategy. Ignored by the v1
    /// synthesis path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<u32>,
}

impl PanelMember {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            temperature: None,
            seed: None,
            weight: None,
        }
    }
}

/// The judge model that synthesizes the panel responses into one answer.
/// `model` references a direct model alias.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Judge {
    /// Model alias for the direct model that synthesizes panel responses.
    pub model: String,
    /// Override for the built-in synthesis prompt template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis_prompt: Option<String>,
}

impl Judge {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            synthesis_prompt: None,
        }
    }
}

/// Default minimum number of successful panel responses required before
/// the judge synthesizes, when the operator does not set `min_responses`.
/// Always clamped to the panel size by [`EnsembleConfig::min_responses_or_default`],
/// so a single-member (self-ensemble) panel still only needs one response.
const DEFAULT_MIN_RESPONSES: usize = 2;

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EnsembleConfig {
    /// Direct models called concurrently for each ensemble request.
    pub panel: Vec<PanelMember>,
    /// Direct model that combines successful panel responses.
    pub judge: Judge,
    /// Minimum successful panel responses required before judge synthesis. When omitted, the gateway requires the smaller of 2 and the panel size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_responses: Option<u32>,
    /// Per-call upstream deadline applied to each panel member and the judge. Set `0` or omit it to disable the ensemble-level deadline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

impl EnsembleConfig {
    /// Effective minimum successful panel responses. Defaults to
    /// [`DEFAULT_MIN_RESPONSES`], never exceeds the panel size, and is at
    /// least 1 for a non-empty panel.
    pub fn min_responses_or_default(&self) -> usize {
        let requested = self.min_responses.map(|n| n as usize);
        requested
            .unwrap_or(DEFAULT_MIN_RESPONSES)
            .min(self.panel.len())
            .max(1)
    }

    /// Per-call upstream deadline applied to each panel member and the
    /// judge call. Folds the `0`/absent sentinel into `None` like
    /// [`Model::request_timeout`](super::Model::request_timeout) so callers
    /// can apply it unconditionally.
    pub fn timeout(&self) -> Option<std::time::Duration> {
        self.timeout_ms
            .filter(|&ms| ms > 0)
            .map(std::time::Duration::from_millis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_full_ensemble_block() {
        let json = r#"{
            "panel": [
                {"model": "gpt", "temperature": 0.5, "seed": 7, "weight": 1},
                {"model": "claude", "temperature": 1.0}
            ],
            "judge": {"model": "opus", "synthesis_prompt": "combine them"},
            "min_responses": 2,
            "timeout_ms": 45000
        }"#;
        let e: EnsembleConfig = serde_json::from_str(json).unwrap();
        assert_eq!(e.panel.len(), 2);
        assert_eq!(e.panel[0].model, "gpt");
        assert_eq!(e.panel[0].temperature, Some(0.5));
        assert_eq!(e.panel[0].seed, Some(7));
        assert_eq!(e.panel[0].weight, Some(1));
        assert_eq!(e.panel[1].temperature, Some(1.0));
        assert_eq!(e.judge.model, "opus");
        assert_eq!(e.judge.synthesis_prompt.as_deref(), Some("combine them"));
        assert_eq!(e.min_responses_or_default(), 2);
        assert_eq!(e.timeout(), Some(std::time::Duration::from_millis(45_000)));
    }

    #[test]
    fn minimal_ensemble_block_uses_defaults() {
        let e: EnsembleConfig = serde_json::from_str(
            r#"{"panel":[{"model":"a"},{"model":"b"},{"model":"c"}],"judge":{"model":"j"}}"#,
        )
        .unwrap();
        // Default min_responses is DEFAULT_MIN_RESPONSES (2), not the panel size.
        assert_eq!(e.min_responses_or_default(), 2);
        assert!(e.panel[0].temperature.is_none());
        assert!(e.judge.synthesis_prompt.is_none());
        assert_eq!(e.timeout(), None);
        assert!(!e.panel.is_empty());
    }

    #[test]
    fn min_responses_clamps_to_panel_size() {
        let e: EnsembleConfig = serde_json::from_str(
            r#"{"panel":[{"model":"a"},{"model":"b"}],"judge":{"model":"j"},"min_responses":10}"#,
        )
        .unwrap();
        assert_eq!(e.min_responses_or_default(), 2);
    }

    #[test]
    fn single_member_panel_defaults_to_one_response() {
        // Self-ensemble shrunk to one member: default min(2, 1) = 1.
        let e: EnsembleConfig =
            serde_json::from_str(r#"{"panel":[{"model":"solo"}],"judge":{"model":"j"}}"#).unwrap();
        assert_eq!(e.min_responses_or_default(), 1);
    }

    #[test]
    fn timeout_zero_folds_to_none() {
        let e: EnsembleConfig = serde_json::from_str(
            r#"{"panel":[{"model":"a"}],"judge":{"model":"j"},"timeout_ms":0}"#,
        )
        .unwrap();
        assert_eq!(e.timeout(), None);
    }

    #[test]
    fn rejects_unknown_ensemble_field() {
        let r: Result<EnsembleConfig, _> =
            serde_json::from_str(r#"{"panel":[{"model":"a"}],"judge":{"model":"j"},"foo":1}"#);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_unknown_panel_member_field() {
        let r: Result<PanelMember, _> = serde_json::from_str(r#"{"model":"a","bogus":true}"#);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_unknown_judge_field() {
        let r: Result<Judge, _> = serde_json::from_str(r#"{"model":"j","bogus":true}"#);
        assert!(r.is_err());
    }

    #[test]
    fn panel_member_new_has_no_overrides() {
        let m = PanelMember::new("x");
        assert_eq!(m.model, "x");
        assert!(m.temperature.is_none());
        assert!(m.seed.is_none());
        assert!(m.weight.is_none());
    }
}
