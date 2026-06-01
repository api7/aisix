//! Compose multiple guardrails into one. First [`GuardrailVerdict::Block`]
//! short-circuits the chain; subsequent guardrails are not consulted.
//! A [`GuardrailVerdict::Rewrite`] propagates the modified payload to all
//! subsequent guardrails via `Cow<ChatFormat>` — the heap allocation is
//! deferred until an actual rewrite occurs.
//! Useful for building a single `Arc<dyn Guardrail>` to hand to the
//! proxy from a config-driven list.

use std::borrow::Cow;
use std::sync::Arc;

use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;

use crate::{Guardrail, GuardrailVerdict, StreamOutputPolicy};

#[derive(Clone)]
pub struct GuardrailChain {
    guardrails: Vec<Arc<dyn Guardrail>>,
}

impl std::fmt::Debug for GuardrailChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&'static str> = self.guardrails.iter().map(|g| g.name()).collect();
        f.debug_struct("GuardrailChain")
            .field("guardrails", &names)
            .finish()
    }
}

impl GuardrailChain {
    pub fn new(guardrails: Vec<Arc<dyn Guardrail>>) -> Self {
        Self { guardrails }
    }

    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.guardrails.len()
    }

    pub fn is_empty(&self) -> bool {
        self.guardrails.is_empty()
    }
}

#[async_trait]
impl Guardrail for GuardrailChain {
    fn name(&self) -> &'static str {
        "chain"
    }

    fn is_empty(&self) -> bool {
        self.guardrails.is_empty()
    }

    /// The strictest streamed-output policy across the chain's members.
    /// If any member wants hold-back, the whole stream holds back and
    /// the full chain's `check_output` runs on the held content.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        self.guardrails
            .iter()
            .map(|g| g.stream_output_policy())
            .fold(
                StreamOutputPolicy::EndOfStreamCheck,
                StreamOutputPolicy::stricter,
            )
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        // `current` starts as a borrow; flips to Owned only if a Rewrite fires.
        // This keeps the common (no-rewrite) path allocation-free.
        let mut current: Cow<'_, ChatFormat> = Cow::Borrowed(req);
        let mut bypass: Option<String> = None;

        for g in &self.guardrails {
            let verdict = g.check_input(current.as_ref()).await;
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block { .. } => return verdict,
                GuardrailVerdict::Bypass { reason } => {
                    // First bypass sticks; downstream guardrails still
                    // get to inspect the request (they may Block).
                    if bypass.is_none() {
                        bypass = Some(reason);
                    }
                }
                GuardrailVerdict::Rewrite { payload } => {
                    // Substitute the rewritten payload for all remaining
                    // guardrails. A Block from a later guardrail still wins.
                    current = Cow::Owned(*payload);
                }
            }
        }

        match current {
            Cow::Owned(rewritten) => {
                // If a Bypass also fired earlier in the chain, surface it in
                // the audit trail. The Rewrite takes precedence for routing
                // but the bypass reason must not disappear from logs.
                if let Some(ref reason) = bypass {
                    tracing::info!(
                        bypass_reason = %reason,
                        "guardrail bypass shadowed by Rewrite verdict; bypass recorded"
                    );
                }
                GuardrailVerdict::Rewrite {
                    payload: Box::new(rewritten),
                }
            }
            Cow::Borrowed(_) => match bypass {
                Some(reason) => GuardrailVerdict::Bypass { reason },
                None => GuardrailVerdict::Allow,
            },
        }
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        let mut bypass: Option<String> = None;
        for g in &self.guardrails {
            let verdict = g.check_output(resp).await;
            match verdict {
                GuardrailVerdict::Allow => continue,
                GuardrailVerdict::Block { .. } => return verdict,
                GuardrailVerdict::Bypass { reason } => {
                    if bypass.is_none() {
                        bypass = Some(reason);
                    }
                }
                // Rewrite on the output path is valid but rare; the chain
                // ignores it (output rewrites would need a mutable `resp`
                // which the trait doesn't provide). Treat as Allow.
                GuardrailVerdict::Rewrite { .. } => {}
            }
        }
        match bypass {
            Some(reason) => GuardrailVerdict::Bypass { reason },
            None => GuardrailVerdict::Allow,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KeywordBlocklist, KeywordRule, MaxContentLength};
    use aisix_gateway::{ChatMessage, FinishReason, UsageStats};

    fn req(msg: &str) -> ChatFormat {
        ChatFormat::new("m", vec![ChatMessage::user(msg)])
    }

    fn resp(content: &str) -> ChatResponse {
        ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: ChatMessage::assistant(content),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(0, 0),
        }
    }

    #[tokio::test]
    async fn empty_chain_allows_everything() {
        let chain = GuardrailChain::empty();
        assert_eq!(chain.check_input(&req("hi")).await, GuardrailVerdict::Allow);
        assert_eq!(
            chain.check_output(&resp("hi")).await,
            GuardrailVerdict::Allow,
        );
    }

    #[tokio::test]
    async fn first_block_short_circuits_subsequent_guardrails() {
        // Both would block on the same input; the first wins so the
        // reason is deterministic.
        let chain = GuardrailChain::new(vec![
            Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("alpha")])),
            Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("beta")])),
        ]);
        let v = chain.check_input(&req("alpha and beta")).await;
        if let GuardrailVerdict::Block { reason } = v {
            assert!(reason.contains("alpha"));
        } else {
            panic!("expected Block");
        }
    }

    #[tokio::test]
    async fn allow_falls_through_to_next_guardrail() {
        // First guardrail allows everything; second blocks on length.
        let chain = GuardrailChain::new(vec![
            Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal(
                "nope-not-here",
            )])),
            Arc::new(MaxContentLength::input_only(5)),
        ]);
        let v = chain.check_input(&req("this is way too long")).await;
        assert!(v.is_block());
    }

    /// Bypass doesn't short-circuit: a downstream Block must still
    /// fire. This is the failure mode that matters when an operator
    /// stacks a Bedrock guardrail (which can bypass on AWS 5xx) on
    /// top of a keyword guardrail (which is local + always available).
    #[tokio::test]
    async fn bypass_does_not_short_circuit_keyword_block() {
        struct AlwaysBypass;
        #[async_trait]
        impl Guardrail for AlwaysBypass {
            fn name(&self) -> &'static str {
                "always-bypass"
            }
            async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
                GuardrailVerdict::Bypass {
                    reason: "test".into(),
                }
            }
        }
        let chain = GuardrailChain::new(vec![
            Arc::new(AlwaysBypass),
            Arc::new(KeywordBlocklist::new(vec![KeywordRule::literal("AKIA")])),
        ]);
        // Bypass first, then a keyword Block — Block must win.
        let v = chain.check_input(&req("here is AKIAEXAMPLE")).await;
        assert!(v.is_block(), "expected Block, got {v:?}");
    }

    /// When no guardrail blocks but at least one bypassed, the chain's
    /// verdict is the first bypass reason — chat handler attaches
    /// it to the telemetry event.
    #[tokio::test]
    async fn bypass_propagates_when_no_block_fires() {
        struct AlwaysBypass(&'static str);
        #[async_trait]
        impl Guardrail for AlwaysBypass {
            fn name(&self) -> &'static str {
                "always-bypass"
            }
            async fn check_input(&self, _req: &ChatFormat) -> GuardrailVerdict {
                GuardrailVerdict::Bypass {
                    reason: self.0.into(),
                }
            }
        }
        let chain = GuardrailChain::new(vec![
            Arc::new(AlwaysBypass("first")),
            Arc::new(AlwaysBypass("second")),
        ]);
        let v = chain.check_input(&req("hello")).await;
        match v {
            GuardrailVerdict::Bypass { reason } => assert_eq!(reason, "first"),
            other => panic!("expected Bypass, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn output_check_short_circuits_on_first_block() {
        let chain = GuardrailChain::new(vec![
            Arc::new(KeywordBlocklist::output_only(vec![KeywordRule::literal(
                "secret",
            )])),
            Arc::new(MaxContentLength::output_only(2)),
        ]);
        // The keyword guardrail fires before length.
        let v = chain.check_output(&resp("the secret answer")).await;
        if let GuardrailVerdict::Block { reason } = v {
            assert!(reason.contains("secret"));
        } else {
            panic!("expected Block");
        }
    }
}
