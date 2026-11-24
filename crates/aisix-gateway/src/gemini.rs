//! Google Gemini response fields shared by Vertex AI and the Gemini API.

use serde::Deserialize;

use crate::{FinishReason, ServiceTier, UsageStats};

#[derive(Debug, Default, Deserialize)]
pub struct GeminiUsageMetadata {
    /// `None` when omitted, apart from a reported `0`: only an omitted
    /// count is estimated.
    #[serde(default, rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(default, rename = "totalTokenCount")]
    total_token_count: u32,
    /// Context-cache hits, which Gemini reports as a SUBSET of
    /// `promptTokenCount` — the OpenAI accounting shape, so it maps onto
    /// `UsageStats::cached_prompt_tokens` and must not be added to the
    /// prompt total (AISIX-Cloud#1404).
    #[serde(default, rename = "cachedContentTokenCount")]
    cached_content_token_count: u32,
    /// Thinking tokens. Gemini bills these as output but reports them
    /// in their own counter, and — depending on model version — either
    /// inside `candidatesTokenCount` or beside it. OpenAI accounting has
    /// no third bucket: reasoning is a SUBSET of `completion_tokens`.
    /// See [`GeminiUsageMetadata::into_usage_stats`].
    #[serde(default, rename = "thoughtsTokenCount")]
    thoughts_token_count: u32,
    /// Tokens fed back to the model from tool results — INPUT that
    /// `promptTokenCount` does not include. It is also a term in
    /// Gemini's own `totalTokenCount` identity, which is what decides
    /// whether `candidatesTokenCount` already contains the thoughts; a
    /// version of this code that assumed the term away would
    /// double-count thinking tokens on tool-calling turns.
    #[serde(default, rename = "toolUsePromptTokenCount")]
    tool_use_prompt_token_count: u32,
    /// Per-modality breakdowns, `[{modality, tokenCount}]`.
    #[serde(default, rename = "promptTokensDetails")]
    prompt_tokens_details: Option<serde_json::Value>,
    #[serde(default, rename = "candidatesTokensDetails")]
    candidates_tokens_details: Option<serde_json::Value>,
    /// The tier that served the request. Raw, so an unexpected type cannot
    /// fail the decode.
    #[serde(default, rename = "serviceTier")]
    service_tier: Option<serde_json::Value>,
}

impl GeminiUsageMetadata {
    pub fn service_tier(&self) -> Option<ServiceTier> {
        crate::chat::gemini_usage_service_tier(
            self.service_tier
                .as_ref()
                .and_then(serde_json::Value::as_str),
        )
    }

    pub fn reports_counts(&self) -> bool {
        self.prompt_token_count.is_some()
            || self.candidates_token_count.is_some()
            || self.total_token_count > 0
            || self.cached_content_token_count > 0
            || self.thoughts_token_count > 0
            || self.tool_use_prompt_token_count > 0
    }

    /// Project Gemini's counters onto [`UsageStats`]' OpenAI accounting
    /// shape. Shared by the streaming and non-streaming decoders so the
    /// two cannot report a different completion total for the same call.
    ///
    /// `thoughtsTokenCount` is the subtle one. Gemini changed its mind
    /// about whether `candidatesTokenCount` includes it, so neither
    /// answer can be hard-coded; `totalTokenCount` settles it, exactly
    /// as LiteLLM's adapter decides the same question. Left unmapped
    /// (the pre-fix behaviour) a thinking model reported
    /// `reasoning_tokens: 0` and a `completion_tokens` short of what
    /// Google billed, which every client-facing protocol then inherited
    /// (AISIX-Cloud#1447).
    pub fn into_usage_stats(self) -> UsageStats {
        // Gemini's identity is
        //   total = prompt + candidates + toolUsePrompt + thoughts
        // so `candidates` still summing to the total means it has
        // already absorbed the thoughts, and adding them again would
        // bill the thinking twice.
        let candidates_token_count = self.candidates_token_count.unwrap_or(0);
        let prompt_tokens = self
            .prompt_token_count
            .unwrap_or(0)
            .saturating_add(self.tool_use_prompt_token_count);
        //
        // `thoughts <= candidates` is part of the test, not a safety
        // clamp: "the candidates already contain the thoughts" cannot be
        // true of a count smaller than the thoughts it supposedly
        // contains. A streaming frame mid-thinking hits exactly that —
        // thoughts accrue while `candidatesTokenCount` is still 0, and a
        // `totalTokenCount` that has not caught up satisfies the sum. On
        // the inclusive reading that frame yields
        // `reasoning_tokens > completion_tokens`, which cp-api rejects
        // outright, dropping the request from Logs and billing with only
        // a warning. Reading it as exclusive is also simply correct.
        let candidates_inclusive = self.total_token_count > 0
            && prompt_tokens.saturating_add(candidates_token_count) == self.total_token_count
            && self.thoughts_token_count <= candidates_token_count;
        //
        // The fold is client-facing accounting only. The UsageEvent records
        // Gemini's own counters — `candidatesTokenCount` as the completion
        // and `totalTokenCount` as the total — by subtracting
        // `reasoning_folded_into_completion` back out; cp-api reads the
        // total identity itself.
        let reasoning_folded_into_completion = if candidates_inclusive {
            0
        } else {
            self.thoughts_token_count
        };
        let completion_tokens =
            candidates_token_count.saturating_add(reasoning_folded_into_completion);
        UsageStats {
            prompt_tokens,
            completion_tokens,
            reported: crate::ReportedCounts {
                prompt: self.prompt_token_count.is_some(),
                completion: self.candidates_token_count.is_some(),
            },
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            cached_prompt_tokens: self.cached_content_token_count,
            reasoning_tokens: self.thoughts_token_count,
            upstream_total_tokens: self.total_token_count,
            reasoning_folded_into_completion,
            modality: crate::ModalityTokens::from_gemini_details(
                self.prompt_tokens_details.as_ref(),
                self.candidates_tokens_details.as_ref(),
            ),
            ..Default::default()
        }
    }
}

/// The finish reason of a turn that called tools. Gemini reports `STOP`
/// for one, which the OpenAI shape spells `tool_calls`; a turn cut short
/// by the token limit or a block keeps saying so.
pub fn finish_with_tool_calls(reason: Option<&str>) -> FinishReason {
    match reason {
        None | Some("STOP") => FinishReason::ToolCalls,
        other => map_gemini_finish_reason(other),
    }
}

/// Map Gemini's `finishReason` strings to the gateway's enum. Per
/// <https://ai.google.dev/api/generate-content#FinishReason>:
///
/// - `STOP` → `FinishReason::Stop`
/// - `MAX_TOKENS` → `FinishReason::Length`
/// - `SAFETY` / `RECITATION` / `BLOCKLIST` / `PROHIBITED_CONTENT` /
///   `SPII` / `IMAGE_SAFETY` / `IMAGE_PROHIBITED_CONTENT` / `IMAGE_RECITATION`
///   / `IMAGE_OTHER` / `NO_IMAGE` / `LANGUAGE` → `FinishReason::ContentFilter`
/// - `MALFORMED_FUNCTION_CALL` / `UNEXPECTED_TOOL_CALL` / `OTHER` /
///   `FINISH_REASON_UNSPECIFIED` / unknown → `FinishReason::Stop`
///
/// **Audit LOW-3:** `IMAGE_SAFETY` and `LANGUAGE` previously fell
/// through to `Stop` — misleading for tracing, because a customer
/// would see a successful "stop" when Google in fact filtered the
/// response.
pub fn map_gemini_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("STOP") => FinishReason::Stop,
        Some("MAX_TOKENS") => FinishReason::Length,
        Some("SAFETY")
        | Some("RECITATION")
        | Some("BLOCKLIST")
        | Some("PROHIBITED_CONTENT")
        | Some("SPII")
        | Some("IMAGE_SAFETY")
        | Some("IMAGE_PROHIBITED_CONTENT")
        | Some("IMAGE_RECITATION")
        | Some("IMAGE_OTHER")
        | Some("NO_IMAGE")
        | Some("LANGUAGE") => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}
