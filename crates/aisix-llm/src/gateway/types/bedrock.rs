use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockConverseRequest {
    pub messages: Vec<BedrockMessage>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<BedrockSystemContentBlock>>,

    #[serde(rename = "inferenceConfig", skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<BedrockInferenceConfig>,

    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<BedrockToolConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockMessage {
    pub role: BedrockRole,
    pub content: Vec<BedrockContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BedrockRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockSystemContentBlock {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockInferenceConfig {
    #[serde(rename = "maxTokens", skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockToolConfig {
    pub tools: Vec<BedrockTool>,

    #[serde(rename = "toolChoice", skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<BedrockToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockTool {
    #[serde(rename = "toolSpec")]
    pub tool_spec: BedrockToolSpecification,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockToolSpecification {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(rename = "inputSchema")]
    pub input_schema: BedrockToolInputSchema,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockToolInputSchema {
    pub json: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum BedrockToolChoice {
    Auto { auto: BedrockEmptyObject },
    Any { any: BedrockEmptyObject },
    Tool { tool: BedrockSpecificToolChoice },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct BedrockEmptyObject {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BedrockSpecificToolChoice {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum BedrockContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        #[serde(rename = "toolUse")]
        tool_use: BedrockToolUseBlock,
    },
    ToolResult {
        #[serde(rename = "toolResult")]
        tool_result: BedrockToolResultBlock,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockToolUseBlock {
    #[serde(rename = "toolUseId")]
    pub tool_use_id: String,
    pub name: String,
    pub input: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockToolResultBlock {
    #[serde(rename = "toolUseId")]
    pub tool_use_id: String,
    pub content: Vec<BedrockToolResultContentBlock>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum BedrockToolResultContentBlock {
    Json { json: Value },
    Text { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockConverseResponse {
    pub output: BedrockConverseOutput,

    #[serde(rename = "stopReason", skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<BedrockUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockConverseOutput {
    pub message: BedrockMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BedrockUsage {
    #[serde(rename = "inputTokens")]
    pub input_tokens: u32,

    #[serde(rename = "outputTokens")]
    pub output_tokens: u32,

    #[serde(rename = "totalTokens")]
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockConverseStreamFrame {
    #[serde(rename = "type")]
    pub event_type: String,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BedrockMessageStartEvent {
    pub role: BedrockRole,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockContentBlockStartEvent {
    #[serde(rename = "contentBlockIndex")]
    pub content_block_index: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<BedrockContentBlockStart>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct BedrockContentBlockStart {
    #[serde(rename = "toolUse", skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<BedrockToolUseBlockStart>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BedrockToolUseBlockStart {
    #[serde(rename = "toolUseId")]
    pub tool_use_id: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockContentBlockDeltaEvent {
    #[serde(rename = "contentBlockIndex")]
    pub content_block_index: usize,
    pub delta: BedrockContentBlockDelta,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct BedrockContentBlockDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(rename = "toolUse", skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<BedrockToolUseBlockDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BedrockToolUseBlockDelta {
    pub input: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BedrockContentBlockStopEvent {
    #[serde(rename = "contentBlockIndex")]
    pub content_block_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BedrockMessageStopEvent {
    #[serde(rename = "stopReason", skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,

    #[serde(
        rename = "additionalModelResponseFields",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_model_response_fields: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct BedrockConverseStreamMetadataEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<BedrockUsage>,
}
