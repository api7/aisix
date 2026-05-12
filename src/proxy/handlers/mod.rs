pub mod chat_completions;
pub mod embeddings;
mod format_handler;
pub mod messages;
pub mod models;
pub(crate) mod openai_error;
pub mod responses;

pub(crate) use format_handler::{FormatHandlerAdapter, format_handler};
