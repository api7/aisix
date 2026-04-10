mod auth;
mod trace;

pub use auth::auth;
pub use trace::trace;

#[derive(Clone)]
pub struct RequestModel(pub String);
