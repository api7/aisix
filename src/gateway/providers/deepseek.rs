use crate::gateway::providers::macros::provider;

provider!(DeepSeek {
    display_name: "deepseek",
    base_url: "https://api.deepseek.com",
    auth: bearer,
});
