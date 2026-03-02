use std::sync::Arc;

use axum::Router;

mod auth;

pub const TEST_ADMIN_KEY: &str = "test_admin_key";

pub async fn create_router() -> Router {
    let config = ai_gateway::config::Config {
        deployment: ai_gateway::config::Deployment {
            etcd: ai_gateway::config::etcd::Config {
                host: vec!["http://localhost:2379".to_string()],
                prefix: "/test".to_string(),
                timeout: 5,
                user: None,
                password: None,
            },
            admin: Some(ai_gateway::config::DeploymentAdmin {
                admin_key: Some(vec![ai_gateway::config::AdminKey {
                    key: TEST_ADMIN_KEY.to_string(),
                }]),
            }),
        },
    };

    let config_provider = Arc::new(
        ai_gateway::config::etcd::EtcdConfigProvider::new(config.clone().deployment.etcd)
            .await
            .unwrap(),
    );

    let state = ai_gateway::admin::AppState::new(
        config,
        config_provider.clone(),
        Arc::new(ai_gateway::config::entities::ResourceRegistry::new(config_provider).await),
    );

    ai_gateway::admin::create_router(state)
}
