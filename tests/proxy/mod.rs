mod timeout;

use std::sync::Arc;

use axum::Router;
use uuid::Uuid;

pub const TEST_ADMIN_KEY: &str = "test_admin_key";

pub async fn create_routers(etcd_prefix: Option<&str>) -> (Router, Router) {
    let config = Arc::new(ai_gateway::config::Config {
        deployment: ai_gateway::config::Deployment {
            etcd: ai_gateway::config::etcd::Config {
                host: vec!["http://127.0.0.1:2379".to_string()],
                prefix: etcd_prefix
                    .unwrap_or(&format!("/{}", Uuid::new_v4()))
                    .to_string(),
                timeout: 5,
                user: None,
                password: None,
            },
            admin: Some(ai_gateway::config::DeploymentAdmin {
                listen: "127.0.0.1:3001".parse().unwrap(),
                admin_key: Some(vec![ai_gateway::config::AdminKey {
                    key: TEST_ADMIN_KEY.to_string(),
                }]),
            }),
        },
        listen: "127.0.0.1:3000".parse().unwrap(),
    });

    let config_provider = Arc::new(
        ai_gateway::config::etcd::EtcdConfigProvider::new(config.deployment.etcd.clone())
            .await
            .unwrap(),
    );
    let resources = Arc::new(
        ai_gateway::config::entities::ResourceRegistry::new(config_provider.clone()).await,
    );

    let admin_router = ai_gateway::admin::create_router(ai_gateway::admin::AppState::new(
        config.clone(),
        config_provider,
        resources.clone(),
        None,
    ));
    let proxy_router =
        ai_gateway::proxy::create_router(ai_gateway::proxy::AppState::new(config, resources));

    (admin_router, proxy_router)
}
