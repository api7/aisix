mod apikeys_api;
mod auth;
mod models_api;
mod ui;

use std::sync::Arc;

use axum::Router;
use uuid::Uuid;

pub const TEST_ADMIN_KEY: &str = "test_admin_key";

pub async fn create_router(etcd_prefix: Option<&str>) -> Router {
    let config = Arc::new(aisix::config::Config {
        deployment: aisix::config::Deployment {
            etcd: aisix::config::etcd::Config {
                host: vec!["http://127.0.0.1:2379".to_string()],
                prefix: etcd_prefix
                    .unwrap_or(&format!("/test/{}", Uuid::new_v4()))
                    .to_string(),
                timeout: 5,
                user: None,
                password: None,
            },
            admin: aisix::config::DeploymentAdmin {
                admin_key: vec![aisix::config::AdminKey {
                    key: TEST_ADMIN_KEY.to_string(),
                }],
            },
        },
        server: aisix::config::Server {
            proxy: aisix::config::ServerProxy {
                listen: "127.0.0.1:3000".parse().unwrap(),
                tls: aisix::config::ServerCommonTls {
                    enabled: false,
                    cert_file: None,
                    key_file: None,
                },
            },
            admin: aisix::config::ServerAdmin {
                listen: "127.0.0.1:3001".parse().unwrap(),
                tls: aisix::config::ServerCommonTls {
                    enabled: false,
                    cert_file: None,
                    key_file: None,
                },
            },
        },
    });

    let config_provider = Arc::new(
        aisix::config::etcd::EtcdConfigProvider::new(config.deployment.etcd.clone())
            .await
            .unwrap(),
    );

    let state = aisix::admin::AppState::new(
        config,
        config_provider.clone(),
        Arc::new(aisix::config::entities::ResourceRegistry::new(config_provider).await),
        None,
    );

    aisix::admin::create_router(state)
}
