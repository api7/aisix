use log::info;

use crate::{config::entities::ResourceRegistry, handlers::AppState};

mod config;
mod handlers;
mod providers;

#[tokio::main]
async fn main() {
    env_logger::init();

    let config = config::load().expect("Failed to load configuration");
    info!("Loaded config: {:?}", config);

    let config_provider = config::create_provider(config.clone()).await;
    let resources = ResourceRegistry::init(config_provider).await;

    serve(handlers::AppState::new(config.clone(), resources.clone())).await;
}

async fn serve(state: AppState) {
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    info!("Server listening on http://0.0.0.0:3000");

    let _ = tokio::join!(axum::serve(listener, handlers::create_router(state),),);
}
