use aisix::{Args, run};
use anyhow::{Context, Result};
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    aisix::utils::instance::init().context("failed to initialize instance")?;

    run(args.config).await
}
