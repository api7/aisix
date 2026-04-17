use std::process::exit;

use aisix::{Args, run};
use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if let Err(e) = run(args.config).await {
        eprintln!("Error: {e:#}");
        exit(1);
    }
    Ok(())
}
