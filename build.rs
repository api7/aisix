use std::{env, fs, process};

use anyhow::{Result, anyhow};
use vergen_git2::{Emitter, Git2Builder};

fn main() -> Result<()> {
    build_ui()?;

    Emitter::default()
        .add_instructions(&Git2Builder::default().sha(true).build()?)?
        .emit()?;

    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}

fn build_ui() -> Result<()> {
    if env::var("CARGO_FEATURE_BUILD_UI").is_ok() {
        println!("cargo:rerun-if-changed=ui");

        let status = process::Command::new("pnpm")
            .args(["install", "--frozen-lockfile"])
            .current_dir("ui")
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()?;

        if !status.success() {
            return Err(anyhow!(
                "failed to install ui dependencies with status: {}",
                status
            ));
        }

        let status = process::Command::new("pnpm")
            .args(["run", "build"])
            .current_dir("ui")
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()?;

        if !status.success() {
            return Err(anyhow!("failed to build ui with status: {}", status));
        }
    } else {
        fs::create_dir_all("ui/dist")?;
    }
    Ok(())
}
