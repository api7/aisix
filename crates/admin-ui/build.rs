use std::{env, fs, process};

use anyhow::{Result, bail};

fn main() -> Result<()> {
    build_ui()?;

    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}

fn build_ui() -> Result<()> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    fs::create_dir_all(format!("{manifest_dir}/ui-dist"))?;

    if env::var("CARGO_FEATURE_BUILD_UI").is_ok() {
        println!("cargo:rerun-if-changed=ui/");

        let status = process::Command::new("pnpm")
            .args(["install", "--frozen-lockfile"])
            .current_dir(format!("{manifest_dir}/ui"))
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()?;

        if !status.success() {
            bail!("failed to install ui dependencies with status: {}", status);
        }

        let status = process::Command::new("pnpm")
            .args(["run", "build"])
            .current_dir(format!("{manifest_dir}/ui"))
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()?;

        if !status.success() {
            bail!("failed to build ui with status: {}", status);
        }
    }
    Ok(())
}
