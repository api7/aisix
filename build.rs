use std::{env, fs, process};

use anyhow::{anyhow, Result};
use vergen_git2::{Emitter, Git2Builder};

fn main() -> Result<()> {
    build_ui()?;

    // When built as a git dependency (no .git directory present), vergen_git2
    // cannot read the SHA. We emit a fallback value and continue rather than
    // failing the build.
    let result = Emitter::default()
        .add_instructions(&Git2Builder::default().sha(true).build()?)?
        .emit();

    if result.is_err() {
        // Emit a placeholder so that code referencing VERGEN_GIT_SHA compiles.
        println!("cargo:rustc-env=VERGEN_GIT_SHA=unknown");
    }

    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}

fn build_ui() -> Result<()> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    if env::var("CARGO_FEATURE_BUILD_UI").is_ok() {
        println!("cargo:rerun-if-changed=ui");

        let status = process::Command::new("pnpm")
            .args(["install", "--frozen-lockfile"])
            .current_dir(format!("{manifest_dir}/ui"))
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
            .current_dir(format!("{manifest_dir}/ui"))
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()?;

        if !status.success() {
            return Err(anyhow!("failed to build ui with status: {}", status));
        }
    } else {
        fs::create_dir_all(format!("{manifest_dir}/ui/dist"))?;
    }
    Ok(())
}
