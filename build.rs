use std::{env, fs, path, process};

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

        let output = process::Command::new("pnpm")
            .args(["run", "build"])
            .current_dir("ui")
            .output()?;

        println!(
            "build ui stdout:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );

        // stderr
        eprintln!(
            "build ui stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );

        if !output.status.success() {
            return Err(anyhow!("UI build failed with status: {}", output.status));
        }
    } else {
        fs::create_dir_all(&path::PathBuf::from(env::var("OUT_DIR")?).join("ui/dist"))?;
    }
    Ok(())
}
