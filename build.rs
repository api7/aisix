use std::fs;

use vergen_git2::{Emitter, Git2Builder};

fn main() -> anyhow::Result<()> {
    fs::create_dir_all("ui/dist").unwrap();

    Emitter::default()
        .add_instructions(&Git2Builder::default().sha(true).build()?)?
        .emit()?;

    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}
