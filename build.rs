use std::fs;

fn main() {
    fs::create_dir_all("ui/dist").unwrap();

    println!("cargo:rerun-if-changed=build.rs");
}
