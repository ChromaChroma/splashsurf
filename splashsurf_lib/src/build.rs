use spirv_builder::{MetadataPrintout, SpirvBuilder};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    SpirvBuilder::new("shaders/reconstruction", "spirv-unknown-vulkan1.1")
        .print_metadata(MetadataPrintout::Full)
        .build()?;
    Ok(())
}
// fn main() {
//     // Tell Cargo that if the given file changes, to rerun this build script.
//     println!("cargo::rerun-if-changed=src/shaders/reconstruction/**");
//
//
//
// }