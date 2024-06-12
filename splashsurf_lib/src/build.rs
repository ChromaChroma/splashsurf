// use spirv_builder::{MetadataPrintout, SpirvBuilder};
// fn main() -> Result<(), Box<dyn std::error::Error>> {
// // println!("cargo::rerun-if-changed=src/shaders/reconstruction/**");
//     SpirvBuilder::new("shaders/reconstruction", "spirv-unknown-vulkan1.1")
//         .print_metadata(MetadataPrintout::Full)
//         .build()?;
//     Ok(())
// }