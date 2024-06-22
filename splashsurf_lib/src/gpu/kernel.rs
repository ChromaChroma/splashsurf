use std::collections::HashMap;

use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::kernel::Kernel;
use opencl3::platform::Platform;
use opencl3::program::{CL_STD_2_0, Program};

use crate::profile;

const SHADER_CODE: &str = include_str!("shaders.cl");
// Kernel function names
pub const COMPUTE_BOUNDS_FUNCTION: &str = "compute_lower_and_upper";
pub const DENSITY_GRID_LOOP_FUNCTION: &str = "density_grid_loop";
pub const RECONSTRUCT_FUNCTION: &str = "reconstruct";

#[derive()]
pub struct OpenCLData {
    pub platform: Platform,
    pub device: Device,
    pub context: Context,
    pub program: Program,
    pub kernels: HashMap<&'static str, Kernel>,
    // pub kernel: Kernel,
}

pub fn init_kernels() -> opencl3::Result<OpenCLData> {
    profile!(p, "GPU Kernel initialization");

    let platform = {
        profile!("load platform", parent = p);
        opencl3::platform::get_platforms()
            .expect("No OpenCL platforms found.")
            .first()
            .expect("No OpenCL (first) platform found.")
            .clone()
    };

    let device = {
        profile!("get device", parent = p);
        let device = *platform
            .get_devices(CL_DEVICE_TYPE_GPU)?
            .first()
            .expect("No device found in platform");
        Device::new(device)
    };

    // Create a Context on an OpenCL device
    let context = {
        profile!("create context", parent = p);
        Context::from_device(&device).expect("Context::from_device failed")
    };

    // Build the OpenCL program source and create the kernel.
    let program = {
        profile!("build program", parent = p);
        Program::create_and_build_from_source(&context, SHADER_CODE, CL_STD_2_0)
            .expect("Program::create_and_build_from_source failed")
    };
    {
        profile!("create kernels", parent = p);
        // let lower_upper_kernel = {
        //     Kernel::create(&program, COMPUTE_BOUNDS_FUNCTION)
        //         .expect(format!("Kernel::create failed: {}", COMPUTE_BOUNDS_FUNCTION).as_str())
        // };
        // let density_grid_kernel = {
        //     Kernel::create(&program, DENSITY_GRID_LOOP_FUNCTION)
        //         .expect(format!("Kernel::create failed: {}", DENSITY_GRID_LOOP_FUNCTION).as_str())
        // };

        let reconstruct_kernel = {
            Kernel::create(&program, RECONSTRUCT_FUNCTION)
                .expect(format!("Kernel::create failed: {}", RECONSTRUCT_FUNCTION).as_str())
        };

        Ok(OpenCLData {
            platform,
            device,
            context,
            program,
            kernels: HashMap::from([
                (RECONSTRUCT_FUNCTION, reconstruct_kernel),
                // (COMPUTE_BOUNDS_FUNCTION, lower_upper_kernel),
                // (DENSITY_GRID_LOOP_FUNCTION, density_grid_kernel),
            ]),
        })
    }
}