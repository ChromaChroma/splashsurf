use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::kernel::Kernel;
use opencl3::platform::Platform;
use opencl3::program::{CL_STD_2_0, Program};
use crate::profile;

const SHADER_CODE: &str = include_str!("shaders.cl");
// Kernel function names
pub const TEST_KERNEL: &str = "fill_arr";

#[derive()]
pub struct OpenCLData {
    pub platform: Platform,
    pub device: Device,
    pub context: Context,
    pub program: Program,
    pub kernel: Kernel,
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

    let context = {
        profile!("create context", parent = p);
        // Create a Context on an OpenCL device
        Context::from_device(&device).expect("Context::from_device failed")
    };

    let program = {
        profile!("build program", parent = p);
        // Build the OpenCL program source and create the kernel.
        Program::create_and_build_from_source(&context, SHADER_CODE, CL_STD_2_0)
            .expect("Program::create_and_build_from_source failed")
    };

    let kernel = {
        profile!("create kernel", parent = p);
        Kernel::create(&program, TEST_KERNEL).expect("Kernel::create failed")
    };


    Ok(OpenCLData { platform, device, context, program, kernel })
}