
// fn main() {
//     // println!("cargo::rerun-if-changed=src/shaders/reconstruction.cl");
//     shaders_code = include_str!("shaders/reconstruct.cl");
//     const KERNEL_NAME2: &str = "reconstruct";
//
//
//     let platforms = opencl3::platform::get_platforms()?;
//     let platform = platforms.first().expect("No OpenCL platforms found.");
//     let device = *platform
//         .get_devices(CL_DEVICE_TYPE_GPU)?
//         .first().expect("no device found in platform");
//     let device = Device::new(device);
//
//     // Create a Context on an OpenCL device
//     let context = Context::from_device(&device).expect("Context::from_device failed");
//
//     // Create a command_queue on the Context's device
//     let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
//         .expect("CommandQueue::create_default failed");
//
//     // Build the OpenCL program source and create the kernel.
//     let program = Program::create_and_build_from_source(&context, shaders_code, CL_STD_2_0)
//         .expect("Program::create_and_build_from_source failed");
//     const kernel: Kernel = Kernel::create(&program, KERNEL_NAME2).expect("Kernel::create failed");
// }

