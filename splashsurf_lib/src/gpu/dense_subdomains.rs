use std::ptr;
use std::sync::{Arc, Mutex};
use log::info;
use nalgebra::Vector3;
use opencl3::command_queue::{cl_event, CL_NON_BLOCKING, CL_QUEUE_ON_DEVICE, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, cl_double, Device};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::{CL_STD_2_0, Program};
use opencl3::types::cl_int;
use crate::{Index, OpenCLData, profile, Real};
use crate::dense_subdomains::{ParametersSubdomainGrid, subdomain_classification, Subdomains};
use crate::gpu::kernel::init_kernels;

/// Performs classification and decomposition of particles into a regular grid of subdomains
pub(crate) fn decomposition<
    I: Index,
    R: Real,
    C: subdomain_classification::ParticleToSubdomainClassifier<I, R>,
>(parameters: &ParametersSubdomainGrid<I, R>,
  particles: &[Vector3<R>],
  ocl_data: Arc<Mutex<OpenCLData>>,
) -> Result<Subdomains<I>, anyhow::Error> {
    profile!(p, "GPU::decomposition");
    info!("GPU::Starting classification of particles into subdomains.");

    const WORKGROUP_SIZE: usize = 262_144;

    // Create a command_queue on the Context's device
    let queue = {
        profile!("create command queue", parent = p);
        CommandQueue::create_default(
            &ocl_data.lock().unwrap().context, CL_QUEUE_PROFILING_ENABLE,
        ).expect("CommandQueue::create_default failed")
    };


    // Output Buffers
    // let levelset_grid_f64_arr: &[cl_double] = &*convert_slice_to_cl_double(&levelset_grid_f64);

    let mut output_buffer = unsafe {
        profile!("create output buffer", parent = p);
        Buffer::<cl_int>::create(&ocl_data.lock().unwrap().context, CL_MEM_READ_WRITE, WORKGROUP_SIZE, ptr::null_mut()).expect("Could not create output_buffer")
    };
    let _output_buffer_write_event = unsafe {
        profile!("enqueue write output buffer", parent = p);
        queue.enqueue_write_buffer(&mut output_buffer, CL_NON_BLOCKING, 0, &[0; WORKGROUP_SIZE], &[]).expect("Could not enqueue output_buffer")
    };

    let kernel_event = unsafe {
        profile!("create and enqueue kernel", parent = p);
        ExecuteKernel::new(&ocl_data.lock().unwrap().kernel)
            .set_arg(&output_buffer)
            .set_global_work_size(WORKGROUP_SIZE)
            .set_wait_event(&_output_buffer_write_event)
            .enqueue_nd_range(&queue)
            .expect("Could not run Kernel")
    };

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    let mut results = &mut vec![0 as cl_int; WORKGROUP_SIZE];
    {
        profile!("enqueue read buffer", parent = p);
        let read_event = unsafe {
            queue.enqueue_read_buffer(&output_buffer, CL_NON_BLOCKING, 0, &mut results, &events).expect("Could not enqueue result read buffer")
        };
        // Wait for the read_event to complete.
        read_event.wait().expect("Could not read event for retrieveing data from gpu bufer");
    }

    // println!("{:?}", results);

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start().unwrap();
    let end_time = kernel_event.profiling_command_end().unwrap();
    let duration = end_time - start_time;
    println!("kernel execution duration (ns): {}", duration);


    // // Create a command_queue on the Context's device
    // let queue = CommandQueue::create_default(
    //     &ocl_data.lock().unwrap().context, CL_QUEUE_PROFILING_ENABLE,
    // ).expect("CommandQueue::create_default failed");
    //
    // const WORKGROUP_SIZE: usize = 2048;
    //
    //
    // // Output Buffers
    // // let levelset_grid_f64_arr: &[cl_double] = &*convert_slice_to_cl_double(&levelset_grid_f64);
    //
    // let mut output_buffer = unsafe {
    //     Buffer::<cl_int>::create(&ocl_data.lock().unwrap().context, CL_MEM_READ_WRITE, WORKGROUP_SIZE, ptr::null_mut()).expect("Could not create output_buffer")
    // };
    // let _output_buffer_write_event = unsafe {
    //     queue.enqueue_write_buffer(&mut output_buffer, CL_NON_BLOCKING, 0, &[0; WORKGROUP_SIZE], &[]).expect("Could not enqueue output_buffer")
    // };
    //
    // let kernel_event = unsafe {
    //     ExecuteKernel::new(&ocl_data.lock().unwrap().kernel)
    //         .set_arg(&output_buffer)
    //         .set_global_work_size(WORKGROUP_SIZE)
    //         .set_wait_event(&_output_buffer_write_event)
    //         .enqueue_nd_range(&queue)
    //         .expect("Could not run Kernel")
    // };
    //
    // let mut events: Vec<cl_event> = Vec::default();
    // events.push(kernel_event.get());
    //
    // let mut results = &mut vec![0 as cl_int; WORKGROUP_SIZE];
    // let read_event = unsafe { queue.enqueue_read_buffer(&output_buffer, CL_NON_BLOCKING, 0, &mut results, &events).expect("Could not enqueue result read buffer") };
    // // Wait for the read_event to complete.
    // read_event.wait().expect("Could not read event for retrieveing data from gpu bufer");
    //
    // println!("{:?}", results);
    //
    // // Calculate the kernel duration, from the kernel_event
    // let start_time = kernel_event.profiling_command_start().unwrap();
    // let end_time = kernel_event.profiling_command_end().unwrap();
    // let duration = end_time - start_time;
    // println!("kernel execution duration (ns): {}", duration);


    // init_kernels().expect(" Gpu failed initializing");
    // panic!(" Well Done!")

    Ok(Subdomains {
        flat_subdomain_indices: Vec::new(),
        per_subdomain_particles: Vec::new(),
    })
}
