use std::ptr;

use opencl3::command_queue::{CL_BLOCKING, CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::event::{cl_int, Event};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, ClMem};
use opencl3::platform::Platform;
use opencl3::program::{CL_STD_2_0, Program};
use opencl3::Result;
use opencl3::types::{cl_double, cl_long, cl_mem_flags, CL_NON_BLOCKING, cl_ulong};
use opencl3::types::cl_event;

pub struct KernelData {
    pub platform: Platform,
    pub device: Device,
    pub context: Context,
    pub program: Program,
    pub kernel: Kernel,
}


pub(crate) fn init_kernel() -> Result<KernelData> {
    let shaders_code = include_str!("shaders/shaders.cl");
    const KERNEL_NAME: &str = "reconstruct2";

    let platforms = opencl3::platform::get_platforms()?;
    let platform = platforms.first().expect("No OpenCL platforms found.");
    let device = *platform
        .get_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("No device found in platform");
    let device = Device::new(device);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, shaders_code, CL_STD_2_0)
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    Ok(KernelData {
        platform: *platform,
        device,
        context,
        program,
        kernel,
    })
}

fn usize_diff<const N: usize>(lower: [u64; N], upper: [u64; N]) -> [usize; N] {
    upper.into_iter().zip(lower)
        .map(|(a, b)| (a - b) as usize)
        .collect::<Vec<usize>>()
        .try_into()
        .expect("Could not compute lower/upper diff as usize")
}

pub(crate) fn combine_cl_int_args(
    [subdomain_ijk_x, subdomain_ijk_y, subdomain_ijk_z]: [u64; 3],
    [cells_per_subdomain_x, cells_per_subdomain_y, cells_per_subdomain_z]: [u64; 3],
    [n_points_per_dim_x, n_points_per_dim_y, n_points_per_dim_z]: [u64; 3],
    [global_marching_cubes_grid__np_x, global_marching_cubes_grid__np_y, global_marching_cubes_grid__np_z]: [u64; 3],
) -> [cl_int; 12] {
    [subdomain_ijk_x as cl_int,
        subdomain_ijk_y as cl_int,
        subdomain_ijk_z as cl_int,
        cells_per_subdomain_x as cl_int,
        cells_per_subdomain_y as cl_int,
        cells_per_subdomain_z as cl_int,
        n_points_per_dim_x as cl_int,
        n_points_per_dim_y as cl_int,
        n_points_per_dim_z as cl_int,
        global_marching_cubes_grid__np_x as cl_int,
        global_marching_cubes_grid__np_y as cl_int,
        global_marching_cubes_grid__np_z as cl_int
    ]
}

pub(crate) fn prepare_cl_int_buffer<const N: usize>(
    context: &Context,
    queue: &CommandQueue,
    cl_int_args: [cl_int; N],
) -> Result<(Buffer<cl_int>, Event)> {
    let mut p_i_buffer = unsafe {
        Buffer::<cl_int>::create(context, CL_MEM_READ_ONLY, N, ptr::null_mut())?
    };
    let p_i_event = unsafe {
        (*queue).enqueue_write_buffer(&mut p_i_buffer, CL_BLOCKING, 0, &cl_int_args, &[])?
    };
    Ok((p_i_buffer, p_i_event))
}

pub(crate) fn combine_cl_double_args(
    [gmcg_aabb_min_x, gmcg_aabb_min_y, gmcg_aabb_min_z]: [f64; 3],
    gmcg_cell_size: f64,
    [p_i_x, p_i_y, p_i_z]: [f64; 3],
    rho_i: f64,
    [p0, p1, p2, p3, p4]: [f64; 5],
) -> [cl_double; 13] {
    [gmcg_aabb_min_x as cl_double,
        gmcg_aabb_min_y as cl_double,
        gmcg_aabb_min_z as cl_double,
        gmcg_cell_size as cl_double,
        p_i_x as cl_double,
        p_i_y as cl_double,
        p_i_z as cl_double,
        rho_i as cl_double,
        p0 as cl_double,
        p1 as cl_double,
        p2 as cl_double,
        p3 as cl_double,
        p4 as cl_double,
    ]
}

pub(crate) fn prepare_cl_double_buffer<const N: usize>(
    context: &Context,
    queue: &CommandQueue,
    cl_int_args: [cl_double; N],
) -> Result<(Buffer<cl_double>, Event)> {
    let mut p_i_buffer = unsafe {
        Buffer::<cl_double>::create(context, CL_MEM_READ_ONLY, N, ptr::null_mut())?
    };
    let p_i_event = unsafe {
        (*queue).enqueue_write_buffer(&mut p_i_buffer, CL_BLOCKING, 0, &cl_int_args, &[])?
    };

    Ok((p_i_buffer, p_i_event))
}

pub(crate) fn gpu_small_reconstruct(
    kernel_data: &KernelData,
    // levelset_grid_f64:  &mut Vec<f64>,
    levelset_grid_f64: &[f64],

    // Range
    lower: [u64; 3],
    upper: [u64; 3],

    //-- Sameover all
    subdomain_ijk: [u64; 3],
    cells_per_subdomain: [u64; 3],
    n_points_per_dim: [u64; 3],

    // Global -- Sameover all
    global_marching_cubes_grid__np: [u64; 3],
    global_marching_cubes_grid__aabb_min: [f64; 3],
    global_marching_cubes_grid__cell_size: f64,
    p_i: [f64; 3],
    rho_i: f64,
    parameters: [f64; 5],
) -> Result<Vec<cl_double>> {
    let shaders_code = include_str!("shaders/reconstruct.cl");
    const KERNEL_NAME2: &str = "reconstruct";

    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, shaders_code, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME2).expect("Kernel::create failed");


    // let KernelData {
    //     platform,
    //     device,
    //     context,
    //     program,
    //     kernel
    // } = kernel_data;
    // // Create a command_queue on the Context's device
    // let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
    //     .expect("CommandQueue::create_default failed");


    ////

    let cl_int_args = combine_cl_int_args(
        subdomain_ijk,
        cells_per_subdomain,
        n_points_per_dim,
        global_marching_cubes_grid__np,
    );
    let (int_buffer, _int_args_event) = prepare_cl_int_buffer(
        &context,
        &queue,
        cl_int_args,
    ).expect("Could not combine cl_int values into one buffer");


    let cl_double_args = combine_cl_double_args(
        global_marching_cubes_grid__aabb_min,
        global_marching_cubes_grid__cell_size,
        p_i,
        rho_i,
        parameters,
    );
    let (float_buffer, _float_args_event) = prepare_cl_double_buffer(
        &context,
        &queue,
        cl_double_args,
    ).expect("Could not combine cl_double values into one buffer");


    ////////////////
    // Compute data

    // let output_size: usize = delta_workgroup_sizes[0] * delta_workgroup_sizes[1] * delta_workgroup_sizes[2];
    let output_size: usize = levelset_grid_f64.len();
    let levelset_grid_f64_arr: &[cl_double] = &*convert_slice_to_cl_double(levelset_grid_f64);
    let mut output_buffer = unsafe {Buffer::<cl_double>::create(&context, CL_MEM_WRITE_ONLY, output_size, ptr::null_mut())? };
    let _output_buffer_write_event = unsafe {  queue.enqueue_write_buffer(&mut output_buffer, CL_BLOCKING, 0,  &levelset_grid_f64_arr, &[])? };


    // Create OpenCL device buffers
    let mut x_np = unsafe { Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };
    let mut x_subdomain_ijk = unsafe { Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };
    let mut x_cells_per_subdomain = unsafe { Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };

    let mut x_global_marching_cubes_grid__np = unsafe { Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };
    let mut x_global_marching_cubes_grid__aabb_min = unsafe { Buffer::<cl_double>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };

    // Blocking write
    let _x_write_event = unsafe { queue.enqueue_write_buffer(&mut x_np, CL_BLOCKING, 0, &n_points_per_dim.map(|x| x as cl_int), &[])? };
    let _subdomain_ijk_write_event = unsafe { queue.enqueue_write_buffer(&mut x_subdomain_ijk, CL_BLOCKING, 0, &subdomain_ijk.map(|x| x as cl_int), &[])? };
    let _cells_per_subdomain_write_event = unsafe { queue.enqueue_write_buffer(&mut x_cells_per_subdomain, CL_BLOCKING, 0, &cells_per_subdomain.map(|x| x as cl_int), &[])? };
    let _global_marching_cubes_grid__np_event = unsafe { queue.enqueue_write_buffer(&mut x_global_marching_cubes_grid__np, CL_BLOCKING, 0, &global_marching_cubes_grid__np.map(|x| x as cl_int), &[])? };
    let _global_marching_cubes_grid__aabb_min_event = unsafe { queue.enqueue_write_buffer(&mut x_global_marching_cubes_grid__aabb_min, CL_BLOCKING, 0, &global_marching_cubes_grid__aabb_min.map(|x| x as cl_double), &[])? };


    // // Non-blocking write, wait for y_write_event
    // let y_write_event =
    //     unsafe { queue.enqueue_write_buffer(&mut y, CL_NON_BLOCKING, 0, &sums, &[])? };


    let mut p_i_buffer = unsafe { Buffer::<cl_double>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };
    let _p_i_event = unsafe { queue.enqueue_write_buffer(&mut p_i_buffer, CL_BLOCKING, 0, &p_i.map(|x| x as cl_double), &[])? };

    let mut parameters_buffer = unsafe { Buffer::<cl_double>::create(&context, CL_MEM_READ_ONLY, 5, ptr::null_mut())? };
    let _parameters_event = unsafe { queue.enqueue_write_buffer(&mut parameters_buffer, CL_BLOCKING, 0, &parameters.map(|x| x as cl_double), &[])? };


    let mut delta_workgroup_sizes: [usize; 3] = usize_diff(lower, upper);

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            // Set 3d range
            // ranges is difference between lowe and high.
            // Low needs to be passed so in CL Low.x + x can be done

            // .set_global_work_sizes(&delta_workgroup_sizes)
            .set_global_work_sizes(&[2, 2])


            .set_arg(&output_buffer)
            // .set_arg(&x_np)
            // .set_arg(&(delta_workgroup_sizes[1] as cl_int))
            // .set_arg(&(delta_workgroup_sizes[2] as cl_int))
            // .set_arg(&x_subdomain_ijk)
            // .set_arg(&x_cells_per_subdomain)
            // //Global
            // .set_arg(&x_global_marching_cubes_grid__np)
            // .set_arg(&x_global_marching_cubes_grid__aabb_min)
            // .set_arg(&(global_marching_cubes_grid__cell_size as cl_double))
            //
            // .set_arg(&p_i_buffer)
            // .set_arg(&(rho_i as cl_double))
            //
            // .set_arg(&parameters_buffer)
            // .set_wait_event(&y_write_event)
            .enqueue_nd_range(&queue)?
    };

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Create a results array to hold the results from the OpenCL device
    // and enqueue a read command to read the device buffer into the array
    // after the kernel event completes.

    let mut results = &mut vec![0 as cl_double; output_size];
    // let mut results: [cl_int; OUTPUT_SIZE] = [0; OUTPUT_SIZE];

    let read_event =
        unsafe { queue.enqueue_read_buffer(&output_buffer, CL_NON_BLOCKING, 0, &mut results, &events)? };

    // Wait for the read_event to complete.
    read_event.wait()?;

    // Output the first and last results
    // println!("results front: {:?}", results);

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;
    println!("kernel execution duration (ns): {}. For {} items", duration, output_size);


    let mut lsg_vec = Vec::default();
    for x in results {
        if *x > 0.0 {
            lsg_vec.push(x)
        }
    }
    for (i, x) in lsg_vec.into_iter().enumerate() {
        println!("Val:: {:?}->{:?}", i, x, );
    }
    // println!("results front: {:?} :: {:?} :: {:?}", results[0], results[1], results[2]);
    panic!("ss");


    // Check if to_owned is performant or not
    Ok(results.to_owned())
}

pub(crate) fn new_queue_buffer<D: From<T>, T, const N: usize>(context: &Context, queue: &CommandQueue, values: [T; N], cl_mem_flags: cl_mem_flags) -> Buffer<D> {
    let mut buffer = unsafe { Buffer::<D>::create(&context, cl_mem_flags, N, ptr::null_mut()).unwrap() };
    let _parameters_event = unsafe { (*queue).enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, &values.map(|x| D::from(x)), &[]).unwrap() };
    return buffer;
}

pub(crate) fn convert_slice_to_cl_double(input: &[f64]) -> Box<[cl_double]> {
    let output: Vec<cl_double> = input.iter().map(|&x| x as cl_double).collect();
    output.into_boxed_slice()
}
pub(crate) fn convert_slice_to_cl_long(input: &[u64]) -> Box<[cl_long]> {
    let output: Vec<cl_long> = input.iter().map(|&x| x as cl_long).collect();
    output.into_boxed_slice()
}


pub(crate) fn gpu_img(
    kernel_data: &KernelData,
    levelset_grid_f64: &[f64],
    // Range
    lower: [u64; 3],
    upper: [u64; 3],
    n_points_per_dim: [u64; 3],
    subdomain_ijk: [u64; 3],
    cells_per_subdomain: [u64; 3],
    global_marching_cubes_grid__np: [u64; 3],
    global_marching_cubes_grid__aabb_min: [f64; 3],
    global_marching_cubes_grid__cell_size: f64,
    p_i: [f64; 3],
    rho_i: f64,
    parameters: [f64; 5],
) -> Result<Vec<cl_double>> {
    let KernelData { platform, device, context, program, kernel } = kernel_data;

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    /////////////////////////////////////////////////////////////////////
    // Compute data


    let output_size: usize = levelset_grid_f64.len();


    // Output Buffers
    let levelset_grid_f64_arr: &[cl_double] = &*convert_slice_to_cl_double(levelset_grid_f64);
    let mut output_buffer = unsafe { Buffer::<cl_double>::create(&context, CL_MEM_READ_WRITE, output_size, ptr::null_mut())? };
    let _output_buffer_write_event = unsafe { queue.enqueue_write_buffer(&mut output_buffer, CL_BLOCKING, 0, &levelset_grid_f64_arr, &[])? };

    // Input Argument Buffers
    let lower_buffer: Buffer<cl_ulong> = new_queue_buffer(&context, &queue, lower, CL_MEM_READ_ONLY);
    let x_np: Buffer<cl_ulong> = new_queue_buffer(&context, &queue, n_points_per_dim, CL_MEM_READ_ONLY);
    let sd_ijk: Buffer<cl_ulong> = new_queue_buffer(&context, &queue, subdomain_ijk, CL_MEM_READ_ONLY);
    let sp_sd: Buffer<cl_ulong> = new_queue_buffer(&context, &queue, cells_per_subdomain, CL_MEM_READ_ONLY);
    let gmcg_np: Buffer<cl_ulong> = new_queue_buffer(&context, &queue, global_marching_cubes_grid__np, CL_MEM_READ_ONLY);


    let gmcg_aabb_min: Buffer<cl_double> = new_queue_buffer(&context, &queue, global_marching_cubes_grid__aabb_min, CL_MEM_READ_ONLY);
    let p_i_buffer: Buffer<cl_double> = new_queue_buffer(&context, &queue, p_i, CL_MEM_READ_ONLY);
    let parameters_buffer: Buffer<cl_double> = new_queue_buffer(&context, &queue, parameters, CL_MEM_READ_ONLY);

    let delta_workgroup_sizes: [usize; 3] = usize_diff(lower, upper);

    // Use the ExecuteKernel builder to set the kernel buffer and
    // cl_float value arguments, before setting the one dimensional
    // global_work_size for the call to enqueue_nd_range.
    // Unwraps the Result to get the kernel execution event.
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&output_buffer)
            .set_arg(&lower_buffer)
            .set_arg(&x_np)
            .set_arg(&sd_ijk)
            .set_arg(&sp_sd)
            .set_arg(&gmcg_np)
            .set_arg(&gmcg_aabb_min)
            .set_arg(&(global_marching_cubes_grid__cell_size as cl_double))
            .set_arg(&p_i_buffer)
            .set_arg(&(rho_i as cl_double))
            .set_arg(&parameters_buffer)
            .set_global_work_sizes(&delta_workgroup_sizes)
            .enqueue_nd_range(&queue)?
    };

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Create a results array to hold the results from the OpenCL device
    // and enqueue a read command to read the device buffer into the array
    // after the kernel event completes.
    let mut results = &mut vec![0 as cl_double; output_size];

    let read_event = unsafe { queue.enqueue_read_buffer(&output_buffer, CL_NON_BLOCKING, 0, &mut results, &events)? };

    // Wait for the read_event to complete.
    read_event.wait()?;

    // // Calculate the kernel duration, from the kernel_event
    // let start_time = kernel_event.profiling_command_start()?;
    // let end_time = kernel_event.profiling_command_end()?;
    // let duration = end_time - start_time;
    // println!("kernel execution duration (ns): {}", duration);

    Ok(results.to_vec())
}
