// #![warn(
//     clippy::all,
//     clippy::await_holding_lock,
//     clippy::char_lit_as_u8,
//     clippy::checked_conversions,
//     clippy::dbg_macro,
//     clippy::debug_assert_with_mut_call,
//     clippy::doc_markdown,
//     clippy::empty_enum,
//     clippy::enum_glob_use,
//     clippy::exit,
//     clippy::expl_impl_clone_on_copy,
//     clippy::explicit_deref_methods,
//     clippy::explicit_into_iter_loop,
//     clippy::fallible_impl_from,
//     clippy::filter_map_next,
//     clippy::float_cmp_const,
//     clippy::fn_params_excessive_bools,
//     clippy::if_let_mutex,
//     clippy::implicit_clone,
//     clippy::imprecise_flops,
//     clippy::inefficient_to_string,
//     clippy::invalid_upcast_comparisons,
//     clippy::large_types_passed_by_value,
//     clippy::let_unit_value,
//     clippy::linkedlist,
//     clippy::lossy_float_literal,
//     clippy::macro_use_imports,
//     clippy::manual_ok_or,
//     clippy::map_err_ignore,
//     clippy::map_flatten,
//     clippy::map_unwrap_or,
//     clippy::match_on_vec_items,
//     clippy::match_same_arms,
//     clippy::match_wildcard_for_single_variants,
//     clippy::mem_forget,
//     clippy::mismatched_target_os,
//     clippy::mut_mut,
//     clippy::mutex_integer,
//     clippy::needless_borrow,
//     clippy::needless_continue,
//     clippy::option_option,
//     clippy::path_buf_push_overwrite,
//     clippy::ptr_as_ptr,
//     clippy::ref_option_ref,
//     clippy::rest_pat_in_fully_bound_structs,
//     clippy::same_functions_in_if_condition,
//     clippy::semicolon_if_nothing_returned,
//     clippy::string_add_assign,
//     clippy::string_add,
//     clippy::string_lit_as_bytes,
//     clippy::string_to_string,
//     clippy::todo,
//     clippy::trait_duplication_in_bounds,
//     clippy::unimplemented,
//     clippy::unnested_or_patterns,
//     clippy::unused_self,
//     clippy::useless_transmute,
//     clippy::verbose_file_reads,
//     clippy::zero_sized_map_values,
//     future_incompatible,
//     nonstandard_style,
//     rust_2018_idioms
// )]
//
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::types::{cl_float, CL_NON_BLOCKING};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_BLOCKING};
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::event::cl_int;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::Image;
use opencl3::program::{Program, CL_STD_2_0};
use opencl3::types::cl_event;
use opencl3::Result;
use std::ptr;
use opencl3::platform::Platform;

pub struct KernelData {
    pub platform: Platform,
    pub device: Device,
    pub context: Context,
    pub program: Program,
    pub kernel: Kernel,
}


pub(crate) fn init_kernel() -> Result<KernelData> {
    let shaders_code = include_str!("shaders/reconstruct.cl");
    const KERNEL_NAME2: &str = "reconstruct";

    let platforms = opencl3::platform::get_platforms()?;
    let platform = platforms.first()
        .expect("No OpenCL platforms found.");
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
    let kernel = Kernel::create(&program, KERNEL_NAME2).expect("Kernel::create failed");

    Ok(KernelData {
        platform: *platform,
        device: device,
        context: context,
        program: program,
        kernel: kernel,
    })
}


pub(crate) fn gpu_small_reconstruct(
    kernel_data: &KernelData,
    // levelset_grid_f64:  &mut Vec<f64>,
    levelset_grid_f64: &[f64],
    lower: [u64; 3],
    upper: [u64; 3],
    subdomain_ijk: [u64; 3],
    cells_per_subdomain: [u64; 3],
    // mc_grid: UniformCartesianCubeGrid3d<I, R>
    n_points_per_dim: [u64; 3],

    // Global
    global_marching_cubes_grid__np: [u64; 3],
    global_marching_cubes_grid__aabb_min: [f64; 3],
    global_marching_cubes_grid__cell_size: f64,
    p_i: [f64; 3],
    rho_i: f64,
    parameters: [f64; 5],
) -> Result<Vec<cl_float>> {
    let KernelData {
        platform,
        device,
        context,
        program,
        kernel
    } = kernel_data;


    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");


    ////////////////
    // Compute data

    // Lower
    let [lx, ly, lz] = lower;
    // Higher
    let [hx, hy, hz] = upper;

    //[delta_x, delta_y, delta_z]
    let mut delta_workgroup_sizes = [
        (hx - lx) as usize,
        (hy - ly) as usize,
        (hz - lz) as usize,
    ];

    // let output_size: usize = delta_workgroup_sizes[0] * delta_workgroup_sizes[1] * delta_workgroup_sizes[2];
    let output_size: usize = levelset_grid_f64.len();

    let mut output_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, output_size, ptr::null_mut())?
    };

    fn convert_slice_to_cl_float(input: &[f64]) -> Box<[cl_float]> {
        let output: Vec<cl_float> = input.iter().map(|&x| x as cl_float).collect();
        output.into_boxed_slice()
    }

    // let levelset_grid_f64_arr1: [f64] = *levelset_grid_f64;
    let levelset_grid_f64_arr: &[cl_float] = &*convert_slice_to_cl_float(levelset_grid_f64);

    // levelset_grid_f64.into_iter()
    // .map(|x| *x as cl_float)
    // // .collect();
    // .collect::<Vec<cl_float>>()
    // .try_into()
    // .unwrap();

    let _output_buffer_write_event = unsafe {
        queue.enqueue_write_buffer(&mut output_buffer, CL_BLOCKING, 0,
                                   &levelset_grid_f64_arr,
                                   &[])?
    };


    // Create OpenCL device buffers
    let mut x = unsafe { Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };
    let mut x_subdomain_ijk = unsafe { Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };
    let mut x_cells_per_subdomain = unsafe { Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };

    let mut x_global_marching_cubes_grid__np = unsafe { Buffer::<cl_int>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };
    let mut x_global_marching_cubes_grid__aabb_min = unsafe { Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };

    // Blocking write
    let _x_write_event = unsafe { queue.enqueue_write_buffer(&mut x, CL_BLOCKING, 0, &n_points_per_dim.map(|x| x as cl_int), &[])? };
    let _subdomain_ijk_write_event = unsafe { queue.enqueue_write_buffer(&mut x_subdomain_ijk, CL_BLOCKING, 0, &subdomain_ijk.map(|x| x as cl_int), &[])? };
    let _cells_per_subdomain_write_event = unsafe { queue.enqueue_write_buffer(&mut x_cells_per_subdomain, CL_BLOCKING, 0, &cells_per_subdomain.map(|x| x as cl_int), &[])? };
    let _global_marching_cubes_grid__np_event = unsafe { queue.enqueue_write_buffer(&mut x_global_marching_cubes_grid__np, CL_BLOCKING, 0, &global_marching_cubes_grid__np.map(|x| x as cl_int), &[])? };
    let _global_marching_cubes_grid__aabb_min_event = unsafe { queue.enqueue_write_buffer(&mut x_global_marching_cubes_grid__aabb_min, CL_BLOCKING, 0, &global_marching_cubes_grid__aabb_min.map(|x| x as cl_float), &[])? };


    // // Non-blocking write, wait for y_write_event
    // let y_write_event =
    //     unsafe { queue.enqueue_write_buffer(&mut y, CL_NON_BLOCKING, 0, &sums, &[])? };


    let mut p_i_buffer = unsafe { Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, 3, ptr::null_mut())? };
    let _p_i_event = unsafe { queue.enqueue_write_buffer(&mut p_i_buffer, CL_BLOCKING, 0, &p_i.map(|x| x as cl_float), &[])? };

    let mut parameters_buffer = unsafe { Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, 5, ptr::null_mut())? };
    let _parameters_event = unsafe { queue.enqueue_write_buffer(&mut parameters_buffer, CL_BLOCKING, 0, &parameters.map(|x| x as cl_float), &[])? };


    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            // Set 3d range
            // ranges is difference between lowe and high.
            // Low needs to be passed so in CL Low.x + x can be done
            .set_global_work_sizes(&delta_workgroup_sizes)
            .set_arg(&output_buffer)
            .set_arg(&x)
            .set_arg(&(delta_workgroup_sizes[1] as cl_int))
            .set_arg(&(delta_workgroup_sizes[2] as cl_int))
            .set_arg(&x_subdomain_ijk)
            .set_arg(&x_cells_per_subdomain)
            //Global
            .set_arg(&x_global_marching_cubes_grid__np)
            .set_arg(&x_global_marching_cubes_grid__aabb_min)
            .set_arg(&(global_marching_cubes_grid__cell_size as cl_float))

            .set_arg(&p_i_buffer)
            .set_arg(&(rho_i as cl_float))

            .set_arg(&parameters_buffer)
            // .set_wait_event(&y_write_event)
            .enqueue_nd_range(&queue)?
    };

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Create a results array to hold the results from the OpenCL device
    // and enqueue a read command to read the device buffer into the array
    // after the kernel event completes.

    let mut results = &mut vec![0 as cl_float; output_size];
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

    // Check if to_owned is performant or not
    Ok(results.to_owned())
}


//
// const PROGRAM_SOURCE: &str = r#"
// kernel void colorize(write_only image2d_t image)
// {
//     const size_t x = get_global_id(0);
//     const size_t y = get_global_id(1);
//     write_imageui(image, (int2)(x, y), (uint4)(x, y, 0, 255));
// }"#;
//
// const KERNEL_NAME: &str = "colorize";

// pub(crate) fn gpu_img() -> Result<()> {
//     let platforms = opencl3::platform::get_platforms()?;
//     let platform = platforms.first()
//         .expect("No OpenCL platforms found.");
//     let device = *platform
//         .get_devices(CL_DEVICE_TYPE_GPU)?
//         .first()
//         .expect("no device found in platform");
//     let device = Device::new(device);
//
//     // Create a Context on an OpenCL device
//     let context = Context::from_device(&device).expect("Context::from_device failed");
//
//     // Print some information about the device
//     println!(
//         "CL_DEVICE_IMAGE_SUPPORT: {:?}",
//         device.image_support().unwrap()
//     );
//     println!(
//         "CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: {:?}",
//         device.max_read_write_image_args().unwrap()
//     );
//     println!(
//         "CL_DEVICE_MAX_READ_IMAGE_ARGS: {:?}",
//         device.max_read_image_args().unwrap()
//     );
//     println!(
//         "CL_DEVICE_MAX_WRITE_IMAGE_ARGS: {:?}",
//         device.max_write_image_args().unwrap()
//     );
//     println!(
//         "CL_DEVICE_MAX_SAMPLERS: {:?}",
//         device.max_device_samples().unwrap()
//     );
//
//     let supported_formats =
//         context.get_supported_image_formats(CL_MEM_WRITE_ONLY, CL_MEM_OBJECT_IMAGE2D)?;
//     if supported_formats
//         .iter()
//         .filter(|f| {
//             f.image_channel_order == CL_RGBA && f.image_channel_data_type == CL_UNSIGNED_INT8
//         })
//         .count()
//         <= 0
//     {
//         println!("Device does not support CL_RGBA with CL_UNSIGNED_INT8 for CL_MEM_WRITE_ONLY!");
//         return Err(CL_IMAGE_FORMAT_NOT_SUPPORTED.into());
//     }
//
//     // Build the OpenCL program source and create the kernel.
//     let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, CL_STD_2_0)
//         .expect("Program::create_and_build_from_source failed");
//     let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");
//
//
//     // Create a command_queue on the Context's device
//     let queue = CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
//         .expect("CommandQueue::create_default_with_properties failed");
//
//     // Create an image
//     let mut image = unsafe {
//         Image::create(
//             &context,
//             CL_MEM_WRITE_ONLY,
//             &cl_image_format {
//                 image_channel_order: CL_RGBA,
//                 image_channel_data_type: CL_UNSIGNED_INT8,
//             },
//             &cl_image_desc {
//                 image_type: CL_MEM_OBJECT_IMAGE2D,
//                 image_width: 10 as usize,
//                 image_height: 10 as usize,
//                 image_depth: 1,
//                 image_array_size: 1,
//                 image_row_pitch: 0,
//                 image_slice_pitch: 0,
//                 num_mip_levels: 0,
//                 num_samples: 0,
//                 buffer: std::ptr::null_mut(),
//             },
//             std::ptr::null_mut(),
//         ).expect("Image::create failed")
//     };
//
//
//     // Run the kernel on the input data
//     let kernel_event = unsafe {
//         ExecuteKernel::new(&kernel)
//             .set_arg(&image)
//             .set_global_work_sizes(&[10usize, 10usize])
//             .enqueue_nd_range(&queue)?
//     };
//
//     let mut events: Vec<cl_event> = Vec::default();
//     events.push(kernel_event.get());
//
//     // Fill the middle of the image with a solid color
//     let fill_color = [11u32, 22u32, 33u32, 44u32];
//     let fill_event = unsafe {
//         queue.enqueue_fill_image(
//             &mut image,
//             fill_color.as_ptr() as *const c_void,
//             &[3usize, 3usize, 0usize] as *const usize,
//             &[4usize, 4usize, 1usize] as *const usize,
//             &events,
//         )?
//     };
//
//     let mut events: Vec<cl_event> = Vec::default();
//     events.push(fill_event.get());
//
//     // Read the image data from the device
//     let mut image_data = [0u8; 10 * 10 * 4];
//     let read_event = unsafe {
//         queue.enqueue_read_image(
//             &image,
//             CL_NON_BLOCKING,
//             &[0usize, 0usize, 0usize] as *const usize,
//             &[10usize, 10usize, 1usize] as *const usize,
//             0,
//             0,
//             image_data.as_mut_ptr() as *mut c_void,
//             &events,
//         )?
//     };
//
//     // Wait for the read_event to complete.
//     read_event.wait()?;
//
//     // Print the image data
//     println!("image_data: ");
//     for y in 0..10 {
//         for x in 0..10 {
//             let offset = (y * 10 + x) * 4;
//             print!(
//                 "({:>3}, {:>3}, {:>3}, {:>3}) ",
//                 image_data[offset],
//                 image_data[offset + 1],
//                 image_data[offset + 2],
//                 image_data[offset + 3]
//             );
//         }
//         println!();
//     }
//
//     Ok(())
// }
