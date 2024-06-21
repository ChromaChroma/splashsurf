// TODO: Reduce code duplication between dense and sparse
pub(crate) fn reconstruction<I: Index, R: Real>(
    parameters: &ParametersSubdomainGrid<I, R>,
    global_particles: &[Vector3<R>],
    global_particle_densities: &[R],
    subdomains: &Subdomains<I>,
) -> Vec<SurfacePatch<I, R>> {
    profile!(parent, "reconstruction");

    let squared_support = parameters.compact_support_radius * parameters.compact_support_radius;
    // Add 1% so that we don't exclude grid points that are just on the kernel boundary
    let squared_support_with_margin = squared_support * to_real!(1.01);
    // Compute radial distance in terms of grid points we have to evaluate for each particle
    let cube_radius = I::from((parameters.compact_support_radius / parameters.cube_size).ceil())
        .expect("kernel radius in cubes has to fit in index type");
    // Kernel
    let kernel = CubicSplineKernel::new(parameters.compact_support_radius);
    //let kernel = DiscreteSquaredDistanceCubicKernel::new::<f64>(1000, parameters.compact_support_radius);

    let mc_total_cells = parameters.subdomain_cubes.cubed();
    let mc_total_points = (parameters.subdomain_cubes + I::one()).cubed();

    assert!(
        mc_total_points.to_usize().is_some(),
        "number of mc cubes per subdomain must be fit into usize"
    );

    let max_particles = subdomains
        .per_subdomain_particles
        .iter()
        .map(|p| p.len())
        .max()
        .unwrap_or(0);
    info!("Largest subdomain has {} particles.", max_particles);

    // Maximum number of particles such that a subdomain will be considered "sparse"
    let sparse_limit = (to_real!(max_particles) * to_real!(0.05))
        .ceil()
        .to_usize()
        .unwrap()
        .max(100);
    info!(
        "Subdomains with {} or less particles will be considered sparse.",
        sparse_limit
    );

    info!("Starting reconstruction (level-set evaluation and local triangulation).");

    // Returns a unique identifier for any edge index of a subdomain that can be later used for stitching
    let globalize_local_edge = |mc_grid: &UniformCartesianCubeGrid3d<I, R>,
                                subdomain_grid: &UniformCartesianCubeGrid3d<I, R>,
                                subdomain_index: I,
                                local_edge: &EdgeIndex<I>|
                                -> (I, EdgeIndex<I>) {
        // We globalize the boundary edge index by translating the local edge index to the subdomain
        // where it lies on the lower boundary of that domain.

        let max_mc_point_index = mc_grid.points_per_dim().map(|i| i - I::one());
        let max_subdomain_index = subdomain_grid
            .cells_per_dim()
            .map(|i| i.saturating_sub(&I::one()).max(I::zero()));

        // Check along which axes this edge is on the max boundary
        let is_max = local_edge.axis().orthogonal_axes().map(|orth_axis| {
            if local_edge.origin().index()[orth_axis.dim()] == max_mc_point_index[orth_axis.dim()] {
                // We are on the max side of this domain along the axis
                true
            } else {
                // We are either
                //  - On the min side of this domain along the axis
                //  - Somewhere in the middle (in this case this axis is irrelevant)
                false
            }
        });

        if !is_max[0] && !is_max[1] {
            // Edge is already in the correct subdomain
            (subdomain_index, local_edge.clone())
        } else {
            // We have to translate to the neighboring subdomain (+1 in all directions where is_max == true)
            let subdomain_cell = subdomain_grid
                .try_unflatten_cell_index(subdomain_index)
                .expect("invalid subdomain index");

            let mut target_subdomain_ijk = subdomain_cell.index().clone();
            let mut target_local_origin_ijk = local_edge.origin().index().clone();

            // Obtain index of new subdomain and new origin point
            for (&orth_axis, &is_max) in local_edge
                .axis()
                .orthogonal_axes()
                .iter()
                .zip(is_max.iter())
            {
                if is_max {
                    // Clamp the step to the subdomain grid because we are not interested in subdomains outside the grid
                    // (globalization is not needed on the outermost boundary of the entire problem domain)
                    target_subdomain_ijk[orth_axis.dim()] = (target_subdomain_ijk[orth_axis.dim()]
                        + I::one())
                        .min(max_subdomain_index[orth_axis.dim()]);
                    // Move origin point from max boundary to min boundary
                    target_local_origin_ijk[orth_axis.dim()] = I::zero();
                }
            }

            let target_subdomain = subdomain_grid
                .get_cell(target_subdomain_ijk)
                .expect("target subdomain has to exist");
            let flat_target_subdomain = subdomain_grid.flatten_cell_index(&target_subdomain);

            // We re-use the same marching cubes domain here because the domain is anyway rectangular,
            // therefore this shift gives the same result
            let new_local_edge = mc_grid
                .get_edge(target_local_origin_ijk, local_edge.axis())
                .expect("failed to translate edge");

            (flat_target_subdomain, new_local_edge)
        }
    };

    #[derive(Default)]
    struct SubdomainWorkspace<I: Index, R: Real> {
        // Particle positions of this subdomain
        subdomain_particles: Vec<Vector3<R>>,
        // Per particle density values of this subdomain
        subdomain_particle_densities: Vec<R>,
        // Cache for the level-set values
        levelset_grid: Vec<R>,
        // Cache for indices
        index_cache: Vec<I>,
    }

    let workspace_tls = ThreadLocal::<RefCell<SubdomainWorkspace<I, R>>>::new();

    let reconstruct_subdomain = |flat_subdomain_idx: I, subdomain_particle_indices: &Vec<usize>, is_sparse: bool| {
        // Obtain thread local workspace and clear it
        let mut workspace = workspace_tls.get_or_default().borrow_mut();

        let SubdomainWorkspace {
            subdomain_particles,
            subdomain_particle_densities,
            levelset_grid,
            index_cache,
        } = &mut *workspace;

        let flat_subdomain_idx: I = flat_subdomain_idx;
        let subdomain_particle_indices: &[usize] = subdomain_particle_indices.as_slice();

        // Collect all particle positions and densities of this subdomain
        {
            //profile!("collect subdomain data");
            gather_subdomain_data(
                global_particles,
                subdomain_particle_indices,
                subdomain_particles,
            );
            gather_subdomain_data(
                global_particle_densities,
                subdomain_particle_indices,
                subdomain_particle_densities,
            );
        }

        // Get the cell index and AABB of the subdomain
        let subdomain_idx = parameters
            .subdomain_grid
            .try_unflatten_cell_index(flat_subdomain_idx)
            .expect("Subdomain cell does not exist");
        let subdomain_aabb = parameters.subdomain_grid.cell_aabb(&subdomain_idx);

        let mc_grid = UniformCartesianCubeGrid3d::new(
            subdomain_aabb.min(),
            &[parameters.subdomain_cubes; 3],
            parameters.cube_size,
        )
            .unwrap();

        levelset_grid.fill(R::zero());
        levelset_grid.resize(mc_total_points.to_usize().unwrap(), R::zero());


        {
            profile!("density grid loop");

            let extents = mc_grid.points_per_dim();

            for (p_i, rho_i) in subdomain_particles
                .iter()
                .copied()
                .zip(subdomain_particle_densities.iter().copied())
            {
                // Get grid cell containing particle
                let particle_cell = mc_grid.enclosing_cell(&p_i);

                // Compute lower and upper bounds of the grid points possibly affected by the particle
                // We want to loop over the vertices of the enclosing cells plus all points in `cube_radius` distance from the cell

                let lower = [
                    (particle_cell[0] - cube_radius).max(I::zero()),
                    (particle_cell[1] - cube_radius).max(I::zero()),
                    (particle_cell[2] - cube_radius).max(I::zero()),
                ];

                let upper = [
                    // We add 2 because
                    //  - we want to loop over all grid points of the cell (+1 for upper points) + the radius
                    //  - the upper range limit is exclusive (+1)
                    (particle_cell[0] + cube_radius + I::two()).min(extents[0]),
                    (particle_cell[1] + cube_radius + I::two()).min(extents[1]),
                    (particle_cell[2] + cube_radius + I::two()).min(extents[2]),
                ];

                // Loop over all grid points around the enclosing cell
                for i in I::range(lower[0], upper[0]).iter() {
                    for j in I::range(lower[1], upper[1]).iter() {
                        for k in I::range(lower[2], upper[2]).iter() {
                            let point_ijk = [i, j, k];
                            let local_point = mc_grid
                                .get_point(point_ijk)
                                .expect("point has to be part of the subdomain grid");
                            //let point_coordinates = mc_grid.point_coordinates(&point);

                            let subdomain_ijk = subdomain_idx.index();
                            let mc_cells_per_subdomain = mc_grid.cells_per_dim();

                            fn local_to_global_point_ijk<I: Index>(
                                local_point_ijk: [I; 3],
                                subdomain_ijk: [I; 3],
                                cells_per_subdomain: [I; 3],
                            ) -> [GlobalIndex; 3] {
                                let local_point_ijk = local_point_ijk
                                    .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
                                let subdomain_ijk = subdomain_ijk
                                    .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
                                let cells_per_subdomain = cells_per_subdomain
                                    .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
                                let [i, j, k] = local_point_ijk;

                                [
                                    subdomain_ijk[0] * cells_per_subdomain[0] + i,
                                    subdomain_ijk[1] * cells_per_subdomain[1] + j,
                                    subdomain_ijk[2] * cells_per_subdomain[2] + k,
                                ]
                            }

                            // Use global coordinate calculation for consistency with neighboring domains
                            let global_point_ijk = local_to_global_point_ijk(
                                point_ijk,
                                subdomain_ijk.clone(),
                                mc_cells_per_subdomain.clone(),
                            );
                            let global_point = parameters
                                .global_marching_cubes_grid
                                .get_point(global_point_ijk)
                                .expect("point has to be part of the global mc grid");
                            let point_coordinates = parameters
                                .global_marching_cubes_grid
                                .point_coordinates(&global_point);

                            let dx = p_i - point_coordinates;
                            let dx_norm_sq = dx.norm_squared();

                            if dx_norm_sq < squared_support_with_margin {
                                let v_i = parameters.particle_rest_mass / rho_i;
                                let r = dx_norm_sq.sqrt();
                                let w_ij = kernel.evaluate(r);
                                //let w_ij = kernel.evaluate(dx_norm_sq);

                                let interpolated_value = v_i * w_ij;

                                let flat_point_idx = mc_grid.flatten_point_index(&local_point);
                                let flat_point_idx = flat_point_idx.to_usize().unwrap();
                                levelset_grid[flat_point_idx] += interpolated_value;

                                if is_sparse && levelset_grid[flat_point_idx] > parameters.surface_threshold {
                                    for c in mc_grid
                                        .cells_adjacent_to_point(
                                            &mc_grid.get_point_neighborhood(&local_point),
                                        )
                                        .iter()
                                        .flatten()
                                    {
                                        let flat_cell_index = mc_grid.flatten_cell_index(c);
                                        index_cache.push(flat_cell_index);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


        let mut vertices = Vec::new();
        let mut triangles = Vec::new();

        let mut vertex_inside_count = 0;
        let mut triangle_inside_count = 0;

        let mut vertex_inside_flags = Vec::new();
        let mut triangle_inside_flags = Vec::new();

        let mut exterior_vertex_edge_indices = Vec::new();

        let mut edge_to_vertex = new_map();


        {
            profile!("mc triangulation loop");

            let indexes = if is_sparse {
                index_cache.sort_unstable();
                index_cache.iter().copied().dedup()
            } else {
                I::range(I::zero(), mc_total_cells).iter()
            };


            for flat_cell_idx in indexes {
                let cell = mc_grid.try_unflatten_cell_index(flat_cell_idx).unwrap();


                let mut vertices_inside = [true; 8];
                for local_point_index in 0..8 {
                    let point = cell.global_point_index_of(local_point_index).unwrap();
                    let flat_point_idx = mc_grid.flatten_point_index(&point);
                    let flat_point_idx = flat_point_idx.to_usize().unwrap();
                    // Get value of density map
                    let density_value = levelset_grid[flat_point_idx];
                    // Update inside/outside surface flag
                    vertices_inside[local_point_index] =
                        density_value > parameters.surface_threshold;
                }

                for triangle in marching_cubes_triangulation_iter(&vertices_inside) {
                    let mut global_triangle = [0; 3];
                    for (v_idx, local_edge_index) in triangle.iter().copied().enumerate() {
                        let edge = cell
                            .global_edge_index_of(local_edge_index as usize)
                            .unwrap();
                        let vertex_index = *edge_to_vertex.entry(edge).or_insert_with(|| {
                            // TODO: Nonlinear interpolation

                            let origin_coords = mc_grid.point_coordinates(&edge.origin());
                            let target_coords = mc_grid.point_coordinates(&edge.target());

                            let flat_origin_idx = mc_grid
                                .flatten_point_index(&edge.origin())
                                .to_usize()
                                .unwrap();
                            let flat_target_idx = mc_grid
                                .flatten_point_index(&edge.target())
                                .to_usize()
                                .unwrap();

                            let origin_value = levelset_grid[flat_origin_idx];
                            let target_value = levelset_grid[flat_target_idx];

                            let alpha = (parameters.surface_threshold - origin_value)
                                / (target_value - origin_value);
                            let interpolated_coords =
                                origin_coords * (R::one() - alpha) + target_coords * alpha;
                            let vertex_coords = interpolated_coords;

                            vertices.push(vertex_coords);
                            let vertex_index = vertices.len() - 1;

                            let is_interior_vertex = !mc_grid.is_boundary_edge(&edge);
                            vertex_inside_count += is_interior_vertex as usize;
                            vertex_inside_flags.push(is_interior_vertex);

                            if !is_interior_vertex {
                                exterior_vertex_edge_indices.push(globalize_local_edge(
                                    &mc_grid,
                                    &parameters.subdomain_grid,
                                    flat_subdomain_idx,
                                    &edge,
                                ));
                            }

                            vertex_index
                        });

                        global_triangle[v_idx] = vertex_index;
                    }

                    let all_tri_vertices_inside = global_triangle
                        .iter()
                        .copied()
                        .all(|v_idx| vertex_inside_flags[v_idx]);

                    triangles.push(global_triangle);
                    triangle_inside_count += all_tri_vertices_inside as usize;
                    triangle_inside_flags.push(all_tri_vertices_inside);
                }
            }
        }

        SurfacePatch {
            vertices,
            triangles,
            vertex_inside_count,
            triangle_inside_count,
            vertex_inside_flags,
            triangle_inside_flags,
            exterior_vertex_edge_indices,
        }
    };

    let mut surface_patches = Vec::with_capacity(subdomains.flat_subdomain_indices.len());
    subdomains.flat_subdomain_indices
        .par_iter()
        .copied()
        .zip(subdomains.per_subdomain_particles.par_iter())
        .map(|(flat_subdomain_idx, subdomain_particle_indices)| {
            if subdomain_particle_indices.len() <= sparse_limit {
                profile!("subdomain reconstruction (sparse)", parent = parent);
                reconstruct_subdomain(flat_subdomain_idx, subdomain_particle_indices, true)
            } else {
                profile!("subdomain reconstruction (dense)", parent = parent);
                reconstruct_subdomain(flat_subdomain_idx, subdomain_particle_indices, false)
            }
        })
        .collect_into_vec(&mut surface_patches);

    surface_patches
}


use std::cell::RefCell;
use std::ptr;
use std::sync::{Arc, Mutex};
use itertools::Itertools;
use log::info;
use nalgebra::Vector3;
use num_traits::NumCast;
use opencl3::command_queue::{cl_event, CL_NON_BLOCKING, CL_QUEUE_ON_DEVICE, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, cl_double, Device};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::{CL_STD_2_0, Program};
use opencl3::types::cl_int;
use rayon::iter::IntoParallelRefIterator;
use thread_local::ThreadLocal;
use crate::{Index, new_map, OpenCLData, profile, Real};
use crate::dense_subdomains::{gather_subdomain_data, GlobalIndex, ParametersSubdomainGrid, subdomain_classification, Subdomains, SurfacePatch, to_real};
use crate::gpu::kernel::init_kernels;
use crate::kernel::{CubicSplineKernel, SymmetricKernel3d};
use crate::marching_cubes::marching_cubes_lut::marching_cubes_triangulation_iter;
use crate::uniform_grid::{EdgeIndex, UniformCartesianCubeGrid3d};

// /// Performs classification and decomposition of particles into a regular grid of subdomains
// pub(crate) fn decomposition<
//     I: Index,
//     R: Real,
//     C: subdomain_classification::ParticleToSubdomainClassifier<I, R>,
// >(parameters: &ParametersSubdomainGrid<I, R>,
//   particles: &[Vector3<R>],
//   ocl_data: Arc<Mutex<OpenCLData>>,
// ) -> Result<Subdomains<I>, anyhow::Error> {
//     profile!(p, "GPU::decomposition");
//     info!("GPU::Starting classification of particles into subdomains.");
//
//     const WORKGROUP_SIZE: usize = 262_144;
//
//     // Create a command_queue on the Context's device
//     let queue = {
//         profile!("create command queue", parent = p);
//         CommandQueue::create_default(
//             &ocl_data.lock().unwrap().context, CL_QUEUE_PROFILING_ENABLE,
//         ).expect("CommandQueue::create_default failed")
//     };
//
//
//     // Output Buffers
//     // let levelset_grid_f64_arr: &[cl_double] = &*convert_slice_to_cl_double(&levelset_grid_f64);
//
//     let mut output_buffer = unsafe {
//         profile!("create output buffer", parent = p);
//         Buffer::<cl_int>::create(&ocl_data.lock().unwrap().context, CL_MEM_READ_WRITE, WORKGROUP_SIZE, ptr::null_mut()).expect("Could not create output_buffer")
//     };
//     let _output_buffer_write_event = unsafe {
//         profile!("enqueue write output buffer", parent = p);
//         queue.enqueue_write_buffer(&mut output_buffer, CL_NON_BLOCKING, 0, &[0; WORKGROUP_SIZE], &[]).expect("Could not enqueue output_buffer")
//     };
//
//     let kernel_event = unsafe {
//         profile!("create and enqueue kernel", parent = p);
//         ExecuteKernel::new(&ocl_data.lock().unwrap().kernel)
//             .set_arg(&output_buffer)
//             .set_global_work_size(WORKGROUP_SIZE)
//             .set_wait_event(&_output_buffer_write_event)
//             .enqueue_nd_range(&queue)
//             .expect("Could not run Kernel")
//     };
//
//     let mut events: Vec<cl_event> = Vec::default();
//     events.push(kernel_event.get());
//
//     let mut results = &mut vec![0 as cl_int; WORKGROUP_SIZE];
//     {
//         profile!("enqueue read buffer", parent = p);
//         let read_event = unsafe {
//             queue.enqueue_read_buffer(&output_buffer, CL_NON_BLOCKING, 0, &mut results, &events).expect("Could not enqueue result read buffer")
//         };
//         // Wait for the read_event to complete.
//         read_event.wait().expect("Could not read event for retrieveing data from gpu bufer");
//     }
//
//     // println!("{:?}", results);
//
//     // Calculate the kernel duration, from the kernel_event
//     let start_time = kernel_event.profiling_command_start().unwrap();
//     let end_time = kernel_event.profiling_command_end().unwrap();
//     let duration = end_time - start_time;
//     println!("kernel execution duration (ns): {}", duration);
//
//
//     // // Create a command_queue on the Context's device
//     // let queue = CommandQueue::create_default(
//     //     &ocl_data.lock().unwrap().context, CL_QUEUE_PROFILING_ENABLE,
//     // ).expect("CommandQueue::create_default failed");
//     //
//     // const WORKGROUP_SIZE: usize = 2048;
//     //
//     //
//     // // Output Buffers
//     // // let levelset_grid_f64_arr: &[cl_double] = &*convert_slice_to_cl_double(&levelset_grid_f64);
//     //
//     // let mut output_buffer = unsafe {
//     //     Buffer::<cl_int>::create(&ocl_data.lock().unwrap().context, CL_MEM_READ_WRITE, WORKGROUP_SIZE, ptr::null_mut()).expect("Could not create output_buffer")
//     // };
//     // let _output_buffer_write_event = unsafe {
//     //     queue.enqueue_write_buffer(&mut output_buffer, CL_NON_BLOCKING, 0, &[0; WORKGROUP_SIZE], &[]).expect("Could not enqueue output_buffer")
//     // };
//     //
//     // let kernel_event = unsafe {
//     //     ExecuteKernel::new(&ocl_data.lock().unwrap().kernel)
//     //         .set_arg(&output_buffer)
//     //         .set_global_work_size(WORKGROUP_SIZE)
//     //         .set_wait_event(&_output_buffer_write_event)
//     //         .enqueue_nd_range(&queue)
//     //         .expect("Could not run Kernel")
//     // };
//     //
//     // let mut events: Vec<cl_event> = Vec::default();
//     // events.push(kernel_event.get());
//     //
//     // let mut results = &mut vec![0 as cl_int; WORKGROUP_SIZE];
//     // let read_event = unsafe { queue.enqueue_read_buffer(&output_buffer, CL_NON_BLOCKING, 0, &mut results, &events).expect("Could not enqueue result read buffer") };
//     // // Wait for the read_event to complete.
//     // read_event.wait().expect("Could not read event for retrieveing data from gpu bufer");
//     //
//     // println!("{:?}", results);
//     //
//     // // Calculate the kernel duration, from the kernel_event
//     // let start_time = kernel_event.profiling_command_start().unwrap();
//     // let end_time = kernel_event.profiling_command_end().unwrap();
//     // let duration = end_time - start_time;
//     // println!("kernel execution duration (ns): {}", duration);
//
//
//     // init_kernels().expect(" Gpu failed initializing");
//     // panic!(" Well Done!")
//
//     Ok(Subdomains {
//         flat_subdomain_indices: Vec::new(),
//         per_subdomain_particles: Vec::new(),
//     })
// }
