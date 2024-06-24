use std::cell::RefCell;
use std::sync::Arc;

use itertools::Itertools;
use log::info;
use nalgebra::{SVector, Vector3};
use num_traits::{NumCast, ToPrimitive, Zero};
use opencl3::event::Event;
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::Buffer;
use opencl3::program::CL_BUILD_ERROR;
use opencl3::types::{cl_double, cl_ulong};
use parking_lot::Mutex;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use thread_local::ThreadLocal;

use crate::{exec_time, Index, new_map, OpenCLData, profile, Real};
use crate::dense_subdomains::{gather_subdomain_data, GlobalIndex, ParametersSubdomainGrid, Subdomains, SurfacePatch, to_real};
use crate::gpu::debug::{log_kernel_exec_time, print_first_last_non_zero_index, print_non_zero_values, print_nr_of_zero};
use crate::gpu::kernel::RECONSTRUCT_FUNCTION;
use crate::gpu::utils::{as_read_buffer_non_blocking, new_queue, new_write_buffer, read_buffer_into, to_u64_svec};
use crate::kernel::{CubicSplineKernel, SymmetricKernel3d};
use crate::marching_cubes::marching_cubes_lut::marching_cubes_triangulation_iter;
use crate::uniform_grid::{EdgeIndex, UniformCartesianCubeGrid3d};

// // TODO: Reduce code duplication between dense and sparse
pub(crate) fn reconstruction<I: Index, R: Real>(
    parameters: &ParametersSubdomainGrid<I, R>,
    global_particles: &[Vector3<R>],
    global_particle_densities: &[R],
    subdomains: &Subdomains<I>,
    ocl_data: Arc<Mutex<OpenCLData>>,
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

    let reconstruct_dense = |flat_subdomain_idx: I, subdomain_particle_indices: &Vec<usize>| {
        // Obtain thread local workspace and clear it
        let mut workspace = workspace_tls.get_or_default().borrow_mut();

        let SubdomainWorkspace {
            subdomain_particles,
            subdomain_particle_densities,
            levelset_grid,
            index_cache: _index_cache,
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
        ).unwrap();

        let subdomain_ijk = subdomain_idx.index();
        let subdomain_ijk = subdomain_ijk.clone()
            .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());

        let mc_cells_per_subdomain = mc_grid.cells_per_dim();
        let cells_per_subdomain = mc_cells_per_subdomain.clone()
            .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());


        levelset_grid.fill(R::zero());
        levelset_grid.resize(mc_total_points.to_usize().unwrap(), R::zero());

        let extents = mc_grid.points_per_dim();

        {
            profile!("density grid loop");
            let nr_of_particles: usize = subdomain_particles.len();

            // Create a command_queue on the Context's device
            let queue = {
                profile!("queue                   ");
                new_queue(&ocl_data.lock().context)
            };

            // let (upper_lower_buffer, write_event) =
            //     new_write_buffer(&ocl_data.lock().context, &queue, 2 * nr_of_particles);
            let (levelset_grid_buffer, write_event): (Buffer<cl_double>, Event) = {
                profile!("write_buffer            ");
                new_write_buffer(&ocl_data.lock().context, &queue, levelset_grid.len())
            };


            let (sd_particles_buffer, write_event2): (Buffer<cl_double>, Event) = {
                profile!("read_buffer             ");
                // Buffer of subdomain particles - Flattened Point Coordinates [x0, y0, z0, x1, y1, z0, ....]
                let flattened_ps = subdomain_particles
                    .iter()
                    .copied()
                    .zip(subdomain_particle_densities.iter().copied())
                    .flat_map(|(vec, rho)| [vec.x.to_f64().unwrap(), vec.y.to_f64().unwrap(), vec.z.to_f64().unwrap(), rho.to_f64().unwrap()])
                    .collect::<Vec<f64>>();
                as_read_buffer_non_blocking(&ocl_data.lock().context, &queue, &flattened_ps)
                    .expect("Failed to create particle positions buffer")
            };


            // Create a command_queue on the Context's device
            let queue = {
                profile!("queue                   ");
                new_queue(&ocl_data.lock().context)
            };

            let (long_args_buffer, write_event5): (Buffer<cl_ulong>, Event) ={
                profile!("read_buffer             ");
                let np = mc_grid.points_per_dim().into_iter()
                    .map(|x| x.to_u64().unwrap())
                    .collect::<Vec<u64>>();
                let gmcg_n = parameters.global_marching_cubes_grid.points_per_dim()
                    .into_iter().map(|x| x.to_u64().unwrap())
                    .collect::<Vec<u64>>();
                as_read_buffer_non_blocking(&ocl_data.lock().context, &queue, &[
                    np[0], np[1], np[2],
                    subdomain_ijk[0], subdomain_ijk[1], subdomain_ijk[2],
                    cells_per_subdomain[0], cells_per_subdomain[1], cells_per_subdomain[2],
                    gmcg_n[0], gmcg_n[1], gmcg_n[2],
                ]).expect("Failed to create particle density buffer")
            };



            let (gmcg_aabb_min, buffer_w_e8): (Buffer<cl_double>, Event) = {
                profile!("read_buffer             ");
                as_read_buffer_non_blocking(&ocl_data.lock().context, &queue, &to_u64_svec(*parameters.global_marching_cubes_grid.aabb().min()))
                    .expect("Failed to create particle positions buffer")
            };

            let kernel_event = unsafe {
                let min = mc_grid.aabb().min();
                profile!("enqueue range           ");
                ExecuteKernel::new(ocl_data.lock().kernels.get(RECONSTRUCT_FUNCTION).unwrap())
                    .set_arg(&squared_support_with_margin.to_f64().unwrap())
                    .set_arg(&parameters.particle_rest_mass.to_f64().unwrap())
                    .set_arg(&parameters.global_marching_cubes_grid.cell_size().to_f64().unwrap())
                    .set_arg(&cube_radius)
                    .set_arg(&extents[0]).set_arg(&extents[1]).set_arg(&extents[2])
                    .set_arg(&mc_grid.cell_size().to_f64().unwrap())
                    .set_arg(&min.x.to_f64().unwrap()).set_arg(&min.y.to_f64().unwrap()).set_arg(&min.z.to_f64().unwrap())
                    .set_arg(&parameters.compact_support_radius.to_f64().unwrap())
                    .set_arg(&long_args_buffer)
                    .set_wait_event(&write_event5)
                    .set_arg(&gmcg_aabb_min)
                    .set_wait_event(&buffer_w_e8)
                    .set_arg(&sd_particles_buffer)
                    .set_wait_event(&write_event2)
                    // Output buffer
                    .set_arg(&levelset_grid_buffer)
                    .set_wait_event(&write_event)
                    .set_global_work_size(nr_of_particles)
                    .enqueue_nd_range(&queue)
                    .expect("Could not build, or execute kernel")
            };

            // // TODO: No need to read if next kernel does not depend on it.
            let mut results = vec![cl_double::zero(); levelset_grid.len()];
            {
                profile!("read results            ");
                read_buffer_into(&queue, &kernel_event, &levelset_grid_buffer, &mut results);
            }

            // exec_time!(&kernel_event);

            // exec_time!(&bounds_kernel_event, COMPUTE_BOUNDS_FUNCTION);
            // print_non_zero_values(results.clone());
            print_nr_of_zero(results.clone());
            print_first_last_non_zero_index(results.clone());
            // println!("{}, {}", nr_of_particles, results.len());
            // write_non_zero_values(
            //     format!("z-{:?}-gpu.txt", thread::current().id()).to_string(),
            //     results.clone().to_vec(),
            // );

            {
                profile!("results -> levelset_grid");
                *levelset_grid = results
                    .into_iter()
                    .map(|x| R::from(x).unwrap())
                    .collect();
            }


            // results
            //     .into_iter()
            //     .enumerate()
            //     .for_each(|(i, x)|levelset_grid[i] = R::from(x).unwrap());


            // for (inxdx, (p_i, rho_i)) in subdomain_particles
            //     .iter()
            //     .copied()
            //     .zip(subdomain_particle_densities.iter().copied())
            //     .enumerate()
            // {
            //
            //
            //     // Get grid cell containing particle
            //     let particle_cell = mc_grid.enclosing_cell(&p_i);
            //
            //     // Compute lower and upper bounds of the grid points possibly affected by the particle
            //     // We want to loop over the vertices of the enclosing cells plus all points in `cube_radius` distance from the cell
            //
            //     let lower = [
            //         (particle_cell[0] - cube_radius).max(I::zero()),
            //         (particle_cell[1] - cube_radius).max(I::zero()),
            //         (particle_cell[2] - cube_radius).max(I::zero()),
            //     ];
            //
            //
            //     // We add 2 because
            //     //  - we want to loop over all grid points of the cell (+1 for upper points) + the radius
            //     //  - the upper range limit is exclusive (+1)
            //     let upper = [
            //         (particle_cell[0] + cube_radius + I::two()).min(extents[0]),
            //         (particle_cell[1] + cube_radius + I::two()).min(extents[1]),
            //         (particle_cell[2] + cube_radius + I::two()).min(extents[2]),
            //     ];
            //
            //
            //     // // Loop over all grid points around the enclosing cell
            //     for i in I::range(lower[0], upper[0]).iter() {
            //         for j in I::range(lower[1], upper[1]).iter() {
            //             for k in I::range(lower[2], upper[2]).iter() {
            //                 let point_ijk = [i, j, k];
            //                 let local_point = mc_grid
            //                     .get_point(point_ijk)
            //                     .expect("point has to be part of the subdomain grid");
            //                 //let point_coordinates = mc_grid.point_coordinates(&point);
            //
            //                 let [i, j, k] = point_ijk
            //                     .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
            //                 // Use global coordinate calculation for consistency with neighboring domains
            //                 let global_point_ijk = [
            //                     subdomain_ijk[0] * cells_per_subdomain[0] + i,
            //                     subdomain_ijk[1] * cells_per_subdomain[1] + j,
            //                     subdomain_ijk[2] * cells_per_subdomain[2] + k,
            //                 ];
            //
            //
            //                 let global_point = parameters
            //                     .global_marching_cubes_grid
            //                     .get_point(global_point_ijk)
            //                     .expect("point has to be part of the global mc grid");
            //                 let point_coordinates = parameters
            //                     .global_marching_cubes_grid
            //                     .point_coordinates(&global_point);
            //
            //
            //                 let dx = p_i - point_coordinates;
            //
            //                 let dx_norm_sq = dx.norm_squared();
            //
            //
            //                 // levelset_grid[inxdx] +=  R::from(
            //                 //     dx_norm_sq
            //                 // ).expect("dddd");
            //
            //                 if dx_norm_sq < squared_support_with_margin {
            //                     let v_i = parameters.particle_rest_mass / rho_i;
            //                     let r = dx_norm_sq.sqrt();
            //
            //
            //                     let w_ij = kernel.evaluate(r);
            //
            //
            //                     // levelset_grid[inxdx] += w_ij;
            //
            //                     let interpolated_value = v_i * w_ij;
            //
            //                     let flat_point_idx = mc_grid.flatten_point_index(&local_point);
            //                     let flat_point_idx = flat_point_idx.to_usize().unwrap();
            //                     // levelset_grid[flat_point_idx] += interpolated_value;
            //
            //
            //                     if flat_point_idx == 234626 {
            //                         levelset_grid[0] += interpolated_value;
            //                         // println!(
            //                         //     "234626 (p:{}): {},{},{}\n{:?}",
            //                         //      inxdx, i, j, k,
            //                         //     p_i
            //                         // );
            //                     }
            //
            //
            //                     // if (flat_point_idx == 71089) {
            //                     //     levelset_grid[inxdx] += interpolated_value;
            //                     //     levelset_grid[0] += interpolated_value;
            //                     //     // out[id] += interpolated_value;
            //                     // }
            //                 }
            //             }
            //         }
            //     }
            // }

            // results.clone()
            //     .iter()
            //     .zip(levelset_grid.clone().iter())
            //     .for_each(|(x, y)| println!("{}\n{}", x, y));
            //
            // write_non_zero_values(
            //     format!("z-{:?}-cpu.txt", thread::current().id()),
            //     levelset_grid.clone().to_vec(),
            // );


            // let list: Vec<(usize, f64)> = results.clone()
            //     .into_iter()
            //     .zip(levelset_grid.clone().into_iter())
            //     .enumerate()
            //     .map(|(i, (x, y))| (i, x - y.to_f64().unwrap()))
            //     .sorted_by(|(i0, a), (i1, b)| PartialOrd::partial_cmp(a, b).unwrap())
            //
            //     // .map(|(i, (x, y))| (i, x - y.to_f64().unwrap()))
            //     // .sorted_by(|(i1, a), (i2, b)| PartialOrd::partial_cmp(b, a).unwrap())
            //     .collect();
            // write_non_zero_indexed_values(
            //     format!("z-{:?}-diff.txt", thread::current().id()),
            //     list.clone().to_vec(),
            // );
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

            for flat_cell_idx in I::range(I::zero(), mc_total_cells).iter() {
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

    let reconstruct_sparse = |flat_subdomain_idx: I, subdomain_particle_indices: &Vec<usize>| {
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

        index_cache.clear();

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

                                if levelset_grid[flat_point_idx] > parameters.surface_threshold {
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

            index_cache.sort_unstable();
            for flat_cell_idx in index_cache.iter().copied().dedup() {
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
    subdomains
        .flat_subdomain_indices
        .par_iter()
        .copied()
        .zip(subdomains.per_subdomain_particles.par_iter())
        .map(|(flat_subdomain_idx, subdomain_particle_indices)| {
            if subdomain_particle_indices.len() <= sparse_limit {
                profile!("subdomain reconstruction (sparse)", parent = parent);
                reconstruct_sparse(flat_subdomain_idx, subdomain_particle_indices)
            } else {
                profile!("subdomain reconstruction (dense)", parent = parent);
                reconstruct_dense(flat_subdomain_idx, subdomain_particle_indices)
            }
        })
        .collect_into_vec(&mut surface_patches);

    surface_patches
}



// // TODO: Reduce code duplication between dense and sparse
// pub(crate) fn reconstruction<I: Index, R: Real>(
//     parameters: &ParametersSubdomainGrid<I, R>,
//     global_particles: &[Vector3<R>],
//     global_particle_densities: &[R],
//     subdomains: &Subdomains<I>,
//     ocl_data: Arc<Mutex<OpenCLData>>,
// ) -> Vec<SurfacePatch<I, R>> {
//     profile!(parent, "reconstruction");
//
//     let squared_support = parameters.compact_support_radius * parameters.compact_support_radius;
//     // Add 1% so that we don't exclude grid points that are just on the kernel boundary
//     let squared_support_with_margin = squared_support * to_real!(1.01);
//     // Compute radial distance in terms of grid points we have to evaluate for each particle
//     let cube_radius = I::from((parameters.compact_support_radius / parameters.cube_size).ceil())
//         .expect("kernel radius in cubes has to fit in index type");
//     // Kernel
//     let kernel = CubicSplineKernel::new(parameters.compact_support_radius);
//     //let kernel = DiscreteSquaredDistanceCubicKernel::new::<f64>(1000, parameters.compact_support_radius);
//
//     let mc_total_cells = parameters.subdomain_cubes.cubed();
//     let mc_total_points = (parameters.subdomain_cubes + I::one()).cubed();
//
//     assert!(
//         mc_total_points.to_usize().is_some(),
//         "number of mc cubes per subdomain must be fit into usize"
//     );
//
//     let max_particles = subdomains
//         .per_subdomain_particles
//         .iter()
//         .map(|p| p.len())
//         .max()
//         .unwrap_or(0);
//     info!("Largest subdomain has {} particles.", max_particles);
//
//     // Maximum number of particles such that a subdomain will be considered "sparse"
//     let sparse_limit = (to_real!(max_particles) * to_real!(0.05))
//         .ceil()
//         .to_usize()
//         .unwrap()
//         .max(100);
//     info!(
//         "Subdomains with {} or less particles will be considered sparse.",
//         sparse_limit
//     );
//
//     info!("Starting reconstruction (level-set evaluation and local triangulation).");
//
//     // Returns a unique identifier for any edge index of a subdomain that can be later used for stitching
//     let globalize_local_edge = |mc_grid: &UniformCartesianCubeGrid3d<I, R>,
//                                 subdomain_grid: &UniformCartesianCubeGrid3d<I, R>,
//                                 subdomain_index: I,
//                                 local_edge: &EdgeIndex<I>|
//                                 -> (I, EdgeIndex<I>) {
//         // We globalize the boundary edge index by translating the local edge index to the subdomain
//         // where it lies on the lower boundary of that domain.
//
//         let max_mc_point_index = mc_grid.points_per_dim().map(|i| i - I::one());
//         let max_subdomain_index = subdomain_grid
//             .cells_per_dim()
//             .map(|i| i.saturating_sub(&I::one()).max(I::zero()));
//
//         // Check along which axes this edge is on the max boundary
//         let is_max = local_edge.axis().orthogonal_axes().map(|orth_axis| {
//             if local_edge.origin().index()[orth_axis.dim()] == max_mc_point_index[orth_axis.dim()] {
//                 // We are on the max side of this domain along the axis
//                 true
//             } else {
//                 // We are either
//                 //  - On the min side of this domain along the axis
//                 //  - Somewhere in the middle (in this case this axis is irrelevant)
//                 false
//             }
//         });
//
//         if !is_max[0] && !is_max[1] {
//             // Edge is already in the correct subdomain
//             (subdomain_index, local_edge.clone())
//         } else {
//             // We have to translate to the neighboring subdomain (+1 in all directions where is_max == true)
//             let subdomain_cell = subdomain_grid
//                 .try_unflatten_cell_index(subdomain_index)
//                 .expect("invalid subdomain index");
//
//             let mut target_subdomain_ijk = subdomain_cell.index().clone();
//             let mut target_local_origin_ijk = local_edge.origin().index().clone();
//
//             // Obtain index of new subdomain and new origin point
//             for (&orth_axis, &is_max) in local_edge
//                 .axis()
//                 .orthogonal_axes()
//                 .iter()
//                 .zip(is_max.iter())
//             {
//                 if is_max {
//                     // Clamp the step to the subdomain grid because we are not interested in subdomains outside the grid
//                     // (globalization is not needed on the outermost boundary of the entire problem domain)
//                     target_subdomain_ijk[orth_axis.dim()] = (target_subdomain_ijk[orth_axis.dim()]
//                         + I::one())
//                         .min(max_subdomain_index[orth_axis.dim()]);
//                     // Move origin point from max boundary to min boundary
//                     target_local_origin_ijk[orth_axis.dim()] = I::zero();
//                 }
//             }
//
//             let target_subdomain = subdomain_grid
//                 .get_cell(target_subdomain_ijk)
//                 .expect("target subdomain has to exist");
//             let flat_target_subdomain = subdomain_grid.flatten_cell_index(&target_subdomain);
//
//             // We re-use the same marching cubes domain here because the domain is anyway rectangular,
//             // therefore this shift gives the same result
//             let new_local_edge = mc_grid
//                 .get_edge(target_local_origin_ijk, local_edge.axis())
//                 .expect("failed to translate edge");
//
//             (flat_target_subdomain, new_local_edge)
//         }
//     };
//
//     #[derive(Default)]
//     struct SubdomainWorkspace<I: Index, R: Real> {
//         // Particle positions of this subdomain
//         subdomain_particles: Vec<Vector3<R>>,
//         // Per particle density values of this subdomain
//         subdomain_particle_densities: Vec<R>,
//         // Cache for the level-set values
//         levelset_grid: Vec<R>,
//         // Cache for indices
//         index_cache: Vec<I>,
//     }
//
//     let workspace_tls = ThreadLocal::<RefCell<SubdomainWorkspace<I, R>>>::new();
//
//     let reconstruct_subdomain = |flat_subdomain_idx: I, subdomain_particle_indices: &Vec<usize>, is_sparse: bool| {
//         // Obtain thread local workspace and clear it
//         let mut workspace = workspace_tls.get_or_default().borrow_mut();
//
//         let SubdomainWorkspace {
//             subdomain_particles,
//             subdomain_particle_densities,
//             levelset_grid,
//             index_cache,
//         } = &mut *workspace;
//
//         let flat_subdomain_idx: I = flat_subdomain_idx;
//         let subdomain_particle_indices: &[usize] = subdomain_particle_indices.as_slice();
//
//         // Collect all particle positions and densities of this subdomain
//         {
//             //profile!("collect subdomain data");
//             gather_subdomain_data(
//                 global_particles,
//                 subdomain_particle_indices,
//                 subdomain_particles,
//             );
//             gather_subdomain_data(
//                 global_particle_densities,
//                 subdomain_particle_indices,
//                 subdomain_particle_densities,
//             );
//         }
//
//         // Get the cell index and AABB of the subdomain
//         let subdomain_idx = parameters
//             .subdomain_grid
//             .try_unflatten_cell_index(flat_subdomain_idx)
//             .expect("Subdomain cell does not exist");
//         let subdomain_aabb = parameters.subdomain_grid.cell_aabb(&subdomain_idx);
//         let subdomain_ijk = subdomain_idx.index()
//             .clone()
//             .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
//
//         let mc_grid = UniformCartesianCubeGrid3d::new(
//             subdomain_aabb.min(),
//             &[parameters.subdomain_cubes; 3],
//             parameters.cube_size,
//         ).unwrap();
//         let cells_per_subdomain = mc_grid.cells_per_dim()
//             .clone()
//             .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
//
//
//         levelset_grid.fill(R::zero());
//         levelset_grid.resize(mc_total_points.to_usize().unwrap(), R::zero());
//
//         let extents = mc_grid.points_per_dim();
//
//         {
//             profile!("density grid loop");
//
//             // let nr_of_particles: usize = subdomain_particles.len();
//             //
//             // // Create a command_queue on the Context's device
//             // let queue = new_queue(&ocl_data.lock().context);
//             //
//             // // let (upper_lower_buffer, write_event) =
//             // //     new_write_buffer(&ocl_data.lock().context, &queue, 2 * nr_of_particles);
//             // let (levelset_grid_buffer, write_event): (Buffer<cl_double>, Event) =
//             //     new_write_buffer(&ocl_data.lock().context, &queue, levelset_grid.len());
//             //
//             // // Buffer of subdomain particles - Flattened Point Coordinates [x0, y0, z0, x1, y1, z0, ....]
//             // let flattened_ps = subdomain_particles
//             //     .iter()
//             //     .copied()
//             //     .zip(subdomain_particle_densities.iter().copied())
//             //     .flat_map(|(vec, rho)| [vec.x.to_f64().unwrap(), vec.y.to_f64().unwrap(), vec.z.to_f64().unwrap(), rho.to_f64().unwrap()])
//             //     .collect::<Vec<f64>>();
//             // let (sd_particles_buffer, write_event2): (Buffer<cl_double>, Event) =
//             //     as_read_buffer_non_blocking(&ocl_data.lock().context, &queue, &flattened_ps)
//             //         .expect("Failed to create particle positions buffer");
//             //
//             // let min = mc_grid.aabb().min();
//             // let bounds_kernel_event = unsafe {
//             //     // ExecuteKernel::new(ocl_data.lock().kernels.get(COMPUTE_BOUNDS_FUNCTION).unwrap())
//             //     ExecuteKernel::new(ocl_data.lock().kernels.get(RECONSTRUCT_FUNCTION).unwrap())
//             //         .set_arg(&cube_radius)
//             //         .set_arg(&extents[0]).set_arg(&extents[1]).set_arg(&extents[2])
//             //         .set_arg(&mc_grid.cell_size())
//             //         .set_arg(&min.x).set_arg(&min.y).set_arg(&min.z)
//             //
//             //         .set_arg(&parameters.compact_support_radius.to_f64().unwrap())
//             //
//             //         .set_arg(&sd_particles_buffer)
//             //         .set_wait_event(&write_event2)
//             //
//             //         // Output buffer
//             //         .set_arg(&levelset_grid_buffer)
//             //         .set_wait_event(&write_event)
//             //         // .set_arg(&upper_lower_buffer)
//             //         // .set_wait_event(&write_event)
//             //
//             //         .set_global_work_size(nr_of_particles)
//             //         .enqueue_nd_range(&queue)
//             //         .expect("Could not build, or execute kernel")
//             // };
//             //
//             // // TODO: No need to read if next kernel does not depend on it.
//             // let mut results = &mut vec![cl_double::zero(); levelset_grid.len()];
//             // read_buffer_into(&queue, &bounds_kernel_event, &levelset_grid_buffer, &mut results);
//             //
//             // exec_time!(&bounds_kernel_event, COMPUTE_BOUNDS_FUNCTION);
//
//
//
//             ////
//             ////
//             ////
//
//             // let queue = new_queue(&ocl_data.lock().context);
//
//             // let spd = subdomain_particle_densities.into_iter()
//             //     .map(|x| x.to_f64().unwrap())
//             //     .collect::<Vec<f64>>();
//             // let (sd_particle_density_buffer, write_event4): (Buffer<cl_double>, Event) =
//             //     as_read_buffer_non_blocking(&ocl_data.lock().context, &queue, &spd)
//             //         .expect("Failed to create particle density buffer");
//             //
//             //
//             // let np = mc_grid.points_per_dim().into_iter()
//             //     .map(|x| x.to_u64().unwrap())
//             //     .collect::<Vec<u64>>();
//             // let gmcg_n = parameters.global_marching_cubes_grid.points_per_dim()
//             //     .into_iter().map(|x| x.to_u64().unwrap())
//             //     .collect::<Vec<u64>>();
//             // let (long_args_buffer, write_event5): (Buffer<cl_ulong>, Event) =
//             //     as_read_buffer_non_blocking(&ocl_data.lock().context, &queue, &[
//             //         np[0], np[1], np[2],
//             //         subdomain_ijk[0], subdomain_ijk[1], subdomain_ijk[2],
//             //         cells_per_subdomain[0], cells_per_subdomain[1], cells_per_subdomain[2],
//             //         gmcg_n[0], gmcg_n[1], gmcg_n[2],
//             //     ]).expect("Failed to create particle density buffer");
//
//             //
//             // let density_kernel_event = unsafe {
//             //     ExecuteKernel::new(ocl_data.lock().kernels.get(DENSITY_GRID_LOOP_FUNCTION).unwrap())
//             //         .set_arg(&parameters.compact_support_radius.to_f64().unwrap())
//             //         .set_arg(&sd_particle_density_buffer)
//             //         .set_wait_event(&write_event4)
//             //         .set_arg(&sd_particles_buffer) // .set_wait_event(&write_event2) // sd_particles_buffer Already written at previous kernel
//             //         .set_arg(&upper_lower_buffer) // .set_wait_event(&write_event) // sd_particles_buffer Already written at previous kernel
//             //
//             //         .set_arg(&long_args_buffer)
//             //         .set_wait_event(&write_event5)
//             //
//             //         .set_global_work_size(nr_of_particles)
//             //         .enqueue_nd_range(&queue)
//             //         .expect("Could not build, or execute kernel")
//             // };
//
//             // exec_time!(&density_kernel_event, DENSITY_GRID_LOOP_FUNCTION);
//
//
//
//             for (p_i, rho_i) in subdomain_particles
//                 .iter()
//                 .copied()
//                 .zip(subdomain_particle_densities.iter().copied())
//             {
//                 // Get grid cell containing particle
//                 let particle_cell = mc_grid.enclosing_cell(&p_i);
//
//                 // Compute lower and upper bounds of the grid points possibly affected by the particle
//                 // We want to loop over the vertices of the enclosing cells plus all points in `cube_radius` distance from the cell
//
//                 let lower = [
//                     (particle_cell[0] - cube_radius).max(I::zero()),
//                     (particle_cell[1] - cube_radius).max(I::zero()),
//                     (particle_cell[2] - cube_radius).max(I::zero()),
//                 ];
//
//                 let upper = [
//                     // We add 2 because
//                     //  - we want to loop over all grid points of the cell (+1 for upper points) + the radius
//                     //  - the upper range limit is exclusive (+1)
//                     (particle_cell[0] + cube_radius + I::two()).min(extents[0]),
//                     (particle_cell[1] + cube_radius + I::two()).min(extents[1]),
//                     (particle_cell[2] + cube_radius + I::two()).min(extents[2]),
//                 ];
//
//                 // Loop over all grid points around the enclosing cell
//                 for i in I::range(lower[0], upper[0]).iter() {
//                     for j in I::range(lower[1], upper[1]).iter() {
//                         for k in I::range(lower[2], upper[2]).iter() {
//                             let point_ijk = [i, j, k];
//                             let local_point = mc_grid
//                                 .get_point(point_ijk)
//                                 .expect("point has to be part of the subdomain grid");
//
//
//                             let local_point_ijk = point_ijk
//                                 .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
//
//                             let [i, j, k] = local_point_ijk;
//
//                             // Use global coordinate calculation for consistency with neighboring domains
//                             let global_point_ijk = [
//                                 subdomain_ijk[0] * cells_per_subdomain[0] + i,
//                                 subdomain_ijk[1] * cells_per_subdomain[1] + j,
//                                 subdomain_ijk[2] * cells_per_subdomain[2] + k,
//                             ];
//
//                             let global_point = parameters
//                                 .global_marching_cubes_grid
//                                 .get_point(global_point_ijk)
//                                 .expect("point has to be part of the global mc grid");
//                             let point_coordinates = parameters
//                                 .global_marching_cubes_grid
//                                 .point_coordinates(&global_point);
//
//                             let dx = p_i - point_coordinates;
//                             let dx_norm_sq = dx.norm_squared();
//
//                             if dx_norm_sq < squared_support_with_margin {
//                                 let v_i = parameters.particle_rest_mass / rho_i;
//                                 let r = dx_norm_sq.sqrt();
//                                 let w_ij = kernel.evaluate(r);
//                                 //let w_ij = kernel.evaluate(dx_norm_sq);
//
//                                 let interpolated_value = v_i * w_ij;
//
//                                 let flat_point_idx = mc_grid.flatten_point_index(&local_point);
//                                 let flat_point_idx = flat_point_idx.to_usize().unwrap();
//                                 levelset_grid[flat_point_idx] += interpolated_value;
//
//                                 if is_sparse && levelset_grid[flat_point_idx] > parameters.surface_threshold {
//                                     for c in mc_grid
//                                         .cells_adjacent_to_point(
//                                             &mc_grid.get_point_neighborhood(&local_point),
//                                         )
//                                         .iter()
//                                         .flatten()
//                                     {
//                                         let flat_cell_index = mc_grid.flatten_cell_index(c);
//                                         index_cache.push(flat_cell_index);
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         };
//
//
//         let mut vertices = Vec::new();
//         let mut triangles = Vec::new();
//
//         let mut vertex_inside_count = 0;
//         let mut triangle_inside_count = 0;
//
//         let mut vertex_inside_flags = Vec::new();
//         let mut triangle_inside_flags = Vec::new();
//
//         let mut exterior_vertex_edge_indices = Vec::new();
//
//         let mut edge_to_vertex = new_map();
//
//
//         {
//             profile!("mc triangulation loop");
//             // let x = index_cache.iter().copied().dedup();
//
//             // let indexes = I::range(I::zero(), mc_total_cells).iter();
//             //TODO Fix this code:
//
//             // //: dyn Iterator<Item = I>
//             // let indexes: Box<dyn Iterator<Item = I>> = if is_sparse {
//             //     index_cache.sort_unstable();
//             //     Box::new(index_cache.iter().copied())
//             // } else {
//             //     Box::new(I::range(I::zero(), mc_total_cells).iter())
//             // };
//             // for flat_cell_idx in indexes {
//             for flat_cell_idx in I::range(I::zero(), mc_total_cells).iter() {
//
//                 let cell = mc_grid.try_unflatten_cell_index(flat_cell_idx).unwrap();
//
//
//                 let mut vertices_inside = [true; 8];
//                 for local_point_index in 0..8 {
//                     profile!("local_point_index");
//                     let point = cell.global_point_index_of(local_point_index).unwrap();
//                     let flat_point_idx = mc_grid.flatten_point_index(&point);
//                     let flat_point_idx = flat_point_idx.to_usize().unwrap();
//                     // Get value of density map
//                     let density_value = levelset_grid[flat_point_idx];
//                     // Update inside/outside surface flag
//                     vertices_inside[local_point_index] =
//                         density_value > parameters.surface_threshold;
//                 }
//
//                 for triangle in marching_cubes_triangulation_iter(&vertices_inside) {
//                     profile!("triangle");
//                     let mut global_triangle = [0; 3];
//                     for (v_idx, local_edge_index) in triangle.iter().copied().enumerate() {
//                         profile!("v_idx");
//                         let edge = cell
//                             .global_edge_index_of(local_edge_index as usize)
//                             .unwrap();
//                         let vertex_index = *edge_to_vertex.entry(edge).or_insert_with(|| {
//                             // TODO: Nonlinear interpolation
//
//                             let origin_coords = mc_grid.point_coordinates(&edge.origin());
//                             let target_coords = mc_grid.point_coordinates(&edge.target());
//
//                             let flat_origin_idx = mc_grid
//                                 .flatten_point_index(&edge.origin())
//                                 .to_usize()
//                                 .unwrap();
//                             let flat_target_idx = mc_grid
//                                 .flatten_point_index(&edge.target())
//                                 .to_usize()
//                                 .unwrap();
//
//                             let origin_value = levelset_grid[flat_origin_idx];
//                             let target_value = levelset_grid[flat_target_idx];
//
//                             let alpha = (parameters.surface_threshold - origin_value)
//                                 / (target_value - origin_value);
//                             let interpolated_coords =
//                                 origin_coords * (R::one() - alpha) + target_coords * alpha;
//                             let vertex_coords = interpolated_coords;
//
//                             vertices.push(vertex_coords);
//                             let vertex_index = vertices.len() - 1;
//
//                             let is_interior_vertex = !mc_grid.is_boundary_edge(&edge);
//                             vertex_inside_count += is_interior_vertex as usize;
//                             vertex_inside_flags.push(is_interior_vertex);
//
//                             if !is_interior_vertex {
//                                 exterior_vertex_edge_indices.push(globalize_local_edge(
//                                     &mc_grid,
//                                     &parameters.subdomain_grid,
//                                     flat_subdomain_idx,
//                                     &edge,
//                                 ));
//                             }
//
//                             vertex_index
//                         });
//
//                         global_triangle[v_idx] = vertex_index;
//                     }
//
//                     let all_tri_vertices_inside = global_triangle
//                         .iter()
//                         .copied()
//                         .all(|v_idx| vertex_inside_flags[v_idx]);
//
//                     triangles.push(global_triangle);
//                     triangle_inside_count += all_tri_vertices_inside as usize;
//                     triangle_inside_flags.push(all_tri_vertices_inside);
//                 }
//             }
//         }
//
//         SurfacePatch {
//             vertices,
//             triangles,
//             vertex_inside_count,
//             triangle_inside_count,
//             vertex_inside_flags,
//             triangle_inside_flags,
//             exterior_vertex_edge_indices,
//         }
//     };
//     let reconstruct_sparse = |flat_subdomain_idx: I, subdomain_particle_indices: &Vec<usize>| {
//         // Obtain thread local workspace and clear it
//         let mut workspace = workspace_tls.get_or_default().borrow_mut();
//
//         let SubdomainWorkspace {
//             subdomain_particles,
//             subdomain_particle_densities,
//             levelset_grid,
//             index_cache,
//         } = &mut *workspace;
//
//         let flat_subdomain_idx: I = flat_subdomain_idx;
//         let subdomain_particle_indices: &[usize] = subdomain_particle_indices.as_slice();
//
//         // Collect all particle positions and densities of this subdomain
//         {
//             //profile!("collect subdomain data");
//             gather_subdomain_data(
//                 global_particles,
//                 subdomain_particle_indices,
//                 subdomain_particles,
//             );
//             gather_subdomain_data(
//                 global_particle_densities,
//                 subdomain_particle_indices,
//                 subdomain_particle_densities,
//             );
//         }
//
//         // Get the cell index and AABB of the subdomain
//         let subdomain_idx = parameters
//             .subdomain_grid
//             .try_unflatten_cell_index(flat_subdomain_idx)
//             .expect("Subdomain cell does not exist");
//         let subdomain_aabb = parameters.subdomain_grid.cell_aabb(&subdomain_idx);
//
//         let mc_grid = UniformCartesianCubeGrid3d::new(
//             subdomain_aabb.min(),
//             &[parameters.subdomain_cubes; 3],
//             parameters.cube_size,
//         )
//             .unwrap();
//
//         levelset_grid.fill(R::zero());
//         levelset_grid.resize(mc_total_points.to_usize().unwrap(), R::zero());
//
//         index_cache.clear();
//
//         {
//             profile!("density grid loop");
//
//             let extents = mc_grid.points_per_dim();
//
//             for (p_i, rho_i) in subdomain_particles
//                 .iter()
//                 .copied()
//                 .zip(subdomain_particle_densities.iter().copied())
//             {
//                 // Get grid cell containing particle
//                 let particle_cell = mc_grid.enclosing_cell(&p_i);
//
//                 // Compute lower and upper bounds of the grid points possibly affected by the particle
//                 // We want to loop over the vertices of the enclosing cells plus all points in `cube_radius` distance from the cell
//
//                 let lower = [
//                     (particle_cell[0] - cube_radius).max(I::zero()),
//                     (particle_cell[1] - cube_radius).max(I::zero()),
//                     (particle_cell[2] - cube_radius).max(I::zero()),
//                 ];
//
//                 let upper = [
//                     // We add 2 because
//                     //  - we want to loop over all grid points of the cell (+1 for upper points) + the radius
//                     //  - the upper range limit is exclusive (+1)
//                     (particle_cell[0] + cube_radius + I::two()).min(extents[0]),
//                     (particle_cell[1] + cube_radius + I::two()).min(extents[1]),
//                     (particle_cell[2] + cube_radius + I::two()).min(extents[2]),
//                 ];
//
//                 // Loop over all grid points around the enclosing cell
//                 for i in I::range(lower[0], upper[0]).iter() {
//                     for j in I::range(lower[1], upper[1]).iter() {
//                         for k in I::range(lower[2], upper[2]).iter() {
//                             let point_ijk = [i, j, k];
//                             let local_point = mc_grid
//                                 .get_point(point_ijk)
//                                 .expect("point has to be part of the subdomain grid");
//                             //let point_coordinates = mc_grid.point_coordinates(&point);
//
//                             let subdomain_ijk = subdomain_idx.index();
//                             let mc_cells_per_subdomain = mc_grid.cells_per_dim();
//
//                             fn local_to_global_point_ijk<I: Index>(
//                                 local_point_ijk: [I; 3],
//                                 subdomain_ijk: [I; 3],
//                                 cells_per_subdomain: [I; 3],
//                             ) -> [GlobalIndex; 3] {
//                                 let local_point_ijk = local_point_ijk
//                                     .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
//                                 let subdomain_ijk = subdomain_ijk
//                                     .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
//                                 let cells_per_subdomain = cells_per_subdomain
//                                     .map(|i| <GlobalIndex as NumCast>::from(i).unwrap());
//                                 let [i, j, k] = local_point_ijk;
//
//                                 [
//                                     subdomain_ijk[0] * cells_per_subdomain[0] + i,
//                                     subdomain_ijk[1] * cells_per_subdomain[1] + j,
//                                     subdomain_ijk[2] * cells_per_subdomain[2] + k,
//                                 ]
//                             }
//
//                             // Use global coordinate calculation for consistency with neighboring domains
//                             let global_point_ijk = local_to_global_point_ijk(
//                                 point_ijk,
//                                 subdomain_ijk.clone(),
//                                 mc_cells_per_subdomain.clone(),
//                             );
//                             let global_point = parameters
//                                 .global_marching_cubes_grid
//                                 .get_point(global_point_ijk)
//                                 .expect("point has to be part of the global mc grid");
//                             let point_coordinates = parameters
//                                 .global_marching_cubes_grid
//                                 .point_coordinates(&global_point);
//
//                             let dx = p_i - point_coordinates;
//                             let dx_norm_sq = dx.norm_squared();
//
//                             if dx_norm_sq < squared_support_with_margin {
//                                 let v_i = parameters.particle_rest_mass / rho_i;
//                                 let r = dx_norm_sq.sqrt();
//                                 let w_ij = kernel.evaluate(r);
//                                 //let w_ij = kernel.evaluate(dx_norm_sq);
//
//                                 let interpolated_value = v_i * w_ij;
//
//                                 let flat_point_idx = mc_grid.flatten_point_index(&local_point);
//                                 let flat_point_idx = flat_point_idx.to_usize().unwrap();
//                                 levelset_grid[flat_point_idx] += interpolated_value;
//
//                                 if levelset_grid[flat_point_idx] > parameters.surface_threshold {
//                                     for c in mc_grid
//                                         .cells_adjacent_to_point(
//                                             &mc_grid.get_point_neighborhood(&local_point),
//                                         )
//                                         .iter()
//                                         .flatten()
//                                     {
//                                         let flat_cell_index = mc_grid.flatten_cell_index(c);
//                                         index_cache.push(flat_cell_index);
//                                     }
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//
//         let mut vertices = Vec::new();
//         let mut triangles = Vec::new();
//
//         let mut vertex_inside_count = 0;
//         let mut triangle_inside_count = 0;
//
//         let mut vertex_inside_flags = Vec::new();
//         let mut triangle_inside_flags = Vec::new();
//
//         let mut exterior_vertex_edge_indices = Vec::new();
//
//         let mut edge_to_vertex = new_map();
//
//         {
//             profile!("mc triangulation loop");
//
//             index_cache.sort_unstable();
//             for flat_cell_idx in index_cache.iter().copied().dedup() {
//                 let cell = mc_grid.try_unflatten_cell_index(flat_cell_idx).unwrap();
//
//                 let mut vertices_inside = [true; 8];
//                 for local_point_index in 0..8 {
//                     let point = cell.global_point_index_of(local_point_index).unwrap();
//                     let flat_point_idx = mc_grid.flatten_point_index(&point);
//                     let flat_point_idx = flat_point_idx.to_usize().unwrap();
//                     // Get value of density map
//                     let density_value = levelset_grid[flat_point_idx];
//                     // Update inside/outside surface flag
//                     vertices_inside[local_point_index] =
//                         density_value > parameters.surface_threshold;
//                 }
//
//                 for triangle in marching_cubes_triangulation_iter(&vertices_inside) {
//                     let mut global_triangle = [0; 3];
//                     for (v_idx, local_edge_index) in triangle.iter().copied().enumerate() {
//                         let edge = cell
//                             .global_edge_index_of(local_edge_index as usize)
//                             .unwrap();
//                         let vertex_index = *edge_to_vertex.entry(edge).or_insert_with(|| {
//                             // TODO: Nonlinear interpolation
//
//                             let origin_coords = mc_grid.point_coordinates(&edge.origin());
//                             let target_coords = mc_grid.point_coordinates(&edge.target());
//
//                             let flat_origin_idx = mc_grid
//                                 .flatten_point_index(&edge.origin())
//                                 .to_usize()
//                                 .unwrap();
//                             let flat_target_idx = mc_grid
//                                 .flatten_point_index(&edge.target())
//                                 .to_usize()
//                                 .unwrap();
//
//                             let origin_value = levelset_grid[flat_origin_idx];
//                             let target_value = levelset_grid[flat_target_idx];
//
//                             let alpha = (parameters.surface_threshold - origin_value)
//                                 / (target_value - origin_value);
//                             let interpolated_coords =
//                                 origin_coords * (R::one() - alpha) + target_coords * alpha;
//                             let vertex_coords = interpolated_coords;
//
//                             vertices.push(vertex_coords);
//                             let vertex_index = vertices.len() - 1;
//
//                             let is_interior_vertex = !mc_grid.is_boundary_edge(&edge);
//                             vertex_inside_count += is_interior_vertex as usize;
//                             vertex_inside_flags.push(is_interior_vertex);
//
//                             if !is_interior_vertex {
//                                 exterior_vertex_edge_indices.push(globalize_local_edge(
//                                     &mc_grid,
//                                     &parameters.subdomain_grid,
//                                     flat_subdomain_idx,
//                                     &edge,
//                                 ));
//                             }
//
//                             vertex_index
//                         });
//
//                         global_triangle[v_idx] = vertex_index;
//                     }
//
//                     let all_tri_vertices_inside = global_triangle
//                         .iter()
//                         .copied()
//                         .all(|v_idx| vertex_inside_flags[v_idx]);
//
//                     triangles.push(global_triangle);
//                     triangle_inside_count += all_tri_vertices_inside as usize;
//                     triangle_inside_flags.push(all_tri_vertices_inside);
//                 }
//             }
//         }
//
//         SurfacePatch {
//             vertices,
//             triangles,
//             vertex_inside_count,
//             triangle_inside_count,
//             vertex_inside_flags,
//             triangle_inside_flags,
//             exterior_vertex_edge_indices,
//         }
//     };
//
//     let mut surface_patches = Vec::with_capacity(subdomains.flat_subdomain_indices.len());
//     subdomains.flat_subdomain_indices
//         .par_iter()
//         .copied()
//         .zip(subdomains.per_subdomain_particles.par_iter())
//         .map(|(flat_subdomain_idx, subdomain_particle_indices)| {
//             if subdomain_particle_indices.len() <= sparse_limit {
//                 profile!("subdomain reconstruction (sparse)", parent = parent);
//                 // reconstruct_sparse(flat_subdomain_idx, subdomain_particle_indices)
//                 reconstruct_subdomain(flat_subdomain_idx, subdomain_particle_indices, true)
//             } else {
//                 profile!("subdomain reconstruction (dense)", parent = parent);
//                 // reconstruct_subdomain(flat_subdomain_idx, subdomain_particle_indices)
//                 reconstruct_subdomain(flat_subdomain_idx, subdomain_particle_indices, false)
//             }
//         })
//         .collect_into_vec(&mut surface_patches);
//
//     surface_patches
// }
