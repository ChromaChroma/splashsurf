// #pragma OPENCL EXTENSION cl_khr_gl_sharing : enable
// #pragma OPENCL EXTENSION cl_khr_gl_depth_images : enable

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL SELECT_ROUNDING_MODE rtp
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define CL_M_PI 3.14159265358979323846264338327950288

void AtomicAdd(__global double *val, double delta) {
  union {
    double f;
    ulong i;
  } old;
  union {
    double f;
    ulong i;
  } new;

  do {
    old.f = *val;
    new.f = old.f + delta;
  } while (atom_cmpxchg((volatile __global ulong *)val, old.i, new.i) != old.i);
}


// double atomic_add_float_global(__global double* p, double val)
// {
//     double prev;
//     asm volatile(
//         "atom.global.add.f64 %0, [%1], %2;"
//         : "=f"(prev)
//         : "l"(p) , "f"(val)
//         : "memory"
//     );
//     return prev;
// }

kernel void reconstruct(
    const double sswm, // squared_support_with_margin
    const double prm,  // particle_rest_mass
    const double gmcg_cs, const long cube_radius, const long extents_x,
    const long extents_y, const long extents_z, const double lmc_cell_size,
    const double local_mcg_aab_min_x, const double local_mcg_aab_min_y,
    const double local_mcg_aab_min_z,
    const double csr, // compact_support_radius
    // id + (0-2) : np
    // id + (3-5) : subdomain_ijk
    // id + (6-8) : cells_per_subdomain
    // id + (9-11) : gmcg_np
    global const long *long_args,
    global const double *gmcg_aabb_np, // global_marching_cubes_grid__aabb_min

    // xyz coords of all points + density ([x, y, z, rho])
    global const double *point_coords,
    // // Output buffer (2x longer than point_coords)
    // // Array with xyz of lower and upper bounds
    // // 6 indexes (first 3 for lower-xyz, second 3 for
    // upper-xyz) global long *bounds

    // Output buffer
    global double *levelset_grid) {
  const size_t id = get_global_id(0);

  const double pi_x = point_coords[id * 4];
  const double pi_y = point_coords[id * 4 + 1];
  const double pi_z = point_coords[id * 4 + 2];
  const double3 p_i_double3 =
      (double3)(point_coords[id * 4], point_coords[id * 4 + 1],
                point_coords[id * 4 + 2]);

  long3 particle_cell =
      (long3)((long)floor((pi_x - local_mcg_aab_min_x) / lmc_cell_size),
              (long)floor((pi_y - local_mcg_aab_min_y) / lmc_cell_size),
              (long)floor((pi_z - local_mcg_aab_min_z) / lmc_cell_size));

  const long3 lower =
      (long3)((long)max(particle_cell.x - cube_radius, (long)0),
              (long)max(particle_cell.y - cube_radius, (long)0),
              (long)max(particle_cell.z - cube_radius, (long)0));

  const long3 upper =
      (long3)((long)min(particle_cell.x + cube_radius + 2, extents_x),
              (long)min(particle_cell.y + cube_radius + 2, extents_y),
              (long)min(particle_cell.z + cube_radius + 2, extents_z));

  long lower_x = (long)max(particle_cell.x - cube_radius, (long)0);
  long lower_y = (long)max(particle_cell.y - cube_radius, (long)0);
  long lower_z = (long)max(particle_cell.z - cube_radius, (long)0);
  long upper_x = (long)min(particle_cell.x + cube_radius + 2, extents_x);
  long upper_y = (long)min(particle_cell.y + cube_radius + 2, extents_y);
  long upper_z = (long)min(particle_cell.z + cube_radius + 2, extents_z);

  double normalization_sigma = 8.0 / (csr * csr * csr);

  // levelset_grid[id] = (upper_x- lower_x) * (upper_y - lower_y) * (upper_z - lower_z) ;

  for (int x = lower_x; x < upper_x; x++) {
    for (int y = lower_y; y < upper_y; y++) {
      for (int z = lower_z; z < upper_z; z++) {

        if (x >= long_args[0] || y >= long_args[1] || z >= long_args[2] ||
            x < 0 || y < 0 || z < 0) { // Use error out bool?
          return;
        }

        // Global Point
        long3 gp = (long3)((long)(long_args[3] * long_args[6] + x),
                           (long)(long_args[4] * long_args[7] + y),
                           (long)(long_args[5] * long_args[8] + z));

        if (gp.x >= long_args[9] || gp.y >= long_args[10] ||
            gp.z >= long_args[11] || gp.x < 0 || gp.y < 0 ||
            gp.z < 0) { // Use error out bool?
          return;
        }

        double3 gmcg_aabb_min =
            (double3)(gmcg_aabb_np[0], gmcg_aabb_np[1],
                      gmcg_aabb_np[2]); // Todouble3(gmcg_aabb_np);
        double3 point_coordinates =
            gmcg_aabb_min + (double3)((double)(gp.x * gmcg_cs),
                                      (double)(gp.y * gmcg_cs),
                                      (double)(gp.z * gmcg_cs));
        double3 dx = (double3)p_i_double3 - point_coordinates;

        double dx_norm_sq = dot(dx, dx);

        if (dx_norm_sq < sswm) {

          // // Particle rest mass (parameters[1]) divided by Particle density
          double v_i = prm / point_coords[id * 4 + 3];
          double r = sqrt(dx_norm_sq);

          // kernel.evaluate(r)
          double q = (r + r) / csr;
          double cube_q = 0.0;
          if (q < 1.0) {
            cube_q = (3.000000 / (2.000000 * CL_M_PI)) *
                     ((2.000000 / 3.000000) - q * q + 0.5000000 * q * q * q);
          } else if (q < 2.000000) {
            double x = 2.0 - q;
            cube_q = (1.000000 / (4.0000000 * CL_M_PI)) * x * x * x;
          }
          double w_ij = normalization_sigma * cube_q;

          double interpolated_value = v_i * w_ij;

          long flat_point_idx =
              x * long_args[1] * long_args[2] + y * long_args[2] + z;
          AtomicAdd(levelset_grid + flat_point_idx, interpolated_value);

          // atom_add(levelset_grid + flat_point_idx, interpolated_value);
          // levelset_grid[flat_point_idx] += interpolated_value;

          //   if (flat_point_idx == 234626) {

          //     // double prev;
          //     // asm volatile(
          //     //     "atom.global.add.f64 %0, [%1], %2;"
          //     //     : "=f"(prev)
          //     //     : "l"(levelset_grid) , "f"(1.0)
          //     //     : "memory"
          //     // );

          //     // atomic_add_float_global(levelset_grid, 1);

          //     // atom_add(levelset_grid, interpolated_value);
          //     AtomicAdd(levelset_grid, interpolated_value);
          //     // levelset_grid[0] += interpolated_value;
          //   }
          // };

          // if is_sparse && levelset_grid[flat_point_idx] >
          // parameters.surface_threshold {
          //                           for c in mc_grid
          //                               .cells_adjacent_to_point(
          //                                   &mc_grid.get_point_neighborhood(&local_point),
          //                               )
          //                               .iter()
          //                               .flatten()
          //                           {
          //                               let flat_cell_index =
          //                               mc_grid.flatten_cell_index(c);
          //                               index_cache.push(flat_cell_index);
          //                           }
        }
      }
    }
  }
}
