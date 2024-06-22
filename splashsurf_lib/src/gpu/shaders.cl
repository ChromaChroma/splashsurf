#define CL_M_PI 3.141592653589793115998

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
  const double3 p_i_double3 = (double3)(
      point_coords[id*4    ],
      point_coords[id*4 + 1],
      point_coords[id*4 + 2]);
    

 long3 particle_cell = (long3)(
        (long)floor((pi_x - local_mcg_aab_min_x) / lmc_cell_size),
        (long)floor((pi_y - local_mcg_aab_min_y) / lmc_cell_size),
        (long)floor((pi_z - local_mcg_aab_min_z) / lmc_cell_size)
    );

  const long3 lower = (long3)(
      (long)max(particle_cell.x - cube_radius, (long)0),
      (long)max(particle_cell.y - cube_radius, (long)0),
      (long)max(particle_cell.z - cube_radius, (long)0)
  );


  const long3 upper = (long3)(
      (long)min(particle_cell.x + cube_radius + 2, extents_x),
      (long)min(particle_cell.y + cube_radius + 2, extents_y),
      (long)min(particle_cell.z + cube_radius + 2, extents_z)
  );
    


  long lower_x = (long)max(particle_cell.x - cube_radius, (long)0);
  long lower_y = (long)max(particle_cell.y - cube_radius, (long)0);
  long lower_z = (long)max(particle_cell.z - cube_radius, (long)0);
  long upper_x = (long)min(particle_cell.x + cube_radius + 2, extents_x);
  long upper_y = (long)min(particle_cell.y + cube_radius + 2, extents_y);
  long upper_z = (long)min(particle_cell.z + cube_radius + 2, extents_z);

  double normalization_sigma = 8.0 / (csr * csr * csr);

  levelset_grid[id] = (long)max(particle_cell.x, (long)0.0);

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

        // // double3 gmcg_aabb_min = Todouble3(gmcg_aabb_np);
        // // double3 point_coordinates =
        // //     gmcg_aabb_min + (double3)((double)(gp.x * gmcg_cs),
        // //                               (double)(gp.y * gmcg_cs),
        // //                               (double)(gp.z * gmcg_cs));
        // // double3 dx = (double3) p_i_double3 - point_coordinates;
        double3 dx =
            (double3)((double)pi_x -
                          (gmcg_aabb_np[0] + (double)(gp.x * gmcg_cs)),
                      (double)pi_y -
                          (gmcg_aabb_np[1] + (double)(gp.y * gmcg_cs)),
                      (double)pi_z -
                          (gmcg_aabb_np[2] + (double)(gp.z * gmcg_cs)));

        double dx_norm_sq = dot(dx, dx);

        long flat_point_idx = x * long_args[1] * long_args[2] + y * long_args[2] + z;
        levelset_grid[flat_point_idx] += dx_norm_sq;
        levelset_grid[id] = dx_norm_sq;

        // if dx_norm_sq < squared_support_with_margin (parameters[0])
        if (dx_norm_sq < sswm) {

          // // Particle rest mass (parameters[1]) divided by Particle density
          double v_i = prm / point_coords[id *4 + 3];
          double r = sqrt(dx_norm_sq);

          // kernel.evaluate(r)
          double q = (r + r) / csr;
          double cube_q = 0.0;
          if (q < 1.0) {
            cube_q = (3.0 / (2.0 * CL_M_PI)) *
                     ((2.0 / 3.0) - q * q + 0.5 * q * q * q);
          } else if (q < 2.0) {
            double x = 2.0 - q;
            cube_q = (1.0 / (4.0 * CL_M_PI)) * x * x * x;
          }
          double w_ij = normalization_sigma * cube_q;

          double interpolated_value = v_i * w_ij;

          long flat_point_idx = x * long_args[1] * long_args[2] + y * long_args[2] + z;
          levelset_grid[flat_point_idx] += w_ij;
        };




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
        //                       }
      }
    }
  }
}

// kernel void compute_lower_and_upper(
//     const long cube_radius, const long extents_x, const long extents_y,
//     const long extents_z, const float lmc_cell_size,
//     const float local_mcg_aab_min_x, const float local_mcg_aab_min_y,
//     const float local_mcg_aab_min_z,
//     // xyz coords of all points
//     global const double *point_coords,
//     // Output buffer (2x longer than point_coords)
//     // Array with xyz of lower and upper bounds
//     // 6 indexes (first 3 for lower-xyz, second 3 for upper-xyz)
//     global long *bounds) {
//   const size_t id = get_global_id(0);

//   const double pi_x = point_coords[id * 3];
//   const double pi_y = point_coords[id * 3 + 1];
//   const double pi_z = point_coords[id * 3 + 2];

//   const long3 particle_cell =
//       (long3)((long)floor((pi_x - local_mcg_aab_min_x) / lmc_cell_size),
//               (long)floor((pi_y - local_mcg_aab_min_y) / lmc_cell_size),
//               (long)floor((pi_z - local_mcg_aab_min_z) / lmc_cell_size));

//   bounds[id * 3] = (long)max(particle_cell.x - cube_radius, (long)0);
//   bounds[id * 3 + 1] = (long)max(particle_cell.y - cube_radius, (long)0);
//   bounds[id * 3 + 2] = (long)max(particle_cell.z - cube_radius, (long)0);
//   bounds[id * 3 + 3] = (long)min(particle_cell.x + cube_radius + 2,
//   extents_x); bounds[id * 3 + 4] = (long)min(particle_cell.y + cube_radius +
//   2, extents_y); bounds[id * 3 + 5] = (long)min(particle_cell.z + cube_radius
//   + 2, extents_z);
// }

// kernel void
// density_grid_loop(const double csr, // compact_support_radius
//                   global const int *rho_i,
//                   // xyz coords of all points
//                   global const double *point_coords,
//                   // Array with xyz of lower and upper bounds
//                   // 6 indexes (first 3 for lower-xyz, second 3 for
//                   upper-xyz) global long *bounds,

//                   // , global long* np
//                   // , global long* subdomain_ijk
//                   // , global long* cells_per_subdomain
//                   // , global long* gmcg_np // global_marching_cubes_grid__np

//                   // id + (0-2) : np
//                   // id + (3-5) : subdomain_ijk
//                   // id + (6-8) : cells_per_subdomain
//                   // id + (9-11) : gmcg_np
//                   global long* long_args

//                 ) {
//   const size_t particle_idx = get_global_id(0);
//   const size_t idx = particle_idx * 6;

//   const double normalization_sigma = 8.0 / (csr * csr * csr);
//   return;

//   for (int x = bounds[idx]; x < bounds[idx + 3]; x++) {
//     for (int y = bounds[idx + 1]; y < bounds[idx + 4]; y++) {
//       for (int z = bounds[idx + 2]; z < bounds[idx + 5]; z++) {
//         // const bool zzz = x >= long_args[0] || y >= long_args[1] || z >=
//         long_args[2] || x < 0 || y < 0 || z < 0;
//         // if (x >= long_args[0] || y >= long_args[1] || z >= long_args[2] ||
//         x < 0 || y < 0 || z < 0) { // Use error out bool?
//         //   return;
//         // }

//         // // Global Point
//         // long3 gp =
//         //     (long3)((long)(long_args[3] * long_args[6] + x),
//         //             (long)(long_args[4] * long_args[7] + y),
//         //             (long)(long_args[5] * long_args[8] + z));

//         // if (gp.x >= long_args[9] || gp.y >= long_args[10] || gp.z >=
//         long_args[11] ||
//         //     gp.x < 0 || gp.y < 0 || gp.z < 0) { // Use error out bool?
//         //   return;
//         // }

//         // // double3 gmcg_aabb_min = Todouble3(gmcg_aabb_np);
//         // // double3 point_coordinates =
//         // //     gmcg_aabb_min + (double3)((double)(gp.x * gmcg_cs),
//         // //                               (double)(gp.y * gmcg_cs),
//         // //                               (double)(gp.z * gmcg_cs));
//         // // double3 dx = (double3) p_i_double3 - point_coordinates;
//         // double3 dx = (double3)((double) point_coords[particle_idx*3    ] -
//         (gmcg_aabb_np[0] +(double)(gp.x * gmcg_cs)),
//         //                        (double) point_coords[particle_idx*3 + 1] -
//         (gmcg_aabb_np[1] +(double)(gp.y * gmcg_cs)),
//         //                        (double) point_coords[particle_idx*3 + 2] -
//         (gmcg_aabb_np[2] +(double)(gp.z * gmcg_cs)))

//         // double dx_norm_sq = dot(dx, dx);

//       //   // if dx_norm_sq < squared_support_with_margin (parameters[0])
//       //   if (dx_norm_sq < parameters[0]) {

//       //     // // Particle rest mass (parameters[1]) divided by Particle
//       //     density double v_i = parameters[1] / rho_id; double r =
//       //     sqrt(dx_norm_sq);

//       //     // kernel.evaluate(r)
//       //     double q = (r + r) / parameters[2];
//       //     double cube_q = 0.0;
//       //     if (q < 1.0) {
//       //       cube_q = (3.0 / (2.0 * CL_M_PI)) *
//       //                ((2.0 / 3.0) - q * q + 0.5 * q * q * q);
//       //     } else if (q < 2.0) {
//       //       double x = 2.0 - q;
//       //       cube_q = (1.0 / (4.0 * CL_M_PI)) * x * x * x;
//       //     }
//       //     double w_ij = normalization_sigma * cube_q;

//       //     double interpolated_value = v_i * w_ij;

//       //     long flat_point_idx = x * np[1] * np[2] + y * np[2] + z;
//       //     out[flat_point_idx] += interpolated_value;
//       //   };

//       // if is_sparse && levelset_grid[flat_point_idx] >
//       parameters.surface_threshold {
//       //                           for c in mc_grid
//       //                               .cells_adjacent_to_point(
//       // &mc_grid.get_point_neighborhood(&local_point),
//       //                               )
//       //                               .iter()
//       //                               .flatten()
//       //                           {
//       //                               let flat_cell_index =
//       mc_grid.flatten_cell_index(c);
//       //                               index_cache.push(flat_cell_index);
//       //                           }
//       //                       }
//       }
//     }
//   }
// }