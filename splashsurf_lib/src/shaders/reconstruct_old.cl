#define CL_M_PI 3.141592653589793115998


__kernel void reconstruct_dense2(
    global int* output,
    global int* np,
    int delta_y,
    int delta_z,
)
{
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const size_t z = get_global_id(1);

    if ( x >= np[0] ||  y >= np[1] ||  z >= np[2] || x < 0 || y < 0 || z < 0) {
    return;
  }

    output[x + delta_y * (y + delta_z * z)] = x + y + z;
}




// Should be ran for range in lower to upper 3d range, (+ lower as value where
// is added on)
__kernel void reconstruct_dense(
    // Output vec
    __global float *levelset_grid, 
		//__global int *index_cache, // not used I think in reconstruct_dense

    // parameters
    // [0] : squared_support_with_margin
    // [1] : particle_rest_mass
    // [2] : compact_support_radius
    // [3] : normalization
    // [4] : surface_threshold
    __constant float *parameters,
    // n_points_per_dim (of MC grid)
    int3 np,

    // n_points_per_dim of global_marching_cubes_grid
    int3 global_marching_cubes_grid__np,
    // Min Vector3 of aabb of global MC grid
    float3 global_marching_cubes_grid__aabb_min,
    float global_marching_cubes_grid__cell_size,

    // subdomain particles
    // float3 p_i,
    float3 p_i, float rho_i,
    // for loop ranges
    int3 lower, int3 upper, int3 subdomain_ijk, int3 cells_per_subdomain) {
  // Check if x_idx is 0->n or lower+0 -> lower+n
  int x_idx = get_global_id(0);
  int z_idx = get_global_id(1);
  int y_idx = get_global_id(2);

  // Check if indices are in range for the computation
  // if ((x_idx < lower[0]) || (x_idx > upper[0])
  // 	|| (y_idx < lower[1]) || (y_idx > upper[1])
  //  	|| (z_idx < lower[2]) || (z_idx > upper[2])) return;
  if ((x_idx < lower.x) || (x_idx > upper.x) || (y_idx < lower.y) ||
      (y_idx > upper.y) || (z_idx < lower.z) || (z_idx > upper.z))
    return;

  // // Gets point on MC grid
  // //[I; 3]
  // int *local_point = mc_grid.get_point([
  // 		lower[0] + x_idx,
  // 		lower[1] + y_idx,
  // 		lower[2] + z_idx,
  // 	])
  // 	.expect("point has to be part of the subdomain grid");
  if ( // If NOT exists return (SHOULD THROW ERROR OR STOP, e.g. .expect("point
       // has to be part of the subdomain grid"))
      x_idx >= np.x || 
			y_idx >= np.y || 
			z_idx >= np.z ||
      x_idx < 0 || y_idx < 0 || z_idx < 0) {
    // Use error out bool?
    return;
  }
  // else

  // int3 global_point_ijk = (int3)(
  // 	subdomain_ijk[0] * cells_per_subdomain[0] + x_idx,
  // 	subdomain_ijk[1] * cells_per_subdomain[1] + y_idx,
  // 	subdomain_ijk[2] * cells_per_subdomain[2] + z_idx
  // 	);
  int3 global_point_ijk =
      (int3)((int)(subdomain_ijk.x * cells_per_subdomain.x + x_idx),
             (int)(subdomain_ijk.y * cells_per_subdomain.y + y_idx),
             (int)(subdomain_ijk.z * cells_per_subdomain.z + z_idx));

  // Check if global point exists
  // int3 global_point = parameters
  // 	.global_marching_cubes_grid
  // 	.get_point(global_point_ijk)
  // 	.expect("point has to be part of the global mc grid");
  if ( // If NOT exists return (SHOULD THROW ERROR OR STOP, e.g. .expect("point
       // has to be part of the global mc grid"))
			global_point_ijk.x >= global_marching_cubes_grid__np.x || 
			global_point_ijk.y >= global_marching_cubes_grid__np.y ||
			global_point_ijk.z >= global_marching_cubes_grid__np.z ||
			global_point_ijk.x < 0 ||
			global_point_ijk.y < 0 || 
			global_point_ijk.z < 0) {
      // !((global_point_ijk.x < global_marching_cubes_grid__np.x &&
      //    global_point_ijk.y < global_marching_cubes_grid__np.y &&
      //    global_point_ijk.z < global_marching_cubes_grid__np.z) &&
      //   (global_point_ijk.x >= 0 && global_point_ijk.y >= 0 &&
      //    global_point_ijk.z >= 0)))
				  { // Use error out bool?
    return;
  }
  // else

  // float3 point_coordinates = parameters
  // 	.global_marching_cubes_grid
  // 	.point_coordinates(&global_point);

  float3 point_coordinates =
      global_marching_cubes_grid__aabb_min +
      (float3)((float)(global_point_ijk.x * global_marching_cubes_grid__cell_size),
               (float)(global_point_ijk.y * global_marching_cubes_grid__cell_size),
               (float)(global_point_ijk.z * global_marching_cubes_grid__cell_size));

  // dx and following computations

  // matrix sum (float3)
  float3 dx = p_i - point_coordinates;

  // norm squared
	//TODO remove dotc().simdreal() and code it in CL code
  float dx_norm_sq = dx.dotc(&dx).simd_real()

      // if dx_norm_sq < squared_support_with_margin (parameters[0])
		if (dx_norm_sq < parameters[0]) {
			// Particle rest mass (parameters[1]) divided by Particle density
			float v_i = parameters[1] / rho_i;

			float r = sqrt(dx_norm_sq);
			float q = (r + r) / parameters[2];
			float cube_q = 0.0;
			if (q < 1.0) {
				cube_q = (3.0 / (2.0 * CL_M_PI)) * ((2.0 / 3.0) - q * q + 0.5 * q * q * q);
			} else if (q < 2.0) {
				float x = 2.0 - q;
				cube_q = (1.0 / (4.0 * CL_M_PI)) * x * x * x;
			}
			float w_ij = parameters[3] * cube_q;

			float interpolated_value = v_i * w_ij;

			/// FUNCTIE HIERONDER IS BELANGRIJK VOOR 3D in 1D ARRAY
			// Kijk of dit ergens ander moet worden gebruikt om buffer of size te bouwen

			// Flattens local point using n_points_per_dim of MC grid
			int flat_point_idx = x_idx * np.x * np.z + y_idx * np.z * z_idx;

			// KAN MISSCHIEN GEIGNORED WORDEN
			// int flat_point_idx = flat_point_idx.to_usize().unwrap()


			// Might need to be read/writen atomically
			levelset_grid[flat_point_idx] += interpolated_value;
  	};
}
