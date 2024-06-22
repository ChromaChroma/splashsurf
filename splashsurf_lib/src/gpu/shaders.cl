
kernel void fill_arr(global int *out) {
  const size_t id = get_global_id(0);
  out[id] = id;
}

kernel void compute_lower_and_upper(
    const long cube_radius, const long extents_x, const long extents_y,
    const long extents_z, const float lmc_cell_size,
    const float local_mcg_aab_min_x, const float local_mcg_aab_min_y,
    const float local_mcg_aab_min_z,
    // xyz coords of all points
    global const double *point_coords,
    // Output buffer (2x longer than point_coords)
    // Array with xyz of lower and upper bounds
    // 6 indexes (first 3 for lower-xyz, second 3 for upper-xyz)
    global long *bounds) {
  const size_t id = get_global_id(0);

  const double pi_x = point_coords[id * 3];
  const double pi_y = point_coords[id * 3 + 1];
  const double pi_z = point_coords[id * 3 + 2];

  const long3 particle_cell =
      (long3)((long)floor((pi_x - local_mcg_aab_min_x) / lmc_cell_size),
              (long)floor((pi_y - local_mcg_aab_min_y) / lmc_cell_size),
              (long)floor((pi_z - local_mcg_aab_min_z) / lmc_cell_size));

  bounds[id * 3] = (long)max(particle_cell.x - cube_radius, (long)0);
  bounds[id * 3 + 1] = (long)max(particle_cell.y - cube_radius, (long)0);
  bounds[id * 3 + 2] = (long)max(particle_cell.z - cube_radius, (long)0);
  bounds[id * 3 + 3] = (long)min(particle_cell.x + cube_radius + 2, extents_x);
  bounds[id * 3 + 4] = (long)min(particle_cell.y + cube_radius + 2, extents_y);
  bounds[id * 3 + 5] = (long)min(particle_cell.z + cube_radius + 2, extents_z);
}