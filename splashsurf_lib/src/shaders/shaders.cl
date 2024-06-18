#define CL_M_PI 3.141592653589793115998

double3 Todouble3(double* doubles) {
    return (double3)((double)doubles[0], (double)doubles[1], (double)doubles[2]);
}
kernel void reconstruct (
    global double* out
    , global long* lower
    , global long* np
    , global long* subdomain_ijk
    , global long* cells_per_subdomain
    , global long* gmcg_np // global_marching_cubes_grid__np
    , global double* gmcg_aabb_np // global_marching_cubes_grid__aabb_min
    , double gmcg_cs// global_marching_cubes_grid__cell_size
    , global double* p_i
    , double rho_i
    , global double* parameters
)
{
    const size_t x1 = get_global_id(0);
    const size_t y1 = get_global_id(1);
    const size_t z1 = get_global_id(2);
    
    const long x = x + lower[0];
    const long y = y + lower[1];
    const long z = z + lower[2];

    if ( x >= np[0] ||  y >= np[1] ||  z >= np[2] || x < 0 || y < 0 || z < 0) 
    { // out[2] += 1;
        return;
    }

    //gp
    long3 gp = (long3)((long)(subdomain_ijk[0] * cells_per_subdomain[0] + x),
                                   (long)(subdomain_ijk[1] * cells_per_subdomain[1] + y),
                                   (long)(subdomain_ijk[2] * cells_per_subdomain[1] + z));

    if ( gp.x >= gmcg_np[0] ||  gp.y >= gmcg_np[1] || gp.z >= gmcg_np[2] ||
        gp.x < 0 || gp.y < 0 ||  gp.z < 0) 
    { // Use error out bool?
        // out[2] += 1;
        return;
    }


    double3 gmcg_aabb_min = Todouble3(gmcg_aabb_np);
    double3 polong_coordinates = gmcg_aabb_min +
      (double3)((double)(gp.x * gmcg_cs),
               (double)(gp.y * gmcg_cs),
               (double)(gp.z * gmcg_cs));

    double3 dx = Todouble3(p_i) - polong_coordinates;

    double dx_norm_sq = dot(dx, dx);

    // if dx_norm_sq < squared_support_with_margin (parameters[0])
    if (dx_norm_sq < parameters[0]) {
        // Particle rest mass (parameters[1]) divided by Particle density
        double v_i = parameters[1] / rho_i;

        double r = sqrt(dx_norm_sq);
        double q = (r + r) / parameters[2];
        double cube_q = 0.0;
        if (q < 1.0) {
            cube_q = (3.0 / (2.0 * CL_M_PI)) * ((2.0 / 3.0) - q * q + 0.5 * q * q * q);
        } else if (q < 2.0) {
            double x = 2.0 - q;
            cube_q = (1.0 / (4.0 * CL_M_PI)) * x * x * x;
        }
        double w_ij = parameters[3] * cube_q;

        double interpolated_value = v_i * w_ij;

        /// FUNCTIE HIERONDER IS BELANGRIJK VOOR 3D in 1D ARRAY
        // Kijk of dit ergens ander moet worden gebruikt om buffer of size te bouwen

        // Flattens local point using n_points_per_dim of MC grid
        long flat_point_idx = x * np[0] * np[2] + y * np[2]* z;


        // KAN MISSCHIEN GEIGNORED WORDEN
        // int flat_point_idx = flat_point_idx.to_usize().unwrap()


        // Might need to be read/writen atomically // e.g. atomic_add()
        // out[flat_point_idx] += interpolated_value;
        
    };

    
    // int flat_point_idx = x * np[0] * np[2] + y * np[2]* z;

    // // out[0] += 2+ x + y + z;
    // out[flat_point_idx] += 2 + x + y + z;
    // // out[1] += 1;
}