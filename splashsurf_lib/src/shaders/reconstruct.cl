#define CL_M_PI 3.141592653589793115998

double3 Todouble3(double* doubles) {
    return  (double3)((double)doubles[0], (double)doubles[1], (double)doubles[2]);
}

int3 ToInt3(int* doubles) {
    return  (int3)((int)doubles[0], (int)doubles[1], (int)doubles[2]);
}

__kernel void reconstruct(
    __global float* levelset_grid
    // , global int* np

    // , int delta_y
    // , int delta_z

    // , global int* subdomain_ijk
    // , global int* cells_per_subdomain

    // // n_points_per_dim of global_marching_cubes_grid
    // , global int* global_marching_cubes_grid__np
    // // Min Vector3 of aabb of global MC grid
    // , global double* global_marching_cubes_grid__aabb_min
    // , double global_marching_cubes_grid__cell_size
    
    // , global double* p_i
    // , double rho_i

    // // [0] : squared_support_with_margin
    // // [1] : particle_rest_mass
    // // [2] : compact_support_radius
    // // [3] : normalization
    // // [4] : surface_threshold
    // ,  global double* parameters
)
{
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const size_t z = get_global_id(2);

    levelset_grid[x + y] = 1.0;
    // Gets point on MC grid
    if ( x >= np[0] ||  y >= np[1] ||  z >= np[2] || x < 0 || y < 0 || z < 0) {
        levelset_grid[1] = 0.3;
        return;
    }

    int3 global_point_ijk = (int3)((int)(subdomain_ijk[0] * cells_per_subdomain[0] + x),
                                   (int)(subdomain_ijk[1] * cells_per_subdomain[1] + y),
                                   (int)(subdomain_ijk[2] * cells_per_subdomain[1] + z));


    if ( global_point_ijk.x >= global_marching_cubes_grid__np[0] || 
            global_point_ijk.y >= global_marching_cubes_grid__np[1] ||
            global_point_ijk.z >= global_marching_cubes_grid__np[2] ||
            global_point_ijk.x < 0 ||
            global_point_ijk.y < 0 || 
            global_point_ijk.z < 0) 
    { // Use error out bool?
        return;
    }
    
    double3 g_mcgrid_aabb_min = Todouble3(global_marching_cubes_grid__aabb_min);
        // (double3)((double)global_marching_cubes_grid__aabb_min[0],
        //         (double)global_marching_cubes_grid__aabb_min[1],
        //         (double)global_marching_cubes_grid__aabb_min[2]);

    double3 point_coordinates =
      g_mcgrid_aabb_min +
      (double3)((double)(global_point_ijk.x * global_marching_cubes_grid__cell_size),
               (double)(global_point_ijk.y * global_marching_cubes_grid__cell_size),
               (double)(global_point_ijk.z * global_marching_cubes_grid__cell_size));



    double3 dx = Todouble3(p_i) - point_coordinates;

    // // // norm squared
    // // //TODO remove dotc().simdreal() and code it in CL code
    // // double dx_norm_sq = dx.dotc(&dx)

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
        int flat_point_idx = x * np[0] * np[2] + y * np[2]* z;


        // KAN MISSCHIEN GEIGNORED WORDEN
        // int flat_point_idx = flat_point_idx.to_usize().unwrap()


        // Might need to be read/writen atomically // e.g. atomic_add()
        levelset_grid[flat_point_idx] += interpolated_value;
        
    };

    
}
