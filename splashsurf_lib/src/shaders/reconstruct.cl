#define CL_M_PI 3.141592653589793115998

float3 ToFloat3(float* floats) {
    return  (float3)((float)floats[0], (float)floats[1], (float)floats[2]);
}

int3 ToInt3(int* floats) {
    return  (int3)((int)floats[0], (int)floats[1], (int)floats[2]);
}

__kernel void reconstruct(
    __global int* levelset_grid
    , global int* np

    , int delta_y
    , int delta_z

    , global int* subdomain_ijk
    , global int* cells_per_subdomain

    // n_points_per_dim of global_marching_cubes_grid
    , global int* global_marching_cubes_grid__np
    // Min Vector3 of aabb of global MC grid
    , global float* global_marching_cubes_grid__aabb_min
    , float global_marching_cubes_grid__cell_size
    
    , global float* p_i
    , float rho_i

    // [0] : squared_support_with_margin
    // [1] : particle_rest_mass
    // [2] : compact_support_radius
    // [3] : normalization
    // [4] : surface_threshold
    ,  global float* parameters
)
{
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const size_t z = get_global_id(1);

    // Gets point on MC grid
    if ( x >= np[0] ||  y >= np[1] ||  z >= np[2] || x < 0 || y < 0 || z < 0) {
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
    
    float3 g_mcgrid_aabb_min = ToFloat3(global_marching_cubes_grid__aabb_min);
        // (float3)((float)global_marching_cubes_grid__aabb_min[0],
        //         (float)global_marching_cubes_grid__aabb_min[1],
        //         (float)global_marching_cubes_grid__aabb_min[2]);

    float3 point_coordinates =
      g_mcgrid_aabb_min +
      (float3)((float)(global_point_ijk.x * global_marching_cubes_grid__cell_size),
               (float)(global_point_ijk.y * global_marching_cubes_grid__cell_size),
               (float)(global_point_ijk.z * global_marching_cubes_grid__cell_size));



    float3 dx = ToFloat3(p_i) - point_coordinates;

    // // norm squared
    // //TODO remove dotc().simdreal() and code it in CL code
    // float dx_norm_sq = dx.dotc(&dx).simd_real()
    float dx_norm_sq = 1.0;

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
        int flat_point_idx = x * np[0] * np[2] + y * np[2]* z;


        // KAN MISSCHIEN GEIGNORED WORDEN
        // int flat_point_idx = flat_point_idx.to_usize().unwrap()


        // Might need to be read/writen atomically // e.g. atomic_add()
        levelset_grid[flat_point_idx] += interpolated_value;
        
    };

    // levelset_grid[x + delta_y * (y + delta_z * z)] = x + y + z;
}
