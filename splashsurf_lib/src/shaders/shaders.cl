#define CL_M_PI 3.141592653589793115998

double3 Todouble3(double* doubles) {
    return (double3)((double)doubles[0], (double)doubles[1], (double)doubles[2]);
}
kernel void reconstruct2 (
      global double* out
    , global double* p_i
    , global double* rho

    
    , global double* local_mcg_aabb_np // local_marching_cubes_grid__aabb_min
    , double lmc_cell_size
    , long cube_radius
    , global long* extents

    , double csr // compact_support_radius

    
    , global long* np
    , global long* subdomain_ijk
    , global long* cells_per_subdomain
    
    , global long* gmcg_np // global_marching_cubes_grid__np
    , global double* gmcg_aabb_np // global_marching_cubes_grid__aabb_min
    , double gmcg_cs// global_marching_cubes_grid__cell_size
    
    // [0] : squared_support_with_margin
    // [1] : particle_rest_mass
    // [2] : compact_support_radius
    // // [3] : normalization
    // // [4] : surface_threshold
    , global double* parameters
) {
    const size_t id = get_global_id(0);


    const double rho_id = rho[id];

    const double pi_x = p_i[id*3   ];
    const double pi_y = p_i[id*3 + 1];
    const double pi_z = p_i[id*3 + 2];
    const double3 p_i_double3 = (double3)(
        p_i[id*3    ],
        p_i[id*3 + 1],
        p_i[id*3 + 2]);
    


    // out[id] += (double)((pi_x - local_mcg_aabb_np[0])  / lmc_cell_size);
    long3 particle_cell = (long3)(
        (long)floor((pi_x - local_mcg_aabb_np[0]) / lmc_cell_size),
        (long)floor((pi_y - local_mcg_aabb_np[1]) / lmc_cell_size),
        (long)floor((pi_z - local_mcg_aabb_np[2]) / lmc_cell_size)
    );

    const long3 lower = (long3)(
        (long)max(particle_cell.x - cube_radius, (long)0),
        (long)max(particle_cell.y - cube_radius, (long)0),
        (long)max(particle_cell.z - cube_radius, (long)0)
    );


    const long3 upper = (long3)(
        (long)min(particle_cell.x + cube_radius + 2, extents[0]),
        (long)min(particle_cell.y + cube_radius + 2, extents[1]),
        (long)min(particle_cell.z + cube_radius + 2, extents[2])
    );
    
    
    const double normalization_sigma = 8.0 / (csr * csr * csr);
    
    
    for (int x = lower.x; x < upper.x; x++)
    {
        for (int y = lower.y; y < upper.y; y++)
        {
            for (int z = lower.z; z < upper.z; z++)
            {

                if ( x >= np[0] ||  y >= np[1] ||  z >= np[2] || x < 0 || y < 0 || z < 0) 
                { // Use error out bool?
                    return;
                }

                // Global Point
                long3 gp = (long3)((long)(subdomain_ijk[0] * cells_per_subdomain[0] + x),
                                    (long)(subdomain_ijk[1] * cells_per_subdomain[1] + y),
                                    (long)(subdomain_ijk[2] * cells_per_subdomain[1] + z));
               
                
                if ( gp.x >= gmcg_np[0] ||  gp.y >= gmcg_np[1] || gp.z >= gmcg_np[2] || gp.x < 0 || gp.y < 0 ||  gp.z < 0) 
                {   // Use error out bool?
                    return;
                }


                double3 gmcg_aabb_min = Todouble3(gmcg_aabb_np);
                double3 point_coordinates = gmcg_aabb_min +
                (double3)((double)(gp.x * gmcg_cs),
                        (double)(gp.y * gmcg_cs),
                        (double)(gp.z * gmcg_cs));

                double3 dx = p_i_double3 - point_coordinates;      

                double dx_norm_sq = dot(dx, dx);

                // out[id] += (double)dx_norm_sq;  
                
                // if dx_norm_sq < squared_support_with_margin (parameters[0])
                if (dx_norm_sq < parameters[0]) {

                    // // Particle rest mass (parameters[1]) divided by Particle density
                    double v_i = parameters[1] / rho_id;
                    double r = sqrt(dx_norm_sq);

                    // kernel.evaluate(r)
                    double q = (r + r) / parameters[2];
                    double cube_q = 0.0;
                    if (q < 1.0) {
                        cube_q = (3.0 / (2.0 * CL_M_PI)) * ((2.0 / 3.0) - q * q + 0.5 * q * q * q);
                    } else if (q < 2.0) {
                        double x = 2.0 - q;
                        cube_q = (1.0 / (4.0 * CL_M_PI)) * x * x * x;
                    }
                    double w_ij = normalization_sigma * cube_q;

                    double interpolated_value = v_i * w_ij;

                    long flat_point_idx = x * np[1] * np[2] + y * np[2] + z;
                    out[flat_point_idx] += interpolated_value;
                    
                };

            }
        }
    }

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

    // Private?
    , global double* p_i
    , double rho_i

    , global double* parameters
)
{
    const size_t x1 = get_global_id(0);
    const size_t y1 = get_global_id(1);
    const size_t z1 = get_global_id(2);
    
    const long x = x1 + lower[0];
    const long y = y1 + lower[1];
    const long z = z1 + lower[2];

    if ( x >= np[0] ||  y >= np[1] ||  z >= np[2] || x < 0 || y < 0 || z < 0) 
    { // Use error out bool?
        return;
    }

    // Global Point
    long3 gp = (long3)((long)(subdomain_ijk[0] * cells_per_subdomain[0] + x),
                        (long)(subdomain_ijk[1] * cells_per_subdomain[1] + y),
                        (long)(subdomain_ijk[2] * cells_per_subdomain[1] + z));



    if ( gp.x >= gmcg_np[0] ||  gp.y >= gmcg_np[1] || gp.z >= gmcg_np[2] ||
        gp.x < 0 || gp.y < 0 ||  gp.z < 0) 
    {   // Use error out bool?
        return;
    }

    double3 gmcg_aabb_min = Todouble3(gmcg_aabb_np);

    double3 point_coordinates = gmcg_aabb_min +
      (double3)((double)(gp.x * gmcg_cs),
               (double)(gp.y * gmcg_cs),
               (double)(gp.z * gmcg_cs));

    double3 dx = Todouble3(p_i) - point_coordinates;

    
    double dx_norm_sq = dot(dx, dx);

    // if dx_norm_sq < squared_support_with_margin (parameters[0])
    if (dx_norm_sq < parameters[0]) {

        // // Particle rest mass (parameters[1]) divided by Particle density
        double v_i = parameters[1] / rho_i;
        double r = sqrt(dx_norm_sq);

        // kernel.evaluate(r)
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

        long flat_point_idx = x * np[1] * np[2] + y * np[2] + z;
        out[flat_point_idx] += interpolated_value;
        
    };
}