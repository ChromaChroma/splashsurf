use std::sync::{Arc, Mutex};
use anyhow::Context;
use nalgebra::Vector3;
use crate::{gpu, Index, OpenCLData, Parameters, profile, Real, SurfaceReconstruction};
use crate::dense_subdomains::initialize_parameters;
use crate::dense_subdomains::subdomain_classification::GhostMarginClassifier;

pub(crate) fn reconstruct_surface_subdomain_grid_gpu<'a, I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
    output_surface: &'a mut SurfaceReconstruction<I, R>,
    ocl_data: Arc<Mutex<OpenCLData>>
) -> Result<(), anyhow::Error> {

    profile!("GPU::surface reconstruction subdomain-grid");

    let internal_parameters = initialize_parameters(parameters, &particle_positions, output_surface)?;
    output_surface.grid = internal_parameters
        .global_marching_cubes_grid()
        .context("failed to convert global marching cubes grid")?;


    let subdomains =
        gpu::dense_subdomains::decomposition::<I, R, GhostMarginClassifier<I>>(&internal_parameters, &particle_positions, ocl_data)?;



    Ok(())
}