use std::sync::{Arc, Mutex};
use log::info;
use nalgebra::Vector3;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use opencl3::kernel::Kernel;
use opencl3::program::{CL_STD_2_0, Program};
use crate::{Index, OpenCLData, profile, Real};
use crate::dense_subdomains::{ParametersSubdomainGrid, subdomain_classification, Subdomains};
use crate::gpu::kernel::init_kernels;

/// Performs classification and decomposition of particles into a regular grid of subdomains
pub(crate) fn decomposition<
    I: Index,
    R: Real,
    C: subdomain_classification::ParticleToSubdomainClassifier<I, R>,
>(parameters: &ParametersSubdomainGrid<I, R>,
  particles: &[Vector3<R>],
  ocl_data: Arc<Mutex<OpenCLData>>,
) -> Result<Subdomains<I>, anyhow::Error> {
    profile!("GPU::decomposition");
    info!("GPU::Starting classification of particles into subdomains.");


    // init_kernels().expect(" Gpu failed initializing");
    // panic!(" Well Done!")

    Ok(Subdomains {
        flat_subdomain_indices: Vec::new(),
        per_subdomain_particles: Vec::new(),
    })
}
