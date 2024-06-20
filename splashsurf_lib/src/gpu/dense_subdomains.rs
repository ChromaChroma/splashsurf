use log::info;
use nalgebra::Vector3;
use crate::{Index, profile, Real};
use crate::dense_subdomains::{ParametersSubdomainGrid, subdomain_classification, Subdomains};

/// Performs classification and decomposition of particles into a regular grid of subdomains
pub(crate) fn decomposition<
    I: Index,
    R: Real,
    C: subdomain_classification::ParticleToSubdomainClassifier<I, R>,
>(
    parameters: &ParametersSubdomainGrid<I, R>,
    particles: &[Vector3<R>],
) -> Result<Subdomains<I>, anyhow::Error> {

    profile!("GPU::decomposition");
    info!("GPU::Starting classification of particles into subdomains.");


    panic!(" Well Done!")



}