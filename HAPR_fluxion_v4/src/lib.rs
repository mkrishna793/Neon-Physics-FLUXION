//! FLUXION v4 Core Library.
//!
//! Exports public modules for Python bindings and CLI.

pub mod algorithm;
pub mod config;
pub mod data;
pub mod gpu;
pub mod parser;

pub use config::Config;
pub use data::circuit::Circuit;
pub use data::placement::Placement;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn fluxion(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add Python bindings here in the future
    Ok(())
}
