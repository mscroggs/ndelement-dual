//! ndelement-dual
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

mod assembly;
mod bc;
mod dual;
mod grid;

pub use assembly::{FunctionSpace, assemble_mass_matrix};
pub use grid::bary_refine;
