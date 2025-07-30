//! ndelement-dual
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

mod assembly;
mod dual;
mod grid;

pub use assembly::{FunctionSpace, assemble_mass_matrix};
pub use dual::DualSpace;
pub use grid::RefinedGrid;
