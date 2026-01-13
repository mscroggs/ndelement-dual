//! ndelement-dual
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

mod assembly;
mod dual;
mod grid;

pub use assembly::{assemble_mass_matrix, assemble_mass_matrix_dual};
pub use dual::{DualSpace, barycentric_representation_coefficients, bc_coefficients};
pub use grid::{DualGrid, RefinedGrid};
