//! ndelement-dual
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

mod assembly;
mod dual;
mod mesh;

pub use assembly::{assemble_mass_matrix, assemble_mass_matrix_dual};
pub use dual::{DualSpace, barycentric_representation_coefficients, bc_coefficients, dual0_coefficients};
pub use mesh::{DualMesh, RefinedMesh};
