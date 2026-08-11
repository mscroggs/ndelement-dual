//! Dual spaces
mod bc;
mod dual0;
mod dual1;
mod representation;
pub use bc::coefficients as bc_coefficients;
pub use dual0::coefficients as dual0_coefficients;
pub use dual1::coefficients as dual1_coefficients;
pub use representation::coefficients as barycentric_representation_coefficients;

use crate::RefinedMesh;
use ndelement::types::ReferenceCellType;
use ndfunctionspace::traits::FunctionSpace;
use ndmesh::{traits::Mesh, types::Scalar};
use std::collections::HashMap;

/// A dual space
pub struct DualSpace<
    'a,
    TMesh: Scalar,
    T: Scalar,
    G: Mesh<T = TMesh, EntityDescriptor = ReferenceCellType>,
    FineG: Mesh<T = TMesh, EntityDescriptor = ReferenceCellType>,
    Space: FunctionSpace<EntityDescriptor = ReferenceCellType, Mesh = FineG>,
> {
    mesh: &'a RefinedMesh<'a, TMesh, G, FineG>,
    fine_space: &'a Space,
    coefficients: Vec<HashMap<usize, T>>,
}

impl<
    'a,
    TMesh: Scalar,
    T: Scalar,
    G: Mesh<T = TMesh, EntityDescriptor = ReferenceCellType>,
    FineG: Mesh<T = TMesh, EntityDescriptor = ReferenceCellType>,
    Space: FunctionSpace<EntityDescriptor = ReferenceCellType, Mesh = FineG>,
> DualSpace<'a, TMesh, T, G, FineG, Space>
{
    /// Create new
    pub fn new(
        mesh: &'a RefinedMesh<'a, TMesh, G, FineG>,
        fine_space: &'a Space,
        coefficients: Vec<HashMap<usize, T>>,
    ) -> Self {
        Self {
            mesh,
            fine_space,
            coefficients,
        }
    }

    /// Mesh
    pub fn mesh(&self) -> &'a RefinedMesh<'a, TMesh, G, FineG> {
        self.mesh
    }

    /// Fine space
    pub fn fine_space(&self) -> &Space {
        self.fine_space
    }

    /// Coefficients
    pub fn coefficients(&self) -> &[HashMap<usize, T>] {
        &self.coefficients
    }

    /// Number of DOFs
    pub fn dim(&self) -> usize {
        self.coefficients.len()
    }
}
