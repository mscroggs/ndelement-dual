//! Dual spaces
mod representation;
pub use representation::barycentric_representation_coefficients;

use crate::RefinedGrid;
use ndelement::types::ReferenceCellType;
use ndfunctionspace::traits::FunctionSpace;
use ndgrid::{traits::Grid, types::Scalar};
use std::collections::HashMap;

/// A dual space
pub struct DualSpace<
    'a,
    TGrid: Scalar,
    T: Scalar,
    G: Grid<T = TGrid, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = TGrid, EntityDescriptor = ReferenceCellType>,
    Space: FunctionSpace<EntityDescriptor = ReferenceCellType, Grid = FineG>,
> {
    grid: &'a RefinedGrid<'a, TGrid, G, FineG>,
    fine_space: &'a Space,
    coefficients: Vec<HashMap<usize, T>>,
}

impl<
    'a,
    TGrid: Scalar,
    T: Scalar,
    G: Grid<T = TGrid, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = TGrid, EntityDescriptor = ReferenceCellType>,
    Space: FunctionSpace<EntityDescriptor = ReferenceCellType, Grid = FineG>,
> DualSpace<'a, TGrid, T, G, FineG, Space>
{
    /// Create new
    pub fn new(
        grid: &'a RefinedGrid<'a, TGrid, G, FineG>,
        fine_space: &'a Space,
        coefficients: Vec<HashMap<usize, T>>,
    ) -> Self {
        Self {
            grid,
            fine_space,
            coefficients,
        }
    }

    /// Grid
    pub fn grid(&self) -> &'a RefinedGrid<'a, TGrid, G, FineG> {
        self.grid
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
