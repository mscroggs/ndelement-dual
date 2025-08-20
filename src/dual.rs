//! Dual spaces
mod bc;
mod representation;
pub use representation::coefficients as barycentric_representation_coefficients;
pub use bc::coefficients as bc_coefficients;

use crate::{FunctionSpace, RefinedGrid};
use ndelement::{
    ciarlet::CiarletElement, map::IdentityMap, traits::ElementFamily, types::ReferenceCellType,
};
use ndgrid::{SingleElementGrid, traits::Grid, types::RealScalar};
use rlst::RlstScalar;
use std::collections::HashMap;

/// A dual space
pub struct DualSpace<
    'a,
    TReal: RealScalar,
    T: RlstScalar<Real = TReal>,
    G: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    F: ElementFamily<CellType = ReferenceCellType>,
> {
    grid: &'a RefinedGrid<'a, TReal, G, FineG>,
    fine_space:
        &'a FunctionSpace<'a, SingleElementGrid<TReal, CiarletElement<TReal, IdentityMap>>, F>,
    coefficients: Vec<HashMap<usize, T>>,
}

impl<
    'a,
    TReal: RealScalar,
    T: RlstScalar<Real = TReal>,
    G: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    F: ElementFamily<CellType = ReferenceCellType>,
> DualSpace<'a, TReal, T, G, FineG, F>
{
    /// Create new
    pub fn new(
        grid: &'a RefinedGrid<'a, TReal, G, FineG>,
        fine_space: &'a FunctionSpace<
            'a,
            SingleElementGrid<TReal, CiarletElement<TReal, IdentityMap>>,
            F,
        >,
        coefficients: Vec<HashMap<usize, T>>,
    ) -> Self {
        Self {
            grid,
            fine_space,
            coefficients,
        }
    }

    /// Grid
    pub fn grid(&self) -> &'a RefinedGrid<'a, TReal, G, FineG> {
        self.grid
    }

    /// Fine space
    pub fn fine_space(
        &self,
    ) -> &'a FunctionSpace<'a, SingleElementGrid<TReal, CiarletElement<TReal, IdentityMap>>, F>
    {
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
