//! Dual spaces
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
    F: ElementFamily<CellType = ReferenceCellType>,
> {
    grid: &'a RefinedGrid<'a, TReal, G>,
    fine_space:
        &'a FunctionSpace<'a, SingleElementGrid<TReal, CiarletElement<TReal, IdentityMap>>, F>,
    coefficients: Vec<HashMap<usize, T>>,
}

impl<
    'a,
    TReal: RealScalar,
    T: RlstScalar<Real = TReal>,
    G: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    F: ElementFamily<CellType = ReferenceCellType>,
> DualSpace<'a, TReal, T, G, F>
{
    /// Create new
    pub fn new(
        grid: &'a RefinedGrid<'a, TReal, G>,
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
    pub fn grid(&self) -> &'a RefinedGrid<'a, TReal, G> {
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
    pub fn coefficients(&self, dof: usize) -> &HashMap<usize, T> {
        &self.coefficients[dof]
    }

    /// Number of DOFs
    pub fn dim(&self) -> usize {
        self.coefficients.len()
    }
}
