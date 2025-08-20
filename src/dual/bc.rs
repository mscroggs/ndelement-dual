//! Buffa-Christiansen dual spaces

use ndgrid::types::RealScalar;
use std::collections::HashMap;
use rlst::{MatrixInverse, RlstScalar};
use ndgrid::traits::Grid;
use ndelement::{traits::{ElementFamily, Map}, ciarlet::CiarletElement, types::{ReferenceCellType, Continuity}};
use crate::{FunctionSpace, RefinedGrid};

/// Generate the coefficients that define the basis functions of a BC space
pub fn coefficients<
    'a,
    TReal: RealScalar,
    T: RlstScalar<Real = TReal> + MatrixInverse,
    G: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    M: Map,
    F: ElementFamily<T = T, CellType = ReferenceCellType, FiniteElement = CiarletElement<T, M>>,
>(
    refined_grid: &'a RefinedGrid<'a, TReal, G, FineG>,
    fine_space: &'a FunctionSpace<'a, FineG, F>,
    continuity: Continuity
) -> Vec<HashMap<usize, T>> {

    assert_eq!(refined_grid.coarse_grid().topology_dim(), 2);
    let mut coeffs = vec![];
    for edge in refined_grid.coarse_grid().entity_iter(1) {
        coeffs.push(HashMap::new());
        if continuity == Continuity::Discontinuous {
            coeffs.push(HashMap::new());
        }
    }
    coeffs
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dual::DualSpace;

    use ndgrid::shapes;
    use ndelement::{ciarlet::{NedelecFirstKindElementFamily, RaviartThomasElementFamily}, types::Continuity};

    #[test]
    fn test_bc_space() {
        let grid = shapes::regular_sphere::<f64>(1);

        let nc = NedelecFirstKindElementFamily::<f64>::new(1, Continuity::Standard);
        let nc_space = FunctionSpace::new(&grid, &nc);

        let rgrid = RefinedGrid::new(&grid);
        let rt = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
        let fine_space = FunctionSpace::new(rgrid.fine_grid(), &rt);
        let bc_space = DualSpace::new(&rgrid, &fine_space, coefficients(&rgrid, &fine_space, Continuity::Standard));
        let dbc_space = DualSpace::new(&rgrid, &fine_space, coefficients(&rgrid, &fine_space, Continuity::Discontinuous));

        assert_eq!(nc_space.dim(), bc_space.dim());
        assert_eq!(2 * nc_space.dim(), dbc_space.dim());
    }
}
