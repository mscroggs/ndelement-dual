//! Buffa-Christiansen dual spaces

use crate::RefinedGrid;
use ndelement::{
    ciarlet::CiarletElement,
    traits::Map,
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::traits::FunctionSpace;
use ndgrid::traits::Grid;
use ndgrid::types::Scalar;
use std::collections::HashMap;

/// Generate the coefficients that define the basis functions of a BC space
pub fn coefficients<
    'a,
    TGeo: Scalar,
    T: Scalar,
    G: Grid<T = TGeo, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = TGeo, EntityDescriptor = ReferenceCellType>,
    M: Map,
>(
    refined_grid: &'a RefinedGrid<'a, TGeo, G, FineG>,
    fine_space: &impl FunctionSpace<
        EntityDescriptor = ReferenceCellType,
        Grid = FineG,
        FiniteElement = CiarletElement<T, M, TGeo>,
    >,
    continuity: Continuity,
) -> Vec<HashMap<usize, T>> {
    assert_eq!(refined_grid.coarse_grid().topology_dim(), 2);
    let mut coeffs = vec![];
    for edge in refined_grid
        .coarse_grid()
        .entity_iter(ReferenceCellType::Interval)
    {
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
    use ndelement::{
        ciarlet::{NedelecFirstKindElementFamily, RaviartThomasElementFamily},
        types::Continuity,
    };
    use ndfunctionspace::FunctionSpaceImpl;
    use ndgrid::shapes;

    #[test]
    fn test_bc_space() {
        let grid = shapes::regular_sphere::<f64>(1);

        let nc = NedelecFirstKindElementFamily::<f64>::new(1, Continuity::Standard);
        let nc_space = FunctionSpaceImpl::new(&grid, &nc);

        let rgrid = RefinedGrid::new(&grid);
        let rt = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
        let fine_space = FunctionSpaceImpl::new(rgrid.fine_grid(), &rt);
        let bc_space = DualSpace::new(
            &rgrid,
            &fine_space,
            coefficients(&rgrid, &fine_space, Continuity::Standard),
        );
        let dbc_space = DualSpace::new(
            &rgrid,
            &fine_space,
            coefficients(&rgrid, &fine_space, Continuity::Discontinuous),
        );

        assert_eq!(nc_space.local_size(), bc_space.dim());
        assert_eq!(2 * nc_space.local_size(), dbc_space.dim());
    }
}
