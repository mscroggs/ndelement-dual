//! Refined grid

use super::RefinedGrid;
use ndelement::types::ReferenceCellType;
use ndgrid::{
    traits::{Entity, Grid, Topology},
    types::RealScalar,
};

/// A barycentric dual grid
pub struct DualGrid<
    'a,
    T: RealScalar,
    G: Grid<T = T, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = T, EntityDescriptor = ReferenceCellType>,
> {
    grids: &'a RefinedGrid<'a, T, G, FineG>,
    subcells: Vec<Vec<usize>>,
}

impl<
    'a,
    T: RealScalar,
    G: Grid<T = T, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = T, EntityDescriptor = ReferenceCellType>,
> DualGrid<'a, T, G, FineG> {
    /// Create new dual grid
    pub fn new(grids: &'a RefinedGrid<'a, T, G, FineG>) -> Self {
        let mut subcells = vec![];
        for _ in grids.coarse_grid().entity_iter(0) {
            subcells.push(vec![]);
        }
        for fine_cell in grids.fine_grid().cell_iter() {
            for v in fine_cell.topology().sub_entity_iter(0) {
                if let Some(i) = grids.coarse_vertex(v) {
                   subcells[i].push(fine_cell.local_index());
                }
            }
        }

        Self {
            grids,
            subcells,
        }
    }

    /// Coarse and fine grids used to define this dual grid
    pub fn grids(&self) -> &'a RefinedGrid<'a, T, G, FineG> {
        self.grids
    }

    /// Number of cells
    pub fn cell_count(&self) -> usize {
        self.subcells.len()
    }

    /// Sub cells of a dual cell
    pub fn subcells(&self, index: usize) -> &[usize] {
        &self.subcells[index]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndgrid::shapes::unit_cube_boundary;

    #[test]
    fn test_dual_triangle() {
        let grid = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle);
        let bgrid = RefinedGrid::new(&grid);
        let dual_grid = DualGrid::new(&bgrid);
        assert_eq!(dual_grid.cell_count(), bgrid.coarse_grid().entity_count(ReferenceCellType::Point));
        assert_eq!((0..dual_grid.cell_count()).map(|i| dual_grid.subcells(i).len()).sum::<usize>(), bgrid.fine_grid().entity_count(ReferenceCellType::Triangle));
    }

    #[test]
    fn test_refine_quadrilateral() {
        let grid = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Quadrilateral);
        let bgrid = RefinedGrid::new(&grid);
        let dual_grid = DualGrid::new(&bgrid);
        assert_eq!(dual_grid.cell_count(), bgrid.coarse_grid().entity_count(ReferenceCellType::Point));
        assert_eq!((0..dual_grid.cell_count()).map(|i| dual_grid.subcells(i).len()).sum::<usize>(), bgrid.fine_grid().entity_count(ReferenceCellType::Triangle));
    }
}

