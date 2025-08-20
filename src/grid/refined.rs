//! Refined grid

use itertools::izip;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndgrid::{
    SingleElementGrid, SingleElementGridBuilder,
    traits::{Builder, Entity, Geometry, Grid, Point, Topology},
    types::RealScalar,
};

/// A grid and its barcentric refinement
pub struct RefinedGrid<
    'a,
    T: RealScalar,
    G: Grid<T = T, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = T, EntityDescriptor = ReferenceCellType>,
> {
    grid: &'a G,
    bgrid: FineG,
    child_map: Vec<Vec<usize>>,
    parent_map: Vec<(usize, usize)>,
    fine_vertices: Vec<usize>,
    coarse_vertices: Vec<Option<usize>>,
}

impl<
    'a,
    T: RealScalar,
    G: Grid<T = T, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = T, EntityDescriptor = ReferenceCellType>,
> RefinedGrid<'a, T, G, FineG>
{
    /// Coarse unrefined grid
    pub fn coarse_grid(&self) -> &'a G {
        self.grid
    }

    /// Barycentrically refined grid
    pub fn fine_grid(&self) -> &FineG {
        &self.bgrid
    }

    /// Indices of cells in fine grid that make up a coarse cell
    pub fn children(&self, coarse_cell_index: usize) -> &[usize] {
        &self.child_map[coarse_cell_index]
    }

    /// Index of cell in coarse grid that contains a fine cell
    pub fn parent(&self, fine_cell_index: usize) -> (usize, usize) {
        self.parent_map[fine_cell_index]
    }

    /// Index of vertex in fine grid that coincides with coarse grid vertex
    pub fn fine_vertex(&self, coarse_vertex_index: usize) -> usize {
        self.fine_vertices[coarse_vertex_index]
    }

    /// Index of vertex in coarse grid that coincides with fine grid vertex,
    /// or None if there is no such vertex
    pub fn coarse_vertex(&self, fine_vertex_index: usize) -> Option<usize> {
        self.coarse_vertices[fine_vertex_index]
    }
}

impl<'a, T: RealScalar, G: Grid<T = T, EntityDescriptor = ReferenceCellType>>
    RefinedGrid<'a, T, G, SingleElementGrid<T, CiarletElement<T, IdentityMap>>>
{
    /// Barycentrically refine a grid
    pub fn new(grid: &'a G) -> Self {
        if grid.topology_dim() != 2 {
            panic!(
                "Barycentric refinement only implemented for grid with topological dimension 2."
            );
        }

        let nv = grid.entity_count(ReferenceCellType::Point);

        // TODO: what if element degree > 1
        let mut b = SingleElementGridBuilder::<T>::new(
            grid.geometry_dim(),
            (ReferenceCellType::Triangle, 1),
        );

        let mut child_map = vec![];
        let mut parent_map = vec![];
        for _ in 0..grid.cell_count() {
            child_map.push(vec![]);
        }

        let mut fine_vertices = vec![0; grid.entity_count(ReferenceCellType::Point)];
        let mut coarse_vertices = vec![];

        let mut vertex_i = 0;
        let mut p = vec![T::zero(); grid.geometry_dim()];
        for v in grid.entity_iter(0) {
            v.geometry().points().next().unwrap().coords(&mut p);
            b.add_point(vertex_i, &p);
            fine_vertices[v.local_index()] = vertex_i;
            coarse_vertices.push(Some(v.local_index()));
            vertex_i += 1;
        }
        let mut q = vec![T::zero(); grid.geometry_dim()];
        let mut r = vec![T::zero(); grid.geometry_dim()];
        for e in grid.entity_iter(1) {
            let g = e.geometry();
            let mut pts = g.points();
            pts.next().unwrap().coords(&mut p);
            pts.next().unwrap().coords(&mut q);
            for (ri, pi, qi) in izip!(&mut r, &p, &q) {
                *ri = (*pi + *qi) / T::from(2).unwrap();
            }
            b.add_point(vertex_i, &r);
            coarse_vertices.push(None);
            vertex_i += 1;
        }

        let mut cell_i = 0;
        let mut s = vec![T::zero(); grid.geometry_dim()];
        let ct = grid.cell_types();
        if ct.contains(&ReferenceCellType::Triangle) {
            for cell in grid.cell_iter_by_type(ReferenceCellType::Triangle) {
                let g = cell.geometry();
                let mut pts = g.points();
                pts.next().unwrap().coords(&mut p);
                pts.next().unwrap().coords(&mut q);
                pts.next().unwrap().coords(&mut r);
                for (si, pi, qi, ri) in izip!(&mut s, &p, &q, &r) {
                    *si = (*pi + *qi + *ri) / T::from(3).unwrap();
                }
                b.add_point(vertex_i, &s);
                coarse_vertices.push(None);

                let t = cell.topology();
                let vertices = t.sub_entity_iter(0).collect::<Vec<_>>();
                let edges = t.sub_entity_iter(1).collect::<Vec<_>>();

                b.add_cell(cell_i, &[vertices[0], nv + edges[2], vertex_i]);
                b.add_cell(cell_i + 1, &[nv + edges[2], vertices[1], vertex_i]);
                b.add_cell(cell_i + 2, &[vertices[1], nv + edges[0], vertex_i]);
                b.add_cell(cell_i + 3, &[nv + edges[0], vertices[2], vertex_i]);
                b.add_cell(cell_i + 4, &[vertices[2], nv + edges[1], vertex_i]);
                b.add_cell(cell_i + 5, &[nv + edges[1], vertices[0], vertex_i]);
                child_map[cell.local_index()] = (0..6).map(|i| cell_i + i).collect::<Vec<_>>();
                for i in 0..6 {
                    parent_map.push((cell.local_index(), vertex_i + i));
                }
                vertex_i += 1;
                cell_i += 6
            }
        }
        if ct.contains(&ReferenceCellType::Quadrilateral) {
            for cell in grid.cell_iter_by_type(ReferenceCellType::Quadrilateral) {
                let g = cell.geometry();
                let mut pts = g.points();
                pts.next().unwrap().coords(&mut p);
                pts.next();
                pts.next();
                pts.next().unwrap().coords(&mut q);
                for (si, pi, qi) in izip!(&mut s, &p, &q) {
                    *si = (*pi + *qi) / T::from(2).unwrap();
                }
                b.add_point(vertex_i, &s);
                coarse_vertices.push(None);

                let t = cell.topology();
                let vertices = t.sub_entity_iter(0).collect::<Vec<_>>();
                let edges = t.sub_entity_iter(1).collect::<Vec<_>>();

                b.add_cell(cell_i, &[vertices[0], nv + edges[0], vertex_i]);
                b.add_cell(cell_i + 1, &[nv + edges[0], vertices[1], vertex_i]);
                b.add_cell(cell_i + 2, &[vertices[1], nv + edges[2], vertex_i]);
                b.add_cell(cell_i + 3, &[nv + edges[2], vertices[3], vertex_i]);
                b.add_cell(cell_i + 4, &[vertices[3], nv + edges[3], vertex_i]);
                b.add_cell(cell_i + 5, &[nv + edges[3], vertices[2], vertex_i]);
                b.add_cell(cell_i + 6, &[vertices[2], nv + edges[1], vertex_i]);
                b.add_cell(cell_i + 7, &[nv + edges[1], vertices[0], vertex_i]);
                child_map[cell.local_index()] = (0..8).map(|i| cell_i + i).collect::<Vec<_>>();
                for i in 0..8 {
                    parent_map.push((cell.local_index(), cell_i + i));
                }
                vertex_i += 1;
                cell_i += 8;
            }
        }

        Self {
            grid,
            bgrid: b.create_grid(),
            child_map,
            parent_map,
            fine_vertices,
            coarse_vertices,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndgrid::shapes::unit_cube_boundary;

    #[test]
    fn test_refine_triangle() {
        let grid = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle);
        let bgrid = RefinedGrid::new(&grid);
        assert_eq!(grid.cell_count(), bgrid.coarse_grid().cell_count());
        assert_eq!(grid.cell_count() * 6, bgrid.fine_grid().cell_count());
    }

    #[test]
    fn test_refine_quadrilateral() {
        let grid = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Quadrilateral);
        let bgrid = RefinedGrid::new(&grid);
        assert_eq!(grid.cell_count(), bgrid.coarse_grid().cell_count());
        assert_eq!(grid.cell_count() * 8, bgrid.fine_grid().cell_count());
    }
}

