//! Refined grid
use itertools::izip;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndgrid::{
    SingleElementGrid, SingleElementGridBuilder,
    traits::{Builder, Entity, Geometry, Grid, Point, Topology},
    types::Scalar,
};
use std::collections::HashMap;

/// A grid and its barcentric refinement
pub struct RefinedGrid<
    'a,
    T: Scalar,
    G: Grid<T = T, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = T, EntityDescriptor = ReferenceCellType>,
> {
    grid: &'a G,
    bgrid: FineG,
    child_map: Vec<Vec<usize>>,
    parent_map: Vec<(usize, usize)>,
    fine_vertices: HashMap<ReferenceCellType, Vec<usize>>,
    coarse_vertices: Vec<Option<usize>>,
}

impl<
    'a,
    T: Scalar,
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

    /// Index of vertex in fine grid that is at the midpoint of an entity
    pub fn fine_vertex(&self, entity_type: ReferenceCellType, entity_index: usize) -> usize {
        self.fine_vertices[&entity_type][entity_index]
    }

    /// Index of vertex in coarse grid that coincides with fine grid vertex,
    /// or None if there is no such vertex
    pub fn coarse_vertex(&self, fine_vertex_index: usize) -> Option<usize> {
        self.coarse_vertices[fine_vertex_index]
    }
}

impl<'a, T: Scalar, G: Grid<T = T, EntityDescriptor = ReferenceCellType>>
    RefinedGrid<'a, T, G, SingleElementGrid<T, CiarletElement<T, IdentityMap, T>>>
{
    /// Barycentrically refine a grid
    pub fn new(grid: &'a G) -> Self {
        if grid.topology_dim() != 2 {
            panic!(
                "Barycentric refinement only implemented for grids with topological dimension 2."
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

        let mut fine_vertices = HashMap::new();
        for d in 0..=grid.topology_dim() {
            for etype in grid.entity_types(d) {
                fine_vertices.insert(*etype, vec![0; grid.entity_count(*etype)]);
            }
        }
        let mut coarse_vertices = vec![];

        let mut vertex_i = 0;
        let mut p = vec![T::zero(); grid.geometry_dim()];
        for v in grid.entity_iter(ReferenceCellType::Point) {
            v.geometry().points().next().unwrap().coords(&mut p);
            b.add_point(vertex_i, &p);
            fine_vertices.get_mut(&ReferenceCellType::Point).unwrap()[v.local_index()] = vertex_i;
            coarse_vertices.push(Some(v.local_index()));
            vertex_i += 1;
        }
        let mut q = vec![T::zero(); grid.geometry_dim()];
        let mut r = vec![T::zero(); grid.geometry_dim()];
        for e in grid.entity_iter(ReferenceCellType::Interval) {
            let g = e.geometry();
            let mut pts = g.points();
            pts.next().unwrap().coords(&mut p);
            pts.next().unwrap().coords(&mut q);
            for (ri, pi, qi) in izip!(&mut r, &p, &q) {
                *ri = (*pi + *qi) / T::from(2).unwrap();
            }
            b.add_point(vertex_i, &r);
            fine_vertices.get_mut(&ReferenceCellType::Interval).unwrap()[e.local_index()] = vertex_i;
            coarse_vertices.push(None);
            vertex_i += 1;
        }

        let mut s = vec![T::zero(); grid.geometry_dim()];
        for (fi, f) in grid.entity_iter(ReferenceCellType::Triangle).enumerate() {
            let g = f.geometry();
            let mut pts = g.points();
            pts.next().unwrap().coords(&mut p);
            pts.next().unwrap().coords(&mut q);
            pts.next().unwrap().coords(&mut r);
            for (si, pi, qi, ri) in izip!(&mut s, &p, &q, &r) {
                *si = (*pi + *qi + *ri) / T::from(3).unwrap();
            }
            b.add_point(vertex_i, &s);
            fine_vertices.get_mut(&ReferenceCellType::Triangle).unwrap()[f.local_index()] = vertex_i;
            coarse_vertices.push(None);

            let t = f.topology();
            let vertices = t
                .sub_entity_iter(ReferenceCellType::Point)
                .collect::<Vec<_>>();
            let edges = t
                .sub_entity_iter(ReferenceCellType::Interval)
                .collect::<Vec<_>>();
            b.add_cell(6 * fi, &[vertices[0], nv + edges[2], vertex_i]);
            b.add_cell(6 * fi + 1, &[nv + edges[2], vertices[1], vertex_i]);
            b.add_cell(6 * fi + 2, &[vertices[1], nv + edges[0], vertex_i]);
            b.add_cell(6 * fi + 3, &[nv + edges[0], vertices[2], vertex_i]);
            b.add_cell(6 * fi + 4, &[vertices[2], nv + edges[1], vertex_i]);
            b.add_cell(6 * fi + 5, &[nv + edges[1], vertices[0], vertex_i]);
            child_map[f.local_index()] = (0..6).map(|i| 6 * fi + i).collect::<Vec<_>>();
            for i in 0..6 {
                parent_map.push((f.local_index(), i));
            }
            vertex_i += 1;
        }

        for (fi, f) in grid
            .entity_iter(ReferenceCellType::Quadrilateral)
            .enumerate()
        {
            let g = f.geometry();
            let mut pts = g.points();
            pts.next().unwrap().coords(&mut p);
            pts.next();
            pts.next();
            pts.next().unwrap().coords(&mut q);
            for (si, pi, qi) in izip!(&mut s, &p, &q) {
                *si = (*pi + *qi) / T::from(2).unwrap();
            }
            b.add_point(vertex_i, &s);
            fine_vertices.get_mut(&ReferenceCellType::Quadrilateral).unwrap()[f.local_index()] = vertex_i;
            coarse_vertices.push(None);

            let t = f.topology();
            let vertices = t
                .sub_entity_iter(ReferenceCellType::Point)
                .collect::<Vec<_>>();
            let edges = t
                .sub_entity_iter(ReferenceCellType::Interval)
                .collect::<Vec<_>>();

            b.add_cell(8 * fi, &[vertices[0], nv + edges[0], vertex_i]);
            b.add_cell(8 * fi + 1, &[nv + edges[0], vertices[1], vertex_i]);
            b.add_cell(8 * fi + 2, &[vertices[1], nv + edges[2], vertex_i]);
            b.add_cell(8 * fi + 3, &[nv + edges[2], vertices[3], vertex_i]);
            b.add_cell(8 * fi + 4, &[vertices[3], nv + edges[3], vertex_i]);
            b.add_cell(8 * fi + 5, &[nv + edges[3], vertices[2], vertex_i]);
            b.add_cell(8 * fi + 6, &[vertices[2], nv + edges[1], vertex_i]);
            b.add_cell(8 * fi + 7, &[nv + edges[1], vertices[0], vertex_i]);
            child_map[f.local_index()] = (0..8).map(|i| 8 * fi + i).collect::<Vec<_>>();
            for i in 0..8 {
                parent_map.push((f.local_index(), i));
            }
            vertex_i += 1;
        }

        Self {
            grid,
            bgrid: b.create_grid(),
            child_map,
            parent_map,
            coarse_vertices,
            fine_vertices,
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
