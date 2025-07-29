use itertools::izip;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndgrid::{
    SingleElementGrid, SingleElementGridBuilder,
    traits::{Builder, Entity, Geometry, Grid, Point, Topology},
    types::RealScalar,
};

/// Barycentrically refine a grid
pub fn bary_refine<T: RealScalar>(
    grid: &impl Grid<T = T, EntityDescriptor = ReferenceCellType>,
) -> SingleElementGrid<T, CiarletElement<T, IdentityMap>> {
    if grid.topology_dim() != 2 {
        panic!("Barycentric refinement only implemented for grid with topological dimension 2.");
    }

    let nv = grid.entity_count(ReferenceCellType::Point);

    // TODO: what if element degree > 1
    let mut b =
        SingleElementGridBuilder::<T>::new(grid.geometry_dim(), (ReferenceCellType::Triangle, 1));

    let mut i = 0;
    let mut p = vec![T::zero(); grid.geometry_dim()];
    for v in grid.entity_iter(0) {
        v.geometry().points().next().unwrap().coords(&mut p);
        b.add_point(i, &p);
        i += 1;
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
        b.add_point(i, &r);
        i += 1;
    }

    let mut s = vec![T::zero(); grid.geometry_dim()];
    for (fi, f) in grid.entity_iter(2).enumerate() {
        let g = f.geometry();
        let mut pts = g.points();
        match f.entity_type() {
            ReferenceCellType::Triangle => {
                pts.next().unwrap().coords(&mut p);
                pts.next().unwrap().coords(&mut q);
                pts.next().unwrap().coords(&mut r);
                for (si, pi, qi, ri) in izip!(&mut s, &p, &q, &r) {
                    *si = (*pi + *qi + *ri) / T::from(3).unwrap();
                }
            }
            ReferenceCellType::Quadrilateral => {
                pts.next().unwrap().coords(&mut p);
                pts.next();
                pts.next();
                pts.next().unwrap().coords(&mut q);
                for (si, pi, qi) in izip!(&mut s, &p, &q) {
                    *si = (*pi + *qi) / T::from(2).unwrap();
                }
            }
            _ => {
                panic!("Unsupported cell type: {:?}", f.entity_type());
            }
        }
        b.add_point(i, &s);

        let t = f.topology();
        let vertices = t.sub_entity_iter(0).collect::<Vec<_>>();
        let edges = t.sub_entity_iter(1).collect::<Vec<_>>();

        match f.entity_type() {
            ReferenceCellType::Triangle => {
                b.add_cell(6 * fi, &[vertices[0], nv + edges[2], i]);
                b.add_cell(6 * fi + 1, &[nv + edges[2], vertices[1], i]);
                b.add_cell(6 * fi + 2, &[vertices[1], nv + edges[0], i]);
                b.add_cell(6 * fi + 3, &[nv + edges[0], vertices[2], i]);
                b.add_cell(6 * fi + 4, &[vertices[2], nv + edges[1], i]);
                b.add_cell(6 * fi + 5, &[nv + edges[1], vertices[0], i]);
            }
            ReferenceCellType::Quadrilateral => {
                b.add_cell(8 * fi, &[vertices[0], nv + edges[0], i]);
                b.add_cell(8 * fi + 1, &[nv + edges[0], vertices[1], i]);
                b.add_cell(8 * fi + 2, &[vertices[1], nv + edges[2], i]);
                b.add_cell(8 * fi + 3, &[nv + edges[2], vertices[3], i]);
                b.add_cell(8 * fi + 4, &[vertices[3], nv + edges[3], i]);
                b.add_cell(8 * fi + 5, &[nv + edges[3], vertices[2], i]);
                b.add_cell(8 * fi + 6, &[vertices[2], nv + edges[1], i]);
                b.add_cell(8 * fi + 7, &[nv + edges[1], vertices[0], i]);
            }
            _ => {
                panic!("Unsupported cell type: {:?}", f.entity_type());
            }
        }

        i += 1;
    }

    b.create_grid()
}

#[cfg(test)]
mod test {
    use super::*;
    use ndgrid::shapes::unit_cube_boundary;

    #[test]
    fn test_refine_triangle() {
        let grid = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle);
        let bgrid = bary_refine(&grid);
        assert_eq!(
            grid.entity_count(ReferenceCellType::Triangle) * 6,
            bgrid.entity_count(ReferenceCellType::Triangle)
        );
    }

    #[test]
    fn test_refine_quadrilateral() {
        let grid = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Quadrilateral);
        let bgrid = bary_refine(&grid);
        assert_eq!(
            grid.entity_count(ReferenceCellType::Quadrilateral) * 8,
            bgrid.entity_count(ReferenceCellType::Triangle)
        );
    }
}
