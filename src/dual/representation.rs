//! Representing a space in terms of a barycentric space
use crate::RefinedGrid;
use itertools::izip;
use ndelement::{
    ciarlet::CiarletElement,
    traits::{FiniteElement, Map, MappedFiniteElement},
    types::ReferenceCellType,
};
use ndfunctionspace::traits::FunctionSpace;
use ndgrid::{
    traits::{Entity, Geometry, GeometryMap, Grid, Topology},
    types::Scalar,
};
use rlst::{DynArray, RlstScalar, rlst_dynamic_array};

use std::collections::HashMap;

/// Compute coefficients for the barycentric representation of a space
pub fn coefficients<
    'a,
    TGeo: Scalar,
    T: Scalar,
    G: Grid<T = TGeo, EntityDescriptor = ReferenceCellType>,
    FineG: Grid<T = TGeo, EntityDescriptor = ReferenceCellType>,
    M: Map,
>(
    grid: &'a RefinedGrid<'a, TGeo, G, FineG>,
    coarse_space: &impl FunctionSpace<
        EntityDescriptor = ReferenceCellType,
        Grid = G,
        FiniteElement = CiarletElement<T, M, TGeo>,
    >,
    fine_space: &impl FunctionSpace<
        EntityDescriptor = ReferenceCellType,
        Grid = FineG,
        FiniteElement = CiarletElement<T, M, TGeo>,
    >,
) -> Vec<HashMap<usize, T>> {
    assert_eq!(
        grid.coarse_grid() as *const G,
        coarse_space.grid() as *const G
    );
    assert_eq!(
        grid.fine_grid() as *const FineG,
        fine_space.grid() as *const FineG
    );

    let mut coefficients = vec![HashMap::new(); coarse_space.local_size()];

    assert_eq!(fine_space.elements().len(), 1);
    let fine_e = &fine_space.elements()[0];

    for ct in grid.coarse_grid().cell_types() {
        let child_to_parent_maps = match ct {
            ReferenceCellType::Triangle => vec![
                |x: &[TGeo]| {
                    [
                        x[0] / TGeo::from(2).unwrap() + x[1] / TGeo::from(3).unwrap(),
                        x[1] / TGeo::from(3).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        TGeo::from(0.5).unwrap() + x[0] / TGeo::from(2).unwrap()
                            - x[1] / TGeo::from(6).unwrap(),
                        x[1] / TGeo::from(3).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        TGeo::from(1.0).unwrap()
                            - x[0] / TGeo::from(2).unwrap()
                            - x[1] * TGeo::from(2).unwrap() / TGeo::from(3).unwrap(),
                        x[0] / TGeo::from(2).unwrap() + x[1] / TGeo::from(3).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        TGeo::from(0.5).unwrap()
                            - x[0] / TGeo::from(2).unwrap()
                            - x[1] / TGeo::from(6).unwrap(),
                        TGeo::from(0.5).unwrap() + x[0] / TGeo::from(2).unwrap()
                            - x[1] / TGeo::from(6).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        x[1] / TGeo::from(3).unwrap(),
                        TGeo::from(1.0).unwrap()
                            - x[0] / TGeo::from(2).unwrap()
                            - x[1] * TGeo::from(2).unwrap() / TGeo::from(3).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        x[1] / TGeo::from(3).unwrap(),
                        TGeo::from(0.5).unwrap()
                            - x[0] / TGeo::from(2).unwrap()
                            - x[1] / TGeo::from(6).unwrap(),
                    ]
                },
            ],
            ReferenceCellType::Quadrilateral => vec![
                |x: &[TGeo]| {
                    [
                        x[0] / TGeo::from(2).unwrap() + x[1] / TGeo::from(2).unwrap(),
                        x[1] / TGeo::from(2).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        TGeo::from(0.5).unwrap() + x[0] / TGeo::from(2).unwrap(),
                        x[1] / TGeo::from(2).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        TGeo::from(1.0).unwrap() - x[1] / TGeo::from(2).unwrap(),
                        x[0] / TGeo::from(2).unwrap() + x[1] / TGeo::from(2).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        TGeo::from(1.0).unwrap() - x[1] / TGeo::from(2).unwrap(),
                        TGeo::from(0.5).unwrap() + x[0] / TGeo::from(2).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        TGeo::from(1.0).unwrap()
                            - x[0] / TGeo::from(2).unwrap()
                            - x[1] / TGeo::from(2).unwrap(),
                        TGeo::from(1.0).unwrap() - x[1] / TGeo::from(2).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        TGeo::from(0.5).unwrap() - x[0] / TGeo::from(2).unwrap(),
                        TGeo::from(1.0).unwrap() - x[1] / TGeo::from(2).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        x[1] / TGeo::from(2).unwrap(),
                        TGeo::from(1.0).unwrap()
                            - x[0] / TGeo::from(2).unwrap()
                            - x[1] / TGeo::from(2).unwrap(),
                    ]
                },
                |x: &[TGeo]| {
                    [
                        x[1] / TGeo::from(2).unwrap(),
                        TGeo::from(0.5).unwrap() - x[0] / TGeo::from(2).unwrap(),
                    ]
                },
            ],
            _ => {
                panic!("Unsupported cell type: {ct:?}");
            }
        };

        let coarse_es = coarse_space
            .elements()
            .iter()
            .filter(|e| e.cell_type() == *ct)
            .collect::<Vec<_>>();
        assert_eq!(coarse_es.len(), 1);
        let coarse_e = coarse_es[0];

        let fine_ndofs = fine_e.dim();
        let coarse_ndofs = coarse_e.dim();
        let mut cell_coeffs = vec![T::zero(); coarse_ndofs * fine_ndofs];

        for cell in grid.coarse_grid().entity_iter(*ct) {
            let coarse_cell_dofs = coarse_space
                .entity_closure_dofs(*ct, cell.local_index())
                .unwrap();
            for (fine_cell_index, map) in
                izip!(grid.children(cell.local_index()), &child_to_parent_maps)
            {
                let fine_cell = grid
                    .fine_grid()
                    .entity(fine_e.cell_type(), *fine_cell_index)
                    .unwrap();
                let fine_cell_dofs = fine_space
                    .entity_closure_dofs(fine_e.cell_type(), *fine_cell_index)
                    .unwrap();
                let mut dof_i = 0;
                for (dim, (pts_list, wts_list)) in izip!(
                    fine_e.interpolation_points(),
                    fine_e.interpolation_weights()
                )
                .enumerate()
                {
                    for (entity_i, (pts, wts)) in izip!(pts_list, wts_list).enumerate() {
                        if pts.shape()[1] > 0 {
                            let fine_entity_dofs = fine_e
                                .entity_dofs(dim, entity_i)
                                .unwrap()
                                .iter()
                                .map(|i| fine_cell_dofs[*i])
                                .collect::<Vec<_>>();

                            let mut mapped_pts = DynArray::<TGeo, 2>::from_shape(pts.shape());
                            for i in 0..pts.shape()[1] {
                                [
                                    *mapped_pts.get_mut([0, i]).unwrap(),
                                    *mapped_pts.get_mut([1, i]).unwrap(),
                                ] = map(pts.r().slice::<1>(1, i).data().unwrap());
                            }
                            let mut table = DynArray::<T, 4>::from_shape(
                                coarse_e.tabulate_array_shape(0, pts.shape()[1]),
                            );
                            coarse_e.tabulate(&mapped_pts, 0, &mut table);

                            let coarse_gmap = grid.coarse_grid().geometry_map(
                                *ct,
                                cell.geometry().degree(),
                                &pts,
                            );
                            let fine_gmap = grid.fine_grid().geometry_map(
                                fine_e.cell_type(),
                                fine_cell.geometry().degree(),
                                &mapped_pts,
                            );

                            let npts = pts.shape()[1];
                            let mut jacobians = rlst_dynamic_array!(
                                TGeo,
                                [
                                    grid.coarse_grid().geometry_dim(),
                                    grid.coarse_grid().topology_dim(),
                                    npts
                                ]
                            );
                            let mut jinv = rlst_dynamic_array!(
                                TGeo,
                                [
                                    grid.coarse_grid().topology_dim(),
                                    grid.coarse_grid().geometry_dim(),
                                    npts
                                ]
                            );
                            let mut jdets = vec![TGeo::zero(); npts];
                            let mut physical_values = rlst_dynamic_array!(
                                T,
                                [
                                    table.shape()[0],
                                    table.shape()[1],
                                    table.shape()[2],
                                    coarse_e.physical_value_size(3),
                                ]
                            );

                            coarse_gmap.jacobians_inverses_dets(
                                cell.local_index(),
                                &mut jacobians,
                                &mut jinv,
                                &mut jdets,
                            );
                            coarse_e.push_forward(
                                &table,
                                0,
                                &jacobians,
                                &jdets,
                                &jinv,
                                &mut physical_values,
                            );
                            let mut sub_table = vec![T::zero(); physical_values.shape()[2]];
                            for i3 in 0..physical_values.shape()[3] {
                                for i1 in 0..physical_values.shape()[1] {
                                    for i0 in 0..physical_values.shape()[0] {
                                        for (i2, st) in sub_table.iter_mut().enumerate() {
                                            *st = physical_values[[i0, i1, i2, i3]];
                                        }
                                        coarse_e.apply_dof_permutations_and_transformations(
                                            &mut sub_table,
                                            cell.topology().orientation(),
                                        );
                                        for (i2, st) in sub_table.iter().enumerate() {
                                            physical_values[[i0, i1, i2, i3]] = *st;
                                        }
                                    }
                                }
                            }
                            fine_gmap.jacobians_inverses_dets(
                                fine_cell.local_index(),
                                &mut jacobians,
                                &mut jinv,
                                &mut jdets,
                            );
                            fine_e.pull_back(
                                &physical_values,
                                0,
                                &jacobians,
                                &jdets,
                                &jinv,
                                &mut table,
                            );

                            for (coarse_dof_i, _coarse_dof) in coarse_cell_dofs.iter().enumerate() {
                                for (fine_dof_i, _fine_dof) in fine_entity_dofs.iter().enumerate() {
                                    cell_coeffs[coarse_dof_i * fine_ndofs + dof_i + fine_dof_i] =
                                        (0..table.shape()[1])
                                            .map(|pt_i| {
                                                (0..coarse_e.value_size())
                                                    .map(|vs| {
                                                        *wts.get([fine_dof_i, vs, pt_i]).unwrap()
                                                            * *table
                                                                .get([0, pt_i, coarse_dof_i, vs])
                                                                .unwrap()
                                                    })
                                                    .sum()
                                            })
                                            .sum();
                                }
                            }
                            dof_i += fine_entity_dofs.len();
                        }
                    }
                }
                for (coarse_dof_i, coarse_dof) in coarse_cell_dofs.iter().enumerate() {
                    let co = &mut cell_coeffs
                        [coarse_dof_i * fine_ndofs..(coarse_dof_i + 1) * fine_ndofs];
                    fine_e.apply_dof_permutations_and_transformations(
                        // TODO: inverse
                        co,
                        fine_cell.topology().orientation(),
                    );
                    for (fine_dof, value) in izip!(fine_cell_dofs, co) {
                        if value.abs().re() > T::from(1e-10).unwrap().re() {
                            coefficients[*coarse_dof].insert(*fine_dof, *value);
                        }
                    }
                }
            }
        }
    }

    coefficients
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{DualSpace, assemble_mass_matrix, assemble_mass_matrix_dual};
    use approx::*;
    use ndelement::{
        ciarlet::{LagrangeElementFamily, RaviartThomasElementFamily},
        traits::{ElementFamily, MappedFiniteElement},
        types::Continuity,
    };
    use ndfunctionspace::FunctionSpaceImpl;
    use ndgrid::{
        shapes,
        traits::{GeometryMap, Topology},
    };
    use quadraturerules::{Domain, QuadratureRule, single_integral_quadrature};
    use rlst::rlst_dynamic_array;

    #[test]
    fn test_lagrange_triangles() {
        let grid = shapes::regular_sphere::<f64>(1);
        let rgrid = RefinedGrid::new(&grid);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let space = FunctionSpaceImpl::new(&grid, &family);
        let fine_space = FunctionSpaceImpl::new(rgrid.fine_grid(), &family);

        let coefficients = coefficients(&rgrid, &space, &fine_space);
        let bary_space = DualSpace::new(&rgrid, &fine_space, coefficients);

        let result = assemble_mass_matrix(&space, &space);
        let bary_result = assemble_mass_matrix_dual(&bary_space, &bary_space);

        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                assert_relative_eq!(result[[i, j]], bary_result[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_lagrange_triangles_mixed_degree() {
        let grid = shapes::regular_sphere::<f64>(1);
        let rgrid = RefinedGrid::new(&grid);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let fine_family = LagrangeElementFamily::<f64>::new(2, Continuity::Standard);
        let space = FunctionSpaceImpl::new(&grid, &family);
        let fine_space = FunctionSpaceImpl::new(rgrid.fine_grid(), &fine_family);

        let coefficients = coefficients(&rgrid, &space, &fine_space);
        let bary_space = DualSpace::new(&rgrid, &fine_space, coefficients);

        let result = assemble_mass_matrix(&space, &space);
        let bary_result = assemble_mass_matrix_dual(&bary_space, &bary_space);

        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                assert_relative_eq!(result[[i, j]], bary_result[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_lagrange_triangles_mixed_continuity() {
        let grid = shapes::regular_sphere::<f64>(1);
        let rgrid = RefinedGrid::new(&grid);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let fine_family = LagrangeElementFamily::<f64>::new(1, Continuity::Discontinuous);
        let space = FunctionSpaceImpl::new(&grid, &family);
        let fine_space = FunctionSpaceImpl::new(rgrid.fine_grid(), &fine_family);

        let coefficients = coefficients(&rgrid, &space, &fine_space);
        let bary_space = DualSpace::new(&rgrid, &fine_space, coefficients);

        let result = assemble_mass_matrix(&space, &space);
        let bary_result = assemble_mass_matrix_dual(&bary_space, &bary_space);

        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                assert_relative_eq!(result[[i, j]], bary_result[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_lagrange_quads() {
        let grid = shapes::screen::<f64>(1, ReferenceCellType::Quadrilateral);
        let rgrid = RefinedGrid::new(&grid);

        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let fine_family = LagrangeElementFamily::<f64>::new(2, Continuity::Standard);
        let space = FunctionSpaceImpl::new(&grid, &family);
        let fine_space = FunctionSpaceImpl::new(rgrid.fine_grid(), &fine_family);

        let coefficients = coefficients(&rgrid, &space, &fine_space);

        assert_relative_eq!(coefficients[0][&0], 1.0);
        assert_relative_eq!(coefficients[0][&24], 3.0 / 4.0);
        assert_relative_eq!(coefficients[0][&21], 1.0 / 2.0);
        assert_relative_eq!(coefficients[0][&23], 1.0 / 4.0);
        assert_relative_eq!(coefficients[0][&5], 3.0 / 4.0);
        assert_relative_eq!(coefficients[0][&4], 9.0 / 16.0);
        assert_relative_eq!(coefficients[0][&22], 3.0 / 8.0);
        assert_relative_eq!(coefficients[0][&19], 3.0 / 16.0);
        assert_relative_eq!(coefficients[0][&1], 1.0 / 2.0);
        assert_relative_eq!(coefficients[0][&3], 3.0 / 8.0);
        assert_relative_eq!(coefficients[0][&2], 1.0 / 4.0);
        assert_relative_eq!(coefficients[0][&16], 1.0 / 8.0);
        assert_relative_eq!(coefficients[0][&8], 1.0 / 4.0);
        assert_relative_eq!(coefficients[0][&7], 3.0 / 16.0);
        assert_relative_eq!(coefficients[0][&10], 1.0 / 8.0);
        assert_relative_eq!(coefficients[0][&13], 1.0 / 16.0);

        assert_relative_eq!(coefficients[1][&6], 1.0);
        assert_relative_eq!(coefficients[1][&8], 3.0 / 4.0);
        assert_relative_eq!(coefficients[1][&1], 1.0 / 2.0);
        assert_relative_eq!(coefficients[1][&5], 1.0 / 4.0);
        assert_relative_eq!(coefficients[1][&11], 3.0 / 4.0);
        assert_relative_eq!(coefficients[1][&7], 9.0 / 16.0);
        assert_relative_eq!(coefficients[1][&3], 3.0 / 8.0);
        assert_relative_eq!(coefficients[1][&4], 3.0 / 16.0);
        assert_relative_eq!(coefficients[1][&9], 1.0 / 2.0);
        assert_relative_eq!(coefficients[1][&10], 3.0 / 8.0);
        assert_relative_eq!(coefficients[1][&2], 1.0 / 4.0);
        assert_relative_eq!(coefficients[1][&22], 1.0 / 8.0);
        assert_relative_eq!(coefficients[1][&14], 1.0 / 4.0);
        assert_relative_eq!(coefficients[1][&13], 3.0 / 16.0);
        assert_relative_eq!(coefficients[1][&16], 1.0 / 8.0);
        assert_relative_eq!(coefficients[1][&19], 1.0 / 16.0);

        assert_relative_eq!(coefficients[2][&18], 1.0);
        assert_relative_eq!(coefficients[2][&23], 3.0 / 4.0);
        assert_relative_eq!(coefficients[2][&21], 1.0 / 2.0);
        assert_relative_eq!(coefficients[2][&24], 1.0 / 4.0);
        assert_relative_eq!(coefficients[2][&20], 3.0 / 4.0);
        assert_relative_eq!(coefficients[2][&19], 9.0 / 16.0);
        assert_relative_eq!(coefficients[2][&22], 3.0 / 8.0);
        assert_relative_eq!(coefficients[2][&4], 3.0 / 16.0);
        assert_relative_eq!(coefficients[2][&15], 1.0 / 2.0);
        assert_relative_eq!(coefficients[2][&16], 3.0 / 8.0);
        assert_relative_eq!(coefficients[2][&2], 1.0 / 4.0);
        assert_relative_eq!(coefficients[2][&3], 1.0 / 8.0);
        assert_relative_eq!(coefficients[2][&17], 1.0 / 4.0);
        assert_relative_eq!(coefficients[2][&13], 3.0 / 16.0);
        assert_relative_eq!(coefficients[2][&10], 1.0 / 8.0);
        assert_relative_eq!(coefficients[2][&7], 1.0 / 16.0);

        assert_relative_eq!(coefficients[3][&12], 1.0);
        assert_relative_eq!(coefficients[3][&14], 3.0 / 4.0);
        assert_relative_eq!(coefficients[3][&9], 1.0 / 2.0);
        assert_relative_eq!(coefficients[3][&11], 1.0 / 4.0);
        assert_relative_eq!(coefficients[3][&17], 3.0 / 4.0);
        assert_relative_eq!(coefficients[3][&13], 9.0 / 16.0);
        assert_relative_eq!(coefficients[3][&10], 3.0 / 8.0);
        assert_relative_eq!(coefficients[3][&7], 3.0 / 16.0);
        assert_relative_eq!(coefficients[3][&15], 1.0 / 2.0);
        assert_relative_eq!(coefficients[3][&16], 3.0 / 8.0);
        assert_relative_eq!(coefficients[3][&2], 1.0 / 4.0);
        assert_relative_eq!(coefficients[3][&3], 1.0 / 8.0);
        assert_relative_eq!(coefficients[3][&20], 1.0 / 4.0);
        assert_relative_eq!(coefficients[3][&19], 3.0 / 16.0);
        assert_relative_eq!(coefficients[3][&22], 1.0 / 8.0);
        assert_relative_eq!(coefficients[3][&4], 1.0 / 16.0);

        let bary_space = DualSpace::new(&rgrid, &fine_space, coefficients);

        let result = assemble_mass_matrix(&space, &space);
        let bary_result = assemble_mass_matrix_dual(&bary_space, &bary_space);

        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                assert_relative_eq!(result[[i, j]], bary_result[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_lagrange_integral() {
        //! Test that integral(v) is the same for Lagrange and barycentric Lagrange
        let grid = shapes::regular_sphere::<f64>(2);
        let rgrid = RefinedGrid::new(&grid);

        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);

        let coarse_space = FunctionSpaceImpl::new(&grid, &family);
        let fine_space = FunctionSpaceImpl::new(rgrid.fine_grid(), &family);

        let coefficients = coefficients(&rgrid, &coarse_space, &fine_space);

        let (p, w) =
            single_integral_quadrature(QuadratureRule::XiaoGimbutas, Domain::Triangle, 2).unwrap();
        let npts = w.len();
        let mut pts = rlst_dynamic_array!(f64, [2, npts]);
        for i in 0..npts {
            for j in 0..2 {
                *pts.get_mut([j, i]).unwrap() = p[3 * i + j];
            }
        }
        let wts = w.iter().map(|i| *i / 2.0).collect::<Vec<_>>();

        let e = family.element(ReferenceCellType::Triangle);

        let mut table = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, npts));
        e.tabulate(&pts, 0, &mut table);

        let mut jacobians = rlst_dynamic_array!(f64, [3, 2, npts]);
        let mut jinv = rlst_dynamic_array!(f64, [2, 3, npts]);
        let mut jdets = vec![0.0; npts];

        let mut coarse_result = vec![0.0; coarse_space.local_size()];
        let gmap = grid.geometry_map(ReferenceCellType::Triangle, 1, &pts);
        for cell in grid.entity_iter(ReferenceCellType::Triangle) {
            let dofs = coarse_space
                .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
                .unwrap();
            gmap.jacobians_inverses_dets(cell.local_index(), &mut jacobians, &mut jinv, &mut jdets);
            for (i, dof) in dofs.iter().enumerate() {
                coarse_result[*dof] += wts
                    .iter()
                    .enumerate()
                    .map(|(w_i, w)| jdets[w_i] * w * table.get([0, w_i, i, 0]).unwrap())
                    .sum::<f64>();
            }
        }

        let fine_grid = rgrid.fine_grid();

        let mut fine_result = vec![0.0; fine_space.local_size()];
        let gmap = fine_grid.geometry_map(ReferenceCellType::Triangle, 1, &pts);
        for cell in fine_grid.entity_iter(ReferenceCellType::Triangle) {
            let dofs = fine_space
                .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
                .unwrap();
            gmap.jacobians_inverses_dets(cell.local_index(), &mut jacobians, &mut jinv, &mut jdets);
            for (i, dof) in dofs.iter().enumerate() {
                fine_result[*dof] += wts
                    .iter()
                    .enumerate()
                    .map(|(w_i, w)| jdets[w_i] * w * table.get([0, w_i, i, 0]).unwrap())
                    .sum::<f64>();
            }
        }

        for (i, c) in coefficients.iter().enumerate() {
            assert_relative_eq!(
                coarse_result[i],
                c.iter()
                    .map(|(dof_i, weight)| weight * fine_result[*dof_i])
                    .sum::<f64>()
            );
        }
    }

    #[test]
    fn test_rt_integral() {
        //! Test that integral(v[0]) is the same for RT and barycentric RT
        let grid = shapes::screen::<f64>(1, ReferenceCellType::Triangle);
        let rgrid = RefinedGrid::new(&grid);

        let family = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);

        let coarse_space = FunctionSpaceImpl::new(&grid, &family);
        let fine_space = FunctionSpaceImpl::new(rgrid.fine_grid(), &family);

        let coefficients = coefficients(&rgrid, &coarse_space, &fine_space);

        let (p, w) =
            single_integral_quadrature(QuadratureRule::XiaoGimbutas, Domain::Triangle, 2).unwrap();
        let npts = w.len();
        let mut pts = rlst_dynamic_array!(f64, [2, npts]);
        for i in 0..npts {
            for j in 0..2 {
                *pts.get_mut([j, i]).unwrap() = p[3 * i + j];
            }
        }
        let wts = w.iter().map(|i| *i / 2.0).collect::<Vec<_>>();

        let e = family.element(ReferenceCellType::Triangle);

        let mut table = DynArray::<f64, 4>::from_shape(e.tabulate_array_shape(0, npts));
        e.tabulate(&pts, 0, &mut table);

        let mut jacobians = rlst_dynamic_array!(f64, [3, 2, npts]);
        let mut jinv = rlst_dynamic_array!(f64, [2, 3, npts]);
        let mut jdets = vec![0.0; npts];

        let mut physical_values = rlst_dynamic_array!(
            f64,
            [
                table.shape()[0],
                table.shape()[1],
                table.shape()[2],
                e.physical_value_size(3),
            ]
        );

        let mut local_vector = vec![0.0; 3];

        let mut coarse_result = vec![0.0; coarse_space.local_size()];
        let gmap = grid.geometry_map(ReferenceCellType::Triangle, 1, &pts);
        for cell in grid.entity_iter(ReferenceCellType::Triangle) {
            let dofs = coarse_space
                .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
                .unwrap();
            gmap.jacobians_inverses_dets(cell.local_index(), &mut jacobians, &mut jinv, &mut jdets);
            e.push_forward(&table, 0, &jacobians, &jdets, &jinv, &mut physical_values);
            for (i, _dof) in dofs.iter().enumerate() {
                local_vector[i] = wts
                    .iter()
                    .enumerate()
                    .map(|(w_i, w)| jdets[w_i] * w * physical_values.get([0, w_i, i, 0]).unwrap())
                    .sum();
            }
            e.apply_dof_permutations_and_transformations(
                &mut local_vector,
                cell.topology().orientation(),
            );
            for (dof, value) in izip!(dofs, &local_vector) {
                coarse_result[*dof] += value;
            }
        }

        let fine_grid = rgrid.fine_grid();

        let mut fine_result = vec![0.0; fine_space.local_size()];
        let gmap = fine_grid.geometry_map(ReferenceCellType::Triangle, 1, &pts);
        for cell in fine_grid.entity_iter(ReferenceCellType::Triangle) {
            let dofs = fine_space
                .entity_closure_dofs(ReferenceCellType::Triangle, cell.local_index())
                .unwrap();
            gmap.jacobians_inverses_dets(cell.local_index(), &mut jacobians, &mut jinv, &mut jdets);
            e.push_forward(&table, 0, &jacobians, &jdets, &jinv, &mut physical_values);
            for (i, _dof) in dofs.iter().enumerate() {
                local_vector[i] = wts
                    .iter()
                    .enumerate()
                    .map(|(w_i, w)| jdets[w_i] * w * physical_values.get([0, w_i, i, 0]).unwrap())
                    .sum();
            }
            e.apply_dof_permutations_and_transformations(
                &mut local_vector,
                cell.topology().orientation(),
            );
            for (dof, value) in izip!(dofs, &local_vector) {
                fine_result[*dof] += value;
            }
        }

        dbg!(&coefficients[3]);

        for (i, c) in coefficients.iter().enumerate() {
            assert_relative_eq!(
                coarse_result[i],
                c.iter()
                    .map(|(dof_i, weight)| weight * fine_result[*dof_i])
                    .sum::<f64>()
            );
        }
    }
}
