//! Representing a space in terms of a barycentric space
use crate::RefinedGrid;
use itertools::izip;
use ndelement::{
    ciarlet::CiarletElement,
    traits::{FiniteElement, Map},
    types::ReferenceCellType,
};
use ndfunctionspace::traits::FunctionSpace;
use ndgrid::{
    traits::{Entity, Grid},
    types::Scalar,
};
use rlst::{DynArray, RlstScalar};
use std::collections::HashMap;

/// Compute coefficients for the barycentric representation of a space
pub fn barycentric_representation_coefficients<
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
        for cell in grid.coarse_grid().entity_iter(*ct) {
            let coarse_cell_dofs = coarse_space
                .entity_closure_dofs(*ct, cell.local_index())
                .unwrap();
            for (fine_cell, map) in izip!(grid.children(cell.local_index()), &child_to_parent_maps)
            {
                let fine_cell_dofs = fine_space
                    .entity_closure_dofs(fine_e.cell_type(), *fine_cell)
                    .unwrap();
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
                            for (coarse_dof_i, coarse_dof) in coarse_cell_dofs.iter().enumerate() {
                                for (fine_dof_i, fine_dof) in fine_entity_dofs.iter().enumerate() {
                                    let value: T = (0..table.shape()[1])
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
                                    if value.abs().re() > T::from(1e-10).unwrap().re() {
                                        coefficients[*coarse_dof].insert(*fine_dof, value);
                                    }
                                }
                            }
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
    use ndelement::{ciarlet::LagrangeElementFamily, types::Continuity};
    use ndfunctionspace::FunctionSpaceImpl;
    use ndgrid::shapes;

    #[test]
    fn test_lagrange_triangles() {
        let grid = shapes::regular_sphere::<f64>(1);
        let rgrid = RefinedGrid::new(&grid);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let space = FunctionSpaceImpl::new(&grid, &family);
        let fine_space = FunctionSpaceImpl::new(rgrid.fine_grid(), &family);

        let coefficients = barycentric_representation_coefficients(&rgrid, &space, &fine_space);
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

        let coefficients = barycentric_representation_coefficients(&rgrid, &space, &fine_space);
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

        let coefficients = barycentric_representation_coefficients(&rgrid, &space, &fine_space);
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

        let coefficients = barycentric_representation_coefficients(&rgrid, &space, &fine_space);

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
}
