//! Max matrix assembly
use crate::{FunctionSpace, DualSpace};
use ndelement::{
    traits::{ElementFamily, FiniteElement},
    types::ReferenceCellType,
};
use ndgrid::{
    traits::{Entity, GeometryMap, Grid, Topology},
    types::{Array2D, RealScalar},
};
use quadraturerules::{Domain, QuadratureRule, single_integral_quadrature};
use rlst::{
    RandomAccessByRef, RandomAccessMut, RawAccess, RlstScalar, Shape, rlst_dynamic_array2, RawAccessMut,
    rlst_dynamic_array4,
};
use std::cmp::max;

/// Assemble a mass matrix using dual spaces
pub fn assemble_dual<
    'a,
    TReal: RealScalar,
    T: RlstScalar<Real = TReal>,
    G: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    TestF: ElementFamily<T = T, CellType = ReferenceCellType>,
    TrialF: ElementFamily<T = T, CellType = ReferenceCellType>,
>(
    test_space: &DualSpace<'a, TReal, T, G, TestF>,
    trial_space: &DualSpace<'a, TReal, T, G, TrialF>,
) -> Array2D<T> {
    let fine_mat = assemble(test_space.fine_space(), trial_space.fine_space());

    let mut matrix = rlst_dynamic_array2!(T, [test_space.dim(), trial_space.dim()]);

    for (test_i, test_coeffs) in test_space.coefficients().iter().enumerate() {
        for (trial_i, trial_coeffs) in trial_space.coefficients().iter().enumerate() {
            *matrix.get_mut([test_i, trial_i]).unwrap() = test_coeffs.iter().map(
                |(test_dof, test_c)| *test_c * trial_coeffs.iter().map(
                    |(trial_dof, trial_c)| *trial_c * *fine_mat.get([*test_dof, *trial_dof]).unwrap()
                ).sum()
            ).sum();
        }
    }
    matrix
}

/// Assemble a mass matrix
pub fn assemble<
    'a,
    T: RlstScalar,
    G: Grid<T = T::Real, EntityDescriptor = ReferenceCellType>,
    TestF: ElementFamily<T = T, CellType = ReferenceCellType>,
    TrialF: ElementFamily<T = T, CellType = ReferenceCellType>,
>(
    test_space: &FunctionSpace<'a, G, TestF>,
    trial_space: &FunctionSpace<'a, G, TrialF>,
) -> Array2D<T> {
    assert_eq!(
        test_space.grid() as *const G,
        trial_space.grid() as *const G
    );

    let mut matrix = rlst_dynamic_array2!(T, [test_space.dim(), trial_space.dim()]);

    for ct in test_space.grid().cell_types() {
        let test_e = test_space.family().element(*ct);
        let trial_e = trial_space.family().element(*ct);
        let (points, weights) = match ct {
            ReferenceCellType::Triangle => {
                let (p, w) = single_integral_quadrature(
                    QuadratureRule::XiaoGimbutas,
                    Domain::Triangle,
                    test_e.embedded_superdegree() + trial_e.embedded_superdegree(),
                )
                .unwrap();
                let mut pts = rlst_dynamic_array2!(T::Real, [2, w.len()]);
                for i in 0..w.len() {
                    for j in 0..2 {
                        *pts.get_mut([j, i]).unwrap() = T::from(p[3 * i + j]).unwrap().re();
                    }
                }
                (
                    pts,
                    w.iter()
                        .map(|i| T::from(*i / 2.0).unwrap())
                        .collect::<Vec<_>>(),
                )
            }
            ReferenceCellType::Quadrilateral => {
                println!("{}",
                    (max(
                        test_e.embedded_superdegree(),
                        trial_e.embedded_superdegree(),
                    ) + 1) / 2);
                let (p, w) = single_integral_quadrature(
                    QuadratureRule::GaussLobattoLegendre,
                    Domain::Interval,
                    (max(
                        test_e.embedded_superdegree(),
                        trial_e.embedded_superdegree(),
                    ) + 1) / 2,
                )
                .unwrap();
                println!("HERE");
                let mut pts = rlst_dynamic_array2!(T::Real, [2, w.len() * w.len()]);
                let mut wts = vec![T::zero(); w.len() * w.len()];
                for (i, wi) in w.iter().enumerate() {
                    for (j, wj) in w.iter().enumerate() {
                        wts[w.len() * i + j] = T::from(wi * wj).unwrap();
                        *pts.get_mut([0, w.len() * i + j]).unwrap() = T::from(p[i]).unwrap().re();
                        *pts.get_mut([1, w.len() * i + j]).unwrap() = T::from(p[j]).unwrap().re();
                    }
                }
                (pts, wts)
            }
            _ => {
                panic!("Unsupported cell type: {ct:?}");
            }
        };

        let npts = weights.len();

        let mut test_table = rlst_dynamic_array4!(T, test_e.tabulate_array_shape(0, npts));
        test_e.tabulate(&points, 0, &mut test_table);

        let mut trial_table = rlst_dynamic_array4!(T, trial_e.tabulate_array_shape(0, npts));
        trial_e.tabulate(&points, 0, &mut trial_table);

        let gmap = test_space.grid().geometry_map(*ct, points.data());

        let mut jacobians =
            vec![
                T::zero().re();
                test_space.grid().geometry_dim() * test_space.grid().topology_dim() * npts
            ];
        let mut jdets = vec![T::zero().re(); npts];
        let mut normals = vec![T::zero().re(); test_space.grid().geometry_dim() * npts];

        let mut local_matrix = rlst_dynamic_array2!(T, [test_table.shape()[2], trial_table.shape()[2]]);

        for cell in test_space.grid().cell_iter() {
            if cell.entity_type() == *ct {
                let test_dofs = test_space.cell_dofs(cell.local_index());
                let trial_dofs = trial_space.cell_dofs(cell.local_index());
                gmap.jacobians_dets_normals(
                    cell.local_index(),
                    &mut jacobians,
                    &mut jdets,
                    &mut normals,
                );
                for (test_i, _test_dof) in test_dofs.iter().enumerate() {
                    for (trial_i, _trial_dof) in trial_dofs.iter().enumerate() {
                        *local_matrix.get_mut([test_i, trial_i]).unwrap() = weights.iter().enumerate().map(|(i, w)| 
                            (0..test_table.shape()[3]).map(|j|
                                T::from(jdets[i]).unwrap() * *w * *test_table.get([0, i, test_i, j]).unwrap() * *trial_table.get([0, i, trial_i, j]).unwrap()
                            ).sum()
                        ).sum();
                    }
                }
                for i in 0..local_matrix.shape()[1] {
                    test_e.apply_dof_permutations_and_transformations(local_matrix.r_mut().slice(1, i).data_mut(), cell.topology().orientation());
                }
                trial_e.apply_dof_permutations_and_transformations(local_matrix.data_mut(), cell.topology().orientation());                
                for (test_i, test_dof) in test_dofs.iter().enumerate() {
                    for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                        *matrix.get_mut([*test_dof, *trial_dof]).unwrap() += *local_matrix.get([test_i, trial_i]).unwrap();
                    }
                }
            }
        }
    }
    matrix
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;
    use ndelement::{ciarlet::{LagrangeElementFamily, RaviartThomasElementFamily, NedelecFirstKindElementFamily}, types::Continuity};
    use ndgrid::{shapes, traits::{Geometry, Builder, Point}, SingleElementGridBuilder};
    use rand::{seq::SliceRandom, rng};

    #[test]
    fn test_lagrange_assembly() {
        let grid = shapes::regular_sphere::<f64>(0);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let space = FunctionSpace::new(&grid, &family);
        let result = assemble(&space, &space);

        for i in 0..6 {
            assert_relative_eq!(result[[i, i]], 0.5773502691896255, epsilon = 1e-10);
        }
        for i in 0..6 {
            for j in 0..6 {
                if i != j && result[[i, j]].abs() > 0.001 {
                    assert_relative_eq!(result[[i, j]], 0.1443375672974061, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_rt_nc_assembly() {
        let grid = shapes::regular_sphere::<f64>(0);
        let rt = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
        let nc = NedelecFirstKindElementFamily::<f64>::new(1, Continuity::Standard);
        let rt_space = FunctionSpace::new(&grid, &rt);
        let nc_space = FunctionSpace::new(&grid, &nc);
        let result = assemble(&rt_space, &nc_space);

        for i in 0..6 {
            for j in 0..6 {
                if result[[i, j]].abs() > 0.001 {
                    assert_relative_eq!(result[[i, j]].abs(), f64::sqrt(3.0) / 6.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_rt_nc_assembly_randomly_numbered() {
        let grid1 = shapes::unit_cube_boundary::<f64>(3, 3, 3, ReferenceCellType::Triangle);
        let grid2 = {
            let mut b = SingleElementGridBuilder::new_with_capacity(3, 6, 8, (ReferenceCellType::Triangle, 1));
            let points = grid1.entity_iter(0).map(|v| {
                let mut p = vec![0.0; 3];
                v.geometry().points().next().unwrap().coords(&mut p);
                p
            }).collect::<Vec<_>>();
            let mut indices = (0..points.len()).collect::<Vec<_>>();
            indices.shuffle(&mut rng());
            let mut index_map = vec![0; indices.len()];
            for (i, j) in indices.iter().enumerate() {
                b.add_point(i, &points[*j]);
                index_map[*j] = i;
            }
            for (i, cell) in grid1.cell_iter().enumerate() {
                b.add_cell(i, &cell.geometry().points().map(|p| index_map[p.index()]).collect::<Vec<_>>());
            }
            b.create_grid()
        };

        let rt = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
        let nc = NedelecFirstKindElementFamily::<f64>::new(1, Continuity::Standard);

        let rt_space = FunctionSpace::new(&grid1, &rt);
        let nc_space = FunctionSpace::new(&grid1, &nc);
        let result1 = assemble(&rt_space, &nc_space);

        let rt_space = FunctionSpace::new(&grid2, &rt);
        let nc_space = FunctionSpace::new(&grid2, &nc);
        let result2 = assemble(&rt_space, &nc_space);

        for i in 0..result1.shape()[0] {
            for j in 0..result1.shape()[1] {
                assert_relative_eq!(result1[[i, j]].abs(), result2[[i, j]].abs(), epsilon = 1e-10);
            }
        }
    }
}
