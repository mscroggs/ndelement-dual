//! Max matrix assembly
use super::FunctionSpace;
use ndelement::{
    traits::{ElementFamily, FiniteElement},
    types::ReferenceCellType,
};
use ndgrid::{
    traits::{Entity, GeometryMap, Grid},
    types::Array2D,
};
use quadraturerules::{Domain, QuadratureRule, single_integral_quadrature};
use rlst::{
    RandomAccessByRef, RandomAccessMut, RawAccess, RlstScalar, Shape, rlst_dynamic_array2,
    rlst_dynamic_array4,
};
use std::cmp::max;

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
                let (p, w) = single_integral_quadrature(
                    QuadratureRule::GaussLobattoLegendre,
                    Domain::Quadrilateral,
                    max(
                        test_e.embedded_superdegree(),
                        trial_e.embedded_superdegree(),
                    ),
                )
                .unwrap();
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
                for (test_i, test_dof) in test_dofs.iter().enumerate() {
                    for (trial_i, trial_dof) in trial_dofs.iter().enumerate() {
                        for (i, w) in weights.iter().enumerate() {
                            for j in 0..test_table.shape()[3] {
                                *matrix.get_mut([*test_dof, *trial_dof]).unwrap() +=
                                    T::from(jdets[i]).unwrap()
                                        * *w
                                        * *test_table.get([0, i, test_i, j]).unwrap()
                                        * *trial_table.get([0, i, trial_i, j]).unwrap();
                            }
                        }
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
    use ndelement::{ciarlet::LagrangeElementFamily, types::Continuity};
    use ndgrid::shapes::regular_sphere;

    #[test]
    fn test_assembly() {
        let grid = regular_sphere::<f64>(0);
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
}
