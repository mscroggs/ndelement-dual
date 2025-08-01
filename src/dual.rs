//! Dual spaces
use crate::{FunctionSpace, RefinedGrid};
use itertools::izip;
use ndelement::{
    ciarlet::CiarletElement, map::IdentityMap, traits::{ElementFamily, Map, FiniteElement}, types::ReferenceCellType,
};
use ndgrid::{SingleElementGrid, traits::{Entity, Grid}, types::RealScalar};
use rlst::{RlstScalar, MatrixInverse, RawAccess, Shape, rlst_dynamic_array4, rlst_dynamic_array2, RandomAccessByRef, RandomAccessMut};
use std::collections::HashMap;

/// A dual space
pub struct DualSpace<
    'a,
    TReal: RealScalar,
    T: RlstScalar<Real = TReal>,
    G: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    F: ElementFamily<CellType = ReferenceCellType>,
> {
    grid: &'a RefinedGrid<'a, TReal, G>,
    fine_space:
        &'a FunctionSpace<'a, SingleElementGrid<TReal, CiarletElement<TReal, IdentityMap>>, F>,
    coefficients: Vec<HashMap<usize, T>>,
}

impl<
    'a,
    TReal: RealScalar,
    T: RlstScalar<Real = TReal>,
    G: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    F: ElementFamily<CellType = ReferenceCellType>,
> DualSpace<'a, TReal, T, G, F>
{
    /// Create new
    pub fn new(
        grid: &'a RefinedGrid<'a, TReal, G>,
        fine_space: &'a FunctionSpace<
            'a,
            SingleElementGrid<TReal, CiarletElement<TReal, IdentityMap>>,
            F,
        >,
        coefficients: Vec<HashMap<usize, T>>,
    ) -> Self {
        Self {
            grid,
            fine_space,
            coefficients,
        }
    }

    /// Grid
    pub fn grid(&self) -> &'a RefinedGrid<'a, TReal, G> {
        self.grid
    }

    /// Fine space
    pub fn fine_space(
        &self,
    ) -> &'a FunctionSpace<'a, SingleElementGrid<TReal, CiarletElement<TReal, IdentityMap>>, F>
    {
        self.fine_space
    }

    /// Coefficients
    pub fn coefficients(&self) -> &[HashMap<usize, T>] {
        &self.coefficients
    }

    /// Number of DOFs
    pub fn dim(&self) -> usize {
        self.coefficients.len()
    }
}

/// Create new
pub fn barycentric_representation<
    'a,
    TReal: RealScalar,
    T: RlstScalar<Real = TReal> + MatrixInverse,
    G: Grid<T = TReal, EntityDescriptor = ReferenceCellType>,
    M: Map,
    F: ElementFamily<T=T, CellType = ReferenceCellType, FiniteElement = CiarletElement<T, M>>
>(coarse_space: &'a FunctionSpace<'a, G, F>) -> (RefinedGrid<'a, TReal, G>, Vec<HashMap<usize, T>>) {
    let grid = RefinedGrid::new(coarse_space.grid());
    let fine_space = FunctionSpace::new(grid.fine_grid(), coarse_space.family());
    let mut coefficients = vec![HashMap::new(); coarse_space.dim()];

    let fine_e = fine_space.family().element(ReferenceCellType::Triangle);

    for ct in grid.coarse_grid().cell_types() {
        println!("{ct:?}");

        let child_to_parent_maps = match ct {
            ReferenceCellType::Triangle => vec![
                |x: &[TReal]| [x[0] / T::from(2).unwrap().re() + x[1] / T::from(3).unwrap().re(), x[1] / T::from(3).unwrap().re()],
                |x: &[TReal]| [T::from(0.5).unwrap().re() + x[0] / T::from(2).unwrap().re() - x[1] / T::from(6).unwrap().re(), x[1] / T::from(3).unwrap().re()],
                |x: &[TReal]| [T::from(1.0).unwrap().re() - x[0] / T::from(2).unwrap().re() - x[1] * T::from(2).unwrap().re() / T::from(3).unwrap().re(), x[0] / T::from(2).unwrap().re() + x[1] / T::from(3).unwrap().re()],
                |x: &[TReal]| [T::from(0.5).unwrap().re() - x[0] / T::from(2).unwrap().re() - x[1] / T::from(6).unwrap().re(), T::from(0.5).unwrap().re() + x[0] / T::from(2).unwrap().re() - x[1] / T::from(6).unwrap().re()],
                |x: &[TReal]| [x[1] / T::from(3).unwrap().re(), T::from(1.0).unwrap().re() - x[0] / T::from(2).unwrap().re() - x[1] * T::from(2).unwrap().re() / T::from(3).unwrap().re()],
                |x: &[TReal]| [x[1] / T::from(3).unwrap().re(), T::from(0.5).unwrap().re() - x[0] / T::from(2).unwrap().re() - x[1] / T::from(6).unwrap().re()],
            ],
            ReferenceCellType::Quadrilateral => vec![
                |x: &[TReal]| [x[0] / T::from(2).unwrap().re() + x[1] / T::from(2).unwrap().re(), x[1] / T::from(2).unwrap().re()],
                |x: &[TReal]| [T::from(0.5).unwrap().re() + x[0] / T::from(2).unwrap().re(), x[1] / T::from(2).unwrap().re()],
                |x: &[TReal]| [T::from(1.0).unwrap().re() - x[1] / T::from(2).unwrap().re(), x[0] / T::from(2).unwrap().re() + x[1] / T::from(2).unwrap().re()],
                |x: &[TReal]| [T::from(1.0).unwrap().re() - x[1] / T::from(2).unwrap().re(), T::from(0.5).unwrap().re() + x[0] / T::from(2).unwrap().re()],
                |x: &[TReal]| [T::from(1.0).unwrap().re() - x[0] / T::from(2).unwrap().re() - x[1] / T::from(2).unwrap().re(), T::from(1.0).unwrap().re() - x[1] / T::from(2).unwrap().re()],
                |x: &[TReal]| [T::from(0.5).unwrap().re() - x[0] / T::from(2).unwrap().re(), T::from(1.0).unwrap().re() - x[1] / T::from(2).unwrap().re()],
                |x: &[TReal]| [x[1] / T::from(2).unwrap().re(), T::from(1.0).unwrap().re() - x[0] / T::from(2).unwrap().re() - x[1] / T::from(2).unwrap().re()],
                |x: &[TReal]| [x[1] / T::from(2).unwrap().re(), T::from(0.5).unwrap().re() - x[0] / T::from(2).unwrap().re()],
            ],
            _ => { panic!("Unsupported cell type: {ct:?}"); }
        };

        let coarse_e = coarse_space.family().element(*ct);
        for cell in grid.coarse_grid().cell_iter() {
            if cell.entity_type() == *ct {
                let coarse_cell_dofs = coarse_space.cell_dofs(cell.local_index());
                for (fine_cell, map) in izip!(grid.children(cell.local_index()), &child_to_parent_maps) {
                    let fine_cell_dofs = fine_space.cell_dofs(*fine_cell);
                    for (dim, (pts_list, wts_list)) in izip!(fine_e.interpolation_points(), fine_e.interpolation_weights()).enumerate() {
                        for (entity_i, (pts, wts)) in izip!(pts_list, wts_list).enumerate() {
                            if pts.shape()[1] > 0 {
                                let fine_entity_dofs = fine_e.entity_dofs(dim, entity_i).unwrap().iter().map(|i| fine_cell_dofs[*i]).collect::<Vec<_>>();

                                let mut mapped_pts = rlst_dynamic_array2!(T::Real, pts.shape());
                                for i in 0..pts.shape()[1] {
                                    [
                                        *mapped_pts.get_mut([0, i]).unwrap(),
                                        *mapped_pts.get_mut([1, i]).unwrap(),
                                    ] = map(&pts.r().slice(1, i).data());
                                }
                                let mut table = rlst_dynamic_array4!(T, coarse_e.tabulate_array_shape(0, pts.shape()[1]));
                                coarse_e.tabulate(&mapped_pts, 0, &mut table);
                                println!("{} {fine_cell}, {coarse_cell_dofs:?} {fine_entity_dofs:?} {:?}", cell.local_index(), mapped_pts.data());
                                for (coarse_dof_i, coarse_dof) in coarse_cell_dofs.iter().enumerate() {
                                    for (fine_dof_i, fine_dof) in fine_entity_dofs.iter().enumerate() {
                                        let value: T =                                         (0..table.shape()[1]).map(|pt_i|
                                            (0..coarse_e.value_size()).map(|vs|
                                                *wts.get([fine_dof_i, vs, pt_i]).unwrap() * *table.get([0, pt_i, coarse_dof_i, vs]).unwrap()
                                            ).sum()
                                        ).sum();
                                        if value.abs() > T::from(1e-10).unwrap().re() {
                                            if *coarse_dof == 0 {
                                            print!("  c[{coarse_dof},{fine_dof}] = ");
                                            for vs in 0..coarse_e.value_size() {
                                                for pt_i in 0..table.shape()[1] {
                                                    print!("{} * {}    ", *wts.get([coarse_dof_i, vs, pt_i]).unwrap(), *table.get([0, pt_i, fine_dof_i, vs]).unwrap());
                                                }
                                            }
                                            println!();
                                            }
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
    }

    for c in &coefficients {
        println!("{c:?}");
    }
    (grid, coefficients)
}

#[cfg(test)]
mod test {
    use super::*;
    use ndgrid::shapes;
    use ndelement::{ciarlet::LagrangeElementFamily, types::Continuity};
    use crate::{assemble_mass_matrix, assemble_mass_matrix_dual};
    use rlst::Shape;
    use approx::*;

    #[test]
    fn test_barycentric_representation_lagrange_triangles() {
        let grid = shapes::regular_sphere::<f64>(1);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let space = FunctionSpace::new(&grid, &family);

        let (rgrid, coefficients) = barycentric_representation(&space);
        let fine_space = FunctionSpace::new(rgrid.fine_grid(), &family);
        let bary_space = DualSpace::new(&rgrid, &fine_space, coefficients);

        let result = assemble_mass_matrix(&space, &space);
        let bary_result = assemble_mass_matrix_dual(&bary_space, &bary_space);

        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                assert_relative_eq!(result[[i, j]], bary_result[[i, j]], epsilon=1e-10);
            }
        }
    }

    #[test]
    fn test_barycentric_representation_lagrange_quads() {
        let grid = shapes::screen_quadrilaterals::<f64>(1);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let space = FunctionSpace::new(&grid, &family);

        let (rgrid, coefficients) = barycentric_representation(&space);
        let fine_space = FunctionSpace::new(rgrid.fine_grid(), &family);
        let bary_space = DualSpace::new(&rgrid, &fine_space, coefficients);

        let result = assemble_mass_matrix(&space, &space);
        let bary_result = assemble_mass_matrix_dual(&bary_space, &bary_space);

        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                assert_relative_eq!(result[[i, j]], bary_result[[i, j]], epsilon=1e-10);
            }
        }
    }
}
