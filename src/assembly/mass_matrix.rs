//! Max matrix assembly
use rlst::{RlstScalar, rlst_dynamic_array2, RandomAccessMut};
use ndgrid::{traits::{Grid, Entity}, types::Array2D};
use ndelement::{traits::ElementFamily, types::ReferenceCellType};
use super::FunctionSpace;

/// Assemble a mass matrix
pub fn assemble<'a, T: RlstScalar, TestG: Grid<T=T::Real, EntityDescriptor = ReferenceCellType>, TrialG: Grid<T=T::Real, EntityDescriptor = ReferenceCellType>, TestF: ElementFamily<T=T, CellType = ReferenceCellType>, TrialF: ElementFamily<T=T, CellType = ReferenceCellType>>(
    test_space: &FunctionSpace<'a, TestG, TestF>,
    trial_space: &FunctionSpace<'a, TrialG, TrialF>,
) -> Array2D<T> {
    let mut matrix = rlst_dynamic_array2!(T, [test_space.dim(), trial_space.dim()]);
    for test_ct in test_space.grid().cell_types() {
        for trial_ct in trial_space.grid().cell_types() {
            println!("{test_ct:?} {trial_ct:?}");
        }
    }

    for test_cell in test_space.grid().cell_iter() {
        let test_dofs = test_space.cell_dofs(test_cell.local_index());
        for trial_cell in trial_space.grid().cell_iter() {
            let trial_dofs = trial_space.cell_dofs(trial_cell.local_index());
            for test_i in test_dofs {
                for trial_i in trial_dofs {
                    *matrix.get_mut([*test_i, *trial_i]).unwrap() += T::one();
                }
            }
            println!("{:?} {:?}", test_dofs, trial_dofs);
        }
    }
    matrix
}

#[cfg(test)]
mod test {
    use super::*;
    use ndelement::{ciarlet::LagrangeElementFamily, types::Continuity};
    use ndgrid::shapes::regular_sphere;
    use rlst::RawAccess;

    #[test]
    fn test_assembly() {
        let grid = regular_sphere::<f64>(1);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let space = FunctionSpace::new(&grid, &family);
        let result = assemble(&space, &space);

        println!("{:?}", result.data());

        assert_eq!(1,0);
    }
}
