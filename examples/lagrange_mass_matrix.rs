use ndelement::{
    ciarlet::LagrangeElementFamily,
    types::{Continuity, ReferenceCellType},
};
use ndelement_dual::assemble_mass_matrix;
use ndfunctionspace::FunctionSpaceImpl;
use ndgrid::{shapes::regular_sphere, traits::Grid};
use rlst::SingularValueDecomposition;

fn main() {
    for i in 0..5 {
        let grid = regular_sphere::<f64>(i);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let space = FunctionSpaceImpl::new(&grid, &family);
        let matrix = assemble_mass_matrix(&space, &space);

        let svals = matrix.singular_values().unwrap();

        println!(
            "Number of cells:  {}",
            grid.entity_count(ReferenceCellType::Triangle)
        );
        println!(
            "Condition number: {}",
            svals[[0]] / svals[[svals.len() - 1]]
        );
        println!();
    }
}
