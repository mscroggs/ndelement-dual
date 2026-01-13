use ndelement::{
    ciarlet::{NedelecFirstKindElementFamily, RaviartThomasElementFamily},
    types::{Continuity, ReferenceCellType},
};
use ndelement_dual::assemble_mass_matrix;
use ndfunctionspace::FunctionSpaceImpl;
use ndgrid::{shapes::regular_sphere, traits::Grid};
use rlst::{Shape, SingularValueDecomposition};

fn main() {
    for i in 0..5 {
        let grid = regular_sphere::<f64>(i);
        let rt = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
        let nc = NedelecFirstKindElementFamily::<f64>::new(1, Continuity::Standard);

        let rt_space = FunctionSpaceImpl::new(&grid, &rt);
        let nc_space = FunctionSpaceImpl::new(&grid, &nc);
        let matrix = assemble_mass_matrix(&rt_space, &nc_space);

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
