use ndelement::{
    ciarlet::LagrangeElementFamily,
    types::{Continuity, ReferenceCellType},
};
use ndelement_dual::{FunctionSpace, assemble_mass_matrix};
use ndgrid::{shapes::regular_sphere, traits::Grid};
use rlst::Shape;

fn main() {
    for i in 0..5 {
        let grid = regular_sphere::<f64>(i);
        let family = LagrangeElementFamily::<f64>::new(1, Continuity::Standard);
        let space = FunctionSpace::new(&grid, &family);
        let matrix = assemble_mass_matrix(&space, &space);

        let mut svals = vec![0.0; matrix.shape()[0]];
        matrix.into_singular_values_alloc(svals.as_mut()).unwrap();

        println!(
            "Number of cells:  {}",
            grid.entity_count(ReferenceCellType::Triangle)
        );
        println!("Condition number: {}", svals[0] / svals[svals.len() - 1]);
        println!();
    }
}
