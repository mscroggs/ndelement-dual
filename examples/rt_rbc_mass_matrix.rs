use ndelement::{
    ciarlet::{NedelecFirstKindElementFamily, RaviartThomasElementFamily},
    types::{Continuity, ReferenceCellType},
};
use ndelement_dual::{DualSpace, FunctionSpace, assemble_mass_matrix_dual, RefinedGrid,
    barycentric_representation_coefficients, bc_coefficients};
use ndgrid::{shapes::regular_sphere, traits::Grid};
use rlst::Shape;

fn main() {
    for i in 0..5 {
        let grid = regular_sphere::<f64>(i);
        let rgrid = RefinedGrid::new(&grid);

        let nc = NedelecFirstKindElementFamily::<f64>::new(1, Continuity::Standard);
        let fine_nc_space = FunctionSpace::new(rgrid.fine_grid(), &nc);
        let rbc_space = DualSpace::new(&rgrid, &fine_nc_space, bc_coefficients(&rgrid, &fine_nc_space, Continuity::Standard));

        let rt = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
        let coarse_rt_space = FunctionSpace::new(&grid, &rt);
        let fine_rt_space = FunctionSpace::new(rgrid.fine_grid(), &rt);
        let rt_space = DualSpace::new(&rgrid, &fine_rt_space, barycentric_representation_coefficients(&rgrid, &coarse_rt_space, &fine_rt_space));

        let matrix = assemble_mass_matrix_dual(&rt_space, &rbc_space);

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
