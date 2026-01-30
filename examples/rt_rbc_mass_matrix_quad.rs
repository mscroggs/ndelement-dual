use ndelement::{
    ciarlet::{NedelecFirstKindElementFamily, RaviartThomasElementFamily},
    types::{Continuity, ReferenceCellType},
};
use ndelement_dual::{
    DualSpace, RefinedGrid, assemble_mass_matrix, assemble_mass_matrix_dual,
    barycentric_representation_coefficients, bc_coefficients,
};
use ndfunctionspace::FunctionSpaceImpl;
use ndgrid::{shapes::regular_sphere, traits::Grid};
use rlst::SingularValueDecomposition;

fn main() {
    let ct = ReferenceCellType::Quadrilateral;
    for i in 0..4 {
        //let grid = regular_sphere::<f64>(i);
        let n = usize::pow(2, i);
        let grid = ndgrid::shapes::unit_cube_boundary::<f64>(n, n, n, ct);
        println!("Number of cells:  {}", grid.entity_count(ct));

        let rt = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
        let nc = NedelecFirstKindElementFamily::<f64>::new(1, Continuity::Standard);

        // RT-NC
        let rt_space = FunctionSpaceImpl::new(&grid, &rt);
        let nc_space = FunctionSpaceImpl::new(&grid, &nc);
        let matrix = assemble_mass_matrix(&rt_space, &nc_space);

        let svals = matrix.singular_values().unwrap();

        println!(
            "Condition number (RT-NC): {}",
            svals[[0]] / svals[[svals.len() - 1]]
        );

        // RT-RBC
        let rgrid = RefinedGrid::new(&grid);

        let fine_nc_space = FunctionSpaceImpl::new(rgrid.fine_grid(), &nc);
        let rbc_space = DualSpace::new(
            &rgrid,
            &fine_nc_space,
            bc_coefficients(&rgrid, &fine_nc_space, Continuity::Standard),
        );

        let coarse_rt_space = FunctionSpaceImpl::new(&grid, &rt);
        let fine_rt_space = FunctionSpaceImpl::new(rgrid.fine_grid(), &rt);
        let rt_space = DualSpace::new(
            &rgrid,
            &fine_rt_space,
            barycentric_representation_coefficients(&rgrid, &coarse_rt_space, &fine_rt_space),
        );

        let matrix = assemble_mass_matrix_dual(&rt_space, &rbc_space);

        let svals = matrix.singular_values().unwrap();

        println!(
            "Condition number (RT-RBC): {}",
            svals[[0]] / svals[[svals.len() - 1]]
        );
        println!();
    }
}
