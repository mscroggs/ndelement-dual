use ndelement::{
    ciarlet::{NedelecFirstKindElementFamily, RaviartThomasElementFamily},
    types::{Continuity, ReferenceCellType},
};
use ndelement_dual::{
    DualSpace, RefinedMesh, assemble_mass_matrix, assemble_mass_matrix_dual,
    barycentric_representation_coefficients, bc_coefficients,
};
use ndfunctionspace::FunctionSpaceImpl;
use ndmesh::traits::Mesh;
use rlst::SingularValueDecomposition;

fn main() {
    for i in 0..4 {
        for ct in [
            ReferenceCellType::Triangle,
            ReferenceCellType::Quadrilateral,
        ] {
            println!("{ct:?}");
            let n = usize::pow(2, i);
            let mesh = ndmesh::shapes::unit_cube_boundary::<f64>(n, n, n, ct);
            println!("Number of cells:  {}", mesh.entity_count(ct));

            let rt = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
            let nc = NedelecFirstKindElementFamily::<f64>::new(1, Continuity::Standard);

            // RT-NC
            let rt_space = FunctionSpaceImpl::new(&mesh, &rt);
            let nc_space = FunctionSpaceImpl::new(&mesh, &nc);
            let matrix = assemble_mass_matrix(&rt_space, &nc_space);

            let svals = matrix.singular_values().unwrap();

            println!(
                "Condition number (RT-NC): {}",
                svals[[0]] / svals[[svals.len() - 1]]
            );

            // RT-RBC
            let rmesh = RefinedMesh::new(&mesh);

            let fine_nc_space = FunctionSpaceImpl::new(rmesh.fine_mesh(), &nc);
            let rbc_space = DualSpace::new(
                &rmesh,
                &fine_nc_space,
                bc_coefficients(&rmesh, &fine_nc_space, Continuity::Standard),
            );

            let coarse_rt_space = FunctionSpaceImpl::new(&mesh, &rt);
            let fine_rt_space = FunctionSpaceImpl::new(rmesh.fine_mesh(), &rt);
            let rt_space = DualSpace::new(
                &rmesh,
                &fine_rt_space,
                barycentric_representation_coefficients(&rmesh, &coarse_rt_space, &fine_rt_space),
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
}
