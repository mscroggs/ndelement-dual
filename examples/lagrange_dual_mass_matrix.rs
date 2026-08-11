use ndelement::{
    ciarlet::{LagrangeElementFamily, LagrangeVariant},
    types::{Continuity, ReferenceCellType},
};
use ndelement_dual::{
    DualSpace, RefinedMesh, assemble_mass_matrix, assemble_mass_matrix_dual,
    barycentric_representation_coefficients, dual0_coefficients,
};
use ndfunctionspace::FunctionSpaceImpl;
use ndmesh::{shapes::regular_sphere, traits::Mesh};
use rlst::SingularValueDecomposition;

fn main() {
    for i in 0..4 {
        for ct in [
            ReferenceCellType::Triangle,
            ReferenceCellType::Quadrilateral,
        ] {
            println!("{ct:?}");
            let mesh = regular_sphere::<f64>(i, ct, 1);
            println!("Number of cells:  {}", mesh.entity_count(ct));

            let p1 = LagrangeElementFamily::<f64>::new(
                1,
                Continuity::Standard,
                LagrangeVariant::Equispaced,
            );
            let dp0 = LagrangeElementFamily::<f64>::new(
                0,
                Continuity::Discontinuous,
                LagrangeVariant::Equispaced,
            );

            // P1-DP0
            let p1_space = FunctionSpaceImpl::new(&mesh, &p1);
            let dp0_space = FunctionSpaceImpl::new(&mesh, &dp0);
            let matrix = assemble_mass_matrix(&p1_space, &dp0_space);

            let svals = matrix.singular_values().unwrap();

            println!(
                "Condition number (P1-DP0): {}",
                svals[[0]] / svals[[svals.len() - 1]]
            );

            // P1-DUAL
            let rmesh = RefinedMesh::new(&mesh);

            let fine_dp0_space = FunctionSpaceImpl::new(rmesh.fine_mesh(), &dp0);
            let dual_space = DualSpace::new(
                &rmesh,
                &fine_dp0_space,
                dual0_coefficients(&rmesh, &fine_dp0_space, Continuity::Standard),
            );

            let coarse_p1_space = FunctionSpaceImpl::new(&mesh, &p1);
            let fine_p1_space = FunctionSpaceImpl::new(rmesh.fine_mesh(), &p1);
            let p1_space = DualSpace::new(
                &rmesh,
                &fine_p1_space,
                barycentric_representation_coefficients(&rmesh, &coarse_p1_space, &fine_p1_space),
            );

            let matrix = assemble_mass_matrix_dual(&p1_space, &dual_space);

            let svals = matrix.singular_values().unwrap();

            println!(
                "Condition number (P1-DUAL0): {}",
                svals[[0]] / svals[[svals.len() - 1]]
            );
            println!();
        }
    }
}
