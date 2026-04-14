use ndelement::{
    ciarlet::{LagrangeElementFamily, LagrangeVariant},
    types::{Continuity, ReferenceCellType},
};
use ndelement_dual::assemble_mass_matrix;
use ndfunctionspace::FunctionSpaceImpl;
use ndmesh::{shapes::regular_sphere, traits::Mesh};
use rlst::SingularValueDecomposition;

fn main() {
    for i in 0..4 {
        let mesh = regular_sphere::<f64>(i, ReferenceCellType::Triangle);
        let family =
            LagrangeElementFamily::<f64>::new(1, Continuity::Standard, LagrangeVariant::Equispaced);
        let space = FunctionSpaceImpl::new(&mesh, &family);
        let matrix = assemble_mass_matrix(&space, &space);

        let svals = matrix.singular_values().unwrap();

        println!(
            "Number of cells:  {}",
            mesh.entity_count(ReferenceCellType::Triangle)
        );
        println!(
            "Condition number: {}",
            svals[[0]] / svals[[svals.len() - 1]]
        );
        println!();
    }
}
