use approx::assert_relative_eq;
use itertools::izip;
use ndelement::{
    ciarlet::{NedelecFirstKindElementFamily, RaviartThomasElementFamily},
    types::{Continuity, ReferenceCellType},
};
use ndelement_dual::{
    DualSpace, RefinedMesh, assemble_mass_matrix, assemble_mass_matrix_dual,
    barycentric_representation_coefficients, bc_coefficients,
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
            let mesh = regular_sphere::<f64>(i, ct);
            println!("Number of cells:  {}", mesh.entity_count(ct));

            let rt = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
            let nc = NedelecFirstKindElementFamily::<f64>::new(1, Continuity::Standard);

            // RT-NC
            let rt_space = FunctionSpaceImpl::new(&mesh, &rt);
            let nc_space = FunctionSpaceImpl::new(&mesh, &nc);
            let matrix = assemble_mass_matrix(&rt_space, &nc_space);

            let svals = matrix.singular_values().unwrap();

            if ct == ReferenceCellType::Triangle && i == 0 {
                for (a, b) in izip!(
                    svals.iter_value(),
                    [
                        0.9428090415820631,
                        0.942809041582063,
                        0.9428090415820629,
                        0.9428090415820629,
                        0.9428090415820628,
                        0.9428090415820626,
                        5.193434761422128e-16,
                        3.617058416257079e-16,
                        2.329968774872454e-16,
                        2.010394653368211e-16,
                        5.1211811170143e-17,
                        4.411965373903633e-17,
                    ]
                ) {
                    assert_relative_eq!(a, b / 2.0, epsilon = 1e-10);
                }
            }

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

            if ct == ReferenceCellType::Triangle && i == 0 {
                for (a, b) in izip!(
                    svals.iter_value(),
                    [
                        0.9428090415820628,
                        0.9428090415820626,
                        0.9428090415820621,
                        0.8642416214502241,
                        0.864241621450224,
                        0.864241621450224,
                        0.6285393610547082,
                        0.6285393610547081,
                        0.6285393610547079,
                        0.5892556509887891,
                        0.589255650988789,
                        0.47140452079103085,
                    ]
                ) {
                    assert_relative_eq!(a, b / f64::sqrt(2.0), epsilon = 1e-10);
                }
            }

            println!(
                "Condition number (RT-RBC): {}",
                svals[[0]] / svals[[svals.len() - 1]]
            );
            println!();
        }
    }
}
