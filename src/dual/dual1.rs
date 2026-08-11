//! Piecewise linear dual spaces

use crate::RefinedMesh;
use ndelement::{
    ciarlet::CiarletElement,
    traits::Map,
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::traits::FunctionSpace;
use ndmesh::traits::{Entity, Mesh, Topology};
use ndmesh::types::Scalar;
use std::collections::HashMap;

/// Generate the coefficients that define the basis functions of a DUAL1 space
pub fn coefficients<
    'a,
    TGeo: Scalar,
    T: Scalar,
    G: Mesh<T = TGeo, EntityDescriptor = ReferenceCellType>,
    FineG: Mesh<T = TGeo, EntityDescriptor = ReferenceCellType>,
    M: Map,
>(
    refined_mesh: &'a RefinedMesh<'a, TGeo, G, FineG>,
    fine_space: &impl FunctionSpace<
        EntityDescriptor = ReferenceCellType,
        Mesh = FineG,
        FiniteElement = CiarletElement<T, M, TGeo>,
    >,
    continuity: Continuity,
) -> Vec<HashMap<usize, T>> {
    let fine_mesh = refined_mesh.fine_mesh();
    let coarse_mesh = refined_mesh.coarse_mesh();
    assert_eq!(coarse_mesh.topology_dim(), 2);
    assert_eq!(continuity, Continuity::Standard, "Discontinuous degree 1 dual spaces not supported");
    assert_eq!(fine_mesh.entity_types(2).len(), 1);
    assert_eq!(fine_mesh.entity_types(2)[0], ReferenceCellType::Triangle);

    let mut coeffs = vec![];
    for ct in coarse_mesh.entity_types(2) {
        for coarse_cell in coarse_mesh.entity_iter(*ct) {
            let mut c = HashMap::new();
            let fine_v_index = refined_mesh.fine_vertex(*ct, coarse_cell.local_index());
            let dofs = fine_space
                .entity_dofs(ReferenceCellType::Point, fine_v_index)
                .unwrap();
            c.insert(dofs[0], T::one());

            for coarse_edge_index in coarse_cell.topology().sub_entity_iter(ReferenceCellType::Interval) {
                let fine_v_on_edge = refined_mesh.fine_vertex(ReferenceCellType::Interval, coarse_edge_index);
                let dofs = fine_space
                .entity_dofs(ReferenceCellType::Point, fine_v_on_edge)
                .unwrap();
                c.insert(dofs[0], T::from(0.5).unwrap());
            }

            for coarse_vertex_index in coarse_cell.topology().sub_entity_iter(ReferenceCellType::Point) {
                let coarse_vertex = coarse_mesh.entity(ReferenceCellType::Point, coarse_vertex_index).unwrap();
                let fine_v_at_vertex = refined_mesh.fine_vertex(ReferenceCellType::Point, coarse_vertex_index);
                let dofs = fine_space
                    .entity_dofs(ReferenceCellType::Point, fine_v_at_vertex)
                    .unwrap();
                let ncells = coarse_mesh
                    .entity_types(2)
                    .iter()
                    .map(|t| coarse_vertex.topology().connected_entity_iter(*t).count())
                    .sum::<usize>();
                c.insert(dofs[0], T::one() / T::from(ncells).unwrap());
            }


            // Loop over vertices of coarse cell
                // set coeff to 1/n_fine_triangles at vertices
            dbg!(&c);println!();
            coeffs.push(c);
        }
    }
    coeffs
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dual::DualSpace;
    use ndelement::{
        ciarlet::{LagrangeElementFamily, LagrangeVariant},
        types::Continuity,
    };
    use ndfunctionspace::FunctionSpaceImpl;
    use ndmesh::shapes;

    #[test]
    fn test_dual1_space() {
        let mesh = shapes::regular_sphere::<f64>(1, ReferenceCellType::Triangle, 1);

        let dp0 =
            LagrangeElementFamily::<f64>::new(0, Continuity::Discontinuous, LagrangeVariant::Equispaced);
        let dp0_space = FunctionSpaceImpl::new(&mesh, &dp0);

        let rmesh = RefinedMesh::new(&mesh);
        let p1 = LagrangeElementFamily::<f64>::new(
            1,
            Continuity::Standard,
            LagrangeVariant::Equispaced,
        );
        let fine_space = FunctionSpaceImpl::new(rmesh.fine_mesh(), &p1);
        let dual_space = DualSpace::new(
            &rmesh,
            &fine_space,
            coefficients(&rmesh, &fine_space, Continuity::Standard),
        );

        assert_eq!(dp0_space.process_size(), dual_space.dim());
    }
}
