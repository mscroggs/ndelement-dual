//! Buffa-Christiansen dual spaces

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

/// Generate the coefficients that define the basis functions of a BC space
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
    _continuity: Continuity,
) -> Vec<HashMap<usize, T>> {
    let fine_mesh = refined_mesh.fine_mesh();
    let coarse_mesh = refined_mesh.coarse_mesh();
    assert_eq!(coarse_mesh.topology_dim(), 2);
    assert_eq!(fine_mesh.entity_types(2).len(), 1);
    assert_eq!(fine_mesh.entity_types(2)[0], ReferenceCellType::Triangle);

    let mut coeffs = vec![];
    for vertex in coarse_mesh.entity_iter(ReferenceCellType::Point) {
        let mut c = HashMap::new();
        let fine_v_index = refined_mesh.fine_vertex(ReferenceCellType::Point, vertex.local_index());
        let fine_v = fine_mesh
            .entity(ReferenceCellType::Point, fine_v_index)
            .unwrap();

        for fine_face in fine_v
            .topology()
            .connected_entity_iter(ReferenceCellType::Triangle)
        {
            let dofs = fine_space
                .entity_dofs(ReferenceCellType::Triangle, fine_face)
                .unwrap();
            c.insert(dofs[0], T::one());
        }
        let n = T::from(c.len()).unwrap();
        for i in c.values_mut() {
            *i /= n;
        }
        coeffs.push(c);
    }
    coeffs
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dual::DualSpace;
    use ndelement::{
        ciarlet::{NedelecFirstKindElementFamily, RaviartThomasElementFamily},
        types::Continuity,
    };
    use ndfunctionspace::FunctionSpaceImpl;
    use ndmesh::shapes;

    #[test]
    fn test_bc_space() {
        let mesh = shapes::regular_sphere::<f64>(1, ReferenceCellType::Triangle, 1);

        let nc = NedelecFirstKindElementFamily::<f64>::new(1, Continuity::Standard);
        let nc_space = FunctionSpaceImpl::new(&mesh, &nc);

        let rmesh = RefinedMesh::new(&mesh);
        let rt = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);
        let fine_space = FunctionSpaceImpl::new(rmesh.fine_mesh(), &rt);
        let bc_space = DualSpace::new(
            &rmesh,
            &fine_space,
            coefficients(&rmesh, &fine_space, Continuity::Standard),
        );
        let dbc_space = DualSpace::new(
            &rmesh,
            &fine_space,
            coefficients(&rmesh, &fine_space, Continuity::Discontinuous),
        );

        assert_eq!(nc_space.process_size(), bc_space.dim());
        assert_eq!(2 * nc_space.process_size(), dbc_space.dim());
    }
}
