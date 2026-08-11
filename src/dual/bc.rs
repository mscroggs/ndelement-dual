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
    continuity: Continuity,
) -> Vec<HashMap<usize, T>> {
    let fine_mesh = refined_mesh.fine_mesh();
    let coarse_mesh = refined_mesh.coarse_mesh();

    assert_eq!(coarse_mesh.topology_dim(), 2);
    assert_eq!(fine_mesh.entity_types(2).len(), 1);
    assert_eq!(fine_mesh.entity_types(2)[0], ReferenceCellType::Triangle);

    let mut coeffs = vec![];
    for edge in coarse_mesh.entity_iter(ReferenceCellType::Interval) {
        let edge_point = refined_mesh.fine_vertex(ReferenceCellType::Interval, edge.local_index());

        let mut c = HashMap::new();
        for coarse_v_index in edge.topology().sub_entity_iter(ReferenceCellType::Point) {
            let fine_v_index = refined_mesh.fine_vertex(ReferenceCellType::Point, coarse_v_index);
            let fine_v = fine_mesh
                .entity(ReferenceCellType::Point, fine_v_index)
                .unwrap();

            let mut fine_faces = fine_v
                .topology()
                .connected_entity_iter(ReferenceCellType::Triangle)
                .collect::<Vec<_>>();

            let fine_edges = fine_v
                .topology()
                .connected_entity_iter(ReferenceCellType::Interval)
                .collect::<Vec<_>>();

            let mut ordered_edges = vec![];
            let mut next_edge = *fine_edges
                .iter()
                .find(|i| {
                    let vs = fine_mesh
                        .entity(ReferenceCellType::Interval, **i)
                        .unwrap()
                        .topology()
                        .sub_entity_iter(ReferenceCellType::Point)
                        .collect::<Vec<_>>();
                    vs.contains(&edge_point)
                })
                .unwrap();

            let mut first_face = None;
            let mut last_face = None;

            while !fine_faces.is_empty() {
                ordered_edges.push(next_edge);
                let mut used = None;
                for (i, face_index) in fine_faces.iter().enumerate() {
                    let face = fine_mesh
                        .entity(ReferenceCellType::Triangle, *face_index)
                        .unwrap();
                    let edges = face
                        .topology()
                        .sub_entity_iter(ReferenceCellType::Interval)
                        .collect::<Vec<_>>();
                    if edges.contains(&next_edge) {
                        for e in &edges {
                            if fine_edges.contains(e) && *e != next_edge {
                                if first_face.is_none() {
                                    first_face = Some(face);
                                } else {
                                    last_face = Some(face);
                                }
                                next_edge = *e;
                                break;
                            }
                        }
                        used = Some(i);
                        break;
                    }
                }
                fine_faces.remove(used.unwrap());
            }

            let total = ordered_edges.len();
            let mut n = total as i32 / 2 - 1;

            for e in ordered_edges.iter().skip(1) {
                if n != 0 {
                    let v = fine_mesh
                        .entity(ReferenceCellType::Interval, *e)
                        .unwrap()
                        .topology()
                        .sub_entity_iter(ReferenceCellType::Point)
                        .find(|i| *i != fine_v_index)
                        .unwrap();
                    let edge_sign = if v > fine_v_index {
                        T::one()
                    } else {
                        -T::one()
                    };
                    let edofs = fine_space
                        .entity_dofs(ReferenceCellType::Interval, *e)
                        .unwrap();
                    c.insert(
                        edofs[0],
                        edge_sign * T::from(n).unwrap() / T::from(total).unwrap(),
                    );
                }
                n -= 1;
            }

            for e in first_face
                .unwrap()
                .topology()
                .sub_entity_iter(ReferenceCellType::Interval)
            {
                let vs = fine_mesh
                    .entity(ReferenceCellType::Interval, e)
                    .unwrap()
                    .topology()
                    .sub_entity_iter(ReferenceCellType::Point)
                    .collect::<Vec<_>>();
                if !vs.contains(&fine_v_index) {
                    let v = if vs[0] == edge_point { vs[1] } else { vs[0] };
                    let edge_sign = if v > edge_point { T::one() } else { -T::one() };
                    let edofs = fine_space
                        .entity_dofs(ReferenceCellType::Interval, e)
                        .unwrap();
                    c.insert(edofs[0], edge_sign * T::from(0.5).unwrap());
                    break;
                }
            }
            for e in last_face
                .unwrap()
                .topology()
                .sub_entity_iter(ReferenceCellType::Interval)
            {
                let vs = fine_mesh
                    .entity(ReferenceCellType::Interval, e)
                    .unwrap()
                    .topology()
                    .sub_entity_iter(ReferenceCellType::Point)
                    .collect::<Vec<_>>();
                if !vs.contains(&fine_v_index) {
                    let v = if vs[0] == edge_point { vs[1] } else { vs[0] };
                    let edge_sign = if v > edge_point { T::one() } else { -T::one() };
                    let edofs = fine_space
                        .entity_dofs(ReferenceCellType::Interval, e)
                        .unwrap();
                    c.insert(edofs[0], edge_sign * T::from(-0.5).unwrap());
                    break;
                }
            }

            if continuity == Continuity::Discontinuous {
                coeffs.push(c);
                c = HashMap::new();
            }
        }
        if continuity == Continuity::Standard {
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
