//! Refined mesh
use itertools::izip;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{
    SingleElementMesh, SingleElementMeshBuilder,
    traits::{Builder, Entity, Geometry, Mesh, Point, Topology},
    types::Scalar,
};
use std::collections::HashMap;

/// A mesh and its barcentric refinement
pub struct RefinedMesh<
    'a,
    T: Scalar,
    G: Mesh<T = T, EntityDescriptor = ReferenceCellType>,
    FineG: Mesh<T = T, EntityDescriptor = ReferenceCellType>,
> {
    mesh: &'a G,
    bmesh: FineG,
    child_map: Vec<Vec<usize>>,
    parent_map: Vec<(usize, usize)>,
    fine_vertices: HashMap<ReferenceCellType, Vec<usize>>,
    coarse_vertices: Vec<Option<usize>>,
}

impl<
    'a,
    T: Scalar,
    G: Mesh<T = T, EntityDescriptor = ReferenceCellType>,
    FineG: Mesh<T = T, EntityDescriptor = ReferenceCellType>,
> RefinedMesh<'a, T, G, FineG>
{
    /// Coarse unrefined mesh
    pub fn coarse_mesh(&self) -> &'a G {
        self.mesh
    }

    /// Barycentrically refined mesh
    pub fn fine_mesh(&self) -> &FineG {
        &self.bmesh
    }

    /// Indices of cells in fine mesh that make up a coarse cell
    pub fn children(&self, coarse_cell_index: usize) -> &[usize] {
        &self.child_map[coarse_cell_index]
    }

    /// Index of cell in coarse mesh that contains a fine cell
    pub fn parent(&self, fine_cell_index: usize) -> (usize, usize) {
        self.parent_map[fine_cell_index]
    }

    /// Index of vertex in fine mesh that is at the midpoint of an entity
    pub fn fine_vertex(&self, entity_type: ReferenceCellType, entity_index: usize) -> usize {
        self.fine_vertices[&entity_type][entity_index]
    }

    /// Index of vertex in coarse mesh that coincides with fine mesh vertex,
    /// or None if there is no such vertex
    pub fn coarse_vertex(&self, fine_vertex_index: usize) -> Option<usize> {
        self.coarse_vertices[fine_vertex_index]
    }
}

impl<'a, T: Scalar, G: Mesh<T = T, EntityDescriptor = ReferenceCellType>>
    RefinedMesh<'a, T, G, SingleElementMesh<T, CiarletElement<T, IdentityMap, T>>>
{
    /// Barycentrically refine a mesh
    pub fn new(mesh: &'a G) -> Self {
        if mesh.topology_dim() != 2 {
            panic!(
                "Barycentric refinement only implemented for meshes with topological dimension 2."
            );
        }

        let nv = mesh.entity_count(ReferenceCellType::Point);

        // TODO: what if element degree > 1
        let mut b = SingleElementMeshBuilder::<T>::new(
            mesh.geometry_dim(),
            (ReferenceCellType::Triangle, 1),
        );

        let mut child_map = vec![];
        let mut parent_map = vec![];
        for _ in 0..mesh.cell_count() {
            child_map.push(vec![]);
        }

        let mut fine_vertices = HashMap::new();
        for d in 0..=mesh.topology_dim() {
            for etype in mesh.entity_types(d) {
                fine_vertices.insert(*etype, vec![0; mesh.entity_count(*etype)]);
            }
        }
        let mut coarse_vertices = vec![];

        let mut vertex_i = 0;
        let mut p = vec![T::zero(); mesh.geometry_dim()];
        for v in mesh.entity_iter(ReferenceCellType::Point) {
            v.geometry().points().next().unwrap().coords(&mut p);
            b.add_point(vertex_i, &p);
            fine_vertices.get_mut(&ReferenceCellType::Point).unwrap()[v.local_index()] = vertex_i;
            coarse_vertices.push(Some(v.local_index()));
            vertex_i += 1;
        }
        let mut q = vec![T::zero(); mesh.geometry_dim()];
        let mut r = vec![T::zero(); mesh.geometry_dim()];
        for e in mesh.entity_iter(ReferenceCellType::Interval) {
            let g = e.geometry();
            let mut pts = g.points();
            pts.next().unwrap().coords(&mut p);
            pts.next().unwrap().coords(&mut q);
            for (ri, pi, qi) in izip!(&mut r, &p, &q) {
                *ri = (*pi + *qi) / T::from(2).unwrap();
            }
            b.add_point(vertex_i, &r);
            fine_vertices.get_mut(&ReferenceCellType::Interval).unwrap()[e.local_index()] =
                vertex_i;
            coarse_vertices.push(None);
            vertex_i += 1;
        }

        let mut s = vec![T::zero(); mesh.geometry_dim()];
        for (fi, f) in mesh.entity_iter(ReferenceCellType::Triangle).enumerate() {
            let g = f.geometry();
            let mut pts = g.points();
            pts.next().unwrap().coords(&mut p);
            pts.next().unwrap().coords(&mut q);
            pts.next().unwrap().coords(&mut r);
            for (si, pi, qi, ri) in izip!(&mut s, &p, &q, &r) {
                *si = (*pi + *qi + *ri) / T::from(3).unwrap();
            }
            b.add_point(vertex_i, &s);
            fine_vertices.get_mut(&ReferenceCellType::Triangle).unwrap()[f.local_index()] =
                vertex_i;
            coarse_vertices.push(None);

            let t = f.topology();
            let vertices = t
                .sub_entity_iter(ReferenceCellType::Point)
                .collect::<Vec<_>>();
            let edges = t
                .sub_entity_iter(ReferenceCellType::Interval)
                .collect::<Vec<_>>();
            b.add_cell(6 * fi, &[vertices[0], nv + edges[0], vertex_i]);
            b.add_cell(6 * fi + 1, &[nv + edges[0], vertices[1], vertex_i]);
            b.add_cell(6 * fi + 2, &[vertices[1], nv + edges[2], vertex_i]);
            b.add_cell(6 * fi + 3, &[nv + edges[2], vertices[2], vertex_i]);
            b.add_cell(6 * fi + 4, &[vertices[2], nv + edges[1], vertex_i]);
            b.add_cell(6 * fi + 5, &[nv + edges[1], vertices[0], vertex_i]);
            child_map[f.local_index()] = (0..6).map(|i| 6 * fi + i).collect::<Vec<_>>();
            for i in 0..6 {
                parent_map.push((f.local_index(), i));
            }
            vertex_i += 1;
        }

        for (fi, f) in mesh
            .entity_iter(ReferenceCellType::Quadrilateral)
            .enumerate()
        {
            let g = f.geometry();
            let mut pts = g.points();
            pts.next().unwrap().coords(&mut p);
            pts.next();
            pts.next();
            pts.next().unwrap().coords(&mut q);
            for (si, pi, qi) in izip!(&mut s, &p, &q) {
                *si = (*pi + *qi) / T::from(2).unwrap();
            }
            b.add_point(vertex_i, &s);
            fine_vertices
                .get_mut(&ReferenceCellType::Quadrilateral)
                .unwrap()[f.local_index()] = vertex_i;
            coarse_vertices.push(None);

            let t = f.topology();
            let vertices = t
                .sub_entity_iter(ReferenceCellType::Point)
                .collect::<Vec<_>>();
            let edges = t
                .sub_entity_iter(ReferenceCellType::Interval)
                .collect::<Vec<_>>();

            b.add_cell(8 * fi, &[vertices[0], nv + edges[0], vertex_i]);
            b.add_cell(8 * fi + 1, &[nv + edges[0], vertices[1], vertex_i]);
            b.add_cell(8 * fi + 2, &[vertices[1], nv + edges[2], vertex_i]);
            b.add_cell(8 * fi + 3, &[nv + edges[2], vertices[3], vertex_i]);
            b.add_cell(8 * fi + 4, &[vertices[3], nv + edges[3], vertex_i]);
            b.add_cell(8 * fi + 5, &[nv + edges[3], vertices[2], vertex_i]);
            b.add_cell(8 * fi + 6, &[vertices[2], nv + edges[1], vertex_i]);
            b.add_cell(8 * fi + 7, &[nv + edges[1], vertices[0], vertex_i]);
            child_map[f.local_index()] = (0..8).map(|i| 8 * fi + i).collect::<Vec<_>>();
            for i in 0..8 {
                parent_map.push((f.local_index(), i));
            }
            vertex_i += 1;
        }

        Self {
            mesh,
            bmesh: b.create_mesh(),
            child_map,
            parent_map,
            coarse_vertices,
            fine_vertices,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndmesh::shapes::unit_cube_boundary;

    #[test]
    fn test_refine_triangle() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle, 1);
        let bmesh = RefinedMesh::new(&mesh);
        assert_eq!(mesh.cell_count(), bmesh.coarse_mesh().cell_count());
        assert_eq!(mesh.cell_count() * 6, bmesh.fine_mesh().cell_count());
    }

    #[test]
    fn test_refine_quadrilateral() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Quadrilateral, 1);
        let bmesh = RefinedMesh::new(&mesh);
        assert_eq!(mesh.cell_count(), bmesh.coarse_mesh().cell_count());
        assert_eq!(mesh.cell_count() * 8, bmesh.fine_mesh().cell_count());
    }
}
