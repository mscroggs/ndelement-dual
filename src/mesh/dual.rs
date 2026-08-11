//! Refined mesh

use super::RefinedMesh;
use ndelement::types::ReferenceCellType;
use ndmesh::{
    traits::{Entity, Mesh, Topology},
    types::Scalar,
};

/// A barycentric dual mesh
pub struct DualMesh<
    'a,
    T: Scalar,
    G: Mesh<T = T, EntityDescriptor = ReferenceCellType>,
    FineG: Mesh<T = T, EntityDescriptor = ReferenceCellType>,
> {
    meshes: &'a RefinedMesh<'a, T, G, FineG>,
    subcells: Vec<Vec<usize>>,
}

impl<
    'a,
    T: Scalar,
    G: Mesh<T = T, EntityDescriptor = ReferenceCellType>,
    FineG: Mesh<T = T, EntityDescriptor = ReferenceCellType>,
> DualMesh<'a, T, G, FineG>
{
    /// Create new dual mesh
    pub fn new(meshes: &'a RefinedMesh<'a, T, G, FineG>) -> Self {
        let mut subcells = vec![];
        for _ in meshes.coarse_mesh().entity_iter(ReferenceCellType::Point) {
            subcells.push(vec![]);
        }
        let fine_mesh = meshes.fine_mesh();
        assert_eq!(fine_mesh.entity_types(fine_mesh.topology_dim()).len(), 1);
        for fine_cell in fine_mesh.entity_iter(fine_mesh.entity_types(fine_mesh.topology_dim())[0])
        {
            for v in fine_cell
                .topology()
                .sub_entity_iter(ReferenceCellType::Point)
            {
                if let Some(i) = meshes.coarse_vertex(v) {
                    subcells[i].push(fine_cell.local_index());
                }
            }
        }

        Self { meshes, subcells }
    }

    /// Coarse and fine meshes used to define this dual mesh
    pub fn meshes(&self) -> &'a RefinedMesh<'a, T, G, FineG> {
        self.meshes
    }

    /// Number of cells
    pub fn cell_count(&self) -> usize {
        self.subcells.len()
    }

    /// Sub cells of a dual cell
    pub fn subcells(&self, index: usize) -> &[usize] {
        &self.subcells[index]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndmesh::shapes::unit_cube_boundary;

    #[test]
    fn test_dual_triangle() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Triangle, 1);
        let bmesh = RefinedMesh::new(&mesh);
        let dual_mesh = DualMesh::new(&bmesh);
        assert_eq!(
            dual_mesh.cell_count(),
            bmesh.coarse_mesh().entity_count(ReferenceCellType::Point)
        );
        assert_eq!(
            (0..dual_mesh.cell_count())
                .map(|i| dual_mesh.subcells(i).len())
                .sum::<usize>(),
            bmesh.fine_mesh().entity_count(ReferenceCellType::Triangle)
        );
    }

    #[test]
    fn test_refine_quadrilateral() {
        let mesh = unit_cube_boundary::<f64>(2, 2, 2, ReferenceCellType::Quadrilateral, 1);
        let bmesh = RefinedMesh::new(&mesh);
        let dual_mesh = DualMesh::new(&bmesh);
        assert_eq!(
            dual_mesh.cell_count(),
            bmesh.coarse_mesh().entity_count(ReferenceCellType::Point)
        );
        assert_eq!(
            (0..dual_mesh.cell_count())
                .map(|i| dual_mesh.subcells(i).len())
                .sum::<usize>(),
            bmesh.fine_mesh().entity_count(ReferenceCellType::Triangle)
        );
    }
}
