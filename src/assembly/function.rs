use ndelement::{
    traits::{ElementFamily, FiniteElement},
    types::ReferenceCellType,
};
use ndgrid::traits::{Entity, Grid, Topology};
use std::collections::HashMap;

/// A function space
pub struct FunctionSpace<
    'a,
    G: Grid<EntityDescriptor = ReferenceCellType>,
    F: ElementFamily<CellType = ReferenceCellType>,
> {
    grid: &'a G,
    family: &'a F,
    cell_dofs: Vec<Vec<usize>>,
    dim: usize,
}

impl<
    'a,
    G: Grid<EntityDescriptor = ReferenceCellType>,
    F: ElementFamily<CellType = ReferenceCellType>,
> FunctionSpace<'a, G, F>
{
    /// Create new
    pub fn new(grid: &'a G, family: &'a F) -> Self {
        let mut dofmap: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut cell_dofs = vec![];
        for _ in 0..grid.cell_count() {
            cell_dofs.push(vec![]);
        }
        let mut dof_n = 0;
        for ct in grid.cell_types() {
            let element = family.element(*ct);

            for cell in grid.cell_iter_by_type(*ct) {
                for d in 0..=grid.topology_dim() {
                    for (i, e) in cell.topology().sub_entity_iter(d).enumerate() {
                        let entity_dofs = element.entity_dofs(d, i).unwrap();
                        if !entity_dofs.is_empty() {
                            for dof in dofmap.entry((d, e)).or_insert_with(|| {
                                dof_n += entity_dofs.len();
                                entity_dofs
                                    .iter()
                                    .enumerate()
                                    .map(|(i, _)| dof_n - entity_dofs.len() + i)
                                    .collect::<Vec<_>>()
                            }) {
                                cell_dofs[cell.local_index()].push(*dof);
                            }
                        }
                    }
                }
            }
        }

        Self {
            grid,
            family,
            cell_dofs,
            dim: dof_n,
        }
    }

    /// The grid
    pub fn grid(&self) -> &'a G {
        self.grid
    }

    /// The element family
    pub fn family(&self) -> &'a F {
        self.family
    }

    /// Space dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Cell dofs
    pub fn cell_dofs(&self, cell_n: usize) -> &[usize] {
        self.cell_dofs.get(cell_n).expect("Cell not found")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndelement::{ciarlet::RaviartThomasElementFamily, types::Continuity};
    use ndgrid::shapes::regular_sphere;

    #[test]
    fn test_function_space() {
        let grid = regular_sphere::<f64>(1);
        let family = RaviartThomasElementFamily::<f64>::new(1, Continuity::Standard);

        let space = FunctionSpace::new(&grid, &family);

        assert_eq!(space.dim(), grid.entity_count(ReferenceCellType::Interval));
    }
}
