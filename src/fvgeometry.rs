use crate::geoprimitives::*;
use ndarray::Array2;

#[derive(Clone)]
pub enum FVShape {
    Rectangle(Rectangle<f64>),
    Circle(Circle<f64>),
    Polygon(Polygon<f64>),
    BeveledRect(BeveledRect<f64>),
    ExpandedPolygon(ExpandedPolygon<f64>)
}

#[derive(Default, Clone)]
pub struct FVGeometry {
    shapes: Vec<FVShape>,
}

#[derive(Default, Clone, Copy)]
pub struct InterpIndex {
    id: [usize; 2],
    fact: f64
}

#[derive(Default)]
pub struct GhostCell {
    interp_coefs: [InterpIndex; 4],
    cell: [usize; 2],
    normal: Vec2D<f64>
}

#[derive(Default)]

/*
geometry: Contains the geometry of the case represented as shapes

cell_types: differentiates fluid cells, ghost cells and wall cells,
    - 0 - fluid cell
    - 1 - wall cell
    - 2 - ghost cell
*/
pub struct FVWalls {
    geometry: FVGeometry,
    cell_types: Array2<u8>,
    dim: (usize, usize),
    ghost_cells: Vec<GhostCell>,
    fmask: Array2<f64>
}

impl FVShape {
    pub fn contains(&self, point: (f64, f64)) -> bool {
        match self {
            FVShape::Rectangle(rectangle) => rectangle.contains(point.into()),
            FVShape::Circle(circle) => circle.distance(point.into()) <= 0.,
            FVShape::Polygon(polygon) => polygon.contains(point.into()),
            FVShape::BeveledRect(beveled_rect) => beveled_rect.contains(point.into()),
            FVShape::ExpandedPolygon(polygon) => polygon.contains(point.into())
        }
    }

    pub fn distance(&self, point: (f64, f64)) -> f64 {
        match self {
            FVShape::Rectangle(rectangle) => rectangle.distance(point.into()),
            FVShape::Circle(circle) => circle.distance(point.into()),
            FVShape::Polygon(polygon) => polygon.distance(point.into()),
            FVShape::BeveledRect(beveled_rect) => beveled_rect.distance(point.into()),
            FVShape::ExpandedPolygon(polygon) => polygon.distance(point.into())
        }
    }

    pub fn normal(&self, point: (f64, f64)) -> Vec2D<f64> {
        match self {
            FVShape::Rectangle(rectangle) => rectangle.normal(point.into()),
            FVShape::Circle(circle) => circle.normal(point.into()),
            FVShape::Polygon(polygon) => polygon.normal(point.into()),
            FVShape::BeveledRect(beveled_rect) => beveled_rect.normal(point.into()),
            FVShape::ExpandedPolygon(polygon) => polygon.normal(point.into())
        }
    }
}

impl FVGeometry {
    pub fn new() -> Self {
        Self { shapes: Default::default() }
    }

    pub fn contains(&self, point: (f64, f64)) -> bool {
        for shape in self.shapes.iter() {
            if shape.contains(point) {
                return true;
            }
        }

        return false;
    }

    pub fn closest_normal(&self, point: (f64, f64)) -> Vec2D<f64> {
        self.shapes.iter()
            .min_by(|shape_a, shape_b| {
                let da = shape_a.distance(point).abs();
                let db = shape_b.distance(point).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
            .normal(point)
    }

    pub fn closest_distance(&self, point: (f64, f64)) -> f64 {
        self.shapes.iter()
            .min_by(|shape_a, shape_b| {
                let da = shape_a.distance(point).abs();
                let db = shape_b.distance(point).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
            .distance(point)
            .abs()
    }

    pub fn add_rectangle(&mut self, point_a: (f64, f64), point_b: (f64, f64)) {
        self.shapes.push(FVShape::Rectangle(Rectangle::from_points(point_a.into(), point_b.into())));
    }

    pub fn add_circle(&mut self, center: (f64, f64), radius: f64) {
        self.shapes.push(FVShape::Circle(Circle::new(center.into(), radius)));
    }

    pub fn add_polygon(&mut self, points: Vec<Point<f64>>) {
        self.shapes.push(FVShape::Polygon(Polygon::new(points)));
    }

    pub fn add_expanded_polygon(&mut self, points: Vec<Point<f64>>, expand: f64) {
        self.shapes.push(FVShape::ExpandedPolygon(ExpandedPolygon::new(points, expand)));
    }

    pub fn add_beveled_rect(&mut self, pointA: (f64, f64), pointB: (f64, f64), bevel: f64) {
        self.shapes.push(FVShape::BeveledRect(BeveledRect::new(pointA.into(), pointB.into(), bevel)));
    }

    pub fn shape_iterator(&self) -> std::slice::Iter<'_, FVShape>  {
        self.shapes.iter()
    }

    pub fn create_mask(&self, dims: (usize, usize)) -> Array2<u8> {
        let mut mask = Array2::<u8>::from_elem(dims, 0);
        
        for ((i, j), m) in mask.indexed_iter_mut() {
            if self.contains((i as f64, j as f64)) {
                *m = 1;
            }
        }

        mask
    }
}

fn find_edge_cells(mask: &mut Array2<u8>) {
    let (dimx, dimy) = mask.dim();
    
    //0 - fluid cell
    //1 - wall cell

    //if a cell is on the edge, its value in the mask is set to 2

    for i in 1..dimx-1 {
        for j in 1..dimy-1 {
            if mask[[i,j]] > 0 {
                if mask[[i-1,j]] == 0 || mask[[i+1,j]] == 0 || mask[[i,j+1]] == 0 || mask[[i, j-1]] == 0 {
                    mask[[i,j]] = 2;
                }
            }
        }
    }
}

fn mirror_point(cell: (usize, usize), normal: Vec2D<f64>, distance: f64) -> Point<f64> {
    let shift = normal * distance * 2.;

    (cell.0 as f64 + shift.x1, cell.1 as f64 + shift.x2).into()
}

impl GhostCell {
    fn create(ghost_cell: (usize, usize), normal: Vec2D<f64>, distance: f64, mask: &Array2<u8>) -> Self {
        //calculate the mirror point
        let mp = mirror_point(ghost_cell, normal, distance);

        //check if it falls on the grid
        if mp.x == mp.x.floor() && mp.y == mp.y.floor() {
            let i = mp.x.floor() as usize;
            let j = mp.y.floor() as usize;
            let interp = InterpIndex{ id: [i, j], fact: 1.0 };
            let interp_0 = InterpIndex{ id: [0, 0], fact: 0.0 };

            return GhostCell{
                interp_coefs: [interp, interp_0, interp_0, interp_0],
                cell: [i, j],
                normal,
            };
        }
        
        //prepare the weights

        //now calculate the interpolation factor from each point 
        let blx = mp.x.floor(); let bly = mp.y.floor();
        let bl: Point<f64> = (blx,bly).into();
        let mut w1 = (1./(bl - mp).norm()).powi(2);
        if blx as usize == ghost_cell.0 && bly as usize == ghost_cell.1 {
            w1 = 0.0;
        }

        let brx = mp.x.ceil(); let bry = mp.y.floor();
        let br: Point<f64> = (brx,bry).into();
        let mut w2 = (1./(br - mp).norm()).powi(2);
        if brx as usize == ghost_cell.0 && bry as usize == ghost_cell.1 {
            w2 = 0.0;
        }

        let tlx = mp.x.floor(); let tly = mp.y.ceil();
        let tl: Point<f64> = (tlx,tly).into();
        let mut w3 = (1./(tl - mp).norm()).powi(2);
        if tlx as usize == ghost_cell.0 && tly as usize == ghost_cell.1 {
            w3 = 0.0;
        }

        let trx = mp.x.ceil(); let try_ = mp.y.ceil();
        let tr: Point<f64> = (trx,try_).into();
        let mut w4 = (1./(tr - mp).norm()).powi(2);
        if trx as usize == ghost_cell.0 && try_ as usize == ghost_cell.1 {
            w4 = 0.0;
        }

        let wsum = w1 + w2 + w3 + w4;
        w1 = w1 / wsum; w2 = w2 / wsum; w3 = w3 / wsum; w4 = w4 / wsum; //w_bp = w_bp / wsum;

        let interp_coefs = [
            InterpIndex{ id: [blx as usize, bly as usize], fact: w1 },
            InterpIndex{ id: [brx as usize, bry as usize], fact: w2 },
            InterpIndex{ id: [tlx as usize, tly as usize], fact: w3 },
            InterpIndex{ id: [trx as usize, try_ as usize], fact: w4 },
        ];

        let cell = [ghost_cell.0, ghost_cell.1];

        Self { interp_coefs, cell, normal }

    }

    //neumann bc for things like pressure, water column height...
    fn apply_to_scalar_field(&self, phi: &mut Array2<f64>) {
        let mut mp = 0.0;
        
        //perform the interpolation
        for id in self.interp_coefs {
            mp += phi[id.id] * id.fact;
        }
        //mp += phi[self.cell] * self.bp_interp;

        phi[self.cell] = mp;
    }

    fn apply_to_velocity_field(&self, u: &mut Array2<f64>, v: &mut Array2<f64>) {
        let mut mpu = 0.0;
        let mut mpv = 0.0;

        for id in self.interp_coefs {
            mpu += u[id.id] * id.fact;
            mpv += v[id.id] * id.fact;
        }
        //mpu += u[self.cell] * self.bp_interp;
        //mpv += u[self.cell] * self.bp_interp;

        let mp = (mpu, mpv).into();
        let normal = self.normal * self.normal.dot(mp);
        let tangent = self.normal.orth() * self.normal.orth().dot(mp);

        let gp = tangent - normal;
        u[self.cell] = gp.x1;
        v[self.cell] = gp.x2;
    }

}

impl FVWalls {
    pub fn new(geometry: FVGeometry, dim: (usize, usize)) -> Self {
        let mut s = Self{ geometry, cell_types: Default::default(), dim, ghost_cells: Default::default(), fmask: Default::default()};
        s.compute_boundaries();
        s
    }

    fn compute_boundaries(&mut self) {
        //compute which cells are ghost and which are not

        //clear data
        self.ghost_cells = Default::default();
        
        //get the raw mask - 0 is fluid, 1 is wall 
        let mut mask = self.geometry.create_mask(self.dim);

        //find the ghost cells
        find_edge_cells(&mut mask);

        //iterate over the ghost cells to fill the vector
        mask.indexed_iter()
            .filter(|((_,_),m)| { **m == 2 })
            .for_each(|((i,j),_)| {
                let normal = self.geometry.closest_normal((i as f64, j as f64).into());
                let distance = self.geometry.closest_distance((i as f64, j as f64).into());

                self.ghost_cells.push(GhostCell::create((i,j), normal, distance, &mask));
            });

        self.fmask = mask.mapv(|v| {
            if v > 0 {
                0.0
            } else {1.0}
        }
        );

        self.cell_types = mask;
    }

    pub fn get_mask(&self) -> &Array2<f64> {
        &self.fmask
    }

    pub fn apply_to_scalar_field(&self, phi: &mut Array2<f64>) {
        for ghost_cell in self.ghost_cells.iter() {
            ghost_cell.apply_to_scalar_field(phi);
        }
    }

    pub fn apply_to_velocity_field(&self, u: &mut Array2<f64>, v: &mut Array2<f64>) {
        for ghost_cell in self.ghost_cells.iter() {
            ghost_cell.apply_to_velocity_field(u, v);
        }
    }
}