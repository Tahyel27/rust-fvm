use std::{fs::OpenOptions, io::{Read, Write}, path::Path};

use crate::{FVGeometry, fv::{FVBoundary, FVData, FVDomain, FVFields, FVParams}, fvgeometry};
use crate::fvgeometry::FVShape;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
enum Field {
    Zero,
    LevelSet(f64),
    Array(Vec<f64>)
}

#[derive(Serialize, Deserialize)]
struct SWData {
    h: Field,
    hu: Field,
    hv: Field
}

#[derive(Serialize, Deserialize)]
enum Shape {
    Circle(Circle),
    Rectangle(Rectangle),
    Polygon(Polygon)
}

#[derive(Serialize, Deserialize)]
struct Circle {
    position: (f64, f64), radius: f64
}

#[derive(Serialize, Deserialize)]
struct Rectangle { point: (f64, f64), size: (f64, f64), bevel: f64}

#[derive(Serialize, Deserialize)]
struct Polygon {
    points: Vec<(f64,f64)>,
    bevel: f64
}

#[derive(Default,Serialize, Deserialize)]
struct Geometry {
    shapes: Vec<Shape>
}

#[derive(Serialize, Deserialize)]
pub struct SimulationSave {
    dimensions: (usize, usize),
    dt: f64,
    domain: FVDomain,
    geometry: Geometry,
    data: SWData,
}

impl From<fvgeometry::FVShape> for Shape {
    fn from(value: fvgeometry::FVShape) -> Self {
        match value {
            FVShape::Rectangle(rectangle) => {
                Self::Rectangle(Rectangle { point: (rectangle.point.x, rectangle.point.y),
                    size: (rectangle.size.x1, rectangle.size.x2), bevel: (0.0) })
            },
            FVShape::Circle(circle) => {
                Self::Circle(Circle { position: (circle.p.x, circle.p.y), radius: (circle.r) })
            },
            FVShape::Polygon(polygon) => {
                Self::Polygon(Polygon { points: polygon.points.into_iter().map(|p| (p.x, p.y)).collect(), bevel: (0.0) })
            },
            FVShape::BeveledRect(beveled_rect) => {
                Self::Rectangle(Rectangle { point: (beveled_rect.rect.point.x, beveled_rect.rect.point.y),
                    size: (beveled_rect.rect.size.x1, beveled_rect.rect.size.x2), bevel: (beveled_rect.bevel) })
            },
            FVShape::ExpandedPolygon(expanded_polygon) => {
                Self::Polygon(Polygon { 
                    points: expanded_polygon.polygon.points.into_iter().map(|p| (p.x, p.y)).collect(), bevel: (expanded_polygon.expand) })
            },
        }
    }
}

impl Geometry {
    fn from_fv(geometry: &FVGeometry) -> Self {
        Self { shapes: geometry.shape_iterator().map(|s| s.clone().into()).collect() }
    }

    fn to_fv(self) -> FVGeometry {
        let mut geom = FVGeometry::new();
        
        for shape in self.shapes.into_iter() {
            match shape {
                Shape::Circle(circle) => geom.add_circle(circle.position, circle.radius),
                Shape::Rectangle(rectangle) => geom.add_beveled_rect(rectangle.point.into(), rectangle.point.into(), rectangle.bevel),
                Shape::Polygon(polygon) => {
                    let shape_vec = polygon.points.into_iter().map(|p| p.into()).collect();

                    geom.add_expanded_polygon(shape_vec, polygon.bevel);
                },
            }
        }

        geom
    }
}

impl Field {
    fn to_ndarray(self, dim: (usize, usize)) -> Array2<f64> {
        match self {
            Field::Zero => Array2::<f64>::zeros(dim),
            Field::LevelSet(v) => Array2::<f64>::from_elem(dim, v),
            Field::Array(items) => Array2::<f64>::from_shape_vec(dim, items).unwrap(),
        }
    }
}

pub fn save_to_file_sw(path: &Path, data: FVFields, dt: f64, domain: FVDomain, geometry: &fvgeometry::FVGeometry) -> Result<(), std::io::Error> {

    let dimensions = data.h.dim();

    let h_v = data.h.into_raw_vec_and_offset();
    let hu_v = data.hu.into_raw_vec_and_offset();
    let hv_v = data.hv.into_raw_vec_and_offset();

    let geometry = Geometry::from_fv(geometry);

    let save = SimulationSave {
        dimensions,
        dt,
        domain,
        geometry,
        data: SWData { h: Field::Array(h_v.0), hu: Field::Array(hu_v.0), hv: Field::Array(hv_v.0) }
    };

    let string = serde_json::to_string_pretty(&save)?;

    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open(path)?;

    file.write_all(string.as_bytes())?;

    Ok(())
}

pub fn save_default_case() {

    let domain = FVDomain {
        top: FVBoundary::Wall,
        bottom: FVBoundary::Wall,
        left: FVBoundary::Wall,
        right: FVBoundary::Wall
    };

    let save = SimulationSave {
        dimensions: (600,600),
        dt: 0.1,
        domain,
        geometry: Default::default(),
        data: SWData { h: Field::LevelSet(1.0), hu: Field::LevelSet(0.0), hv: Field::LevelSet(0.0) }
    };

    let string = serde_json::to_string_pretty(&save).unwrap();

    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open("default.json").unwrap();

    file.write_all(string.as_bytes()).unwrap();
}

pub fn save_windtunnel_case() {
    let domain = FVDomain {
        top: FVBoundary::Open,
        bottom: FVBoundary::Open,
        left: FVBoundary::Inlet(0.3, 0.5),
        right: FVBoundary::Wall
    };

    let mut geometry = FVGeometry::new();
    geometry.add_circle((300.,300.), 30.);

    let save = SimulationSave {
        dimensions: (600,600),
        dt: 0.1,
        domain,
        geometry: Geometry::from_fv(&geometry),
        data: SWData { h: Field::LevelSet(0.3), hu: Field::LevelSet(0.0), hv: Field::LevelSet(0.0) }
    };

    let string = serde_json::to_string_pretty(&save).unwrap();

    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open("windtunnel_circle.json").unwrap();

    file.write_all(string.as_bytes()).unwrap();
}

pub fn open_case_sw(path: &Path) -> Result<(FVData, FVParams, Array2<u8>), std::io::Error> {
    let mut file = OpenOptions::new()
        .read(true)
        .open(path)?;

    let mut string = String::new();

    let _ = file.read_to_string(&mut string)?;

    let save: SimulationSave = serde_json::from_str(&string)?;

    let dim = save.dimensions;

    let domain = save.domain;
    
    let geometry = save.geometry.to_fv();

    let mask = geometry.create_mask(dim);

    let case = FVData::new_case_geom(
        save.data.h.to_ndarray(dim),
        save.data.hu.to_ndarray(dim), 
        save.data.hv.to_ndarray(dim),
        geometry
    );

    let params = FVParams {
        dt: save.dt,
        domain: domain
    };

    Ok((case, params, mask))

}
