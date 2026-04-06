use std::ops;

use num::{Float, Integer};
pub trait SDF<T: Float> {
    fn distance(&self, point: Point<T>) -> T;

    fn normal(&self, point: Point<T>) -> Vec2D<T>;
}
#[derive(Default, Clone, Copy)]
pub struct Vec2D<T: Float> {
    pub x1: T,
    pub x2: T
}

#[derive(Clone, Copy)]
pub struct Point<T: Float> {
    pub x: T,
    pub y: T
}

impl<T: Float> Default for Point<T> {
    fn default() -> Self {
        Self { x: T::zero(), y: T::zero() }
    }
}

#[derive(Default)]
pub struct Line<T: Float> {
    point: Point<T>,
    direction: Vec2D<T>
}

#[derive(Default)]
pub struct LineSegment<T: Float> {
    p1: Point<T>,
    p2: Point<T>
}

#[derive(Default, Clone)]
pub struct Rectangle<T: Float> {
    pub point: Point<T>,
    pub size: Vec2D<T>
}

#[derive(Default, Clone)]
pub struct Circle<T: Float> {
    pub r: T,
    pub p: Point<T>
}

#[derive(Default, Clone)]
pub struct Polygon<T: Float> {
    pub points: Vec<Point<T>>
}

pub struct PolygonLineIterator<'a ,T: Float> {
    polygon: &'a Polygon<T>,
    index: usize
}

#[derive(Clone)]
pub struct BeveledRect<T: Float> {
    pub rect: Rectangle<T>,
    pub bevel: T
}

#[derive(Clone)]
pub struct ExpandedPolygon<T: Float> {
    pub polygon: Polygon<T>,
    pub expand: T
}

//ARITHMETIC OF PRIMITIVES

impl<T: Float> ops::Add for Vec2D<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {x1: self.x1 + rhs.x1, x2: self.x2 + rhs.x2}
    }
}

impl<T: Float> ops::Mul<T> for Vec2D<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {x1: self.x1 * rhs, x2: self.x2 * rhs}
    }
}

impl<T: Float> ops::Div<T> for Vec2D<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self {x1: self.x1 / rhs, x2: self.x2 / rhs}
    }
}

impl<T: Float> ops::Sub for Vec2D<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {x1: self.x1 - rhs.x1, x2: self.x2 - rhs.x2}
    }
}

//CONVERSIONS BETWEEN PRIMITIVES

impl<T: Float> From<Point<T>> for Vec2D<T> {
    fn from(value: Point<T>) -> Self {
        Self { x1: value.x, x2: value.y }
    }
}

impl<T: Float> From<Vec2D<T>> for Point<T> {
    fn from(value: Vec2D<T>) -> Self {
        Self { x: value.x1, y: value.x2 }
    }
}

impl<T: Float> From<(T,T)> for Point<T> {
    fn from(value: (T,T)) -> Self {
        Self { x: value.0, y: value.1 }
    }
}

impl<T: Float> From<(T,T)> for Vec2D<T> {
    fn from(value: (T,T)) -> Self {
        Self { x1: value.0, x2: value.1 }
    }
}

impl<T: Float> From<Point<T>> for (T, T) {
    fn from(value: Point<T>) -> Self {
        (value.x, value.y)
    }
}

impl<T: Float> From<Vec2D<T>> for (T, T) {
    fn from(value: Vec2D<T>) -> Self {
        (value.x1, value.x2)
    }
}

impl<T: Float> ops::Sub for Point<T>  {
    type Output = Vec2D<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec2D {x1: self.x - rhs.x, x2: self.y - rhs.y}
    }
}

impl<T: Float> Point<T> {
    pub fn new(x: T, y: T) -> Self { (x,y).into() }
}

//IMPLEMENTATIONS OF PRIMITIVES AND SHAPES

impl<T: Float> Vec2D<T> {
    pub fn norm(self) -> T { (self.x1.powi(2) + self.x2.powi(2)).sqrt() }

    pub fn norm_sq(self) -> T { self.dot(self) }

    pub fn orth(self) -> Self {
        Self {x1: -self.x2, x2: self.x1  }
    }

    pub fn normalize(self) -> Self { self / self.norm()}

    pub fn dot(self, rhs: Self) -> T { self.x1 * rhs.x1 + self.x2 * rhs.x2 }

    //pub fn angle(self, rhs: Self) -> T

}

impl<T: Float> Line<T> {
    pub fn new(point: Point<T>, direction: Vec2D<T> ) -> Self { Self { point, direction: direction.normalize() } }

    pub fn from_unit(point: Point<T>, direction: Vec2D<T>) -> Self { Self { point, direction } }

    pub fn from_normal(point: Point<T>, normal: Vec2D<T> ) -> Self {
        Self { point, direction: normal.orth() }
    }

    pub fn slope(&self) -> T { self.direction.x2 / self.direction.x2 }

    pub fn intercept(&self) -> T { self.point.y - self.point.x * self.slope() }

    pub fn line_params(&self) -> (T, T) {
        let slope = self.slope();
        (slope, self.point.y - self.point.x * slope)
    }
}

impl<T: Float> SDF<T> for Line<T> {
    fn distance(&self, point: Point<T>) -> T {
        let norm = self.direction.orth();

        (norm.dot(point.into()) - norm.dot(self.point.into())).abs()
    }

    fn normal(&self, point: Point<T>) -> Vec2D<T> {
        let norm = self.direction.orth();

        norm * (norm.dot(point.into()) - norm.dot(self.point.into())).signum()
    }
}

impl<T: Float> LineSegment<T> {
    pub fn new(p1: Point<T>, p2: Point<T>) -> Self { 
        Self {p1, p2}
    }

    pub fn normal_vect(&self) -> Vec2D<T> { (self.p2 - self.p1).orth().normalize() }

    pub fn slope(&self) -> T { (self.p2.y - self.p1.y)/(self.p2.x - self.p1.x) }

    pub fn intercept(&self) -> T { self.p1.y - self.p1.x * self.slope() }

    pub fn line_params(&self) -> (T, T) {
        let slope = self.slope();
        (slope, self.p1.y - self.p1.x * slope)
    }

    pub fn length(&self) -> T { (self.p2 - self.p1).norm() }

    pub fn is_close(&self, point: Point<T>) -> bool {
        let lengthsq = (self.p2 - self.p1).norm_sq();
        if (point - self.p1).norm_sq() < lengthsq || (point - self.p2).norm_sq() < lengthsq {
            true
        }
        else {
            false
        }
    }

    fn isect_horizontal_point(&self, y_line: T) -> Option<T> {
        let y1 = self.p1.y;
        let y2 = self.p2.y;

        //y = a*x + b
        //x = y/a - b/a

        if ((y1 <= y_line) && (y_line <= y2)) || ((y2 <= y_line) && (y_line <= y1)) {
            let (slope, intercept) = self.line_params();

            if self.p1.x == self.p2.x {
                return Some(self.p1.x);
            }

            Some((y_line - intercept)/slope)
        } 
        else {
            None
        }
    }

    fn isect_vertical_point(&self, x_line: T) -> Option<T> {
        let x1 = self.p1.x;
        let x2 = self.p2.x;

        if ((x1 <= x_line) && (x_line <= x2)) || ((x2 <= x_line) && (x_line <= x1)) {
            let (slope, intercept) = self.line_params();

            Some(x_line*slope + intercept)
        } 
        else {
            None
        }
    }
}

impl<T: Float> SDF<T> for Point<T> {
    fn normal(&self, point: Point<T>) -> Vec2D<T> {
        (point - *self).normalize()
    }
    
    fn distance(&self, point: Point<T>) -> T {
        (point - *self).norm()
    }
}

impl<T: Float> SDF<T> for LineSegment<T> {
    fn normal(&self, point: Point<T>) -> Vec2D<T> { 
        let ap = point - self.p1;
        let ab = self.p2 - self.p1;

        let h = ap.dot(ab)/ab.dot(ab);
        if h < T::zero() {
            ap.normalize()
        }
        else if h < T::one() {
            let n = ab.orth().normalize();
            n * n.dot(ap).signum()
        }
        else {
            self.p2.normal(point)
        }
    }
    
    fn distance(&self, point: Point<T>) -> T {
        let ap = point - self.p1;
        let ab = self.p2 - self.p1;

        let h = (ap.dot(ab)/ab.dot(ab)).clamp(T::zero(), T::one());
        (ap - ab * h).norm()
    }
}

impl<T: Float> Circle<T> {
    pub fn new(center: Point<T>, radius: T) -> Self {
        Self { r: radius, p: center }
    }
}

impl<T: Float> SDF<T> for Circle<T> {
    fn normal(&self, point: Point<T>) -> Vec2D<T> { self.p.normal(point) }

    fn distance(&self, point: Point<T>) -> T {
        self.p.distance(point) - self.r
    }
}

impl<T: Float> Rectangle<T> {
    pub fn from_points(p1: Point<T>, p2: Point<T>) -> Self {
        let sx = (p2.x - p1.x).abs(); 
        let sy = (p2.y - p1.y).abs();


        let px = p1.x.min(p2.x);
        let py = p1.y.min(p2.y);
        Self { point: (px, py).into(), size: (sx, sy).into() }
    }

    pub fn contains(&self, point: Point<T>) -> bool {
        point.x > self.point.x && point.x < self.point.x + self.size.x1 && point.y > self.point.y && point.y < self.point.y + self.size.x2
    }
}

impl<T: Float> SDF<T> for Rectangle<T> {
    fn distance(&self, point: Point<T>) -> T {
        let xdist = (self.point.x + self.size.x1 * T::from(0.5).unwrap() - point.x).abs() - self.size.x1 * T::from(0.5).unwrap();

        let ydist = (self.point.y + self.size.x2 * T::from(0.5).unwrap() - point.y).abs() - self.size.x2 * T::from(0.5).unwrap();

        xdist.min(ydist)
    }

    fn normal(&self, point: Point<T>) -> Vec2D<T> {
        if point.x > self.point.x {
            (T::one(),T::zero()).into()
        }
        else {
            (T::zero(), T::one()).into()
        }
    }
    
}

impl<T: Float> Polygon<T> {
    pub fn new(points: Vec<Point<T>>) -> Self {
        Self { points }
    }

    pub fn vertices(&self) -> usize {self.points.len()}

    pub fn line_iter<'a>(&'a self) -> PolygonLineIterator<'a, T> {
        PolygonLineIterator{ polygon: &self, index: 0}
    }

    pub fn contains(&self, point: Point<T>) -> bool {
        let mut count = 0;

        let mut pmax: Point<T> = (T::min_value(), T::min_value()).into();
        let mut pmin: Point<T> = (T::max_value(), T::max_value()).into();

        self.line_iter().for_each(|l| {
            if let Some(x) = l.isect_horizontal_point(point.y) {
                if x < point.x { count += 1 }
            }

            if point.y == l.p1.y && l.p1.x < point.x { 
                count -= 1;
                if l.p2.y == point.y {
                    count -= 1;
                }
            }

            if l.p1.y > pmax.y {pmax = l.p1}
            if l.p1.y < pmin.y {pmin = l.p1}

            //if point.y == l.p2.y && l.p2.x < point.x { count += 1 }
        });

        if point.y == pmax.y && pmax.x < point.x {
            count += 1;
        }

        if point.y == pmin.y && pmin.x < point.x {
            count += 1;
        }

        if count.is_odd() {
            true
        }
        else {
            false
        }
    }
}

impl<'a, T: Float> Iterator for PolygonLineIterator<'a, T> {
    type Item = LineSegment<T>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.polygon.vertices() - 1 {
            let i = self.index;
            self.index += 1;
            Some(LineSegment { p1: self.polygon.points[i], p2: self.polygon.points[i + 1] })
        }
        else if self.index == self.polygon.vertices() - 1 {
            let i = self.index;
            self.index += 1;
            Some(LineSegment { p1: self.polygon.points[i], p2: self.polygon.points[0] })
        }
        else {
            None
        }
    }
}

impl<T: Float> SDF<T> for Polygon<T> {
    fn distance(&self, point: Point<T>) -> T {
        let contained = self.contains(point);

        let mut dist = T::max_value();

        self.line_iter().for_each(|l| {
            let check_dist = l.distance(point);

            if check_dist < dist {
                dist = check_dist;
            }
        });

        if contained {
            -dist
        }
        else {
            dist
        }
    }

    fn normal(&self, point: Point<T>) -> Vec2D<T> {
        let contained = self.contains(point);

        let mut dist = T::max_value();

        let mut p1 = Point::<T>::default();
        let mut p2 = Point::<T>::default();

        self.line_iter().for_each(|(l)| {
            let check_dist = l.distance(point);

            if check_dist < dist {
                dist = check_dist;
                p1 = l.p1;
                p2 = l.p2;
            }
        });

        let normal = LineSegment::new(p1, p2).normal(point);

        if contained {
            normal * -T::one()
        }
        else {
            normal
        }
    }
}

impl<T: Float> BeveledRect<T> {
    pub fn new(p1: Point<T>, p2: Point<T>, bevel: T) -> Self {
        let mut rect = Rectangle::<T>::from_points(p1, p2);
        rect.point.x = rect.point.x + bevel;
        rect.point.y = rect.point.y + bevel;

        rect.size.x1 = rect.size.x1 - bevel * T::from(2.).unwrap();
        rect.size.x2 = rect.size.x2 - bevel * T::from(2.).unwrap();

        Self { rect, bevel }
    }

    pub fn contains(&self, point: Point<T>) -> bool {
        if self.distance(point) < T::zero() || self.rect.contains(point) {true} else {false}
    }
}

impl<T: Float> SDF<T> for BeveledRect<T> {
    fn distance(&self, point: Point<T>) -> T {
        let p = self.rect.point;
        let s = self.rect.size;

        Polygon::new(vec![p, (p.x, p.y + s.x2).into(), (p.x + s.x1, p.y + s.x2).into(), (p.x + s.x1, p.y).into()])
            .distance(point) - self.bevel
    }

    fn normal(&self, point: Point<T>) -> Vec2D<T> {
        let p = self.rect.point;
        let s = self.rect.size;
        
        let n = Polygon::new(vec![p, (p.x, p.y + s.x2).into(), (p.x + s.x1, p.y + s.x2).into(), (p.x + s.x1, p.y).into()])
            .normal(point);

        n
    }
}

impl<T: Float> ExpandedPolygon<T> {
    pub fn new(points: Vec<Point<T>>, expand: T) -> Self {
        Self { polygon: Polygon::new(points), expand }
    }

    pub fn vertices(&self) -> usize {self.polygon.points.len()}

    pub fn line_iter<'a>(&'a self) -> PolygonLineIterator<'a, T> {
        self.polygon.line_iter()
    }

    pub fn contains(&self, point: Point<T>) -> bool {
        if self.distance(point) <= T::zero() {true} else {false} 
    }
}

impl<T: Float> SDF<T> for ExpandedPolygon<T> {
    fn distance(&self, point: Point<T>) -> T {
        self.polygon.distance(point) - self.expand
    }

    fn normal(&self, point: Point<T>) -> Vec2D<T> {
        self.polygon.normal(point)
    }
}