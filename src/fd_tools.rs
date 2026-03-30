use num::{self, Integer, Float};
use ndarray::Array2;

pub type MaskBuilderU8 = MaskBuilder<u8>;
pub struct MaskBuilder<T: Integer + Clone> {
    mask: Array2<T>,
}

impl<T: Integer + Clone> Default for MaskBuilder<T> {
    fn default() -> Self {
        Self {
            mask: Array2::<T>::zeros((0, 0)),
        }
    }
}

impl<T: Integer + Clone> From<Array2<T>> for MaskBuilder<T> {
    fn from(value: Array2<T>) -> Self {
        Self { mask: value }
    }
}

impl<T: Integer + Clone> MaskBuilder<T> {
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            mask: Array2::<T>::zeros(shape),
        }
    }

    pub fn mask(self) -> Array2<T> {
        self.mask
    }

    pub fn set(mut self, value: T) -> Self {
        self.mask.iter_mut().for_each(|v| *v = value.clone());
        self
    }

    pub fn mul(mut self, value: T) -> Self {
        self.mask.iter_mut().for_each(|v| *v = v.clone().mul(value.clone()));
        self
    }

    pub fn add(mut self, value: T) -> Self {
        self.mask.iter_mut().for_each(|v| *v = v.clone().add(value.clone()));
        self
    }

    pub fn add_circle(mut self, center: (f64, f64), radius: f64, value: T) -> Self {
        let r2 = radius * radius;
        for ((i, j), v) in self.mask.indexed_iter_mut() {
            let dx = i as f64 - center.0;
            let dy = j as f64 - center.1;
            if dx * dx + dy * dy <= r2 {
                *v = v.clone() + value.clone();
            }
        }
        self
    }

    pub fn set_circle(mut self, center: (f64, f64), radius: f64, value: T) -> Self {
        let r2 = radius * radius;
        for ((i, j), v) in self.mask.indexed_iter_mut() {
            let dx = i as f64 - center.0;
            let dy = j as f64 - center.1;
            if dx * dx + dy * dy <= r2 {
                *v = value.clone();
            }
        }
        self
    }

    pub fn add_rectangle(mut self, pos: (usize, usize), size: (usize, usize), value: T) -> Self {
        let (dim_i, dim_j) = self.mask.dim();
        // Clamp to prevent out-of-bounds panics
        let i_end = (pos.0 + size.0).min(dim_i);
        let j_end = (pos.1 + size.1).min(dim_j);

        for i in pos.0..i_end {
            for j in pos.1..j_end {
                self.mask[[i, j]] = self.mask[[i, j]].clone() + value.clone();
            }
        }
        self
    }

    pub fn set_rectangle(mut self, pos: (usize, usize), size: (usize, usize), value: T) -> Self {
        let (dim_i, dim_j) = self.mask.dim();
        // Clamp to prevent out-of-bounds panics
        let i_end = (pos.0 + size.0).min(dim_i);
        let j_end = (pos.1 + size.1).min(dim_j);

        for i in pos.0..i_end {
            for j in pos.1..j_end {
                self.mask[[i, j]] = value.clone();
            }
        }
        self
    }

    // Helper function for point-in-polygon (Ray Casting algorithm)
    fn is_point_in_polygon(x: f64, y: f64, points: &[(f64, f64)]) -> bool {
        let mut inside = false;
        if points.is_empty() {
            return false;
        }
        
        let mut j = points.len() - 1;
        for i in 0..points.len() {
            let pi = points[i];
            let pj = points[j];

            let intersect = ((pi.1 > y) != (pj.1 > y))
                && (x < (pj.0 - pi.0) * (y - pi.1) / (pj.1 - pi.1) + pi.0);
            
            if intersect {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    pub fn add_polygon(mut self, points: Vec<(f64, f64)>, value: T) -> Self {
        for ((i, j), v) in self.mask.indexed_iter_mut() {
            if Self::is_point_in_polygon(i as f64, j as f64, &points) {
                *v = v.clone() + value.clone(); 
            }
        }
        self
    }

    pub fn set_polygon(mut self, points: Vec<(f64, f64)>, value: T) -> Self {
        for ((i, j), v) in self.mask.indexed_iter_mut() {
            if Self::is_point_in_polygon(i as f64, j as f64, &points) {
                *v = value.clone(); 
            }
        }
        self
    }

    pub fn invert(mut self) -> Self {
        for v in self.mask.iter_mut() {
            if *v > T::zero() {
                *v = T::zero();
            } else if *v == T::zero() {
                *v = T::one();
            }
        }
        self
    }

    pub fn set_for_each<F>(mut self, mut f: F) -> Self
    where
        F: FnMut((usize, usize)) -> T,
    {
        for ((i, j), v) in self.mask.indexed_iter_mut() {
            *v = f((i, j));
        }
        self
    }
}

pub trait FieldExt<T: Float> {
    fn add_circle_mut(&mut self, center: (f64, f64), radius: f64, value: T);
    fn set_circle_mut(&mut self, center: (f64, f64), radius: f64, value: T);
    fn add_rectangle_mut(&mut self, pos: (usize, usize), size: (usize, usize), value: T);
    fn set_rectangle_mut(&mut self, pos: (usize, usize), size: (usize, usize), value: T);
    fn add_gaussian_mut(&mut self, center: (f64, f64), width: f64, height: f64);
    fn set_gaussian_mut(&mut self, center: (f64, f64), width: f64, height: f64);
}

impl<T: Float> FieldExt<T> for Array2<T> {
    fn add_circle_mut(&mut self, center: (f64, f64), radius: f64, value: T) {
        let r2 = radius * radius;
        for ((i, j), v) in self.indexed_iter_mut() {
            let dx = i as f64 - center.0;
            let dy = j as f64 - center.1;
            if dx * dx + dy * dy <= r2 {
                *v = *v + value;
            }
        }
    }

    fn set_circle_mut(&mut self, center: (f64, f64), radius: f64, value: T) {
        let r2 = radius * radius;
        for ((i, j), v) in self.indexed_iter_mut() {
            let dx = i as f64 - center.0;
            let dy = j as f64 - center.1;
            if dx * dx + dy * dy <= r2 {
                *v = value;
            }
        }
    }

    fn add_rectangle_mut(&mut self, pos: (usize, usize), size: (usize, usize), value: T) {
        let (dim_i, dim_j) = self.dim();
        let i_end = (pos.0 + size.0).min(dim_i);
        let j_end = (pos.1 + size.1).min(dim_j);

        for i in pos.0..i_end {
            for j in pos.1..j_end {
                self[[i, j]] = self[[i, j]] + value;
            }
        }
    }

    fn set_rectangle_mut(&mut self, pos: (usize, usize), size: (usize, usize), value: T) {
        let (dim_i, dim_j) = self.dim();
        let i_end = (pos.0 + size.0).min(dim_i);
        let j_end = (pos.1 + size.1).min(dim_j);

        for i in pos.0..i_end {
            for j in pos.1..j_end {
                self[[i, j]] = value;
            }
        }
    }

    fn add_gaussian_mut(&mut self, center: (f64, f64), width: f64, height: f64) {
        let w2 = width * width;
        let two_w2 = 2.0 * w2;
        
        for ((i, j), v) in self.indexed_iter_mut() {
            let dx = i as f64 - center.0;
            let dy = j as f64 - center.1;
            let dist_sq = dx * dx + dy * dy;
            
            let gauss_val = height * (-dist_sq / two_w2).exp();
            // Convert f64 to T safely using num::Float
            if let Some(t_val) = T::from(gauss_val) {
                *v = *v + t_val;
            }
        }
    }

    fn set_gaussian_mut(&mut self, center: (f64, f64), width: f64, height: f64) {
        let w2 = width * width;
        let two_w2 = 2.0 * w2;
        
        for ((i, j), v) in self.indexed_iter_mut() {
            let dx = i as f64 - center.0;
            let dy = j as f64 - center.1;
            let dist_sq = dx * dx + dy * dy;
            
            let gauss_val = height * (-dist_sq / two_w2).exp();
            if let Some(t_val) = T::from(gauss_val) {
                *v = t_val;
            }
        }
    }
}

// 2. The Builder Pattern
pub struct FieldBuilder<T: num::Float> {
    field: Array2<T>,
}

impl<T: num::Float> Default for FieldBuilder<T> {
    fn default() -> Self {
        Self { field: Array2::<T>::zeros((0, 0)) }
    }
}

impl<T: num::Float> From<Array2<T>> for FieldBuilder<T> {
    fn from(value: Array2<T>) -> Self {
        Self { field: value }
    }
}

impl<T: num::Float> FieldBuilder<T> {
    pub fn new(shape: (usize, usize)) -> Self {
        Self { field: Array2::<T>::zeros(shape) }
    }

    pub fn from_val(shape: (usize, usize), value: T) -> Self {
        Self { field: Array2::<T>::from_elem(shape, value) }
    }

    pub fn field(self) -> Array2<T> {
        self.field
    }

    pub fn set(mut self, value: T) -> Self {
        self.field.mapv_inplace(|_| value);
        self
    }

    pub fn add(mut self, value: T) -> Self {
        self.field.mapv_inplace(|v| v + value);
        self
    }

    pub fn add_circle(mut self, center: (f64, f64), radius: f64, value: T) -> Self {
        self.field.add_circle_mut(center, radius, value);
        self
    }

    pub fn set_circle(mut self, center: (f64, f64), radius: f64, value: T) -> Self {
        self.field.set_circle_mut(center, radius, value);
        self
    }

    pub fn add_rectangle(mut self, pos: (usize, usize), size: (usize, usize), value: T) -> Self {
        self.field.add_rectangle_mut(pos, size, value);
        self
    }

    pub fn set_rectangle(mut self, pos: (usize, usize), size: (usize, usize), value: T) -> Self {
        self.field.set_rectangle_mut(pos, size, value);
        self
    }

    pub fn add_gaussian(mut self, center: (f64, f64), width: f64, height: f64) -> Self {
        self.field.add_gaussian_mut(center, width, height);
        self
    }

    pub fn set_gaussian(mut self, center: (f64, f64), width: f64, height: f64) -> Self {
        self.field.set_gaussian_mut(center, width, height);
        self
    }

    pub fn set_for_each<F>(mut self, mut f: F) -> Self
    where
        F: FnMut((usize, usize)) -> T,
    {
        for ((i, j), v) in self.field.indexed_iter_mut() {
            *v = f((i, j));
        }
        self
    }

    pub fn add_for_each<F>(mut self, mut f: F) -> Self
    where
        F: FnMut((usize, usize)) -> T,
    {
        for ((i, j), v) in self.field.indexed_iter_mut() {
            *v = *v + f((i, j));
        }
        self
    }
}