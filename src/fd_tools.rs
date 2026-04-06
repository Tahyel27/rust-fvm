use num::{self, Integer, Float};
use ndarray::{s, Array2};

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

#[derive(Default)]
struct NeumannCell {
    cell: (usize, usize),
    //clockwise from top
    interp_cof: [f64; 4]
}

#[derive(Default)]
struct FDWalls {
    mask: Array2<f64>,
    neum_cells: Vec<NeumannCell>,
    dir_cells: Vec<(usize, usize)>
}

impl FDWalls {
    pub fn new(dimx: usize, dimy: usize) -> Self {
        Self { mask: Array2::<f64>::zeros((dimx, dimy)), dir_cells: Default::default(), neum_cells: Default::default()}
    }

    pub fn from_mask(mask: &Array2<u8>) -> Self {
        let mut s = Self::default();
        s.precalculate_setup(mask);
        s
    }

    pub fn change_mask(&mut self, mask: &Array2<u8>) {
        self.precalculate_setup(mask);
    }

    pub fn get_mask(&self) -> &Array2<f64> { &self.mask}

    fn precalculate_setup(&mut self, input_mask: &Array2<u8>) {
        let (dimx, dimy) = input_mask.dim();

        let mut mask = Array2::<f64>::zeros((dimx, dimy));

        //1 - wall, 0 - fluid
        let mask_s = input_mask.slice(s![1..-1,1..-1]);

        self.neum_cells = Default::default();
        self.dir_cells = Default::default();

        mask_s.indexed_iter().for_each(|((i_f,j_f), m)| {
            let (i, j) = (i_f + 1, j_f + 1);
            mask[[i,j]] = if *m > 0 { 0.0 } else { 1.0 };  

            //fluid cells count as one, used for counting up fluid cells
            let u = if input_mask[[i,j+1]] > 0 { 0.0 } else {1.0};
            let r = if input_mask[[i+1,j]] > 0 { 0.0 } else {1.0};
            let l = if input_mask[[i-1,j]] > 0 { 0.0 } else {1.0};
            let d = if input_mask[[i,j-1]] > 0 { 0.0 } else {1.0};
            
            if *m == 2 {
                let sum: f64 = u + r + d + l;

                //create the normalized list
                let n = sum;
                let coffs = [u / n, r / n, d / n, l / n];
                
                //push it into the vector of neumann
                let cell = NeumannCell { cell: (i,j), interp_cof: coffs};

                if sum > 0.1 { self.neum_cells.push(cell); };
            }
            else if *m == 1 {
                let sum = u + r +d + l;

                if sum > 0.1 {self.dir_cells.push((i,j));}
            }
        });

        self.mask = mask;
    }

    pub fn apply_neumann(&self, u: &mut Array2<f64>) {
        self.neum_cells.iter().for_each(|cell| {
            let (i, j) = cell.cell;

            let [up, r, d, l] = cell.interp_cof;

            u[[i,j]] = up * u[[i,j+1]] + r * u[[i+1,j]] + d * u[[i,j-1]] + l * u[[i-1,j]];

        });
    }

    pub fn apply_dirichlet(&self, u: &mut Array2<f64>, val: f64) {
        self.dir_cells.iter().for_each(|(i,j)| {
            u[[*i,*j]] = val;
        });
    }
}