use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};

use crate::{fvgeometry::{FVGeometry, FVWalls}, simhandler::SimulationData};

#[derive(Clone)]
pub struct FVParams {
    pub dt: f64,
    pub domain: FVDomain
}

#[derive(Default, Clone)]
pub struct FVFields {
    pub h: Array2<f64>,
    pub hu: Array2<f64>,
    pub hv: Array2<f64>
}

#[derive(Default)]
struct FVSlopes {
    h: Array2<f64>,
    hu: Array2<f64>,
    hv: Array2<f64>
}

#[derive(Default)]
pub struct FVData {
    q: FVFields,
    q_slopes_x: FVSlopes,
    q_slopes_y: FVSlopes,
    flux_x: FVFields,
    flux_y: FVFields,
    buffer: Array2<f64>,
    geometry: FVWalls
}

impl Default for FVParams {
    fn default() -> Self {
        Self { dt: 0.1, domain: Default::default() }
    }
}

impl FVData {
    pub fn new(dims: (usize,usize)) -> Self {
        let h = Array2::<f64>::from_elem(dims, 1.0);
        let z = Array2::<f64>::zeros(dims);
        Self {
            q: FVFields { h, hu: z.clone(), hv: z.clone() },
            q_slopes_x: FVSlopes { h: z.clone(), hu: z.clone(), hv: z.clone() },
            q_slopes_y: FVSlopes { h: z.clone(), hu: z.clone(), hv: z.clone() },
            flux_x: FVFields { h: z.clone(), hu: z.clone(), hv: z.clone() },
            flux_y: FVFields { h: z.clone(), hu: z.clone(), hv: z.clone() },
            buffer: z.clone(),
            geometry: Default::default()
        }
    }

    pub fn new_case(h: Array2<f64>, hu: Array2<f64>, hv: Array2<f64>, mask: Array2<u8>) -> Self {
        let z = Array2::<f64>::zeros(h.dim());

        Self {
            q: FVFields { h, hu, hv },
            q_slopes_x: FVSlopes { h: z.clone(), hu: z.clone(), hv: z.clone() },
            q_slopes_y: FVSlopes { h: z.clone(), hu: z.clone(), hv: z.clone() },
            flux_x: FVFields { h: z.clone(), hu: z.clone(), hv: z.clone() },
            flux_y: FVFields { h: z.clone(), hu: z.clone(), hv: z.clone() },
            buffer: z.clone(),
            geometry: Default::default()
        }
    }

    pub fn new_case_geom(h: Array2<f64>, hu: Array2<f64>, hv: Array2<f64>, geometry: FVGeometry) -> Self {
        let z = Array2::<f64>::zeros(h.dim());

        let geometry = FVWalls::new(geometry, z.dim());

        Self {
            q: FVFields { h, hu, hv },
            q_slopes_x: FVSlopes { h: z.clone(), hu: z.clone(), hv: z.clone() },
            q_slopes_y: FVSlopes { h: z.clone(), hu: z.clone(), hv: z.clone() },
            flux_x: FVFields { h: z.clone(), hu: z.clone(), hv: z.clone() },
            flux_y: FVFields { h: z.clone(), hu: z.clone(), hv: z.clone() },
            buffer: z.clone(),
            geometry
        }
    }

    pub fn add_to_h(&mut self, h: Array2<f64>) { self.q.h += &h; }

    pub fn add_to_hu(&mut self, hu: Array2<f64>) {self.q.hu += &hu; }

    pub fn set_fields(&mut self, h: Array2<f64>, hu: Array2<f64>, hv: Array2<f64>) {
        self.q.h = h;
        self.q.hu = hu;
        self.q.hv = hv;
    }

    pub fn clone_fields(&self) -> FVFields {
        self.q.clone()
    }

    pub fn dim(&self) -> (usize, usize) {self.q.h.dim()}

}

fn reconstruct_slopes(q: &Array2<f64>, q_sl_x: &mut Array2<f64>, q_sl_y: &mut Array2<f64>) {
    let q_windows = q.windows((3,3));

    let mut q_sl_x_slice = q_sl_x.slice_mut(s![1..-1,1..-1]);
    let mut q_sl_y_slice = q_sl_y.slice_mut(s![1..-1,1..-1]);

    ndarray::Zip::from(q_windows)
        .and(&mut q_sl_x_slice)
        .and(&mut q_sl_y_slice)
        .par_for_each(|qwin, sl_x, sl_y| {
            let x_forw = qwin[[2,1]] - qwin[[1,1]];
            let x_back = qwin[[1,1]] - qwin[[0,1]];

            let y_forw = qwin[[1,2]] - qwin[[1,1]];
            let y_back = qwin[[1,1]] - qwin[[1,0]];

            *sl_x = minmod(x_forw, x_back);
            *sl_y = minmod(y_forw, y_back);
        });

}

fn reconstruct_slopes_x(q: &Array2<f64>, q_sl_x: &mut Array2<f64>) {
    let q_windows = q.windows((3,3));

    let mut q_sl_x_slice = q_sl_x.slice_mut(s![1..-1,1..-1]);

    ndarray::Zip::from(q_windows)
        .and(&mut q_sl_x_slice)
        .par_for_each(|qwin, sl_x| {
            let x_forw = qwin[[2,1]] - qwin[[1,1]];
            let x_back = qwin[[1,1]] - qwin[[0,1]];

            *sl_x = minmod(x_forw, x_back);
        });
}

fn reconstruct_slopes_y(q: &Array2<f64>, q_sl_y: &mut Array2<f64>) {
    let q_windows = q.windows((3,3));

    let mut q_sl_y_slice = q_sl_y.slice_mut(s![1..-1,1..-1]);

    ndarray::Zip::from(q_windows)
        .and(q_sl_y_slice)
        .par_for_each(|qwin, sl_y| {
            let y_forw = qwin[[1,2]] - qwin[[1,1]];
            let y_back = qwin[[1,1]] - qwin[[1,0]];

            *sl_y = minmod(y_forw, y_back);
        });
    
}

fn calc_flux_horizontal(q: &FVFields, q_sl_x: &FVSlopes, f_x: &mut FVFields) {

    let g = 1.0;

    let mut f_hslice = f_x.h.slice_mut(s![0..-1,..]);
    let mut f_huslice = f_x.hu.slice_mut(s![0..-1,..]);
    let mut f_hvslice = f_x.hv.slice_mut(s![0..-1,..]);

    ndarray::Zip::indexed(&mut f_hslice)
        .and(&mut f_huslice)
        .and(&mut f_hvslice)
        .par_for_each(|(i,j), f_h, f_hu, f_hv| {
            let h_l = unsafe {q.h.uget((i,j))};
            let h_r = unsafe {q.h.uget((i+1,j))};

            let hu_l = unsafe {q.hu.uget((i,j))};
            let hu_r = unsafe {q.hu.uget((i+1,j))};
            
            let hv_l = unsafe {q.hv.uget((i,j))};
            let hv_r = unsafe {q.hv.uget((i+1,j))};

            let h_sl_l = unsafe {q_sl_x.h.uget((i,j))};
            let h_sl_r = unsafe {q_sl_x.h.uget((i+1,j))};

            let hu_sl_l = unsafe {q_sl_x.hu.uget((i,j))};
            let hu_sl_r = unsafe {q_sl_x.hu.uget((i+1,j))};

            let hv_sl_l = unsafe {q_sl_x.hv.uget((i,j))};
            let hv_sl_r = unsafe {q_sl_x.hv.uget((i+1,j))};

            let ql = (h_l + 0.5*h_sl_l, hu_l + 0.5*hu_sl_l);
            let qr = (h_r - 0.5*h_sl_r, hu_r - 0.5*hu_sl_r);

            let qm = hlle_qm(ql, hv_l + 0.5*hv_sl_l, qr, hv_r - 0.5*hv_sl_r, 1.0);
            let (flh, flhu, flhv) = sw_flux_x(qm, g);
            *f_h = flh;
            *f_hu = flhu;
            *f_hv = flhv;
 
        });
}

fn calc_flux_vertical(q: &FVFields, q_sl_y: &FVSlopes, f_y: &mut FVFields) {
    let g = 1.0;

    let mut f_hslice = f_y.h.slice_mut(s![..,0..-1]);
    let mut f_huslice = f_y.hu.slice_mut(s![..,0..-1]);
    let mut f_hvslice = f_y.hv.slice_mut(s![..,0..-1]);

    ndarray::Zip::indexed(&mut f_hslice)
        .and(&mut f_huslice)
        .and(&mut f_hvslice)
        .par_for_each(|(i, j), f_h, f_hu, f_hv| {
            // Fetch states from the "bottom" (j) and "top" (j+1) cells
            let h_b = unsafe { q.h.uget((i, j)) };
            let h_t = unsafe { q.h.uget((i, j + 1)) };

            let hu_b = unsafe { q.hu.uget((i, j)) };
            let hu_t = unsafe { q.hu.uget((i, j + 1)) };

            let hv_b = unsafe { q.hv.uget((i, j)) };
            let hv_t = unsafe { q.hv.uget((i, j + 1)) };

            // Fetch y-slopes for reconstruction
            let h_sl_b = unsafe { q_sl_y.h.uget((i, j)) };
            let h_sl_t = unsafe { q_sl_y.h.uget((i, j + 1)) };

            let hu_sl_b = unsafe { q_sl_y.hu.uget((i, j)) };
            let hu_sl_t = unsafe { q_sl_y.hu.uget((i, j + 1)) };

            let hv_sl_b = unsafe { q_sl_y.hv.uget((i, j)) };
            let hv_sl_t = unsafe { q_sl_y.hv.uget((i, j + 1)) };

            // In the y-direction, normal momentum is hv.
            let ql = (h_b + 0.5 * h_sl_b, hv_b + 0.5 * hv_sl_b);
            let qr = (h_t - 0.5 * h_sl_t, hv_t - 0.5 * hv_sl_t);

            // Solve the Riemann problem with hu as the transverse momentum.
            let qm_hlle = hlle_qm(
                ql, hu_b + 0.5 * hu_sl_b, 
                qr, hu_t - 0.5 * hu_sl_t, 
                1.0
            );

            let qm = (qm_hlle.0, qm_hlle.2, qm_hlle.1);

            // Assuming a corresponding sw_flux_y exists that returns (flux_h, flux_hu, flux_hv)
            let (flh, flhu, flhv) = sw_flux_y(qm, g);
            
            *f_h = flh;
            *f_hu = flhu;
            *f_hv = flhv;
        });
}

fn euler_step(q: &mut Array2<f64>, f_x: &Array2<f64>, f_y: &Array2<f64>, dt: f64, mask: &Array2<f64>) {
    let f_x_slice = f_x.slice(s![0..-1,1..-1]);
    let f_y_slice = f_y.slice(s![1..-1,0..-1]);

    let f_x_wins = f_x_slice.windows((2,1));
    let f_y_wins = f_y_slice.windows((1,2));

    ndarray::Zip::from(q.slice_mut(s![1..-1,1..-1]))
        .and(f_x_wins)
        .and(f_y_wins)
        .and(mask.slice(s![1..-1,1..-1]))
        .par_for_each(|u, f_xw, f_yw, m| {
            let f_x_l = f_xw[[0,0]]; let f_x_r = f_xw[[1,0]];
            let f_y_b = f_yw[[0,0]]; let f_y_t = f_yw[[0,1]];

            *u = *u + m * dt*(f_x_l - f_x_r + f_y_b - f_y_t);
        });
}

fn mask_field(u: &mut Array2<f64>, mask: &Array2<f64>) {
    ndarray::Zip::from(u)
        .and(mask)
        .par_for_each(|u, mask| {
            *u = mask * *u;
        });
}

fn hlle_qm(ql: (f64, f64), phi_l: f64, qr: (f64, f64), phi_r: f64, g: f64) -> (f64, f64, f64) {
    let h_l = ql.0;
    let hu_l = ql.1;
    let h_r = qr.0;
    let hu_r = qr.1;

    let u_l = hu_l / h_l;
    let u_r = hu_r / h_r;

    //roots 
    let h_r_sqrt = h_r.sqrt();
    let h_l_sqrt = h_l.sqrt();

    //Roe averages:
    let h_hat = (h_r + h_l) / 2.;
    let u_hat = (h_r_sqrt * u_r + h_l_sqrt*u_l) / (h_r_sqrt + h_l_sqrt);
    let c_hat = (g*h_hat).sqrt();

    let lam_1l = u_l - g*h_l_sqrt;
    let lam_2r = u_r + g*h_r_sqrt;

    let s1 = lam_1l.min(u_hat - c_hat);
    let s2 = lam_2r.max(u_hat + c_hat);

    let h_m = (hu_r - hu_l -s2*h_r + s1*h_l)/(s1 - s2);
    let hu_m = (hu_r*u_r - hu_l*u_l + 0.5*g*(h_r.powi(2) - h_l.powi(2)) 
        - s2*hu_r + s1*hu_l)/(s1-s2);

    let use_l = {
        if s1 > 0. { 1.0 } else {0.0}
    };

    let use_r = {
        if s2 < 0. { 1.0 } else {0.0}
    };

    let use_m = {
        if use_l == 0. && use_r == 0. { 1.0 } else {0.0}
    };

    let phi_m = {
        if hu_m > 0.0 {phi_l} else {phi_r}
    };

    (use_l*h_l + use_m*h_m + use_r*h_r,use_l*hu_l + use_m*hu_m + use_r*hu_r, phi_m)
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b > 0. {
        a.signum() * a.abs().min(b.abs())
    } else {
        0.0
    }
}

fn sw_flux_x(qm: (f64, f64, f64), g: f64) -> (f64, f64, f64) {
    (qm.1, qm.1.powi(2)/qm.0 + 0.5*g*qm.0.powi(2),qm.1*qm.2/qm.0)
}

fn sw_flux_y(qm: (f64, f64, f64), g: f64) -> (f64, f64, f64) {
    (qm.2, qm.1*qm.2/qm.0, qm.2.powi(2)/qm.0 + 0.5*g*qm.0.powi(2))
}

fn van_leer(a: f64, b: f64) -> f64 {

    //if a.abs() <= 0.002 { return 0.5 * (a + b);};
    //if a.abs() <= 10. { return 0.5 * (a + b);};

    let ab_abs = a.abs() + b.abs();

    let epsilon = 1e-12;

    (a.abs() * b + b.abs() * a) / (ab_abs + epsilon)
}

fn dissipation(u: &mut Array2<f64>, buffer: &mut Array2<f64>, c: f64, dt: f64) {
    let uwins = u.windows((3,3));
    
    ndarray::Zip::from(buffer.slice_mut(s![1..-1,1..-1]))
        .and(uwins)
        .par_for_each(|buf, uw| {
            *buf = -4. * uw[[1,1]] + uw[[0,1]] + uw[[2,1]] + uw[[1,2]] + uw[[1,0]];
        });
    
    ndarray::Zip::from(u.slice_mut(s![1..-1,1..-1]))
        .and(buffer.slice_mut(s![1..-1,1..-1]))
        .par_for_each(|u, buf| {
            *u = *u + *buf * dt * c;
        });
}

impl SimulationData for FVData {
    type SimParams = FVParams;

    type SimRes = Array2<f64>;

    fn update(&mut self, ctx: &crate::simhandler::SimulationContext<Self::SimParams>) -> () {
        let dt = ctx.get_params().dt;
        
        let q = &mut self.q;
        let q_sl_x = &mut self.q_slopes_x;
        let q_sl_y = &mut self.q_slopes_y;

        /*reconstruct_slopes(&q.h, &mut q_sl_x.h, &mut q_sl_y.h);
        reconstruct_slopes(&q.hu, &mut q_sl_x.hu, &mut q_sl_y.hu);
        reconstruct_slopes(&q.hv, &mut q_sl_x.hv, &mut q_sl_y.hv);

        calc_flux_horizontal(q, q_sl_x, &mut self.flux_x);
        calc_flux_vertical(q, q_sl_y, &mut self.flux_y);

        euler_step(&mut q.h, &self.flux_x.h, &self.flux_y.h, dt);
        euler_step(&mut q.hu, &self.flux_x.hu, &self.flux_y.hu, dt);
        euler_step(&mut q.hv, &self.flux_x.hv, &self.flux_y.hv, dt);

        self.walls.apply_x_walls(&mut q.h, &mut q.hu, &mut q.hv);
        self.walls.apply_y_walls(&mut q.h, &mut q.hv, &mut q.hu);*/

        reconstruct_slopes_x(&q.h, &mut q_sl_x.h);
        reconstruct_slopes_x(&q.hu, &mut q_sl_x.hu);
        reconstruct_slopes_x(&q.hv, &mut q_sl_x.hv);

        //self.walls.apply_x_walls(&mut q.h, &mut q.hu, &mut q.hv);

        calc_flux_horizontal(q, q_sl_x, &mut self.flux_x);

        reconstruct_slopes_y(&q.h, &mut q_sl_y.h);
        reconstruct_slopes_y(&q.hu, &mut q_sl_y.hu);
        reconstruct_slopes_y(&q.hv, &mut q_sl_y.hv);

        //self.walls.apply_y_walls(&mut q.h, &mut q.hv, &mut q.hu);

        calc_flux_vertical(q, q_sl_y, &mut self.flux_y);


        let mask = self.geometry.get_mask();
        euler_step(&mut q.h, &self.flux_x.h, &self.flux_y.h, dt, mask);
        euler_step(&mut q.hu, &self.flux_x.hu, &self.flux_y.hu, dt, mask);
        euler_step(&mut q.hv, &self.flux_x.hv, &self.flux_y.hv, dt, mask);

        //mask_field(&mut q.h, mask);
        //mask_field(&mut q.hu, mask);
        //mask_field(&mut q.hv, mask);

        self.geometry.apply_to_scalar_field(&mut q.h);
        self.geometry.apply_to_velocity_field(&mut q.hu, &mut q.hv);

        /*dissipation(&mut q.h, &mut self.buffer, 0.1, dt);
        dissipation(&mut q.hu, &mut self.buffer, 0.1, dt);
        dissipation(&mut q.hv, &mut self.buffer, 0.1, dt);*/

        ctx.get_params().domain.apply_bcs(&mut q.h, &mut q.hu, &mut q.hv);
        
    }

    fn send_result(&self, ctx: &crate::simhandler::SimulationContext<Self::SimParams>) -> Self::SimRes {
        self.q.h.clone()
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub enum FVBoundary {
    #[default]
    Wall,
    Open,
    Inlet(f64, f64)
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct FVDomain {
    pub top: FVBoundary,
    pub bottom: FVBoundary,
    pub left: FVBoundary,
    pub right: FVBoundary
}

fn bot_neumann(u: &mut Array2<f64>) {
    for i in 0..u.dim().0  {
        u[[i,0]] = u[[i,1]];
    }
}

fn top_neumann(u: &mut Array2<f64>) {
    let y = u.dim().1 - 1;

    for i in 0..u.dim().0  {
        u[[i,y]] = u[[i,y-1]];
    }
}

fn left_neumann(u: &mut Array2<f64>) {
    for i in 0..u.dim().1 {
        u[[0,i]] = u[[1,i]];
    }
}

fn right_neumann(u: &mut Array2<f64>) {
    let x = u.dim().0 - 1;
    
    for i in 0..u.dim().1 {
        u[[x,i]] = u[[x-1,i]];
    }
}

fn bot_reflect(u: &mut Array2<f64>) {
    for i in 0..u.dim().0  {
        u[[i,0]] = -u[[i,1]];
    }
}

fn top_reflect(u: &mut Array2<f64>) {
    let y = u.dim().1 - 1;

    for i in 0..u.dim().0  {
        u[[i,y]] = -u[[i,y-1]];
    }
}

fn left_reflect(u: &mut Array2<f64>) {
    for i in 0..u.dim().1 {
        u[[0,i]] = -u[[1,i]];
    }
}

fn right_reflect(u: &mut Array2<f64>) {
    let x = u.dim().0 - 1;
    
    for i in 0..u.dim().1 {
        u[[x,i]] = -u[[x-1,i]];
    }
}

fn bot_fixed(u: &mut Array2<f64>, v: f64) {
    for i in 0..u.dim().0  {
        u[[i,0]] = v;
    }
}

fn top_fixed(u: &mut Array2<f64>, v: f64) {
    let y = u.dim().1 - 1;

    for i in 0..u.dim().0  {
        u[[i,y]] = v;
    }
}

fn left_fixed(u: &mut Array2<f64>, v: f64) {
    for i in 0..u.dim().1 {
        u[[0,i]] = v;
    }
}

fn right_fixed(u: &mut Array2<f64>, v: f64) {
    let x = u.dim().0 - 1;
    
    for i in 0..u.dim().1 {
        u[[x,i]] = v;
    }
}

impl FVDomain {
    fn apply_bcs(&self,h: &mut Array2<f64>, hu: &mut Array2<f64>, hv: &mut Array2<f64>) {
        match self.top {
            FVBoundary::Wall => {
                top_neumann(h);
                top_reflect(hv);
                top_neumann(hu);
            },
            FVBoundary::Open => {
                top_neumann(h);
                top_neumann(hu);
                top_neumann(hv);
            },
            FVBoundary::Inlet(h0, u0) => {
                top_fixed(h, h0);
                top_neumann(hu);
                top_fixed(hv, u0);
            },
        }

        match self.bottom {
            FVBoundary::Wall => {
                bot_neumann(h);
                bot_reflect(hv);
                bot_neumann(hu);
            },
            FVBoundary::Open => {
                bot_neumann(h);
                bot_neumann(hu);
                bot_neumann(hv);
            },
            FVBoundary::Inlet(h0, u0) => {
                bot_fixed(h, h0);
                bot_neumann(hu);
                bot_fixed(hv, u0);
            },
        }

        match self.left {
            FVBoundary::Wall => {
                left_neumann(h);
                left_reflect(hu);
                left_neumann(hv);
            },
            FVBoundary::Open => {
                left_neumann(h);
                left_neumann(hu);
                left_neumann(hv);
            },
            FVBoundary::Inlet(h0, u0) => {
                left_fixed(h, h0);
                left_neumann(hv);
                left_fixed(hu, u0);
            },
        }

        match self.right {
            FVBoundary::Wall => {
                right_neumann(h);
                right_reflect(hu);
                right_neumann(hv);
            },
            FVBoundary::Open => {
                right_neumann(h);
                right_neumann(hu);
                right_neumann(hv);
            },
            FVBoundary::Inlet(h0, u0) => {
                right_fixed(h, h0);
                right_neumann(hv);
                right_fixed(hu, u0);
            },
        }
    }

}

#[derive(Default)]
struct WallCell {
    cell: (usize, usize),
    dir: i32
}