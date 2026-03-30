use ndarray::{s, Array2};

use crate::simhandler::SimulationData;

#[derive(Clone)]
pub struct FVParams {
    pub dt: f64
}

#[derive(Default)]
struct FVFields {
    h: Array2<f64>,
    hu: Array2<f64>,
    hv: Array2<f64>
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
}

impl Default for FVParams {
    fn default() -> Self {
        Self { dt: 0.15 }
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
            flux_y: FVFields { h: z.clone(), hu: z.clone(), hv: z.clone() }
        }
    }

    pub fn new_case(h: Array2<f64>, hu: Array2<f64>, hv: Array2<f64>) -> Self {
        let z = Array2::<f64>::zeros(h.dim());

        Self {
            q: FVFields { h, hu, hv },
            q_slopes_x: FVSlopes { h: z.clone(), hu: z.clone(), hv: z.clone() },
            q_slopes_y: FVSlopes { h: z.clone(), hu: z.clone(), hv: z.clone() },
            flux_x: FVFields { h: z.clone(), hu: z.clone(), hv: z.clone() },
            flux_y: FVFields { h: z.clone(), hu: z.clone(), hv: z.clone() },
        }
    }

    pub fn add_to_h(&mut self, h: Array2<f64>) { self.q.h += &h; }

    pub fn set_fields(&mut self, h: Array2<f64>, hu: Array2<f64>, hv: Array2<f64>) {
        self.q.h = h;
        self.q.hu = hu;
        self.q.hv = hv;
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

fn euler_step(q: &mut Array2<f64>, f_x: &Array2<f64>, f_y: &Array2<f64>, dt: f64) {
    let f_x_slice = f_x.slice(s![0..-1,1..-1]);
    let f_y_slice = f_y.slice(s![1..-1,0..-1]);

    let f_x_wins = f_x_slice.windows((2,1));
    let f_y_wins = f_y_slice.windows((1,2));

    ndarray::Zip::from(q.slice_mut(s![1..-1,1..-1]))
        .and(f_x_wins)
        .and(f_y_wins)
        .par_for_each(|u, f_xw, f_yw| {
            let f_x_l = f_xw[[0,0]]; let f_x_r = f_xw[[1,0]];
            let f_y_b = f_yw[[0,0]]; let f_y_t = f_yw[[0,1]];

            *u = *u + dt*(f_x_l - f_x_r + f_y_b - f_y_t);
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

impl SimulationData for FVData {
    type SimParams = FVParams;

    type SimRes = Array2<f64>;

    fn update(&mut self, ctx: &crate::simhandler::SimulationContext<Self::SimParams>) -> () {
        let dt = ctx.get_params().dt;
        
        let q = &mut self.q;
        let q_sl_x = &mut self.q_slopes_x;
        let q_sl_y = &mut self.q_slopes_y;

        reconstruct_slopes(&q.h, &mut q_sl_x.h, &mut q_sl_y.h);
        reconstruct_slopes(&q.hu, &mut q_sl_x.hu, &mut q_sl_y.hu);
        reconstruct_slopes(&q.hv, &mut q_sl_x.hv, &mut q_sl_y.hv);

        calc_flux_horizontal(q, q_sl_x, &mut self.flux_x);
        calc_flux_vertical(q, q_sl_y, &mut self.flux_y);

        euler_step(&mut q.h, &self.flux_x.h, &self.flux_y.h, dt);
        euler_step(&mut q.hu, &self.flux_x.hu, &self.flux_y.hu, dt);
        euler_step(&mut q.hv, &self.flux_x.hv, &self.flux_y.hv, dt);
        
    }

    fn send_result(&self, ctx: &crate::simhandler::SimulationContext<Self::SimParams>) -> Self::SimRes {
        self.q.h.clone()
    }
}