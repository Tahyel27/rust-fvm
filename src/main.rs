mod fd_tools;
mod heatmap;
mod simhandler;
mod mpscsingle;
mod colormap;
mod fv;

use crate::{fv::{FVData, FVParams}, simhandler::*};
use fd_tools::*;
use colormap::*;
use heatmap::*;

use eframe::egui;
use ndarray::Array2;

#[derive(Default)]
struct UiParams {
    brush_width: f64,
    brush_height: f64,
    draw_mask: bool,
    def_h: f64,
    dimx: usize,
    dimy: usize,
    cmap_max: f64,
    cmap_min: f64
}
struct App {
    sim: SimulationHandler<fv::FVData>,
    heatmap: HeatmapState,
    params: fv::FVParams,
    ui_params: UiParams,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {

        let dims = (600, 600);

        //simulation set up

        let h = FieldBuilder::new(dims)
            .set(0.1)
            .add_gaussian((300., 300.), 40. , 0.8)
            .field();

        let v = FieldBuilder::new(dims).set(0.0).field();

        let case = FVData::new_case(h, v.clone(), v);
        
        let params = FVParams::default();

        let sim = SimulationHandler::new(case, params.clone());

        //ui set up

        let heatmap = HeatmapState::default();

        let ui_params = UiParams {
            brush_height: 0.2,
            brush_width: 20.,
            draw_mask: true,
            def_h: 1.0,
            dimx: dims.0,
            dimy: dims.1,
            cmap_max: 1.5,
            cmap_min: 0.5
        };

        Self {
            sim,
            heatmap,
            params, 
            ui_params,
        }
    }
}

impl eframe::App for App {

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        egui::Panel::top("top_panel").min_size(100.).show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {

                    ui.set_min_width(220.);

                    let desired_width = ui.available_width().min(200.);

                    ui.spacing_mut().slider_width = desired_width;
                    if ui.add(egui::Slider::new(&mut self.params.dt, 0.0..=0.25).text("dt")).changed() {
                        self.sim.update_params(|p| p.dt = self.params.dt);
                    }

                    ui.add(egui::Slider::new(&mut self.ui_params.cmap_max, 0.0..=2.0).text("max"));
                    ui.add(egui::Slider::new(&mut self.ui_params.cmap_min, 0.0..=2.0).text("min"));
                    /*if ui.add(egui::Slider::new(&mut self.params.dx, 0.0..=1.0).text("dx")).changed() {
                        self.sim.update_params(|p| p.dx = self.params.dx);
                    };*/
                });
                ui.add_space(40.);
                ui.vertical(|ui| {
                    ui.set_max_width(200.);

                    let width = ui.available_width();
                    if ui.add_sized([width, 20.], egui::Button::new("Run!")).clicked() {
                        self.sim.run();
                    }
                    if ui.add_sized([width, 20.], egui::Button::new("Pause!")).clicked() {
                        self.sim.pause();
                    }
                    if ui.add_sized([width, 20.], egui::Button::new("Resume!")).clicked() {
                        self.sim.resume();
                    }
                    if ui.add_sized([width, 20.], egui::Button::new("Clear!")).clicked() {
                        let dim= (self.ui_params.dimx,self.ui_params.dimy);
                        let h0 = self.ui_params.def_h;
                        self.sim.modify_data(move |data| {
                            let h = Array2::<f64>::from_elem(dim, h0);
                            let z = Array2::<f64>::zeros(dim);

                            data.set_fields(h, z.clone(), z.clone());
                        });
                    }
                });
                ui.add_space(40.);
                ui.vertical(|ui| {
                    ui.set_max_width(200.);

                    ui.spacing_mut().slider_width =  ui.available_width();

                    ui.add(egui::Slider::new(&mut self.ui_params.brush_width, 0.0..=100.0).text("Pulse width"));
                    ui.add(egui::Slider::new(&mut self.ui_params.brush_height, -1.0..=2.0).text("Pulse height"));
                    ui.add(egui::Slider::new(&mut self.ui_params.def_h, 0.0..=1.0).text("Base height"));
                    ui.checkbox(&mut self.ui_params.draw_mask, "Show mask!");
                })
            });
        });

        if let Some(data) = self.sim.try_receive() {
            //self.heatmap.render_data(&data, ui.ctx(), BRWColormap::new(0.5, 1.5));
            self.heatmap.render_data(&data, ui.ctx(), BRWColormap::new(self.ui_params.cmap_min, self.ui_params.cmap_max));
            //self.heatmap.render_data(&data, ui.ctx(), BRWColormap::new(-0.5, 0.5));
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let res = HeatmapPlot::new(&self.heatmap)
                .show_mask(self.ui_params.draw_mask)
                .show(ui);
            if res.response.clicked() {
                if let Some(p) = res.pos {
                    let w = self.ui_params.brush_width;
                    let h = self.ui_params.brush_height;
                    self.sim.modify_data(move |data| {
                        let (dimx, dimy) = data.dim();
                        let (ox, oy) = ((dimx / 2) as f64, ( dimy / 2)  as f64);
                        let gauss = fd_tools::FieldBuilder::new((dimx,dimy))
                            .set_gaussian((p.x + ox, (dimy as f64) - p.y - oy), w, h)
                            .field();
                        data.add_to_h(gauss);
                    });
                }
            }
        });

        ui.ctx().request_repaint();
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        
    }
}

fn main() {

    let native_options = eframe::NativeOptions::default();
    eframe::run_native("Wave simulation", native_options, 
    Box::new(|cc| Ok(Box::new(App::new(cc)))));
}
