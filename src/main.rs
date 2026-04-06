mod fd_tools;
mod heatmap;
mod simhandler;
mod mpscsingle;
mod colormap;
mod geoprimitives;
mod fv;
mod fvgeometry;
mod files;

use std::{sync::mpsc, time::Duration};
use rfd::FileDialog;

use crate::files::*;
use fv::{FVBoundary, FVData, FVDomain, FVParams};
use crate::fvgeometry::FVGeometry;
use crate::simhandler::SimulationHandler;

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
    geometry: fvgeometry::FVGeometry
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {

        let dims = (600, 600);

        //simulation set up

        let h = FieldBuilder::new(dims)
            .set(0.05)
            //.add_gaussian((300., 300.), 40. , 0.8)
            //.add_rectangle((0,0), (dims.0,dims.0/6), 0.8)
            //.add_rectangle((0,0), (dims.1/6,dims.1), 0.8)
            .field();

        let u = FieldBuilder::new(dims).set(0.0).field();
        let v = FieldBuilder::new(dims).set(0.0).field();

        let mask = MaskBuilderU8::new(dims)
            .set(0)
            //.add_circle((300.,300.), 50., 1)
            .add_rectangle((200,200), (200,50),1)
            .mask();

        let mut geometry = FVGeometry::new();
        geometry.add_circle((100.,300.), 5.1);
        //geometry.add_circle((300.,300.), 20.1);
        geometry.add_circle((350.,300.), 100.1);
        //geometry.add_expanded_polygon(vec![(200.1,300.).into(), (400.1,250.).into(), (400.,350.).into()],20.5);
        //geometry.add_beveled_rect((200.1,270.1), (400.1,329.9), 20.);
        //geometry.add_polygon(vec![(20.,40.).into(), (500., 200.).into(), (400.,400.).into(), (50.,100.).into()]);
        let hmask = geometry.create_mask(dims);
        
        //let case = FVData::new_case(h, v.clone(), v, mask);
        let case = FVData::new_case_geom(h, u, v, geometry.clone());
        
        let mut params = FVParams::default();
        params.domain = FVDomain {
            top: FVBoundary::Open,
            bottom: FVBoundary::Open,
            left: FVBoundary::Inlet(0.31, 0.4),
            right: FVBoundary::Open
        };

        let sim = SimulationHandler::new(case, params.clone());

        //ui set up

        let mut heatmap = HeatmapState::default();
        heatmap.set_mask(&hmask, &cc.egui_ctx);

        let ui_params = UiParams {
            brush_height: 0.2,
            brush_width: 20.,
            draw_mask: true,
            def_h: 1.0,
            dimx: dims.0,
            dimy: dims.1,
            cmap_max: 1.8,
            cmap_min: 0.1
        };

        Self {
            sim,
            heatmap,
            params, 
            ui_params,
            geometry
        }
    }

    fn save_file(&mut self) {
        let file = FileDialog::new()
            .add_filter("JSON simulation config", &["json"])
            .save_file();

        if let Some(file) = file {
            
            let (tx, rx) = mpsc::channel();

            self.sim.modify_data(move |data| {
                tx.send(data.clone_fields());
            });

            let received = rx.recv_timeout(Duration::from_secs(1));

            self.sim.pause();

            if let Ok(data) = received {
                save_to_file_sw(&file, data, self.params.dt, self.params.domain.clone(), &self.geometry).unwrap();
            }
        }
    }

    fn open_simulation(&mut self) {
        let file = FileDialog::new()
            .add_filter("JSON simulation config", &["json"])
            .pick_file();

        if let Some(file) = file {
            let (case, params) = open_case_sw(&file).unwrap();

            self.sim.set_data(case);
            self.sim.set_params(params.clone());

            self.params = params;
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
                    if ui.button("Step one!").clicked() {
                        self.sim.resume();
                        std::thread::sleep(Duration::from_millis(5));
                        self.sim.pause();
                    }
                });
                ui.add_space(40.);
                ui.vertical(|ui| {
                    ui.set_max_width(200.);

                    ui.spacing_mut().slider_width =  ui.available_width();

                    let width = ui.available_width();

                    if ui.add_sized([width, 20.], egui::Button::new("Save simulation")).clicked() {
                        self.save_file();
                    }
                    if ui.add_sized([width, 20.], egui::Button::new("Open simulation")).clicked() {
                        self.open_simulation();
                    }
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
