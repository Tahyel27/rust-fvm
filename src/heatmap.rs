use crate::colormap::Colormap;

use eframe::egui::{self, Response, TextureHandle, Vec2};
use egui_plot::{Plot, PlotImage, PlotPoint, PlotResponse};
use ndarray::Array2;
use num::Float;

pub trait HeatmapStateGeneric {
    fn mask<'a>(&'a self) -> &'a Option<TextureHandle>;

    fn texture<'a>(&'a self) -> &'a Option<TextureHandle>;

    fn dims(&self) -> (usize, usize);
}

#[derive(Default)]
pub struct HeatmapState {
    texture: Option<TextureHandle>,
    mask: Option<TextureHandle>,
    dims: (usize, usize)
}

#[derive(Default)]
pub struct HeatmapStateOwned<N: Float> {
    state: HeatmapState,
    data: Array2<N>
}

pub struct HeatmapPlot<'a, State: HeatmapStateGeneric> {
    state: &'a State,
    data: HeatmapData
}

pub struct HeatmapResponse {
    pub response: Response,
    pub pos: Option<PlotPoint>
}

pub struct HeatmapData {
    scale: f32,
    show_axes: bool,
    show_mask: bool,
    show_grid: bool
}

impl Default for HeatmapData {
    fn default() -> Self {
        Self { 
            scale: 1.0,
            show_axes: true,
            show_mask: true,
            show_grid: true
         }
    }
}

impl HeatmapStateGeneric for HeatmapState {
    fn dims(&self) -> (usize, usize) {
        self.dims
    }

    fn mask<'a>(&'a self) -> &'a Option<TextureHandle> {
        &self.mask
    }

    fn texture<'a>(&'a self) -> &'a Option<TextureHandle> {
        &self.texture
    }
}

impl<N: Float> HeatmapStateGeneric for HeatmapStateOwned<N> {
    fn dims(&self) -> (usize, usize) {
        self.state.dims
    }

    fn mask<'a>(&'a self) -> &'a Option<TextureHandle> {
        &self.state.mask
    }
    
    fn texture<'a>(&'a self) -> &'a Option<TextureHandle> {
        &self.state.texture
    }
}

impl HeatmapState {

    pub fn render_data<N: Float, CMAP: Colormap>(&mut self, data: &Array2<N>, ctx: &egui::Context, cmap: CMAP) {
        self.dims = data.dim();
        self.texture = Some(data_to_texture(data, ctx, cmap));
    }

    pub fn set_mask(&mut self, mask: &Array2<u8>, ctx: &egui::Context) {
        self.mask = Some(create_mask(mask, ctx));
    }

    pub fn remove_mask(&mut self) {self.mask = None}
}

impl<N: Float> HeatmapStateOwned<N> {
    pub fn render_data<CMAP: Colormap>(&mut self, data: Array2<N>, ctx: &egui::Context, cmap: CMAP) {
        self.state.dims = data.dim();
        self.data = data;
        self.state.texture = Some(data_to_texture(&self.data, ctx, cmap));
    }

    pub fn apply_cmap<CMAP: Colormap>(&mut self, ctx: &egui::Context, cmap: CMAP) {
        self.state.texture =  Some(data_to_texture(&self.data, ctx, cmap));
    }
}

fn data_to_texture<CMAP: Colormap, N: Float>(data: &Array2<N>, ctx: &egui::Context, cmap: CMAP) -> TextureHandle {
        let (rows, cols) = data.dim();
        let mut pixels: Vec<egui::Color32> = Vec::with_capacity(rows * cols);

        pixels.extend(data.t().iter().map(|v| {
            if let Some(vf) = v.to_f64() {
                return cmap.egui_color(vf);
            }
            else {
                return egui::Color32::TRANSPARENT;
            }
    }));

    let image = egui::ColorImage::new([rows, cols], pixels);

    ctx.load_texture("heatmap_texture", image,  Default::default())
}

fn create_mask(mask: &Array2<u8>, ctx: &egui::Context) -> TextureHandle {
    let (rows, cols) = mask.dim();
    let mut pixels = Vec::with_capacity(rows * cols);

    pixels.extend(mask.t().iter().map(|m| {
        if *m == 0 {
            egui::Color32::TRANSPARENT
        }
        else {
            egui::Color32::from_black_alpha(255)
        }
    }));

    let image = egui::ColorImage::new([rows, cols], pixels);

    ctx.load_texture("mask_texture", image, Default::default())
}


impl<'a, State: HeatmapStateGeneric> HeatmapPlot<'a, State> {
    pub fn new(state: &'a State) -> Self {
        HeatmapPlot { state, data: Default::default() }
    }

    pub fn set_scale(mut self, scale: f32) -> Self {
        self.data.scale = scale;
        self
    }

    pub fn show_axes(mut self, on: bool) -> Self {
        self.data.show_axes = on;
        self
    }

    pub fn show_mask(mut self, on: bool) -> Self {
        self.data.show_mask = on;
        self
    }

    pub fn show_grid(mut self, on: bool) -> Self {
        self.data.show_grid = on;
        self
    }

    fn plot_empty(&self, ui: &mut egui::Ui) -> PlotResponse<()> {
        Plot::new("empty_heatmap_plot")
            .show_axes(self.data.show_axes)
            .show_grid(self.data.show_grid)
            .data_aspect(1.0)
            .show(ui, |_| {})
    }

    fn plot(&self, ui: &mut egui::Ui, texture: &TextureHandle) -> PlotResponse<()> {
        let (dimx, dimy) = self.state.dims();
        let scale = self.data.scale;
        let (sx, sy) = ((dimx as f32) * scale, (dimy as f32) * scale);

        Plot::new("heatmap_plot")
            .show_axes(self.data.show_axes)
            .show_grid(self.data.show_grid)
            .data_aspect(1.0)
            .show(ui, |plt_ut| {
                plt_ut.image(PlotImage::new("heatmap_image", texture.id(),
                PlotPoint::new(0.0, 0.0),  Vec2::new(sx, sy)));

                if let Some(mask) = self.state.mask() && self.data.show_mask {
                    plt_ut.image(PlotImage::new("mask_plot",
                        mask.id(), 
                        PlotPoint::new(0., 0.), 
                        Vec2::new(sx, sy)
                    ));
                }
            })
    }

    pub fn show(&self, ui: &mut egui::Ui) -> HeatmapResponse {
        
        let response = match &self.state.texture() {
            Some(texture) => self.plot(ui, texture),
            None => self.plot_empty(ui)
        };

        let pointer_pos = response.response.hover_pos();

        let pos = pointer_pos.map(|p_pos| response.transform.value_from_position(p_pos));

        HeatmapResponse { response: response.response, pos }
    }
}