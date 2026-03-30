use eframe::egui::Color32;

pub trait Colormap {
    fn rgb(&self, value: f64) -> (u8, u8, u8);

    fn egui_color(&self, value: f64) -> Color32 {
        let (r, g, b) = self.rgb(value);
        Color32::from_rgb(r, g, b)
    }
}

pub struct BWColormap {
    min: f64,
    max: f64   
}

impl Default for BWColormap {
    fn default() -> Self {
        BWColormap { min: 0.0, max: 1.0 }
    }
}

impl BWColormap {
    pub fn new(min: f64, max: f64) -> Self {
        BWColormap { min, max }
    }
}

impl Colormap for BWColormap {
    fn rgb(&self, value: f64) -> (u8, u8, u8) {
        let scale = self.max - self.min;
        let v_rescaled = (value - self.min) / scale;
        let rgb_val = (v_rescaled * 255.0) as u8;
        (rgb_val, rgb_val, rgb_val)
    }
}

pub struct RainbowColormap {
    min: f64,
    max: f64
}

impl RainbowColormap {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
}

impl Default for RainbowColormap {
    fn default() -> Self {
        Self { min: 0.0, max: 1.0 }
    }
}

impl Colormap for RainbowColormap {
    fn rgb(&self, value: f64) -> (u8, u8, u8) {
        // 1. Normalize the value to a 0.0 - 1.0 range
        let range = self.max - self.min;
        let t = if range.abs() < f64::EPSILON {
            0.5 // Default to middle of the spectrum if min == max
        } else {
            ((value - self.min) / range).clamp(0.0, 1.0)
        };

        // 2. Map 't' to the 4 segments of our rainbow
        // Segment 1: Blue (0,0,1) to Cyan (0,1,1)
        // Segment 2: Cyan (0,1,1) to Green (0,1,0)
        // Segment 3: Green (0,1,0) to Yellow (1,1,0)
        // Segment 4: Yellow (1,1,0) to Red (1,0,0)
        let (r, g, b) = if t <= 0.25 {
            (0.0, t * 4.0, 1.0)
        } else if t <= 0.5 {
            (0.0, 1.0, 1.0 - (t - 0.25) * 4.0)
        } else if t <= 0.75 {
            ((t - 0.5) * 4.0, 1.0, 0.0)
        } else {
            (1.0, 1.0 - (t - 0.75) * 4.0, 0.0)
        };

        // 3. Convert normalized [0, 1] floats to u8 [0, 255]
        (
            (r * 255.0).round() as u8,
            (g * 255.0).round() as u8,
            (b * 255.0).round() as u8,
        )
    }
}

pub struct ColormapPoint {
    color: (u8, u8, u8),
    value: f64
}

impl ColormapPoint {
    fn new(color: (u8, u8, u8), value: f64) -> Self {
        Self { color, value }
    }
}
pub struct LinearColormap {
    points: Vec<ColormapPoint>,
    min: f64,
    max: f64
}

impl LinearColormap {
    fn from_points(points: Vec<ColormapPoint>) -> Self {
        let max = points.iter()
            .max_by(|a, b| a.value.total_cmp(&b.value))
            .unwrap_or(&ColormapPoint { color: (0,0,0), value: (1.0) })
            .value;

        let min = points.iter()
            .min_by(|a, b| a.value.total_cmp(&b.value))
            .unwrap_or(&ColormapPoint { color: (0,0,0), value: (0.0) })
            .value;

        let scale = 1.0 / (max - min);

        let mut normalized_points = points.iter()
            .map(|p| ColormapPoint::new(p.color, (p.value - min)*scale))
            .collect::<Vec<_>>();

        normalized_points.sort_by(|a,b| a.value.total_cmp(&b.value));

        Self { points: normalized_points, min, max}
    }
}

impl Colormap for LinearColormap {
    fn rgb(&self, value: f64) -> (u8, u8, u8) {
        // Handle empty colormap safely
        if self.points.is_empty() {
            return (0, 0, 0); 
        }

        // 1. Normalize the input value between 0.0 and 1.0 based on min/max
        let range = self.max - self.min;
        let normalized_val = if range == 0.0 {
            0.0
        } else {
            ((value - self.min) / range).clamp(0.0, 1.0)
        };

        // 2. Handle bounds (assuming self.points is sorted by `value`)
        if normalized_val <= self.points.first().unwrap().value {
            return self.points.first().unwrap().color;
        }
        if normalized_val >= self.points.last().unwrap().value {
            return self.points.last().unwrap().color;
        }

        // 3. Find the segment to interpolate
        for i in 0..self.points.len() - 1 {
            let p1 = &self.points[i];
            let p2 = &self.points[i + 1];

            if normalized_val >= p1.value && normalized_val <= p2.value {
                // Calculate the interpolation factor `t` between 0.0 and 1.0 for this segment
                let segment_range = p2.value - p1.value;
                let t = if segment_range == 0.0 {
                    0.0
                } else {
                    (normalized_val - p1.value) / segment_range
                };

                // Interpolate each color channel
                let r = lerp(p1.color.0, p2.color.0, t);
                let g = lerp(p1.color.1, p2.color.1, t);
                let b = lerp(p1.color.2, p2.color.2, t);

                return (r, g, b);
            }
        }

        // Fallback for edge cases, though it shouldn't be reached
        self.points.last().unwrap().color
    }
}

/// Helper function to linearly interpolate between two u8 color values
fn lerp(c1: u8, c2: u8, t: f64) -> u8 {
    let c1 = c1 as f64;
    let c2 = c2 as f64;
    (c1 + t * (c2 - c1)).round() as u8
}

pub struct BRWColormap {
    min: f64,
    max: f64
}

impl BRWColormap {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    } 
}

impl Colormap for BRWColormap {
    fn rgb(&self, value: f64) -> (u8, u8, u8) {
        let scale = self.max - self.min;
        let v_rescaled = (value - self.min) / scale;
        if v_rescaled < 0.5 {
            let b = (v_rescaled * 2.) * 255. + (1. - v_rescaled * 2.) * 0.;
            (b as u8, b as u8, 255 as u8)
        }
        else {
            let t = (v_rescaled - 0.5) * 2.;
            let r = t * 0. +  (1. - t) * 255.;
            (255, r as u8, r as u8)
        }
    }
}

pub struct BRBColormap {
    min: f64,
    max: f64
}

impl BRBColormap {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }
}

impl Colormap for BRBColormap {
    fn rgb(&self, value: f64) -> (u8, u8, u8) {
        let scale = self.max - self.min;
        let v_rescaled = (value - self.min) / scale;
        if v_rescaled < 0.5 {
            let b = (1.0 - v_rescaled * 2.) *255.;
            (0, 0, b as u8)
        }
        else {
            let r = (v_rescaled - 0.5) * 255. *2.;
            (r as u8, 0, 0)
        }
    }
}